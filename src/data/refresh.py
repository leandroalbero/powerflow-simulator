"""Incremental, idempotent InfluxDB → parquet refresh."""

import argparse
import importlib.util
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.data.config import InfluxConfig, load_influx_config
from src.data.entities import DATA_START, MAX_WORKERS, OVERLAP, Entity, resolve_entities
from src.data.influx import drop_at_or_after, empty_frame, fetch_range
from src.data.store import (
    ManifestEntity,
    entity_path,
    iso_z,
    merge_frames,
    read_frame,
    read_manifest,
    write_frame,
    write_manifest,
)

UTC = timezone.utc


class EntitySummary(TypedDict):
    rows_added: int
    rows: int
    first_ts: str | None
    last_ts: str | None
    gaps_over_1h: int
    fetched: int


class RefreshSummary(TypedDict):
    updated_at: str
    entities: dict[str, EntitySummary]


def utc_now() -> datetime:
    return datetime.now(UTC)


def floor_minute(ts: datetime) -> datetime:
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    return aware.astimezone(UTC).replace(second=0, microsecond=0)


def parse_day(value: date | datetime | str) -> datetime:
    match value:
        case str() as text:
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
        case datetime() as ts:
            aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
            return aware.astimezone(UTC)
        case date() as day:
            return datetime(day.year, day.month, day.day, tzinfo=UTC)


def count_gaps(frame: pd.DataFrame) -> int:
    if len(frame) < 2:
        return 0
    delta = frame["ts"].sort_values().diff()
    return int((delta > pd.Timedelta(hours=1)).sum())


def _bounds(
    entity: Entity,
    since: datetime | None,
    until: datetime | None,
    full: bool,
    now: datetime,
) -> tuple[datetime, datetime, bool]:
    end = floor_minute(now)
    if until is not None:
        end = min(end, until + timedelta(days=1))
    existing = read_frame(entity_path(entity.key))
    replace = full or existing.empty
    if full:
        start = since if since is not None else DATA_START
    elif since is not None:
        start = since
    elif existing.empty:
        start = DATA_START
    else:
        last = existing["ts"].iloc[-1].to_pydatetime()
        start = max(DATA_START, last - OVERLAP)
    if start < DATA_START and since is None:
        start = DATA_START
    return start, end, replace


def _summarize(existing: pd.DataFrame, fetched: pd.DataFrame, merged: pd.DataFrame) -> EntitySummary:
    if existing.empty:
        added = len(merged)
    else:
        added = int((~merged["ts"].isin(existing["ts"])).sum())
    first_ts = iso_z(merged["ts"].iloc[0]) if not merged.empty else None
    last_ts = iso_z(merged["ts"].iloc[-1]) if not merged.empty else None
    return {
        "rows_added": added,
        "rows": int(len(merged)),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "gaps_over_1h": count_gaps(fetched),
        "fetched": int(len(fetched)),
    }


def _refresh_entity(
    cfg: InfluxConfig,
    entity: Entity,
    since: datetime | None,
    until: datetime | None,
    full: bool,
    now: datetime,
) -> tuple[EntitySummary, ManifestEntity]:
    start, end, replace = _bounds(entity, since, until, full, now)
    existing = read_frame(entity_path(entity.key))
    fetched = empty_if_inverted(cfg, entity, start, end)
    fetched = drop_at_or_after(fetched, end)
    merged = fetched if replace else merge_frames(existing, fetched)
    write_frame(entity_path(entity.key), merged)
    summary = _summarize(existing, fetched, merged)
    record: ManifestEntity = {
        "entity_id": entity.entity_id,
        "measurement": entity.measurement,
        "first_ts": summary["first_ts"],
        "last_ts": summary["last_ts"],
        "rows": summary["rows"],
    }
    return summary, record


def empty_if_inverted(cfg: InfluxConfig, entity: Entity, start: datetime, end: datetime) -> pd.DataFrame:
    if start >= end:
        return empty_frame()
    return fetch_range(cfg, entity, start, end)


def _emit(progress: Callable[[str], None] | None, line: str) -> None:
    print(line, flush=True)
    if progress is not None:
        progress(line)


def _line(key: str, summary: EntitySummary) -> str:
    return (
        f"{key:<8} added={summary['rows_added']:<6} rows={summary['rows']:<8} "
        f"{summary['first_ts'] or '-'} .. {summary['last_ts'] or '-'}  "
        f"gaps>1h={summary['gaps_over_1h']}"
    )


def _extend_irradiance() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_irradiance.py"
    spec = importlib.util.spec_from_file_location("fetch_irradiance", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_ingestion(end=date.today())


def refresh(
    entities: Sequence[str] | None = None,
    since: date | datetime | str | None = None,
    full: bool = False,
    until: date | datetime | str | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    with_irradiance: bool = True,
) -> RefreshSummary:
    """Refresh parquet extracts. Incremental by default; ``full`` replaces."""
    selected = resolve_entities(list(entities) if entities is not None else None)
    since_dt = parse_day(since) if since is not None else None
    until_dt = parse_day(until) if until is not None else None
    now = utc_now()
    cfg = load_influx_config()
    summaries: dict[str, EntitySummary] = {}
    records: dict[str, ManifestEntity] = {}
    workers = min(MAX_WORKERS, len(selected)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_refresh_entity, cfg, entity, since_dt, until_dt, full, now): entity.key
            for entity in selected
        }
        for fut in as_completed(futures):
            key = futures[fut]
            summary, record = fut.result()
            summaries[key] = summary
            records[key] = record
            _emit(progress, _line(key, summary))

    manifest = read_manifest()
    manifest["source"] = "influxdb"
    manifest["updated_at"] = floor_minute(now).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["entities"].update(records)
    write_manifest(manifest)

    _emit(progress, f"{'entity':<8} {'added':>8} {'rows':>8} {'first':<22} {'last':<22} {'gaps>1h':>8}")
    for entity in selected:
        item = summaries[entity.key]
        _emit(
            progress,
            f"{entity.key:<8} {item['rows_added']:>8} {item['rows']:>8} "
            f"{item['first_ts'] or '-':<22} {item['last_ts'] or '-':<22} {item['gaps_over_1h']:>8}",
        )
    if with_irradiance:
        _extend_irradiance()
    return {"updated_at": manifest["updated_at"] or "", "entities": summaries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh InfluxDB extracts to parquet")
    parser.add_argument("--since", type=str, default=None, help="Inclusive start date YYYY-MM-DD (UTC)")
    parser.add_argument("--until", type=str, default=None, help="Inclusive end date YYYY-MM-DD (UTC)")
    parser.add_argument("--full", action="store_true", help="Replace from 2022-08-25 or --since")
    parser.add_argument("--entities", type=str, default=None, help="Comma-separated keys, default all")
    parser.add_argument("--with-irradiance", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    keys = [part.strip() for part in args.entities.split(",") if part.strip()] if args.entities else None
    refresh(entities=keys, since=args.since, full=args.full, until=args.until, with_irradiance=args.with_irradiance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
