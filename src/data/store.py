"""Parquet store for minute means. Writes are atomic (tmp + os.replace)."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from src.data.entities import ENTITIES
from src.data.influx import empty_frame

SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("ns", tz="UTC")),
        pa.field("mean", pa.float64()),
        pa.field("count", pa.int32()),
    ]
)


class ManifestEntity(TypedDict):
    entity_id: str
    measurement: str
    first_ts: str | None
    last_ts: str | None
    rows: int


class Manifest(TypedDict):
    source: str
    updated_at: str | None
    entities: dict[str, ManifestEntity]


class StoreError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def data_dir() -> Path:
    """Repo ``data/``, or POWERFLOW_DATA_DIR when set."""
    override = os.environ.get("POWERFLOW_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


def influx_dir() -> Path:
    return data_dir() / "influx"


def entity_path(key: str) -> Path:
    return influx_dir() / f"{key}.parquet"


def manifest_path() -> Path:
    return influx_dir() / "manifest.json"


def _as_utc(ts: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def validate_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if list(frame.columns) != ["ts", "mean", "count"]:
        raise StoreError("frame columns must be ts, mean, count")
    ts = frame["ts"]
    if getattr(ts.dt, "tz", None) is None:
        raise StoreError("ts must be UTC")
    if str(ts.dt.tz) != "UTC":
        raise StoreError("ts must be UTC")
    if not bool(ts.is_unique):
        raise StoreError("ts must be unique")
    if not bool(ts.is_monotonic_increasing):
        raise StoreError("ts must be strictly increasing")
    if bool(ts.diff().iloc[1:].eq(pd.Timedelta(0)).any()):
        raise StoreError("ts must be strictly increasing")


def atomic_replace(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    ordered = frame.loc[:, ["ts", "mean", "count"]].copy()
    ordered["ts"] = pd.to_datetime(ordered["ts"], utc=True).astype("datetime64[ns, UTC]")
    ordered["mean"] = ordered["mean"].astype("float64")
    ordered["count"] = ordered["count"].astype("int32")
    ordered = ordered.sort_values("ts").reset_index(drop=True)
    validate_frame(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    table = pa.Table.from_pandas(ordered, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    atomic_replace(tmp, path)


def read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return empty_frame()
    table = pq.read_table(path, columns=["ts", "mean", "count"])
    raw = table.to_pandas()
    built = pd.DataFrame(
        {
            "ts": pd.to_datetime(raw["ts"], utc=True).astype("datetime64[ns, UTC]"),
            "mean": raw["mean"].astype("float64"),
            "count": raw["count"].astype("int32"),
        }
    )
    built.sort_values("ts", inplace=True)
    built.reset_index(drop=True, inplace=True)
    return built


def merge_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concat and keep the new row when timestamps overlap."""
    if existing.empty:
        combined = new
    elif new.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts"], keep="last")
    return combined.sort_values("ts").reset_index(drop=True)


def iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def read_entity(key: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """Minute means for one entity. Index is ts UTC. ``end`` is exclusive."""
    if key not in ENTITIES:
        raise KeyError(key)
    frame = read_frame(entity_path(key))
    indexed = frame.set_index("ts")
    if start is not None:
        indexed = indexed.loc[indexed.index >= _as_utc(start)]
    if end is not None:
        indexed = indexed.loc[indexed.index < _as_utc(end)]
    return indexed.loc[:, ["mean", "count"]]


def empty_manifest() -> Manifest:
    return {"source": "influxdb", "updated_at": None, "entities": {}}


def read_manifest() -> Manifest:
    path = manifest_path()
    if not path.is_file():
        return empty_manifest()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise StoreError("manifest is not an object")
    entities = raw.get("entities")
    if not isinstance(entities, dict):
        entities = {}
    updated = raw.get("updated_at")
    updated_at = updated if isinstance(updated, str) else None
    return {"source": "influxdb", "updated_at": updated_at, "entities": entities}


def write_manifest(manifest: Manifest) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    atomic_replace(tmp, path)
