"""Influx refresh: merge, idempotence, atomic write, chunk edges. No network."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.config import load_influx_config, parse_env_file
from src.data.entities import ENTITIES
from src.data.influx import concat_frames, iter_chunks
from src.data.refresh import refresh
from src.data.store import entity_path, read_entity, read_manifest, write_frame

UTC = timezone.utc


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFLUX_URL", "http://influx.test")
    monkeypatch.setenv("INFLUX_DB", "db")
    monkeypatch.setenv("INFLUX_USER", "user")
    monkeypatch.setenv("INFLUX_PASSWORD", "secret")
    return tmp_path


def _frame(stamps: list[datetime], mean: float, count: int = 10) -> pd.DataFrame:
    ts = pd.to_datetime(stamps, utc=True).astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {
            "ts": ts,
            "mean": pd.Series([mean] * len(stamps), dtype="float64"),
            "count": pd.Series([count] * len(stamps), dtype="int32"),
        }
    )


def _at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_env_parser_quotes_export_and_comments() -> None:
    parsed = parse_env_file(
        "\n".join(
            [
                "# comment",
                "export INFLUX_URL=\"http://from-file\"",
                "INFLUX_DB='ha'",
                "INFLUX_USER=user",
                "INFLUX_PASSWORD=secret",
                "ignored",
            ]
        )
    )
    assert parsed["INFLUX_URL"] == "http://from-file"
    assert parsed["INFLUX_DB"] == "ha"
    assert "ignored" not in parsed


def test_process_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFLUX_URL", "http://from-env")
    monkeypatch.setenv("INFLUX_DB", "db")
    monkeypatch.setenv("INFLUX_USER", "user")
    monkeypatch.setenv("INFLUX_PASSWORD", "secret")
    path = tmp_path / ".env"
    path.write_text("INFLUX_URL=http://from-file\nINFLUX_DB=other\nINFLUX_USER=u\nINFLUX_PASSWORD=p\n")
    cfg = load_influx_config(path)
    assert cfg.url == "http://from-env"
    assert cfg.db == "db"


def test_env_file_fallback_when_process_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("INFLUX_URL", "INFLUX_DB", "INFLUX_USER", "INFLUX_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "INFLUX_URL=http://from-file\nINFLUX_DB=ha\nINFLUX_USER=user\nINFLUX_PASSWORD=secret\n"
    )
    cfg = load_influx_config(path)
    assert cfg.url == "http://from-file"
    assert cfg.db == "ha"


def test_incremental_merge_keeps_new_row_on_overlap(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = _at(2026, 10, 11, 10, 0)
    later = stamp + timedelta(minutes=1)
    write_frame(entity_path("pv"), _frame([stamp], mean=1.0))

    def fake_range(cfg: object, entity: object, start: datetime, end: datetime) -> pd.DataFrame:
        return _frame([stamp, later], mean=9.0)

    monkeypatch.setattr("src.data.refresh.fetch_range", fake_range)
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 11, 12, 30))

    refresh(entities=["pv"], with_irradiance=False)

    frame = read_entity("pv")
    assert frame.loc[pd.Timestamp(stamp), "mean"] == 9.0
    assert pd.Timestamp(later) in frame.index
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


def test_second_run_is_idempotent(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _at(2026, 10, 11, 10, 0)

    def fake_range(cfg: object, entity: object, start: datetime, end: datetime) -> pd.DataFrame:
        return _frame([stamp], mean=4.0, count=3)

    monkeypatch.setattr("src.data.refresh.fetch_range", fake_range)
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 11, 12, 0))

    refresh(entities=["pv"], since="2026-10-11", with_irradiance=False)
    first = read_entity("pv")
    refresh(entities=["pv"], since="2026-10-11", with_irradiance=False)
    second = read_entity("pv")
    pd.testing.assert_frame_equal(first, second)


def test_atomic_write_leaves_no_tmp(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _at(2026, 10, 11, 10, 0)
    monkeypatch.setattr(
        "src.data.refresh.fetch_range",
        lambda cfg, entity, start, end: _frame([stamp], mean=1.0),
    )
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 11, 12, 0))
    refresh(entities=["pv"], since="2026-10-11", with_irradiance=False)
    leftovers = list((isolated / "influx").glob("*.tmp"))
    assert leftovers == []
    assert (isolated / "influx" / "pv.parquet").is_file()


def test_open_minute_is_excluded(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = _at(2026, 10, 11, 11, 59)
    open_minute = _at(2026, 10, 11, 12, 0)
    monkeypatch.setattr(
        "src.data.refresh.fetch_range",
        lambda cfg, entity, start, end: _frame([closed, open_minute], mean=2.0),
    )
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 11, 12, 0) + timedelta(seconds=30))
    refresh(entities=["pv"], since="2026-10-11", with_irradiance=False)
    frame = read_entity("pv")
    assert pd.Timestamp(closed) in frame.index
    assert pd.Timestamp(open_minute) not in frame.index


def test_manifest_records_entity(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _at(2026, 10, 11, 10, 0)
    monkeypatch.setattr(
        "src.data.refresh.fetch_range",
        lambda cfg, entity, start, end: _frame([stamp], mean=5.0, count=8),
    )
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 11, 12, 0))
    refresh(entities=["pv"], since="2026-10-11", with_irradiance=False)
    manifest = read_manifest()
    assert manifest["source"] == "influxdb"
    assert manifest["updated_at"] == "2026-10-11T12:00:00Z"
    entity = manifest["entities"]["pv"]
    assert entity["entity_id"] == ENTITIES["pv"].entity_id
    assert entity["measurement"] == ENTITIES["pv"].measurement
    assert entity["rows"] == 1
    assert entity["first_ts"] == "2026-10-11T10:00:00Z"
    assert entity["last_ts"] == "2026-10-11T10:00:00Z"


def test_seven_day_chunks_have_no_gap_or_duplicate_at_the_edge(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[datetime, datetime]] = []

    def fake_chunk(cfg: object, entity: object, start: datetime, end: datetime) -> pd.DataFrame:
        calls.append((start, end))
        return _frame([start, end - timedelta(minutes=1)], mean=1.0)

    monkeypatch.setattr("src.data.influx.fetch_chunk", fake_chunk)
    monkeypatch.setattr("src.data.refresh.utc_now", lambda: _at(2026, 10, 15) + timedelta(seconds=30))
    refresh(entities=["pv"], since="2026-10-01", with_irradiance=False)

    assert calls == [
        (_at(2026, 10, 1), _at(2026, 10, 8)),
        (_at(2026, 10, 8), _at(2026, 10, 15)),
    ]
    frame = read_entity("pv")
    assert frame.index.is_unique
    assert pd.Timestamp(_at(2026, 10, 7, 23, 59)) in frame.index
    assert pd.Timestamp(_at(2026, 10, 8)) in frame.index
    assert len(frame) == 4


def test_iter_chunks_are_half_open_and_contiguous() -> None:
    start = _at(2026, 10, 1)
    end = _at(2026, 10, 15)
    chunks = iter_chunks(start, end)
    assert chunks[0][1] == chunks[1][0]
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all(b[0] < b[1] for b in chunks)


def test_concat_keeps_one_row_when_chunks_both_emit_the_boundary() -> None:
    edge = _at(2026, 10, 8)
    left = _frame([edge], mean=1.0)
    right = _frame([edge], mean=2.0)
    merged = concat_frames([left, right])
    assert len(merged) == 1
    assert merged["mean"].iloc[0] == 2.0
