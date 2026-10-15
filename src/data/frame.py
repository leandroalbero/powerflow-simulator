"""Fixed 1-minute power frame built from per-entity, per-minute sensor data.

Every simulation runs on the frame returned by :func:`build_power_frame`: one row
per minute on a fixed grid (tz Europe/Madrid), power in kW, and a ``valid`` flag
that marks minutes where the house load is unknown (HA/Influx outage). A row at
``ts`` covers ``[ts, ts + 1 min)``.

Entity data comes from ``src.data.store.read_entity`` (the parquet store) or from
any reader with the same contract: ``reader(key, start, end) -> DataFrame`` indexed
by tz-aware UTC start-of-minute ``ts`` with columns ``mean`` (W, or % for soc) and
``count`` (raw samples in that minute); rows exist only for minutes with samples.
"""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Madrid"
ENTITY_KEYS = ("pv", "load", "battery", "grid", "soc", "ev")

# Sensors write on change, so a minute without samples usually means "unchanged".
# SoC changes in 1 % steps and can stay put for hours, hence no limit.
FFILL_LIMIT_MIN: Dict[str, Optional[int]] = {
    "pv": 15, "load": 15, "battery": 15, "grid": 15, "ev": 15, "soc": None,
}
_SOC_LOOKBACK = pd.Timedelta(days=7)

DateLike = Union[str, _dt.date, _dt.datetime, pd.Timestamp, None]
EntityReader = Callable[[str, Optional[pd.Timestamp], Optional[pd.Timestamp]], Optional[pd.DataFrame]]


def store_reader(key: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    # Imported lazily: the parquet store is owned by the data layer and may be absent.
    from src.data.store import read_entity

    try:
        return read_entity(key, start=start, end=end)
    except FileNotFoundError:
        return None
    except KeyError as exc:
        # Only an unknown entity means "no data"; a KeyError from a malformed file must not
        # silently turn into an empty frame.
        if exc.args[:1] == (key,):
            return None
        raise


def memory_reader(entities: Mapping[str, pd.DataFrame], fallback: Optional[EntityReader] = None) -> EntityReader:
    """Reader over in-memory entity frames, deferring unknown keys to ``fallback``."""

    def _read(key: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        df = entities.get(key)
        if df is None:
            return fallback(key, start, end) if fallback is not None else None
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index < end]
        return df

    return _read


def samples_to_entity_frame(samples: pd.Series) -> pd.DataFrame:
    """Aggregate raw timestamped samples into the store's per-minute ``mean``/``count`` format."""
    s = pd.to_numeric(samples, errors="coerce").dropna()
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=True, format="ISO8601"))
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    s = pd.Series(s.to_numpy(dtype="float64"), index=idx.floor("min"))
    if s.empty:
        out = pd.DataFrame({"mean": pd.Series(dtype="float64"), "count": pd.Series(dtype="int32")})
        out.index = pd.DatetimeIndex([], tz="UTC", name="ts")
        return out
    g = s.groupby(level=0, sort=True)
    out = pd.DataFrame({"mean": g.mean(), "count": g.size().astype("int32")})
    out.index.name = "ts"
    return out


def csv_to_entity_frame(source: Union[bytes, str, Path]) -> pd.DataFrame:
    """Convert a Home Assistant ``last_changed,state`` CSV into the per-minute entity format."""
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    df = pd.read_csv(buf, usecols=["last_changed", "state"], dtype={"state": "string"})
    ts = pd.to_datetime(df["last_changed"], utc=True, format="ISO8601")
    s = pd.Series(pd.to_numeric(df["state"], errors="coerce").to_numpy(dtype="float64"), index=ts)
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return samples_to_entity_frame(s)


def to_utc_bound(value: DateLike, is_end: bool) -> Optional[pd.Timestamp]:
    """Convert a user bound to a UTC instant; the returned end is exclusive."""
    if value is None:
        return None
    whole_day = isinstance(value, _dt.date) and not isinstance(value, _dt.datetime)
    if isinstance(value, str) and len(value.strip()) == 10:
        whole_day = True
        value = _dt.date.fromisoformat(value.strip())
    if whole_day:
        day = pd.Timestamp(value)
        if is_end:
            day = day + pd.Timedelta(days=1)
        # Local midnight always exists in Europe/Madrid (DST switches at 02:00/03:00).
        return day.tz_localize(LOCAL_TZ).tz_convert("UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    ts = ts.tz_convert("UTC").floor("min")
    return ts + pd.Timedelta(minutes=1) if is_end else ts


def _read(reader: EntityReader, key: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.Series]:
    df = reader(key, start, end)
    if df is None or len(df) == 0:
        return None
    s = df["mean"].astype("float64").dropna()
    idx = pd.DatetimeIndex(s.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s = pd.Series(s.to_numpy(), index=idx)
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index < end]
    return s if len(s) else None


def _on_grid(s: Optional[pd.Series], grid: pd.DatetimeIndex, limit_min: Optional[int]) -> np.ndarray:
    if s is None:
        return np.full(len(grid), np.nan)
    tolerance = pd.Timedelta(minutes=limit_min) if limit_min is not None else None
    return s.reindex(grid, method="ffill", tolerance=tolerance).to_numpy(dtype="float64")


def _empty_frame(include_measured: bool) -> pd.DataFrame:
    cols = ["pv_kw", "load_kw", "ev_kw"] + (["grid_kw", "battery_kw", "soc_pct"] if include_measured else [])
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    df["valid"] = pd.Series(dtype="bool")
    df.index = pd.DatetimeIndex([], tz=LOCAL_TZ, name="ts")
    return df


def _has_load(reader: EntityReader, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> bool:
    return _read(reader, "load", start, end) is not None


def _clamp_to_load_span(
    reader: EntityReader, load: Optional[pd.Series], start_utc: Optional[pd.Timestamp],
    end_utc: Optional[pd.Timestamp],
) -> tuple:
    """Clamp ``[start_utc, end_utc)`` to the GLOBAL span of load data, not the window's.

    An outage at a window edge must stay in the frame as invalid rows, so the
    window is only shortened where the load series itself begins or ends.
    """
    first = load.index[0] if load is not None else None
    last_end = load.index[-1] + pd.Timedelta(minutes=1) if load is not None else None
    if start_utc is None:
        lo = first
    elif (first is not None and first <= start_utc) or _has_load(reader, None, start_utc):
        lo = start_utc
    else:
        lo = first
    if end_utc is None:
        hi = last_end
    elif (last_end is not None and last_end >= end_utc) or _has_load(reader, end_utc, None):
        hi = end_utc
    else:
        hi = last_end
    return lo, hi


def build_power_frame(
    start: DateLike = None,
    end: DateLike = None,
    exclude_ev: bool = False,
    include_measured: bool = False,
    reader: Optional[EntityReader] = None,
) -> pd.DataFrame:
    """Build the fixed 1-minute frame for ``[start, end]``.

    ``'YYYY-MM-DD'`` bounds are Europe/Madrid local days and ``end`` is inclusive
    (the whole end day). Timestamps are taken to the minute, ``end`` inclusive.
    The range is clamped to the overall span of the load data; minutes inside
    it without load data are rows with ``valid=False``.
    """
    reader = reader or store_reader
    start_utc = to_utc_bound(start, is_end=False)
    end_utc = to_utc_bound(end, is_end=True)

    def lookback(key: str) -> Optional[pd.Timestamp]:
        if start_utc is None:
            return None
        limit = FFILL_LIMIT_MIN[key]
        return start_utc - (pd.Timedelta(minutes=limit) if limit is not None else _SOC_LOOKBACK)

    load = _read(reader, "load", lookback("load"), end_utc)
    lo, hi = _clamp_to_load_span(reader, load, start_utc, end_utc)
    if lo is None or hi is None or lo >= hi:
        return _empty_frame(include_measured)
    grid = pd.date_range(lo, hi, freq="1min", inclusive="left")

    load_kw = _on_grid(load, grid, FFILL_LIMIT_MIN["load"]) / 1000.0
    pv_kw = _on_grid(_read(reader, "pv", lookback("pv"), end_utc), grid, FFILL_LIMIT_MIN["pv"]) / 1000.0
    ev_kw = _on_grid(_read(reader, "ev", lookback("ev"), end_utc), grid, FFILL_LIMIT_MIN["ev"]) / 1000.0

    pv_kw = np.clip(np.nan_to_num(pv_kw, nan=0.0), 0.0, None)
    ev_kw = np.clip(np.nan_to_num(ev_kw, nan=0.0), 0.0, None)
    valid = ~np.isnan(load_kw)
    load_kw = np.clip(load_kw, 0.0, None)
    if exclude_ev:
        load_kw = np.clip(load_kw - ev_kw, 0.0, None)

    cols: Dict[str, np.ndarray] = {"pv_kw": pv_kw, "load_kw": load_kw, "ev_kw": ev_kw}
    if include_measured:
        grid_kw = _on_grid(_read(reader, "grid", lookback("grid"), end_utc), grid, FFILL_LIMIT_MIN["grid"]) / 1000.0
        batt_kw = _on_grid(_read(reader, "battery", lookback("battery"), end_utc), grid,
                           FFILL_LIMIT_MIN["battery"]) / 1000.0
        soc = _on_grid(_read(reader, "soc", lookback("soc"), end_utc), grid, FFILL_LIMIT_MIN["soc"])
        soc[~valid] = np.nan
        cols.update({"grid_kw": grid_kw, "battery_kw": batt_kw, "soc_pct": soc})
    cols["valid"] = valid

    frame = pd.DataFrame(cols, index=grid.tz_convert(LOCAL_TZ))
    frame.index.name = "ts"
    return frame


def frame_coverage(frame: pd.DataFrame, min_gap_minutes: int = 60) -> dict:
    """Summarise valid coverage and list invalid runs of at least ``min_gap_minutes``."""
    n = len(frame)
    if n == 0:
        return {"valid_fraction": 0.0, "valid_hours": 0.0, "excluded_hours": 0.0, "gaps": []}
    valid = frame["valid"].to_numpy(dtype=bool)
    invalid = ~valid
    edges = np.diff(np.concatenate(([0], invalid.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    idx = frame.index
    step = pd.Timedelta(minutes=1)
    gaps = []
    for s, e in zip(starts, ends):
        minutes = int(e - s)
        if minutes < min_gap_minutes:
            continue
        gaps.append({
            "start": idx[s].isoformat(),
            "end": (idx[e - 1] + step).isoformat(),
            "hours": round(minutes / 60.0, 2),
        })
    return {
        "valid_fraction": float(valid.mean()),
        "valid_hours": float(valid.sum()) / 60.0,
        "excluded_hours": float(invalid.sum()) / 60.0,
        "gaps": gaps,
    }
