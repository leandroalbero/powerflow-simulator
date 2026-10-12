"""Bucketed min/max envelope downsampler.

Each bucket emits its most extreme low and high samples in time order so a
single-minute spike survives. ``step_cost`` is the sum of finite costs in the
bucket, attached to the later emitted sample.
"""

from dataclasses import dataclass

import numpy as np

_POWER = (
    "battery_level",
    "grid_import",
    "grid_export",
    "solar_power",
    "house_consumption",
    "battery_charge",
    "battery_discharge",
)


@dataclass(frozen=True, slots=True)
class Envelope:
    ts_ms: np.ndarray
    columns: dict[str, np.ndarray]
    step_cost: np.ndarray
    downsampled: bool
    resolution_minutes: float


def envelope_downsample(
    ts_ms: np.ndarray,
    columns: dict[str, np.ndarray],
    step_cost: np.ndarray,
    max_points: int,
) -> Envelope:
    ts = np.asarray(ts_ms, dtype=np.int64)
    cols = {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}
    cost = np.asarray(step_cost, dtype=np.float64)
    n = len(ts)
    if n == 0 or n <= max_points:
        return Envelope(ts, cols, cost, False, 1.0)

    n_buckets = max(1, int(max_points))
    t0 = int(ts[0])
    t1 = int(ts[-1])
    edges = np.linspace(t0, t1, n_buckets + 1)
    edges = np.rint(edges).astype(np.int64)
    edges[-1] = t1 + 1
    bounds = np.searchsorted(ts, edges, side="left")

    power_keys = [key for key in _POWER if key in cols] or list(cols)
    gap_key = "grid_import" if "grid_import" in cols else power_keys[0]

    out_idx: list[int] = []
    out_cost: list[float] = []
    gap_rows: set[int] = set()
    for bucket in range(n_buckets):
        lo = int(bounds[bucket])
        hi = int(bounds[bucket + 1])
        if lo >= hi:
            continue
        gap = cols[gap_key][lo:hi]
        finite_gap = np.isfinite(gap)
        if not finite_gap.any():
            out_idx.append(lo)
            out_cost.append(np.nan)
            gap_rows.add(len(out_idx) - 1)
            continue
        chosen = _extremes(cols, power_keys, lo, hi)
        bucket_cost = _bucket_cost(cost[lo:hi])
        for pos, idx in enumerate(chosen):
            out_idx.append(idx)
            out_cost.append(bucket_cost if pos == len(chosen) - 1 else 0.0)

    idx = np.asarray(out_idx, dtype=np.intp)
    gathered = {key: values[idx].copy() for key, values in cols.items()}
    for row in gap_rows:
        for values in gathered.values():
            values[row] = np.nan
    span_min = max(t1 - t0, 0) / 60_000.0
    resolution = span_min / n_buckets if n_buckets else 1.0
    return Envelope(ts[idx], gathered, np.asarray(out_cost, dtype=np.float64), True, float(resolution))


def _extremes(cols: dict[str, np.ndarray], keys: list[str], lo: int, hi: int) -> list[int]:
    width = hi - lo
    sample_min = np.full(width, np.inf)
    sample_max = np.full(width, -np.inf)
    finite = np.zeros(width, dtype=bool)
    for key in keys:
        values = cols[key][lo:hi]
        ok = np.isfinite(values)
        finite |= ok
        sample_min = np.where(ok, np.minimum(sample_min, values), sample_min)
        sample_max = np.where(ok, np.maximum(sample_max, values), sample_max)
    if not finite.any():
        return [lo]
    sample_min = np.where(finite, sample_min, np.inf)
    sample_max = np.where(finite, sample_max, -np.inf)
    return sorted({lo + int(np.argmin(sample_min)), lo + int(np.argmax(sample_max))})


def _bucket_cost(cost: np.ndarray) -> float:
    if not np.isfinite(cost).any():
        return float("nan")
    return float(np.nansum(cost))
