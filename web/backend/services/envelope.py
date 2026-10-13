"""Bucketed min/max envelope downsampler.

Each bucket emits the per-column argmin and argmax, plus one NaN index for any
column that has a gap in the bucket, so a spike in one series is not dropped
because another series (kWh vs kW) is larger, and a partial gap breaks that
series. ``step_cost`` is the sum of finite costs in the bucket, attached to the
later emitted sample. ``max_points`` is a cap: the bucket count is chosen so
the unioned indices fit.
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

    keys = [key for key in _POWER if key in cols] or list(cols)
    # Columns the caller passed but that are not power keys still need their own extremes.
    keys = list(dict.fromkeys([*keys, *cols]))
    per_bucket = max(1, 3 * len(keys))
    n_buckets = max(1, min(n, max_points // per_bucket)) if per_bucket <= max_points else 1
    t0 = int(ts[0])
    t1 = int(ts[-1])
    out_idx, out_cost = _collect(ts, cols, cost, keys, n_buckets, t0, t1)
    if len(out_idx) > max_points:
        out_idx, out_cost = _trim(out_idx, out_cost, max_points)

    idx = np.asarray(out_idx, dtype=np.intp)
    gathered = {key: values[idx].copy() for key, values in cols.items()}
    span_min = max(t1 - t0, 0) / 60_000.0
    resolution = span_min / n_buckets if n_buckets else 1.0
    return Envelope(ts[idx], gathered, np.asarray(out_cost, dtype=np.float64), True, float(resolution))


def _collect(ts, cols, cost, keys, n_buckets, t0, t1):
    edges = np.linspace(t0, t1, n_buckets + 1)
    edges = np.rint(edges).astype(np.int64)
    edges[-1] = t1 + 1
    bounds = np.searchsorted(ts, edges, side="left")
    out_idx: list[int] = []
    out_cost: list[float] = []
    for bucket in range(n_buckets):
        lo = int(bounds[bucket])
        hi = int(bounds[bucket + 1])
        if lo >= hi:
            continue
        chosen = _column_indices(cols, keys, lo, hi)
        bucket_cost = _bucket_cost(cost[lo:hi])
        last = len(chosen) - 1
        for pos, index in enumerate(chosen):
            out_idx.append(index)
            out_cost.append(bucket_cost if pos == last else 0.0)
    return out_idx, out_cost


def _column_indices(cols: dict[str, np.ndarray], keys: list[str], lo: int, hi: int) -> list[int]:
    chosen: set[int] = set()
    for key in keys:
        values = cols[key][lo:hi]
        finite = np.isfinite(values)
        if finite.any():
            finite_values = np.where(finite, values, np.inf)
            chosen.add(lo + int(np.argmin(finite_values)))
            chosen.add(lo + int(np.argmax(np.where(finite, values, -np.inf))))
        if not finite.all():
            chosen.add(lo + int(np.flatnonzero(~finite)[0]))
    if not chosen:
        return [lo]
    return sorted(chosen)


def _trim(indices: list[int], costs: list[float], max_points: int) -> tuple[list[int], list[float]]:
    keep = np.unique(np.linspace(0, len(indices) - 1, max_points).astype(int))
    total = float(np.nansum(costs))
    trimmed = [0.0] * len(keep)
    trimmed[-1] = total if np.isfinite(costs).any() else float("nan")
    return [indices[int(pos)] for pos in keep], trimmed


def _bucket_cost(cost: np.ndarray) -> float:
    if not np.isfinite(cost).any():
        return float("nan")
    return float(np.nansum(cost))
