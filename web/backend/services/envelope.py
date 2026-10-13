"""Bucketed min/max envelope downsampler.

Each bucket emits the per-column argmin and argmax of the power series, plus
one NaN index for a power column that has a gap, so a spike is not dropped
because another series is larger. ``soc_pct`` and prices are read at those
rows; they do not add indices of their own. ``step_cost`` is the sum of finite
costs in the bucket, attached to the later emitted sample. Bucket count follows
the realized union so ``max_points`` is a cap, not a 3-per-column reservation.
"""

from dataclasses import dataclass

import numpy as np

_POWER = (
    "solar_power",
    "house_consumption",
    "grid_import",
    "grid_export",
    "battery_charge",
    "battery_discharge",
    "battery_level",
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

    power = [key for key in _POWER if key in cols] or list(cols)
    n_power = len(power)
    if max_points < 2 * n_power:
        n_buckets = max(1, min(n, max_points // 2 or 1))
        out_idx, out_cost = _priority_collect(ts, cols, cost, power, n_buckets, max_points)
    else:
        n_buckets, out_idx, out_cost = _fit_buckets(ts, cols, cost, power, max_points)

    idx = np.asarray(out_idx, dtype=np.intp)
    gathered = {key: values[idx].copy() for key, values in cols.items()}
    span_min = max(int(ts[-1]) - int(ts[0]), 0) / 60_000.0
    resolution = span_min / n_buckets if n_buckets else 1.0
    return Envelope(ts[idx], gathered, np.asarray(out_cost, dtype=np.float64), True, float(resolution))


def _fit_buckets(ts, cols, cost, power, max_points: int):
    """Largest bucket count whose realized union still fits in ``max_points``."""
    n = len(ts)
    limit = min(n, max_points)
    best_n = 1
    best_idx, best_cost = _collect(ts, cols, cost, power, 1)
    if len(best_idx) > max_points:
        trimmed_idx, trimmed_cost = _priority_collect(ts, cols, cost, power, 1, max_points)
        return 1, trimmed_idx, trimmed_cost
    lo, hi = 1, limit
    while lo <= hi:
        mid = (lo + hi) // 2
        idx, bucket_cost = _collect(ts, cols, cost, power, mid)
        if len(idx) <= max_points:
            best_n, best_idx, best_cost = mid, idx, bucket_cost
            lo = mid + 1
        else:
            hi = mid - 1
    return best_n, best_idx, best_cost


def _bounds(ts: np.ndarray, n_buckets: int) -> np.ndarray:
    t0 = int(ts[0])
    t1 = int(ts[-1])
    edges = np.rint(np.linspace(t0, t1, n_buckets + 1)).astype(np.int64)
    edges[-1] = t1 + 1
    return np.searchsorted(ts, edges, side="left")


def _collect(ts, cols, cost, keys, n_buckets: int):
    bounds = _bounds(ts, n_buckets)
    out_idx: list[int] = []
    out_cost: list[float] = []
    for bucket in range(n_buckets):
        lo = int(bounds[bucket])
        hi = int(bounds[bucket + 1])
        if lo >= hi:
            continue
        chosen = _union(cols, keys, lo, hi)
        _append_bucket(out_idx, out_cost, chosen, _bucket_cost(cost[lo:hi]))
    return out_idx, out_cost


def _priority_collect(ts, cols, cost, keys, n_buckets: int, max_points: int):
    bounds = _bounds(ts, n_buckets)
    buckets: list[tuple[int, int, list[int]]] = []
    for bucket in range(n_buckets):
        lo = int(bounds[bucket])
        hi = int(bounds[bucket + 1])
        if lo >= hi:
            continue
        buckets.append((lo, hi, _ranked(cols, keys, lo, hi)))
    selected: list[list[int]] = [[] for _ in buckets]
    remaining = max_points
    depth = 0
    while remaining > 0 and buckets:
        added = False
        for i, (_lo, _hi, ordered) in enumerate(buckets):
            if remaining <= 0 or depth >= len(ordered):
                continue
            index = ordered[depth]
            if index in selected[i]:
                continue
            selected[i].append(index)
            remaining -= 1
            added = True
        if not added:
            break
        depth += 1
    out_idx: list[int] = []
    out_cost: list[float] = []
    for i, (lo, hi, _ordered) in enumerate(buckets):
        if not selected[i]:
            continue
        _append_bucket(out_idx, out_cost, sorted(selected[i]), _bucket_cost(cost[lo:hi]))
    return out_idx, out_cost


def _append_bucket(out_idx: list[int], out_cost: list[float], chosen: list[int], bucket_cost: float) -> None:
    last = len(chosen) - 1
    for pos, index in enumerate(chosen):
        out_idx.append(index)
        out_cost.append(bucket_cost if pos == last else 0.0)


def _union(cols: dict[str, np.ndarray], keys: list[str], lo: int, hi: int) -> list[int]:
    return sorted(set(_ranked(cols, keys, lo, hi)))


def _ranked(cols: dict[str, np.ndarray], keys: list[str], lo: int, hi: int) -> list[int]:
    ranked: list[tuple[int, int]] = []
    for priority, key in enumerate(keys):
        values = cols[key][lo:hi]
        finite = np.isfinite(values)
        if finite.any():
            ranked.append((priority * 2, lo + int(np.argmin(np.where(finite, values, np.inf)))))
            ranked.append((priority * 2 + 1, lo + int(np.argmax(np.where(finite, values, -np.inf)))))
        if not finite.all():
            ranked.append((1000 + priority, lo + int(np.flatnonzero(~finite)[0])))
    ranked.sort()
    ordered: list[int] = []
    seen: set[int] = set()
    for _priority, index in ranked:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    if not ordered:
        ordered.append(lo)
    return ordered


def _bucket_cost(cost: np.ndarray) -> float:
    if not np.isfinite(cost).any():
        return float("nan")
    return float(np.nansum(cost))
