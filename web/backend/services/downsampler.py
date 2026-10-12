"""Largest-Triangle-Three-Buckets (LTTB) downsampling implemented with numpy."""

from typing import List, Tuple

import numpy as np


def lttb_downsample(
    x: np.ndarray,
    y: np.ndarray,
    threshold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample parallel x/y arrays to *threshold* points using LTTB.

    If the data already has <= threshold points, it is returned as-is.

    Parameters
    ----------
    x : 1-D array (typically epoch seconds or an ordinal sequence)
    y : 1-D array of the same length
    threshold : desired number of output points (>= 2)

    Returns
    -------
    (x_out, y_out) : downsampled arrays
    """
    n = len(x)
    if threshold >= n or threshold < 2:
        return x, y
    if threshold == 2:
        return x[[0, n - 1]], y[[0, n - 1]]

    # Always keep first and last point
    indices = np.empty(threshold, dtype=np.intp)
    indices[0] = 0
    indices[threshold - 1] = n - 1

    bucket_size = (n - 2) / (threshold - 2)

    a_index = 0  # index of the previously selected point

    for i in range(1, threshold - 1):
        # Calculate the average point for the *next* bucket
        next_bucket_start = int((i + 0) * bucket_size) + 1
        next_bucket_end = int((i + 1) * bucket_size) + 1
        if next_bucket_end > n - 1:
            next_bucket_end = n - 1

        avg_x = np.mean(x[next_bucket_start : next_bucket_end + 1])
        avg_y = np.mean(y[next_bucket_start : next_bucket_end + 1])

        # Current bucket range
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1

        # Triangle area for each point in the current bucket
        # Using the cross-product formula for triangle area
        ax = float(x[a_index])
        ay = float(y[a_index])

        candidates_x = x[bucket_start:bucket_end]
        candidates_y = y[bucket_start:bucket_end]

        areas = np.abs(
            (ax - avg_x) * (candidates_y - ay)
            - (ax - candidates_x) * (avg_y - ay)
        )

        best = int(np.argmax(areas))
        indices[i] = bucket_start + best
        a_index = indices[i]

    return x[indices], y[indices]


def downsample_indices(columns: dict, max_points: int = 2000) -> np.ndarray:
    """Row indices to keep so every column fits in about ``max_points`` points.

    LTTB runs on the valid (non-NaN) rows of the highest-variance column; the
    first row of every NaN run is kept as well so charts still break at gaps.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in columns.values()]
    n = len(arrays[0]) if arrays else 0
    if n <= max_points:
        return np.arange(n)

    best_var = -1.0
    ref_y = np.zeros(n)
    for arr in arrays:
        v = float(np.nanvar(arr)) if np.isfinite(arr).any() else 0.0
        if v > best_var:
            best_var = v
            ref_y = arr

    finite = np.isfinite(ref_y)
    gap_starts = np.flatnonzero(~finite & np.concatenate(([True], finite[:-1])))
    valid_idx = np.flatnonzero(finite)
    budget = max(2, max_points - len(gap_starts))
    if len(valid_idx) > budget:
        x_down, _ = lttb_downsample(valid_idx.astype(np.float64), ref_y[valid_idx], budget)
        keep = x_down.astype(np.intp)
    else:
        keep = valid_idx
    return np.union1d(keep, gap_starts)


def downsample_timeseries(
    timestamps: List[str],
    columns: dict,
    max_points: int = 2000,
) -> Tuple[List[str], dict]:
    """Downsample parallel timeseries columns (NaN-aware); see :func:`downsample_indices`."""
    idx = downsample_indices(columns, max_points)
    if len(idx) == len(timestamps):
        return timestamps, columns
    return ([timestamps[i] for i in idx],
            {name: np.asarray(values, dtype=np.float64)[idx].tolist() for name, values in columns.items()})
