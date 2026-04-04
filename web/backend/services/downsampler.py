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


def downsample_timeseries(
    timestamps: List[str],
    columns: dict,
    max_points: int = 2000,
) -> Tuple[List[str], dict]:
    """Downsample multiple parallel timeseries columns.

    Parameters
    ----------
    timestamps : list of ISO timestamp strings
    columns : dict mapping column name -> list[float]
    max_points : target point count

    Returns
    -------
    (timestamps_out, columns_out) with each list trimmed to <= max_points
    """
    n = len(timestamps)
    if n <= max_points:
        return timestamps, columns

    x = np.arange(n, dtype=np.float64)

    # Use the column with the highest variance as the reference for index selection
    # so the most interesting features are preserved.
    best_var = -1.0
    ref_y: np.ndarray = np.zeros(n)
    for col_values in columns.values():
        arr = np.asarray(col_values, dtype=np.float64)
        v = float(np.var(arr))
        if v > best_var:
            best_var = v
            ref_y = arr

    x_down, _ = lttb_downsample(x, ref_y, max_points)
    indices = x_down.astype(int)

    ts_out = [timestamps[i] for i in indices]
    cols_out = {}
    for name, values in columns.items():
        arr = np.asarray(values, dtype=np.float64)
        cols_out[name] = arr[indices].tolist()

    return ts_out, cols_out
