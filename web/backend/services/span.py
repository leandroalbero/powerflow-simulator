"""Inclusive local-day spans. A plain YYYY-MM-DD end covers that whole day."""

import pandas as pd

from src.data.frame import to_utc_bound


def inclusive_day_count(start: str, end: str) -> int:
    start_ts = to_utc_bound(start, is_end=False)
    end_ts = to_utc_bound(end, is_end=True)
    if start_ts is None or end_ts is None or start_ts >= end_ts:
        raise ValueError(f"start ({start}) must not be after end ({end})")
    return int((end_ts - start_ts) / pd.Timedelta(days=1))
