"""Inclusive local-day spans. A plain YYYY-MM-DD end covers that whole day."""

import math
import re
from datetime import date

import pandas as pd

from src.data.frame import to_utc_bound

_PLAIN_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def inclusive_day_count(start: str, end: str) -> int:
    start_text = start.strip()
    end_text = end.strip()
    if _PLAIN_DATE.fullmatch(start_text) and _PLAIN_DATE.fullmatch(end_text):
        start_day = date.fromisoformat(start_text)
        end_day = date.fromisoformat(end_text)
        if start_day > end_day:
            raise ValueError(f"start ({start}) must not be after end ({end})")
        return (end_day - start_day).days + 1
    start_ts = to_utc_bound(start, is_end=False)
    end_ts = to_utc_bound(end, is_end=True)
    if start_ts is None or end_ts is None or start_ts >= end_ts:
        raise ValueError(f"start ({start}) must not be after end ({end})")
    # Floor drops a day when a spring-forward makes the UTC span an hour short.
    span_days = float((end_ts - start_ts) / pd.Timedelta(days=1))
    return math.ceil(span_days - 1e-9)
