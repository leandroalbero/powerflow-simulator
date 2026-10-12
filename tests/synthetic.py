"""Small synthetic frames for fast simulator/strategy tests."""

import numpy as np
import pandas as pd


def synthetic_frame(start: str = "2026-03-04", days: int = 3, outage: bool = True, spike: bool = True,
                    seed: int = 0) -> pd.DataFrame:
    """1-min frame: sunny / cloudy / sunny days, evening load, an 11 kW spike and a 2 h outage."""
    rng = np.random.default_rng(seed)
    utc0 = pd.Timestamp(start).tz_localize("Europe/Madrid").tz_convert("UTC")
    utc1 = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize("Europe/Madrid").tz_convert("UTC")
    idx = pd.date_range(utc0, utc1, freq="1min", inclusive="left").tz_convert("Europe/Madrid")
    hour = idx.hour + idx.minute / 60.0
    day = (idx.normalize() - idx[0].normalize()).days
    sun = np.clip(np.sin((hour - 7.5) / 11.0 * np.pi), 0.0, None) * 4.8
    cloud = np.where(day % 3 == 1, rng.uniform(0.05, 0.5, len(idx)), rng.uniform(0.85, 1.0, len(idx)))
    pv = sun * cloud
    load = 0.35 + rng.uniform(0.0, 0.3, len(idx))
    load += np.where((hour >= 7) & (hour < 9), 1.2, 0.0)
    load += np.where((hour >= 19) & (hour < 23), 1.8, 0.0)
    if spike:
        load[(day == 0) & (hour >= 20) & (hour < 20 + 20 / 60)] += 9.0
    valid = np.ones(len(idx), dtype=bool)
    if outage and days > 1:
        valid[(day == 1) & (hour >= 10) & (hour < 12)] = False
    load = np.where(valid, load, np.nan)
    frame = pd.DataFrame({"pv_kw": pv, "load_kw": load, "ev_kw": np.zeros(len(idx)), "valid": valid}, index=idx)
    frame.index.name = "ts"
    return frame
