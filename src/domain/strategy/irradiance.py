"""Open-Meteo irradiance files (``data/irradiance/YYYY-MM.csv``) on a UTC hourly index.

The files' naive ``timestamp`` labels are hour-ENDING means (Open-Meteo
radiation is the mean of the preceding hour) on a FIXED UTC offset, not DST-aware
Madrid time: the March and October files have 744 rows with no DST gap or repeat,
and the daily GHI centroid sits at the same label hour in winter and summer. The
offset of each file is inferred from that centroid against the site's solar
noon, and every value is re-keyed to the UTC start of the hour it covers.
Checked against measured PV: with this mapping the best lag is 0 h in both CET
and CEST months (hourly correlation 0.914, vs 0.892 for the naive local +1 h).
"""

import os
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SITE_LONGITUDE = -0.8654  # same site as scripts/fetch_irradiance.py
_SOLAR_NOON_UTC_H = 12.0 - SITE_LONGITUDE / 15.0


def irradiance_dir() -> Path:
    base = os.environ.get("POWERFLOW_DATA_DIR")
    return (Path(base) if base else REPO_ROOT / "data") / "irradiance"


def _label_offset_hours(labels: pd.Series, ghi: pd.Series) -> int:
    hour = labels.dt.hour + labels.dt.minute / 60.0
    weight = ghi.fillna(0.0).clip(lower=0.0)
    if weight.sum() <= 0:
        return 2
    centroid = float((hour * weight).sum() / weight.sum())
    return int(round(centroid - 0.5 - _SOLAR_NOON_UTC_H))


def load_irradiance(directory: Optional[Path] = None) -> pd.DataFrame:
    """All irradiance files as one frame indexed by the UTC start of each covered hour."""
    directory = Path(directory) if directory is not None else irradiance_dir()
    parts = []
    for path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(path)
        if "timestamp" not in df.columns or "ghi" not in df.columns:
            continue
        labels = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        offset = _label_offset_hours(labels, pd.to_numeric(df["ghi"], errors="coerce"))
        start = (labels - pd.Timedelta(hours=offset + 1)).dt.tz_localize("UTC")
        cols = {c: pd.to_numeric(df[c], errors="coerce") for c in ("ghi", "ghi_forecast") if c in df.columns}
        parts.append(pd.DataFrame({k: v.to_numpy() for k, v in cols.items()}, index=pd.DatetimeIndex(start)))
    if not parts:
        return pd.DataFrame({"ghi": pd.Series(dtype=float), "ghi_forecast": pd.Series(dtype=float)},
                            index=pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "ghi_forecast" not in out.columns:
        out["ghi_forecast"] = np.nan
    return out


class HourlyGhi:
    """Dense hourly GHI forecast lookup with a month/hour climatology for missing hours."""

    def __init__(self, irradiance: pd.DataFrame, column: str = "ghi_forecast"):
        series = irradiance[column].dropna() if column in irradiance.columns else pd.Series(dtype=float)
        if series.empty:
            self._base = 0
            self._values = np.zeros(0)
            self._clim = np.zeros((13, 24))
            return
        clim = series.groupby([series.index.month, series.index.hour]).mean()
        self._clim = np.zeros((13, 24))
        for key, v in clim.items():
            m, h = cast(tuple[int, int], key)
            self._clim[m, h] = v
        hours = series.index.asi8 // 3_600_000_000_000
        self._base = int(hours[0])
        self._values = np.full(int(hours[-1]) - self._base + 1, np.nan)
        self._values[hours - self._base] = series.to_numpy()

    def at(self, ts_ns: np.ndarray) -> np.ndarray:
        """GHI (W/m2) for the UTC hours containing ``ts_ns`` (int64 ns), climatology where missing."""
        hours = ts_ns // 3_600_000_000_000
        pos = hours - self._base
        out = np.full(len(ts_ns), np.nan)
        ok = (pos >= 0) & (pos < len(self._values))
        out[ok] = self._values[pos[ok]]
        missing = np.isnan(out)
        if missing.any():
            utc = pd.DatetimeIndex(ts_ns[missing], tz="UTC")
            out[missing] = self._clim[utc.month, utc.hour]
        return out


def daily_forecast_sums(irradiance: pd.DataFrame, tz: str = "Europe/Madrid") -> Dict:
    """{local date: sum of the day's hourly ghi_forecast (Wh/m2)}."""
    s = irradiance["ghi_forecast"].dropna()
    if s.empty:
        return {}
    local = s.index.tz_convert(tz)
    sums = s.groupby(local.date).sum()
    return {d: float(v) for d, v in sums.items()}
