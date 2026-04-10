"""MPC strategy: rolling-horizon LP with solar/load forecasts."""

import csv
from datetime import date as date_type, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


class LoadForecaster:
    """Builds load forecasts from historical averages per (hour, is_weekend)."""

    def __init__(self, load_df: pd.DataFrame) -> None:
        self._profile: dict[tuple[int, bool], float] = {}
        if load_df.empty:
            return
        df = load_df.copy()
        df["hour"] = df.index.hour
        df["is_weekend"] = df.index.weekday >= 5
        grouped = df.groupby(["hour", "is_weekend"])["state"].mean()
        for (hour, is_wknd), val in grouped.items():
            self._profile[(int(hour), bool(is_wknd))] = float(val) / 1000.0  # W -> kW

    def forecast_24h(
        self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15,
    ) -> np.ndarray:
        forecast = np.zeros(steps)
        for i in range(steps):
            ts = start + timedelta(minutes=i * step_minutes)
            key = (ts.hour, ts.weekday() >= 5)
            forecast[i] = self._profile.get(key, 0.0)
        return forecast


GHI_TO_PV_FACTOR = 4.592  # W/m2 GHI -> W PV (empirical)


class SolarForecaster:
    """Converts hourly GHI forecasts to kW PV output for a 24h horizon."""

    def __init__(self, hourly_forecasts: dict[date_type, dict[int, float]]) -> None:
        self._forecasts = hourly_forecasts

    def forecast_24h(
        self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15,
    ) -> np.ndarray:
        forecast = np.zeros(steps)
        for i in range(steps):
            ts = start + timedelta(minutes=i * step_minutes)
            day_forecast = self._forecasts.get(ts.date())
            if day_forecast is None:
                continue
            ghi = day_forecast.get(ts.hour, 0.0)
            forecast[i] = ghi * GHI_TO_PV_FACTOR / 1000.0  # W -> kW
        return forecast


IRRADIANCE_DIR = Path("data/irradiance")


def load_hourly_ghi_forecasts(
    irradiance_dir: Path = IRRADIANCE_DIR,
) -> dict[date_type, dict[int, float]]:
    forecasts: dict[date_type, dict[int, float]] = {}
    for csv_path in sorted(irradiance_dir.glob("*.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "ghi_forecast" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw = row.get("ghi_forecast", "").strip()
                if not raw:
                    continue
                dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                d = dt.date()
                h = dt.hour
                if d not in forecasts:
                    forecasts[d] = {}
                forecasts[d][h] = float(raw)
    return forecasts
