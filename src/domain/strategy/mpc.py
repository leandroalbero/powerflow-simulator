"""MPC strategy: rolling-horizon LP with solar/load forecasts."""

import csv
from datetime import date as date_type, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow
from src.domain.strategy.oracle import solve_oracle_lp


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


def learn_monthly_ghi_factors(
    solar_df: pd.DataFrame,
    hourly_forecasts: dict[date_type, dict[int, float]],
) -> dict[int, float]:
    """Learn monthly correction factors: actual_pv / (ghi_forecast * GHI_TO_PV_FACTOR).

    Returns dict mapping month (1-12) -> correction multiplier.
    """
    solar_h = solar_df[["state"]].resample("1h").mean().fillna(0.0)
    monthly_actual: dict[int, float] = {}
    monthly_predicted: dict[int, float] = {}

    for ts in solar_h.index:
        day_fc = hourly_forecasts.get(ts.date())
        if day_fc is None:
            continue
        ghi = day_fc.get(ts.hour, 0.0)
        predicted_pv = ghi * GHI_TO_PV_FACTOR
        actual_pv = float(solar_h.loc[ts, "state"])
        if predicted_pv < 10 and actual_pv < 10:
            continue  # skip nighttime
        m = ts.month
        monthly_actual[m] = monthly_actual.get(m, 0.0) + actual_pv
        monthly_predicted[m] = monthly_predicted.get(m, 0.0) + predicted_pv

    factors: dict[int, float] = {}
    for m in range(1, 13):
        pred = monthly_predicted.get(m, 0.0)
        if pred > 0:
            factors[m] = monthly_actual.get(m, 0.0) / pred
        else:
            factors[m] = 1.0
    return factors


class CalibratedSolarForecaster:
    """GHI forecast with monthly correction factors learned from historical data.

    Corrects the systematic seasonal bias in the fixed GHI_TO_PV_FACTOR
    (winter underestimate ~1.4-1.7x, summer overestimate ~0.84-0.93x).
    """

    def __init__(
        self,
        hourly_forecasts: dict[date_type, dict[int, float]],
        monthly_factors: dict[int, float],
    ) -> None:
        self._forecasts = hourly_forecasts
        self._factors = monthly_factors

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
            factor = self._factors.get(ts.month, 1.0)
            forecast[i] = ghi * GHI_TO_PV_FACTOR * factor / 1000.0  # W -> kW
        return forecast


class PerfectSolarForecaster:
    """Uses actual solar data for backtesting — perfect foresight solar forecast.

    Resamples to the requested step resolution on first use of that resolution.
    """

    def __init__(self, solar_df: pd.DataFrame) -> None:
        self._solar_df = solar_df
        self._cache: dict[int, tuple[np.ndarray, pd.Timestamp]] = {}

    def _get_resampled(self, step_minutes: int) -> tuple[np.ndarray, pd.Timestamp]:
        if step_minutes not in self._cache:
            resampled = self._solar_df[["state"]].resample(f"{step_minutes}min").mean().fillna(0.0)
            arr = (resampled["state"] / 1000.0).values
            start = resampled.index[0] if len(resampled) > 0 else None
            self._cache[step_minutes] = (arr, start)
        return self._cache[step_minutes]

    def forecast_24h(
        self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15,
    ) -> np.ndarray:
        arr, start_ts = self._get_resampled(step_minutes)
        if start_ts is None or len(arr) == 0:
            return np.zeros(steps)
        offset_min = (start - start_ts).total_seconds() / 60.0
        start_idx = int(round(offset_min / step_minutes))
        end_idx = start_idx + steps
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(arr):
            end_idx = len(arr)
        valid = end_idx - start_idx
        if valid <= 0:
            return np.zeros(steps)
        forecast = np.zeros(steps)
        forecast[:valid] = arr[start_idx:end_idx]
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


MPC_HORIZON_STEPS = 96     # 24h at 15-min resolution
MPC_STEP_MINUTES = 15
MPC_RESOLVE_MINUTES = 15


class MpcStrategy(BaseEnergyStrategy):
    """Rolling-horizon LP strategy with forecast-based planning.

    Re-solves a 24h LP at configurable intervals using solar/load forecasts.
    Between solves, replays the cached plan.
    """

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        load_forecaster: LoadForecaster,
        solar_forecaster: SolarForecaster,
        step_minutes: int = 15,
        resolve_minutes: int = 15,
        horizon_hours: int = 24,
    ) -> None:
        super().__init__(battery, grid, tariff)
        self._load_forecaster = load_forecaster
        self._solar_forecaster = solar_forecaster
        self._step_minutes = step_minutes
        self._resolve_minutes = resolve_minutes
        self._horizon_steps = horizon_hours * 60 // step_minutes
        self._cached_plan: dict[str, np.ndarray] | None = None
        self._plan_start: pd.Timestamp | None = None
        self._current_ts: pd.Timestamp | None = None
        self._solve_count = 0

    def set_timestamp(self, ts: pd.Timestamp) -> None:
        """Called by the simulator before each step to set the current time."""
        self._current_ts = ts

    def _should_resolve(self) -> bool:
        if self._cached_plan is None or self._plan_start is None:
            return True
        elapsed = (self._current_ts - self._plan_start).total_seconds() / 60.0
        return elapsed >= self._resolve_minutes

    def _solve_horizon(self) -> None:
        ts = self._current_ts
        n = self._horizon_steps
        sm = self._step_minutes
        solar_fc = self._solar_forecaster.forecast_24h(ts, n, sm)
        load_fc = self._load_forecaster.forecast_24h(ts, n, sm)

        step_times = [ts + timedelta(minutes=i * sm) for i in range(n)]
        hours = np.array([t.hour for t in step_times])

        # Build rates with weekend awareness (Spain 2.0TD: valley all day on weekends)
        import_rates = np.empty(n)
        export_rates = np.empty(n)
        for i, t in enumerate(step_times):
            self.tariff.update_datetime(t)
            import_rates[i] = self.tariff.get_import_rate(t.hour)
            export_rates[i] = self.tariff.get_export_rate(t.hour)

        dt = sm / 60.0

        result = solve_oracle_lp(
            solar=solar_fc,
            load=load_fc,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=self.battery.capacity,
            max_charge_rate=self.battery.max_charge_rate,
            max_discharge_rate=self.battery.max_discharge_rate,
            efficiency=self.battery.efficiency,
            initial_soc=self.battery.current_charge / self.battery.capacity,
            min_soc_frac=self.min_battery_level,
            max_grid_import=self.grid.max_import,
            max_grid_export=self.grid.max_export,
        )

        if result.success:
            self._cached_plan = {
                "charge": result.charge,
                "discharge": result.discharge,
                "grid_import": result.grid_import,
                "grid_export": result.grid_export,
                "soc": result.soc,
            }
        self._plan_start = ts
        self._solve_count += 1

    def _get_plan_step(self) -> int:
        if self._plan_start is None:
            return 0
        elapsed_min = (self._current_ts - self._plan_start).total_seconds() / 60.0
        return min(int(elapsed_min / self._step_minutes), self._horizon_steps - 1)

    def calculate_energy_flows(
        self, solar_power: float, load_power: float, hour: int, duration: float,
    ) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        if self._current_ts is not None and self._should_resolve():
            self._solve_horizon()

        if self._cached_plan is None:
            flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration, force_discharge=True)
            return flows

        t = self._get_plan_step()
        plan = self._cached_plan
        flows = EnergyFlow()

        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        flows.battery_charge = float(plan["charge"][t])
        flows.battery_discharge = float(plan["discharge"][t])
        flows.grid_import = float(plan["grid_import"][t])
        flows.grid_export = float(plan["grid_export"][t])

        self.battery.current_charge = float(plan["soc"][t])

        return flows
