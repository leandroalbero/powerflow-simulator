"""MPC strategy: rolling-horizon LP with out-of-sample solar/load forecasts.

Forecasts only use information available at decision time: the load profile
comes from the 28 days before the plan's local day, the GHI-to-PV correction
from history before the run start, and the GHI forecast is the day-ahead one.
Horizon prices are pure ``tariff.rate_at`` lookups.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.holidays import is_offpeak_day
from src.domain.power_tariff.model import EnergyDirection, PowerTariff
from src.domain.strategy.irradiance import HourlyGhi, load_irradiance
from src.domain.strategy.model import GHI_TO_PV_FACTOR, BaseEnergyStrategy, Decision
from src.domain.strategy.oracle import solve_oracle_lp

LOCAL_TZ = "Europe/Madrid"
LOAD_PROFILE_DAYS = 28
_NS_PER_MIN = 60_000_000_000
_NS_PER_HOUR = 3_600_000_000_000
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def _step_index(start: pd.Timestamp, steps: int, step_minutes: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize(LOCAL_TZ)
    return pd.DatetimeIndex(start + pd.to_timedelta(np.arange(steps) * step_minutes, unit="min")).tz_convert(LOCAL_TZ)


def _as_kw_series(data: Union[pd.DataFrame, pd.Series], column: str) -> pd.Series:
    """Accept a kW series or a legacy DataFrame with a ``state`` column in W."""
    if isinstance(data, pd.Series):
        s = data.astype(float)
    elif column in data.columns:
        s = data[column].astype(float)
    else:
        s = data["state"].astype(float) / 1000.0
    if len(s) and s.index.tz is None:
        s = s.tz_localize(LOCAL_TZ)
    return s.tz_convert(LOCAL_TZ) if len(s) else s


class LoadForecaster:
    """Hourly load profile per (hour, off-peak day) from the 28 days strictly before the plan's local day.

    The profile is recomputed once per local day.
    """

    def __init__(self, load: Union[pd.DataFrame, pd.Series], days: int = LOAD_PROFILE_DAYS) -> None:
        s = _as_kw_series(load, "load_kw").dropna()
        self._days = days
        self._cache: Dict[date, Dict[Tuple[int, bool], float]] = {}
        if s.empty:
            self._hourly = pd.DataFrame(columns=["kw", "date", "hour", "offpeak"])
            return
        hourly = s.resample("1h").mean().dropna()
        dates = hourly.index.date
        self._hourly = pd.DataFrame({
            "kw": hourly.to_numpy(), "date": dates, "hour": hourly.index.hour,
            "offpeak": [is_offpeak_day(d) for d in dates],
        })

    def profile(self, day: date) -> Dict[Tuple[int, bool], float]:
        prof = self._cache.get(day)
        if prof is not None:
            return prof
        h = self._hourly
        lo = date.fromordinal(day.toordinal() - self._days)
        window = h[(h["date"] >= lo) & (h["date"] < day)] if len(h) else h
        prof = {}
        if len(window):
            by_key = window.groupby(["hour", "offpeak"])["kw"].mean()
            by_hour = window.groupby("hour")["kw"].mean()
            overall = float(window["kw"].mean())
            for hour in range(24):
                for off in (False, True):
                    v = by_key.get((hour, off))
                    if v is None or np.isnan(v):
                        v = by_hour.get(hour, overall)
                    prof[(hour, off)] = float(v)
        self._cache[day] = prof
        return prof

    def forecast_24h(self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15) -> np.ndarray:
        times = _step_index(start, steps, step_minutes)
        prof = self.profile(times[0].date())
        if not prof:
            return np.zeros(steps)
        table = np.array([[prof[(h, off)] for h in range(24)] for off in (False, True)])
        wall = times.tz_localize(None).asi8
        hours = (wall // _NS_PER_HOUR) % 24
        days = wall // (24 * _NS_PER_HOUR)
        offpeak = {d: is_offpeak_day(date.fromordinal(_EPOCH_ORDINAL + int(d))) for d in np.unique(days)}
        return table[np.array([offpeak[d] for d in days], dtype=int), hours]


class SolarForecaster:
    """Day-ahead GHI forecast -> PV kW, optionally with monthly correction factors."""

    def __init__(self, ghi: HourlyGhi, monthly_factors: Optional[Dict[int, float]] = None,
                 ghi_to_pv_factor: float = GHI_TO_PV_FACTOR) -> None:
        self._ghi = ghi
        self._factors = np.ones(13)
        for m, f in (monthly_factors or {}).items():
            self._factors[m] = f
        self._k = ghi_to_pv_factor / 1000.0

    def forecast_24h(self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15) -> np.ndarray:
        times = _step_index(start, steps, step_minutes)
        # A step can straddle an hour boundary; average the minutes it covers.
        offsets = np.arange(step_minutes) * _NS_PER_MIN
        ns = times.asi8[:, None] + offsets[None, :]
        ghi = self._ghi.at(ns.ravel()).reshape(ns.shape).mean(axis=1)
        return ghi * self._k * self._factors[times.month]


class CalibratedSolarForecaster(SolarForecaster):
    """SolarForecaster with monthly factors learned from history (see learn_monthly_ghi_factors)."""


class PerfectSolarForecaster:
    """Actual PV for backtesting: the mean of the measured minutes in each step."""

    def __init__(self, pv: Union[pd.DataFrame, pd.Series]) -> None:
        s = _as_kw_series(pv, "pv_kw").fillna(0.0)
        if len(s):
            s = s.resample("1min").mean().fillna(0.0)
        self._start_ns = int(s.index.asi8[0]) if len(s) else 0
        self._cum = np.concatenate(([0.0], np.cumsum(s.to_numpy(dtype=float))))

    def forecast_24h(self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15) -> np.ndarray:
        n = len(self._cum) - 1
        first = (pd.Timestamp(start).value - self._start_ns) // _NS_PER_MIN
        edges = np.clip(first + np.arange(steps + 1) * step_minutes, 0, n)
        width = np.diff(edges)
        sums = self._cum[edges[1:]] - self._cum[edges[:-1]]
        return np.where(width > 0, sums / np.maximum(width, 1), 0.0)


def learn_monthly_ghi_factors(pv: Union[pd.DataFrame, pd.Series], ghi: HourlyGhi,
                              before: Optional[pd.Timestamp] = None,
                              ghi_to_pv_factor: float = GHI_TO_PV_FACTOR) -> Dict[int, float]:
    """Monthly actual_pv / (ghi_forecast * factor) from PV strictly before ``before``; 1.0 without data."""
    s = _as_kw_series(pv, "pv_kw").dropna()
    if before is not None and len(s):
        s = s[s.index < pd.Timestamp(before)]
    factors = {m: 1.0 for m in range(1, 13)}
    if s.empty:
        return factors
    hourly = s.tz_convert("UTC").resample("1h").mean().dropna()
    predicted = ghi.at(hourly.index.asi8) * ghi_to_pv_factor / 1000.0
    actual = hourly.to_numpy()
    day = (predicted > 0.01) | (actual > 0.01)
    months = hourly.index.tz_convert(LOCAL_TZ).month
    df = pd.DataFrame({"m": months[day], "a": actual[day], "p": predicted[day]})
    for m, g in df.groupby("m"):
        if g["p"].sum() > 0:
            factors[int(m)] = float(g["a"].sum() / g["p"].sum())
    return factors


class MpcStrategy(BaseEnergyStrategy):
    """Rolling-horizon LP re-solved every ``resolve_minutes`` on forecasts; the plan's net
    battery command is applied to the actual battery through ``_settle``."""

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        load_forecaster: LoadForecaster,
        solar_forecaster,
        step_minutes: int = 15,
        resolve_minutes: int = 15,
        horizon_hours: int = 24,
        cycle_cost_eur_kwh: float = 0.0,
        terminal_value_eur_kwh: Optional[float] = None,
        min_battery_level: float = 0.1,
        max_charge_power: Optional[float] = None,
    ) -> None:
        super().__init__(battery, grid, tariff, min_battery_level, max_charge_power)
        self._load_forecaster = load_forecaster
        self._solar_forecaster = solar_forecaster
        self._step_minutes = step_minutes
        self._resolve_minutes = resolve_minutes
        self._horizon_steps = horizon_hours * 60 // step_minutes
        self._cycle_cost_eur_kwh = cycle_cost_eur_kwh
        self._terminal_value_eur_kwh = terminal_value_eur_kwh
        self._cached_plan: Optional[Dict[str, np.ndarray]] = None
        self._plan_start: Optional[pd.Timestamp] = None
        self._prices: Dict[int, Tuple[float, float]] = {}
        self._solve_count = 0

    def _horizon_prices(self, times: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        imp = np.empty(len(times))
        exp = np.empty(len(times))
        for i, (key, ts) in enumerate(zip(times.asi8 // _NS_PER_HOUR, times)):
            rates = self._prices.get(key)
            if rates is None:
                rates = (self.tariff.rate_at(ts, EnergyDirection.IMPORT).price,
                         self.tariff.rate_at(ts, EnergyDirection.EXPORT).price)
                self._prices[key] = rates
            imp[i], exp[i] = rates
        return imp, exp

    def _should_resolve(self) -> bool:
        if self._cached_plan is None or self._plan_start is None:
            return True
        elapsed = (pd.Timestamp(self.now) - self._plan_start).total_seconds() / 60.0
        return elapsed >= self._resolve_minutes or elapsed < 0

    def _solve_horizon(self, solar_now: float, load_now: float) -> None:
        ts = pd.Timestamp(self.now)
        n, sm = self._horizon_steps, self._step_minutes
        solar_fc = np.asarray(self._solar_forecaster.forecast_24h(ts, n, sm), dtype=float)
        load_fc = np.asarray(self._load_forecaster.forecast_24h(ts, n, sm), dtype=float)
        # The controller measures the current power, so the first step uses it.
        solar_fc[0], load_fc[0] = solar_now, load_now
        imp, exp = self._horizon_prices(_step_index(ts, n, sm))
        b = self.battery
        tv = self._terminal_value_eur_kwh
        if tv is None:
            tv = float(imp.mean()) * b.discharge_efficiency
        result = solve_oracle_lp(
            solar=solar_fc, load=load_fc, import_rates=imp, export_rates=exp, dt=sm / 60.0,
            battery_capacity=b.capacity, max_charge_rate=b.max_charge_rate,
            max_discharge_rate=b.max_discharge_rate, efficiency=b.charge_efficiency,
            discharge_efficiency=b.discharge_efficiency,
            initial_soc=b.current_charge / b.capacity, min_soc_frac=self.min_battery_level,
            max_grid_import=self.grid.max_import, max_grid_export=self.grid.max_export,
            cycle_cost_eur_kwh=self._cycle_cost_eur_kwh, terminal_value_eur_kwh=tv,
            taper_start=b.taper_start, taper_factor=b.taper_factor, max_grid_charge=self.grid_charge_power,
        )
        if result.success:
            self._cached_plan = {"solar": solar_fc, "load": load_fc,
                                 "charge": result.charge, "discharge": result.discharge}
        self._plan_start = ts
        self._solve_count += 1

    def _plan_step(self) -> int:
        elapsed = (pd.Timestamp(self.now) - self._plan_start).total_seconds() / 60.0
        return min(int(elapsed // self._step_minutes), self._horizon_steps - 1)

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self.now is None:
            return self._self_consume(solar_kw, load_kw)
        if self._should_resolve():
            self._solve_horizon(solar_kw, load_kw)
        if self._cached_plan is None:
            return self._self_consume(solar_kw, load_kw)
        t = self._plan_step()
        plan = self._cached_plan
        charge = float(plan["charge"][t])
        planned_surplus = max(0.0, float(plan["solar"][t] - plan["load"][t]))
        return Decision(charge - float(plan["discharge"][t]), allow_grid_charge=charge > planned_surplus + 1e-6)


@dataclass(frozen=True)
class MpcVariant:
    solar: str  # "ghi" | "calibrated" | "perfect"
    step_minutes: int


MPC_VARIANTS: Dict[str, MpcVariant] = {
    "mpc": MpcVariant("ghi", 15),
    "mpc_calibrated": MpcVariant("calibrated", 15),
    "mpc_5min": MpcVariant("calibrated", 5),
    "mpc_5min_perfect": MpcVariant("perfect", 5),
    "mpc_perfect": MpcVariant("perfect", 15),
}


def build_mpc(
    strategy_id: str,
    battery: Battery,
    grid: Grid,
    tariff: PowerTariff,
    frame: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    ghi: Optional[HourlyGhi] = None,
    horizon_hours: int = 24,
    cycle_cost_eur_kwh: float = 0.0,
    min_battery_level: float = 0.1,
    max_charge_power: Optional[float] = None,
) -> MpcStrategy:
    """Build one of the MPC variants for a run over ``frame``.

    ``history`` is a frame covering (ideally) 60 days before the run start up to
    its end; the load forecaster reads only the days before each plan and the
    solar calibration only the data before ``frame``'s start.
    """
    variant = MPC_VARIANTS[strategy_id]
    hist = history if history is not None else frame
    load_hist = hist["load_kw"].where(hist["valid"])
    if variant.solar == "perfect":
        solar = PerfectSolarForecaster(frame["pv_kw"])
    else:
        ghi = ghi if ghi is not None else HourlyGhi(load_irradiance())
        factors = None
        if variant.solar == "calibrated":
            pv_hist = hist["pv_kw"].where(hist["valid"])
            factors = learn_monthly_ghi_factors(pv_hist, ghi, before=frame.index[0] if len(frame) else None)
        solar = SolarForecaster(ghi, factors)
    return MpcStrategy(
        battery, grid, tariff,
        load_forecaster=LoadForecaster(load_hist),
        solar_forecaster=solar,
        step_minutes=variant.step_minutes,
        resolve_minutes=variant.step_minutes,
        horizon_hours=horizon_hours,
        cycle_cost_eur_kwh=cycle_cost_eur_kwh,
        min_battery_level=min_battery_level,
        max_charge_power=max_charge_power,
    )
