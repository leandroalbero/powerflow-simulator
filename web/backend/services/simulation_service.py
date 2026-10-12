"""Wraps domain simulation code, runs strategies in background threads, stores results."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import pytz

from src.data.frame import to_utc_bound
from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import run as simulate
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.power_tariff.pvpc import PvpcTariff
from src.domain.strategy.forecast_loader import load_daily_forecasts
from src.domain.strategy.irradiance import HourlyGhi, load_irradiance
from src.domain.strategy.model import (
    BaseRateAwareStrategy,
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    ForecastChargeStrategy,
    NoBatteryStrategy,
    SelfConsumeStrategy,
    SmartDischargeStrategy,
    ValleyChargePeakDischargeStrategy,
)
from src.domain.strategy.mpc import MPC_VARIANTS, MpcStrategy, build_mpc
from src.domain.strategy.oracle import OracleStrategy
from web.backend.models.schemas import (
    StrategyInfo,
    SystemConfig,
)
from web.backend.services.data_service import DataService
from web.backend.services.pvpc_service import PvpcService

LOCAL_TZ = pytz.timezone("Europe/Madrid")
logger = logging.getLogger(__name__)

# --- Strategy registry ---

STRATEGY_REGISTRY: List[StrategyInfo] = [
    StrategyInfo(
        id="no_battery",
        name="No Battery",
        description="Baseline without a battery: solar covers load directly, the rest is "
        "imported or exported. Every other strategy is compared against this.",
        group="core",
    ),
    StrategyInfo(
        id="self_consume",
        name="Self Consume",
        description="Maximise self-consumption of solar energy. "
        "Excess solar charges battery, deficit discharges battery, then imports from grid.",
        group="core",
    ),
    StrategyInfo(
        id="charge_night",
        name="Force Charge at Night",
        description="Charge the battery from grid during cheapest night-time valley rate. "
        "Discharge during the day to reduce peak-rate imports.",
        group="core",
    ),
    StrategyInfo(
        id="force_valleys",
        name="Force Charge at Valleys",
        description="Charge battery from grid during pre-peak shoulder periods. "
        "Discharge only during peak-rate hours.",
        group="advanced",
    ),
    StrategyInfo(
        id="force_valleys_pre_peak",
        name="Force Charge Valleys & Pre-Peak",
        description="Charge battery during valley and pre-peak shoulder windows. "
        "Discharge during peak hours for maximum savings.",
        group="advanced",
    ),
    StrategyInfo(
        id="forecast_charge",
        name="Forecast Charge",
        description="Uses day-ahead solar irradiance forecasts to set night charge target. "
        "Skips charging on sunny days, charges fully on cloudy days.",
        group="core",
    ),
    StrategyInfo(
        id="smart_discharge",
        name="Smart Discharge",
        description="Charge at valley, discharge morning shoulder + evening peak. "
        "Morning discharge creates room for solar; holds battery for evening when solar is gone.",
        group="advanced",
    ),
    StrategyInfo(
        id="valley_charge_peak_discharge",
        name="Valley Charge, Peak Discharge",
        description="Charge at cheapest valley rate, discharge only during peak-rate hours. "
        "Preserves battery for highest-value periods instead of wasting on shoulder.",
        group="advanced",
    ),
    StrategyInfo(
        id="oracle",
        name="Oracle Optimizer",
        description="Computes the theoretical minimum cost using linear programming with "
        "perfect foresight. Not deployable — serves as a benchmark.",
        group="core",
    ),
    StrategyInfo(
        id="mpc",
        name="Model Predictive Control",
        description="Rolling 24h LP with solar forecasts and learned load profiles. "
        "Re-solves every 15 minutes. Deployable in real-time.",
        group="core",
    ),
    StrategyInfo(
        id="mpc_calibrated",
        name="MPC (Calibrated Solar)",
        description="MPC with monthly-calibrated GHI forecast. Learns seasonal correction "
        "factors from historical actual-vs-forecast data to fix systematic bias.",
        group="advanced",
    ),
    StrategyInfo(
        id="mpc_5min",
        name="MPC 5-min (Calibrated)",
        description="MPC at 5-min resolution with calibrated solar forecast. "
        "Finer battery control and faster re-solving for better load tracking.",
        group="advanced",
    ),
    StrategyInfo(
        id="mpc_5min_perfect",
        name="MPC 5-min (Perfect Solar)",
        description="MPC at 5-min resolution with perfect solar foresight. "
        "Shows the ceiling for high-resolution MPC.",
        group="advanced",
    ),
    StrategyInfo(
        id="mpc_perfect",
        name="MPC (Perfect Solar)",
        description="MPC with perfect solar foresight — uses actual solar data instead of "
        "GHI forecast. Shows MPC ceiling without forecast error.",
        group="core",
    ),
    StrategyInfo(
        id="dqn_agent",
        name="DQN Agent (experimental)",
        description="Deep Q-Network reinforcement learning agent trained on historical data. "
        "Uses learned policy for charge/discharge decisions. Requires a trained policy file.",
        group="experimental",
    ),
]

STRATEGY_MAP = {s.id: s for s in STRATEGY_REGISTRY}

# Maps strategy id -> domain strategy class
_STRATEGY_CLASSES = {
    "no_battery": NoBatteryStrategy,
    "self_consume": SelfConsumeStrategy,
    "charge_night": ForceChargeAtNightStrategy,
    "force_valleys": ForceChargeAtValleyStrategy,
    "force_valleys_pre_peak": ForceChargeValleyAndPrePeakStrategy,
    "forecast_charge": ForecastChargeStrategy,
    "smart_discharge": SmartDischargeStrategy,
    "valley_charge_peak_discharge": ValleyChargePeakDischargeStrategy,
    "oracle": OracleStrategy,
    "mpc": MpcStrategy,
    "mpc_calibrated": MpcStrategy,
    "mpc_5min": MpcStrategy,
    "mpc_5min_perfect": MpcStrategy,
    "mpc_perfect": MpcStrategy,
    # dqn_agent: lazy-imported in _run_strategy to avoid torch import at startup
}


# --- Run storage ---

@dataclass
class StrategyRunResult:
    strategy_id: str
    status: str = "pending"  # pending | running | completed | failed
    progress: int = 0  # 0-100 percent
    metrics: Optional[Dict[str, Any]] = None
    timeseries: Optional[Dict[str, np.ndarray]] = None
    error: Optional[str] = None


@dataclass
class SimulationRun:
    run_id: str
    status: str = "running"  # running | completed | failed
    strategies: Dict[str, StrategyRunResult] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)
    start: Optional[str] = None
    end: Optional[str] = None
    config: Optional[SystemConfig] = None


# Progress callback type: (strategy_id, percent 0-100)
ProgressCallback = Callable[[str, int], None]
# Strategy-done callback: (strategy_id, status, metrics_or_none)
StrategyDoneCallback = Callable[[str, str, Optional[Dict[str, Any]]], None]
# Run-done callback: ()
RunDoneCallback = Callable[[], None]

HISTORY_DAYS = 60


@dataclass
class RunInputs:
    frame: pd.DataFrame
    history: Optional[pd.DataFrame] = None
    daily_forecasts: Optional[Dict] = None
    ghi: Optional[HourlyGhi] = None
    pvpc_prices: Optional[Dict[datetime, float]] = None


def _end_of_tomorrow(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """23:00 Europe/Madrid on the calendar day after ``now``.

    ``Timedelta(days=2)`` is 48 elapsed hours and is an hour off on DST eves.
    """
    if now is None:
        now = pd.Timestamp.now(tz="Europe/Madrid")
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("Europe/Madrid")
        else:
            now = now.tz_convert("Europe/Madrid")
    day = now.date() + timedelta(days=2)
    return pd.Timestamp(day).tz_localize("Europe/Madrid") - pd.Timedelta(hours=1)


class SimulationService:
    """Manages simulation runs in background threads."""

    def __init__(
        self,
        data_service: DataService,
        pvpc_service: Optional[PvpcService] = None,
    ) -> None:
        self.data_service = data_service
        self.pvpc_service = pvpc_service or PvpcService()
        self.config = SystemConfig()
        self.runs: Dict[str, SimulationRun] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    # --- config ---

    def get_config(self) -> SystemConfig:
        return self.config

    def update_config(self, config: SystemConfig) -> SystemConfig:
        self.config = config
        return self.config

    # --- strategies ---

    @staticmethod
    def list_strategies() -> List[StrategyInfo]:
        return STRATEGY_REGISTRY

    # --- build domain objects from config ---

    def _build_tariff(
        self,
        config: Optional[SystemConfig] = None,
        prices: Optional[Dict[datetime, float]] = None,
    ) -> PowerTariff:
        config = config or self.config
        if config.tariff.use_pvpc and prices is not None:
            export_price = 0.0
            for r in config.tariff.rates:
                if r.direction == "export":
                    export_price = r.price
                    break
            # PvpcTariff is structurally compatible with PowerTariff but is
            # not a subclass (the existing PowerTariff is a dataclass).
            return cast(
                PowerTariff,
                PvpcTariff(
                    prices_eur_kwh=prices,
                    export_price=export_price,
                    export_follows_pvpc=config.tariff.export_follows_pvpc,
                    export_pvpc_discount=config.tariff.export_pvpc_discount,
                ),
            )

        schedule = {}
        for r in config.tariff.rates:
            direction = EnergyDirection.IMPORT if r.direction == "import" else EnergyDirection.EXPORT
            schedule[(r.start_hour, r.end_hour)] = Rate(price=r.price, energy_direction=direction)
        weekend = None
        if config.tariff.weekend_rate is not None:
            weekend = Rate(price=config.tariff.weekend_rate.price, energy_direction=EnergyDirection.IMPORT)
        return PowerTariff(rate_schedule=schedule, weekend_rate=weekend)

    def _build_battery(self, config: Optional[SystemConfig] = None) -> Battery:
        bc = (config or self.config).battery
        return Battery(
            capacity=bc.capacity,
            max_charge_rate=bc.max_charge_rate,
            max_discharge_rate=bc.max_discharge_rate,
            efficiency=bc.efficiency,
            initial_soc=bc.initial_soc,
            taper_start=bc.taper_start,
            taper_factor=bc.taper_factor,
        )

    def _build_grid(self, config: Optional[SystemConfig] = None) -> Grid:
        gc = (config or self.config).grid
        return Grid(max_import=gc.max_import, max_export=gc.max_export)

    def _build_strategy(self, strategy_id: str, config: SystemConfig, inputs: RunInputs, tariff: PowerTariff):
        battery = self._build_battery(config)
        grid = self._build_grid(config)
        sc = config.strategy
        common = dict(min_battery_level=sc.min_battery_level, max_charge_power=sc.max_charge_power)
        if strategy_id in MPC_VARIANTS:
            return build_mpc(
                strategy_id, battery, grid, tariff, inputs.frame, history=inputs.history, ghi=inputs.ghi,
                horizon_hours=config.tariff.mpc_horizon_hours,
                cycle_cost_eur_kwh=config.battery.cycle_cost_eur_kwh, **common,
            )
        if strategy_id == "oracle":
            return OracleStrategy(battery, grid, tariff, inputs.frame,
                                  cycle_cost_eur_kwh=config.battery.cycle_cost_eur_kwh, **common)
        if strategy_id == "dqn_agent":
            from src.domain.strategy.dqn_strategy import load_dqn_strategy

            return load_dqn_strategy(battery, grid, tariff, min_battery_level=sc.min_battery_level)
        if strategy_id == "forecast_charge":
            return ForecastChargeStrategy(battery, grid, tariff, inputs.daily_forecasts or {},
                                          valley_charge_target=sc.valley_charge_target, **common)
        cls = _STRATEGY_CLASSES[strategy_id]
        if issubclass(cls, BaseRateAwareStrategy):
            return cls(battery, grid, tariff, valley_charge_target=sc.valley_charge_target, **common)
        return cls(battery, grid, tariff, **common)

    # --- run simulation ---

    def _history_frame(self, frame: pd.DataFrame, exclude_ev: bool) -> pd.DataFrame:
        """Frame from up to 60 days before the run start to its end, for out-of-sample forecasters."""
        if len(frame) == 0:
            return frame
        start = (frame.index[0] - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
        return self.data_service.get_frame(start, frame.index[-1].strftime("%Y-%m-%d"), exclude_ev=exclude_ev)

    @staticmethod
    def validate_request(strategy_ids: List[str], start: Optional[str], end: Optional[str]) -> None:
        if not strategy_ids:
            raise ValueError("Select at least one strategy")
        unknown = sorted(set(strategy_ids) - set(STRATEGY_MAP))
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("Duplicate strategies in request")
        bounds = []
        for label, value, is_end in (("start", start, False), ("end", end, True)):
            try:
                bounds.append(to_utc_bound(value, is_end=is_end))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid {label} date {value!r}: expected YYYY-MM-DD or ISO datetime") from exc
        # A date-only end covers that whole local day, so compare against the exclusive end.
        if bounds[0] is not None and bounds[1] is not None and bounds[0] >= bounds[1]:
            raise ValueError(f"start ({start}) must not be after end ({end})")

    def _pvpc_window(self, frame: pd.DataFrame, config: SystemConfig,
                     now: Optional[pd.Timestamp] = None) -> Optional[tuple]:
        if len(frame) == 0:
            return None
        lookahead = pd.Timedelta(hours=max(24, config.tariff.mpc_horizon_hours))
        needed_end = frame.index[-1] + lookahead
        # Prices are published for tomorrow at the latest; PvpcTariff falls back for later hours.
        tomorrow_end = _end_of_tomorrow(now)
        end = min(needed_end, tomorrow_end)
        start = frame.index[0]
        if end < start:
            end = start
        return start.to_pydatetime(), end.to_pydatetime()

    def start_run(
        self,
        strategy_ids: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        config: Optional[SystemConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_strategy_done: Optional[StrategyDoneCallback] = None,
        on_run_done: Optional[RunDoneCallback] = None,
    ) -> str:
        """Validate and start a simulation run; returns the run_id. Raises ValueError for bad requests."""
        self.validate_request(strategy_ids, start, end)
        # Snapshot config so threads don't race with PUT /api/config
        config_snapshot = (config or self.config).model_copy(deep=True)

        frame = self.data_service.get_frame(start, end, exclude_ev=config_snapshot.exclude_ev)
        if len(frame) == 0 or not frame["valid"].any():
            raise ValueError(f"No data between {start or 'the beginning'} and {end or 'the end'}")

        run_id = uuid.uuid4().hex[:12]
        run = SimulationRun(run_id=run_id, start=start, end=end, config=config_snapshot)
        for sid in strategy_ids:
            run.strategies[sid] = StrategyRunResult(strategy_id=sid)
        self.runs[run_id] = run

        inputs = RunInputs(frame=frame)
        try:
            if any(s in MPC_VARIANTS for s in strategy_ids):
                inputs.history = self._history_frame(frame, config_snapshot.exclude_ev)
            if "forecast_charge" in strategy_ids:
                inputs.daily_forecasts = load_daily_forecasts()
            if any(MPC_VARIANTS[s].solar != "perfect" for s in strategy_ids if s in MPC_VARIANTS):
                inputs.ghi = HourlyGhi(load_irradiance())
            if config_snapshot.tariff.use_pvpc:
                window = self._pvpc_window(frame, config_snapshot)
                if window is not None:
                    inputs.pvpc_prices = self.pvpc_service.get_prices(*window, strict=False)
        except Exception as exc:
            logger.exception("Preparing run %s failed", run_id)
            self._fail_run(run, strategy_ids, f"{type(exc).__name__}: {exc}", on_strategy_done, on_run_done)
            return run_id

        for sid in strategy_ids:
            self._executor.submit(
                self._run_strategy, run, sid, inputs, config_snapshot,
                on_progress, on_strategy_done, on_run_done, strategy_ids,
            )
        return run_id

    @staticmethod
    def _fail_run(run: SimulationRun, strategy_ids: List[str], error: str,
                  on_strategy_done: Optional[StrategyDoneCallback], on_run_done: Optional[RunDoneCallback]) -> None:
        with run.lock:
            for sid in strategy_ids:
                run.strategies[sid].status = "failed"
                run.strategies[sid].error = error
            run.status = "failed"
        if on_strategy_done is not None:
            for sid in strategy_ids:
                on_strategy_done(sid, "failed", None)
        if on_run_done is not None:
            on_run_done()

    def _run_strategy(
        self,
        run: SimulationRun,
        strategy_id: str,
        inputs: RunInputs,
        config: SystemConfig,
        on_progress: Optional[ProgressCallback],
        on_strategy_done: Optional[StrategyDoneCallback],
        on_run_done: Optional[RunDoneCallback],
        all_strategy_ids: List[str],
    ) -> None:
        result = run.strategies[strategy_id]
        result.status = "running"

        try:
            tariff = self._build_tariff(config, prices=inputs.pvpc_prices)
            strategy = self._build_strategy(strategy_id, config, inputs, tariff)

            def progress(pct: int) -> None:
                result.progress = pct
                if on_progress is not None:
                    on_progress(strategy_id, pct)

            sim = simulate(inputs.frame, strategy, tariff, on_progress=progress)

            with run.lock:
                result.status = "completed"
                result.metrics = sim.metrics
                result.timeseries = sim.timeseries

            if on_strategy_done is not None:
                on_strategy_done(strategy_id, "completed", sim.metrics)

        except Exception as e:
            logger.exception("Strategy %s failed in run %s", strategy_id, run.run_id)
            with run.lock:
                result.status = "failed"
                result.error = str(e) or type(e).__name__

            if on_strategy_done is not None:
                on_strategy_done(strategy_id, "failed", None)

        # Check if all strategies finished (only first thread fires callback)
        should_fire_done = False
        with run.lock:
            all_done = all(
                run.strategies[sid].status in ("completed", "failed")
                for sid in all_strategy_ids
            )
            if all_done and run.status == "running":
                all_failed = all(
                    run.strategies[sid].status == "failed"
                    for sid in all_strategy_ids
                )
                run.status = "failed" if all_failed else "completed"
                should_fire_done = True

        if should_fire_done and on_run_done is not None:
            on_run_done()

    # --- query results ---

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        return self.runs.get(run_id)

    def get_run_timeseries(self, run_id: str, strategy_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Raw per-minute numpy arrays (``ts`` epoch ms; NaN on invalid minutes)."""
        run = self.runs.get(run_id)
        if run is None:
            return None
        result = run.strategies.get(strategy_id)
        if result is None or result.timeseries is None:
            return None
        return result.timeseries
