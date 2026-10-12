"""Wraps domain simulation code, runs strategies in background threads, stores results."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, cast

import pandas as pd
import pytz

from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import run as simulate
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.power_tariff.pvpc import PvpcTariff
from src.domain.strategy.forecast_loader import load_daily_forecasts
from src.domain.strategy.irradiance import HourlyGhi, load_irradiance
from src.domain.strategy.model import (
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
    ),
    StrategyInfo(
        id="self_consume",
        name="Self Consume",
        description="Maximise self-consumption of solar energy. "
        "Excess solar charges battery, deficit discharges battery, then imports from grid.",
    ),
    StrategyInfo(
        id="charge_night",
        name="Force Charge at Night",
        description="Charge the battery from grid during cheapest night-time valley rate. "
        "Discharge during the day to reduce peak-rate imports.",
    ),
    StrategyInfo(
        id="force_valleys",
        name="Force Charge at Valleys",
        description="Charge battery from grid during pre-peak shoulder periods. "
        "Discharge only during peak-rate hours.",
    ),
    StrategyInfo(
        id="force_valleys_pre_peak",
        name="Force Charge Valleys & Pre-Peak",
        description="Charge battery during valley and pre-peak shoulder windows. "
        "Discharge during peak hours for maximum savings.",
    ),
    StrategyInfo(
        id="forecast_charge",
        name="Forecast Charge",
        description="Uses day-ahead solar irradiance forecasts to set night charge target. "
        "Skips charging on sunny days, charges fully on cloudy days.",
    ),
    StrategyInfo(
        id="smart_discharge",
        name="Smart Discharge",
        description="Charge at valley, discharge morning shoulder + evening peak. "
        "Morning discharge creates room for solar; holds battery for evening when solar is gone.",
    ),
    StrategyInfo(
        id="valley_charge_peak_discharge",
        name="Valley Charge, Peak Discharge",
        description="Charge at cheapest valley rate, discharge only during peak-rate hours. "
        "Preserves battery for highest-value periods instead of wasting on shoulder.",
    ),
    StrategyInfo(
        id="oracle",
        name="Oracle Optimizer",
        description="Computes the theoretical minimum cost using linear programming with "
        "perfect foresight. Not deployable — serves as a benchmark.",
    ),
    StrategyInfo(
        id="mpc",
        name="Model Predictive Control",
        description="Rolling 24h LP with solar forecasts and learned load profiles. "
        "Re-solves every 15 minutes. Deployable in real-time.",
    ),
    StrategyInfo(
        id="mpc_calibrated",
        name="MPC (Calibrated Solar)",
        description="MPC with monthly-calibrated GHI forecast. Learns seasonal correction "
        "factors from historical actual-vs-forecast data to fix systematic bias.",
    ),
    StrategyInfo(
        id="mpc_5min",
        name="MPC 5-min (Calibrated)",
        description="MPC at 5-min resolution with calibrated solar forecast. "
        "Finer battery control and faster re-solving for better load tracking.",
    ),
    StrategyInfo(
        id="mpc_5min_perfect",
        name="MPC 5-min (Perfect Solar)",
        description="MPC at 5-min resolution with perfect solar foresight. "
        "Shows the ceiling for high-resolution MPC.",
    ),
    StrategyInfo(
        id="mpc_perfect",
        name="MPC (Perfect Solar)",
        description="MPC with perfect solar foresight — uses actual solar data instead of "
        "GHI forecast. Shows MPC ceiling without forecast error.",
    ),
    StrategyInfo(
        id="dqn_agent",
        name="DQN Agent",
        description="Deep Q-Network reinforcement learning agent trained on historical data. "
        "Uses learned policy for charge/discharge decisions.",
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
    timeseries: Optional[Dict[str, list]] = None
    error: Optional[str] = None


@dataclass
class SimulationRun:
    run_id: str
    status: str = "running"  # running | completed | failed
    strategies: Dict[str, StrategyRunResult] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


# Progress callback type: (strategy_id, percent 0-100)
ProgressCallback = Callable[[str, int], None]
# Strategy-done callback: (strategy_id, status, metrics_or_none)
StrategyDoneCallback = Callable[[str, str, Optional[Dict[str, Any]]], None]
# Run-done callback: ()
RunDoneCallback = Callable[[], None]


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
            direction = (
                EnergyDirection.IMPORT
                if r.direction == "import"
                else EnergyDirection.EXPORT
            )
            schedule[(r.start_hour, r.end_hour)] = Rate(
                price=r.price, energy_direction=direction
            )
        weekend = None
        if config.tariff.weekend_rate is not None:
            wr = config.tariff.weekend_rate
            weekend = Rate(
                price=wr.price,
                energy_direction=EnergyDirection.IMPORT if wr.direction == "import" else EnergyDirection.EXPORT,
            )
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

    # --- run simulation ---

    def _history_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Frame from up to 60 days before the run start to its end, for out-of-sample forecasters."""
        if len(frame) == 0:
            return frame
        first = frame.index[0]
        start = (first - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        return self.data_service.get_frame(start, frame.index[-1].strftime("%Y-%m-%d"))

    def start_run(
        self,
        strategy_ids: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_strategy_done: Optional[StrategyDoneCallback] = None,
        on_run_done: Optional[RunDoneCallback] = None,
    ) -> str:
        """Start a simulation run. Returns the run_id."""
        run_id = uuid.uuid4().hex[:12]
        run = SimulationRun(run_id=run_id)
        for sid in strategy_ids:
            run.strategies[sid] = StrategyRunResult(strategy_id=sid)
        self.runs[run_id] = run

        # Snapshot config so threads don't race with PUT /api/config
        config_snapshot = self.config

        # Fetch data once (shared across strategies — read only)
        frame = self.data_service.get_frame(start, end)
        needs_mpc = any(s in MPC_VARIANTS for s in strategy_ids)
        history = self._history_frame(frame) if needs_mpc else None

        # Load forecasts once if needed (shared across threads)
        daily_forecasts = (
            load_daily_forecasts() if "forecast_charge" in strategy_ids else None
        )
        ghi = (
            HourlyGhi(load_irradiance())
            if any(MPC_VARIANTS[s].solar != "perfect" for s in strategy_ids if s in MPC_VARIANTS)
            else None
        )

        # Pre-fetch PVPC prices synchronously so failures short-circuit the
        # whole run with a clear message instead of surfacing per-thread. All
        # strategy threads then share the same price dict.
        pvpc_prices: Optional[Dict[datetime, float]] = None
        if config_snapshot.tariff.use_pvpc and len(frame.index) > 0:
            first_ts = frame.index[0]
            last_ts = frame.index[-1]
            # MPC-style strategies read prices ~24h ahead of the current
            # simulation step. Fetch an extra day so the final simulation day
            # has the lookahead prices available.
            # +2 days minimum, or horizon+1 day whichever is larger (48h MPC
            # horizon needs extra lookahead on the final simulation day).
            horizon_buffer_days = max(2, config_snapshot.tariff.mpc_horizon_hours // 24 + 1)
            last_ts_ext = last_ts + pd.Timedelta(days=horizon_buffer_days)
            try:
                pvpc_prices = self.pvpc_service.get_prices(
                    first_ts.to_pydatetime(), last_ts_ext.to_pydatetime()
                )
            except Exception as exc:
                logger.exception("PVPC fetch failed for run %s", run_id)
                err = f"PVPC fetch failed: {exc}"
                with run.lock:
                    for sid in strategy_ids:
                        sr = run.strategies[sid]
                        sr.status = "failed"
                        sr.error = err
                    run.status = "failed"
                if on_strategy_done is not None:
                    for sid in strategy_ids:
                        on_strategy_done(sid, "failed", None)
                if on_run_done is not None:
                    on_run_done()
                return run_id

        for sid in strategy_ids:
            self._executor.submit(
                self._run_strategy,
                run,
                sid,
                frame,
                config_snapshot,
                on_progress,
                on_strategy_done,
                on_run_done,
                strategy_ids,
                daily_forecasts,
                pvpc_prices,
                history,
                ghi,
            )

        return run_id

    def _run_strategy(
        self,
        run: SimulationRun,
        strategy_id: str,
        frame: pd.DataFrame,
        config: SystemConfig,
        on_progress: Optional[ProgressCallback],
        on_strategy_done: Optional[StrategyDoneCallback],
        on_run_done: Optional[RunDoneCallback],
        all_strategy_ids: List[str],
        daily_forecasts: Optional[Dict] = None,
        pvpc_prices: Optional[Dict[datetime, float]] = None,
        history: Optional[pd.DataFrame] = None,
        ghi: Optional[HourlyGhi] = None,
    ) -> None:
        result = run.strategies[strategy_id]
        result.status = "running"

        try:
            tariff = self._build_tariff(config, prices=pvpc_prices)
            battery = self._build_battery(config)
            grid = self._build_grid(config)

            strategy_cls = _STRATEGY_CLASSES.get(strategy_id)
            if strategy_cls is None and strategy_id != "dqn_agent":
                raise ValueError(f"Unknown strategy: {strategy_id}")

            if strategy_id == "forecast_charge":
                strategy = strategy_cls(battery, grid, tariff, daily_forecasts)
            elif strategy_id in MPC_VARIANTS:
                strategy = build_mpc(
                    strategy_id, battery, grid, tariff, frame, history=history, ghi=ghi,
                    horizon_hours=config.tariff.mpc_horizon_hours,
                    cycle_cost_eur_kwh=config.battery.cycle_cost_eur_kwh,
                    min_battery_level=config.strategy.min_battery_level,
                    max_charge_power=config.strategy.max_charge_power,
                )
            elif strategy_id == "dqn_agent":
                import os

                from src.domain.strategy.dqn.agent import DqnAgent
                from src.domain.strategy.dqn_strategy import DqnStrategy as DqnStrategyCls
                agent = DqnAgent(state_dim=16, action_dim=9)
                model_path = "output_files/dqn_policy.pt"
                if os.path.exists(model_path):
                    agent.load(model_path)
                strategy = DqnStrategyCls(battery, grid, tariff, agent=agent)
            elif strategy_id == "oracle":
                strategy = strategy_cls(
                    battery, grid, tariff, frame,
                    min_battery_level=config.strategy.min_battery_level,
                    max_charge_power=config.strategy.max_charge_power,
                    cycle_cost_eur_kwh=config.battery.cycle_cost_eur_kwh,
                )
            else:
                strategy = strategy_cls(battery, grid, tariff)
            # Apply strategy config
            sc = config.strategy
            strategy.min_battery_level = sc.min_battery_level
            strategy.max_charge_power = sc.max_charge_power
            if hasattr(strategy, 'valley_charge_target'):
                strategy.valley_charge_target = sc.valley_charge_target

            def progress(pct: int) -> None:
                result.progress = pct
                if on_progress is not None:
                    on_progress(strategy_id, pct)

            sim = simulate(frame, strategy, tariff, on_progress=progress)
            metrics = sim.metrics
            ts_arr = sim.timeseries
            keep = frame["valid"].to_numpy()
            timeseries = {
                "timestamps": [t.isoformat() for t in frame.index[keep]],
                "battery_level": ts_arr["battery_level_kwh"][keep].tolist(),
                "grid_import": ts_arr["grid_import_kw"][keep].tolist(),
                "grid_export": ts_arr["grid_export_kw"][keep].tolist(),
                "solar_power": ts_arr["solar_kw"][keep].tolist(),
                "house_consumption": ts_arr["load_kw"][keep].tolist(),
            }

            with run.lock:
                result.status = "completed"
                result.metrics = metrics
                result.timeseries = timeseries

            if on_strategy_done is not None:
                on_strategy_done(strategy_id, "completed", metrics)

        except Exception as e:
            with run.lock:
                result.status = "failed"
                result.error = str(e)

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

    def get_run_timeseries(
        self, run_id: str, strategy_id: str
    ) -> Optional[Dict[str, list]]:
        run = self.runs.get(run_id)
        if run is None:
            return None
        result = run.strategies.get(strategy_id)
        if result is None or result.timeseries is None:
            return None
        return result.timeseries
