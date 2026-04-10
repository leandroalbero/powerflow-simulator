"""Wraps domain simulation code, runs strategies in background threads, stores results."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

from src.domain.battery.models import Battery
from src.domain.energy_load.model import EnergyLoad
from src.domain.energy_simulator.models import EnergySimulator
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.strategy.forecast_loader import load_daily_forecasts
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    ForecastChargeStrategy,
    SelfConsumeStrategy,
    SmartDischargeStrategy,
    ValleyChargePeakDischargeStrategy,
)
from src.domain.strategy.oracle import OracleStrategy
from web.backend.models.schemas import (
    StrategyInfo,
    SystemConfig,
)
from web.backend.services.data_service import DataService

LOCAL_TZ = pytz.timezone("Europe/Madrid")

# --- Strategy registry ---

STRATEGY_REGISTRY: List[StrategyInfo] = [
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
]

STRATEGY_MAP = {s.id: s for s in STRATEGY_REGISTRY}

# Maps strategy id -> domain strategy class
_STRATEGY_CLASSES = {
    "self_consume": SelfConsumeStrategy,
    "charge_night": ForceChargeAtNightStrategy,
    "force_valleys": ForceChargeAtValleyStrategy,
    "force_valleys_pre_peak": ForceChargeValleyAndPrePeakStrategy,
    "forecast_charge": ForecastChargeStrategy,
    "smart_discharge": SmartDischargeStrategy,
    "valley_charge_peak_discharge": ValleyChargePeakDischargeStrategy,
    "oracle": OracleStrategy,
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

    def __init__(self, data_service: DataService) -> None:
        self.data_service = data_service
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

    def _build_tariff(self, config: Optional[SystemConfig] = None) -> PowerTariff:
        config = config or self.config
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
        battery = Battery(
            capacity=bc.capacity,
            max_charge_rate=bc.max_charge_rate,
            max_discharge_rate=bc.max_discharge_rate,
            efficiency=bc.efficiency,
        )
        battery.current_charge = battery.capacity * bc.initial_soc
        battery._charge_taper_start = bc.taper_start
        battery._taper_factor = bc.taper_factor
        return battery

    def _build_grid(self, config: Optional[SystemConfig] = None) -> Grid:
        gc = (config or self.config).grid
        return Grid(max_import=gc.max_import, max_export=gc.max_export)

    # --- run simulation ---

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
        solar_df, load_df = self.data_service.get_filtered_data(start, end)

        # Load forecasts once if needed (shared across threads)
        daily_forecasts = (
            load_daily_forecasts() if "forecast_charge" in strategy_ids else None
        )

        for sid in strategy_ids:
            self._executor.submit(
                self._run_strategy,
                run,
                sid,
                solar_df,
                load_df,
                config_snapshot,
                on_progress,
                on_strategy_done,
                on_run_done,
                strategy_ids,
                daily_forecasts,
            )

        return run_id

    def _run_strategy(
        self,
        run: SimulationRun,
        strategy_id: str,
        solar_df: pd.DataFrame,
        load_df: pd.DataFrame,
        config: SystemConfig,
        on_progress: Optional[ProgressCallback],
        on_strategy_done: Optional[StrategyDoneCallback],
        on_run_done: Optional[RunDoneCallback],
        all_strategy_ids: List[str],
        daily_forecasts: Optional[Dict] = None,
    ) -> None:
        result = run.strategies[strategy_id]
        result.status = "running"

        try:
            tariff = self._build_tariff(config)
            battery = self._build_battery(config)
            grid = self._build_grid(config)

            load = EnergyLoad(load_df)
            solar = SolarGenerator(solar_df)

            strategy_cls = _STRATEGY_CLASSES.get(strategy_id)
            if strategy_cls is None:
                raise ValueError(f"Unknown strategy: {strategy_id}")

            if strategy_id == "forecast_charge":
                strategy = strategy_cls(battery, grid, tariff, daily_forecasts)
            elif strategy_id == "oracle":
                timestamps = load_df.index
                solar_kw = np.array([
                    solar_df["state"].get(ts, 0.0) / 1000.0 for ts in timestamps
                ])
                load_kw = np.array([
                    load_df["state"].get(ts, 0.0) / 1000.0 for ts in timestamps
                ])
                hours = np.array([ts.hour for ts in timestamps])
                diffs = pd.Series(timestamps).diff().dt.total_seconds() / 3600.0
                diffs.iloc[0] = 1.0 / 60.0
                durations = diffs.values
                strategy = strategy_cls(
                    battery, grid, tariff,
                    solar=solar_kw, load=load_kw,
                    hours=hours, durations=durations,
                )
            else:
                strategy = strategy_cls(battery, grid, tariff)
            # Apply strategy config
            sc = config.strategy
            strategy.min_battery_level = sc.min_battery_level
            strategy.max_charge_power = sc.max_charge_power
            if hasattr(strategy, 'valley_charge_target'):
                strategy.valley_charge_target = sc.valley_charge_target
            sim = EnergySimulator(battery, load, grid, tariff, solar, strategy=strategy)

            timestamps = load_df.index
            total = len(timestamps)
            prev_timestamp = None
            last_reported = -1

            for i, timestamp in enumerate(timestamps):
                if hasattr(strategy, 'current_date'):
                    strategy.current_date = timestamp.date()
                sim.step(timestamp, prev_timestamp)
                prev_timestamp = timestamp

                # Report progress every ~1%
                if total > 0:
                    pct = int((i + 1) * 100 / total)
                    if pct != last_reported:
                        last_reported = pct
                        result.progress = pct
                        if on_progress is not None:
                            on_progress(strategy_id, pct)

            metrics = sim.get_metrics()

            timeseries = {
                "timestamps": [t.isoformat() for t in sim.timestamps],
                "battery_level": list(sim.battery_levels),
                "grid_import": list(sim.grid_imports),
                "grid_export": list(sim.grid_exports),
                "solar_power": list(sim.solar_powers),
                "house_consumption": list(sim.house_loads),
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
