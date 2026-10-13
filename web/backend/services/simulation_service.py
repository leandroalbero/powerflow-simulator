"""Wraps domain simulation code, runs strategies in background threads, stores results."""

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Optional, cast
from zoneinfo import ZoneInfo

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
from web.backend.services.cancel import RunCancelled
from web.backend.services.client_errors import public_error
from web.backend.services.data_service import DataService
from web.backend.services.history_store import HistoryStore, config_digest
from web.backend.services.pvpc_service import PvpcService

LOCAL_TZ = pytz.timezone("Europe/Madrid")
logger = logging.getLogger(__name__)
_DQN_POLICY = Path(__file__).resolve().parents[3] / "output_files" / "dqn_policy.pt"
_TERMINAL = ("completed", "failed", "cancelled")
_TS_LIMIT = 8

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
    status: str = "running"  # running | completed | failed | cancelled
    strategies: Dict[str, StrategyRunResult] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)
    start: Optional[str] = None
    end: Optional[str] = None
    config: Optional[SystemConfig] = None
    label: Optional[str] = None
    created_at: str = ""
    cancel: Event = field(default_factory=Event)
    futures: Dict[str, Any] = field(default_factory=dict)
    deleted: bool = False


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


class RunStillActive(Exception):
    def __init__(self) -> None:
        super().__init__("run still active, cancel first")


def _run_active(run: SimulationRun) -> bool:
    if run.status == "running":
        return True
    return any(result.status in ("pending", "running") for result in run.strategies.values())


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
        self._registry_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.history = HistoryStore()
        self._ts_order: OrderedDict[str, None] = OrderedDict()
        self.pool = None
        self._progress_q = None

    # --- config ---

    def get_config(self) -> SystemConfig:
        return self.config

    def update_config(self, config: SystemConfig) -> SystemConfig:
        self.config = config
        return self.config

    # --- strategies ---

    @staticmethod
    def list_strategies() -> List[StrategyInfo]:
        missing = not _DQN_POLICY.exists()
        reason = "DQN policy not trained: output_files/dqn_policy.pt missing" if missing else None
        listed = []
        for info in STRATEGY_REGISTRY:
            if info.id == "dqn_agent" and missing:
                listed.append(info.model_copy(update={"available": False, "unavailable_reason": reason}))
            else:
                listed.append(info.model_copy(update={"available": True, "unavailable_reason": None}))
        return listed

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

    def _build_strategy(
        self,
        strategy_id: str,
        config: SystemConfig,
        inputs: RunInputs,
        tariff: PowerTariff,
        on_chunk: Optional[Callable[[], None]] = None,
    ):
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
            return OracleStrategy(
                battery, grid, tariff, inputs.frame,
                cycle_cost_eur_kwh=config.battery.cycle_cost_eur_kwh, on_chunk=on_chunk, **common,
            )
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
        label: Optional[str] = None,
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
        created_at = datetime.now(ZoneInfo("Europe/Madrid")).isoformat()
        run = SimulationRun(
            run_id=run_id, start=start, end=end, config=config_snapshot,
            label=label, created_at=created_at,
        )
        if self.pool is not None:
            run.cancel = self.pool.manager.Event()
        for sid in strategy_ids:
            run.strategies[sid] = StrategyRunResult(strategy_id=sid)
        with self._registry_lock:
            self.runs[run_id] = run

        try:
            inputs = self.prepare_inputs(frame, strategy_ids, config_snapshot)
        except Exception:
            public_error("simulation failed")
            self._fail_run(run, strategy_ids, "simulation failed", on_strategy_done, on_run_done)
            return run_id

        if self.pool is not None:
            for sid in strategy_ids:
                self._submit_pooled(run, sid, inputs, config_snapshot, strategy_ids)
        else:
            for sid in strategy_ids:
                fut = self._executor.submit(
                    self._run_strategy, run, sid, inputs, config_snapshot,
                    on_progress, on_strategy_done, on_run_done, strategy_ids,
                )
                run.futures[sid] = fut
        return run_id

    def prepare_inputs(
        self,
        frame: pd.DataFrame,
        strategy_ids: List[str],
        config: SystemConfig,
        *,
        load_pvpc: bool = True,
    ) -> RunInputs:
        """History, irradiance and forecasts shared by /simulate, sweep and backtest."""
        inputs = RunInputs(frame=frame)
        if any(sid in MPC_VARIANTS for sid in strategy_ids):
            inputs.history = self._history_frame(frame, config.exclude_ev)
        if "forecast_charge" in strategy_ids:
            inputs.daily_forecasts = load_daily_forecasts()
        if any(MPC_VARIANTS[sid].solar != "perfect" for sid in strategy_ids if sid in MPC_VARIANTS):
            inputs.ghi = HourlyGhi(load_irradiance())
        if load_pvpc and config.tariff.use_pvpc:
            window = self._pvpc_window(frame, config)
            if window is not None:
                inputs.pvpc_prices = self.pvpc_service.get_prices(*window, strict=False)
        return inputs

    def attach_pool(self, pool) -> None:
        """Use the process pool created on the main thread. Progress comes back on a manager queue."""
        self.pool = pool
        self._progress_q = pool.manager.Queue()
        threading.Thread(target=self._drain_progress, args=(self._progress_q,), daemon=True).start()

    def _drain_progress(self, queue) -> None:
        while True:
            try:
                item = queue.get()
            except (EOFError, OSError):
                return
            if item is None:
                return
            run_id, strategy_id, pct = item
            run = self.runs.get(run_id)
            if run is None or strategy_id not in run.strategies:
                continue
            run.strategies[strategy_id].progress = pct

    def _submit_pooled(self, run: SimulationRun, strategy_id: str, inputs: RunInputs,
                       config: SystemConfig, all_strategy_ids: List[str]) -> None:
        from web.backend.services.sim_worker import CasePayload, execute_case

        result = run.strategies[strategy_id]
        with run.lock:
            if run.cancel.is_set() or result.status in _TERMINAL:
                if result.status not in _TERMINAL:
                    result.status = "cancelled"
                self._finish_if_done(run, all_strategy_ids)
                return
            result.status = "running"
        payload = CasePayload(
            frame=inputs.frame,
            strategy_id=strategy_id,
            config=config,
            pvpc_prices=inputs.pvpc_prices,
            history=inputs.history,
            daily_forecasts=inputs.daily_forecasts,
            cancel=run.cancel,
            progress=self._progress_q,
            job_id=run.run_id,
            ghi=inputs.ghi,
        )
        try:
            fut = self.pool.executor.submit(execute_case, payload)
        except BrokenProcessPool:
            public_error("worker process crashed")
            self._recreate_pool()
            with run.lock:
                if result.status not in _TERMINAL:
                    result.status = "failed"
                    result.error = "worker process crashed"
                self._finish_if_done(run, all_strategy_ids)
            return
        run.futures[strategy_id] = fut

        def _done(future) -> None:
            self._apply_output(run, strategy_id, future, all_strategy_ids)

        fut.add_done_callback(_done)

    def _recreate_pool(self) -> None:
        pool = self.pool
        if pool is not None:
            pool.recreate()

    def _apply_output(self, run: SimulationRun, strategy_id: str, future, all_strategy_ids: List[str]) -> None:
        result = run.strategies[strategy_id]
        try:
            output = future.result()
        except CancelledError:
            output_status, error, metrics, timeseries = "cancelled", None, None, None
        except BrokenProcessPool:
            public_error("worker process crashed")
            self._recreate_pool()
            output_status, error, metrics, timeseries = "failed", "worker process crashed", None, None
        except Exception:  # noqa: BLE001 — pool boundary
            output_status, error, metrics, timeseries = "failed", public_error("simulation failed"), None, None
        else:
            output_status, error = output.status, output.error
            metrics, timeseries = output.metrics, output.timeseries
        with run.lock:
            if result.status in _TERMINAL:
                self._finish_if_done(run, all_strategy_ids)
                return
            result.status = output_status
            result.error = error
            result.metrics = metrics
            result.timeseries = timeseries
            if output_status == "completed":
                result.progress = 100
            self._finish_if_done(run, all_strategy_ids)

    def _finish_if_done(self, run: SimulationRun, all_strategy_ids: List[str]) -> None:
        if not all(run.strategies[sid].status in _TERMINAL for sid in all_strategy_ids):
            return
        if run.status != "running":
            return
        statuses = [run.strategies[sid].status for sid in all_strategy_ids]
        if any(status == "cancelled" for status in statuses):
            run.status = "cancelled"
        elif all(status == "failed" for status in statuses):
            run.status = "failed"
        else:
            run.status = "completed"
        self._persist_run(run)
        self._retain_timeseries(run.run_id)

    def _fail_run(self, run: SimulationRun, strategy_ids: List[str], error: str,
                  on_strategy_done: Optional[StrategyDoneCallback], on_run_done: Optional[RunDoneCallback]) -> None:
        with run.lock:
            # A cancel (and maybe a delete) can land while prepare_inputs runs.
            if run.deleted or run.status in _TERMINAL:
                return
            failed = [sid for sid in strategy_ids if run.strategies[sid].status not in _TERMINAL]
            for sid in failed:
                run.strategies[sid].status = "failed"
                run.strategies[sid].error = error
            run.status = "failed"
            self._persist_run(run)
        if on_strategy_done is not None:
            for sid in failed:
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
        with run.lock:
            if result.status in _TERMINAL or run.cancel.is_set():
                if result.status not in _TERMINAL:
                    result.status = "cancelled"
                skipped = True
            else:
                result.status = "running"
                skipped = False
        if skipped:
            self._finish_thread(run, all_strategy_ids, on_run_done)
            return

        try:
            tariff = self._build_tariff(config, prices=inputs.pvpc_prices)

            def on_chunk() -> None:
                if run.cancel.is_set():
                    raise RunCancelled()

            strategy = self._build_strategy(strategy_id, config, inputs, tariff, on_chunk=on_chunk)

            def progress(pct: int) -> None:
                if run.cancel.is_set():
                    raise RunCancelled()
                result.progress = pct
                if on_progress is not None:
                    on_progress(strategy_id, pct)

            sim = simulate(inputs.frame, strategy, tariff, on_progress=progress)

            with run.lock:
                if result.status in _TERMINAL or run.cancel.is_set():
                    if result.status not in _TERMINAL:
                        result.status = "cancelled"
                    finished = "cancelled"
                else:
                    result.status = "completed"
                    result.metrics = sim.metrics
                    result.timeseries = sim.timeseries
                    finished = "completed"

            if on_strategy_done is not None:
                on_strategy_done(strategy_id, finished, sim.metrics if finished == "completed" else None)

        except RunCancelled:
            with run.lock:
                result.status = "cancelled"
                result.error = None
            if on_strategy_done is not None:
                on_strategy_done(strategy_id, "cancelled", None)
        except Exception:
            failed = False
            with run.lock:
                if result.status not in _TERMINAL:
                    result.status = "failed"
                    result.error = "simulation failed"
                    failed = True
            if failed:
                public_error("simulation failed")
                if on_strategy_done is not None:
                    on_strategy_done(strategy_id, "failed", None)

        self._finish_thread(run, all_strategy_ids, on_run_done)

    def _finish_thread(
        self,
        run: SimulationRun,
        all_strategy_ids: List[str],
        on_run_done: Optional[RunDoneCallback],
    ) -> None:
        should_fire_done = False
        with run.lock:
            all_done = all(run.strategies[sid].status in _TERMINAL for sid in all_strategy_ids)
            if all_done and run.status == "running":
                statuses = [run.strategies[sid].status for sid in all_strategy_ids]
                if any(status == "cancelled" for status in statuses):
                    run.status = "cancelled"
                elif all(status == "failed" for status in statuses):
                    run.status = "failed"
                else:
                    run.status = "completed"
                should_fire_done = True
                self._persist_run(run)

        if should_fire_done:
            self._retain_timeseries(run.run_id)
            if on_run_done is not None:
                on_run_done()

    # --- query results ---

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        run = self.runs.get(run_id)
        if run is not None:
            return run
        return self._hydrate(run_id)

    def get_run_timeseries(self, run_id: str, strategy_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Raw per-minute numpy arrays (``ts`` epoch ms; NaN on invalid minutes)."""
        status, data = self.fetch_timeseries(run_id, strategy_id)
        return data if status == "ok" else None

    def fetch_timeseries(self, run_id: str, strategy_id: str) -> tuple[str, Optional[Dict[str, np.ndarray]]]:
        run = self.runs.get(run_id)
        if run is not None:
            result = run.strategies.get(strategy_id)
            if result is None:
                return "missing", None
            if result.timeseries is not None:
                return "ok", result.timeseries
            if result.status in ("pending", "running"):
                return "not_ready", None
            if result.status == "completed":
                return "evicted", None
            return "missing", None
        row = self.history.get_run(run_id)
        if row is None:
            return "missing", None
        match = next((item for item in row["strategies"] if item["strategy_id"] == strategy_id), None)
        if match is None:
            return "missing", None
        if match.get("status") == "completed":
            return "evicted", None
        return "missing", None

    def cancel_run(self, run_id: str) -> Optional[str]:
        run = self.runs.get(run_id)
        if run is None:
            row = self.history.get_run(run_id)
            return None if row is None else row["status"]
        run.cancel.set()
        for fut in list(run.futures.values()):
            fut.cancel()
        with run.lock:
            for sid, result in run.strategies.items():
                fut = run.futures.get(sid)
                queued = fut is None or fut.cancelled()
                if result.status in ("pending", "running") and queued:
                    result.status = "cancelled"
            self._finish_if_done(run, list(run.strategies))
        return run.status

    def delete_run(self, run_id: str) -> bool:
        with self._registry_lock:
            run = self.runs.get(run_id)
            if run is not None:
                # Status changes and their persist happen under run.lock, so checking and
                # flagging here closes the window where a finishing run is re-inserted.
                with run.lock:
                    if _run_active(run):
                        raise RunStillActive()
                    run.deleted = True
            else:
                row = self.history.get_run(run_id)
                if row is not None and row["status"] == "running":
                    raise RunStillActive()
            removed = self.history.delete_run(run_id)
            if run_id in self.runs:
                del self.runs[run_id]
                self._ts_order.pop(run_id, None)
                removed = True
            return removed

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in self.history.list_runs(limit):
            merged[row["run_id"]] = _public_run(row)
        with self._registry_lock:
            runs = list(self.runs.values())
        for run in runs:
            merged[run.run_id] = _public_run_memory(run)
        items = sorted(merged.values(), key=lambda item: item["created_at"], reverse=True)
        return items[:limit]

    def _persist_run(self, run: SimulationRun) -> None:
        if run.config is None or run.deleted:
            return
        self.history.upsert_run(
            run_id=run.run_id,
            label=run.label,
            created_at=run.created_at,
            status=run.status,
            date_start=run.start,
            date_end=run.end,
            config=run.config.model_dump(mode="json"),
            strategies=[_strategy_blob(result) for result in run.strategies.values()],
        )

    def _retain_timeseries(self, run_id: str) -> None:
        self._ts_order[run_id] = None
        self._ts_order.move_to_end(run_id)
        while len(self._ts_order) > _TS_LIMIT:
            old, _ = self._ts_order.popitem(last=False)
            victim = self.runs.get(old)
            if victim is None or victim.status == "running":
                continue
            for result in victim.strategies.values():
                result.timeseries = None

    def _hydrate(self, run_id: str) -> Optional[SimulationRun]:
        row = self.history.get_run(run_id)
        if row is None:
            return None
        config = SystemConfig.model_validate(row["config"])
        run = SimulationRun(
            run_id=row["run_id"],
            status=row["status"],
            start=row["date_range"].start if row["date_range"] else None,
            end=row["date_range"].end if row["date_range"] else None,
            config=config,
            label=row["label"],
            created_at=row["created_at"],
        )
        for item in row["strategies"]:
            run.strategies[item["strategy_id"]] = StrategyRunResult(
                strategy_id=item["strategy_id"],
                status=item["status"],
                progress=100 if item["status"] in _TERMINAL else 0,
                metrics=item.get("metrics"),
                error=item.get("error"),
            )
        return run


def _strategy_blob(result: StrategyRunResult) -> dict[str, Any]:
    return {
        "strategy_id": result.strategy_id,
        "status": result.status,
        "progress": result.progress,
        "metrics": result.metrics,
        "error": result.error,
    }


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "label": row["label"],
        "created_at": row["created_at"],
        "status": row["status"],
        "date_range": row["date_range"],
        "strategies": [
            {
                "strategy_id": item["strategy_id"],
                "status": item["status"],
                "total_cost": (item.get("metrics") or {}).get("total_cost"),
                "adjusted_cost": (item.get("metrics") or {}).get("adjusted_cost"),
            }
            for item in row["strategies"]
        ],
        "config_digest": config_digest(row["config"]),
    }


def _public_run_memory(run: SimulationRun) -> dict[str, Any]:
    date_range = None
    if run.start and run.end:
        from web.backend.models.schemas import DateRange
        date_range = DateRange(start=run.start, end=run.end)
    return _public_run({
        "run_id": run.run_id,
        "label": run.label,
        "created_at": run.created_at,
        "status": run.status,
        "date_range": date_range,
        "config": run.config.model_dump(mode="json") if run.config else {},
        "strategies": [_strategy_blob(result) for result in run.strategies.values()],
    })
