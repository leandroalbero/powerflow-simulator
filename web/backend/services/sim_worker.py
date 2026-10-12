"""Picklable sweep/sim case. Imported by spawn workers — must not touch app state."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from web.backend.models.schemas import SystemConfig
from web.backend.services.cancel import RunCancelled


@dataclass
class CasePayload:
    frame: pd.DataFrame
    strategy_id: str
    config: SystemConfig
    pvpc_prices: Optional[dict[Any, float]] = None
    history: Optional[pd.DataFrame] = None
    daily_forecasts: Optional[dict[Any, Any]] = None
    cancel: Any = None
    progress: Any = None
    job_id: str = ""


@dataclass
class CaseOutput:
    job_id: str
    strategy_id: str
    status: str
    metrics: Optional[dict[str, Any]] = None
    timeseries: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def execute_case(payload: CasePayload) -> CaseOutput:
    """Rebuild the strategy inside the worker and run it. Cancel is cooperative."""
    from web.backend.services.simulation_service import MPC_VARIANTS, RunInputs, SimulationService

    svc = SimulationService.__new__(SimulationService)
    inputs = RunInputs(frame=payload.frame, history=payload.history, daily_forecasts=payload.daily_forecasts)
    if payload.strategy_id in MPC_VARIANTS and MPC_VARIANTS[payload.strategy_id].solar != "perfect":
        from src.domain.strategy.irradiance import HourlyGhi, load_irradiance

        inputs.ghi = HourlyGhi(load_irradiance())
    try:
        tariff = SimulationService._build_tariff(svc, payload.config, prices=payload.pvpc_prices)
        strategy = SimulationService._build_strategy(
            svc, payload.strategy_id, payload.config, inputs, tariff,
        )

        def progress(pct: int) -> None:
            if payload.cancel is not None and payload.cancel.is_set():
                raise RunCancelled()
            if payload.progress is not None:
                payload.progress.put((payload.job_id, payload.strategy_id, pct))

        from src.domain.energy_simulator.models import run as simulate

        sim = simulate(payload.frame, strategy, tariff, on_progress=progress)
    except RunCancelled:
        return CaseOutput(payload.job_id, payload.strategy_id, "cancelled")
    except Exception as exc:  # noqa: BLE001 — worker boundary; solvers raise varied errors
        return CaseOutput(payload.job_id, payload.strategy_id, "failed", error=str(exc) or type(exc).__name__)
    return CaseOutput(
        payload.job_id,
        payload.strategy_id,
        "completed",
        metrics=sim.metrics,
        timeseries=sim.timeseries,
    )
