import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from src.data.frame import to_utc_bound
from web.backend.models.schemas import (
    RunSummary,
    SimulationRequest,
    SimulationStartResponse,
    StrategyMetrics,
    StrategyResult,
    TimeseriesResponse,
)
from web.backend.services.downsampler import downsample_indices
from web.backend.services.simulation_service import SimulationService

router = APIRouter(prefix="/api", tags=["simulation"])


# --- REST endpoints ---


@router.post("/simulate", response_model=SimulationStartResponse)
def start_simulation(body: SimulationRequest, request: Request) -> SimulationStartResponse:
    sim_service: SimulationService = request.app.state.sim_service

    start = body.date_range.start if body.date_range else None
    end = body.date_range.end if body.date_range else None

    try:
        run_id = sim_service.start_run(
            strategy_ids=body.strategies,
            start=start,
            end=end,
            config=body.config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SimulationStartResponse(run_id=run_id, status="running")


@router.get("/runs/{run_id}/summary", response_model=RunSummary)
def get_run_summary(run_id: str, request: Request) -> RunSummary:
    sim_service: SimulationService = request.app.state.sim_service
    run = sim_service.get_run(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    strategies = []
    for sid, result in run.strategies.items():
        metrics = None
        if result.metrics is not None:
            metrics = StrategyMetrics(**result.metrics)
        strategies.append(
            StrategyResult(
                strategy_id=sid,
                status=result.status,
                metrics=metrics,
                error=result.error,
            )
        )

    return RunSummary(run_id=run.run_id, status=run.status, strategies=strategies)


@router.get("/runs/{run_id}/timeseries", response_model=TimeseriesResponse)
def get_timeseries(
    run_id: str,
    request: Request,
    strategy: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    max_points: int = Query(2000, ge=10, le=50000),
) -> TimeseriesResponse:
    sim_service: SimulationService = request.app.state.sim_service
    ts_data = sim_service.get_run_timeseries(run_id, strategy)

    if ts_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No timeseries for run={run_id}, strategy={strategy}",
        )

    ts_ms = ts_data["ts"]
    columns = {
        "battery_level": ts_data["battery_level_kwh"],
        "grid_import": ts_data["grid_import_kw"],
        "grid_export": ts_data["grid_export_kw"],
        "solar_power": ts_data["solar_kw"],
        "house_consumption": ts_data["load_kw"],
    }

    mask = np.ones(len(ts_ms), dtype=bool)
    try:
        if start:
            mask &= ts_ms >= _epoch_ms(start, is_end=False)
        if end:
            mask &= ts_ms < _epoch_ms(end, is_end=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid time filter: {exc}")
    ts_ms = ts_ms[mask]
    columns = {k: np.asarray(v, dtype=float)[mask] for k, v in columns.items()}

    # Skip downsampling for single-day windows (~1440 points at 1-min resolution)
    if len(ts_ms) > 1500:
        idx = downsample_indices(columns, max_points)
        ts_ms = ts_ms[idx]
        columns = {k: v[idx] for k, v in columns.items()}

    timestamps = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("Europe/Madrid")
    return TimeseriesResponse(
        timestamps=[t.isoformat() for t in timestamps],
        point_count=len(ts_ms),
        **{k: _nan_to_none(v) for k, v in columns.items()},
    )


def _epoch_ms(value: str, is_end: bool) -> int:
    """Same bound convention as /simulate: a date-only end is the whole local day; the end is exclusive."""
    bound = to_utc_bound(value, is_end=is_end)
    if bound is None:
        raise ValueError(f"empty bound {value!r}")
    return int(bound.value // 1_000_000)


def _nan_to_none(values: np.ndarray) -> List[Optional[float]]:
    """JSON has no NaN (Starlette serialises with allow_nan=False): gaps become null."""
    return [None if v != v else float(v) for v in values.tolist()]


# --- WebSocket for live progress ---


@router.websocket("/ws/runs/{run_id}")
async def ws_run_progress(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    sim_service: SimulationService = websocket.app.state.sim_service
    run = sim_service.get_run(run_id)

    if run is None:
        await websocket.send_json({"error": f"Run {run_id} not found"})
        await websocket.close()
        return

    # If the run is already done, send final state immediately
    if run.status in ("completed", "failed"):
        await _send_final_state(websocket, run)
        await websocket.close()
        return

    # Poll for progress updates.
    # In a production system you would use an asyncio.Queue fed by callbacks,
    # but polling is simpler and sufficient for this use case.
    last_pcts: Dict[str, int] = {}
    sent_done: set = set()

    try:
        while True:
            changed = False
            # One snapshot under the run lock: the service finalises run.status under the same
            # lock after the last strategy, so a finished run here implies every strategy's
            # done message (with its metrics) goes out in this pass, before run_complete.
            with run.lock:
                run_done = run.status in ("completed", "failed")
                snapshot = [(sid, r.status, r.progress, r.metrics, r.error) for sid, r in run.strategies.items()]

            for sid, status, progress, metrics, error in snapshot:
                if status == "running":
                    if progress != last_pcts.get(sid, -1):
                        last_pcts[sid] = progress
                        await websocket.send_text(json.dumps({"strategy": sid, "percent": progress}))
                        changed = True
                elif status in ("completed", "failed") and sid not in sent_done:
                    msg: Dict[str, Any] = {"strategy": sid, "status": status}
                    if metrics is not None:
                        msg["metrics"] = metrics
                    if error is not None:
                        msg["error"] = error
                    await websocket.send_text(json.dumps(msg))
                    sent_done.add(sid)
                    changed = True

            if run_done:
                await websocket.send_json({"status": "run_complete"})
                break

            if not changed:
                await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _send_final_state(websocket: WebSocket, run) -> None:  # type: ignore[no-untyped-def]
    for sid, result in run.strategies.items():
        msg: Dict[str, Any] = {"strategy": sid, "status": result.status}
        if result.metrics is not None:
            msg["metrics"] = result.metrics
        if result.error is not None:
            msg["error"] = result.error
        await websocket.send_text(json.dumps(msg))

    await websocket.send_json({"status": "run_complete"})
