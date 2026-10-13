import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from web.backend.models.schemas import (
    RunSummary,
    SimulationRequest,
    SimulationStartResponse,
    StrategyMetrics,
    StrategyResult,
    TimeseriesResponse,
)
from web.backend.services.simulation_service import SimulationService
from web.backend.services.timeseries import present_timeseries

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
            label=body.label,
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
                progress=result.progress,
                metrics=metrics,
                error=result.error,
            )
        )

    date_range = None
    if run.start and run.end:
        from web.backend.models.schemas import DateRange
        date_range = DateRange(start=run.start, end=run.end)
    return RunSummary(
        run_id=run.run_id,
        status=run.status,
        strategies=strategies,
        label=run.label,
        created_at=run.created_at or None,
        date_range=date_range,
        config=run.config,
    )


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
    status, ts_data = sim_service.fetch_timeseries(run_id, strategy)
    if status == "evicted":
        raise HTTPException(status_code=410, detail="timeseries expired, re-run")
    if ts_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No timeseries for run={run_id}, strategy={strategy}",
        )
    run = sim_service.get_run(run_id)
    capacity = 0.0
    if run is not None:
        metrics = run.strategies.get(strategy)
        if metrics is not None and metrics.metrics is not None:
            capacity = float(metrics.metrics.get("battery_capacity") or 0.0)
        elif run.config is not None:
            capacity = run.config.battery.capacity
    try:
        return present_timeseries(ts_data, capacity, start, end, max_points)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid time filter: {exc}") from exc


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

    last_pcts: Dict[str, int] = {}
    sent_done: set[str] = set()
    replay_done = await _replay(websocket, run, last_pcts, sent_done)
    if replay_done:
        await websocket.send_json({"status": "run_complete"})
        await websocket.close()
        return

    try:
        while True:
            # One snapshot under the run lock: the service finalises run.status under the same
            # lock after the last strategy, so a finished run here implies every strategy's
            # done message goes out in this pass, before run_complete.
            with run.lock:
                run_done = run.status in ("completed", "failed", "cancelled")
                snapshot = [(sid, r.status, r.progress) for sid, r in run.strategies.items()]
                results = dict(run.strategies)
            for sid, status, progress in snapshot:
                if status == "running" and progress != last_pcts.get(sid, -1):
                    last_pcts[sid] = progress
                    await websocket.send_text(json.dumps({"strategy": sid, "percent": progress}))
                elif status in ("completed", "failed", "cancelled") and sid not in sent_done:
                    await websocket.send_text(json.dumps(_strategy_message(sid, results[sid])))
                    sent_done.add(sid)
            if run_done:
                await websocket.send_json({"status": "run_complete"})
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return
    await websocket.close()


async def _replay(websocket: WebSocket, run, last_pcts: Dict[str, int], sent_done: set[str]) -> bool:
    with run.lock:
        run_done = run.status in ("completed", "failed", "cancelled")
        snapshot = [(sid, result.status, result.progress, result) for sid, result in run.strategies.items()]
    for sid, status, progress, result in snapshot:
        if status in ("completed", "failed", "cancelled"):
            await websocket.send_text(json.dumps(_strategy_message(sid, result)))
            sent_done.add(sid)
        elif status == "running":
            last_pcts[sid] = progress
            await websocket.send_text(json.dumps({"strategy": sid, "percent": progress}))
    return run_done


def _strategy_message(sid: str, result) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"strategy": sid, "status": result.status}
    if result.metrics is not None:
        msg["metrics"] = result.metrics
    if result.error is not None:
        msg["error"] = result.error
    return msg
