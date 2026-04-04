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
from web.backend.services.downsampler import downsample_timeseries
from web.backend.services.simulation_service import SimulationService

router = APIRouter(prefix="/api", tags=["simulation"])


# --- REST endpoints ---


@router.post("/simulate", response_model=SimulationStartResponse)
def start_simulation(body: SimulationRequest, request: Request) -> SimulationStartResponse:
    sim_service: SimulationService = request.app.state.sim_service

    # Validate strategy ids
    known = {s.id for s in sim_service.list_strategies()}
    unknown = set(body.strategies) - known
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown strategies: {sorted(unknown)}")

    start = body.date_range.start if body.date_range else None
    end = body.date_range.end if body.date_range else None

    run_id = sim_service.start_run(
        strategy_ids=body.strategies,
        start=start,
        end=end,
    )

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

    timestamps = ts_data["timestamps"]
    columns = {
        "battery_level": ts_data["battery_level"],
        "grid_import": ts_data["grid_import"],
        "grid_export": ts_data["grid_export"],
        "solar_power": ts_data["solar_power"],
        "house_consumption": ts_data["house_consumption"],
    }

    # Optional time-range filter
    if start or end:
        filtered_indices = []
        for i, ts_str in enumerate(timestamps):
            if start and ts_str < start:
                continue
            if end and ts_str > end:
                continue
            filtered_indices.append(i)

        timestamps = [timestamps[i] for i in filtered_indices]
        columns = {
            k: [v[i] for i in filtered_indices] for k, v in columns.items()
        }

    # Skip downsampling for single-day windows (~1440 points at 1-min resolution)
    is_single_day = len(timestamps) <= 1500
    if not is_single_day:
        timestamps, columns = downsample_timeseries(timestamps, columns, max_points)

    return TimeseriesResponse(
        timestamps=timestamps,
        point_count=len(timestamps),
        **columns,
    )


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

            for sid, result in run.strategies.items():
                # Nothing to do yet
                if result.status == "pending":
                    continue

                # Stream progress percentage
                if result.status == "running":
                    prev = last_pcts.get(sid, -1)
                    if result.progress != prev:
                        last_pcts[sid] = result.progress
                        await websocket.send_text(json.dumps({
                            "strategy": sid, "percent": result.progress,
                        }))
                        changed = True

                # Strategy finished
                if result.status in ("completed", "failed") and sid not in sent_done:
                    msg: Dict[str, Any] = {"strategy": sid, "status": result.status}
                    if result.metrics is not None:
                        msg["metrics"] = result.metrics
                    if result.error is not None:
                        msg["error"] = result.error
                    await websocket.send_text(json.dumps(msg))
                    sent_done.add(sid)
                    changed = True

            # Check if entire run is done
            if run.status in ("completed", "failed"):
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
