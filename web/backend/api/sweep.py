import asyncio
import json

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from web.backend.models.feature_schemas import SweepDetail, SweepListItem, SweepRequest, SweepStartResponse
from web.backend.services.csv_export import sweep_csv
from web.backend.services.sweep_service import SweepBusy

router = APIRouter(prefix="/api", tags=["sweep"])


@router.post("/sweep", response_model=SweepStartResponse)
def start_sweep(body: SweepRequest, request: Request) -> SweepStartResponse:
    sweep = request.app.state.sweep_service
    if sweep is None:
        raise HTTPException(status_code=503, detail="process pool unavailable")
    try:
        sweep_id, total = sweep.start(body)
    except SweepBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SweepStartResponse(sweep_id=sweep_id, total_cases=total)


@router.get("/sweeps", response_model=list[SweepListItem])
def list_sweeps(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[SweepListItem]:
    sweep = request.app.state.sweep_service
    if sweep is None:
        return []
    return sweep.list_sweeps(limit)


@router.get("/sweep/{sweep_id}", response_model=SweepDetail)
def get_sweep(sweep_id: str, request: Request) -> SweepDetail:
    detail = _require(request, sweep_id)
    return detail


@router.post("/sweep/{sweep_id}/cancel")
def cancel_sweep(sweep_id: str, request: Request) -> dict[str, str]:
    sweep = request.app.state.sweep_service
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
    status = sweep.cancel(sweep_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
    return {"status": status}


@router.get("/sweep/{sweep_id}/export.csv")
def export_sweep(sweep_id: str, request: Request) -> StreamingResponse:
    detail = _require(request, sweep_id)
    filename = f"sweep-{sweep_id}.csv"
    return StreamingResponse(
        sweep_csv([case.model_dump() for case in detail.cases]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.websocket("/ws/sweep/{sweep_id}")
async def ws_sweep(websocket: WebSocket, sweep_id: str) -> None:
    await websocket.accept()
    sweep = websocket.app.state.sweep_service
    detail = None if sweep is None else sweep.get(sweep_id)
    if detail is None:
        await websocket.send_json({"error": f"Sweep {sweep_id} not found"})
        await websocket.close()
        return
    sent: set[int] = set()
    try:
        while True:
            detail = sweep.get(sweep_id)
            if detail is None:
                break
            for idx, case in enumerate(detail.cases):
                if case.status in ("pending", "running") or idx in sent:
                    continue
                await websocket.send_text(json.dumps({
                    "type": "case_complete",
                    "case": case.model_dump(),
                }))
                sent.add(idx)
            await websocket.send_json({"type": "progress", "done": detail.done, "total": detail.total})
            if detail.status in ("completed", "failed", "cancelled"):
                await websocket.send_json({"type": "sweep_complete", "status": detail.status})
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return
    await websocket.close()


def _require(request: Request, sweep_id: str) -> SweepDetail:
    sweep = request.app.state.sweep_service
    detail = None if sweep is None else sweep.get(sweep_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
    return detail
