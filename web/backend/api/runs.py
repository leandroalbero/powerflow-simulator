"""Run history, cancel, and CSV export."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from web.backend.models.feature_schemas import RunListItem
from web.backend.services.csv_export import monthly_csv, summary_csv, timeseries_csv

router = APIRouter(prefix="/api", tags=["runs"])

_EXPIRED = "timeseries expired, re-run"


@router.get("/runs", response_model=list[RunListItem])
def list_runs(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[RunListItem]:
    sim = request.app.state.sim_service
    return [RunListItem.model_validate(item) for item in sim.list_runs(limit)]


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request) -> Response:
    sim = request.app.state.sim_service
    if not sim.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return Response(status_code=204)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request) -> dict[str, str]:
    sim = request.app.state.sim_service
    status = sim.cancel_run(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"status": status}


@router.get("/runs/{run_id}/export.csv")
def export_run(
    run_id: str,
    request: Request,
    kind: str = Query("summary"),
    strategy: str | None = Query(None),
) -> StreamingResponse:
    sim = request.app.state.sim_service
    run = sim.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if kind not in ("summary", "monthly", "timeseries"):
        raise HTTPException(status_code=422, detail="kind must be summary, monthly, or timeseries")
    blobs = [
        {
            "strategy_id": sid,
            "status": result.status,
            "error": result.error,
            "metrics": result.metrics,
        }
        for sid, result in run.strategies.items()
    ]
    if kind == "summary":
        body = summary_csv(blobs)
        filename = f"run-{run_id}-summary.csv"
    elif kind == "monthly":
        body = monthly_csv(blobs)
        filename = f"run-{run_id}-monthly.csv"
    else:
        if not strategy:
            raise HTTPException(status_code=422, detail="timeseries export requires strategy")
        status, ts = sim.fetch_timeseries(run_id, strategy)
        if status == "missing":
            raise HTTPException(status_code=404, detail=f"No timeseries for run={run_id}, strategy={strategy}")
        if status == "evicted":
            raise HTTPException(status_code=410, detail=_EXPIRED)
        if ts is None:
            raise HTTPException(status_code=404, detail=f"No timeseries for run={run_id}, strategy={strategy}")
        capacity = (run.strategies[strategy].metrics or {}).get("battery_capacity") or run.config.battery.capacity
        body = timeseries_csv(ts, float(capacity))
        filename = f"run-{run_id}-{strategy}.csv"
    return StreamingResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
    })
