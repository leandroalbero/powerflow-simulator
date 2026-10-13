import asyncio

from fastapi import APIRouter, HTTPException, Request

from web.backend.models.feature_schemas import BacktestRequest, BacktestResponse
from web.backend.services.backtest_service import run_backtest
from web.backend.services.simulation_service import STRATEGY_MAP

router = APIRouter(prefix="/api", tags=["backtest"])


@router.post("/backtest", response_model=BacktestResponse)
async def post_backtest(body: BacktestRequest, request: Request) -> BacktestResponse:
    if body.strategy_id not in STRATEGY_MAP:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {body.strategy_id}")
    sim = request.app.state.sim_service
    try:
        return await asyncio.to_thread(
            run_backtest, sim, body.date_range.start, body.date_range.end, body.strategy_id, body.hardware,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown preset: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
