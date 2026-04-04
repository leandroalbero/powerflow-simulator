from typing import List

from fastapi import APIRouter, Request

from web.backend.models.schemas import StrategyInfo

router = APIRouter(prefix="/api", tags=["strategies"])


@router.get("/strategies", response_model=List[StrategyInfo])
def list_strategies(request: Request) -> List[StrategyInfo]:
    sim_service = request.app.state.sim_service
    return sim_service.list_strategies()
