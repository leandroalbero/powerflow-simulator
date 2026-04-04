from fastapi import APIRouter, Request

from web.backend.models.schemas import ConfigResponse, SystemConfig

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    sim_service = request.app.state.sim_service
    data_service = request.app.state.data_service
    cfg = sim_service.get_config()
    return ConfigResponse(
        battery=cfg.battery,
        grid=cfg.grid,
        tariff=cfg.tariff,
        data_date_range=data_service.get_date_range(),
    )


@router.put("/config", response_model=ConfigResponse)
def update_config(body: SystemConfig, request: Request) -> ConfigResponse:
    sim_service = request.app.state.sim_service
    data_service = request.app.state.data_service
    updated = sim_service.update_config(body)
    return ConfigResponse(
        battery=updated.battery,
        grid=updated.grid,
        tariff=updated.tariff,
        data_date_range=data_service.get_date_range(),
    )
