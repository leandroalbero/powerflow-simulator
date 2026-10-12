from fastapi import APIRouter

from web.backend.models.feature_schemas import PresetsResponse
from web.backend.services.presets import hardware_presets, tariff_presets

router = APIRouter(prefix="/api", tags=["presets"])


@router.get("/presets", response_model=PresetsResponse)
def get_presets() -> PresetsResponse:
    return PresetsResponse(hardware=hardware_presets(), tariffs=tariff_presets())
