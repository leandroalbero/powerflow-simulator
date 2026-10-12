from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.data.config import MissingCredentialError
from web.backend.models.feature_schemas import RefreshRequest, RefreshStartResponse
from web.backend.models.schemas import DataInfo

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/data/info", response_model=DataInfo)
def get_data_info(request: Request) -> DataInfo:
    data_service = request.app.state.data_service
    return data_service.get_data_info()


@router.post("/data/upload", response_model=DataInfo)
async def upload_data(
    request: Request,
    solar_file: UploadFile | None = File(None),
    load_file: UploadFile | None = File(None),
) -> DataInfo:
    data_service = request.app.state.data_service

    if solar_file is None and load_file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one file: solar_file or load_file",
        )

    try:
        if solar_file is not None:
            content = await solar_file.read()
            data_service.upload_solar(content)

        if load_file is not None:
            content = await load_file.read()
            data_service.upload_load(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return data_service.get_data_info()


@router.post("/data/reload", response_model=DataInfo)
def reload_data(request: Request) -> DataInfo:
    """Reload CSV files from disk without restarting the server."""
    data_service = request.app.state.data_service
    data_service.load_default_data()
    return data_service.get_data_info()


@router.post("/data/refresh", status_code=202, response_model=RefreshStartResponse)
def start_refresh(body: RefreshRequest, request: Request) -> JSONResponse:
    registry = request.app.state.refresh_registry
    try:
        job_id = registry.start(body.full)
    except MissingCredentialError:
        raise HTTPException(status_code=503, detail="INFLUX_* not configured") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/data/refresh/{job_id}")
def get_refresh(job_id: str, request: Request):
    registry = request.app.state.refresh_registry
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Refresh job {job_id} not found")
    return job
