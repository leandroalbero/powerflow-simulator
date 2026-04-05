from fastapi import APIRouter, Request, UploadFile, File, HTTPException

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
