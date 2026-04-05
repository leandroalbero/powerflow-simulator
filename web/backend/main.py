"""FastAPI application for the powerflow-simulator web UI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.backend.api import config, data, simulation, strategies
from web.backend.services.data_service import DataService
from web.backend.services.simulation_service import SimulationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # --- startup ---
    data_service = DataService()
    data_service.load_default_data()

    sim_service = SimulationService(data_service)

    app.state.data_service = data_service
    app.state.sim_service = sim_service

    info = data_service.get_data_info()
    logger.info(
        "Data loaded: solar=%s pts, load=%s pts, range=%s",
        info.solar_point_count,
        info.load_point_count,
        info.date_range,
    )

    yield

    # --- shutdown ---
    sim_service._executor.shutdown(wait=False)


app = FastAPI(
    title="Powerflow Simulator",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(strategies.router)
app.include_router(config.router)
app.include_router(simulation.router)
app.include_router(data.router)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


# Serve built frontend assets (production mode)
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
