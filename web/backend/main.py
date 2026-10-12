"""FastAPI application for the powerflow-simulator web UI."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.backend.api import backtest, config, data, presets, runs, simulation, strategies, sweep
from web.backend.services.data_service import DataService
from web.backend.services.pool import SimPool
from web.backend.services.refresh_jobs import RefreshRegistry
from web.backend.services.simulation_service import SimulationService
from web.backend.services.sweep_service import SweepService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # --- startup ---
    data_service = DataService()
    data_service.load_default_data()

    sim_service = SimulationService(data_service)
    pool = None
    if os.environ.get("POWERFLOW_EAGER_POOL", "1") != "0":
        pool = SimPool()
        sim_service.attach_pool(pool)
    sweep_service = SweepService(sim_service, pool, sim_service.history) if pool is not None else None

    app.state.data_service = data_service
    app.state.sim_service = sim_service
    app.state.sim_pool = pool
    app.state.sweep_service = sweep_service
    app.state.refresh_registry = RefreshRegistry(data_service.load_default_data)

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
    if pool is not None:
        try:
            if sim_service._progress_q is not None:
                sim_service._progress_q.put(None)
        except (EOFError, ConnectionRefusedError, OSError):
            pass
        pool.shutdown()
    sim_service.history.close()


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
app.include_router(runs.router)
app.include_router(data.router)
app.include_router(presets.router)
app.include_router(sweep.router)
app.include_router(backtest.router)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


# Serve built frontend assets (production mode)
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
