"""FastAPI application factory for the calibration dashboard."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aisteer360.workbenches.vector_calibration import (
    CalibrationBuilderConfig,
)

from .state import ServerState
from .ws import ConnectionManager

logger = logging.getLogger(__name__)


def create_app(
    config: CalibrationBuilderConfig,
    save_dir: str | Path = "./runs",
) -> FastAPI:
    """Build and return the FastAPI application."""

    state = ServerState(config=config, save_dir=save_dir)
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.set_loop(asyncio.get_running_loop())
        state.rebuild_builder()
        try:
            yield
        finally:
            if state.builder is not None:
                state.builder.cleanup()

    app = FastAPI(title="AISteer360 — Vector Calibration", lifespan=lifespan)

    app.state.server = state
    app.state.ws_manager = manager

    from .routes_catalog import router as catalog_router
    from .routes_config import router as config_router
    from .routes_model import router as model_router
    from .routes_results import router as results_router
    from .routes_run import router as run_router
    from .ws import router as ws_router

    app.include_router(config_router, prefix="/api")
    app.include_router(run_router, prefix="/api")
    app.include_router(results_router, prefix="/api")
    app.include_router(model_router, prefix="/api")
    app.include_router(catalog_router, prefix="/api")
    app.include_router(ws_router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

    return app
