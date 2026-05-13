"""FastAPI application factory for the calibration dashboard.

The server is inert: it persists run metadata in SQLite, stores artefacts on disk, and relays
agent progress to browsers over WebSocket. It never loads a model and never imports torch.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .db import Database, resolve_data_root
from .relay import ProgressRelay

logger = logging.getLogger(__name__)

DEFAULT_AGENT_COMMAND_NAME = "aisteer360-agent"


def create_app(
    data_root: str | Path | None = None,
    *,
    public_server_url: str | None = None,
    agent_command_name: str = DEFAULT_AGENT_COMMAND_NAME,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        data_root: Root directory for `runs.db` and per-run artefact folders. Defaults to `./runs`
            or the `AISTEER_WORKBENCH_DATA_ROOT` env var.
        public_server_url: Base URL shown in the agent command hint (e.g. `https://steer.example`).
            Defaults to the request's own base URL.
        agent_command_name: Name of the CLI script (overridable for tests).
    """
    resolved_root = resolve_data_root(data_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_root.mkdir(parents=True, exist_ok=True)
        db = Database(resolved_root / "runs.db")
        await db.connect()
        relay = ProgressRelay()
        app.state.db = db
        app.state.relay = relay
        app.state.data_root = resolved_root
        app.state.public_server_url = public_server_url
        app.state.agent_command_name = agent_command_name
        logger.info("Workbench server up. data_root=%s", resolved_root)
        try:
            yield
        finally:
            await db.close()

    app = FastAPI(title="AISteer360 — Vector Calibration", lifespan=lifespan)

    # route wiring (imports lazy so the module is importable before routes exist at boot)
    from .routes_agent import router as agent_router
    from .routes_catalog import router as catalog_router
    from .routes_model import router as model_router
    from .routes_runs import router as runs_router
    from .routes_secrets import router as secrets_router
    from .ws import router as ws_router

    app.include_router(runs_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")
    app.include_router(catalog_router, prefix="/api")
    app.include_router(model_router, prefix="/api")
    app.include_router(secrets_router, prefix="/api")
    app.include_router(ws_router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

    return app
