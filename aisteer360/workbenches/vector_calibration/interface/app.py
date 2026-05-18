"""FastAPI application factory for the vector calibration dashboard.

The server is inert: it persists run metadata in SQLite, stores artefacts on disk, and relays
agent progress to browsers over WebSocket. It never loads a model and never imports torch.

This module is now a thin wrapper around the shared `create_workbench_app()` factory; it adds
only the VC-specific routers (runs, agent, model probe) and mounts the VC dashboard static dir.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aisteer360.workbenches.common.interface.app_factory import create_workbench_app

logger = logging.getLogger(__name__)

DEFAULT_AGENT_COMMAND_NAME = "aisteer360-agent"


def create_app(
    data_root: str | Path | None = None,
    *,
    public_server_url: str | None = None,
    agent_command_name: str = DEFAULT_AGENT_COMMAND_NAME,
    solo_mode: bool = False,
) -> FastAPI:
    """Build the vector-calibration FastAPI app.

    Args:
        data_root: Root directory for `runs.db` and per-run artefact folders. Defaults to `./runs`
            or the `AISTEER_WORKBENCH_DATA_ROOT` env var.
        public_server_url: Base URL shown in the agent command hint (e.g. `https://steer.example`).
            Defaults to the request's own base URL.
        agent_command_name: Name of the CLI script (overridable for tests).
        solo_mode: True when the server was started by the single-user CLI wrapper. When set, runs
            with no compute config default to local dispatch instead of showing the manual modal.
    """
    app = create_workbench_app(
        title="AISteer360 — Vector Calibration",
        data_root=data_root,
        public_server_url=public_server_url,
        agent_command_name=agent_command_name,
        solo_mode=solo_mode,
    )

    from .routes_agent import router as agent_router
    from .routes_runs import router as runs_router

    app.include_router(runs_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

    return app
