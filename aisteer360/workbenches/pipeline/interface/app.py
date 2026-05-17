"""FastAPI application factory for the pipeline workbench.

Like vector calibration, the server is inert: it persists session metadata in SQLite, relays
inference requests browser↔agent, and never loads a model. All compute lives in the agent.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aisteer360.workbenches.common.interface.app_factory import create_workbench_app
from aisteer360.workbenches.common.interface.db import Database

logger = logging.getLogger(__name__)

DEFAULT_AGENT_COMMAND_NAME = "aisteer360-pipeline-agent"


async def _install_sessions(db: Database) -> None:
    await db.install_sessions_schema()
    await db.fail_orphaned_sessions()


def create_app(
    data_root: str | Path | None = None,
    *,
    public_server_url: str | None = None,
    agent_command_name: str = DEFAULT_AGENT_COMMAND_NAME,
    solo_mode: bool = False,
) -> FastAPI:
    """Build the pipeline-workbench FastAPI app."""
    app = create_workbench_app(
        title="AISteer360 — Pipeline Workbench",
        data_root=data_root,
        public_server_url=public_server_url,
        agent_command_name=agent_command_name,
        solo_mode=solo_mode,
        extend_schema=_install_sessions,
    )

    from .routes_agent import router as agent_router
    from .routes_sessions import router as sessions_router

    app.include_router(sessions_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

    return app


__all__ = ["create_app"]
