"""Shared FastAPI application factory.

Each workbench builds its own FastAPI app via `create_workbench_app()`, which wires up the
infrastructure that every workbench needs (database, progress relay, request relay, owner-token
auth, model catalog, secrets vault, WebSocket relay) and mounts the shared static directory at
`/static/common/`. The workbench then adds its own routers and (typically) mounts a workbench-
specific static directory at `/`.
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .db import Database, resolve_data_root
from .relay import ProgressRelay, RequestRelay

logger = logging.getLogger(__name__)

ExtendSchemaHook = Callable[[Database], Awaitable[None]]
LifespanHook = Callable[[FastAPI], Awaitable[None]]

_COMMON_STATIC_DIR = Path(__file__).parent / "static"


def create_workbench_app(
    *,
    title: str,
    data_root: str | Path | None = None,
    public_server_url: str | None = None,
    agent_command_name: str = "aisteer360-agent",
    solo_mode: bool = False,
    extend_schema: ExtendSchemaHook | None = None,
    on_startup: LifespanHook | None = None,
    on_shutdown: LifespanHook | None = None,
) -> FastAPI:
    """Build a FastAPI app pre-wired with shared workbench infrastructure.

    Args:
        title: FastAPI app title (shown in `/docs`).
        data_root: Root directory for `runs.db` and per-run/session artefacts. Defaults to `./runs`
            or the `AISTEER_WORKBENCH_DATA_ROOT` env var.
        public_server_url: Base URL shown in agent command hints (e.g. `https://steer.example`).
            Defaults to the request's own base URL when None.
        agent_command_name: Name of the local CLI script (`aisteer360-agent` by default). Each
            workbench may override this if it ships its own console script.
        solo_mode: True when started from the per-workbench solo-dev launcher. When set, runs and
            sessions with no compute config default to local dispatch.
        extend_schema: Optional async hook invoked after the database connects. Use this to add
            workbench-specific tables (e.g. the `sessions` table for the pipeline workbench).
        on_startup: Optional async hook invoked once the database is connected, the relays are on
            `app.state`, and orphan-cleanup has run.
        on_shutdown: Optional async hook invoked before the database closes on shutdown. Local
            agent subprocesses recorded on `app.state.local_agents` are terminated independently.

    Returns:
        A FastAPI app with shared routes mounted. The caller adds workbench-specific routers and
        typically mounts a workbench-specific static directory at `/`.
    """
    resolved_root = resolve_data_root(data_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_root.mkdir(parents=True, exist_ok=True)
        db = Database(resolved_root / "runs.db")
        await db.connect()
        if extend_schema is not None:
            await extend_schema(db)
        await db.fail_orphaned_runs()
        relay = ProgressRelay()
        request_relay = RequestRelay()
        app.state.db = db
        app.state.relay = relay
        app.state.request_relay = request_relay
        app.state.data_root = resolved_root
        app.state.public_server_url = public_server_url
        app.state.agent_command_name = agent_command_name
        app.state.solo_mode = solo_mode
        app.state.local_agents: dict[str, subprocess.Popen] = {}
        logger.info(
            "Workbench server up. title=%r data_root=%s solo_mode=%s",
            title, resolved_root, solo_mode,
        )
        if on_startup is not None:
            await on_startup(app)
        try:
            yield
        finally:
            if on_shutdown is not None:
                try:
                    await on_shutdown(app)
                except Exception as exc:
                    logger.warning("on_shutdown hook raised: %s", exc)
            agents = getattr(app.state, "local_agents", {})
            for rid, proc in agents.items():
                if proc.poll() is None:
                    proc.terminate()
                    logger.info("Terminated local agent for %s on shutdown", rid)
            agents.clear()
            await db.close()

    app = FastAPI(title=title, lifespan=lifespan)

    @app.get("/api/server-info")
    async def server_info() -> dict:
        return {"solo_mode": getattr(app.state, "solo_mode", False)}

    from .routes_catalog import router as catalog_router
    from .routes_methods import router as methods_router
    from .routes_secrets import router as secrets_router
    from .ws import router as ws_router

    app.include_router(catalog_router, prefix="/api")
    app.include_router(methods_router, prefix="/api")
    app.include_router(secrets_router, prefix="/api")
    app.include_router(ws_router)

    if _COMMON_STATIC_DIR.exists():
        app.mount(
            "/static/common",
            StaticFiles(directory=str(_COMMON_STATIC_DIR)),
            name="common-static",
        )

    return app


__all__ = ["create_workbench_app"]
