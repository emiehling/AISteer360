"""Per-run and per-session WebSocket progress relay."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .auth import authorise_ws_owner, authorise_ws_session_owner
from .db import Database
from .relay import ProgressRelay

logger = logging.getLogger(__name__)

router = APIRouter()


async def _pump_events(websocket: WebSocket, queue: asyncio.Queue, label: str) -> None:
    try:
        while True:
            recv_task = asyncio.create_task(websocket.receive_text())
            send_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if recv_task in done:
                try:
                    recv_task.result()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    raise WebSocketDisconnect()
                continue
            event = send_task.result()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.debug("WS disconnect for %s", label)
    except Exception as exc:
        logger.debug("WS for %s closed with error: %s", label, exc)


@router.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    """Subscribe the browser to progress events for one run.

    Authentication is via `?token=dt-...` query param or `Authorization: Bearer ...` header. The
    server validates that the token's SHA-256 hash matches the run's `owner_token_hash`.
    """
    db: Database = websocket.app.state.db
    relay: ProgressRelay = websocket.app.state.relay

    run = await authorise_ws_owner(run_id, websocket, db)
    if run is None:
        return

    await websocket.accept()
    queue = await relay.subscribe(run_id)

    await websocket.send_json({"event": "snapshot", **run.to_summary()})
    try:
        await _pump_events(websocket, queue, f"run {run_id}")
    finally:
        await relay.unsubscribe(run_id, queue)


@router.websocket("/ws/sessions/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    """Subscribe the browser to events for one composition session.

    Used to deliver `inference_result`, `model_info`, and lifecycle status events. Auth follows
    the same pattern as `/ws/runs/{id}`.
    """
    db: Database = websocket.app.state.db
    relay: ProgressRelay = websocket.app.state.relay

    session = await authorise_ws_session_owner(session_id, websocket, db)
    if session is None:
        return

    await websocket.accept()
    queue = await relay.subscribe(session_id)

    await websocket.send_json({"event": "snapshot", **session.to_summary()})
    try:
        await _pump_events(websocket, queue, f"session {session_id}")
    finally:
        await relay.unsubscribe(session_id, queue)
