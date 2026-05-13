"""Per-run WebSocket progress relay."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .auth import authorise_ws_owner
from .db import Database
from .relay import ProgressRelay

logger = logging.getLogger(__name__)

router = APIRouter()


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

    # emit a hello with the latest known state so the UI can paint immediately
    latest = run.to_summary()
    await websocket.send_json({"event": "snapshot", **latest})

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
                # client closed (or sent a ping); either way treat as keepalive
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
        logger.debug("WS disconnect for run %s", run_id)
    except Exception as exc:
        logger.debug("WS for run %s closed with error: %s", run_id, exc)
    finally:
        await relay.unsubscribe(run_id, queue)
