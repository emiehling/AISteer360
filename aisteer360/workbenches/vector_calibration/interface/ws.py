"""WebSocket connection manager and /ws/progress endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts events."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]

    async def _send(self, message: dict[str, Any]) -> None:
        """Send to all connections, removing any that have closed."""
        payload = json.dumps(message)
        alive: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
                alive.append(ws)
            except Exception:
                logger.debug("Dropping closed WebSocket connection.")
        self._connections = alive

    def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Thread-safe broadcast callable from the builder's on_progress callback.

        The builder runs in a worker thread (via asyncio.to_thread), so this
        schedules the actual send on the event loop.
        """
        message = {"event": event_type, **data}
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send(message), self._loop)


router = APIRouter()


@router.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline progress.

    Events sent to the client:

      - `{"event": "phase",         "phase": ...}`
      - `{"event": "progress",      "phase": ..., "completed": int, "total": int, ...}`
      - `{"event": "cell_complete", "layer": int, "multiplier": float, ...}`

    Clients do not need to send anything; this is a server-push channel.
    """
    manager: ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket closed with error: %s", exc)
        manager.disconnect(websocket)
