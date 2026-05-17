"""Per-run progress fan-out and per-session request queue.

`ProgressRelay` is the agent-to-browser direction: agents POST `/api/agent/.../progress` (or any
similar event-emitting endpoint) and the handler publishes to the relay; any browser WebSocket
subscribed to `/ws/runs/{id}` (or `/ws/sessions/{id}`) receives the event. Events are dropped
(not buffered) when no subscriber is listening — the checkpoint files on disk are the source of
truth for replay.

`RequestRelay` is the browser-to-agent direction used by the composition workbench: the browser
POSTs an inference request, the server enqueues it on a single-slot per-session queue, and the
agent long-polls. New requests supersede any pending request — the slot only holds the latest.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ProgressRelay:
    """Asyncio queue-based fan-out keyed by run_id (or session_id)."""

    def __init__(self, queue_size: int = 128):
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._queue_size = queue_size

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subs[run_id].add(queue)
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subs.get(run_id)
            if subs is None:
                return
            subs.discard(queue)
            if not subs:
                self._subs.pop(run_id, None)

    async def publish(self, run_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            subs = list(self._subs.get(run_id, set()))
        for queue in subs:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Dropped progress event for %s (subscriber slow).", run_id)

    async def clear(self, run_id: str) -> None:
        async with self._lock:
            self._subs.pop(run_id, None)


class RequestRelay:
    """Per-session FIFO with a single slot.

    The browser pushes inference requests via `push()`; the agent long-polls via `poll()`. Pushing
    a new request while one is already queued *supersedes* it — the previous request is dropped
    because the user only cares about the latest pipeline state. This is intentional for the
    "I changed a slider, re-run" UX.

    Each session id gets its own `asyncio.Queue(maxsize=1)`. Queues are created lazily on first
    use and removed on `clear()`.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def _queue_for(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        async with self._lock:
            queue = self._queues.get(session_id)
            if queue is None:
                queue = asyncio.Queue(maxsize=1)
                self._queues[session_id] = queue
            return queue

    async def push(self, session_id: str, request: dict[str, Any]) -> str:
        """Enqueue a request. Returns the (possibly client-supplied) request id.

        If the request lacks a `request_id`, one is generated. If a previous request is still
        queued, it is discarded so the new one takes its place.
        """
        request_id = request.get("request_id") or uuid.uuid4().hex
        request = {**request, "request_id": request_id}
        queue = await self._queue_for(session_id)
        if queue.full():
            try:
                stale = queue.get_nowait()
                logger.debug(
                    "Superseded queued request %s for session %s",
                    stale.get("request_id"), session_id,
                )
            except asyncio.QueueEmpty:
                pass
        await queue.put(request)
        return request_id

    async def poll(self, session_id: str, timeout_s: float = 30.0) -> dict[str, Any] | None:
        """Block up to `timeout_s` for the next request. Returns None on timeout."""
        queue = await self._queue_for(session_id)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def clear(self, session_id: str) -> None:
        """Discard all pending requests for the session and forget its queue."""
        async with self._lock:
            queue = self._queues.pop(session_id, None)
        if queue is None:
            return
        try:
            while True:
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass


__all__ = ["ProgressRelay", "RequestRelay"]
