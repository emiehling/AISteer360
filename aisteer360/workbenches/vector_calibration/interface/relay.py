"""Per-run progress fan-out.

Agents POST `/api/agent/runs/{id}/progress`; the handler publishes to the relay; any browser
WebSocket subscribed to `/ws/runs/{id}` receives the event. Events are dropped (not buffered)
when no subscriber is listening — the checkpoint files on disk are the source of truth for replay.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ProgressRelay:
    """Asyncio queue-based fan-out keyed by run_id."""

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


__all__ = ["ProgressRelay"]
