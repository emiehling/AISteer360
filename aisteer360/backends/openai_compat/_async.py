"""Sync-over-async driver for OpenAI-compatible backends.

The pipeline API is synchronous, but the OpenAI client is fastest driven asynchronously with bounded
concurrency. `run_coros` executes a list of coroutine factories under an `asyncio.Semaphore`,
returning results in submission order. When called from within a running event loop (e.g. a
notebook), it falls back to a dedicated loop thread so it never deadlocks.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable


async def _gather_bounded(factories: list[Callable[[], Awaitable[Any]]], concurrency: int) -> list[Any]:
    """Run coroutine factories under a semaphore, preserving submission order.

    Exceptions are captured per slot and returned in place (never raised here) so callers can decide
    how to aggregate partial failures.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(factory: Callable[[], Awaitable[Any]]) -> Any:
        async with semaphore:
            try:
                return await factory()
            except Exception as exc:  # captured; caller aggregates
                return exc

    return await asyncio.gather(*[_one(factory) for factory in factories])


def run_coros(factories: list[Callable[[], Awaitable[Any]]], concurrency: int) -> list[Any]:
    """Drive coroutine factories to completion from sync code, bounded by `concurrency`.

    Args:
        factories: Zero-argument callables each returning a fresh coroutine.
        concurrency: Maximum number of coroutines in flight at once.

    Returns:
        Results in submission order; a slot holds the raised exception when its coroutine failed.
    """
    if not factories:
        return []

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # no running loop: safe to own one for this call
        return asyncio.run(_gather_bounded(factories, concurrency))

    # a loop is already running (e.g. notebook): run in a dedicated thread with its own loop
    result: list[Any] = []
    error: list[BaseException] = []

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result.extend(loop.run_until_complete(_gather_bounded(factories, concurrency)))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            error.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result
