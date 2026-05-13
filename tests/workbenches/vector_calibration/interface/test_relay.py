"""Tests for the ProgressRelay fan-out."""
from __future__ import annotations

import asyncio

import pytest

from aisteer360.workbenches.vector_calibration.interface.relay import ProgressRelay


@pytest.mark.asyncio
async def test_publish_delivers_to_all_subscribers() -> None:
    relay = ProgressRelay()
    q1 = await relay.subscribe("r1")
    q2 = await relay.subscribe("r1")
    await relay.publish("r1", {"event": "progress", "n": 1})
    assert (await asyncio.wait_for(q1.get(), timeout=1.0))["n"] == 1
    assert (await asyncio.wait_for(q2.get(), timeout=1.0))["n"] == 1


@pytest.mark.asyncio
async def test_publish_scoped_by_run_id() -> None:
    relay = ProgressRelay()
    q_a = await relay.subscribe("a")
    q_b = await relay.subscribe("b")
    await relay.publish("a", {"event": "x"})
    evt_a = await asyncio.wait_for(q_a.get(), timeout=1.0)
    assert evt_a["event"] == "x"
    assert q_b.empty()


@pytest.mark.asyncio
async def test_unsubscribe_removes_subscriber() -> None:
    relay = ProgressRelay()
    q = await relay.subscribe("r")
    await relay.unsubscribe("r", q)
    await relay.publish("r", {"event": "y"})
    assert q.empty()
