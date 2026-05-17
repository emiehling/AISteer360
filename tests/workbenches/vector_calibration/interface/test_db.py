"""Tests for the aiosqlite persistence layer."""
from __future__ import annotations

import pytest

from aisteer360.workbenches.common.interface.db import (
    Database,
    STATUS_CLAIMED,
    STATUS_CREATED,
    hash_agent_token,
    mint_agent_token,
    sha256_hex,
    verify_agent_token,
)


@pytest.mark.asyncio
async def test_create_and_get_run(db: Database, tmp_path) -> None:
    token = mint_agent_token()
    run = await db.create_run(
        run_id="warmth_abc",
        behavior="warmth",
        steered_model="x/model",
        config={"k": 1},
        stages=["generation", "extraction", "calibration"],
        owner_token_hash=sha256_hex("dt-aaaa"),
        agent_token_hash=hash_agent_token(token),
        run_dir=tmp_path / "warmth_abc",
    )
    assert run.id == "warmth_abc"
    assert run.status == STATUS_CREATED
    got = await db.get_run("warmth_abc")
    assert got is not None
    assert got.config == {"k": 1}


@pytest.mark.asyncio
async def test_list_scoped_by_owner(db: Database, tmp_path) -> None:
    h1 = sha256_hex("dt-one")
    h2 = sha256_hex("dt-two")
    for i in range(2):
        await db.create_run(
            run_id=f"a_{i}",
            behavior="a",
            steered_model="m",
            config={},
            stages=["generation", "extraction", "calibration"],
            owner_token_hash=h1,
            agent_token_hash=hash_agent_token(mint_agent_token()),
            run_dir=tmp_path / f"a_{i}",
        )
    await db.create_run(
        run_id="b_0",
        behavior="b",
        steered_model="m",
        config={},
        stages=["generation", "extraction", "calibration"],
        owner_token_hash=h2,
        agent_token_hash=hash_agent_token(mint_agent_token()),
        run_dir=tmp_path / "b_0",
    )
    r1 = await db.list_runs_for_owner(h1)
    r2 = await db.list_runs_for_owner(h2)
    assert {r.id for r in r1} == {"a_0", "a_1"}
    assert {r.id for r in r2} == {"b_0"}


@pytest.mark.asyncio
async def test_heartbeat_and_claim(db: Database, tmp_path) -> None:
    run = await db.create_run(
        run_id="z",
        behavior="z",
        steered_model="m",
        config={},
        stages=["generation", "extraction", "calibration"],
        owner_token_hash=sha256_hex("dt-x"),
        agent_token_hash=hash_agent_token(mint_agent_token()),
        run_dir=tmp_path / "z",
    )
    assert run.last_heartbeat is None
    await db.heartbeat("z")
    r2 = await db.get_run("z")
    assert r2.last_heartbeat is not None
    await db.claim("z")
    r3 = await db.get_run("z")
    assert r3.status == STATUS_CLAIMED
    assert r3.claimed_at is not None


def test_agent_token_roundtrip() -> None:
    token = mint_agent_token()
    hashed = hash_agent_token(token)
    assert verify_agent_token(token, hashed)
    assert not verify_agent_token("sk-run-wrong", hashed)


def test_sha256_is_stable() -> None:
    assert sha256_hex("dt-abc") == sha256_hex("dt-abc")
    assert sha256_hex("dt-abc") != sha256_hex("dt-abd")
