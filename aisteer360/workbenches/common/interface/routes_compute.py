"""Compute config endpoints shared across workbenches.

Persists per-owner compute settings (local vs SSH, host, credentials) and exposes a one-shot
SSH connectivity probe. Schemas live alongside the routes since they're used nowhere else.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from .auth import OwnerTokenHash, get_db
from .db import Database
from .dispatch import test_ssh

logger = logging.getLogger(__name__)

router = APIRouter(tags=["compute"])


class ComputeConfig(BaseModel):
    mode: Literal["local", "ssh"] = "local"
    host: str | None = None
    port: int = 22
    username: str | None = None
    auth_method: Literal["key", "password"] | None = None
    credential: str | None = None
    python_path: str = "python3"


class ComputeConfigResponse(BaseModel):
    """Compute config returned to the browser. Credentials are never sent back in plaintext."""
    mode: Literal["local", "ssh"] = "local"
    host: str | None = None
    port: int = 22
    username: str | None = None
    auth_method: Literal["key", "password"] | None = None
    credential_set: bool = False
    python_path: str = "python3"


class ComputeTestResponse(BaseModel):
    ok: bool
    error: str | None = None
    device: str | None = None
    device_name: str | None = None
    device_count: int | None = None
    server_reachable: bool | None = None
    reachability_error: str | None = None


def _public_server_url(request: Request) -> str:
    override = getattr(request.app.state, "public_server_url", None)
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")


@router.get("/compute/config", response_model=ComputeConfigResponse)
async def get_compute(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> ComputeConfigResponse:
    config = await db.get_compute_config(owner_hash)
    if config is None:
        return ComputeConfigResponse(mode="local")
    return ComputeConfigResponse(
        mode=config.get("mode", "local"),
        host=config.get("host"),
        port=config.get("port", 22),
        username=config.get("username"),
        auth_method=config.get("auth_method"),
        credential_set=bool(config.get("credential")),
        python_path=config.get("python_path") or "python3",
    )


@router.put("/compute/config")
async def put_compute(
    body: ComputeConfig,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    payload = body.model_dump()
    if payload.get("credential") is None:
        payload.pop("credential", None)
    await db.upsert_compute_config(owner_hash, payload)
    return {"status": "ok"}


@router.post("/compute/test", response_model=ComputeTestResponse)
async def post_compute_test(
    body: ComputeConfig,
    request: Request,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> ComputeTestResponse:
    payload = body.model_dump()
    if not payload.get("credential"):
        existing = await db.get_compute_config(owner_hash)
        if existing and existing.get("credential"):
            payload["credential"] = existing["credential"]
    server_url = _public_server_url(request)
    result = test_ssh(payload, server_url)
    return ComputeTestResponse(**result)
