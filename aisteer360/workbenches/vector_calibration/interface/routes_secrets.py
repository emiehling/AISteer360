"""Owner-scoped API-key vault.

Browsers set API keys via `PUT /api/secrets` and query which keys are set via
`GET /api/secrets/status`. Plaintext values never leave the server once stored: status queries
return booleans only, and decrypted values are only sent to the agent through the authenticated
claim endpoint.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .auth import OwnerTokenHash, get_db
from .db import SECRET_FIELDS, Database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["secrets"])


class SecretsUpdate(BaseModel):
    """Partial update for owner secrets.

    Any field left unset is preserved on the server. An explicit empty string clears the stored
    value. A non-empty string replaces the stored (encrypted) value.
    """

    hf_token: str | None = None
    anthropic_key: str | None = None
    openai_key: str | None = None


class SecretsStatusResponse(BaseModel):
    hf_token: bool
    anthropic_key: bool
    openai_key: bool


@router.put("/secrets", response_model=SecretsStatusResponse)
async def put_secrets(
    body: SecretsUpdate,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> SecretsStatusResponse:
    updates = body.model_dump(exclude_unset=True)
    if updates:
        await db.upsert_secrets(owner_hash, updates)
    status = await db.get_secrets_status(owner_hash)
    return SecretsStatusResponse(**{name: status.get(name, False) for name in SECRET_FIELDS})


@router.get("/secrets/status", response_model=SecretsStatusResponse)
async def get_secrets_status(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> SecretsStatusResponse:
    status = await db.get_secrets_status(owner_hash)
    return SecretsStatusResponse(**{name: status.get(name, False) for name in SECRET_FIELDS})
