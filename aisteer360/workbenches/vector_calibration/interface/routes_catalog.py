"""Model catalog and provider-status endpoints."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .auth import OwnerTokenHash, get_db
from .catalog import (
    ALL_PROVIDERS,
    ALL_ROLES,
    CatalogEntry,
    load_catalog,
    provider_status,
    save_catalog,
)
from .db import Database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])

_SUGGESTION_TTL_S = 60.0
_suggestion_cache: dict[tuple[str, str, str], tuple[float, list[dict]]] = {}


def _cache_get(key: tuple[str, str, str]) -> list[dict] | None:
    hit = _suggestion_cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.monotonic() - ts > _SUGGESTION_TTL_S:
        _suggestion_cache.pop(key, None)
        return None
    return value


def _cache_put(key: tuple[str, str, str], value: list[dict]) -> None:
    _suggestion_cache[key] = (time.monotonic(), value)


class CatalogEntrySchema(BaseModel):
    model_config = {"protected_namespaces": ()}

    label: str
    model_id: str
    provider: str = "hf"
    endpoint: str | None = None
    roles: list[str] = Field(default_factory=lambda: list(ALL_ROLES))


class CatalogResponse(BaseModel):
    entries: list[CatalogEntrySchema]
    providers: list[str] = Field(default_factory=lambda: list(ALL_PROVIDERS))
    roles: list[str] = Field(default_factory=lambda: list(ALL_ROLES))


class ProviderStatusResponse(BaseModel):
    providers: dict[str, dict[str, Any]]


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog(_: OwnerTokenHash) -> CatalogResponse:
    entries = load_catalog()
    return CatalogResponse(
        entries=[CatalogEntrySchema(**asdict(e)) for e in entries],
        providers=list(ALL_PROVIDERS),
        roles=list(ALL_ROLES),
    )


@router.put("/catalog", response_model=CatalogResponse)
def put_catalog(body: list[CatalogEntrySchema], _: OwnerTokenHash) -> CatalogResponse:
    entries = [
        CatalogEntry(
            label=e.label,
            model_id=e.model_id,
            provider=e.provider,
            endpoint=e.endpoint,
            roles=list(e.roles),
        )
        for e in body
    ]
    save_catalog(entries)
    return get_catalog(_)


@router.get("/catalog/providers/status", response_model=ProviderStatusResponse)
async def get_provider_status(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> ProviderStatusResponse:
    providers = provider_status()
    stored = await db.get_secrets_status(owner_hash)
    provider_key_map = {
        "anthropic": "anthropic_key",
        "openai": "openai_key",
        "openai_compatible": "openai_key",
    }
    for provider, key_name in provider_key_map.items():
        if provider in providers and stored.get(key_name):
            providers[provider]["env_present"] = True
    return ProviderStatusResponse(providers=providers)


class ModelSuggestion(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    label: str | None = None


class ModelSuggestionsResponse(BaseModel):
    entries: list[ModelSuggestion]
    provider: str
    query: str
    note: str | None = None


def _order_matches(ids: list[str], q: str, limit: int) -> list[str]:
    q_low = q.lower()
    prefix = [i for i in ids if i.lower().startswith(q_low)]
    substring = [i for i in ids if q_low in i.lower() and not i.lower().startswith(q_low)]
    return (prefix + substring)[:limit]


def _hf_search(q: str, limit: int) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    results = api.list_models(search=q, limit=limit)
    return [m.modelId for m in results if getattr(m, "modelId", None)]


async def _openai_list(base_url: str, api_key: str) -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        payload = resp.json()
    return [m["id"] for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]


async def _anthropic_list(api_key: str) -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    return [m["id"] for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]


@router.get("/catalog/model-suggestions", response_model=ModelSuggestionsResponse)
async def get_model_suggestions(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
    provider: str = Query("hf"),
    q: str = Query(..., min_length=2),
    limit: int = Query(8, ge=1, le=25),
    base_url: str | None = Query(None),
) -> ModelSuggestionsResponse:
    q = q.strip()
    cache_key = (provider, q, base_url or "")
    cached = _cache_get(cache_key)
    if cached is not None:
        return ModelSuggestionsResponse(
            entries=[ModelSuggestion(**e) for e in cached],
            provider=provider,
            query=q,
        )

    entries: list[dict] = []
    note: str | None = None

    try:
        if provider == "hf":
            try:
                ids = await asyncio.to_thread(_hf_search, q, limit)
            except ImportError:
                raise HTTPException(501, "huggingface_hub is required for HF autocomplete.")
            entries = [{"model_id": i} for i in ids[:limit]]

        elif provider in ("openai", "openai_compatible"):
            secrets = await db.get_secrets(owner_hash)
            api_key = secrets.get("openai_key")
            if not api_key:
                note = "no api key stored"
            else:
                effective_base = (base_url or "https://api.openai.com").strip()
                try:
                    ids = await _openai_list(effective_base, api_key)
                except httpx.HTTPError as exc:
                    raise HTTPException(502, f"OpenAI models lookup failed: {exc}") from exc
                matches = _order_matches(ids, q, limit)
                entries = [{"model_id": i} for i in matches]

        elif provider == "anthropic":
            secrets = await db.get_secrets(owner_hash)
            api_key = secrets.get("anthropic_key")
            if not api_key:
                note = "no api key stored"
            else:
                try:
                    ids = await _anthropic_list(api_key)
                except httpx.HTTPError as exc:
                    raise HTTPException(502, f"Anthropic models lookup failed: {exc}") from exc
                matches = _order_matches(ids, q, limit)
                entries = [{"model_id": i} for i in matches]

        else:
            # custom / unknown: no autocomplete
            entries = []
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("model-suggestions failed for provider=%s q=%r: %s", provider, q, exc)
        raise HTTPException(500, f"model-suggestions failed: {exc}") from exc

    _cache_put(cache_key, entries)
    return ModelSuggestionsResponse(
        entries=[ModelSuggestion(**e) for e in entries],
        provider=provider,
        query=q,
        note=note,
    )
