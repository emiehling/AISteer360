"""Model catalog and provider-status endpoints."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .catalog import (
    ALL_PROVIDERS,
    ALL_ROLES,
    CatalogEntry,
    load_catalog,
    provider_status,
    save_catalog,
)

router = APIRouter(tags=["catalog"])


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
def get_catalog() -> CatalogResponse:
    entries = load_catalog()
    return CatalogResponse(
        entries=[CatalogEntrySchema(**asdict(e)) for e in entries],
        providers=list(ALL_PROVIDERS),
        roles=list(ALL_ROLES),
    )


@router.put("/catalog", response_model=CatalogResponse)
def put_catalog(body: list[CatalogEntrySchema]) -> CatalogResponse:
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
    return get_catalog()


@router.get("/catalog/providers/status", response_model=ProviderStatusResponse)
def get_provider_status() -> ProviderStatusResponse:
    return ProviderStatusResponse(providers=provider_status())
