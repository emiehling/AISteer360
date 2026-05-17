"""Persistent model catalog.

Stores a small user-managed list of models that the dashboard offers as selections in the target / generator / judge
slots. The catalog lives at `~/.aisteer360/model_catalog.json` and is edited via the settings modal.

Only entries with provider `"hf"` are actually usable by the pipeline today; other providers (`openai_compatible`,
`anthropic`, `custom`) are stored and presented in the UI but flagged as not wired.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Provider = Literal["hf", "openai", "openai_compatible", "anthropic", "custom"]
Role = Literal["target", "generator", "judge"]

WIRED_PROVIDERS: set[str] = {"hf", "openai", "openai_compatible", "anthropic"}
ALL_PROVIDERS: tuple[str, ...] = ("hf", "openai", "openai_compatible", "anthropic", "custom")
ALL_ROLES: tuple[str, ...] = ("target", "generator", "judge")

PROVIDER_ENV: dict[str, str | None] = {
    "hf": None,
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "custom": None,
}

DEFAULT_CATALOG_PATH = Path.home() / ".aisteer360" / "model_catalog.json"


@dataclass
class CatalogEntry:
    """A single model entry in the catalog.

    Attributes:
        label: Human-readable name shown in the UI.
        model_id: HF id (for `hf`) or provider-specific id.
        provider: One of `hf`, `openai_compatible`, `anthropic`, `custom`.
        endpoint: Optional HTTP base URL for network-backed providers.
        roles: Which slots this model can fill (`target` is HF-only by construction).
    """

    label: str
    model_id: str
    provider: str = "hf"
    endpoint: str | None = None
    roles: list[str] = field(default_factory=lambda: ["target", "generator", "judge"])

    def sanitized(self) -> "CatalogEntry":
        roles = [r for r in self.roles if r in ALL_ROLES]
        if self.provider != "hf":
            roles = [r for r in roles if r != "target"]
        return CatalogEntry(
            label=self.label.strip() or self.model_id,
            model_id=self.model_id.strip(),
            provider=self.provider if self.provider in ALL_PROVIDERS else "hf",
            endpoint=(self.endpoint or "").strip() or None,
            roles=roles or ["generator", "judge"],
        )


def _default_entries() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            label="Granite 3.3 2B Instruct",
            model_id="ibm-granite/granite-3.3-2b-instruct",
            provider="hf",
            roles=list(ALL_ROLES),
        ),
        CatalogEntry(
            label="Granite 3.3 8B Instruct",
            model_id="ibm-granite/granite-3.3-8b-instruct",
            provider="hf",
            roles=list(ALL_ROLES),
        ),
    ]


def load_catalog(path: Path | None = None) -> list[CatalogEntry]:
    """Read the catalog, returning defaults if the file is missing or unreadable."""
    path = path or DEFAULT_CATALOG_PATH
    if not path.exists():
        return _default_entries()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read catalog at %s (%s); returning defaults.", path, exc)
        return _default_entries()
    entries = []
    for raw in data if isinstance(data, list) else []:
        if not isinstance(raw, dict) or "model_id" not in raw:
            continue
        entry = CatalogEntry(
            label=str(raw.get("label") or raw["model_id"]),
            model_id=str(raw["model_id"]),
            provider=str(raw.get("provider") or "hf"),
            endpoint=raw.get("endpoint"),
            roles=list(raw.get("roles") or ALL_ROLES),
        ).sanitized()
        entries.append(entry)
    if not entries:
        return _default_entries()
    return entries


def save_catalog(entries: list[CatalogEntry], path: Path | None = None) -> None:
    """Write the catalog atomically to disk."""
    path = path or DEFAULT_CATALOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [asdict(e.sanitized()) for e in entries if e.model_id.strip()]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cleaned, indent=2))
    tmp.replace(path)


def provider_status() -> dict[str, dict[str, object]]:
    """Report which providers are usable.

    Returns a mapping from provider name to `{wired, env_var, env_present}`. `env_present` only reflects whether the
    env var is set; values are never echoed back.
    """
    status: dict[str, dict[str, object]] = {}
    for provider in ALL_PROVIDERS:
        env_var = PROVIDER_ENV.get(provider)
        status[provider] = {
            "wired": provider in WIRED_PROVIDERS,
            "env_var": env_var,
            "env_present": bool(env_var and os.environ.get(env_var)),
        }
    return status
