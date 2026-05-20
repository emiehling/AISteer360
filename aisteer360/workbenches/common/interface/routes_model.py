"""Model probe + search endpoints.

The former `/model/info` endpoint (which needed a loaded HF model) is gone — model info for a run
is now POSTed by the agent and read from `runs.model_info_json` via `GET /api/runs/{id}`.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .auth import OwnerTokenHash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model"])

_PROBE_CACHE: dict[str, dict] = {}

# small TTL cache for HF search responses so typing fast doesn't hammer the Hub.
# key: search query (case-folded). value: (timestamp, results).
_SEARCH_CACHE: dict[str, tuple[float, list[dict]]] = {}
_SEARCH_TTL_SECONDS = 60.0


class ModelProbeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    model_type: str | None = None
    source: str = "hf"


class ModelSearchHit(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    downloads: int | None = None
    likes: int | None = None


class ModelSearchResponse(BaseModel):
    query: str
    results: list[ModelSearchHit]


@router.get("/model/probe", response_model=ModelProbeResponse)
def probe_model(
    _: OwnerTokenHash,
    model_id: str = Query(..., min_length=1),
) -> ModelProbeResponse:
    """Probe a HF model's architecture without downloading weights.

    Fetches only `config.json` via `huggingface_hub.hf_hub_download` (or reads it locally if
    `model_id` is a filesystem path). No GPU needed.
    """
    model_id = model_id.strip()
    if model_id in _PROBE_CACHE:
        return ModelProbeResponse(**_PROBE_CACHE[model_id])

    config_data: dict | None = None

    local_path = Path(model_id)
    if local_path.is_dir() and (local_path / "config.json").exists():
        try:
            config_data = json.loads((local_path / "config.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(400, f"Could not read local config.json: {exc}") from exc
    else:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
        except ImportError as exc:
            raise HTTPException(501, "huggingface_hub is required for model probing.") from exc
        try:
            cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
            config_data = json.loads(Path(cfg_path).read_text())
        except RepositoryNotFoundError as exc:
            raise HTTPException(404, f"Model '{model_id}' not found on the Hub.") from exc
        except HfHubHTTPError as exc:
            raise HTTPException(502, f"Hub error probing '{model_id}': {exc}") from exc
        except Exception as exc:
            error_msg = str(exc)
            if "Entry Not Found" in error_msg:
                raise HTTPException(404, f"Model '{model_id}' exists but has no config.json.") from exc
            raise HTTPException(500, f"Failed to probe '{model_id}': {exc}") from exc

    payload = {
        "model_id": model_id,
        "num_hidden_layers": config_data.get("num_hidden_layers"),
        "hidden_size": config_data.get("hidden_size"),
        "num_attention_heads": config_data.get("num_attention_heads"),
        "num_key_value_heads": config_data.get("num_key_value_heads"),
        "intermediate_size": config_data.get("intermediate_size"),
        "vocab_size": config_data.get("vocab_size"),
        "max_position_embeddings": config_data.get("max_position_embeddings"),
        "model_type": config_data.get("model_type"),
        "source": "hf",
    }

    _SUB_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder_config")
    _ARCH_FIELDS = (
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "max_position_embeddings",
    )
    for field_name in _ARCH_FIELDS:
        if payload[field_name] is None:
            for sub_key in _SUB_CONFIG_KEYS:
                sub = config_data.get(sub_key)
                if isinstance(sub, dict) and sub.get(field_name) is not None:
                    payload[field_name] = sub[field_name]
                    break

    _PROBE_CACHE[model_id] = payload
    return ModelProbeResponse(**payload)


@router.get("/model/search", response_model=ModelSearchResponse)
def search_models(
    _: OwnerTokenHash,
    q: str = Query(..., min_length=1, max_length=128),
    limit: int = Query(10, ge=1, le=20),
) -> ModelSearchResponse:
    """Autocomplete proxy for the HuggingFace model search API.

    The browser can't call huggingface.co directly (CORS), so this server-side
    helper queries `huggingface_hub.list_models(search=q, limit=limit, sort='downloads')`
    and returns a small list of candidates. Results are TTL-cached for 60s.
    """
    query = q.strip()
    if not query:
        return ModelSearchResponse(query=q, results=[])

    key = query.casefold()
    now = time.monotonic()
    cached = _SEARCH_CACHE.get(key)
    if cached and now - cached[0] < _SEARCH_TTL_SECONDS:
        cached_results = cached[1][:limit]
        return ModelSearchResponse(
            query=q,
            results=[ModelSearchHit(**hit) for hit in cached_results],
        )

    try:
        from huggingface_hub import list_models
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        raise HTTPException(501, "huggingface_hub is required for model search.") from exc

    hits: list[dict] = []
    try:
        for info in list_models(search=query, limit=limit, sort="downloads", direction=-1):
            hits.append({
                "model_id": getattr(info, "id", None) or getattr(info, "modelId", None) or "",
                "downloads": getattr(info, "downloads", None),
                "likes": getattr(info, "likes", None),
            })
    except HfHubHTTPError as exc:
        raise HTTPException(502, f"Hub error searching '{query}': {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Failed to search '{query}': {exc}") from exc

    hits = [h for h in hits if h["model_id"]]
    _SEARCH_CACHE[key] = (now, hits)
    return ModelSearchResponse(
        query=q,
        results=[ModelSearchHit(**hit) for hit in hits],
    )
