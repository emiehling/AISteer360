"""Model probe endpoint.

The former `/model/info` endpoint (which needed a loaded HF model) is gone — model info for a run
is now POSTed by the agent and read from `runs.model_info_json` via `GET /api/runs/{id}`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .auth import OwnerTokenHash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model"])

_PROBE_CACHE: dict[str, dict] = {}


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
    _PROBE_CACHE[model_id] = payload
    return ModelProbeResponse(**payload)
