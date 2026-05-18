"""Fixtures for workbench server + agent tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import json

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from aisteer360.workbenches.common.interface import catalog as catalog_module
from aisteer360.workbenches.vector_calibration.interface.app import create_app
from aisteer360.workbenches.common.interface.db import (
    Database,
    hash_agent_token,
    mint_agent_token,
    mint_owner_token,
    sha256_hex,
)


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def owner_token() -> str:
    return mint_owner_token()


@pytest.fixture
def owner_header(owner_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {owner_token}"}


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the catalog at a tmp file containing the synthetic test models.

    Run creation now validates generator/judge models against this catalog, so tests need an
    entry for the `test/model` id used by `minimal_config()`.
    """
    catalog_path = tmp_path / "model_catalog.json"
    catalog_path.write_text(json.dumps([
        {
            "label": "Test Model",
            "model_id": "test/model",
            "provider": "hf",
            "endpoint": None,
            "roles": ["target", "inference"],
        },
    ]))
    monkeypatch.setattr(catalog_module, "DEFAULT_CATALOG_PATH", catalog_path)
    return catalog_path


@pytest.fixture
def client(tmp_data_root: Path):
    app = create_app(data_root=tmp_data_root)
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def db(tmp_data_root: Path) -> Database:
    d = Database(tmp_data_root / "runs.db")
    await d.connect()
    try:
        yield d
    finally:
        await d.close()


ALL_STAGES: list[str] = ["generation", "extraction", "calibration"]


def minimal_config(model: str = "test/model", behavior: str = "warmth") -> dict[str, Any]:
    """A minimal but schema-valid CalibrationBuilderConfig as a dict."""
    return {
        "steered_model": model,
        "generation": {
            "generator_model": model,
            "behavior": behavior,
            "positive_prompt": "be warm",
            "negative_prompt": "be cold",
            "seed_prompts": ["hi", "hello"],
        },
        "extraction": {},
        "calibration": {
            "judge": {
                "model": model,
                "criteria": "Rate warmth 1-5. Response: {response}",
            },
        },
    }


def run_body(
    *,
    config: dict[str, Any] | None = None,
    stages: list[str] | None = None,
    behavior: str = "warmth",
) -> dict[str, Any]:
    """Build a valid /api/runs request body. Stages default to the full pipeline."""
    return {
        "config": config if config is not None else minimal_config(behavior=behavior),
        "stages": list(stages) if stages is not None else list(ALL_STAGES),
    }


@pytest.fixture
def cfg() -> dict[str, Any]:
    return minimal_config()


def _agent_token_and_hash() -> tuple[str, str]:
    token = mint_agent_token()
    return token, hash_agent_token(token)


@pytest.fixture
def make_agent_token():
    return _agent_token_and_hash


@pytest.fixture
def sha256_hex_fn():
    return sha256_hex
