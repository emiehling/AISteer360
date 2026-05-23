"""Tests for `CausalPoolMemory`."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from aisteer360.algorithms.input_control.common.memory.base import Memory
from aisteer360.algorithms.input_control.cpo.dml import train_causal_reward_model
from aisteer360.algorithms.input_control.cpo.memory import CausalPoolMemory


def _make_memory(seed: int = 0) -> CausalPoolMemory:
    rng = np.random.default_rng(seed)
    n, d_x, d_z = 60, 3, 2
    x = rng.normal(size=(n, d_x)).astype(np.float32)
    z = rng.normal(size=(n, d_z)).astype(np.float32)
    y = (x.sum(axis=1) + (x[:, 0] * z[:, 0])).astype(np.float32)
    crm = train_causal_reward_model(
        query_embeddings=x, prompt_embeddings=z, outcomes=y, n_folds=3, rng_seed=seed,
    )
    pool = ["template_a", "template_b", "template_c"]
    pool_embeddings = rng.normal(size=(3, d_z)).astype(np.float32)
    return CausalPoolMemory(
        pool=pool,
        pool_embeddings=pool_embeddings,
        causal_model=crm,
        query_embedder_name_or_path="my-query-embedder",
        prompt_embedder_name_or_path="my-prompt-embedder",
    )


def test_causal_pool_memory_defaults():
    memory = _make_memory()
    assert memory.model_type == "causal_pool"


def test_causal_pool_memory_save_directory_structure(tmp_path):
    memory = _make_memory()
    base = str(tmp_path / "memory")
    memory.save(base)

    saved_dir = base + ".cpm"
    assert os.path.isdir(saved_dir)
    for fname in ("meta.json", "pool.json", "pool_embeddings.npy", "causal_model.joblib"):
        assert os.path.exists(os.path.join(saved_dir, fname)), f"missing {fname}"

    with open(os.path.join(saved_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_type"] == "causal_pool"
    assert meta["pool_size"] == 3


def test_causal_pool_memory_round_trip(tmp_path):
    memory = _make_memory(seed=7)
    base = str(tmp_path / "rt")
    memory.save(base)
    loaded = CausalPoolMemory.load(base)

    assert loaded.pool == memory.pool
    np.testing.assert_allclose(loaded.pool_embeddings, memory.pool_embeddings)
    assert loaded.query_embedder_name_or_path == memory.query_embedder_name_or_path
    assert loaded.prompt_embedder_name_or_path == memory.prompt_embedder_name_or_path

    rng = np.random.default_rng(123)
    q = rng.normal(size=(5, memory.causal_model.metadata["d_x"])).astype(np.float32)
    z = rng.normal(size=(5, memory.causal_model.metadata["d_z"])).astype(np.float32)
    np.testing.assert_allclose(
        loaded.causal_model.predict(q, z),
        memory.causal_model.predict(q, z),
    )


def test_causal_pool_memory_load_rejects_wrong_type(tmp_path):
    memory = _make_memory()
    base = str(tmp_path / "wrongtype")
    memory.save(base)
    saved_dir = base + ".cpm"

    meta_path = os.path.join(saved_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["model_type"] = "text"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="model_type"):
        CausalPoolMemory.load(base)


def test_causal_pool_memory_satisfies_memory_protocol():
    memory = _make_memory()
    assert isinstance(memory, Memory)
