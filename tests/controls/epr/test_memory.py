"""Tests for `RetrievalMemory`."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from aisteer360.algorithms.input_control.common.memory.base import Memory
from aisteer360.algorithms.input_control.epr.memory import RetrievalMemory
from aisteer360.algorithms.input_control.epr.retrieval import build_bm25_index


def _corpus() -> list[dict]:
    return [
        {"input": "what is 2+2", "output": "4"},
        {"input": "what is 3+3", "output": "6"},
        {"input": "capital of france", "output": "paris"},
    ]


def test_retrieval_memory_defaults():
    memory = RetrievalMemory(corpus=_corpus(), mode="bm25")
    assert memory.model_type == "retrieval"


def test_retrieval_memory_save_bm25_round_trip(tmp_path):
    corpus = _corpus()
    bm25 = build_bm25_index([row["output"] for row in corpus])
    memory = RetrievalMemory(corpus=corpus, mode="bm25", bm25_state=bm25)
    base = str(tmp_path / "rt")
    memory.save(base)

    saved_dir = base + ".rmem"
    assert os.path.isdir(saved_dir)
    for fname in ("meta.json", "corpus.jsonl", "bm25_state.joblib"):
        assert os.path.exists(os.path.join(saved_dir, fname)), f"missing {fname}"

    loaded = RetrievalMemory.load(base)
    assert loaded.mode == "bm25"
    assert loaded.corpus == corpus
    assert loaded.bm25_state is not None
    assert "vectorizer" in loaded.bm25_state
    assert "doc_matrix" in loaded.bm25_state


def test_retrieval_memory_save_dense_round_trip(tmp_path):
    corpus = _corpus()
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(corpus), 8)).astype(np.float32)
    memory = RetrievalMemory(
        corpus=corpus,
        mode="dense",
        dense_embeddings=embeddings,
        input_encoder_name_or_path="some-encoder",
        prompt_encoder_name_or_path="some-encoder",
        encoder_pooling="mean",
    )
    base = str(tmp_path / "rt")
    memory.save(base)
    loaded = RetrievalMemory.load(base)
    assert loaded.mode == "dense"
    assert loaded.corpus == corpus
    np.testing.assert_allclose(loaded.dense_embeddings, embeddings)
    assert loaded.input_encoder_name_or_path == "some-encoder"
    assert loaded.encoder_pooling == "mean"


def test_retrieval_memory_save_epr_round_trip(tmp_path):
    corpus = _corpus()
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(len(corpus), 4)).astype(np.float32)
    saved_dir = str(tmp_path / "epr_mem.rmem")
    input_path = os.path.join(saved_dir, "input_encoder")
    prompt_path = os.path.join(saved_dir, "prompt_encoder")
    memory = RetrievalMemory(
        corpus=corpus,
        mode="epr",
        dense_embeddings=embeddings,
        input_encoder_name_or_path=input_path,
        prompt_encoder_name_or_path=prompt_path,
        encoder_pooling="cls",
    )
    memory.save(saved_dir)

    loaded = RetrievalMemory.load(saved_dir)
    assert loaded.mode == "epr"
    np.testing.assert_allclose(loaded.dense_embeddings, embeddings)
    assert loaded.input_encoder_name_or_path == input_path
    assert loaded.prompt_encoder_name_or_path == prompt_path
    assert loaded.encoder_pooling == "cls"


def test_retrieval_memory_load_rejects_wrong_type(tmp_path):
    corpus = _corpus()
    bm25 = build_bm25_index([row["output"] for row in corpus])
    memory = RetrievalMemory(corpus=corpus, mode="bm25", bm25_state=bm25)
    base = str(tmp_path / "wrongtype")
    memory.save(base)
    saved_dir = base + ".rmem"

    meta_path = os.path.join(saved_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["model_type"] = "text"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="model_type"):
        RetrievalMemory.load(base)


def test_retrieval_memory_satisfies_memory_protocol():
    memory = RetrievalMemory(corpus=_corpus(), mode="bm25")
    assert isinstance(memory, Memory)
