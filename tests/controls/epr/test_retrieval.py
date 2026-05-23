"""Tests for retrieval helpers (BM25/TF-IDF + dense top-k)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aisteer360.algorithms.input_control.epr.retrieval import (
    bm25_search,
    build_bm25_index,
    dense_top_k,
)


def test_bm25_returns_top_k_by_similarity():
    docs = [
        "the cat sat on the mat",
        "dogs run in the park",
        "the cat chased a mouse",
        "the weather is sunny today",
    ]
    index = build_bm25_index(docs)
    top = bm25_search(index, "cat on mat", k=2)
    assert top[0] == 0  # nearly identical query
    assert 2 in top  # also cat-related


def test_bm25_search_handles_k_larger_than_corpus():
    docs = ["alpha beta gamma", "delta epsilon zeta"]
    index = build_bm25_index(docs)
    top = bm25_search(index, "alpha delta", k=10)
    assert sorted(top) == [0, 1]


def test_dense_top_k_numpy():
    corpus = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    indices, scores = dense_top_k(query, corpus, k=2)
    assert list(indices) == [0, 3]
    assert scores[0] >= scores[1]


def test_dense_top_k_k_larger_than_corpus():
    corpus = np.array([[1.0], [2.0]], dtype=np.float32)
    query = np.array([1.0], dtype=np.float32)
    indices, scores = dense_top_k(query, corpus, k=5)
    assert list(indices) == [1, 0]
    assert len(scores) == 2


def test_dense_top_k_faiss_matches_numpy_or_falls_back():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(20, 8)).astype(np.float32)
    q = rng.normal(size=(8,)).astype(np.float32)

    np_indices, _ = dense_top_k(q, corpus, k=5, use_faiss=False)
    try:
        import faiss  # noqa: F401
        faiss_available = True
    except ImportError:
        faiss_available = False

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        faiss_indices, _ = dense_top_k(q, corpus, k=5, use_faiss=True)

    if faiss_available:
        assert list(faiss_indices) == list(np_indices)
    else:
        # fell back; warning emitted; numpy result returned
        assert list(faiss_indices) == list(np_indices)
        assert any("faiss" in str(item.message).lower() for item in w)


def test_dense_top_k_empty_corpus():
    corpus = np.zeros((0, 4), dtype=np.float32)
    q = np.zeros(4, dtype=np.float32)
    indices, scores = dense_top_k(q, corpus, k=3)
    assert indices.shape == (0,)
    assert scores.shape == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
