"""Tests for Stage 1 LM-conditional scoring + contrastive data generation."""
from __future__ import annotations

import numpy as np
import pytest

from aisteer360.algorithms.input_control.epr.retrieval import build_bm25_index
from aisteer360.algorithms.input_control.epr.scoring import (
    generate_contrastive_data,
    score_candidates_with_lm,
)

from tests.controls.epr._stubs import (
    cluster_score_fn,
    make_clustered_corpus,
    overlap_score_fn,
)


def test_score_candidates_returns_one_score_per_candidate():
    candidates = [
        {"input": "alpha", "output": "one"},
        {"input": "beta", "output": "two"},
        {"input": "gamma", "output": "three"},
    ]
    scores = score_candidates_with_lm(
        anchor_input="alpha gamma",
        anchor_output="one",
        candidates=candidates,
        scoring_lm=overlap_score_fn,
        scoring_tokenizer=None,
    )
    assert scores.shape == (3,)


def test_score_candidates_higher_for_more_similar():
    candidates = [
        {"input": "alpha gamma", "output": "one"},  # high overlap with anchor
        {"input": "delta epsilon", "output": "nine"},  # low overlap
    ]
    scores = score_candidates_with_lm(
        anchor_input="alpha gamma",
        anchor_output="one",
        candidates=candidates,
        scoring_lm=overlap_score_fn,
        scoring_tokenizer=None,
    )
    assert scores[0] > scores[1]


def test_generate_contrastive_data_splits_positives_negatives():
    corpus = make_clustered_corpus({"MATH": 8, "LANG": 8}, seed=0)
    bm25 = build_bm25_index([row["output"] for row in corpus])
    labeled = generate_contrastive_data(
        corpus=corpus,
        bm25_index=bm25,
        scoring_lm=cluster_score_fn,
        scoring_tokenizer=None,
        candidate_set_size=10,
        n_positives=3,
        n_negatives=3,
    )
    assert len(labeled) == len(corpus)
    for entry in labeled:
        assert len(entry["positives"]) == 3
        assert len(entry["negatives"]) == 3


def test_generate_contrastive_data_excludes_self():
    corpus = make_clustered_corpus({"MATH": 6, "LANG": 6}, seed=1)
    bm25 = build_bm25_index([row["output"] for row in corpus])
    labeled = generate_contrastive_data(
        corpus=corpus,
        bm25_index=bm25,
        scoring_lm=cluster_score_fn,
        scoring_tokenizer=None,
        candidate_set_size=8,
        n_positives=2,
        n_negatives=2,
    )
    for entry in labeled:
        anchor = entry["anchor"]
        for member in entry["positives"] + entry["negatives"]:
            assert (member["input"], member["output"]) != (anchor["input"], anchor["output"])


def test_generate_contrastive_data_validates_candidate_size():
    corpus = make_clustered_corpus({"MATH": 4}, seed=0)
    bm25 = build_bm25_index([row["output"] for row in corpus])
    with pytest.raises(ValueError, match="candidate_set_size"):
        generate_contrastive_data(
            corpus=corpus,
            bm25_index=bm25,
            scoring_lm=cluster_score_fn,
            scoring_tokenizer=None,
            candidate_set_size=2,
            n_positives=2,
            n_negatives=2,
        )


def test_score_candidates_empty_returns_empty():
    out = score_candidates_with_lm(
        anchor_input="x",
        anchor_output="y",
        candidates=[],
        scoring_lm=overlap_score_fn,
        scoring_tokenizer=None,
    )
    assert out.shape == (0,)


def test_score_candidates_positives_correlate_with_cluster():
    corpus = make_clustered_corpus({"MATH": 5, "LANG": 5}, seed=2)
    anchor = corpus[0]  # MATH item
    candidates = corpus[1:]
    scores = score_candidates_with_lm(
        anchor_input=anchor["input"],
        anchor_output=anchor["output"],
        candidates=candidates,
        scoring_lm=cluster_score_fn,
        scoring_tokenizer=None,
    )
    math_scores = np.array([s for s, c in zip(scores, candidates) if c["input"].startswith("MATH")])
    lang_scores = np.array([s for s, c in zip(scores, candidates) if c["input"].startswith("LANG")])
    assert math_scores.mean() > lang_scores.mean()
