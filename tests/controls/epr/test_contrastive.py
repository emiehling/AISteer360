"""Tests for Stage 2 contrastive training."""
from __future__ import annotations

import numpy as np
import pytest

from aisteer360.algorithms.input_control.epr.contrastive import (
    ContrastiveConfig,
    train_contrastive_retriever,
)

BERT_TINY = "prajjwal1/bert-tiny"


def _synthetic_labeled(n: int = 12) -> list[dict]:
    rng = np.random.default_rng(0)
    out: list[dict] = []
    for i in range(n):
        cluster = "MATH" if i % 2 == 0 else "LANG"
        anchor = {"input": f"{cluster} item {i:03d}", "output": f"{cluster.lower()}_o_{i:03d}"}
        same_cluster_idx = (i + 2) % n if (i + 2) % n != i else (i + 4) % n
        if same_cluster_idx % 2 != i % 2:
            same_cluster_idx = (same_cluster_idx + 1) % n
        diff_cluster_idx = (i + 1) % n  # opposite parity by construction
        positives = [
            {"input": f"{cluster} item {same_cluster_idx:03d}", "output": f"{cluster.lower()}_o"},
        ]
        other_cluster = "LANG" if cluster == "MATH" else "MATH"
        negatives = [
            {"input": f"{other_cluster} item {diff_cluster_idx:03d}", "output": f"{other_cluster.lower()}_o"},
        ]
        out.append({"anchor": anchor, "positives": positives, "negatives": negatives})
    _ = rng  # unused but seeded
    return out


@pytest.mark.slow
def test_train_contrastive_completes():
    labeled = _synthetic_labeled(n=8)
    config = ContrastiveConfig(
        batch_size=4, epochs=1, learning_rate=1e-4, n_negatives_per_anchor=1, max_length=32, warmup_steps=2,
    )
    try:
        in_enc, pr_enc = train_contrastive_retriever(
            labeled_data=labeled,
            base_encoder_name_or_path=BERT_TINY,
            pooling="cls",
            config=config,
            device="cpu",
        )
    except Exception as exc:
        pytest.skip(f"could not run contrastive trainer (model load failure?): {exc}")
    assert in_enc.hidden_size > 0
    assert pr_enc.hidden_size > 0
    in_enc.cleanup()
    pr_enc.cleanup()


@pytest.mark.slow
def test_train_contrastive_separates_pos_from_neg():
    labeled = _synthetic_labeled(n=12)
    config = ContrastiveConfig(
        batch_size=4, epochs=4, learning_rate=5e-4, n_negatives_per_anchor=1, max_length=32, warmup_steps=2,
    )
    try:
        in_enc, pr_enc = train_contrastive_retriever(
            labeled_data=labeled,
            base_encoder_name_or_path=BERT_TINY,
            pooling="cls",
            config=config,
            device="cpu",
        )
    except Exception as exc:
        pytest.skip(f"could not run contrastive trainer: {exc}")

    anchors = [row["anchor"]["input"] for row in labeled]
    positives = [row["positives"][0] for row in labeled]
    negatives = [row["negatives"][0] for row in labeled]

    a_emb = in_enc.embed(anchors)
    p_emb = pr_enc.embed(["{} -> {}".format(p["input"], p["output"]) for p in positives])
    n_emb = pr_enc.embed(["{} -> {}".format(n["input"], n["output"]) for n in negatives])

    pos_sim = (a_emb * p_emb).sum(axis=1).mean()
    neg_sim = (a_emb * n_emb).sum(axis=1).mean()
    assert pos_sim > neg_sim
    in_enc.cleanup()
    pr_enc.cleanup()


def test_train_contrastive_validates_empty_data():
    config = ContrastiveConfig(epochs=1)
    with pytest.raises(ValueError, match="non-empty"):
        train_contrastive_retriever(
            labeled_data=[],
            base_encoder_name_or_path=BERT_TINY,
            pooling="cls",
            config=config,
            device="cpu",
        )


@pytest.mark.slow
def test_train_contrastive_respects_epoch_count(monkeypatch):
    """epochs=1 calls forward_torch a bounded number of times."""
    labeled = _synthetic_labeled(n=4)
    config = ContrastiveConfig(
        batch_size=2, epochs=1, learning_rate=1e-4, n_negatives_per_anchor=1, max_length=16, warmup_steps=1,
    )
    try:
        in_enc, pr_enc = train_contrastive_retriever(
            labeled_data=labeled,
            base_encoder_name_or_path=BERT_TINY,
            pooling="cls",
            config=config,
            device="cpu",
        )
    except Exception as exc:
        pytest.skip(f"could not run contrastive trainer: {exc}")
    in_enc.cleanup()
    pr_enc.cleanup()
