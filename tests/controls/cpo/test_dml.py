"""Tests for the DML training pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from aisteer360.algorithms.input_control.cpo.dml import (
    CausalRewardModel,
    train_causal_reward_model,
)


def _synth_linear(
    n: int = 400,
    d_x: int = 4,
    d_z: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear ground truth: y = m(x) + θ(x)·z + ε, where θ(x) = A·x for known A."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d_x)).astype(np.float32)
    z = rng.normal(size=(n, d_z)).astype(np.float32)

    A = rng.normal(size=(d_x, d_z)).astype(np.float32)
    theta = x @ A  # [n, d_z]

    m = (x.sum(axis=1) * 0.5).astype(np.float32)
    interaction = (theta * z).sum(axis=1).astype(np.float32)
    noise = rng.normal(scale=0.1, size=n).astype(np.float32)
    y = m + interaction + noise
    return x, z, y, A


def test_train_causal_reward_model_recovers_linear_truth():
    x, z, y, A = _synth_linear(n=800, d_x=3, d_z=2, seed=0)

    crm = train_causal_reward_model(
        query_embeddings=x,
        prompt_embeddings=z,
        outcomes=y,
        n_folds=5,
        rng_seed=0,
    )

    rng = np.random.default_rng(99)
    n_test = 200
    x_test = rng.normal(size=(n_test, x.shape[1])).astype(np.float32)
    z_test = rng.normal(size=(n_test, z.shape[1])).astype(np.float32)
    theta_true = x_test @ A
    truth = (theta_true * z_test).sum(axis=1)

    pred = crm.predict(x_test, z_test)

    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.5, f"τ̂ R² = {r2:.3f}; expected > 0.5"


def test_train_with_pca_reduces_z_dim():
    x, z, y, _ = _synth_linear(n=80, d_x=3, d_z=8, seed=1)
    crm = train_causal_reward_model(
        query_embeddings=x,
        prompt_embeddings=z,
        outcomes=y,
        n_folds=4,
        embedding_dim_reduction=4,
        rng_seed=1,
    )
    assert crm.pca is not None
    assert crm.pca.n_components_ == 4
    assert crm.metadata["d_z"] == 4


def test_train_handles_small_data():
    x, z, y, _ = _synth_linear(n=20, d_x=2, d_z=2, seed=2)
    crm = train_causal_reward_model(
        query_embeddings=x,
        prompt_embeddings=z,
        outcomes=y,
        n_folds=5,
        rng_seed=2,
    )
    assert isinstance(crm, CausalRewardModel)
    pred = crm.predict(x[:5], z[:5])
    assert pred.shape == (5,)


def test_train_respects_seed():
    x, z, y, _ = _synth_linear(n=120, d_x=3, d_z=3, seed=3)
    crm_a = train_causal_reward_model(
        query_embeddings=x, prompt_embeddings=z, outcomes=y, n_folds=4, rng_seed=7,
    )
    crm_b = train_causal_reward_model(
        query_embeddings=x, prompt_embeddings=z, outcomes=y, n_folds=4, rng_seed=7,
    )
    pred_a = crm_a.predict(x[:10], z[:10])
    pred_b = crm_b.predict(x[:10], z[:10])
    np.testing.assert_allclose(pred_a, pred_b)


def test_causal_reward_model_predict_shape():
    x, z, y, _ = _synth_linear(n=60, d_x=2, d_z=2, seed=4)
    crm = train_causal_reward_model(
        query_embeddings=x, prompt_embeddings=z, outcomes=y, n_folds=3, rng_seed=4,
    )
    pred = crm.predict(x[:7], z[:7])
    assert pred.shape == (7,)
    assert pred.dtype == np.float32


def test_causal_reward_model_predict_mismatched_shapes():
    x, z, y, _ = _synth_linear(n=40, d_x=2, d_z=2, seed=5)
    crm = train_causal_reward_model(
        query_embeddings=x, prompt_embeddings=z, outcomes=y, n_folds=3, rng_seed=5,
    )
    with pytest.raises(ValueError):
        crm.predict(x[:3], z[:5])
