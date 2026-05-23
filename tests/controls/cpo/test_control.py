"""Unit tests for the CPO control."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from sklearn.ensemble import GradientBoostingRegressor

from aisteer360.algorithms.input_control.cpo import CPO, CPOArgs, CausalPoolMemory
from aisteer360.algorithms.input_control.cpo.dml import CausalRewardModel

from tests.controls.cpo._stubs import (
    StubEmbedder,
    StubTokenizer,
    fixed_prompt_embedder,
    fixed_query_embedder,
    make_synthetic_training_data,
)


def _gbm_effect_factory():
    """Boosted-tree effect estimator. Used in routing tests because the default RF lacks the resolution to
    differentiate categories on the small one-hot synthetic data."""
    def factory():
        return GradientBoostingRegressor(random_state=0)
    return factory


def _build_cpo(
    *,
    n_folds: int = 3,
    selection_temperature: float = 0.0,
    seed: int = 0,
    n_per_category: int = 80,
    effect_estimator_factory=None,
) -> CPO:
    training_data, pool = make_synthetic_training_data(n_per_category=n_per_category)
    args = CPOArgs(
        training_data=training_data,
        prompt_pool=pool,
        query_embedder=fixed_query_embedder(),
        prompt_embedder=fixed_prompt_embedder(),
        n_folds=n_folds,
        selection_temperature=selection_temperature,
        effect_estimator_factory=effect_estimator_factory,
        seed=seed,
    )
    return CPO(args)


def test_cpo_is_not_stateful():
    assert CPO.is_stateful is False


def test_cpo_steer_ignores_model():
    cpo = _build_cpo()
    cpo.steer(model=None, tokenizer=StubTokenizer())
    assert cpo.memory is not None


def test_cpo_steer_populates_memory():
    cpo = _build_cpo()
    cpo.steer(model=None, tokenizer=StubTokenizer())
    assert isinstance(cpo.memory, CausalPoolMemory)
    assert cpo.memory.pool == ["PROMPT_A:", "PROMPT_B:", "PROMPT_BAD:"]
    assert cpo.memory.pool_embeddings.shape == (3, 3)
    assert isinstance(cpo.memory.causal_model, CausalRewardModel)


def test_cpo_adapt_selects_from_pool():
    cpo = _build_cpo()
    tok = StubTokenizer()
    cpo.steer(model=None, tokenizer=tok)

    ids = tok.encode("AAA042 query body")
    adapted = cpo.adapt(ids)
    decoded = tok.decode(adapted)
    assert any(template in decoded for template in cpo.memory.pool)


def test_cpo_adapt_argmax_routing():
    cpo = _build_cpo(effect_estimator_factory=_gbm_effect_factory())
    tok = StubTokenizer()
    cpo.steer(model=None, tokenizer=tok)

    a_query = "AAA017 about something"
    b_query = "BBB024 about something else"

    a_text = tok.decode(cpo.adapt(tok.encode(a_query)))
    b_text = tok.decode(cpo.adapt(tok.encode(b_query)))

    assert "PROMPT_A:" in a_text
    assert "PROMPT_B:" in b_text


def test_cpo_adapt_softmax_sample_diversity():
    cpo = _build_cpo(selection_temperature=1.0, seed=123)
    tok = StubTokenizer()
    cpo.steer(model=None, tokenizer=tok)

    counts = Counter()
    for _ in range(60):
        decoded = tok.decode(cpo.adapt(tok.encode("AAA050 q")))
        for template in cpo.memory.pool:
            if template in decoded:
                counts[template] += 1
                break
    assert len(counts) >= 2


def test_cpo_cleanup_releases_embedders():
    training_data, pool = make_synthetic_training_data(n_per_category=10)
    q_emb = fixed_query_embedder()
    p_emb = fixed_prompt_embedder()
    args = CPOArgs(
        training_data=training_data,
        prompt_pool=pool,
        query_embedder=q_emb,
        prompt_embedder=p_emb,
        n_folds=3,
    )
    cpo = CPO(args)
    cpo.steer(model=None, tokenizer=StubTokenizer())
    cpo.cleanup()

    assert cpo._query_embedder is None
    assert cpo._prompt_embedder is None
    assert q_emb.cleanup_called == 1
    assert p_emb.cleanup_called == 1


def test_cpo_cleanup_does_not_double_release_shared_embedder():
    training_data, pool = make_synthetic_training_data(n_per_category=10)
    shared = StubEmbedder(
        category_map={
            "AAA": np.array([1.0, 0.0], dtype=np.float32),
            "BBB": np.array([0.0, 1.0], dtype=np.float32),
            "PROMPT_A:": np.array([0.5, 0.0], dtype=np.float32),
            "PROMPT_B:": np.array([0.0, 0.5], dtype=np.float32),
            "PROMPT_BAD:": np.array([-0.5, -0.5], dtype=np.float32),
        }
    )
    args = CPOArgs(
        training_data=training_data,
        prompt_pool=pool,
        query_embedder=shared,
        prompt_embedder=None,  # reuse query_embedder
        n_folds=3,
    )
    cpo = CPO(args)
    cpo.steer(model=None, tokenizer=StubTokenizer())
    assert cpo._query_embedder is cpo._prompt_embedder

    cpo.cleanup()
    assert shared.cleanup_called == 1


def test_cpo_adapt_rejects_batched_input():
    cpo = _build_cpo()
    tok = StubTokenizer()
    cpo.steer(model=None, tokenizer=tok)

    batched = torch.tensor([tok.encode("AAA001 a"), tok.encode("BBB001 b")], dtype=torch.long)
    with pytest.raises(NotImplementedError):
        cpo.adapt(batched)
