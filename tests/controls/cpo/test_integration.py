"""End-to-end integration test for CPO.

Exercises the full pipeline:

  1. Synthetic observational data with two query categories and a known best prompt per category.
  2. Stub embedders mapping category prefixes to fixed vectors.
  3. `cpo.steer()` runs the DML training (no task LM involved).
  4. Per-query routing via `cpo.adapt()` selects the correct prompt for each category.
  5. `CausalPoolMemory.save` / `load` round-trips the trained model and reproduces routing.

This validates that Phase 6 lands without amending any prior phase artifact.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from aisteer360.algorithms.input_control.cpo import CPO, CPOArgs, CausalPoolMemory

from tests.controls.cpo._stubs import (
    StubTokenizer,
    fixed_prompt_embedder,
    fixed_query_embedder,
    make_synthetic_training_data,
)


def _gbm_factory():
    def factory():
        return GradientBoostingRegressor(random_state=0)
    return factory


def test_cpo_end_to_end_routing_and_round_trip(tmp_path):
    training_data, pool = make_synthetic_training_data(n_per_category=80)

    args = CPOArgs(
        training_data=training_data,
        prompt_pool=pool,
        query_embedder=fixed_query_embedder(),
        prompt_embedder=fixed_prompt_embedder(),
        n_folds=3,
        effect_estimator_factory=_gbm_factory(),
        seed=0,
    )
    cpo = CPO(args)
    tok = StubTokenizer()
    cpo.steer(model=None, tokenizer=tok)

    a_text = tok.decode(cpo.adapt(tok.encode("AAA042 question")))
    b_text = tok.decode(cpo.adapt(tok.encode("BBB019 question")))
    assert "PROMPT_A:" in a_text
    assert "PROMPT_B:" in b_text

    save_base = str(tmp_path / "cpo_state")
    cpo.memory.save(save_base)
    reloaded = CausalPoolMemory.load(save_base)

    q_emb = cpo._query_embedder.embed(["AAA042 question", "BBB019 question"])
    pool_emb = cpo.memory.pool_embeddings
    a_scores_orig = cpo.memory.causal_model.predict(np.repeat(q_emb[0:1], 3, axis=0), pool_emb)
    a_scores_reload = reloaded.causal_model.predict(np.repeat(q_emb[0:1], 3, axis=0), pool_emb)
    np.testing.assert_allclose(a_scores_orig, a_scores_reload)

    cpo.cleanup()


def test_cpo_steering_method_export():
    """Verify the STEERING_METHOD export shape that the registry crawler reads.

    The toolkit's REGISTRY crawler in `aisteer360.algorithms.core.registry` is currently broken (doubled-path import
    error, see GEPA's matching test). We assert the export shape directly so Phase 6 doesn't depend on that pre-existing
    bug.
    """
    from aisteer360.algorithms.input_control.cpo import STEERING_METHOD

    assert STEERING_METHOD["category"] == "input_control"
    assert STEERING_METHOD["name"] == "cpo"
    assert STEERING_METHOD["control"] is CPO
    assert STEERING_METHOD["args"] is CPOArgs
