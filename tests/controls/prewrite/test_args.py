"""Tests for `PRewriteArgs`."""
from __future__ import annotations

import pytest

from aisteer360.algorithms.input_control.prewrite.args import PRewriteArgs

from tests.controls.prewrite._stubs import StubFeedbackMetric


def _valid_kwargs(**overrides):
    base = dict(
        initial_prompt="respond.",
        rewriter_model_name_or_path="dummy/rewriter",
        training_data=[{"input": "what is 2+2?"}],
        feedback_metric=StubFeedbackMetric(),
    )
    base.update(overrides)
    return base


def test_args_validation_initial_prompt():
    with pytest.raises(ValueError, match="initial_prompt"):
        PRewriteArgs(**_valid_kwargs(initial_prompt=""))


def test_args_validation_training_data():
    with pytest.raises(ValueError, match="training_data"):
        PRewriteArgs(**_valid_kwargs(training_data=[]))


def test_args_validation_mode():
    with pytest.raises(ValueError, match="mode"):
        PRewriteArgs(**_valid_kwargs(mode="bogus"))


def test_args_validation_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        PRewriteArgs(**_valid_kwargs(n_steps=0))
    with pytest.raises(ValueError, match="n_steps"):
        PRewriteArgs(**_valid_kwargs(n_steps=-1))


def test_args_validation_kl_coef():
    with pytest.raises(ValueError, match="kl_coef"):
        PRewriteArgs(**_valid_kwargs(kl_coef=-0.1))


def test_args_validation_row_schema():
    with pytest.raises(ValueError, match="missing 'input' key"):
        PRewriteArgs(**_valid_kwargs(training_data=[{"query": "x"}]))


def test_args_default_mode_is_per_query():
    args = PRewriteArgs(**_valid_kwargs())
    assert args.mode == "per_query"


def test_args_default_meta_prompts_selected_by_mode():
    from aisteer360.algorithms.input_control.prewrite.control import PRewrite
    from aisteer360.algorithms.input_control.prewrite.templates import (
        DEFAULT_PER_QUERY_META_PROMPT,
        DEFAULT_STATIC_META_PROMPT,
    )

    pq = PRewrite(PRewriteArgs(**_valid_kwargs(mode="per_query")))
    assert pq._resolved_meta_prompt() == DEFAULT_PER_QUERY_META_PROMPT

    st = PRewrite(PRewriteArgs(**_valid_kwargs(mode="static")))
    assert st._resolved_meta_prompt() == DEFAULT_STATIC_META_PROMPT
