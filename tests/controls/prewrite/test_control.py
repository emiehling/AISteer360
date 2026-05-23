"""Unit tests for the `PRewrite` input control.

These tests substitute the trainer via the `_trainer_factory` hook so we don't actually run PPO. The rewriter and
task LM are also stubs to avoid loading real models.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.prewrite import PRewrite, PRewriteArgs
from aisteer360.algorithms.input_control.prewrite.memory import ModelMemory

from tests.controls.prewrite._stubs import (
    StubFeedbackMetric,
    StubPRewriteTrainer,
    StubRewriter,
    StubTaskLM,
    StubTokenizer,
)


def _make_args(**overrides) -> PRewriteArgs:
    base = dict(
        initial_prompt="respond.",
        rewriter_model_name_or_path="dummy/rewriter",
        training_data=[{"input": "hi"}, {"input": "hello"}],
        feedback_metric=StubFeedbackMetric(),
        n_steps=1,
        batch_size=1,
    )
    base.update(overrides)
    return PRewriteArgs(**base)


def _build_with_stub_trainer(args: PRewriteArgs, *, rewriter=None) -> tuple[PRewrite, dict]:
    """Build a PRewrite that loads stubs instead of HF models and uses the StubPRewriteTrainer."""
    rewriter = rewriter or StubRewriter()
    rewriter_tokenizer = StubTokenizer()
    holder: dict = {"trainer": None, "rewriter": rewriter, "rewriter_tokenizer": rewriter_tokenizer}

    control = PRewrite(args)

    def fake_load_rewriter(self):
        return rewriter, rewriter_tokenizer

    def factory(**kwargs):
        trainer = StubPRewriteTrainer(**kwargs)
        holder["trainer"] = trainer
        return trainer

    control._trainer_factory = factory
    control._load_rewriter = fake_load_rewriter.__get__(control, PRewrite)
    return control, holder


def test_prewrite_is_not_stateful():
    assert PRewrite.is_stateful is False


def test_prewrite_steer_requires_model():
    args = _make_args()
    control = PRewrite(args)
    with pytest.raises(ValueError, match="task model"):
        control.steer(model=None, tokenizer=StubTokenizer())


def test_prewrite_per_query_produces_model_memory():
    args = _make_args(mode="per_query")
    control, holder = _build_with_stub_trainer(args)
    control.steer(model=StubTaskLM(), tokenizer=StubTokenizer())

    assert holder["trainer"].train_called is True
    assert isinstance(control.memory, ModelMemory)
    assert control.memory.model is holder["rewriter"]
    assert control.memory.extras["mode"] == "per_query"
    assert control.memory.extras["initial_prompt"] == "respond."


def test_prewrite_static_produces_text_memory():
    args = _make_args(mode="static")
    control, holder = _build_with_stub_trainer(args)
    control.steer(model=StubTaskLM(), tokenizer=StubTokenizer())

    assert isinstance(control.memory, TextMemory)
    assert control.memory.instruction == "STATIC_REWRITTEN_INSTRUCTION"
    assert control.memory.extras["mode"] == "static"


def test_prewrite_static_releases_rewriter():
    args = _make_args(mode="static")
    control, _ = _build_with_stub_trainer(args)
    control.steer(model=StubTaskLM(), tokenizer=StubTokenizer())

    assert isinstance(control.memory, TextMemory)
    # static mode discards rewriter — no ModelMemory is held
    assert not isinstance(control.memory, ModelMemory)


def test_prewrite_per_query_adapt_invokes_rewriter():
    args = _make_args(mode="per_query")
    rewriter = StubRewriter(response_text="Better instruction")
    control, _ = _build_with_stub_trainer(args, rewriter=rewriter)
    task_tok = StubTokenizer()
    control.steer(model=StubTaskLM(), tokenizer=task_tok)

    initial_calls = len(rewriter.generate_calls)
    ids = task_tok.encode("user query")
    out = control.adapt(ids)

    assert len(rewriter.generate_calls) == initial_calls + 1
    assert isinstance(out, list)
    decoded = task_tok.decode(out)
    assert "Better instruction" in decoded
    assert "user query" in decoded


def test_prewrite_static_adapt_uses_cached_instruction():
    args = _make_args(mode="static")
    control, holder = _build_with_stub_trainer(args)
    task_tok = StubTokenizer()
    control.steer(model=StubTaskLM(), tokenizer=task_tok)

    # rewriter was discarded — adapt must not invoke it
    rewriter = holder["rewriter"]
    calls_before = len(rewriter.generate_calls)
    ids = task_tok.encode("the user query")
    out = control.adapt(ids)
    assert len(rewriter.generate_calls) == calls_before
    decoded = task_tok.decode(out)
    assert "STATIC_REWRITTEN_INSTRUCTION" in decoded
    assert "the user query" in decoded


def test_prewrite_cleanup_releases_model_memory():
    args = _make_args(mode="per_query")
    control, _ = _build_with_stub_trainer(args)
    control.steer(model=StubTaskLM(), tokenizer=StubTokenizer())

    assert isinstance(control.memory, ModelMemory)
    control.cleanup()
    assert control.memory is None


def test_prewrite_adapt_before_steer_raises():
    args = _make_args()
    control = PRewrite(args)
    with pytest.raises(RuntimeError, match="steered"):
        control.adapt([1, 2, 3])
