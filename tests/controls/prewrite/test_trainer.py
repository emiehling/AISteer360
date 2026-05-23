"""Unit tests for `PRewriteTrainer` helper logic.

These tests don't actually run PPO updates; they exercise reward computation, rewriter-input construction, and the
mode-specific reward-aggregation logic that the trainer exposes.
"""
from __future__ import annotations

from aisteer360.algorithms.input_control.prewrite.templates import (
    DEFAULT_PER_QUERY_META_PROMPT,
    DEFAULT_STATIC_META_PROMPT,
)
from aisteer360.algorithms.input_control.prewrite.trainer import PRewriteTrainer

from tests.controls.prewrite._stubs import (
    StubFeedbackMetric,
    StubRewriter,
    StubTaskLM,
    StubTokenizer,
)


def _make_trainer(mode: str, score=0.7) -> PRewriteTrainer:
    return PRewriteTrainer(
        rewriter_model=StubRewriter(),
        rewriter_tokenizer=StubTokenizer(),
        task_model=StubTaskLM(),
        task_tokenizer=StubTokenizer(),
        feedback_metric=StubFeedbackMetric(score=score),
        meta_prompt=(
            DEFAULT_PER_QUERY_META_PROMPT if mode == "per_query" else DEFAULT_STATIC_META_PROMPT
        ),
        initial_prompt="respond.",
        mode=mode,
        config=None,
        rewriter_gen_kwargs={},
        task_gen_kwargs={},
    )


def test_trainer_builds_rewriter_input_per_query():
    trainer = _make_trainer("per_query")
    text = trainer._build_rewriter_input(query="What is 2+2?")
    assert "respond." in text
    assert "What is 2+2?" in text


def test_trainer_builds_rewriter_input_static():
    trainer = _make_trainer("static")
    text = trainer._build_rewriter_input(query=None)
    assert "respond." in text
    assert "{query}" not in text


def test_trainer_reward_extraction():
    trainer = _make_trainer("per_query", score=0.7)
    reward = trainer._compute_reward("a response", reference=None)
    assert reward == 0.7


def test_trainer_per_query_reward_per_rollout():
    """In per_query mode, each rollout produces a reward independently — verified by counting metric calls."""
    trainer = _make_trainer("per_query", score=[0.1, 0.4, 0.9])

    batch = [{"input": "q1"}, {"input": "q2"}, {"input": "q3"}]
    prompt_ids_list = []
    response_ids_list = []
    rewards = []
    for row in batch:
        rewriter_input = trainer._build_rewriter_input(query=row["input"])
        text, prompt_ids, response_ids = trainer._generate_rewrite(rewriter_input, sample=False)
        response = trainer._run_task_lm(text.strip(), row["input"])
        rewards.append(trainer._compute_reward(response, row.get("expected")))
        prompt_ids_list.append(prompt_ids)
        response_ids_list.append(response_ids)

    assert rewards == [0.1, 0.4, 0.9]
    assert len(trainer.feedback_metric.calls) == 3


def test_trainer_static_reward_is_batch_mean():
    """In static mode, per-query scores collapse into a mean reward used for the policy step."""
    trainer = _make_trainer("static", score=[0.1, 0.5, 0.9])

    batch = [{"input": "q1"}, {"input": "q2"}, {"input": "q3"}]
    rewriter_input = trainer._build_rewriter_input(query=None)
    rewritten_text, _, _ = trainer._generate_rewrite(rewriter_input, sample=False)
    rewritten = rewritten_text.strip()

    per_query_rewards = []
    for row in batch:
        response = trainer._run_task_lm(rewritten, row["input"])
        per_query_rewards.append(trainer._compute_reward(response, row.get("expected")))
    mean = sum(per_query_rewards) / len(per_query_rewards)

    assert per_query_rewards == [0.1, 0.5, 0.9]
    assert abs(mean - 0.5) < 1e-9
