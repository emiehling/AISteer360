"""Tests for FeedbackScorer."""
from __future__ import annotations

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.scorers import TaskLMScorer
from aisteer360.algorithms.input_control.gepa.feedback_metric import ScoreOnlyFeedbackMetric
from aisteer360.algorithms.input_control.gepa.scorer import FeedbackScorer

from tests.controls.common._stubs import StubMetric, StubModel, StubTokenizer, make_data


class _StubFeedbackMetric:
    def __init__(self, score: float, feedback: str) -> None:
        self._score = score
        self._feedback = feedback
        self.calls: list[dict] = []

    def compute_with_feedback(self, responses, references=None, prompts=None):
        self.calls.append({"responses": list(responses), "references": references})
        return [{"score": self._score, "feedback": self._feedback} for _ in responses]


def _adapter(input_ids, memory):
    return input_ids


def test_feedback_scorer_populates_feedback():
    model = StubModel()
    tokenizer = StubTokenizer()
    fm = _StubFeedbackMetric(score=0.5, feedback="needs improvement")
    scorer = FeedbackScorer(model=model, tokenizer=tokenizer, adapter=_adapter, feedback_metric=fm)

    data = make_data(["abc"])
    results = scorer.score([Candidate(memory=None)], data)

    trace = results[0][0]
    assert trace.score == 0.5
    assert trace.feedback == "needs improvement"
    assert trace.metadata["raw_response"] == "XY"


def test_feedback_scorer_score_matches_taskscorer_when_wrapped():
    model = StubModel()
    tokenizer = StubTokenizer()
    base_metric = StubMetric()
    direct = TaskLMScorer(model=model, tokenizer=tokenizer, adapter=_adapter, metric=base_metric)
    wrapped = FeedbackScorer(
        model=model,
        tokenizer=tokenizer,
        adapter=_adapter,
        feedback_metric=ScoreOnlyFeedbackMetric(base_metric),
    )

    data = make_data(["abcd"])
    cand = [Candidate(memory=None)]
    direct_score = direct.score(cand, data)[0][0].score
    wrapped_score = wrapped.score(cand, data)[0][0].score

    assert direct_score == wrapped_score
