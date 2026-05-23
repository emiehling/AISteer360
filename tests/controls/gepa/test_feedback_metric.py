"""Tests for FeedbackMetric Protocol and ScoreOnlyFeedbackMetric."""
from __future__ import annotations

from aisteer360.algorithms.input_control.gepa.feedback_metric import (
    FeedbackMetric,
    ScoreOnlyFeedbackMetric,
)

from tests.controls.common._stubs import StubMetric


class _ProtocolFeedbackMetric:
    def compute_with_feedback(self, responses, references=None, prompts=None):
        return [{"score": 1.0, "feedback": "ok"} for _ in responses]


def test_feedback_metric_protocol_check():
    assert isinstance(_ProtocolFeedbackMetric(), FeedbackMetric)


def test_score_only_wraps_metric():
    base = StubMetric()
    fm = ScoreOnlyFeedbackMetric(base)
    out = fm.compute_with_feedback(["abc"])
    assert len(out) == 1
    assert out[0]["score"] == 3.0  # length-based scoring
    assert "score of 3.0" in out[0]["feedback"]
