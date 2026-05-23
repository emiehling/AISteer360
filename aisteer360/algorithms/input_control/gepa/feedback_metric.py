"""FeedbackMetric: GEPA-flavored metric returning score and textual feedback per response."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from aisteer360.algorithms.input_control.common.scorers.task_lm import TaskLMScorer
from aisteer360.evaluation.metrics.base import Metric


@runtime_checkable
class FeedbackMetric(Protocol):
    """A metric that returns score and textual feedback per response.

    Mirrors the role of `GEPAFeedbackMetric` in DSPy. A FeedbackMetric is given a batch of responses (plus optional
    references and prompts) and returns one result dict per response with keys `"score"` (float) and `"feedback"`
    (str). The feedback is the optimization signal GEPA's reflection step consumes.
    """

    def compute_with_feedback(
        self,
        responses: list[str],
        references: list[str] | None = None,
        prompts: list[str] | None = None,
    ) -> list[dict]:
        """Returns one dict per response with keys `"score"` and `"feedback"`."""
        ...


class ScoreOnlyFeedbackMetric:
    """Wraps a regular `Metric`, producing boilerplate feedback per response.

    Feedback string mirrors DSPy GEPA's default when no per-trajectory feedback is available:
    `"This trajectory got a score of {score}."`

    GEPA still works with this -- but reflection quality degrades because the LM has no signal beyond the scalar.
    Users get better results by implementing a real `FeedbackMetric`.
    """

    def __init__(self, metric: Metric) -> None:
        self.metric = metric

    def compute_with_feedback(
        self,
        responses: list[str],
        references: list[str] | None = None,
        prompts: list[str] | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        for i, response in enumerate(responses):
            metric_input: dict[str, Any] = {"responses": [response]}
            if references is not None and i < len(references):
                metric_input["references"] = [references[i]]
            if prompts is not None and i < len(prompts):
                metric_input["prompts"] = [prompts[i]]
            metric_result = self.metric.compute(**metric_input)
            score = TaskLMScorer._extract_scalar(metric_result)
            results.append({
                "score": score,
                "feedback": f"This trajectory got a score of {score}.",
            })
        return results
