"""FeedbackScorer: TaskLMScorer variant that runs a FeedbackMetric and populates Trace.feedback."""
from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.scorers.task_lm import Adapter, TaskLMScorer
from aisteer360.algorithms.input_control.common.trace import Trace
from aisteer360.algorithms.input_control.gepa.feedback_metric import FeedbackMetric


class FeedbackScorer(TaskLMScorer):
    """`TaskLMScorer` variant that runs a `FeedbackMetric` and populates `Trace.feedback`.

    Args:
        model: The task LM (already loaded and steered if applicable).
        tokenizer: For decoding model outputs and re-encoding adapter output.
        adapter: `Callable[(input_ids, Memory), steered_input_ids]`.
        feedback_metric: A `FeedbackMetric` instance (or a `Metric` wrapped via `ScoreOnlyFeedbackMetric`).
        gen_kwargs: Default generation parameters.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        adapter: Adapter,
        feedback_metric: FeedbackMetric,
        gen_kwargs: dict | None = None,
    ) -> None:
        super().__init__(model, tokenizer, adapter, metric=None, gen_kwargs=gen_kwargs)
        self.feedback_metric = feedback_metric

    def score(
        self,
        candidates: list[Candidate],
        data: list[dict],
    ) -> list[list[Trace]]:
        results: list[list[Trace]] = []
        for candidate in candidates:
            traces_for_candidate: list[Trace] = []
            for example in data:
                rollout = self._rollout(example, candidate.memory)

                reference = example.get("expected")
                fb_kwargs: dict[str, Any] = {"responses": [rollout["response"]]}
                if reference is not None:
                    fb_kwargs["references"] = [reference]
                fb_result = self.feedback_metric.compute_with_feedback(**fb_kwargs)
                score = float(fb_result[0]["score"])
                feedback = fb_result[0].get("feedback")

                traces_for_candidate.append(Trace(
                    input_ids=rollout["input_tensor"],
                    steered_input_ids=rollout["steered_tensor"],
                    output=Output(output_ids=rollout["new_tokens"], runtime_kwargs=None),
                    score=score,
                    feedback=feedback,
                    metadata={"task_example": example, "raw_response": rollout["response"]},
                ))
            results.append(traces_for_candidate)
        return results
