"""TaskLMScorer: runs the task LM on data with each candidate's memory plugged in."""
from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace
from aisteer360.evaluation.metrics.base import Metric

Adapter = Callable[[list[int] | torch.Tensor, Any], list[int] | torch.Tensor]


class TaskLMScorer:
    """Run the task LM with each candidate's memory active; apply a Metric.

    The method that uses this scorer is responsible for supplying an `adapter` callable that knows how to combine its
    memory shape with user input. For most methods this is a thin wrapper around the method's `adapt()` logic
    refactored to take memory as a parameter rather than reading `self.memory`.

    Args:
        model: The task LM (already loaded and steered if applicable).
        tokenizer: For decoding model outputs and re-encoding adapter output.
        adapter: `Callable[(input_ids, Memory), steered_input_ids]`. Receives the user's untransformed input_ids and a
            candidate's memory; returns the steered input_ids ready for `model.generate`.
        metric: A `Metric` instance from `aisteer360/evaluation/metrics/` used to score generations. Called as
            `metric.compute(responses, prompts=..., references=...)`.
        gen_kwargs: Default generation parameters; overridden by per-call kwargs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        adapter: Adapter,
        metric: Metric | None = None,
        gen_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.metric = metric
        self.gen_kwargs = gen_kwargs or {}

    def score(
        self,
        candidates: list[Candidate],
        data: list[dict],
    ) -> list[list[Trace]]:
        """Score each candidate on each data example.

        For each (candidate, example) pair: apply the adapter, run `model.generate`, decode, score via the metric,
        build a Trace.
        """
        if self.metric is None:
            raise RuntimeError(
                "TaskLMScorer.score requires a Metric; either pass `metric=` to the constructor or override "
                "`score()` in a subclass that supplies its own scoring path."
            )

        results: list[list[Trace]] = []
        for candidate in candidates:
            traces_for_candidate: list[Trace] = []
            for example in data:
                rollout = self._rollout(example, candidate.memory)
                reference = example.get("expected")
                metric_input: dict[str, Any] = {"responses": [rollout["response"]]}
                if reference is not None:
                    metric_input["references"] = [reference]
                metric_result = self.metric.compute(**metric_input)
                score = self._extract_scalar(metric_result)

                traces_for_candidate.append(Trace(
                    input_ids=rollout["input_tensor"],
                    steered_input_ids=rollout["steered_tensor"],
                    output=Output(output_ids=rollout["new_tokens"], runtime_kwargs=None),
                    score=score,
                    metadata={"task_example": example, "raw_response": rollout["response"]},
                ))
            results.append(traces_for_candidate)
        return results

    def _rollout(self, example: dict, memory: Any) -> dict[str, Any]:
        """Apply the adapter, run generate, decode. Shared between TaskLMScorer and subclasses."""
        device = self.model.device
        input_ids = example["input_ids"]

        steered = self.adapter(input_ids, memory)
        steered_tensor = self._to_2d_tensor(steered, device)
        input_tensor = self._to_2d_tensor(input_ids, device)

        with torch.no_grad():
            full_output = self.model.generate(
                input_ids=steered_tensor,
                **self.gen_kwargs,
            )

        new_tokens = full_output[:, steered_tensor.size(1):]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        return {
            "input_tensor": input_tensor,
            "steered_tensor": steered_tensor,
            "new_tokens": new_tokens,
            "response": response,
        }

    @staticmethod
    def _to_2d_tensor(ids, device: torch.device) -> torch.Tensor:
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids.to(device)

    @staticmethod
    def _extract_scalar(metric_result: Any) -> float:
        """Extract a scalar score from a Metric's result.

        Convention: metrics return a dict; if a single-key dict is returned with a list value, take its first element.
        Subclasses can override for more elaborate aggregation.
        """
        if isinstance(metric_result, (int, float)):
            return float(metric_result)
        if isinstance(metric_result, dict) and len(metric_result) == 1:
            value = next(iter(metric_result.values()))
            if isinstance(value, list) and value:
                return float(value[0])
            if isinstance(value, (int, float)):
                return float(value)
        raise ValueError(
            f"Could not extract scalar score from metric result: {metric_result!r}. "
            "TaskLMScorer expects a Metric whose `compute` returns a scalar or a single-key dict; "
            "override `_extract_scalar` for richer shapes."
        )
