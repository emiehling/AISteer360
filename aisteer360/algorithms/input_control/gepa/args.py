"""Arguments for the GEPA input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class GEPAArgs(BaseArgs):
    """Arguments for GEPA (Genetic-Pareto reflective prompt evolution).

    Required:
        seed_instruction: Initial instruction string to optimize.
        feedback_metric: A `FeedbackMetric` instance (or `Metric` wrapped via `ScoreOnlyFeedbackMetric`).
        reflection_lm: Either an HF model name/path string (loaded by GEPA at `steer()` time) or a callable
            `(prompt) -> response` (used directly).
        max_metric_calls: Total budget on `FeedbackMetric` evaluations. The loop ends once this is reached.
        train_data: Task examples used for sampling reflection minibatches. Each dict must have an `"input_ids"` key
            (and optionally `"expected"` and `"id"`).

    Optional:
        val_data: Task examples used for the per-instance Pareto archive. Defaults to `train_data` (with a warning).
        reflection_minibatch_size: Number of training examples sampled per reflection step.
        use_merge: Whether to enable the `MergeProposer` step.
        max_merge_invocations: Cap on total merge attempts during the loop.
        merge_interval: Try a merge step every K iterations.
        skip_perfect_score: When True, skip reflection if all minibatch traces hit `perfect_score`.
        perfect_score: Threshold for `skip_perfect_score`.
        seed: RNG seed for minibatch sampling, archive selection, and merge partner selection.
        gen_kwargs: Generation parameters for the task LM (passed to `model.generate`).
        reflection_lm_kwargs: Loader kwargs when `reflection_lm` is a string.
    """

    seed_instruction: str = ""
    feedback_metric: Any = None
    reflection_lm: str | Callable[[str], str] | None = None
    max_metric_calls: int = 0
    train_data: list[dict] = field(default_factory=list)

    val_data: list[dict] | None = None
    reflection_minibatch_size: int = 3
    use_merge: bool = True
    max_merge_invocations: int = 5
    merge_interval: int = 5
    skip_perfect_score: bool = True
    perfect_score: float = 1.0
    seed: int = 0
    gen_kwargs: dict | None = None
    reflection_lm_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if not self.seed_instruction:
            raise ValueError("seed_instruction must be a non-empty string.")
        if self.feedback_metric is None:
            raise ValueError("feedback_metric is required.")
        if self.reflection_lm is None:
            raise ValueError("reflection_lm is required (string model name or callable).")
        if self.max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be positive.")
        if self.reflection_minibatch_size <= 0:
            raise ValueError("reflection_minibatch_size must be positive.")
        if not self.train_data:
            raise ValueError("train_data must be non-empty.")
        if self.max_merge_invocations < 0:
            raise ValueError("max_merge_invocations must be >= 0.")
        if self.merge_interval <= 0:
            raise ValueError("merge_interval must be positive.")
