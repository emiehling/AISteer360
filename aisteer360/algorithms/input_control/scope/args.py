"""Arguments for the SCOPE input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.input_control.scope.memory import Rule


@dataclass
class SCOPEArgs(BaseArgs):
    """Arguments for SCOPE (Self-evolving Context Optimization via Prompt Evolution).

    Required:
        reflection_lm: Either an HF model name/path string (loaded by SCOPE at `steer()` time) or a callable
            `(prompt) -> response` (used directly).

    Optional:
        base_prompt: Static prefix that always appears at the top of the assembled system prompt. None to omit.
        seed_rules: Initial rules pre-populated into strategic/tactical streams.
        n_candidates: Number of guideline candidates the Generator proposes per `observe()` call.
        confidence_threshold: Strategic-stream gate. A guideline classified as "strategic" with confidence below this
            threshold is routed to tactical instead.
        strategic_max_size: Cap on strategic-stream length; the Optimizer is invoked when this is exceeded.
        tactical_max_size: Optional cap on tactical-stream length; oldest entries trimmed when exceeded. None for no
            cap.
        trigger_predicate: Optional `(input_text, response_text) -> bool` gate for synthesis. None to synthesize on
            every `observe()`.
        generator_template: Override for the Generator prompt template.
        selector_template: Override for the Selector prompt template.
        classifier_template: Override for the Classifier prompt template.
        optimizer_templates: Optional dict with keys "conflict_template", "subsumption_template",
            "consolidation_template" overriding the Optimizer's per-step templates.
        reflection_lm_kwargs: Loader kwargs when `reflection_lm` is a string.
        seed: RNG seed reserved for future use.
    """

    reflection_lm: str | Callable[[str], str] | None = None

    base_prompt: str | None = None
    seed_rules: list[Rule] | None = None

    n_candidates: int = 2
    confidence_threshold: float = 0.85

    strategic_max_size: int = 10
    tactical_max_size: int | None = None

    trigger_predicate: Callable[[str, str], bool] | None = None

    generator_template: str | None = None
    selector_template: str | None = None
    classifier_template: str | None = None
    optimizer_templates: dict[str, str] | None = None

    reflection_lm_kwargs: dict | None = None

    seed: int = 0

    def __post_init__(self) -> None:
        if self.reflection_lm is None:
            raise ValueError("reflection_lm is required (string model name or callable).")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.strategic_max_size < 1:
            raise ValueError("strategic_max_size must be >= 1")
        if self.tactical_max_size is not None and self.tactical_max_size < 1:
            raise ValueError("tactical_max_size must be >= 1 if set")
