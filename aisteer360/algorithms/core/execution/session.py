"""The `SteeringSession` protocol, the scope within which steering is in force."""
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import torch

from aisteer360.algorithms.core.execution.items import (
    CaptureResult,
    GenerationItem,
    ItemResult,
    ScoringItem,
)
from aisteer360.algorithms.core.execution.layout import ModelLayout
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.prompts import PreparedPrompt


@runtime_checkable
class SteeringSession(Protocol):
    """One logical operation's scope on a backend; the unit of concurrency.

    A session is opened per logical operation (one generation fan-out, one scoring call, one
    steer-phase fit) on every backend. The `Backend` owns the loaded model or engine; a session
    holds only per-operation state. Session-contract facts, provided by every backend and
    therefore never capability atoms, include token-id prompts, stop rules, minimum tokens,
    multiple candidates, seeded sampling, prompt-logprob scoring, and the model layout.
    """

    @property
    def layout(self) -> ModelLayout:
        """Structural facts about the session's model."""
        ...

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item, in item order."""
        ...

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens, shape
        `[num_items, ref_len]`."""
        ...

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Capture hidden states for `prompts` at `layers`; requires
        `Capability.HIDDEN_CAPTURE`."""
        ...
