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


class SteeredSession:
    """A `SteeringSession` whose `generate` and `score` inject a generation's control entries
    into every item, so a driver's rollouts carry the pipeline's steering without the driver
    knowing entries exist.

    The pipeline builds one wrapper per logical generation and hands it to the decoding
    driver. On an in-process backend the injected tuple is empty, since the session hosts the
    generation's hooks ambiently for the span of the driver's decode; on spec-consuming
    backends the injected entries are the generation's lowered interventions with
    prompt-relative scopes rewritten to absolute positions at the generation's original prompt
    boundary, so re-prefilled continuation tokens are steered at their original positions.

    Attributes:
        inner: The wrapped backend session.
        state_entries: Entries injected ahead of each item's own.
    """

    def __init__(self, inner, state_entries: tuple = ()):
        self.inner = inner
        self.state_entries = tuple(state_entries)

    @property
    def layout(self):
        return self.inner.layout

    @property
    def tokenizer(self):
        return getattr(self.inner, "tokenizer", None)

    def _inject(self, item):
        if not self.state_entries:
            return item
        import dataclasses

        return dataclasses.replace(
            item, state_entries=self.state_entries + tuple(item.state_entries)
        )

    def generate(self, items, params):
        """Generate with the wrapper's entries injected into every item."""
        return self.inner.generate([self._inject(item) for item in items], params)

    def score(self, items, params):
        """Score with the wrapper's entries injected into every item."""
        return self.inner.score([self._inject(item) for item in items], params)

    def capture(self, prompts, layers, mode, location="layer_output"):
        """Capture through the wrapped session, unsteered."""
        return self.inner.capture(prompts, layers, mode, location=location)
