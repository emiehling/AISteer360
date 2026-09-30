"""Logits processor that masks every token a constraint automaton does not permit.

`ConstraintProcessor` drives an automaton that implements `reset(prefix_ids)` and `allowed(prefix_ids)` (the
`ConstraintAutomaton` protocol). At each step, the automaton returns the permitted token ids as one set for the whole
batch or as one set per row. The processor sets every other logit to -inf. Grammar, JSON schema, and regex constraints
are supported by supplying an automaton for them, e.g., an xgrammar or outlines-core matcher adapted to the protocol.
The toolkit does not implement a finite-state engine of its own.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from steerability.algorithms.output_control.common.processors.base import PrefixKeyedProcessor


class ConstraintAutomaton(Protocol):
    """Minimal automaton protocol a `ConstraintProcessor` drives."""

    def reset(self, prefix_ids: torch.Tensor) -> None:
        """Reset automaton state for a new/rewound prefix."""
        ...

    def allowed(self, prefix_ids: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor]:
        """Return the token ids permitted as the next token.

        Args:
            prefix_ids: The prefix token ids of shape `[B, T]`.

        Returns:
            One 1-D `LongTensor` of token ids that applies to every row, or a sequence with one 1-D `LongTensor` for
            each row of `prefix_ids`.
        """
        ...


class ConstraintProcessor(PrefixKeyedProcessor):
    """Mask all logits except the automaton's currently permitted token ids.

    Args:
        automaton: An object implementing the `ConstraintAutomaton` protocol.
    """

    def __init__(self, automaton: ConstraintAutomaton):
        super().__init__()
        self.automaton = automaton

    def reset_state(self, input_ids: torch.Tensor) -> None:
        self.automaton.reset(input_ids)

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Set every logit that the automaton does not permit to -inf.

        A single tensor of permitted token ids applies to every row. A sequence of permitted sets applies row by row.

        Args:
            input_ids: The prefix token ids of shape `[B, T]`.
            scores: The next-token logits of shape `[B, V]`.

        Returns:
            A new tensor of shape `[B, V]` that contains the permitted logits of `scores` and -inf elsewhere.

        Raises:
            ValueError: If the automaton returns a sequence of permitted sets whose length differs from the number of
                rows in `scores`.
        """
        allowed = self.automaton.allowed(input_ids)
        out = torch.full_like(scores, float("-inf"))
        if isinstance(allowed, torch.Tensor):
            allowed = allowed.to(scores.device)
            out[:, allowed] = scores[:, allowed]
            return out
        if len(allowed) != scores.size(0):
            raise ValueError(
                f"The automaton returned {len(allowed)} allowed sets for a batch of {scores.size(0)} rows."
            )
        for row, row_allowed in enumerate(allowed):
            row_allowed = row_allowed.to(scores.device)
            out[row, row_allowed] = scores[row, row_allowed]
        return out
