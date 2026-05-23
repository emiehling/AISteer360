"""Shared value types for steering controls."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Output:
    """The result of one generation call.

    Passed to a subsequent `InputControl.adapt` call as `prior` and to `InputControl.observe`.

    Fields are intentionally minimal at Phase 1. Future phases may add lazy text decoding, logprobs, etc.

    Attributes:
        output_ids: Generated token IDs as a [batch, seq] tensor, excluding the input prompt (i.e. the same slice the
            pipeline returns to the caller by default).
        runtime_kwargs: The runtime_kwargs that produced this output. May be None if no runtime overrides were used.
    """
    output_ids: torch.Tensor
    runtime_kwargs: dict | None = None
