"""Trace: one execution record from a candidate running on one task input."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from aisteer360.algorithms.core.types import Output


@dataclass
class Trace:
    """One record of a candidate executing on one task input.

    A Scorer produces a list of Traces per Candidate; an Archive stores them; a Proposer reads them when constructing
    the next batch of candidates.

    Attributes:
        input_ids: User input before any adaptation (2D tensor `[1, seq]`).
        steered_input_ids: Input after the candidate's adapter ran (2D tensor).
        output: Model response wrapped in the Phase 1 `Output` value type.
        score: Quality score. A scalar for single-objective optimization; a mapping `{dimension_name: value}` for
            multi-objective (e.g. ParetoArchive). Higher is better by convention; Scorers that measure error should
            negate before returning.
        feedback: Textual critique produced by the scorer's metric (when supported), consumed by reflection-based
            proposers. None when the scorer does not produce feedback. Methods that don't need feedback ignore the
            field.
        metadata: Free-form attachments. Conventions used by specific methods:

            - `"reasoning_steps"`: list of intermediate strings (CoT/agent methods)
            - `"errors"`: list of exception messages or model refusals
            - `"raw_response"`: pre-decode model output string
            - `"task_example"`: original dict from `data` that produced this trace

            The framework treats this opaquely.
    """

    input_ids: torch.Tensor
    steered_input_ids: torch.Tensor
    output: Output
    score: float | dict[str, float]
    feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
