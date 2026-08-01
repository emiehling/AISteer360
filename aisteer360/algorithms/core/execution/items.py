"""Per-item units of session work and the per-category control contributions they carry.

A field on an item holds either an artifact, named for what it is (`prompt`, `ref_output_ids`,
`seed`), or the per-call contributions of one control category, named `<category>_entries`. An
entry is one enabled control's contribution for this call, in controls-list order, in whichever
representation the session consumes. An item never holds a control object.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from aisteer360.algorithms.core.execution.constraints import ConstraintSource
from aisteer360.algorithms.core.execution.interventions import (
    InterventionSpec,
    ProcessorSpec,
)
from aisteer360.algorithms.core.execution.prompts import PreparedPrompt
from aisteer360.algorithms.core.output import Output


@dataclass(frozen=True, slots=True, eq=False)
class HookEntry:
    """One state control's torch-hook contribution, consumed by in-process sessions.

    Attributes:
        hooks: Hook specifications keyed by phase (`"pre"`, `"forward"`, `"backward"`), as
            returned by `StateControl.get_hooks`.
    """

    hooks: Mapping[str, list]


@dataclass(frozen=True, slots=True)
class InterventionEntry:
    """One state control's intervention-spec contribution, consumed by intervention-capable
    backends.

    Attributes:
        spec: The serialized intervention.
    """

    spec: InterventionSpec


StateControlEntry = HookEntry | InterventionEntry


@dataclass(frozen=True, slots=True, eq=False)
class StackEntry:
    """One output control's live processor and criteria contribution, consumed by in-process
    sessions.

    Attributes:
        logits_processors: HF `LogitsProcessor`-style objects, in contribution order.
        stopping_criteria: HF `StoppingCriteria`-style objects, in contribution order.
    """

    logits_processors: tuple = ()
    stopping_criteria: tuple = ()


@dataclass(frozen=True, slots=True)
class ConstraintEntry:
    """An output control's contribution as a declarative constrained-decoding source.

    Consumed by backends advertising `Capability.GUIDED_DECODING`, rendered onto the engine's
    native structured-output request parameters in place of the control's live processor.

    Attributes:
        source: The declarative constraint.
    """

    source: ConstraintSource


@dataclass(frozen=True, slots=True)
class ProcessorSpecEntry:
    """One output control's engine-hosted processor contribution.

    Attributes:
        spec: The serialized processor.
    """

    spec: ProcessorSpec


OutputControlEntry = StackEntry | ProcessorSpecEntry | ConstraintEntry


@dataclass(frozen=True, slots=True, eq=False)
class GenerationItem:
    """One prompt's unit of generation work.

    Input controls have no entry because their contribution is already folded into `prompt`;
    structural controls have none because they contribute at steer time through artifacts.

    Attributes:
        prompt: The prepared prompt.
        state_entries: Enabled state controls' contributions, in controls-list order.
        output_entries: Enabled output controls' contributions, in controls-list order.
        seed: Per-item sampling seed, or None for unseeded operation.
    """

    prompt: PreparedPrompt
    state_entries: tuple[StateControlEntry, ...] = ()
    output_entries: tuple[OutputControlEntry, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ScoringItem:
    """One prompt's unit of scoring work (teacher-forced reference tokens).

    Only controls participating in scoring contribute entries, and stopping criteria are never
    applied (there is no loop to stop).

    Attributes:
        prompt: The prepared prompt.
        ref_output_ids: Reference tokens to score, shape `[ref_len]` or `[1, ref_len]`.
        state_entries: Enabled state controls' contributions, in controls-list order.
        output_entries: Scoring-participant output controls' contributions, in controls-list
            order.
    """

    prompt: PreparedPrompt
    ref_output_ids: torch.Tensor
    state_entries: tuple[StateControlEntry, ...] = ()
    output_entries: tuple[OutputControlEntry, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class ItemResult:
    """The result of one generation item.

    Attributes:
        index: Position of the item in the submitted sequence.
        output: The generation record. For `n > 1` the record's batch dimension holds the
            candidates in request order and `finish_reason` reflects the first candidate.
    """

    index: int
    output: Output


@dataclass(frozen=True, slots=True, eq=False)
class CaptureResult:
    """Hidden states captured by `SteeringSession.capture`.

    Attributes:
        hidden: Tensors keyed by 0-based layer id. Shape `[N, T, H]` in `"all_tokens"` mode and
            `[N, H]` in `"last_token"` mode, on CPU, in the model's native dtype.
        attention_mask: Mask of shape `[N, T]` matching the captured prompts, on CPU.
        mode: The capture mode the tensors were produced under.
        location: The capture location (`"layer_output"` or `"layer_input"`).
    """

    hidden: Mapping[int, torch.Tensor]
    attention_mask: torch.Tensor
    mode: str
    location: str


__all__ = [
    "HookEntry",
    "InterventionEntry",
    "StateControlEntry",
    "StackEntry",
    "ProcessorSpecEntry",
    "OutputControlEntry",
    "GenerationItem",
    "ScoringItem",
    "ItemResult",
    "CaptureResult",
    "ConstraintEntry",
]
