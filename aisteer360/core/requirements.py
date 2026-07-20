"""Capability/requirements vocabulary for backend negotiation.

Controls declare `Requirements` (the capabilities they need, and in which phase); backends declare
`BackendCapabilities` (doc 02). `SteeringPipeline.validate()` (doc 04) intersects the two and yields
one `ControlVerdict` per control — `supported`, `degraded`, or `unsupported` — collected into a
`ValidationReport`. This module owns only the control-side and report-side vocabulary; the
backend-side `BackendCapabilities` lives in `backends/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Literal


class Capability(Flag):
    """A capability a backend may grant and a control may require.

    Combinable with `|`; membership tested with `&`. The empty set is `Capability(0)`.
    """

    MESSAGES = auto()            # accepts chat-message prompts
    TEXT = auto()                # accepts text prompts
    TOKEN_IDS = auto()           # accepts token-array prompts
    SCORING = auto()             # ref-continuation logprobs
    RESIDUAL_WRITE = auto()      # additive/ablative edits at layer boundaries
    HIDDEN_READ = auto()         # hidden-state capture / condition scoring
    SERVER_GATING = auto()       # conditions evaluated in-engine
    ATTENTION_WRITE = auto()     # attention-mask/QK edits (PASTA)
    FORWARD_HOOKS = auto()       # arbitrary in-process hooks
    RAW_MODEL = auto()           # direct PreTrainedModel access
    WEIGHT_TRAINING = auto()     # in-process weight updates
    STEPWISE_LOGITS = auto()     # per-decode-step logits access
    LORA_ARTIFACT = auto()       # accepts a LoRA adapter artifact
    CHECKPOINT_ARTIFACT = auto()  # accepts a full-checkpoint artifact


@dataclass(frozen=True)
class Requirements:
    """The capabilities a control needs, and the phase in which it needs them.

    Attributes:
        capabilities: The capabilities the control requires (empty for a disabled control).
        phase: `"steer"` for fitting/training requirements (validated against the steering backend);
            `"generate"` for inference requirements (validated against the inference backend).
        notes: Free-form notes surfaced in validation output (e.g. a pointer to a deferred feature).
    """

    capabilities: Capability = Capability(0)
    phase: Literal["steer", "generate"] = "generate"
    notes: tuple[str, ...] = ()


@dataclass
class ControlVerdict:
    """One control's runnability verdict against a backend.

    Attributes:
        control: The control's display name.
        status: `"supported"`, `"degraded"`, or `"unsupported"`.
        missing: Capabilities the backend does not grant (empty unless `unsupported`).
        note: A semantic note attached when `degraded` (e.g. a latency caveat).
        fix: An actionable remedy attached when `unsupported` (e.g. the backend to run on).
    """

    control: str
    status: Literal["supported", "degraded", "unsupported"]
    missing: Capability = Capability(0)
    note: str = ""
    fix: str = ""


@dataclass
class ValidationReport:
    """The collected per-control verdicts for a pipeline against a backend.

    Attributes:
        verdicts: One `ControlVerdict` per validated control.
    """

    verdicts: list[ControlVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """`True` when no control is `unsupported`."""
        return not any(verdict.status == "unsupported" for verdict in self.verdicts)

    def raise_if_failed(self) -> None:
        """Raise a single actionable error aggregating every `unsupported` verdict's fix.

        Raises:
            RuntimeError: If any control is `unsupported`. The message lists each failing control,
                the capabilities it is missing, and its suggested fix.
        """
        failures = [verdict for verdict in self.verdicts if verdict.status == "unsupported"]
        if not failures:
            return
        lines = ["Pipeline is not runnable on the selected backend:"]
        for verdict in failures:
            missing = _render_capability(verdict.missing)
            line = f"  - {verdict.control}: missing {missing}"
            if verdict.fix:
                line += f" — {verdict.fix}"
            lines.append(line)
        raise RuntimeError("\n".join(lines))

    def __str__(self) -> str:
        if not self.verdicts:
            return "ValidationReport(empty)"
        lines = ["ValidationReport:"]
        for verdict in self.verdicts:
            line = f"  [{verdict.status:>11}] {verdict.control}"
            if verdict.status == "degraded" and verdict.note:
                line += f" — {verdict.note}"
            elif verdict.status == "unsupported":
                line += f" — missing {_render_capability(verdict.missing)}"
                if verdict.fix:
                    line += f"; {verdict.fix}"
            lines.append(line)
        return "\n".join(lines)


def _render_capability(capability: Capability) -> str:
    """Render a capability flag as a `|`-joined list of member names."""
    if not capability:
        return "(none)"
    return " | ".join(member.name for member in Capability if member & capability)
