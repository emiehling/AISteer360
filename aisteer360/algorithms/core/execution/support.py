"""Binary per-control, per-phase support verdicts against a backend pair.

Each verdict is supported or unsupported. Unsupported verdicts name the missing atoms, kind
names, or violated spec constraints and a fix. Only enabled controls impose requirements;
disabled controls (including a pipeline's default identity controls) never gate a backend and
do not appear in the report.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aisteer360.algorithms.core.execution.capabilities import (
    BackendCapabilities,
    Capability,
)
from aisteer360.algorithms.core.execution.requirements import PHASES, Requirements
from aisteer360.algorithms.core.execution.spec import BackendSpec

_DEFAULT_HINT = "run this pipeline on the huggingface backend"


class UnsupportedPipelineError(RuntimeError):
    """Raised when an operation targets a backend pair that does not support the pipeline.

    Attributes:
        report: The `SupportReport` whose failures triggered the error.
    """

    def __init__(self, report: "SupportReport", phases: tuple[str, ...]) -> None:
        self.report = report
        failures = report.failures_for(*phases)
        lines = "\n".join(f"- {failure.message}" for failure in failures)
        super().__init__(
            f"Pipeline is unsupported on the configured backends ({len(failures)} unsupported "
            f"requirement(s)):\n{lines}"
        )


class UnsupportedOperationError(RuntimeError):
    """Raised when a session receives work it cannot execute on its backend."""


@dataclass(frozen=True, slots=True)
class SupportFailure:
    """One unsupported verdict.

    Attributes:
        control: Class name of the failing control.
        phase: The phase the verdict applies to (`"steer"`, `"generate"`, or `"score"`).
        message: Stable, tested message naming the gap and a fix.
    """

    control: str
    phase: str
    message: str


@dataclass(frozen=True, slots=True)
class SupportReport:
    """The result of evaluating every enabled control against a backend pair.

    Attributes:
        steer_spec: The steering backend spec the steer phase was evaluated against.
        inference_spec: The inference backend spec the generate and score phases were evaluated
            against.
        failures: All unsupported verdicts, in controls-list order then phase order.
    """

    steer_spec: BackendSpec
    inference_spec: BackendSpec
    failures: tuple[SupportFailure, ...] = ()

    @property
    def ok(self) -> bool:
        """True when no phase of any enabled control is unsupported."""
        return not self.failures

    def failures_for(self, *phases: str) -> tuple[SupportFailure, ...]:
        """The failures whose phase is among `phases`."""
        return tuple(failure for failure in self.failures if failure.phase in phases)

    def supported(self, *phases: str) -> bool:
        """True when no failure falls in any of `phases`."""
        return not self.failures_for(*phases)

    def raise_for(self, *phases: str) -> None:
        """Raise `UnsupportedPipelineError` listing every failing control in `phases`, if any."""
        if not self.supported(*phases):
            raise UnsupportedPipelineError(self, phases)


def _spec_for_phase(phase: str, steer_spec: BackendSpec, inference_spec: BackendSpec) -> BackendSpec:
    return steer_spec if phase == "steer" else inference_spec


def _phase_failure_message(
    control_name: str,
    phase: str,
    spec: BackendSpec,
    requirements: Requirements,
    capabilities: BackendCapabilities,
) -> str:
    """Build the unsupported message for one control phase, naming the gaps and a fix."""
    alternatives = requirements.for_phase(phase)
    gap_parts = []
    hint = None
    for alternative in alternatives:
        gaps = alternative.missing(capabilities)
        gap_parts.append(" + ".join(gaps) if gaps else "unsatisfied alternative")
        if hint is None and alternative.hint is not None:
            hint = alternative.hint
    if hint is None:
        missing_atom_names = {gap for part in gap_parts for gap in part.split(" + ")}
        if Capability.IN_PROCESS_TORCH.name in missing_atom_names:
            hint = _DEFAULT_HINT
    message = (
        f"{control_name} is unsupported at {phase} on backend kind '{spec.kind}': "
        f"missing {' or '.join(gap_parts)}"
    )
    return f"{message}; {hint}." if hint else f"{message}."


def evaluate_support(
    controls: Iterable[Any],
    steer_spec: BackendSpec,
    inference_spec: BackendSpec,
    steer_capabilities: BackendCapabilities,
    inference_capabilities: BackendCapabilities,
) -> SupportReport:
    """Evaluate every enabled control's requirements against a backend pair.

    For each enabled control, `control.requirements()` is read once and each declared phase is
    checked against the matching backend's capabilities (`steer` against the steering backend,
    `generate` and `score` against the inference backend). Spec constraints are checked against
    the spec of every phase they name. Controls whose `enabled` attribute is False are skipped.

    Args:
        controls: Control instances, in pipeline order.
        steer_spec: The steering backend spec.
        inference_spec: The inference backend spec.
        steer_capabilities: Capability advertisement of the steering backend.
        inference_capabilities: Capability advertisement of the inference backend.

    Returns:
        A `SupportReport` whose `failures` hold one entry per unsupported (control, phase) pair
        and per violated spec constraint.
    """
    failures: list[SupportFailure] = []
    for control in controls:
        if not getattr(control, "enabled", True):
            continue
        control_name = type(control).__name__
        requirements: Requirements = control.requirements()

        for phase in PHASES:
            alternatives = requirements.for_phase(phase)
            if not alternatives:
                continue
            capabilities = steer_capabilities if phase == "steer" else inference_capabilities
            if any(alternative.satisfied_by(capabilities) for alternative in alternatives):
                continue
            spec = _spec_for_phase(phase, steer_spec, inference_spec)
            failures.append(SupportFailure(
                control=control_name,
                phase=phase,
                message=_phase_failure_message(control_name, phase, spec, requirements, capabilities),
            ))

        for constraint in requirements.spec_constraints:
            for phase in constraint.phases:
                spec = _spec_for_phase(phase, steer_spec, inference_spec)
                if constraint.predicate(spec):
                    continue
                failures.append(SupportFailure(
                    control=control_name,
                    phase=phase,
                    message=(
                        f"{control_name} is unsupported at {phase} on backend kind "
                        f"'{spec.kind}': {constraint.description}"
                    ),
                ))

    return SupportReport(steer_spec=steer_spec, inference_spec=inference_spec, failures=tuple(failures))
