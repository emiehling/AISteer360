"""The requirement language controls use to state what a backend must provide.

A control computes a `Requirements` instance from its validated `Args`, separately for the
steer, generate, and score phases. Each phase holds zero or more `Alternative`s; the phase is
satisfied by its first satisfied alternative. An absent phase requires nothing beyond the
session contract. Requirements may additionally carry `SpecConstraint`s, predicates over the
resolved `BackendSpec` options, for facts that are configuration of a backend rather than a
capability of it.
"""
from collections.abc import Callable
from dataclasses import dataclass

from aisteer360.algorithms.core.execution.capabilities import (
    ConstraintKinds,
    BackendCapabilities,
    Capability,
    CaptureKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.spec import BackendSpec

KindSet = InterventionKinds | ProcessorKinds | CaptureKinds | ConstraintKinds

PHASES: tuple[str, ...] = ("steer", "generate", "score")


@dataclass(frozen=True, slots=True)
class Alternative:
    """One way to satisfy a phase requirement, i.e., a conjunction of capability atoms with
    optional kind predicates over the backend's advertised kind sets.

    Attributes:
        atoms: Capability atoms that must all be advertised.
        kinds: Kind sets whose names must all be contained in the backend's advertisement of the
            corresponding kind-set type.
        hint: Optional fix text used in unsupported-verdict messages in place of the default.
    """

    atoms: frozenset[Capability] = frozenset()
    kinds: tuple[KindSet, ...] = ()
    hint: str | None = None

    def satisfied_by(self, capabilities: BackendCapabilities) -> bool:
        """Return True when every atom is advertised and every kind set is contained."""
        if not self.atoms <= capabilities.atoms:
            return False
        for kind_set in self.kinds:
            advertised = _advertised_for(kind_set, capabilities)
            if advertised is None or not advertised.contains(kind_set):
                return False
        return True

    def missing(self, capabilities: BackendCapabilities) -> list[str]:
        """Names of the atoms and kind sets this alternative needs but `capabilities` lacks."""
        gaps = [atom.name for atom in sorted(self.atoms - capabilities.atoms, key=lambda a: a.name)]
        for kind_set in self.kinds:
            advertised = _advertised_for(kind_set, capabilities)
            if advertised is None or not advertised.contains(kind_set):
                gaps.append(f"{type(kind_set).__name__}({_kind_names(kind_set)})")
        return gaps


def _advertised_for(kind_set: KindSet, capabilities: BackendCapabilities) -> KindSet | None:
    """The backend's advertised kind set of the same type as `kind_set`, or None."""
    if isinstance(kind_set, InterventionKinds):
        return capabilities.intervention_kinds
    if isinstance(kind_set, ProcessorKinds):
        return capabilities.processor_kinds
    if isinstance(kind_set, ConstraintKinds):
        return capabilities.constraint_kinds
    return capabilities.capture_kinds


def _kind_names(kind_set: KindSet) -> str:
    """Comma-joined sorted kind names across the set's name-bearing fields."""
    if isinstance(kind_set, InterventionKinds):
        names = kind_set.transforms | kind_set.modifiers | kind_set.scopes | kind_set.gates
    elif isinstance(kind_set, ProcessorKinds):
        names = kind_set.processors
    elif isinstance(kind_set, ConstraintKinds):
        names = kind_set.constraints
    else:
        names = kind_set.kinds | kind_set.locations | kind_set.modes
    return ", ".join(sorted(names))


def needs(
    *atoms: Capability,
    kinds: KindSet | tuple[KindSet, ...] | None = None,
    hint: str | None = None,
) -> tuple[Alternative, ...]:
    """Build a single-alternative phase requirement.

    Args:
        *atoms: Capability atoms that must all be advertised.
        kinds: One kind set, or a tuple of kind sets, whose names must be contained in the
            backend's advertisement.
        hint: Optional fix text for unsupported-verdict messages.

    Returns:
        A one-element tuple of `Alternative`, directly assignable to a `Requirements` phase.
    """
    if kinds is None:
        kind_sets: tuple[KindSet, ...] = ()
    elif isinstance(kinds, tuple):
        kind_sets = kinds
    else:
        kind_sets = (kinds,)
    return (Alternative(atoms=frozenset(atoms), kinds=kind_sets, hint=hint),)


def any_of(*alternatives: tuple[Alternative, ...] | Alternative) -> tuple[Alternative, ...]:
    """Combine alternatives into a disjunction, satisfied by its first satisfied alternative.

    Args:
        *alternatives: `Alternative` instances or tuples of them (as returned by `needs`).

    Returns:
        The flattened tuple of alternatives.
    """
    flattened: list[Alternative] = []
    for alternative in alternatives:
        if isinstance(alternative, Alternative):
            flattened.append(alternative)
        else:
            flattened.extend(alternative)
    return tuple(flattened)


@dataclass(frozen=True, slots=True)
class SpecConstraint:
    """A predicate over a resolved `BackendSpec`, for backend-configuration facts.

    Attributes:
        description: The unsupported-verdict message shown when the predicate fails. It should
            name the conflict and a fix.
        predicate: Callable evaluated against the phase's `BackendSpec`; True means satisfied.
        phases: Phases whose backend spec the predicate is evaluated against.
    """

    description: str
    predicate: Callable[[BackendSpec], bool]
    phases: tuple[str, ...] = ("steer", "generate")

    def __post_init__(self) -> None:
        unknown = [phase for phase in self.phases if phase not in PHASES]
        if unknown:
            raise ValueError(f"Unknown phases {unknown}; phases are {', '.join(PHASES)}.")


@dataclass(frozen=True, slots=True)
class Requirements:
    """Phase-keyed backend requirements computed by a control instance.

    Each phase holds a tuple of `Alternative`s (a disjunction); an empty tuple requires nothing
    beyond the session contract, which includes the model layout.

    Attributes:
        steer: Alternatives for the steer phase, evaluated against the steering backend.
        generate: Alternatives for the generate phase, evaluated against the inference backend.
        score: Alternatives for the score phase, evaluated against the inference backend.
        spec_constraints: Backend-configuration predicates, each evaluated against the spec of
            every phase it names.
    """

    steer: tuple[Alternative, ...] = ()
    generate: tuple[Alternative, ...] = ()
    score: tuple[Alternative, ...] = ()
    spec_constraints: tuple[SpecConstraint, ...] = ()

    def for_phase(self, phase: str) -> tuple[Alternative, ...]:
        """The alternatives for `phase` (one of `"steer"`, `"generate"`, `"score"`).

        Raises:
            ValueError: If `phase` is not a known phase name.
        """
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; phases are {', '.join(PHASES)}.")
        return getattr(self, phase)
