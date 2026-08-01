"""Declarative constrained-decoding source: `ConstraintSource`."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CONSTRAINT_KINDS = ("json_schema", "regex", "grammar", "choice")


@dataclass(frozen=True, slots=True)
class ConstraintSource:
    """A declarative constrained-decoding specification.

    The portable form of the constraint class: one source renders per execution arm, compiled
    into a client-side automaton in process and onto the engine's native structured-output
    request parameters on vLLM backends.

    Attributes:
        kind: The constraint kind: `"json_schema"`, `"regex"`, `"grammar"` (EBNF), or
            `"choice"`.
        value: The constraint payload: a schema string or mapping for `"json_schema"`, a
            pattern string for `"regex"`, a grammar string for `"grammar"`, or a sequence of
            candidate strings for `"choice"`.
    """

    kind: Literal["json_schema", "regex", "grammar", "choice"]
    value: str | Mapping | Sequence[str]

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(
                f"Unknown constraint kind {self.kind!r}; kinds are {', '.join(CONSTRAINT_KINDS)}."
            )
        if self.kind == "json_schema":
            if not isinstance(self.value, (str, Mapping)):
                raise TypeError("A json_schema constraint takes a schema string or mapping.")
        elif self.kind in ("regex", "grammar"):
            if not isinstance(self.value, str):
                raise TypeError(f"A {self.kind} constraint takes a string.")
        else:
            if isinstance(self.value, str) or not isinstance(self.value, Sequence) or not self.value:
                raise TypeError("A choice constraint takes a non-empty sequence of strings.")
            if not all(isinstance(item, str) for item in self.value):
                raise TypeError("A choice constraint takes a non-empty sequence of strings.")
            object.__setattr__(self, "value", tuple(self.value))


def as_constraint_source(value: "ConstraintSource | Mapping[str, Any]") -> ConstraintSource:
    """Coerce a mapping with `kind` and `value` keys into a `ConstraintSource`."""
    if isinstance(value, ConstraintSource):
        return value
    if isinstance(value, Mapping):
        return ConstraintSource(kind=value["kind"], value=value["value"])
    raise TypeError(
        f"Expected a ConstraintSource or a mapping with 'kind' and 'value'; got {type(value).__name__}."
    )
