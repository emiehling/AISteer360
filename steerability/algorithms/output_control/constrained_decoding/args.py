"""Constrained decoding argument validation."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.payloads import ConstraintSource, as_constraint_source


@dataclass
class ConstrainedDecodingArgs(BaseArgs):
    """Arguments for `ConstrainedDecoding`.

    Exactly one constraint must be given, either as a declarative source or as an automaton object. A declarative
    source is given by `source` or by one of the kind fields `json_schema`, `regex`, `grammar`, and `choice`. A kind
    field also sets `source` to the `ConstraintSource` it describes and keeps its own value. A `source` given together
    with one kind field is accepted when both describe the same constraint. A declarative source runs in process or
    on a vLLM backend, and an automaton object runs in process only.

    Attributes:
        source: The declarative constraint, as a `ConstraintSource` or as a mapping with `kind` and `value` keys,
            which is converted to a `ConstraintSource`.
        json_schema: A JSON schema, as a string or a mapping. Equivalent to
            `source=ConstraintSource(kind="json_schema", value=json_schema)`.
        regex: A regular expression. Equivalent to `source=ConstraintSource(kind="regex", value=regex)`.
        grammar: An EBNF grammar. Equivalent to `source=ConstraintSource(kind="grammar", value=grammar)`.
        choice: A sequence of strings, one of which the output must match exactly. Equivalent to
            `source=ConstraintSource(kind="choice", value=choice)`.
        automaton: An in-memory object that implements the `ConstraintAutomaton` protocol (`reset(prefix_ids)` and
            `allowed(prefix_ids)`).
        include_in_scoring: Whether the constraint applies when `compute_logprobs` scores sequences (default True).
            Scoring with the constraint requires the in-process backend, since structured outputs on vLLM do not apply
            to prompt logprobs. With False, scores do not reflect the constraint.

    Raises:
        ValueError: If no constraint or more than one constraint is given, or if `source` is a mapping with an
            unknown kind.
        TypeError: If a constraint value does not have the type its kind requires (e.g., a `choice` that is not a
            non-empty sequence of strings), or if `source` is neither a `ConstraintSource` nor a mapping.
        KeyError: If `source` is a mapping without a `kind` or `value` key.
    """

    source: ConstraintSource | Mapping | None = None
    json_schema: str | Mapping | None = None
    regex: str | None = None
    grammar: str | None = None
    choice: Sequence[str] | None = None
    automaton: Any | None = None
    include_in_scoring: bool = True

    def __post_init__(self):
        convenience = {
            "json_schema": self.json_schema,
            "regex": self.regex,
            "grammar": self.grammar,
            "choice": self.choice,
        }
        supplied = [name for name, value in convenience.items() if value is not None]
        given = int(self.source is not None) + int(self.automaton is not None) + len(supplied)
        # args built from a convenience field contain the derived source next to that field
        derived = given == 2 and len(supplied) == 1 and self.source is not None
        if derived:
            kind = supplied[0]
            derived = as_constraint_source(self.source) == ConstraintSource(kind=kind, value=convenience[kind])
        if given != 1 and not derived:
            raise ValueError(
                "Provide exactly one constraint: source, automaton, or one of "
                "json_schema/regex/grammar/choice."
            )
        if self.source is not None:
            object.__setattr__(self, "source", as_constraint_source(self.source))
        elif supplied:
            kind = supplied[0]
            object.__setattr__(
                self, "source", ConstraintSource(kind=kind, value=convenience[kind])
            )
