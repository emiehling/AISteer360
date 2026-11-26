"""The description of a study's search for an engine that asks a language model for candidates.

Such an engine needs the JSON form of a candidate (`candidate_document`, `parse_document`), a statement of the task
and its goal (`describe_task`), and a description of a structure and its axes (`describe_structure`,
`describe_axes`). The descriptions contain the study's structures, the ranges and help texts of their arguments, and
the goal its author wrote, and no benchmark item text.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import Choice, Interval, Text
from steerability.optimization.evaluation.result import jsonable
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.algorithms.core.recipe import Axis
    from steerability.optimization.study import Study


def candidate_document(*, candidate: Candidate) -> str:
    """The JSON object `{"structure": ..., "values": ...}` of a candidate, with sorted keys.

    Args:
        candidate: The candidate.

    Returns:
        The JSON text, on one line.
    """
    document = {"structure": candidate.structure, "values": jsonable(value=dict(candidate.values))}
    return json.dumps(document, sort_keys=True)


def parse_document(*, text: str) -> Candidate:
    """The candidate of a JSON document, i.e., exactly one object with exactly the keys `structure` and `values`.

    Whether the candidate is valid is `Study.bind`'s decision.

    Args:
        text: The JSON text, with or without surrounding whitespace.

    Returns:
        The candidate.

    Raises:
        ValueError: If the text is not exactly one JSON object, the object has keys other than `"structure"` and
            `"values"`, or their values have the wrong type.
    """
    text = text.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The text does not start with a JSON object ({error}).") from error
    if text[end:].strip():
        raise ValueError("The text contains more than one JSON value.")
    if not isinstance(value, dict):
        raise ValueError(f"The JSON value is a {type(value).__name__}, not an object.")
    if set(value) != {"structure", "values"}:
        raise ValueError(f"The object must have exactly the keys 'structure' and 'values'; got {sorted(value)}.")
    if not isinstance(value["structure"], str) or not isinstance(value["values"], dict):
        raise ValueError("'structure' must be a string and 'values' an object.")
    return Candidate(structure=value["structure"], values=value["values"])


def describe_task(*, goal: str) -> str:
    """The paragraph that states what a candidate is, how it is scored, and what the behavior score measures.

    The goal is written by the study's author, and it is sent to a language model with the rest of the paragraph.
    It must therefore not contain benchmark item text.

    Args:
        goal: One or more sentences that state what the behavior score measures, e.g., "The behavior score measures
            how formal the reply reads."

    Returns:
        The paragraph.

    Raises:
        ValueError: If `goal` is not a non-empty string.
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string that states what the behavior score measures.")
    return (
        "You propose steering configurations for a language model. Each program is one candidate, i.e., a structure "
        "(a fixed arrangement of steering controls) and a value for each of the structure's free settings. A "
        "candidate is scored by how much it raises a behavior score above the unsteered model while the model's "
        "accuracy on capability benchmarks stays within a fixed margin of the unsteered model's, and a candidate "
        f"outside the margin scores 0. {goal.strip()}"
    )


def describe_structure(*, study: Study, structure: str) -> str:
    """One line that gives a structure's controls and the setting the evaluator solves, if any.

    Args:
        study: The study.
        structure: A key of `study.space`.

    Returns:
        The line.

    Raises:
        RecipeError: If the structure is unknown.
    """
    solved = study.solved_axis(structure=structure)
    template = study.space[structure]
    controls = ", ".join(
        _control(name=name, method=item.method) for name, item in zip(template.names, template.entries) if item.enabled
    )
    line = f"The structure `{structure}` contains the controls {controls}."
    if solved is not None:
        line += f" The evaluator solves `{solved.key}` for each candidate, and a program does not set it."
    return line


def describe_axes(*, study: Study, structure: str) -> str:
    """The lines that describe the proposer axes of a structure.

    Each line gives an axis path, its range, and the help text of the argument's `Args` field (or of its entry in
    the `Args` docstring). A shared axis gets a line of its own, followed by one line per member.

    Args:
        study: The study.
        structure: A key of `study.space`.

    Returns:
        The lines, joined by newlines, or an empty string for a structure without proposer axes.

    Raises:
        RecipeError: If the structure is unknown.
    """
    lines: list[str] = []
    for axis in study.proposer_axes(structure=structure).values():
        lines.extend(_describe_axis(study=study, structure=structure, axis=axis))
    return "\n".join(lines)


def _control(*, name: str, method: str) -> str:
    short = method.partition("/")[2]
    return f"`{name}`" if name == short else f"`{name}` ({short})"


def _field_help(*, method: str, argument: str) -> str:
    """The help text of an argument of a registered method, from its `Args` field or its docstring."""
    from steerability.algorithms.core.registry import resolve_method_key

    try:
        args_cls = resolve_method_key(key=method).control_cls.Args
    except Exception:
        return ""
    if args_cls is None:
        return ""
    for item in dataclasses.fields(args_cls):
        if item.name == argument and item.metadata.get("help"):
            return " ".join(str(item.metadata["help"]).split())
    docstring = inspect.getdoc(args_cls) or ""
    match = re.search(r"^Attributes:\s*$(.*?)(?=^\S|\Z)", docstring, re.M | re.S)
    if match is None:
        return ""
    lines: list[str] = []
    for line in match.group(1).splitlines():
        found = re.match(r"^\s{4}(\w+):\s*(.*)$", line)
        if found is not None:
            if lines:
                break
            if found.group(1) == argument:
                lines.append(found.group(2))
        elif lines and line.strip():
            lines.append(line.strip())
    return " ".join(" ".join(lines).split())


def _describe_range(*, member: Any) -> str:
    if isinstance(member, Choice):
        return "one of " + ", ".join(json.dumps(jsonable(value=value)) for value in member.values)
    if isinstance(member, Interval):
        kind = "an integer" if member.integer else "a number"
        scale = " (log scale)" if member.log else ""
        return f"{kind} from {member.low:g} to {member.high:g}{scale}"
    if isinstance(member, Text):
        return f"a string of at most {member.max_chars} characters"
    return repr(member)


def _describe_axis(*, study: Study, structure: str, axis: Axis) -> list[str]:
    template = study.space[structure]
    methods = dict(zip(template.names, (item.method for item in template.entries)))
    lines = []
    if axis.key.startswith("@"):
        members = ", ".join(f"`{path}`" for path in axis.members)
        if axis.continuous:
            lines.append(f"- `{axis.key}`: a fraction from 0 to 1 that moves {members} together.")
        else:
            lines.append(f"- `{axis.key}`: a position index from 0 to {axis.count - 1} that moves {members} together.")
    for path, member in axis.members.items():
        name, _, rest = path.partition(".")
        argument = re.split(r"[.\[]", rest, maxsplit=1)[0]
        help_text = _field_help(method=methods.get(name, ""), argument=argument)
        line = f"- `{path}`: {_describe_range(member=member)}."
        if axis.key.startswith("@"):
            line = f"  - `{path}` (member of `{axis.key}`): {_describe_range(member=member)}."
        lines.append(line + (f" {help_text}" if help_text else ""))
    return lines


__all__ = ["candidate_document", "describe_axes", "describe_structure", "describe_task", "parse_document"]
