"""The result of evaluating a candidate, its operating point, and the ladder diagnostics.

A point's loss on a constraint is `-capability[name].mean`, its gain is `behavior.mean`, and its level is the largest
`loss / margin` over the constraints. `Result.summary` is what a proposer may see. It contains numbers and short
enumerated strings computed from search-pool aggregates only.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from steerability.algorithms.core.identity import canonical_json
from steerability.optimization.measurement.stats import Paired

RESULT_SCHEMA = "result/1"

REASONS = (
    "invalid", "unsupported", "fit_failed", "no_behavior_gain", "capability_fail", "no_feasible_point",
    "not_confirmed", "over_cost", "degenerate", "budget", "timeout",
)


def jsonable(value: Any) -> Any:
    """The JSON form of a candidate or position value.

    Plain data is written as JSON (tuples become lists). Any other value is written in its canonical identity
    encoding, which identifies it but does not decode back to it.
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return json.loads(canonical_json(value))


def compact(value: Any) -> str:
    """Compact JSON text of a value with sorted keys."""
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def level_of(capability: Mapping[str, Paired], margins: Mapping[str, float]) -> float:
    """The largest `loss / margin` over the constraints (0.0 without constraints).

    Args:
        capability: Mapping from constraint benchmark name to its table.
        margins: Mapping from constraint benchmark name to its margin.

    Returns:
        The level.
    """
    levels = [-capability[name].mean / margin for name, margin in margins.items() if name in capability]
    return max(levels) if levels else 0.0


def _tables_json(tables: Mapping[str, Paired]) -> dict[str, list[float]]:
    return {name: table.as_list() for name, table in tables.items()}


def _tables_from_json(data: Mapping[str, Any]) -> dict[str, Paired]:
    return {name: Paired.from_list(values) for name, values in data.items()}


@dataclass(frozen=True)
class LadderPoint:
    """The ladder-block diagnostics of one position of a solved axis; never a reported estimate.

    Attributes:
        position: Mapping from member path to value.
        recipe_id: The position's recipe id.
        resolved_id: Its resolved id.
        capability: Mapping from constraint benchmark name to the differences, candidate minus baseline.
        behavior: The behavior differences.
        rank: The position's place in the confirmation order, or None when it was not kept.
    """

    position: Mapping[str, Any]
    recipe_id: str
    resolved_id: str
    capability: Mapping[str, Paired]
    behavior: Paired
    rank: int | None

    def to_json(self) -> dict:
        return {
            "position": jsonable(dict(self.position)), "recipe_id": self.recipe_id, "resolved_id": self.resolved_id,
            "capability": _tables_json(self.capability), "behavior": self.behavior.as_list(), "rank": self.rank,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LadderPoint:
        return cls(
            position=dict(data["position"]), recipe_id=data["recipe_id"], resolved_id=data["resolved_id"],
            capability=_tables_from_json(data["capability"]), behavior=Paired.from_list(data["behavior"]),
            rank=data["rank"],
        )


@dataclass(frozen=True)
class OperatingPoint:
    """The position a candidate is reported at, with tables from fresh-block items only.

    Attributes:
        position: Mapping from member path to value; empty without a solved axis.
        recipe_id: The position's recipe id.
        resolved_id: Its resolved id.
        capability: Mapping from constraint benchmark name to the differences, candidate minus baseline.
        behavior: The behavior differences.
        cost_ratio: The ratio of generated tokens per item to the baseline's.
        degenerate_increase: The degenerate rate of the behavior completions minus the baseline's.
        eligible: Whether the point passed confirmation and the cost and degeneracy limits.
    """

    position: Mapping[str, Any]
    recipe_id: str
    resolved_id: str
    capability: Mapping[str, Paired]
    behavior: Paired
    cost_ratio: float
    degenerate_increase: float
    eligible: bool

    @property
    def gain(self) -> float:
        """The mean behavior difference."""
        return self.behavior.mean

    def loss(self, name: str) -> float:
        """The loss on a constraint benchmark, `-capability[name].mean`."""
        return -self.capability[name].mean

    def to_json(self) -> dict:
        return {
            "position": jsonable(dict(self.position)), "recipe_id": self.recipe_id, "resolved_id": self.resolved_id,
            "capability": _tables_json(self.capability), "behavior": self.behavior.as_list(),
            "cost_ratio": self.cost_ratio, "degenerate_increase": self.degenerate_increase, "eligible": self.eligible,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> OperatingPoint:
        return cls(
            position=dict(data["position"]), recipe_id=data["recipe_id"], resolved_id=data["resolved_id"],
            capability=_tables_from_json(data["capability"]), behavior=Paired.from_list(data["behavior"]),
            cost_ratio=float(data["cost_ratio"]), degenerate_increase=float(data["degenerate_increase"]),
            eligible=bool(data["eligible"]),
        )


def summarize(
    *,
    capability: Mapping[str, Paired] | None = None,
    behavior: Paired | None = None,
    margins: Mapping[str, float],
    fitness_alpha: float,
    cost_ratio: float | None = None,
    degenerate_increase: float | None = None,
    position: Mapping[str, Any] | None = None,
) -> dict[str, float | str]:
    """The summary of what was measured, with only the keys whose values are known and finite.

    Args:
        capability: Mapping from constraint benchmark name to its table, or None.
        behavior: The behavior table, or None.
        margins: Mapping from constraint benchmark name to its margin.
        fitness_alpha: The level of the gain's lower bound.
        cost_ratio: The cost ratio, or None.
        degenerate_increase: The degenerate increase, or None.
        position: The position, or None.

    Returns:
        A mapping with `gain`, `gain_lower`, `level`, `cost_ratio`, `degenerate_increase`, one `loss.<name>` per
        measured constraint, and `position` as compact JSON text, each when measured.
    """
    summary: dict[str, float | str] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            summary[key] = float(value)

    if behavior is not None and behavior.n:
        put("gain", behavior.mean)
        put("gain_lower", behavior.lower(fitness_alpha))
    if capability:
        measured = {name: table for name, table in capability.items() if table.n}
        for name, table in measured.items():
            put(f"loss.{name}", -table.mean)
        if measured and set(margins) <= set(measured):
            put("level", level_of(measured, margins))
    put("cost_ratio", cost_ratio)
    put("degenerate_increase", degenerate_increase)
    if position is not None:
        summary["position"] = compact(dict(position))
    return summary


@dataclass(frozen=True)
class Result:
    """The outcome of evaluating a candidate, as recorded in the archive.

    Attributes:
        candidate_id: The candidate's id (empty for a candidate that does not bind).
        study_id: The study's id.
        version: The result's version; a re-score appends a version.
        round_index: The round the candidate was calibrated in.
        status: `"accepted"` when the candidate has an eligible operating point, else `"rejected"`.
        stage: The stage that decided the result (0 to 5).
        reason: The rejection reason (one of `REASONS`), or None for an accepted result.
        fitness: The one-sided lower bound of the operating point's gain at `settings.fitness_alpha`, floored at 0;
            0 for a rejected result.
        operating: The operating point, or None.
        ladder: One `LadderPoint` per evaluated position of a solved axis.
        summary: What a proposer may see (see `summarize`).
        generations: The generations this evaluation ran.
        seconds: The wall time of this evaluation.
    """

    candidate_id: str
    study_id: str
    version: int
    round_index: int
    status: Literal["accepted", "rejected"]
    stage: int
    reason: str | None
    fitness: float
    operating: OperatingPoint | None
    ladder: tuple[LadderPoint, ...] = ()
    summary: Mapping[str, float | str] = field(default_factory=dict)
    generations: int = 0
    seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        """Whether the result is accepted."""
        return self.status == "accepted"

    @property
    def level(self) -> float | None:
        """The level of the operating point from the summary, or None when it was not measured."""
        value = self.summary.get("level")
        return None if value is None else float(value)

    def to_json(self) -> dict:
        """The stored JSON form, with each `Paired` as its `as_list()`."""
        return {
            "schema": RESULT_SCHEMA, "candidate_id": self.candidate_id, "study_id": self.study_id,
            "version": self.version, "round_index": self.round_index, "status": self.status, "stage": self.stage,
            "reason": self.reason, "fitness": self.fitness,
            "operating": None if self.operating is None else self.operating.to_json(),
            "ladder": [point.to_json() for point in self.ladder], "summary": dict(self.summary),
            "generations": self.generations, "seconds": self.seconds,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Result:
        """A result from its stored JSON form.

        Raises:
            ValueError: If the schema differs.
        """
        if data.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"Expected a {RESULT_SCHEMA!r} result; got schema {data.get('schema')!r}.")
        operating = data.get("operating")
        return cls(
            candidate_id=data["candidate_id"], study_id=data["study_id"], version=int(data["version"]),
            round_index=int(data["round_index"]), status=data["status"], stage=int(data["stage"]),
            reason=data.get("reason"), fitness=float(data["fitness"]),
            operating=None if operating is None else OperatingPoint.from_json(operating),
            ladder=tuple(LadderPoint.from_json(point) for point in data.get("ladder") or ()),
            summary=dict(data.get("summary") or {}), generations=int(data.get("generations", 0)),
            seconds=float(data.get("seconds", 0.0)),
        )


__all__ = [
    "LadderPoint",
    "OperatingPoint",
    "REASONS",
    "Result",
    "compact",
    "jsonable",
    "level_of",
    "summarize",
]
