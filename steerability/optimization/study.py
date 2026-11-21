"""The definition of a study: what is searched, what is measured, and on which items.

A `Study` searches the steering pipelines described by its `space` (a template per structure) for the one that
maximizes a behavior score (`objective`) while the accuracy on each capability benchmark (`constraints`) stays within
a fixed margin of the unsteered model. Items of every benchmark are split into a search pool and a held-out pool
(`pools`); search decisions use the search pool only, and the result is certified once on the held-out pool.

A `Candidate` is a structure and the values of that structure's proposer axes. Each template's ordered axis that is
not listed in `Study.searched` is its solved axis, which the evaluator solves for each candidate, and every other axis
is a proposer axis.

The study's records are stored in the tables of the `"optimization/2"` schema (`ensure_tables`).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.identity import content_id
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Axis, Recipe, RecipeError, Template
from steerability.optimization.measurement.pools import PoolSplit, check_pools

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.evaluation.items import Benchmark

logger = logging.getLogger(__name__)

STUDY_SCHEMA = "study/2"
OPTIMIZATION_SCHEMA = "optimization/2"

TABLES = (
    """CREATE TABLE IF NOT EXISTS studies (study_id TEXT PRIMARY KEY, name TEXT NOT NULL, definition TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS baselines (study_id TEXT NOT NULL, benchmark_id TEXT NOT NULL,
        resolved_id TEXT NOT NULL, noise_floor REAL NOT NULL, excluded TEXT NOT NULL,
        PRIMARY KEY (study_id, benchmark_id))""",
    """CREATE TABLE IF NOT EXISTS rounds (study_id TEXT NOT NULL, round_index INTEGER NOT NULL, batches TEXT NOT NULL,
        PRIMARY KEY (study_id, round_index))""",
    """CREATE TABLE IF NOT EXISTS candidates (study_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
        structure TEXT NOT NULL, assignment TEXT NOT NULL, proposer TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (study_id, candidate_id))""",
    """CREATE TABLE IF NOT EXISTS results (study_id TEXT NOT NULL, candidate_id TEXT NOT NULL, version INTEGER NOT NULL,
        round_index INTEGER NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (study_id, candidate_id, version))""",
    """CREATE TABLE IF NOT EXISTS certifications (study_id TEXT PRIMARY KEY, finalists TEXT NOT NULL, outcome TEXT,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS held_out_uses (study_id TEXT NOT NULL, benchmark_name TEXT NOT NULL,
        held_out_id TEXT NOT NULL, resolved_id TEXT NOT NULL, used_at TEXT NOT NULL)""",
)

_FLOAT_SETTINGS = (
    "alpha", "power", "ladder_power", "min_gain", "fitness_alpha", "max_cost_ratio", "max_degenerate_increase",
)
_INT_SETTINGS = (
    "round_candidates", "ladder_points", "ladder_refinements", "behavior_ladder_items", "behavior_items",
    "quick_behavior_items", "rescore_top", "finalists",
)


class StudyError(RuntimeError):
    """A study cannot proceed, e.g., its baseline has too many errors or it is certified already."""


class StudyInfeasibleError(StudyError):
    """A study's pools are too small for its margins and settings at the variance it is sized for."""


def now() -> str:
    """The current UTC time in the store's timestamp format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def dumps(value: Any) -> str:
    """Compact JSON with sorted keys, as the optimization tables store it."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def ensure_tables(store: Store) -> None:
    """Create the tables of the `"optimization/2"` schema in a store (once)."""
    store.ensure_schema(OPTIMIZATION_SCHEMA, TABLES)


@dataclass(frozen=True)
class Constraint:
    """A capability benchmark whose accuracy may fall by at most `margin` below the unsteered model's.

    Attributes:
        benchmark: The benchmark; its per-item scores are in [0, 1], and higher is better.
        margin: The allowed loss, as an absolute difference (0.03 is three points).
    """

    benchmark: Benchmark
    margin: float

    def __post_init__(self) -> None:
        if isinstance(self.margin, bool) or not isinstance(self.margin, (int, float)) or not 0 < self.margin < 1:
            raise ValueError(f"A constraint margin must be in (0, 1); got {self.margin!r}.")
        object.__setattr__(self, "margin", float(self.margin))


@dataclass(frozen=True)
class Settings:
    """The statistical and budget settings of a study.

    The capability block sizes are not settings. They follow from each constraint's margin and the variance of its
    baseline (`evidence.block_sizes`). The whole ladder block (`ladder_size`) and `ladder_power` serve the calibration
    of a solved axis outside `study.monotone`, and a monotone solved axis reads the first look of its ladder block.

    Attributes:
        alpha: The one-sided error rate of certification.
        power: The design probability that one confirmed point passes certification.
        round_candidates: The number of new candidates per round.
        ladder_points: The number of positions of a continuous solved axis.
        ladder_power: The power of the provisional check on the ladder of a solved axis outside `study.monotone`.
        ladder_refinements: The number of positions calibration may insert on a monotone continuous axis.
        behavior_ladder_items: The behavior items per ladder position.
        behavior_items: The behavior items a point is reported on.
        quick_behavior_items: The behavior items of the quick look (stage 2).
        min_gain: The smallest gain worth keeping.
        fitness_alpha: The level of the one-sided lower bound on the gain that is a candidate's fitness.
        max_cost_ratio: The largest allowed ratio of generated tokens per item to the baseline's.
        max_degenerate_increase: The largest allowed increase of the degenerate rate over the baseline's.
        rescore_top: The number of accepted results re-scored at every new round.
        finalists: The number of finalists taken to certification.
        guard_band: Whether the guard band applies; False sets every band to zero.
    """

    alpha: float = 0.05
    power: float = 0.80
    round_candidates: int = 16
    ladder_points: int = 6
    ladder_power: float = 0.50
    ladder_refinements: int = 1
    behavior_ladder_items: int = 128
    behavior_items: int = 256
    quick_behavior_items: int = 32
    min_gain: float = 0.02
    fitness_alpha: float = 0.10
    max_cost_ratio: float = 2.0
    max_degenerate_increase: float = 0.10
    rescore_top: int = 4
    finalists: int = 5
    guard_band: bool = True

    def __post_init__(self) -> None:
        for name in _FLOAT_SETTINGS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Settings.{name} must be a finite number; got {value!r}.")
            object.__setattr__(self, name, float(value))
        for name in _INT_SETTINGS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Settings.{name} must be a non-negative integer; got {value!r}.")
        for name in ("alpha", "power", "ladder_power", "fitness_alpha"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"Settings.{name} must be in (0, 1); got {getattr(self, name)}.")
        for name in ("round_candidates", "ladder_points", "behavior_ladder_items", "behavior_items", "finalists"):
            if getattr(self, name) < 1:
                raise ValueError(f"Settings.{name} must be at least 1; got {getattr(self, name)}.")


@dataclass(frozen=True)
class Candidate:
    """A structure and the values of its proposer axes.

    Attributes:
        structure: A key of `Study.space`.
        values: One value per proposer axis, keyed as `Template.bind` accepts (a member path, or the key of a shared
            axis with a position index or a fraction).
    """

    structure: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise RecipeError(f"Candidate values must be a mapping; got {type(self.values).__name__}.")
        object.__setattr__(self, "values", dict(self.values))


@dataclass(frozen=True)
class Study:
    """A search over steering pipelines under accuracy constraints, with its items, settings, and identity.

    Each template of `space` has at most one solved axis, i.e., an ordered axis whose key is not in `searched`. The
    evaluator finds, for each candidate, the position of the solved axis with the best behavior among those inside
    the margin. A solved axis must not change a fit (every position has the same fit digests). An ordered argument that
    is an input of a fit therefore belongs in `searched`. Every other axis (unordered choices, text, and the axes in
    `searched`) is a proposer axis, which a candidate binds.

    `monotone` gives the solved axes along which capability loss is assumed not to fall as the position rises. It
    says nothing about behavior. The study's id covers every field, with benchmarks given by their ids and the
    protocol and templates by theirs.

    Attributes:
        name: The study's name.
        protocol: The measurement protocol.
        space: Mapping from structure name to template.
        objective: The behavior benchmark; its per-item scores are in [0, 1], and higher is better.
        constraints: The capability constraints.
        pools: Mapping from benchmark id to its item split, for the objective and every constraint.
        seed: The base seed of every draw (pools, batches, proposers).
        settings: The study's settings.
        searched: The keys of ordered axes that the proposer searches.
        monotone: The keys of solved axes along which capability loss is assumed not to fall.
        fit_item_ids: Mapping from benchmark id to the items used as fit data, which must be in neither pool.
    """

    name: str
    protocol: Protocol
    space: Mapping[str, Template]
    objective: Benchmark
    constraints: tuple[Constraint, ...]
    pools: Mapping[str, PoolSplit]
    seed: int = 0
    settings: Settings = Settings()
    searched: frozenset[str] = frozenset()
    monotone: frozenset[str] = frozenset()
    fit_item_ids: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, Protocol):
            raise TypeError(f"protocol must be a Protocol; got {type(self.protocol).__name__}.")
        space = {}
        for structure, source in dict(self.space).items():
            if isinstance(source, Recipe):
                source = Template(source.entries)
            if not isinstance(source, Template):
                raise TypeError(f"space[{structure!r}] must be a Template; got {type(source).__name__}.")
            space[str(structure)] = source
        if not space:
            raise ValueError("A study needs at least one structure in its space.")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "constraints", tuple(self.constraints))
        for constraint in self.constraints:
            if not isinstance(constraint, Constraint):
                raise TypeError(f"constraints must be Constraint objects; got {type(constraint).__name__}.")
        object.__setattr__(self, "pools", dict(self.pools))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "searched", frozenset(self.searched))
        object.__setattr__(self, "monotone", frozenset(self.monotone))
        object.__setattr__(self, "fit_item_ids", {
            str(key): frozenset(str(item) for item in items) for key, items in dict(self.fit_item_ids).items()
        })
        object.__setattr__(self, "_definition", None)
        object.__setattr__(self, "_id", None)
        object.__setattr__(self, "_classified", {})

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def benchmarks(self) -> tuple[Benchmark, ...]:
        """The objective followed by the constraint benchmarks, in order."""
        return (self.objective, *(constraint.benchmark for constraint in self.constraints))

    def definition(self) -> dict:
        """The JSON definition of the study, from which its id is computed.

        Returns:
            A mapping with the name, the protocol id, the template id of each structure, the objective and the
            constraints (benchmark ids, names, and margins), the pools by benchmark id, the seed, the settings, the
            searched and monotone axis keys, and the fit items.
        """
        if self._definition is None:
            definition = {
                "schema": STUDY_SCHEMA,
                "name": self.name,
                "protocol_id": self.protocol.id,
                "space": {structure: template.id for structure, template in self.space.items()},
                "objective": {"id": self.objective.id, "name": self.objective.name},
                "constraints": [
                    {"id": constraint.benchmark.id, "name": constraint.benchmark.name, "margin": constraint.margin}
                    for constraint in self.constraints
                ],
                "pools": {
                    benchmark_id: {"search": list(split.search), "held_out": list(split.held_out)}
                    for benchmark_id, split in self.pools.items()
                },
                "seed": self.seed,
                "settings": dataclasses.asdict(self.settings),
                "searched": sorted(self.searched),
                "monotone": sorted(self.monotone),
                "fit_item_ids": {key: sorted(items) for key, items in self.fit_item_ids.items()},
            }
            object.__setattr__(self, "_definition", definition)
        return json.loads(dumps(self._definition))

    @property
    def id(self) -> str:
        """The content id of the study's definition."""
        if self._id is None:
            object.__setattr__(self, "_id", content_id(self.definition()))
        return self._id

    def constraint_for(self, benchmark: Benchmark) -> Constraint | None:
        """The constraint on a benchmark (by id), or None for the objective."""
        for constraint in self.constraints:
            if constraint.benchmark.id == benchmark.id:
                return constraint
        return None

    def classify(self, structure: str) -> tuple[Axis | None, dict[str, Axis]]:
        """The solved axis and the proposer axes of a structure's template.

        Args:
            structure: A key of `space`.

        Returns:
            The solved axis (or None) and the proposer axes by key, in the template's axis order.

        Raises:
            RecipeError: If the structure is unknown, or the template leaves more than one ordered axis out of
                `searched`.
        """
        if structure not in self._classified:
            template = self.space.get(structure)
            if template is None:
                raise RecipeError(f"The study has no structure {structure!r}; its structures are {list(self.space)}.")
            axes = template.axes()
            free = [axis for axis in axes.values() if axis.ordered and axis.key not in self.searched]
            if len(free) > 1:
                raise RecipeError(
                    f"Structure {structure!r} leaves {len(free)} ordered axes free ({[axis.key for axis in free]}); "
                    "a template has at most one solved axis (list all but one in study.searched)."
                )
            solved = free[0] if free else None
            proposer = {key: axis for key, axis in axes.items() if solved is None or key != solved.key}
            self._classified[structure] = (solved, proposer)
        solved, proposer = self._classified[structure]
        return solved, dict(proposer)

    def solved_axis(self, structure: str) -> Axis | None:
        """The solved axis of a structure, or None."""
        return self.classify(structure)[0]

    def proposer_axes(self, structure: str) -> dict[str, Axis]:
        """The proposer axes of a structure, by key."""
        return self.classify(structure)[1]

    def bind(self, candidate: Candidate) -> Template:
        """The template of a candidate: its structure's template with every proposer axis bound.

        Only the solved axis, if any, is left free.

        Args:
            candidate: The candidate.

        Returns:
            The bound template.

        Raises:
            RecipeError: If the structure is unknown, a value is outside its range, a proposer axis is missing, or a
                key gives anything other than a proposer axis or one of its members.
        """
        if not isinstance(candidate, Candidate):
            raise RecipeError(f"Expected a Candidate; got {type(candidate).__name__}.")
        solved, proposer = self.classify(candidate.structure)
        allowed: dict[str, str] = {}
        for key, axis in proposer.items():
            allowed[key] = key
            for path in axis.members:
                allowed[path] = key
        unknown = sorted(key for key in candidate.values if key not in allowed)
        if unknown:
            solved_note = f" (the solved axis {solved.key!r} is left free)" if solved is not None else ""
            raise RecipeError(
                f"Candidate keys {unknown} give no proposer axis of structure {candidate.structure!r}{solved_note}; "
                f"its proposer axes are {list(proposer)}."
            )
        missing = [
            key for key, axis in proposer.items()
            if key not in candidate.values and not any(path in candidate.values for path in axis.members)
        ]
        if missing:
            raise RecipeError(f"Candidate for structure {candidate.structure!r} does not bind proposer axes {missing}.")
        return self.space[candidate.structure].bind(candidate.values)

    def candidate_id(self, candidate: Candidate) -> str:
        """The id of a candidate, `content_id([structure, bound template id])`.

        Two candidates that bind to the same template have the same id.

        Raises:
            RecipeError: If the candidate does not bind.
        """
        return content_id([candidate.structure, self.bind(candidate).id])

    def seed_candidate(self, structure: str) -> Candidate:
        """The fixed starting candidate of a structure.

        Every proposer axis is at its first position if it is discrete, at its midpoint if it is continuous, and an
        empty string for a `Text` range.

        Raises:
            RecipeError: If the structure is unknown or its axes are invalid.
        """
        values: dict[str, Any] = {}
        for axis in self.proposer_axes(structure).values():
            if axis.has_text:
                values.update({path: "" for path in axis.members})
            elif axis.continuous:
                values.update(axis.at(0.5))
            else:
                values.update(axis.at_index(0))
        return Candidate(structure, values)

    def validate(self, store: Store) -> None:
        """Check the study before any model work.

        The checks, in order, are:

        1. The protocol has its `versions` (`Protocol.create` fills them). Callables of the toolkit are versioned by
           the toolkit version in the protocol id, which is part of the id only when `versions` is filled.
        2. For every benchmark, the generation arguments merged from `protocol.generate_defaults` and the benchmark's
           `generate` set `temperature`, since the decoding mode decides which baseline rules apply.
        3. No template has an entry in the structural category.
        4. Every callable reachable from a template or a benchmark is versioned (`versioned`).
        5. The pools pass `check_pools` (the benchmarks are prepared on `store` first).
        6. Each template has at most one solved axis, a discrete solved axis has at most `2 * settings.ladder_points`
           positions, every key in `searched` and `monotone` gives an axis of some template, and every key in
           `monotone` gives a solved axis.
        7. The objective and the constraints have distinct benchmark names.
        8. `make_plan` succeeds for the seed candidate of every structure.

        A warning is logged for a template with a solved axis that contains a control whose fit has no identity,
        since such a control refits at every position. Whether the pools are large enough is checked by
        `StudySession.start` (`evidence.feasibility`), before and after the baseline is measured.

        Args:
            store: The store the benchmarks are prepared on.

        Raises:
            ValueError: If a check fails (`RecipeError` for a template that cannot be bound or planned).
        """
        from steerability.optimization.evaluation.plan import make_plan

        if not self.protocol.versions:
            raise ValueError(
                "The study's protocol has no versions; build it with Protocol.create, which fills them. Callables of "
                "the toolkit are versioned by the toolkit version in the protocol."
            )
        unset = [benchmark.name for benchmark in self.benchmarks
                 if "temperature" not in {**self.protocol.generate_defaults,
                                          **(getattr(benchmark, "generate", None) or {})}]
        if unset:
            raise ValueError(
                f"Benchmarks {unset} have no temperature in protocol.generate_defaults or their own generate "
                "arguments; set it (0 for greedy decoding), since the decoding mode decides which baseline rules apply."
            )

        for structure, template in self.space.items():
            structural = [item.method for item in template.entries
                          if item.enabled and item.category == "structural_control"]
            if structural:
                raise ValueError(
                    f"Structure {structure!r} has structural entries {structural}; a study does not search weight "
                    "changes."
                )

        unversioned: list[str] = []
        for structure, template in self.space.items():
            unversioned.extend(f"{structure}: {target}" for target in _unversioned_refs(template.canonical()))
        for benchmark in self.benchmarks:
            unversioned.extend(f"{benchmark.name}: {target}" for target in _unversioned_callables(benchmark))
        if unversioned:
            raise ValueError(
                f"Callables without a version: {unversioned}; decorate each with @versioned('<version>') from "
                "steerability.algorithms.core.identity; its version enters the identity of what it scores."
            )

        for benchmark in self.benchmarks:
            benchmark.prepare(store)
        check_pools(self)

        all_keys: set[str] = set()
        solved_keys: set[str] = set()
        for structure, template in self.space.items():
            all_keys.update(template.axes())
            solved, _ = self.classify(structure)
            if solved is None:
                continue
            solved_keys.add(solved.key)
            if not solved.continuous and solved.count is not None and solved.count > 2 * self.settings.ladder_points:
                raise ValueError(
                    f"Structure {structure!r}: the solved axis {solved.key!r} has {solved.count} positions, more than "
                    f"2 * ladder_points ({2 * self.settings.ladder_points}); list it in study.searched or use fewer "
                    "positions."
                )
        unknown_searched = sorted(self.searched - all_keys)
        if unknown_searched:
            raise ValueError(f"study.searched gives keys {unknown_searched} that are axes of no template.")
        not_solved = sorted(self.monotone - solved_keys)
        if not_solved:
            raise ValueError(f"study.monotone gives keys {not_solved} that are not solved axes of any template.")

        names = [benchmark.name for benchmark in self.benchmarks]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Benchmark names {repeated} repeat; the objective and constraints need distinct names.")

        for structure in self.space:
            plan = make_plan(self, self.seed_candidate(structure))
            if plan.solved is None:
                continue
            for control in plan.recipes[0].instantiate():
                if control.steer_fits() and control.fit_identity() is None:
                    logger.warning(
                        "Structure %r has a solved axis and a %s control whose fit has no identity; it refits at "
                        "every position of the axis.", structure, type(control).__name__,
                    )


def record_study(store: Store, study: Study) -> None:
    """Record a study's definition in the `studies` table (once)."""
    ensure_tables(store)
    store.execute(
        "INSERT OR IGNORE INTO studies (study_id, name, definition, created_at) VALUES (?, ?, ?, ?)",
        (study.id, study.name, dumps(study.definition()), now()),
    )


def load_definition(store: Store, study_id: str) -> dict | None:
    """The recorded definition of a study, or None."""
    ensure_tables(store)
    row = store.execute("SELECT definition FROM studies WHERE study_id = ?", (study_id,)).fetchone()
    return None if row is None else json.loads(row[0])


def _toolkit_target(target: str) -> bool:
    module = target.partition(":")[0]
    return module == "steerability" or module.startswith("steerability.")


def _unversioned_refs(encoded: Any) -> list[str]:
    """The targets of the `$ref` callables without a version in an encoded form, outside the toolkit."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            target = value.get("$ref")
            if isinstance(target, str) and "version" not in value and not _toolkit_target(target):
                found.append(target)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(encoded)
    return sorted(set(found))


def _unversioned_callables(benchmark: Any) -> list[str]:
    """The callables without a version in a benchmark's public attributes, outside the toolkit."""
    found: list[str] = []
    seen: set[int] = set()

    def walk(value: Any, path: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (str, bytes, int, float, bool)) or value is None:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}[{key!r}]")
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if callable(value) and not isinstance(value, type):
            module = getattr(value, "__module__", None) or ""
            if module == "steerability" or module.startswith("steerability."):
                return
            if not isinstance(getattr(value, "__steerability_version__", None), str):
                found.append(f"{path} ({getattr(value, '__qualname__', type(value).__name__)})")

    for name, value in vars(benchmark).items():
        if not name.startswith("_"):
            walk(value, name)
    return found


__all__ = [
    "Candidate",
    "Constraint",
    "OPTIMIZATION_SCHEMA",
    "Settings",
    "Study",
    "StudyError",
    "StudyInfeasibleError",
    "ensure_tables",
    "load_definition",
    "record_study",
]
