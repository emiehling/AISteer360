"""The stages of evaluating a candidate and the confirmation of a point.

`StudySession.screen` runs stages 0 to 3 and `StudySession.finish` runs stages 4 and 5. Each stage and the reasons
it rejects with are:

- **0. Static**: `make_plan`, and `preflight` on the plan's first recipe (`invalid`, `unsupported`).
- **1. Fits**: `factory.prepare(plan.recipes)` (`fit_failed`).
- **2. Behavior quick look**: probe recipes on the start of the objective's ladder block (`no_behavior_gain`).
- **3. Capability smoke test**: with a solved axis, probe recipes on the start of each constraint's ladder block
  (`capability_fail`).
- **4. Operating point**: calibration with a solved axis, else confirmation of the single recipe
  (`no_feasible_point`, `not_confirmed`, `over_cost`, `degenerate`).
- **5. Report**: the `Result`, recorded in the archive.

Confirmation (`confirm`) is the one routine that judges a point, and every candidate ends in it. A rejection is
recorded with its stage, reason, and summary, except for a candidate that does not bind (it has no id) and a candidate
stopped by a limit (`budget`) or a deadline (`timeout`), whose evidence rows stay so that a later evaluation resumes
from them.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import Point, Recipe, RecipeError
from steerability.algorithms.core.sweeps import preflight
from steerability.optimization.evaluation.plan import EvaluationPlan, make_plan
from steerability.optimization.evaluation.result import OperatingPoint, Result, summarize
from steerability.optimization.measurement.evidence import cost_ratio, degenerate_increase, paired, point_items
from steerability.optimization.measurement.stats import Paired, clearly_over, confirm_step, feasible
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.evaluation.items import Benchmark
    from steerability.optimization.evaluation.session import StudySession

logger = logging.getLogger(__name__)


class Stopped(Exception):
    """A limit or a deadline stopped an evaluation; nothing is recorded.

    Args:
        reason: `"budget"` or `"timeout"`.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Rejection(Exception):
    """A stage decided to reject the candidate; the rejection is recorded.

    Args:
        stage: The stage.
        reason: The reason.
        summary: What was measured.
        message: A description of the cause, or None.
        ladder: The ladder diagnostics of the evaluated positions.
    """

    def __init__(self, stage: int, reason: str, summary: Mapping[str, float | str] | None = None,
                 message: str | None = None, ladder: tuple = ()) -> None:
        super().__init__(message or reason)
        self.stage = stage
        self.reason = reason
        self.summary = dict(summary or {})
        self.message = message
        self.ladder = ladder


class SteeringFailure(Exception):
    """Steering a recipe failed while it was being measured."""


@dataclass
class Partial:
    """An evaluation between `screen` and `finish`.

    Attributes:
        candidate: The candidate.
        candidate_id: Its id, or None when it does not bind.
        plan: Its plan, or None when stage 0 did not produce one.
        round_index: The round whose batches this evaluation uses.
        result: The result, set when stages 0 to 3 already decided (or a recorded result was found).
        proposer: The name of the proposer.
        generations: The generations run so far by this evaluation.
        seconds: The wall time spent so far by this evaluation.
        error: A description of why the candidate was rejected as invalid or unsupported, or None.
    """

    candidate: Candidate
    candidate_id: str | None
    plan: EvaluationPlan | None
    round_index: int
    result: Result | None = None
    proposer: str = ""
    generations: int = 0
    seconds: float = 0.0
    error: str | None = None


@dataclass
class Confirmation:
    """The outcome of judging one point on its fresh-block tables.

    Attributes:
        eligible: Whether the point is eligible.
        reason: The rejection reason, or None.
        resolved_id: The point's resolved id.
        capability: Mapping from constraint benchmark name to the point's table (as far as measured).
        behavior: The behavior table, or None when confirmation stopped before the objective.
        cost_ratio: The cost ratio, or None.
        degenerate_increase: The degenerate increase, or None.
    """

    eligible: bool
    reason: str | None
    resolved_id: str
    capability: dict[str, Paired] = field(default_factory=dict)
    behavior: Paired | None = None
    cost_ratio: float | None = None
    degenerate_increase: float | None = None

    def point(self, position: Mapping[str, Any], recipe_id: str) -> OperatingPoint:
        """The operating point of an eligible confirmation."""
        return OperatingPoint(
            position=dict(position), recipe_id=recipe_id, resolved_id=self.resolved_id,
            capability=dict(self.capability), behavior=self.behavior, cost_ratio=float(self.cost_ratio),
            degenerate_increase=float(self.degenerate_increase), eligible=self.eligible,
        )


def margins(session: StudySession) -> dict[str, float]:
    """The margin of every constraint, by benchmark name."""
    return {constraint.benchmark.name: constraint.margin for constraint in session.study.constraints}


def measure(session: StudySession, partial: Partial, recipe: Recipe, benchmark: Benchmark,
            item_ids: Sequence[str], deadline: float | None) -> str:
    """Measure a recipe on items of a benchmark, charging the generations to the session and the evaluation.

    Returns:
        The recipe's resolved id.

    Raises:
        Stopped: If a limit or the deadline has passed.
        SteeringFailure: If the recipe fails to steer.
    """
    resolved, generated = session.measure(recipe, benchmark, item_ids, deadline)
    partial.generations += generated
    return resolved


def probe_indices(size: int) -> list[int]:
    """The first, middle, and last of `size` ladder positions, or all of them when there are fewer than three."""
    if size < 3:
        return list(range(size))
    return sorted({0, size // 2, size - 1})


def _rejected(session: StudySession, partial: Partial, *, stage: int, reason: str,
              summary: Mapping[str, float | str] | None = None, ladder: tuple = ()) -> Result:
    return Result(
        candidate_id=partial.candidate_id or "", study_id=session.study.id, version=1,
        round_index=partial.round_index, status="rejected", stage=stage, reason=reason, fitness=0.0, operating=None,
        ladder=ladder, summary=dict(summary or {}), generations=partial.generations, seconds=partial.seconds,
    )


def screen(session: StudySession, candidate: Candidate, *, proposer: str, deadline: float | None) -> Partial:
    """Run stages 0 to 3 of a candidate (see `StudySession.screen`)."""
    started = time.monotonic()
    partial = Partial(candidate, None, None, session.round_index, proposer=proposer)
    try:
        partial.candidate_id = session.study.candidate_id(candidate)
    except RecipeError as error:
        partial.error = str(error)
        partial.result = _rejected(session, partial, stage=0, reason="invalid")
        return partial
    session.archive.remember(partial.candidate_id, candidate)
    latest = session.archive.latest(partial.candidate_id)
    if latest is not None:
        partial.result = latest
        return partial
    if session.exhausted():
        partial.result = _rejected(session, partial, stage=0, reason="budget")
        return partial
    progress = _Progress()
    try:
        _screen_stages(session, partial, progress, deadline)
    except Stopped as stop:
        partial.seconds += time.monotonic() - started
        partial.result = _rejected(session, partial, stage=progress.stage, reason=stop.reason)
        return partial
    except SteeringFailure as failure:
        partial.error = str(failure)
        partial.seconds += time.monotonic() - started
        result = _rejected(session, partial, stage=progress.stage, reason="fit_failed")
        partial.result = session.record(candidate, result, proposer=proposer)
        return partial
    except Rejection as rejection:
        partial.error = rejection.message
        partial.seconds += time.monotonic() - started
        result = _rejected(session, partial, stage=rejection.stage, reason=rejection.reason, summary=rejection.summary)
        partial.result = session.record(candidate, result, proposer=proposer)
        return partial
    partial.seconds += time.monotonic() - started
    return partial


@dataclass
class _Progress:
    stage: int = 0


def _screen_stages(session: StudySession, partial: Partial, progress: _Progress, deadline: float | None) -> None:
    """Stages 0 to 3, raising `Rejection` when one rejects; `progress.stage` is the stage being run."""
    study, settings = session.study, session.study.settings
    try:
        plan = make_plan(study, partial.candidate)
    except RecipeError as error:
        raise Rejection(0, "invalid", message=str(error)) from error
    partial.plan = plan
    messages = preflight([(partial.candidate.structure, Point(plan.recipes[0], {}))], study.protocol)
    if messages:
        raise Rejection(0, "unsupported", message="; ".join(messages))

    progress.stage = 1
    session.check(deadline)
    try:
        session.fit(plan.recipes)
    except Exception as error:
        logger.warning("Fitting candidate %s failed.", partial.candidate_id, exc_info=True)
        raise Rejection(1, "fit_failed", message=f"{type(error).__name__}: {error}") from error

    progress.stage = 2
    probes = probe_indices(len(plan.recipes))
    batches = session.batches(partial.round_index)
    objective = study.objective
    items = batches[objective.id].ladder[:settings.quick_behavior_items]
    gains: dict[int, Paired] = {}
    for index in probes:
        resolved = measure(session, partial, plan.recipes[index], objective, items, deadline)
        gains[index] = paired(study, session.store, resolved, objective, items)
        if gains[index].upper(settings.alpha) >= settings.min_gain:
            break
    else:
        best = max(probes, key=lambda index: gains[index].mean)
        raise Rejection(2, "no_behavior_gain", summarize(
            behavior=gains[best], margins=margins(session), fitness_alpha=settings.fitness_alpha,
            position=plan.ladder[best],
        ))

    progress.stage = 3
    if plan.solved is None or not study.constraints:
        return
    smoke = [0] if plan.monotone else probes
    tables: dict[int, dict[str, Paired]] = {}
    for index in smoke:
        tables[index] = {}
        over = False
        for constraint in study.constraints:
            benchmark = constraint.benchmark
            items = batches[benchmark.id].ladder[:session.looks(benchmark)[0]]
            resolved = measure(session, partial, plan.recipes[index], benchmark, items, deadline)
            table = paired(study, session.store, resolved, benchmark, items)
            tables[index][benchmark.name] = table
            if clearly_over(table, constraint.margin, settings.alpha):
                over = True
                break
        if not over:
            return
    best = min(smoke, key=lambda index: max(-table.mean for table in tables[index].values()))
    raise Rejection(3, "capability_fail", summarize(
        capability=tables[best], margins=margins(session), fitness_alpha=settings.fitness_alpha,
        position=plan.ladder[best],
    ))


def judge(session: StudySession, resolved_id: str, calibration_round: int) -> Confirmation:
    """Judge a point on its pooled fresh-block tables without measuring anything.

    The point is eligible when every constraint's table is `feasible` (at `settings.power`) and the cost and
    degeneracy limits hold.

    Args:
        session: The session.
        resolved_id: The point's resolved id.
        calibration_round: The round its candidate was calibrated in.

    Returns:
        The confirmation.
    """
    study, store, settings = session.study, session.store, session.study.settings
    rounds = session.rounds
    outcome = Confirmation(False, None, resolved_id)
    items_by_benchmark: dict[str, tuple[str, ...]] = {}
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        items = point_items(study, store, resolved_id, benchmark, rounds, calibration_round)
        items_by_benchmark[benchmark.id] = items
        table = paired(study, store, resolved_id, benchmark, items)
        outcome.capability[benchmark.name] = table
        if not feasible(table, constraint.margin, n_sealed=session.n_sealed(benchmark), alpha=settings.alpha,
                        power=settings.power, benchmarks=len(study.constraints), guard=settings.guard_band):
            outcome.reason = "not_confirmed"
            return outcome
    return _behavior_and_limits(session, outcome, items_by_benchmark, calibration_round)


def _behavior_and_limits(session: StudySession, outcome: Confirmation,
                         items_by_benchmark: dict[str, tuple[str, ...]], calibration_round: int) -> Confirmation:
    study, store, settings = session.study, session.store, session.study.settings
    objective = study.objective
    behavior_items = point_items(study, store, outcome.resolved_id, objective, session.rounds, calibration_round)
    outcome.behavior = paired(study, store, outcome.resolved_id, objective, behavior_items)
    outcome.cost_ratio = cost_ratio(study, store, outcome.resolved_id,
                                    {**items_by_benchmark, objective.id: behavior_items})
    outcome.degenerate_increase = degenerate_increase(study, store, outcome.resolved_id, behavior_items)
    if outcome.cost_ratio > settings.max_cost_ratio:
        outcome.reason = "over_cost"
    elif outcome.degenerate_increase > settings.max_degenerate_increase:
        outcome.reason = "degenerate"
    else:
        outcome.eligible = True
    return outcome


def confirm_constraints(
    session: StudySession, partial: Partial, recipe: Recipe, *, calibration_round: int, fresh_round: int,
    deadline: float | None,
) -> tuple[bool, str, dict[str, Paired], dict[str, tuple[str, ...]]]:
    """Run the sequential looks of a confirmation on each constraint's fresh block.

    For each constraint the recipe is measured on the fresh block of `fresh_round` up to each size of
    `session.looks(benchmark)` in turn (a look larger than the block uses the whole block), and after each look
    `confirm_step` judges the point's table (built from fresh-block items only). `"accept"` settles the constraint,
    and `"reject"` (or `"continue"` after the last look) stops at that constraint.

    Args:
        session: The session.
        partial: The evaluation the generations are charged to.
        recipe: The point's recipe.
        calibration_round: The round the candidate was calibrated in, whose ladder blocks are excluded.
        fresh_round: The round whose fresh blocks are measured.
        deadline: A `time.monotonic()` deadline, or None.

    Returns:
        Whether every constraint accepted, the point's resolved id, its capability tables by constraint benchmark
        name (as far as measured), and the items of each table by benchmark id.

    Raises:
        Stopped: If a limit or the deadline has passed.
        SteeringFailure: If the recipe fails to steer.
    """
    study, store, settings = session.study, session.store, session.study.settings
    rounds = session.rounds
    items_by_benchmark: dict[str, tuple[str, ...]] = {}
    capability: dict[str, Paired] = {}
    resolved = ""
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        fresh = rounds[fresh_round][benchmark.id].fresh
        decision = "continue"
        table = Paired(0, 0.0, 0.0, 0)
        for look, size in enumerate(session.looks(benchmark)):
            resolved = measure(session, partial, recipe, benchmark, fresh[:size], deadline)
            items = point_items(study, store, resolved, benchmark, rounds, calibration_round)
            items_by_benchmark[benchmark.id] = items
            table = paired(study, store, resolved, benchmark, items)
            decision = confirm_step(
                table, look, constraint.margin, n_sealed=session.n_sealed(benchmark), alpha=settings.alpha,
                power=settings.power, benchmarks=len(study.constraints), guard=settings.guard_band,
            )
            if decision != "continue":
                break
        capability[benchmark.name] = table
        if decision != "accept":
            return False, resolved, capability, items_by_benchmark
    return True, resolved, capability, items_by_benchmark


def confirm(session: StudySession, partial: Partial, recipe: Recipe, *, calibration_round: int, fresh_round: int,
            deadline: float | None) -> Confirmation:
    """Confirm a point: sequential looks on each constraint's fresh block, then the objective's fresh block.

    The looks on the constraints are those of `confirm_constraints`, and a point that a constraint does not accept
    fails with `not_confirmed`. The recipe is then measured on the objective's fresh block, and a point over the cost
    or degeneracy limit fails with `over_cost` or `degenerate`.

    Args:
        session: The session.
        partial: The evaluation the generations are charged to.
        recipe: The point's recipe.
        calibration_round: The round the candidate was calibrated in, whose ladder blocks are excluded.
        fresh_round: The round whose fresh blocks are measured.
        deadline: A `time.monotonic()` deadline, or None.

    Returns:
        The confirmation.

    Raises:
        Stopped: If a limit or the deadline has passed.
        SteeringFailure: If the recipe fails to steer.
    """
    accepted, resolved, capability, items_by_benchmark = confirm_constraints(
        session, partial, recipe, calibration_round=calibration_round, fresh_round=fresh_round, deadline=deadline,
    )
    if not accepted:
        return Confirmation(False, "not_confirmed", resolved, capability)
    objective = session.study.objective
    resolved = measure(session, partial, recipe, objective, session.rounds[fresh_round][objective.id].fresh, deadline)
    outcome = Confirmation(False, None, resolved, capability)
    return _behavior_and_limits(session, outcome, items_by_benchmark, calibration_round)


def summary_of(session: StudySession, outcome: Confirmation, position: Mapping[str, Any]) -> dict[str, float | str]:
    """The summary of a confirmation's tables and limits."""
    return summarize(
        capability=outcome.capability, behavior=outcome.behavior, margins=margins(session),
        fitness_alpha=session.study.settings.fitness_alpha, cost_ratio=outcome.cost_ratio,
        degenerate_increase=outcome.degenerate_increase, position=position,
    )


def accepted_result(session: StudySession, partial: Partial, point: OperatingPoint, *, version: int,
                    round_index: int, ladder: tuple = ()) -> Result:
    """The accepted result of an eligible operating point."""
    settings = session.study.settings
    fitness = max(0.0, point.behavior.lower(settings.fitness_alpha))
    summary = summarize(
        capability=point.capability, behavior=point.behavior, margins=margins(session),
        fitness_alpha=settings.fitness_alpha, cost_ratio=point.cost_ratio,
        degenerate_increase=point.degenerate_increase, position=point.position,
    )
    return Result(
        candidate_id=partial.candidate_id or "", study_id=session.study.id, version=version, round_index=round_index,
        status="accepted", stage=5, reason=None, fitness=fitness, operating=point, ladder=ladder, summary=summary,
        generations=partial.generations, seconds=partial.seconds,
    )


def finish(session: StudySession, partial: Partial, *, deadline: float | None) -> Result:
    """Run stages 4 and 5 of a screened candidate (see `StudySession.finish`)."""
    from steerability.optimization.evaluation.calibration import calibrate

    if partial.result is not None:
        return partial.result
    recorded = session.archive.latest(partial.candidate_id)
    if recorded is not None:
        partial.result = recorded
        return recorded
    started = time.monotonic()
    plan = partial.plan
    try:
        if plan.solved is None:
            outcome = confirm(session, partial, plan.recipes[0], calibration_round=partial.round_index,
                              fresh_round=partial.round_index, deadline=deadline)
            if not outcome.eligible:
                raise Rejection(4, outcome.reason, summary_of(session, outcome, {}))
            point, ladder = outcome.point({}, plan.recipes[0].id), ()
        else:
            point, ladder = calibrate(session, partial, deadline=deadline)
    except Stopped as stop:
        partial.seconds += time.monotonic() - started
        partial.result = _rejected(session, partial, stage=4, reason=stop.reason)
        return partial.result
    except SteeringFailure as failure:
        partial.error = str(failure)
        partial.seconds += time.monotonic() - started
        result = _rejected(session, partial, stage=4, reason="fit_failed")
        partial.result = session.record(partial.candidate, result, proposer=partial.proposer)
        return partial.result
    except Rejection as rejection:
        partial.seconds += time.monotonic() - started
        result = _rejected(session, partial, stage=rejection.stage, reason=rejection.reason,
                           summary=rejection.summary, ladder=rejection.ladder)
        partial.result = session.record(partial.candidate, result, proposer=partial.proposer)
        return partial.result
    partial.seconds += time.monotonic() - started
    result = accepted_result(session, partial, point, version=1, round_index=partial.round_index, ladder=ladder)
    partial.result = session.record(partial.candidate, result, proposer=partial.proposer)
    return partial.result


def recipe_at(plan: EvaluationPlan, position: Mapping[str, Any]) -> Recipe:
    """The recipe of a plan at a position of its solved axis (the plan's one recipe without a solved axis)."""
    if plan.solved is None or not position:
        return plan.recipes[0]
    return plan.template.bind(dict(position)).recipe()


def rescore_result(session: StudySession, result: Result) -> Result | None:
    """A new version of an accepted result, measured on the session's current round.

    The operating recipe is measured on the current round's fresh blocks and its tables are rebuilt. A point that is
    still eligible gets a new accepted version. Otherwise, for a candidate with a solved axis, the next positions in
    the recorded ranking are confirmed on the current round's fresh blocks in turn. A candidate with no position left
    gets a rejected version.

    Args:
        session: The session.
        result: The latest version of an accepted result.

    Returns:
        The new version, or None when the candidate cannot be rebuilt in this process.

    Raises:
        Stopped: If a limit has passed.
        SteeringFailure: If a recipe fails to steer.
    """
    study = session.study
    candidate = session.archive.candidate(result.candidate_id)
    if candidate is None or result.operating is None:
        return None
    try:
        plan = make_plan(study, candidate)
        recipe = recipe_at(plan, result.operating.position)
    except RecipeError:
        logger.warning("Candidate %s no longer binds; it is not re-scored.", result.candidate_id, exc_info=True)
        return None
    if recipe.id != result.operating.recipe_id:
        logger.warning("Candidate %s rebuilds to recipe %s, not %s; it is not re-scored.", result.candidate_id,
                       recipe.id, result.operating.recipe_id)
        return None
    started = time.monotonic()
    partial = Partial(candidate, result.candidate_id, plan, result.round_index, proposer="rescore")
    current = session.round_index
    resolved = result.operating.resolved_id
    for benchmark in (*(constraint.benchmark for constraint in study.constraints), study.objective):
        resolved = measure(session, partial, recipe, benchmark, session.rounds[current][benchmark.id].fresh, None)
    outcome = judge(session, resolved, result.round_index)
    version = result.version + 1
    if outcome.eligible:
        partial.seconds = time.monotonic() - started
        point = outcome.point(result.operating.position, recipe.id)
        return accepted_result(session, partial, point, version=version, round_index=result.round_index,
                               ladder=result.ladder)
    reason = outcome.reason
    summary = summary_of(session, outcome, result.operating.position)
    if plan.solved is not None:
        ranked = sorted((point for point in result.ladder if point.rank is not None), key=lambda point: point.rank)
        current_rank = next((point.rank for point in ranked if point.recipe_id == recipe.id), None)
        following = [point for point in ranked if current_rank is None or point.rank > current_rank]
        for ladder_point in following:
            candidate_recipe = recipe_at(plan, ladder_point.position)
            confirmation = confirm(session, partial, candidate_recipe, calibration_round=result.round_index,
                                   fresh_round=current, deadline=None)
            if confirmation.eligible:
                partial.seconds = time.monotonic() - started
                point = confirmation.point(ladder_point.position, candidate_recipe.id)
                return accepted_result(session, partial, point, version=version, round_index=result.round_index,
                                       ladder=result.ladder)
            summary = summary_of(session, confirmation, ladder_point.position)
        reason = "no_feasible_point"
    partial.seconds = time.monotonic() - started
    return Result(
        candidate_id=result.candidate_id, study_id=study.id, version=version, round_index=result.round_index,
        status="rejected", stage=4, reason=reason, fitness=0.0, operating=None, ladder=result.ladder,
        summary=summary, generations=partial.generations, seconds=partial.seconds,
    )


__all__ = [
    "Confirmation",
    "Partial",
    "Rejection",
    "SteeringFailure",
    "Stopped",
    "accepted_result",
    "confirm",
    "confirm_constraints",
    "finish",
    "judge",
    "measure",
    "probe_indices",
    "recipe_at",
    "rescore_result",
    "screen",
    "summary_of",
]
