"""Calibration of a candidate's solved axis: choose a position, then confirm it on fresh blocks.

A solved axis in `study.monotone` (whose capability loss is assumed not to fall as the position rises) is calibrated
in these steps:

1. **First look**: from the weakest position up, each position is measured on the first look of every constraint's
   ladder block (the items stage 3 read for the weakest position). The first position that is clearly over a margin
   and every stronger one are pruned, and the stronger ones are not measured.
2. **Behavior ladder**: every position that is not pruned is measured on the objective's ladder block. A position
   over the cost or degeneracy limit on its ladder blocks is not eligible.
3. **Search range**: the eligible positions up to the one with the largest lower bound of its ladder-block gain at
   `settings.fitness_alpha` (the weaker position on a tie).
4. **Bisection**: the range is searched by bisection, and each probe runs the sequential looks of a confirmation on
   every constraint's fresh block (`confirm_constraints`). The highest position that every constraint accepts is the
   highest accepted one.
5. **Refine**: on a continuous axis, while `settings.ladder_refinements` remain, the position halfway in fraction
   between the highest accepted one and the next position is measured as in steps 1 and 2 and probed. An accepted
   position becomes the highest accepted one, and any other becomes the next position.
6. **Confirm**: the eligible positions at or below the highest accepted one are confirmed in the order of their gain
   bound (`rank_positions`), and the first eligible confirmation is the operating point. A confirmation re-reads the
   rows of a probe.

A candidate is rejected only when no position is within the limits or no confirmation succeeds. Any other solved axis
is calibrated on its ladder blocks:

1. **Ladder**: every ladder position is measured on the ladder block of each constraint and of the objective (the
   probe rows of stages 2 and 3 are prefixes of these).
2. **Rank**: per constraint, `provisional_positions` keeps the positions worth confirming (at `settings.ladder_power`,
   which is more lenient than `settings.power` since confirmation applies the full band). The positions kept by
   every constraint and within the cost and degeneracy limits on the ladder blocks are ordered by the lower bound of
   their behavior gain (`rank_positions`).
3. **Confirm**: the top-ranked position is confirmed on the fresh blocks, then the next-ranked one, and so on.

Either way, the pick is the position with the best behavior among those that can be confirmed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.fits import fit_digests
from steerability.algorithms.core.recipe import Recipe
from steerability.optimization.evaluation.evaluator import (
    Confirmation,
    Partial,
    Rejection,
    confirm,
    confirm_constraints,
    margins,
    measure,
    summary_of,
)
from steerability.optimization.evaluation.result import LadderPoint, OperatingPoint, level_of, summarize
from steerability.optimization.measurement.evidence import cost_ratio, degenerate_increase, paired
from steerability.optimization.measurement.stats import Paired, clearly_over, provisional_positions, rank_positions

if TYPE_CHECKING:
    from steerability.optimization.evaluation.session import StudySession


@dataclass
class _Rung:
    """One position of the ladder and what was measured for it."""

    position: Mapping[str, Any]
    recipe: Recipe
    fraction: float | None
    resolved_id: str | None = None
    capability: dict[str, Paired] = field(default_factory=dict)
    behavior: Paired | None = None
    cost_ratio: float = 1.0
    degenerate_increase: float = 0.0
    pruned: bool = False


def calibrate(session: StudySession, partial: Partial, *,
              deadline: float | None) -> tuple[OperatingPoint, tuple[LadderPoint, ...]]:
    """Choose and confirm the operating point of a candidate with a solved axis.

    Args:
        session: The session.
        partial: The screened candidate.
        deadline: A `time.monotonic()` deadline, or None.

    Returns:
        The operating point and one `LadderPoint` per measured position, with its place in the ranking.

    Raises:
        Rejection: With reason `no_feasible_point` when no position is eligible or none is confirmed.
        Stopped: If a limit or the deadline has passed.
        SteeringFailure: If a recipe fails to steer.
    """
    if partial.plan.monotone:
        return _calibrate_monotone(session, partial, deadline=deadline)
    return _calibrate_ladder(session, partial, deadline=deadline)


def _ladder_of(partial: Partial) -> list[_Rung]:
    plan = partial.plan
    fractions = plan.fractions or (None,) * len(plan.ladder)
    return [_Rung(position, recipe, fraction)
            for position, recipe, fraction in zip(plan.ladder, plan.recipes, fractions)]


def _closest_rejection(session: StudySession, rungs: list[_Rung], ladder: tuple[LadderPoint, ...]) -> Rejection:
    """The rejection of a candidate without an eligible position, with the tables of the position closest to them."""
    closest = min(rungs, key=lambda rung: level_of(rung.capability, margins(session)))
    summary = summarize(
        capability=closest.capability, behavior=closest.behavior, margins=margins(session),
        fitness_alpha=session.study.settings.fitness_alpha, cost_ratio=closest.cost_ratio,
        degenerate_increase=closest.degenerate_increase, position=closest.position,
    )
    return Rejection(4, "no_feasible_point", summary, ladder=ladder)


def _ladder_points(rungs: list[_Rung], ranking: list[int]) -> tuple[LadderPoint, ...]:
    """One `LadderPoint` per measured rung, in position order (an empty behavior table for a pruned rung)."""
    points = []
    for index, rung in enumerate(rungs):
        if rung.resolved_id is None:
            continue
        points.append(LadderPoint(
            position=dict(rung.position), recipe_id=rung.recipe.id, resolved_id=rung.resolved_id,
            capability=dict(rung.capability),
            behavior=rung.behavior if rung.behavior is not None else Paired(0, 0.0, 0.0, 0),
            rank=ranking.index(index) if index in ranking else None,
        ))
    return tuple(points)


def _confirm_in_order(session: StudySession, partial: Partial, rungs: list[_Rung], ranking: list[int],
                      deadline: float | None) -> tuple[OperatingPoint, tuple[LadderPoint, ...]]:
    """Confirm the ranked rungs in order and return the first eligible one."""
    summary: dict[str, float | str] = {}
    for index in ranking:
        rung = rungs[index]
        outcome = confirm(session, partial, rung.recipe, calibration_round=partial.round_index,
                          fresh_round=partial.round_index, deadline=deadline)
        if outcome.eligible:
            return outcome.point(rung.position, rung.recipe.id), _ladder_points(rungs, ranking)
        summary = summary_of(session, outcome, rung.position)
    raise Rejection(4, "no_feasible_point", summary, ladder=_ladder_points(rungs, ranking))


def _measure_rung(session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> None:
    study, store = session.study, session.store
    batches = session.batches(partial.round_index)
    ladder_items: dict[str, tuple[str, ...]] = {}
    resolved = ""
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        items = batches[benchmark.id].ladder
        resolved = measure(session, partial, rung.recipe, benchmark, items, deadline)
        rung.capability[benchmark.name] = paired(study, store, resolved, benchmark, items)
        ladder_items[benchmark.id] = items
    objective = study.objective
    items = batches[objective.id].ladder
    resolved = measure(session, partial, rung.recipe, objective, items, deadline)
    rung.behavior = paired(study, store, resolved, objective, items)
    ladder_items[objective.id] = items
    rung.resolved_id = resolved
    rung.cost_ratio = cost_ratio(study, store, resolved, ladder_items)
    rung.degenerate_increase = degenerate_increase(study, store, resolved, items)


def _rank(session: StudySession, rungs: list[_Rung]) -> list[int]:
    """The confirmation order of the rungs (indices), best first."""
    study, settings = session.study, session.study.settings
    kept = set(range(len(rungs)))
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        tables = [rung.capability[benchmark.name] for rung in rungs]
        kept &= set(provisional_positions(
            tables, constraint.margin, n_target=session.looks(benchmark)[2], n_held_out=session.n_held_out(benchmark),
            alpha=settings.alpha, power=settings.ladder_power, benchmarks=len(study.constraints),
            guard=settings.guard_band,
        ))
    within = [
        index for index in sorted(kept)
        if rungs[index].cost_ratio <= settings.max_cost_ratio
        and rungs[index].degenerate_increase <= settings.max_degenerate_increase
    ]
    return rank_positions(within, [rung.behavior for rung in rungs], settings.fitness_alpha)


def _calibrate_ladder(session: StudySession, partial: Partial, *,
                      deadline: float | None) -> tuple[OperatingPoint, tuple[LadderPoint, ...]]:
    rungs = _ladder_of(partial)
    for rung in rungs:
        _measure_rung(session, partial, rung, deadline)
    ranking = _rank(session, rungs)
    if not ranking:
        raise _closest_rejection(session, rungs, _ladder_points(rungs, ranking))
    return _confirm_in_order(session, partial, rungs, ranking, deadline)


def _first_look(session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> bool:
    """Measure a rung on the first look of each constraint's ladder block; whether it is clearly over a margin."""
    study, store, settings = session.study, session.store, session.study.settings
    batches = session.batches(partial.round_index)
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        items = batches[benchmark.id].ladder[:session.looks(benchmark)[0]]
        rung.resolved_id = measure(session, partial, rung.recipe, benchmark, items, deadline)
        table = paired(study, store, rung.resolved_id, benchmark, items)
        rung.capability[benchmark.name] = table
        if clearly_over(table, constraint.margin, settings.alpha):
            rung.pruned = True
            return True
    return False


def _behavior_ladder(session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> None:
    """Measure a rung on the objective's ladder block, with its limits on the blocks it was measured on."""
    study, store = session.study, session.store
    batches = session.batches(partial.round_index)
    objective = study.objective
    items = batches[objective.id].ladder
    rung.resolved_id = measure(session, partial, rung.recipe, objective, items, deadline)
    rung.behavior = paired(study, store, rung.resolved_id, objective, items)
    measured = {objective.id: items}
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        measured[benchmark.id] = batches[benchmark.id].ladder[:session.looks(benchmark)[0]]
    rung.cost_ratio = cost_ratio(study, store, rung.resolved_id, measured)
    rung.degenerate_increase = degenerate_increase(study, store, rung.resolved_id, items)


def _eligible(session: StudySession, rung: _Rung) -> bool:
    """Whether a rung has a behavior ladder and is within the cost and degeneracy limits on its ladder blocks."""
    settings = session.study.settings
    return (rung.behavior is not None and not rung.pruned and rung.cost_ratio <= settings.max_cost_ratio
            and rung.degenerate_increase <= settings.max_degenerate_increase)


def _probe(session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> Confirmation:
    """The sequential looks of a confirmation on every constraint's fresh block, measuring nothing else."""
    accepted, resolved, capability, _ = confirm_constraints(
        session, partial, rung.recipe, calibration_round=partial.round_index, fresh_round=partial.round_index,
        deadline=deadline,
    )
    return Confirmation(accepted, None if accepted else "not_confirmed", resolved, capability)


def _calibrate_monotone(session: StudySession, partial: Partial, *,
                        deadline: float | None) -> tuple[OperatingPoint, tuple[LadderPoint, ...]]:
    plan = partial.plan
    settings = session.study.settings
    ladder = _ladder_of(partial)
    cut = len(ladder)
    for index, rung in enumerate(ladder):
        if _first_look(session, partial, rung, deadline):
            cut = index
            break
    for rung in ladder[:cut]:
        _behavior_ladder(session, partial, rung, deadline)
    eligible = [index for index, rung in enumerate(ladder) if _eligible(session, rung)]
    if not eligible:
        measured = [rung for rung in ladder if rung.resolved_id is not None]
        raise _closest_rejection(session, measured, _ladder_points(ladder, []))

    best = max(eligible, key=lambda index: (ladder[index].behavior.lower(settings.fitness_alpha), -index))
    search = [index for index in eligible if index <= best]
    low, high = -1, len(search)
    probe: Confirmation | None = None
    while high - low > 1:
        middle = (low + high) // 2
        probe = _probe(session, partial, ladder[search[middle]], deadline)
        if probe.eligible:
            low = middle
        else:
            high = middle
    if low < 0:
        summary = summary_of(session, probe, ladder[search[0]].position)
        raise Rejection(4, "no_feasible_point", summary, ladder=_ladder_points(ladder, []))
    highest = ladder[search[low]]

    refinements = settings.ladder_refinements if plan.solved.continuous else 0
    digests = fit_digests(plan.recipes[0]) if refinements else None
    while refinements > 0 and ladder.index(highest) < len(ladder) - 1:
        refinements -= 1
        place = ladder.index(highest) + 1
        fraction = (highest.fraction + ladder[place].fraction) / 2
        position = plan.solved.at(fraction)
        if any(position == rung.position for rung in ladder):
            continue
        recipe = plan.template.bind(position).recipe()
        if fit_digests(recipe) != digests:
            continue
        rung = _Rung(position, recipe, fraction)
        ladder.insert(place, rung)
        if _first_look(session, partial, rung, deadline):
            continue
        _behavior_ladder(session, partial, rung, deadline)
        if _eligible(session, rung) and _probe(session, partial, rung, deadline).eligible:
            highest = rung

    top = ladder.index(highest)
    ranking = rank_positions([index for index in range(top + 1) if _eligible(session, ladder[index])],
                             [rung.behavior for rung in ladder], settings.fitness_alpha)
    return _confirm_in_order(session, partial, ladder, ranking, deadline)


__all__ = ["calibrate"]
