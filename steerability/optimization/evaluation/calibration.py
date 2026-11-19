"""Calibration of a candidate's solved axis: choose a position on the ladder blocks, then confirm it on fresh blocks.

The steps are:

1. **Ladder**: every ladder position is measured on the ladder block of each constraint and of the objective (the
   probe rows of stages 2 and 3 are prefixes of these).
2. **Rank**: per constraint, `provisional_positions` keeps the positions worth confirming (at `settings.ladder_power`,
   which is more lenient than `settings.power` since confirmation applies the full band). The positions kept by
   every constraint and within the cost and degeneracy limits on the ladder blocks are ordered by the lower bound of
   their behavior gain (`rank_positions`).
3. **Refine**: for a continuous axis in a monotone plan, while refinements remain, a position halfway between the
   last kept position and the next one is inserted and measured, and the ranking is repeated.
4. **Confirm**: the top-ranked position is confirmed on the fresh blocks, then the next-ranked one, and so on.

The pick is the position with the best behavior among those that can be confirmed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.fits import fit_digests
from steerability.algorithms.core.recipe import Recipe
from steerability.optimization.evaluation.evaluator import Partial, Rejection, confirm, margins, measure, summary_of
from steerability.optimization.evaluation.result import LadderPoint, OperatingPoint, level_of, summarize
from steerability.optimization.measurement.evidence import cost_ratio, degenerate_increase, paired
from steerability.optimization.measurement.stats import Paired, confirm_looks, provisional_positions, rank_positions

if TYPE_CHECKING:
    from steerability.optimization.evaluation.session import StudySession


@dataclass
class _Rung:
    """One measured position of the ladder."""

    position: Mapping[str, Any]
    recipe: Recipe
    fraction: float | None
    resolved_id: str | None = None
    capability: dict[str, Paired] = field(default_factory=dict)
    behavior: Paired | None = None
    cost_ratio: float = 1.0
    degenerate_increase: float = 0.0


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


def _rank(session: StudySession, rungs: list[_Rung], monotone: bool) -> list[int]:
    """The confirmation order of the rungs (indices), best first."""
    study, settings = session.study, session.study.settings
    kept = set(range(len(rungs)))
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        tables = [rung.capability[benchmark.name] for rung in rungs]
        kept &= set(provisional_positions(
            tables, constraint.margin, n_target=confirm_looks(constraint.margin)[2],
            n_sealed=session.n_sealed(benchmark), sd_floor=session.sd_floor(benchmark), monotone=monotone,
            alpha=settings.alpha, power=settings.ladder_power, benchmarks=len(study.constraints),
            guard=settings.guard_band,
        ))
    within = [
        index for index in sorted(kept)
        if rungs[index].cost_ratio <= settings.max_cost_ratio
        and rungs[index].degenerate_increase <= settings.max_degenerate_increase
    ]
    return rank_positions(within, [rung.behavior for rung in rungs], settings.fitness_alpha)


def _ladder_points(rungs: list[_Rung], ranking: list[int]) -> tuple[LadderPoint, ...]:
    points = []
    for index, rung in enumerate(rungs):
        rank = ranking.index(index) if index in ranking else None
        points.append(LadderPoint(
            position=dict(rung.position), recipe_id=rung.recipe.id, resolved_id=rung.resolved_id or "",
            capability=dict(rung.capability), behavior=rung.behavior, rank=rank,
        ))
    return tuple(points)


def calibrate(session: StudySession, partial: Partial, *,
              deadline: float | None) -> tuple[OperatingPoint, tuple[LadderPoint, ...]]:
    """Choose and confirm the operating point of a candidate with a solved axis.

    Args:
        session: The session.
        partial: The screened candidate.
        deadline: A `time.monotonic()` deadline, or None.

    Returns:
        The operating point and one `LadderPoint` per evaluated position, with its place in the ranking.

    Raises:
        Rejection: With reason `no_feasible_point` when no position is kept or none is confirmed.
        Stopped: If a limit or the deadline has passed.
        SteeringFailure: If a recipe fails to steer.
    """
    plan = partial.plan
    settings = session.study.settings
    fractions = plan.fractions or (None,) * len(plan.ladder)
    rungs = [_Rung(position, recipe, fraction)
             for position, recipe, fraction in zip(plan.ladder, plan.recipes, fractions)]
    refinements = settings.ladder_refinements if plan.solved.continuous and plan.monotone else 0
    digests = fit_digests(plan.recipes[0]) if refinements else None
    while True:
        for rung in rungs:
            if rung.resolved_id is None:
                _measure_rung(session, partial, rung, deadline)
        ranking = _rank(session, rungs, plan.monotone)
        if refinements <= 0 or not ranking:
            break
        last = max(ranking)
        if last >= len(rungs) - 1:
            break
        refinements -= 1
        fraction = (rungs[last].fraction + rungs[last + 1].fraction) / 2
        position = plan.solved.at(fraction)
        if any(position == rung.position for rung in rungs):
            continue
        recipe = plan.template.bind(position).recipe()
        if fit_digests(recipe) != digests:
            continue
        rungs.insert(last + 1, _Rung(position, recipe, fraction))

    summary: dict[str, float | str] = {}
    if not ranking:
        closest = min(rungs, key=lambda rung: level_of(rung.capability, margins(session)))
        summary = summarize(
            capability=closest.capability, behavior=closest.behavior, margins=margins(session),
            fitness_alpha=settings.fitness_alpha, cost_ratio=closest.cost_ratio,
            degenerate_increase=closest.degenerate_increase, position=closest.position,
        )
        raise Rejection(4, "no_feasible_point", summary, ladder=_ladder_points(rungs, ranking))
    for index in ranking:
        rung = rungs[index]
        outcome = confirm(session, partial, rung.recipe, calibration_round=partial.round_index,
                          fresh_round=partial.round_index, deadline=deadline)
        if outcome.eligible:
            return outcome.point(rung.position, rung.recipe.id), _ladder_points(rungs, ranking)
        summary = summary_of(session, outcome, rung.position)
    raise Rejection(4, "no_feasible_point", summary, ladder=_ladder_points(rungs, ranking))


__all__ = ["calibrate"]
