"""Calibration of a candidate's solved axis: choose a position, then confirm it on fresh blocks.

Every position of the solved axis is judged on its own evidence, and calibration assumes nothing about how capability
loss or behavior change along the axis. The steps are:

1. **First look**: every position is measured on each constraint's ladder block, which is as large as the first look
   of a confirmation (the items stage 3 read). A position whose first look is clearly over a margin is dropped and
   measured no further.
2. **Behavior ladder**: every other position is measured on the objective's ladder block. A position over the cost or
   degeneracy limit on the blocks it was measured on is dropped.
3. **Confirm**: the remaining positions are ordered by the lower bound of their ladder-block gain at
   `settings.fitness_alpha` (`rank_positions`) and confirmed in that order on fresh blocks (`confirm`). The first
   position that passes is the operating point, and the candidate is rejected (`no_feasible_point`) when none passes.
4. **Refine**: on a continuous axis, while `settings.ladder_refinements` remain, one position is inserted halfway in
   fraction between the operating point and a neighbor that was dropped or failed its confirmation (the one with the
   higher gain bound when both did). It is measured as in steps 1 and 2. It is confirmed when its gain bound is above
   the operating point's, and it becomes the operating point when it passes. A position whose recipe would change a
   fit is not inserted.

The order in which the positions are confirmed changes the cost of a candidate and not the decision. The pick is the
position with the best ladder-block gain among those that pass their own confirmation, and a ladder estimate alone
drops a position only when it is clearly over a margin. The recorded ranking (`LadderPoint.rank`) orders every
position that was not dropped by its gain bound, which is the order in which a re-scoring tries them.
"""
from __future__ import annotations

import math
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
    margins,
    measure,
    summary_of,
)
from steerability.optimization.evaluation.result import LadderPoint, OperatingPoint, level_of, summarize
from steerability.optimization.measurement.evidence import cost_ratio, degenerate_increase, paired
from steerability.optimization.measurement.stats import Paired, clearly_over, rank_positions

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
    failed: bool = False


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
    ladder = _ladder_of(partial=partial)
    for rung in ladder:
        _measure_rung(session=session, partial=partial, rung=rung, deadline=deadline)
    ranking = _ranking(session=session, ladder=ladder)
    if not ranking:
        measured = [rung for rung in ladder if rung.resolved_id is not None]
        raise _closest_rejection(session=session, rungs=measured, ladder=_ladder_points(session=session, ladder=ladder))
    operating: tuple[_Rung, Confirmation] | None = None
    summary: dict[str, float | str] = {}
    for index in ranking:
        rung = ladder[index]
        outcome = _confirm(session=session, partial=partial, rung=rung, deadline=deadline)
        if outcome.eligible:
            operating = (rung, outcome)
            break
        summary = summary_of(session=session, outcome=outcome, position=rung.position)
    if operating is None:
        raise Rejection(4, "no_feasible_point", summary, ladder=_ladder_points(session=session, ladder=ladder))
    rung, outcome = _refine(session=session, partial=partial, ladder=ladder, operating=operating, deadline=deadline)
    return (
        outcome.point(position=rung.position, recipe_id=rung.recipe.id),
        _ladder_points(session=session, ladder=ladder),
    )


def _ladder_of(*, partial: Partial) -> list[_Rung]:
    plan = partial.plan
    fractions = plan.fractions or (None,) * len(plan.ladder)
    return [
        _Rung(position=position, recipe=recipe, fraction=fraction)
        for position, recipe, fraction in zip(plan.ladder, plan.recipes, fractions)
    ]


def _measure_rung(*, session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> None:
    """Take a rung's first look and, when it is not clearly over a margin, its behavior ladder."""
    if not _first_look(session=session, partial=partial, rung=rung, deadline=deadline):
        _behavior_ladder(session=session, partial=partial, rung=rung, deadline=deadline)


def _first_look(*, session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> bool:
    """Measure a rung on each constraint's ladder block; whether it is clearly over a margin.

    The constraints are measured in order, and the first one on which the rung is clearly over ends the look.
    """
    study, store, settings = session.study, session.store, session.study.settings
    batches = session.batches(round_index=partial.round_index)
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        items = batches[benchmark.id].ladder
        rung.resolved_id = measure(
            session=session,
            partial=partial,
            recipe=rung.recipe,
            benchmark=benchmark,
            item_ids=items,
            deadline=deadline,
        )
        table = paired(study=study, store=store, resolved_id=rung.resolved_id, benchmark=benchmark, item_ids=items)
        rung.capability[benchmark.name] = table
        if clearly_over(paired=table, margin=constraint.margin, alpha=settings.alpha):
            rung.pruned = True
            return True
    return False


def _behavior_ladder(*, session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> None:
    """Measure a rung on the objective's ladder block, with its limits on the blocks it was measured on."""
    study, store = session.study, session.store
    batches = session.batches(round_index=partial.round_index)
    objective = study.objective
    items = batches[objective.id].ladder
    rung.resolved_id = measure(
        session=session,
        partial=partial,
        recipe=rung.recipe,
        benchmark=objective,
        item_ids=items,
        deadline=deadline,
    )
    rung.behavior = paired(study=study, store=store, resolved_id=rung.resolved_id, benchmark=objective, item_ids=items)
    measured = {objective.id: items}
    for constraint in study.constraints:
        measured[constraint.benchmark.id] = batches[constraint.benchmark.id].ladder
    rung.cost_ratio = cost_ratio(study=study, store=store, resolved_id=rung.resolved_id, items=measured)
    rung.degenerate_increase = degenerate_increase(
        study=study,
        store=store,
        resolved_id=rung.resolved_id,
        item_ids=items,
    )


def _eligible(*, session: StudySession, rung: _Rung) -> bool:
    """Whether a rung has a behavior ladder and is within the cost and degeneracy limits on its ladder blocks."""
    settings = session.study.settings
    return (rung.behavior is not None and not rung.pruned and rung.cost_ratio <= settings.max_cost_ratio
            and rung.degenerate_increase <= settings.max_degenerate_increase)


def _gain_bound(*, session: StudySession, rung: _Rung) -> float:
    """The lower bound of a rung's ladder-block gain, or minus infinity without a behavior ladder."""
    if rung.behavior is None:
        return -math.inf
    return rung.behavior.lower(alpha=session.study.settings.fitness_alpha)


def _ranking(*, session: StudySession, ladder: list[_Rung]) -> list[int]:
    """The rungs that were not dropped, best gain bound first (`rank_positions`)."""
    kept = [index for index, rung in enumerate(ladder) if _eligible(session=session, rung=rung)]
    return rank_positions(
        positions=kept,
        gains=[rung.behavior for rung in ladder],
        alpha=session.study.settings.fitness_alpha,
    )


def _confirm(*, session: StudySession, partial: Partial, rung: _Rung, deadline: float | None) -> Confirmation:
    """Confirm a rung on the fresh blocks of the candidate's round, marking it failed when it does not pass."""
    outcome = confirm(
        session=session,
        partial=partial,
        recipe=rung.recipe,
        calibration_round=partial.round_index,
        fresh_round=partial.round_index,
        deadline=deadline,
    )
    rung.failed = not outcome.eligible
    return outcome


def _refine(
    *,
    session: StudySession,
    partial: Partial,
    ladder: list[_Rung],
    operating: tuple[_Rung, Confirmation],
    deadline: float | None,
) -> tuple[_Rung, Confirmation]:
    """Insert positions next to the operating point on a continuous axis and confirm those with a higher gain bound.

    Each refinement goes halfway in fraction toward a neighbor of the operating point that was dropped or failed its
    confirmation, the one with the higher gain bound when both did, and the refinements end when no neighbor did. A
    position already on the ladder, or one whose recipe changes a fit, is not inserted and uses up its refinement.

    Returns:
        The operating rung and its confirmation.
    """
    plan = partial.plan
    refinements = session.study.settings.ladder_refinements if plan.solved.continuous else 0
    digests = fit_digests(recipe=plan.recipes[0]) if refinements else None
    while refinements > 0:
        refinements -= 1
        rung = operating[0]
        place = ladder.index(rung)
        neighbors = [ladder[index] for index in (place - 1, place + 1) if 0 <= index < len(ladder)]
        toward = [
            neighbor for neighbor in neighbors
            if neighbor.failed or not _eligible(session=session, rung=neighbor)
        ]
        if not toward:
            break
        target = max(toward, key=lambda neighbor: _gain_bound(session=session, rung=neighbor))
        fraction = (rung.fraction + target.fraction) / 2
        position = plan.solved.at(fraction=fraction)
        if any(position == other.position for other in ladder):
            continue
        recipe = plan.template.bind(assignment=position).recipe()
        if fit_digests(recipe=recipe) != digests:
            continue
        inserted = _Rung(position=position, recipe=recipe, fraction=fraction)
        ladder.insert(max(place, ladder.index(target)), inserted)
        _measure_rung(session=session, partial=partial, rung=inserted, deadline=deadline)
        higher = _gain_bound(session=session, rung=inserted) > _gain_bound(session=session, rung=rung)
        if _eligible(session=session, rung=inserted) and higher:
            outcome = _confirm(session=session, partial=partial, rung=inserted, deadline=deadline)
            if outcome.eligible:
                operating = (inserted, outcome)
    return operating


def _ladder_points(*, session: StudySession, ladder: list[_Rung]) -> tuple[LadderPoint, ...]:
    """One `LadderPoint` per measured rung, in position order (an empty behavior table for a pruned rung)."""
    ranking = _ranking(session=session, ladder=ladder)
    points = []
    for index, rung in enumerate(ladder):
        if rung.resolved_id is None:
            continue
        behavior = rung.behavior if rung.behavior is not None else Paired(n=0, total=0.0, total_sq=0.0, changed=0)
        points.append(
            LadderPoint(
                position=dict(rung.position),
                recipe_id=rung.recipe.id,
                resolved_id=rung.resolved_id,
                capability=dict(rung.capability),
                behavior=behavior,
                rank=ranking.index(index) if index in ranking else None,
            ),
        )
    return tuple(points)


def _closest_rejection(*, session: StudySession, rungs: list[_Rung], ladder: tuple[LadderPoint, ...]) -> Rejection:
    """The rejection of a candidate without an eligible position, with the tables of the position closest to them."""
    limits = margins(session=session)
    closest = min(rungs, key=lambda rung: level_of(capability=rung.capability, margins=limits))
    summary = summarize(
        capability=closest.capability,
        behavior=closest.behavior,
        margins=limits,
        fitness_alpha=session.study.settings.fitness_alpha,
        cost_ratio=closest.cost_ratio,
        degenerate_increase=closest.degenerate_increase,
        position=closest.position,
    )
    return Rejection(4, "no_feasible_point", summary, ladder=ladder)


__all__ = ["calibrate"]
