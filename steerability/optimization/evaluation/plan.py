"""The evaluation plan of a candidate: its bound template, its solved axis, and the recipes of the axis's ladder."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.fits import fit_digests, product_weights
from steerability.algorithms.core.recipe import Axis, Recipe, RecipeError, Template
from steerability.spipe.errors import SpipeSaveError

if TYPE_CHECKING:
    from steerability.optimization.study import Candidate, Study


@dataclass(frozen=True)
class EvaluationPlan:
    """What evaluating a candidate involves, computed without model work.

    Attributes:
        candidate_id: The candidate's id.
        template: The bound candidate, with only the solved axis (if any) free.
        solved: The solved axis, or None.
        monotone: Whether the solved axis is in `study.monotone`.
        ladder: The positions of the solved axis in order (one empty mapping without a solved axis).
        recipes: The recipe of each ladder position, `template.bind(position).recipe()`.
        fractions: The fraction of each ladder position on a continuous solved axis, or None.
    """

    candidate_id: str
    template: Template
    solved: Axis | None
    monotone: bool
    ladder: tuple[Mapping[str, Any], ...]
    recipes: tuple[Recipe, ...]
    fractions: tuple[float, ...] | None = None


def continuous_fractions(axis: Axis, n: int) -> tuple[float, ...]:
    """The fractions of the positions `axis.positions(n)` returns for a continuous axis.

    A position equal to the one before it is dropped, as in `Axis.positions`.
    """
    fractions: list[float] = []
    previous: dict[str, Any] | None = None
    for index in range(n):
        fraction = index / (n - 1) if n > 1 else 0.0
        position = axis.at(fraction)
        if previous is None or position != previous:
            fractions.append(fraction)
            previous = position
    return tuple(fractions)


def _check_products(*, structure: str, recipe: Recipe) -> None:
    """Check that every structural product of a recipe has an identity and is trained once (`product_weights`).

    Raises:
        RecipeError: If a product's fit has no identity or the recipe contains one training twice.
    """
    try:
        product_weights(recipe=recipe)
    except RecipeError:
        raise
    except (TypeError, ValueError, SpipeSaveError) as error:
        raise RecipeError(f"Structure {structure!r}: a structural fit of the recipe has no identity "
                          f"({type(error).__name__}: {error}).") from error


def make_plan(study: Study, candidate: Candidate) -> EvaluationPlan:
    """The plan of a candidate.

    The ladder is `solved.positions(settings.ladder_points)` for a continuous solved axis and every position of a
    discrete one. Every ladder recipe must have the same `fit_digests`. One set of fits then serves the whole ladder,
    and solving the axis costs generations only.

    Args:
        study: The study.
        candidate: The candidate.

    Returns:
        The plan.

    Raises:
        RecipeError: If the candidate does not bind, a ladder recipe has an argument that cannot be encoded, the
            solved axis changes a fit (the message points to `study.searched`), or a recipe contains one training
            twice (`product_weights`).
    """
    candidate_id = study.candidate_id(candidate)
    template = study.bind(candidate)
    expected = study.solved_axis(candidate.structure)
    remaining = template.axes()
    if set(remaining) != ({expected.key} if expected is not None else set()):
        raise RecipeError(
            f"Candidate of structure {candidate.structure!r} leaves axes {sorted(remaining)} free; only the solved "
            "axis may be free."
        )
    if expected is None:
        recipe = template.recipe()
        _ = recipe.id
        _check_products(structure=candidate.structure, recipe=recipe)
        return EvaluationPlan(candidate_id, template, None, False, ({},), (recipe,))
    solved = remaining[expected.key]
    fractions = None
    if solved.continuous:
        positions = solved.positions(study.settings.ladder_points)
        fractions = continuous_fractions(solved, study.settings.ladder_points)
    else:
        positions = solved.positions()
    recipes = tuple(template.bind(position).recipe() for position in positions)
    digests = []
    for recipe in recipes:
        _ = recipe.id
        try:
            digests.append(fit_digests(recipe))
        except (TypeError, ValueError, SpipeSaveError) as error:
            raise RecipeError(f"Structure {candidate.structure!r}: the fits of a ladder recipe have no identity "
                              f"({type(error).__name__}: {error}).") from error
    if any(digest != digests[0] for digest in digests):
        raise RecipeError(
            f"Structure {candidate.structure!r}: the solved axis {solved.key!r} changes a fit, and its positions "
            f"cannot share fitted artifacts; list {solved.key!r} in study.searched to fit each proposed value once."
        )
    _check_products(structure=candidate.structure, recipe=recipes[0])
    return EvaluationPlan(
        candidate_id, template, solved, solved.key in study.monotone, tuple(positions), recipes, fractions,
    )


__all__ = ["EvaluationPlan", "continuous_fractions", "make_plan"]
