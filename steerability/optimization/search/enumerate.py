"""Proposers that enumerate or sample the proposer axes of a study's structures."""
from __future__ import annotations

import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.recipe import RecipeError, assignments
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.study import Study


class GridProposer:
    """Every combination of the proposer axes, structure by structure in the order of `study.space`.

    Candidates already in the archive are skipped.

    Args:
        study: The study.

    Raises:
        ValueError: If a proposer axis is continuous or a `Text` range.
    """

    name = "grid"

    def __init__(self, study: Study) -> None:
        self.study = study
        for structure in study.space:
            for key, axis in study.proposer_axes(structure).items():
                if axis.has_text or axis.continuous:
                    raise ValueError(
                        f"GridProposer needs discrete proposer axes; axis {key!r} of structure {structure!r} is "
                        f"{'a Text range' if axis.has_text else 'continuous'}."
                    )
        self._candidates = self._enumerate()
        self._proposed: set[str] = set()

    def _enumerate(self) -> Iterator[Candidate]:
        for structure in self.study.space:
            axes = self.study.proposer_axes(structure)
            if not axes:
                yield Candidate(structure, {})
                continue
            for values in assignments(axes, strategy="grid"):
                yield Candidate(structure, values)

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """The next `n` grid candidates that are not in the archive."""
        known = view.ids | self._proposed
        proposed: list[Candidate] = []
        for candidate in self._candidates:
            candidate_id = self.study.candidate_id(candidate)
            if candidate_id in known:
                continue
            self._proposed.add(candidate_id)
            proposed.append(candidate)
            if len(proposed) >= n:
                break
        return proposed

    def tell(self, candidate: Candidate, result: Result) -> None:
        """Nothing to learn; the grid is fixed."""


class RandomProposer:
    """Candidates drawn at random: a uniformly chosen structure, and one random draw of its proposer axes.

    Draw `i` uses `derive_seed(seed, i)` for the axes (one index per discrete axis and one fraction per continuous
    axis, as `assignments` draws them), and a structure chosen with `derive_seed(seed, i, "structure")`. `Text`
    ranges cannot be drawn and keep the value of the structure's seed candidate (an empty string). Candidates already
    in the archive are skipped, and at most `10 * n` draws are made per `ask`.

    Args:
        study: The study.
        seed: The proposer's seed.
    """

    name = "random"

    def __init__(self, study: Study, seed: int) -> None:
        self.study = study
        self.seed = int(seed)
        self._structures = list(study.space)
        self._axes: dict[str, dict[str, Any]] = {}
        self._fixed: dict[str, dict[str, Any]] = {}
        for structure in self._structures:
            axes = study.proposer_axes(structure)
            self._axes[structure] = {key: axis for key, axis in axes.items() if not axis.has_text}
            self._fixed[structure] = {
                path: "" for axis in axes.values() if axis.has_text for path in axis.members
            }
        self._draw = 0
        self._proposed: set[str] = set()

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Up to `n` new random candidates."""
        known = view.ids | self._proposed
        proposed: list[Candidate] = []
        for _ in range(10 * n):
            if len(proposed) >= n:
                break
            draw = self._draw
            self._draw += 1
            chooser = random.Random(derive_seed(self.seed, draw, "structure"))
            structure = self._structures[chooser.randrange(len(self._structures))]
            values = dict(self._fixed[structure])
            axes = self._axes[structure]
            if axes:
                values.update(next(assignments(axes, strategy="random", samples=1, seed=derive_seed(self.seed, draw))))
            candidate = Candidate(structure, values)
            try:
                candidate_id = self.study.candidate_id(candidate)
            except RecipeError:
                continue
            if candidate_id in known:
                continue
            known = known | {candidate_id}
            self._proposed.add(candidate_id)
            proposed.append(candidate)
        return proposed

    def tell(self, candidate: Candidate, result: Result) -> None:
        """Nothing to learn; draws do not depend on results."""


__all__ = ["GridProposer", "RandomProposer"]
