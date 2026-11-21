"""A proposer that draws its candidates at random."""
from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.recipe import assignments
from steerability.optimization.search.proposers.base import BaseProposer
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.study import Study

logger = logging.getLogger(__name__)


class RandomProposer(BaseProposer):
    """Candidates drawn at random: a uniformly chosen structure, and one random draw of its proposer axes.

    Draw `i` uses `derive_seed(seed, i)` for the axes (one index per discrete axis and one fraction per continuous
    axis, as `assignments` draws them), and a structure chosen with `derive_seed(seed, i, "structure")`. `Text`
    ranges cannot be drawn and keep the value of the structure's seed candidate (an empty string), and the
    constructor logs a warning for each structure that has one. An unordered `Choice` of strings is drawn like any
    other discrete axis, which makes a text setting searchable. Candidates already in the archive are skipped, and
    at most `10 * n` draws are made per `ask`.

    Args:
        study: The study.
        seed: The proposer's seed.
    """

    name = "random"

    def __init__(self, study: Study, seed: int) -> None:
        super().__init__(study)
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
            if self._fixed[structure]:
                logger.warning(
                    "RandomProposer cannot draw text; the Text axis paths %s of structure %r keep the empty seed "
                    "value. A Choice of strings makes such an axis searchable.", sorted(self._fixed[structure]),
                    structure,
                )
        self._draw = 0

    def _draws(self) -> Iterator[Candidate]:
        while True:
            draw = self._draw
            self._draw += 1
            chooser = random.Random(derive_seed(self.seed, draw, "structure"))
            structure = self._structures[chooser.randrange(len(self._structures))]
            values = dict(self._fixed[structure])
            axes = self._axes[structure]
            if axes:
                values.update(next(assignments(axes, strategy="random", samples=1, seed=derive_seed(self.seed, draw))))
            yield Candidate(structure, values)

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Up to `n` new random candidates."""
        return self._take_new(itertools.islice(self._draws(), 10 * n), view, n)


__all__ = ["RandomProposer"]
