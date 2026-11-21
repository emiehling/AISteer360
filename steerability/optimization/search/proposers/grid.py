"""A proposer that enumerates every combination of the proposer axes."""
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from steerability.algorithms.core.recipe import assignments
from steerability.optimization.search.proposers.base import BaseProposer
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.study import Study


class GridProposer(BaseProposer):
    """Every combination of the proposer axes, structure by structure in the order of `study.space`.

    Candidates already in the archive are skipped.

    Args:
        study: The study.

    Raises:
        ValueError: If a proposer axis is continuous or a `Text` range.
    """

    name = "grid"

    def __init__(self, study: Study) -> None:
        super().__init__(study)
        for structure in study.space:
            for key, axis in study.proposer_axes(structure).items():
                if axis.has_text or axis.continuous:
                    raise ValueError(
                        f"GridProposer needs discrete proposer axes; axis {key!r} of structure {structure!r} is "
                        f"{'a Text range' if axis.has_text else 'continuous'}."
                    )
        self._candidates = self._enumerate()

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
        return self._take_new(self._candidates, view, n)


__all__ = ["GridProposer"]
