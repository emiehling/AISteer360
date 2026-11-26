"""A proposer that enumerates every combination of the proposer axes."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
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
        structures: The structures to enumerate, or None for every structure of the study. Only these structures need
            discrete proposer axes.

    Raises:
        ValueError: If `structures` gives a structure that is not in the study's space, or a proposer axis of a
            selected structure is continuous or a `Text` range.
    """

    name = "grid"

    def __init__(self, study: Study, *, structures: Sequence[str] | None = None) -> None:
        super().__init__(study=study, structures=structures)
        for structure in self.structures:
            for key, axis in study.proposer_axes(structure=structure).items():
                if axis.has_text or axis.continuous:
                    raise ValueError(
                        f"GridProposer needs discrete proposer axes; axis {key!r} of structure {structure!r} is "
                        f"{'a Text range' if axis.has_text else 'continuous'}."
                    )
        self._candidates = self._enumerate()

    def _enumerate(self) -> Iterator[Candidate]:
        for structure in self.structures:
            axes = self.study.proposer_axes(structure=structure)
            if not axes:
                yield Candidate(structure=structure, values={})
                continue
            for values in assignments(axes=axes, strategy="grid"):
                yield Candidate(structure=structure, values=values)

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """The next `n` grid candidates that are not in the archive."""
        return self._take_new(self._candidates, view, n)


__all__ = ["GridProposer"]
