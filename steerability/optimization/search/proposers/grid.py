"""A proposer that enumerates every combination of the proposer axes."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from steerability.algorithms.core.recipe import assignments
from steerability.optimization.search.population import is_trained, trained_values
from steerability.optimization.search.proposers.base import BaseProposer
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.study import Study


class GridProposer(BaseProposer):
    """Every combination of the proposer axes, structure by structure in the order of `study.space`.

    Candidates already in the archive are skipped. When no new structural fit may run (`ArchiveView.fits_open` is
    false), a candidate whose training entries do not all take trained values (`trained_values`) is held back, and
    it is proposed once new structural fits may run again.

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
        self._held: list[Candidate] = []

    def _enumerate(self) -> Iterator[Candidate]:
        for structure in self.structures:
            axes = self.study.proposer_axes(structure=structure)
            if not axes:
                yield Candidate(structure=structure, values={})
                continue
            for values in assignments(axes=axes, strategy="grid"):
                yield Candidate(structure=structure, values=values)

    def _pending(self) -> Iterator[Candidate]:
        """The held candidates, then the rest of the grid; a candidate leaves its source when it is yielded."""
        while self._held:
            yield self._held.pop(0)
        # a for loop keeps the grid open when this generator is closed early, which `yield from` would not
        for candidate in self._candidates:
            yield candidate

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """The next `n` grid candidates that are not in the archive, on trained values when no new fit may run."""
        if view.fits_open:
            return self._take_new(candidates=self._pending(), view=view, n=n)
        trained = {
            structure: trained_values(study=self.study, view=view, structure=structure) for structure in self.structures
        }
        held: list[Candidate] = []

        def available() -> Iterator[Candidate]:
            for candidate in self._pending():
                if is_trained(study=self.study, candidate=candidate, trained=trained[candidate.structure]):
                    yield candidate
                else:
                    held.append(candidate)

        taken = self._take_new(candidates=available(), view=view, n=n)
        self._held = held + self._held
        return taken


__all__ = ["GridProposer"]
