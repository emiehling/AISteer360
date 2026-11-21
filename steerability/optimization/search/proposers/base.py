"""The contract of a proposer and the bookkeeping that proposers share."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from steerability.algorithms.core.recipe import RecipeError

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.study import Candidate, Study


class BaseProposer(ABC):
    """A source of candidates for a study that sees only the archive's view.

    A subclass implements `ask`, and overrides `tell` when it learns from results.

    Args:
        study: The study.

    Attributes:
        name: The proposer's name, recorded with each candidate it proposes.
        study: The study.
    """

    name: ClassVar[str]

    def __init__(self, study: Study) -> None:
        self.study = study
        self._proposed: set[str] = set()

    @abstractmethod
    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Up to `n` candidates to evaluate next; an empty list ends the study."""

    def tell(self, candidate: Candidate, result: Result) -> None:
        """Receive the result of a proposed candidate. The default does nothing."""

    def _take_new(self, candidates: Iterable[Candidate], view: ArchiveView, n: int) -> list[Candidate]:
        """The first `n` of `candidates` that bind and are neither in the archive nor proposed before."""
        known = set(view.ids) | self._proposed
        taken: list[Candidate] = []
        for candidate in candidates:
            try:
                candidate_id = self.study.candidate_id(candidate)
            except RecipeError:
                continue
            if candidate_id in known:
                continue
            known.add(candidate_id)
            self._proposed.add(candidate_id)
            taken.append(candidate)
            if len(taken) >= n:
                break
        return taken


__all__ = ["BaseProposer"]
