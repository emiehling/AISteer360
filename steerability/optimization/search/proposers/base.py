"""The contract of a proposer and the bookkeeping that proposers share."""
from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from steerability.algorithms.core.recipe import RecipeError
from steerability.optimization.search.engine import SearchEngine, select_structures

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.study import Candidate, Study

logger = logging.getLogger(__name__)


class BaseProposer(SearchEngine):
    """A search engine that the toolkit asks for candidates and tells their results (the ask/tell shape).

    A subclass implements `ask`, and overrides `tell` when it learns from results. `run` asks for
    `settings.round_candidates` candidates at a time, prepares their fits together, and evaluates each in order.

    Args:
        study: The study.
        structures: The structures to propose, or None for every structure of the study.

    Attributes:
        name: The proposer's name, recorded with each candidate it proposes.
        study: The study.
        structures: The structures the proposer proposes, in the order of `study.space`.

    Raises:
        ValueError: If `structures` is empty or gives a structure that is not in the study's space.
    """

    def __init__(self, study: Study, *, structures: Sequence[str] | None = None) -> None:
        self.study = study
        self.structures = select_structures(study=study, structures=structures)
        self._proposed: set[str] = set()

    @abstractmethod
    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Up to `n` candidates to evaluate next; an empty list ends the proposer's run."""

    def tell(self, candidate: Candidate, result: Result) -> None:
        """Receive the result of a proposed candidate. The default does nothing."""

    def run(
        self,
        *,
        session: StudySession,
        initial_candidates: Sequence[Candidate] = (),
    ) -> None:
        """Ask for candidates and have the session evaluate them until the session is exhausted.

        Each batch has `settings.round_candidates` candidates. Their fits are prepared together, and each is
        evaluated in order and told to the proposer. The run stops when the session is exhausted, when `ask` returns
        no candidate, or when a batch records no new result (every candidate was already recorded, did not bind, or
        was stopped). The initial candidates are in the archive already, which `ask` reads.

        Args:
            session: The started session of the run.
            initial_candidates: The candidates given to `run_study`, already evaluated.

        Raises:
            ValueError: If the session's study is not the proposer's study.
        """
        if session.study.id != self.study.id:
            raise ValueError(
                f"Proposer {self.name!r} was built for study {self.study.id}, and the session runs study "
                f"{session.study.id}."
            )
        while not session.exhausted():
            candidates = self.ask(view=session.archive.view(), n=self.study.settings.round_candidates)
            if not candidates:
                break
            session.prepare(candidates=candidates)
            recorded = session.recorded
            for candidate in candidates:
                result = session.evaluate(candidate=candidate, proposer=self.name)
                self.tell(candidate=candidate, result=result)
            if session.recorded == recorded and not session.exhausted():
                logger.info("Proposer %r proposed no new candidate; its run stops.", self.name)
                break

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
