"""Proposers and the in-process study loop."""
from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING

from steerability.optimization.evaluation.session import StudySession

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import PipelineFactory
    from steerability.optimization.evaluation.archive import Archive, ArchiveView
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.study import Candidate, Study

logger = logging.getLogger(__name__)


class Proposer(typing.Protocol):
    """A source of candidates that sees only the archive's view.

    Attributes:
        name: The proposer's name, recorded with each candidate it proposes.
    """

    name: str

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Up to `n` candidates to evaluate next; an empty list ends the study."""
        ...

    def tell(self, candidate: Candidate, result: Result) -> None:
        """Receive the result of a proposed candidate."""
        ...


def run_study(
    study: Study,
    proposer: Proposer,
    *,
    store: Store,
    factory: PipelineFactory | None = None,
    max_candidates: int | None = None,
    max_generations: int | None = None,
    max_seconds: float | None = None,
    allow_infeasible: bool = False,
) -> Archive:
    """Run a study with an in-process proposer until it is exhausted or the proposer has nothing new.

    A session is started, and until it is exhausted the proposer is asked for `settings.round_candidates`
    candidates, their fits are prepared together, and each is evaluated in order and told to the proposer. The loop
    stops when the proposer returns no candidate, or when a batch records no new result (every candidate was
    already recorded or did not bind).

    Args:
        study: The study.
        proposer: The proposer.
        store: The store.
        factory: The pipeline factory, or None to build one.
        max_candidates: The number of candidates to record, or None.
        max_generations: The number of generations to run, or None.
        max_seconds: The wall time to run, or None.
        allow_infeasible: Whether to accept a study that fails the feasibility check.

    Returns:
        The study's archive.
    """
    session = StudySession(
        study, store=store, factory=factory, max_candidates=max_candidates, max_generations=max_generations,
        max_seconds=max_seconds, allow_infeasible=allow_infeasible,
    )
    try:
        session.start()
        while not session.exhausted():
            candidates = proposer.ask(session.archive.view(), study.settings.round_candidates)
            if not candidates:
                break
            session.prepare(candidates)
            recorded = len(session.archive.view())
            for candidate in candidates:
                result = session.evaluate(candidate, proposer=proposer.name)
                proposer.tell(candidate, result)
            if len(session.archive.view()) == recorded and not session.exhausted():
                logger.info("Proposer %r proposed no new candidate; the study stops.", proposer.name)
                break
    finally:
        session.close()
    return session.archive


__all__ = ["Proposer", "run_study"]
