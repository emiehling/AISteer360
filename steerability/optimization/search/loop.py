"""A run of a study: its initial candidates, then search phases in order, on one session."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from steerability.optimization.evaluation.session import StudySession, check_limit
from steerability.optimization.search.engine import SearchEngine

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import PipelineFactory
    from steerability.optimization.evaluation.archive import Archive
    from steerability.optimization.study import Candidate, Study

logger = logging.getLogger(__name__)

INITIAL_PROPOSER = "initial"


@dataclass(frozen=True, kw_only=True)
class SearchPhase:
    """One engine and the allowance it runs under.

    The allowance limits the phase on top of the limits of the run (`StudySession.allowance`), and None leaves a
    limit open.

    Attributes:
        engine: The search engine.
        candidates: The number of candidates the phase may record, or None.
        generations: The number of rows the phase may generate, or None.
        seconds: The wall time the phase may run, or None.

    Raises:
        TypeError: If `engine` is not a `SearchEngine`.
        ValueError: If a limit is neither None nor a non-negative number (an integer for a count).
    """

    engine: SearchEngine
    candidates: int | None = None
    generations: int | None = None
    seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine, SearchEngine):
            raise TypeError(f"SearchPhase.engine must be a SearchEngine; got {type(self.engine).__name__}.")
        check_limit(name="candidates", value=self.candidates, count=True)
        check_limit(name="generations", value=self.generations, count=True)
        check_limit(name="seconds", value=self.seconds, count=False)


def run_study(
    *,
    study: Study,
    phases: Sequence[SearchPhase],
    store: Store,
    initial_candidates: Sequence[Candidate] = (),
    factory: PipelineFactory | None = None,
    max_candidates: int | None = None,
    max_generations: int | None = None,
    max_seconds: float | None = None,
    allow_infeasible: bool = False,
) -> Archive:
    """Run a study: evaluate the initial candidates, then run each phase's engine in order on one session.

    The steps are:

    1. `phases` is checked to be a sequence of `SearchPhase`, and every initial candidate is bound
       (`study.candidate_id`), before the session starts. A typo therefore fails before the baseline is measured.
    2. The session is built and started.
    3. The fits of the initial candidates are prepared together, and each is evaluated with the proposer name
       `"initial"` until the session is exhausted. A candidate that is already in the archive costs nothing.
    4. Each phase's engine runs within the phase's allowance and receives the initial candidates. A phase does not
       run once the session is exhausted. One line is logged when a phase starts and one when it ends, with the
       number of candidates it recorded.
    5. The session is closed, also when a step raises an error.

    An empty `phases` evaluates the initial candidates only. Since each engine reads the archive, a later phase
    begins from what the earlier phases found.

    Args:
        study: The study.
        phases: The search phases, in order.
        store: The store.
        initial_candidates: Candidates to evaluate before the first phase.
        factory: The pipeline factory, or None to build one (released when the session closes).
        max_candidates: The number of candidates the run may record, or None.
        max_generations: The number of rows the run may generate, or None.
        max_seconds: The wall time the run may take from the session's start, or None.
        allow_infeasible: Whether to run a study that fails the feasibility check.

    Returns:
        The study's archive.

    Raises:
        TypeError: If `phases` is not a sequence of `SearchPhase`.
        RecipeError: If an initial candidate does not bind.
    """
    if not isinstance(phases, Sequence) or isinstance(phases, str):
        raise TypeError(
            "phases must be a sequence of SearchPhase, e.g., phases=[SearchPhase(engine=engine)]; got a "
            f"{type(phases).__name__}."
        )
    for phase in phases:
        if not isinstance(phase, SearchPhase):
            raise TypeError(
                "phases must contain SearchPhase objects, e.g., SearchPhase(engine=engine); got a "
                f"{type(phase).__name__}."
            )
    initial_candidates = tuple(initial_candidates)
    for candidate in initial_candidates:
        study.candidate_id(candidate=candidate)
    session = StudySession(
        study=study,
        store=store,
        factory=factory,
        max_candidates=max_candidates,
        max_generations=max_generations,
        max_seconds=max_seconds,
        allow_infeasible=allow_infeasible,
    )
    try:
        session.start()
        session.prepare(candidates=initial_candidates)
        for candidate in initial_candidates:
            if session.exhausted():
                break
            session.evaluate(candidate=candidate, proposer=INITIAL_PROPOSER)
        for index, phase in enumerate(phases):
            if session.exhausted():
                logger.info("The session is exhausted; %d phase(s) do not run.", len(phases) - index)
                break
            name = phase.engine.name
            logger.info("Phase %d (%s) starts.", index, name)
            recorded = session.recorded
            with session.allowance(
                candidates=phase.candidates,
                generations=phase.generations,
                seconds=phase.seconds,
            ):
                phase.engine.run(session=session, initial_candidates=initial_candidates)
            logger.info("Phase %d (%s) ends with %d candidate(s) recorded.", index, name, session.recorded - recorded)
    finally:
        session.close()
    return session.archive


__all__ = ["SearchPhase", "run_study"]
