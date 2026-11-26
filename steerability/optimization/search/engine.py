"""The interface of a search engine, i.e., anything that proposes a study's candidates and has a session evaluate them.

An engine comes in one of two shapes. In the ask/tell shape (`BaseProposer`), the toolkit runs the loop and asks
the engine for candidates. In the evaluator shape (`OpenEvolveEngine`), the tool runs its own loop and calls back for
a score through an `EvaluationService` that the engine starts on the session. Either way, `run_study` hands each
engine the started session of the run, and the engine stops when the session is exhausted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.study import Candidate, Study


class SearchEngine(ABC):
    """Something that proposes candidates for a study and has a session evaluate them.

    An engine relies on the following contract with the session it is given:

    - The session is started. The engine does not start or close it.
    - Every evaluation goes through `session.screen`, `session.finish`, or `session.evaluate`, either directly or
      through an `EvaluationService` that the engine starts and stops. The engine passes its `name` as the proposer
      name, which the archive records with each candidate.
    - The engine learns what is known from `session.archive.view()` and from nothing else. No item-level data
      reaches it.
    - The engine returns when `session.exhausted()` is true. It releases whatever it started (processes, the
      service) even when it raises an error.
    - The engine may keep a private working copy of what it has learned (a population, a surrogate model), and the
      archive stays the authoritative record.
    - The `initial_candidates` of `run` are the ones given to `run_study`, which has evaluated them before the first
      engine runs.

    Attributes:
        name: The engine's name, recorded with each candidate it proposes.
    """

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        *,
        session: StudySession,
        initial_candidates: Sequence[Candidate] = (),
    ) -> None:
        """Propose and evaluate candidates until the session is exhausted or the engine has nothing new.

        Args:
            session: The started session of the run.
            initial_candidates: The candidates given to `run_study`, already evaluated.
        """


def select_structures(*, study: Study, structures: Sequence[str] | None) -> tuple[str, ...]:
    """The structures of a study that an engine searches, in the order of `study.space`.

    Args:
        study: The study.
        structures: The names of the structures to search, or None for every structure of the study.

    Returns:
        The selected structure names.

    Raises:
        ValueError: If `structures` is empty or gives a name that is not a structure of the study.
    """
    if structures is None:
        return tuple(study.space)
    selected = set(structures)
    if not selected:
        raise ValueError("structures is empty; give at least one structure name, or None for every structure.")
    unknown = sorted(selected - set(study.space))
    if unknown:
        raise ValueError(f"Structures {unknown} are not in the study's space; its structures are {list(study.space)}.")
    return tuple(structure for structure in study.space if structure in selected)


__all__ = ["SearchEngine", "select_structures"]
