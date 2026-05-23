"""LatestArchive: retain only the most recent candidate (TextGrad-style)."""
from __future__ import annotations

from typing import Iterable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


class LatestArchive:
    """Trivial archive: keeps only the most recent ingested candidate.

    Suitable for methods that overwrite their candidate in place each step (TextGrad-style backprop optimization).
    `best` and `select_for_mutation` both return the single retained candidate.
    """

    def __init__(self) -> None:
        self._current: Candidate | None = None
        self._traces: list[Trace] = []

    def ingest(
        self,
        candidates: list[Candidate],
        traces_per_candidate: list[list[Trace]],
    ) -> None:
        if not candidates:
            return
        self._current = candidates[-1]
        self._traces = list(traces_per_candidate[-1]) if traces_per_candidate else []

    def select_for_mutation(self) -> Candidate:
        if self._current is None:
            raise RuntimeError("LatestArchive is empty; nothing to select.")
        return self._current

    def best(self) -> Candidate:
        return self.select_for_mutation()

    def members(self) -> Iterable[Candidate]:
        return [self._current] if self._current is not None else []

    def traces_for(self, candidate: Candidate) -> list[Trace]:
        if self._current is not None and candidate.id == self._current.id:
            return list(self._traces)
        return []
