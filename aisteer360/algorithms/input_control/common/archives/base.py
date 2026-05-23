"""Archive: stores candidates and their traces; manages selection."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


@runtime_checkable
class Archive(Protocol):
    """Structural type for the train-time storage and selection component.

    The Archive holds candidates and their traces, decides which candidate to surface to the Proposer for the next
    iteration, and exposes the current best candidate when the loop ends. Selection policy is inseparable from storage
    shape (a Pareto archive's "best" is different from a latest-only archive's "best"), so they live in one component.
    """

    def ingest(
        self,
        candidates: list[Candidate],
        traces_per_candidate: list[list[Trace]],
    ) -> None:
        """Add candidates and their traces to the archive.

        Args:
            candidates: New candidates to consider.
            traces_per_candidate: Element-wise parallel to `candidates`. `traces_per_candidate[i]` is the list of
                Traces for `candidates[i]`.
        """
        ...

    def select_for_mutation(self) -> Candidate:
        """Return a candidate the Proposer should mutate next.

        The selection policy is archive-specific (latest-only, non-dominated-uniform, surrogate-driven exploration,
        etc.).
        """
        ...

    def best(self) -> Candidate:
        """Return the current best candidate. Definition is archive-specific."""
        ...

    def members(self) -> Iterable[Candidate]:
        """Iterate over all currently retained candidates."""
        ...

    def traces_for(self, candidate: Candidate) -> list[Trace]:
        """Return the traces associated with a candidate, or an empty list."""
        ...
