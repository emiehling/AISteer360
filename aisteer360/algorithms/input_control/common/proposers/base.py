"""Proposer: generates new candidates from a parent and archive state."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace

if TYPE_CHECKING:
    from aisteer360.algorithms.input_control.common.archives.base import Archive


@runtime_checkable
class Proposer(Protocol):
    """Structural type for the train-time proposal component.

    A Proposer reads a parent candidate and the archive's state and produces one or more new candidates. It
    encapsulates both the diagnostic step ("what went wrong with the parent") and the generative step ("here is a
    revised candidate") because in practice they happen together -- usually in a single LLM call for reflection-based
    proposers, or a single gradient step for parameter-based proposers.
    """

    def propose(
        self,
        parent: Candidate,
        archive: "Archive",
        traces: list[Trace] | None = None,
    ) -> list[Candidate]:
        """Generate new candidates derived from `parent`.

        Args:
            parent: The candidate to mutate. The Proposer typically reads `archive.traces_for(parent)` to ground its
                proposals.
            archive: The full archive, exposing membership and trace history. Proposers may use it for merge-style
                mutations (combining multiple archive members) or for diversity-aware sampling.
            traces: When provided, used as the reflection signal in place of `archive.traces_for(parent)`. Useful for
                proposers that need to operate on fresh rollouts distinct from the archive's stored evaluation traces
                (GEPA reflects on minibatch traces sampled per iteration). When None, the proposer falls back to the
                archive's traces for the parent.

        Returns:
            New candidates. Empty list is permissible (the loop handles it).
        """
        ...
