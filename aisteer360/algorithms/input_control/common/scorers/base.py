"""Scorer: runs candidates against task data, produces traces."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


@runtime_checkable
class Scorer(Protocol):
    """Structural type for the train-time scoring component.

    A Scorer runs one or more candidates against a list of task examples and returns one list of Traces per candidate
    (parallel lists). It is the train-time analogue of running a `UseCase` through a `Metric` -- separated out so that
    methods can plug in surrogate scorers (e.g. learned reward models, Bayesian surrogates) without depending on the
    full Benchmark/UseCase stack.

    The Scorer is also where the bridge between an arbitrary `Memory` and the method's `adapt()` lives: TaskLMScorer
    takes an adapter callable `(input_ids, Memory) -> steered_input_ids` and applies it per-candidate.

    Implementations are NOT required to subclass; they only need to satisfy this Protocol.
    """

    def score(
        self,
        candidates: list[Candidate],
        data: list[dict],
    ) -> list[list[Trace]]:
        """Score `candidates` against `data`.

        Args:
            candidates: Candidates to score.
            data: Task examples. Each dict's schema is method-specific; common keys are `input_ids` (Tensor or
                `list[int]`) and `expected` (whatever the Metric needs).

        Returns:
            One list of Traces per candidate. `len(result) == len(candidates)`. For each candidate `i`, `result[i]`
            contains one Trace per element of `data` (or fewer if examples were filtered, with the metadata documenting
            why).
        """
        ...
