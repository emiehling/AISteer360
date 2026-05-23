"""ParetoArchive: non-dominated frontier (GEPA-style)."""
from __future__ import annotations

import random
from typing import Iterable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


class ParetoArchive:
    """Non-dominated frontier across multiple quality dimensions.

    Each candidate is summarized by a dict `{dimension: scalar}` obtained by averaging trace scores across the data. A
    candidate is retained if no other retained candidate dominates it on every dimension.

    Args:
        score_keys: Names of the dimensions to compare on. The Scorer's traces must produce dict-valued `score` fields
            covering these keys. If a key is missing for a trace, it contributes 0 to the mean.
        rng_seed: Optional seed for `select_for_mutation` reproducibility.
    """

    def __init__(
        self,
        score_keys: list[str],
        rng_seed: int | None = None,
    ) -> None:
        if not score_keys:
            raise ValueError("ParetoArchive requires at least one score_key.")
        self.score_keys = list(score_keys)
        self._rng = random.Random(rng_seed)
        self._frontier: list[Candidate] = []
        self._traces: dict[str, list[Trace]] = {}
        self._summaries: dict[str, dict[str, float]] = {}

    def ingest(
        self,
        candidates: list[Candidate],
        traces_per_candidate: list[list[Trace]],
    ) -> None:
        for cand, traces in zip(candidates, traces_per_candidate):
            self._traces[cand.id] = list(traces)
            self._summaries[cand.id] = self._summarize(traces)

        all_candidates = list(self._frontier) + list(candidates)
        seen_ids: set[str] = set()
        unique: list[Candidate] = []
        for cand in all_candidates:
            if cand.id in seen_ids:
                continue
            seen_ids.add(cand.id)
            unique.append(cand)

        self._frontier = [
            c for c in unique
            if not any(self._dominates(other.id, c.id) for other in unique if other.id != c.id)
        ]

    def select_for_mutation(self) -> Candidate:
        if not self._frontier:
            raise RuntimeError("ParetoArchive is empty; nothing to select.")
        return self._rng.choice(self._frontier)

    def best(self) -> Candidate:
        """Return the frontier member with the highest sum of scores; tie-broken by id."""
        if not self._frontier:
            raise RuntimeError("ParetoArchive is empty.")
        return max(
            self._frontier,
            key=lambda c: (sum(self._summaries[c.id].values()), c.id),
        )

    def members(self) -> Iterable[Candidate]:
        return list(self._frontier)

    def traces_for(self, candidate: Candidate) -> list[Trace]:
        return list(self._traces.get(candidate.id, []))

    def _summarize(self, traces: list[Trace]) -> dict[str, float]:
        if not traces:
            return {k: 0.0 for k in self.score_keys}
        summary = {k: 0.0 for k in self.score_keys}
        for t in traces:
            if isinstance(t.score, dict):
                for k in self.score_keys:
                    summary[k] += float(t.score.get(k, 0.0))
            else:
                raise TypeError(
                    "ParetoArchive requires Trace.score to be a dict; got scalar. "
                    "Use a Scorer that emits multi-dimensional scores."
                )
        return {k: v / len(traces) for k, v in summary.items()}

    def _dominates(self, a_id: str, b_id: str) -> bool:
        """True iff `a` dominates `b`: `a >= b` on all keys, `a > b` on at least one."""
        a = self._summaries[a_id]
        b = self._summaries[b_id]
        ge_all = all(a[k] >= b[k] for k in self.score_keys)
        gt_any = any(a[k] > b[k] for k in self.score_keys)
        return ge_all and gt_any
