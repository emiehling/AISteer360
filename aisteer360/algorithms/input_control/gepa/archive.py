"""PerInstanceParetoArchive: per-task Pareto frontier with coverage-weighted selection."""
from __future__ import annotations

import random
from typing import Iterable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


class PerInstanceParetoArchive:
    """Pareto frontier across per-task scores.

    Semantically distinct from `common.archives.ParetoArchive`:

    - `ParetoArchive` (Phase 3): each candidate's score is a dict `{dimension: scalar}`; "dominated" means `<=` on
      every dimension and `<` on at least one. Use for genuine multi-objective optimization (truthfulness vs.
      informativeness, etc.).

    - `PerInstanceParetoArchive` (this class): each candidate's score is a dict `{instance_id: scalar}` across many
      task instances with a single objective. The frontier contains candidates that achieve the maximum score on at
      least one instance (DSPy GEPA's selection strategy). Selection samples from the frontier with probability
      proportional to coverage (number of instances on which the candidate is top).

    Instance IDs are read from `example.get("id", str(example_index))`. Users wanting stable IDs across re-runs (e.g.,
    for Benchmark resume) should supply explicit `"id"` keys in their data examples.

    Args:
        rng_seed: Optional seed for `select_for_mutation` reproducibility.
    """

    def __init__(self, rng_seed: int | None = None) -> None:
        self._rng = random.Random(rng_seed)
        self._candidates: dict[str, Candidate] = {}
        self._traces: dict[str, list[Trace]] = {}
        self._scores: dict[str, dict[str, float]] = {}
        self._frontier: list[Candidate] = []
        self._coverage: dict[str, int] = {}

    def ingest(
        self,
        candidates: list[Candidate],
        traces_per_candidate: list[list[Trace]],
    ) -> None:
        for cand, traces in zip(candidates, traces_per_candidate):
            self._candidates[cand.id] = cand
            self._traces[cand.id] = list(traces)
            self._scores[cand.id] = self._build_score_map(traces)
        self._recompute_frontier()

    def select_for_mutation(self) -> Candidate:
        if not self._frontier:
            raise RuntimeError("PerInstanceParetoArchive is empty; nothing to select.")
        weights = [self._coverage.get(c.id, 1) for c in self._frontier]
        return self._rng.choices(self._frontier, weights=weights, k=1)[0]

    def best(self) -> Candidate:
        """Return the frontier member with the highest mean score across instances; tie-broken by id."""
        if not self._frontier:
            raise RuntimeError("PerInstanceParetoArchive is empty.")
        return max(
            self._frontier,
            key=lambda c: (self._mean_score(c.id), c.id),
        )

    def members(self) -> Iterable[Candidate]:
        return list(self._frontier)

    def traces_for(self, candidate: Candidate) -> list[Trace]:
        return list(self._traces.get(candidate.id, []))

    @staticmethod
    def _build_score_map(traces: list[Trace]) -> dict[str, float]:
        score_map: dict[str, float] = {}
        for idx, trace in enumerate(traces):
            example = trace.metadata.get("task_example", {}) if trace.metadata else {}
            instance_id = str(example.get("id", str(idx)))
            score = trace.score
            if isinstance(score, dict):
                raise TypeError(
                    "PerInstanceParetoArchive expects scalar Trace.score values; got dict. "
                    "Use ParetoArchive for multi-dimensional scores."
                )
            score_map[instance_id] = float(score)
        return score_map

    def _recompute_frontier(self) -> None:
        all_instance_ids: set[str] = set()
        for score_map in self._scores.values():
            all_instance_ids.update(score_map.keys())

        coverage: dict[str, int] = {cid: 0 for cid in self._candidates}
        for instance_id in all_instance_ids:
            top_score: float | None = None
            top_candidate_ids: list[str] = []
            for cid, score_map in self._scores.items():
                s = score_map.get(instance_id, 0.0)
                if top_score is None or s > top_score:
                    top_score = s
                    top_candidate_ids = [cid]
                elif s == top_score:
                    top_candidate_ids.append(cid)
            for cid in top_candidate_ids:
                coverage[cid] += 1

        self._coverage = coverage
        self._frontier = sorted(
            (self._candidates[cid] for cid, count in coverage.items() if count > 0),
            key=lambda c: c.id,
        )

    def _mean_score(self, candidate_id: str) -> float:
        scores = self._scores.get(candidate_id, {})
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)
