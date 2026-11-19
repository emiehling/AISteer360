"""Search over steering pipelines under accuracy constraints, with certification on sealed items.

A `Study` defines the search space (a template per structure), the behavior benchmark to maximize, the capability
benchmarks whose accuracy must stay within a margin of the unsteered model, and the split of every benchmark's items
into a search pool and a sealed pool. The subpackages are layered:

- `measurement`: pools, the statistics of paired differences, and the evidence decisions are made on.
- `evaluation`: the plan, stages, calibration, and confirmation of a candidate, the archive of results, the
  `StudySession` (the one entry to evaluation), and the `EvaluationService` that serves a session over a socket.
- `search`: proposers, `run_study`, and the OpenEvolve adapter (`run_openevolve`, with the `optimize` extra).
- `release`: certification of the finalists on the sealed items and the bundles of certified points.
"""
from steerability.optimization.measurement.pools import PoolSplit, split_items
from steerability.optimization.study import Candidate, Constraint, Settings, Study, StudyError, StudyInfeasibleError

__all__ = [
    "Candidate",
    "Constraint",
    "PoolSplit",
    "Settings",
    "Study",
    "StudyError",
    "StudyInfeasibleError",
    "split_items",
]
