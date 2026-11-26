"""Search over steering pipelines under accuracy constraints, with certification on held-out items.

A `Study` defines the search space (a template per structure, which `structures` expands from optional and
exclusive controls), the behavior benchmark to maximize, the capability benchmarks whose accuracy must stay within a
margin of the unsteered model, and the split of every benchmark's items into a search pool and a held-out pool. The
subpackages are layered:

- `measurement`: pools, the statistics of paired differences, and the evidence decisions are made on.
- `evaluation`: the plan, stages, calibration, and confirmation of a candidate, the archive of results, the
  `StudySession` (the one entry to evaluation), and the `EvaluationService` that serves a session over a socket.
- `search`: the search engines (the grid and random proposers, and `OpenEvolveEngine` with the `optimize` extra),
  the initial population of an engine, and `run_study`, which runs a study's search phases on one session.
- `release`: certification of the finalists on the held-out items and the bundles of certified points.
"""
from steerability.optimization.measurement.pools import PoolSplit, split_items
from steerability.optimization.space import Slot, one_of, optional, structures
from steerability.optimization.study import Candidate, Constraint, Settings, Study, StudyError, StudyInfeasibleError

__all__ = [
    "Candidate",
    "Constraint",
    "PoolSplit",
    "Settings",
    "Slot",
    "Study",
    "StudyError",
    "StudyInfeasibleError",
    "one_of",
    "optional",
    "split_items",
    "structures",
]
