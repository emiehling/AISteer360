"""The initial population of a search engine: for each structure, the candidates it begins from.

An engine that keeps a population (OpenEvolve's programs, an agent's initial node) begins from candidates taken from
the archive. Whatever an earlier phase of a run found is therefore where a later engine begins, and handing one
engine's results to another needs no interface of its own.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import RecipeError
from steerability.optimization.search.engine import select_structures
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.algorithms.core.recipe import Axis
    from steerability.optimization.evaluation.archive import ArchiveEntry, ArchiveView
    from steerability.optimization.study import Study


def initial_population(
    *,
    study: Study,
    view: ArchiveView,
    initial_candidates: Sequence[Candidate] = (),
    per_structure: int = 1,
    structures: Sequence[str] | None = None,
) -> dict[str, list[Candidate]]:
    """For each structure, the candidates an engine begins from, best first; never empty.

    For each structure, in the order of `study.space` (restricted to `structures` when given), candidates are taken
    until there are `per_structure`, from these sources in order:

    1. The accepted entries of the structure in the view, by fitness, highest first (ties by candidate id).
    2. The `initial_candidates` of the structure, in the order given, also when the archive rejected them.
    3. One candidate composed from other structures (e.g., a `prompt+caa` structure begins from the best prompt of
       `prompt` and the best layer of `caa`). Each proposer axis of the structure takes its value from the accepted
       entry with the highest fitness, of any structure, whose values contain the axis (by its key, or by all of its
       member paths). The composed candidate is used only when every axis found a value and it binds.
    4. The structure's default candidate (`Study.default_candidate`).

    Sources 3 and 4 are fallbacks, i.e., they are used only for a structure that has no candidate from sources 1 and
    2, and they contribute one candidate. Candidates are built from an entry's structure and values, an entry whose
    values do not bind is skipped, and candidates with the same `study.candidate_id` are taken once.

    Args:
        study: The study.
        view: The archive's view.
        initial_candidates: The candidates given to `run_study`.
        per_structure: The largest number of candidates per structure.
        structures: The structures, or None for every structure of the study.

    Returns:
        Mapping from structure name to its candidates, best first, in the order of `study.space`.

    Raises:
        ValueError: If `per_structure` is not a positive integer, or `structures` gives a structure that is not in
            the study's space.
    """
    if isinstance(per_structure, bool) or not isinstance(per_structure, int) or per_structure < 1:
        raise ValueError(f"per_structure must be a positive integer; got {per_structure!r}.")
    accepted = view.accepted()
    population: dict[str, list[Candidate]] = {}
    for structure in select_structures(study=study, structures=structures):
        taken: list[Candidate] = []
        seen: set[str] = set()

        def take(*, candidate: Candidate) -> None:
            try:
                candidate_id = study.candidate_id(candidate=candidate)
            except RecipeError:
                return
            if candidate_id not in seen:
                seen.add(candidate_id)
                taken.append(candidate)

        sources = [
            *(Candidate(structure=entry.structure, values=entry.values)
              for entry in accepted if entry.structure == structure),
            *(candidate for candidate in initial_candidates if candidate.structure == structure),
        ]
        for candidate in sources:
            if len(taken) >= per_structure:
                break
            take(candidate=candidate)
        if not taken:
            composed = _composed(study=study, structure=structure, accepted=accepted)
            if composed is not None:
                take(candidate=composed)
        if not taken:
            taken.append(study.default_candidate(structure=structure))
        population[structure] = taken
    return population


def _composed(*, study: Study, structure: str, accepted: Sequence[ArchiveEntry]) -> Candidate | None:
    """The candidate of a structure whose proposer axes take their values from the best accepted entries, or None."""
    values: dict[str, Any] = {}
    for axis in study.proposer_axes(structure=structure).values():
        bound = (_axis_values(axis=axis, values=entry.values) for entry in accepted)
        found = next((item for item in bound if item is not None), None)
        if found is None:
            return None
        values.update(found)
    return Candidate(structure=structure, values=values)


def _axis_values(*, axis: Axis, values: Mapping[str, Any]) -> dict[str, Any] | None:
    """The values of an axis in a candidate's values: by the axis key, or by all of its member paths, or None."""
    if axis.key in values:
        return {axis.key: values[axis.key]}
    if all(path in values for path in axis.members):
        return {path: values[path] for path in axis.members}
    return None


__all__ = ["initial_population"]
