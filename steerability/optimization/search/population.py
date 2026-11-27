"""What a search engine begins from and what it may propose, read from the archive's view.

An engine that keeps a population (OpenEvolve's programs, an agent's initial node) begins from candidates taken from
the archive (`initial_population`). Whatever an earlier phase of a run found is therefore where a later engine begins,
and handing one engine's results to another needs no interface of its own. When no new training may run
(`ArchiveView.fits_open` is false), an engine proposes only the values of a training that have been trained
(`trained_values`, `is_trained`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.identity import canonical_json
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


def trained_values(*, study: Study, view: ArchiveView, structure: str) -> dict[str, list[dict[str, Any]]]:
    """For each training entry of a structure, the values of its proposer axes under which it has been trained.

    A training entry is an entry that trains or merges (`Study.training_entries`), and its axes are the structure's
    proposer axes with a member in the entry. A combination of their values is listed when an entry of the view, of
    any structure whose entry of that name is the same (the same method and arguments), was decided after stage 1,
    where the fits run. The helper reads the view only. A listed combination may still need a training when it was
    trained under other structural entries before it, and its candidate then comes back with the reason `budget`.

    Args:
        study: The study.
        view: The archive's view.
        structure: A key of `study.space`.

    Returns:
        Mapping from training entry name to its combinations, in entry order. Each combination maps every member
        path of the entry's axes to a value, and the combinations follow the recording order without repeats. An
        entry without proposer axes has the one empty combination once it has been trained.

    Raises:
        RecipeError: If the structure is unknown.
    """
    training = study.training_entries(structure=structure)
    axes = _training_axes(study=study, structure=structure)
    found: dict[str, list[dict[str, Any]]] = {name: [] for name in training}
    seen: dict[str, set[str]] = {name: set() for name in training}
    for entry in view:
        if entry.stage < 2 or entry.structure not in study.space:
            continue
        other = study.training_entries(structure=entry.structure)
        for name, identity in training.items():
            if other.get(name) != identity:
                continue
            combination: dict[str, Any] = {}
            for axis in axes[name]:
                values = _member_values(axis=axis, values=entry.values)
                if values is None:
                    break
                combination.update(values)
            else:
                key = canonical_json(value=combination)
                if key not in seen[name]:
                    seen[name].add(key)
                    found[name].append(combination)
    return found


def is_trained(*, study: Study, candidate: Candidate, trained: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    """Whether every training entry of a candidate's structure takes a combination listed in `trained`.

    Args:
        study: The study.
        candidate: The candidate.
        trained: The `trained_values` of the candidate's structure.

    Returns:
        True when the candidate's values of each training entry's axes equal one of the entry's combinations.

    Raises:
        RecipeError: If the candidate's structure is unknown.
    """
    for name, axes in _training_axes(study=study, structure=candidate.structure).items():
        combination: dict[str, Any] = {}
        for axis in axes:
            values = _member_values(axis=axis, values=candidate.values)
            if values is None:
                return False
            combination.update(values)
        key = canonical_json(value=combination)
        if all(canonical_json(value=listed) != key for listed in trained.get(name, ())):
            return False
    return True


def _training_axes(*, study: Study, structure: str) -> dict[str, list[Axis]]:
    """The proposer axes of a structure with a member in each training entry, by entry name."""
    proposer = study.proposer_axes(structure=structure).values()
    return {
        name: [axis for axis in proposer if any(path.partition(".")[0] == name for path in axis.members)]
        for name in study.training_entries(structure=structure)
    }


def _member_values(*, axis: Axis, values: Mapping[str, Any]) -> dict[str, Any] | None:
    """The value of every member path of an axis in a candidate's values, or None when the values do not give it."""
    if all(path in values for path in axis.members):
        return {path: values[path] for path in axis.members}
    if axis.key not in values:
        return None
    try:
        return axis.at(fraction=values[axis.key]) if axis.continuous else axis.at_index(index=values[axis.key])
    except (RecipeError, TypeError):
        return None


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


__all__ = ["initial_population", "is_trained", "trained_values"]
