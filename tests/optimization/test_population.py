"""Tests for the initial population of a search engine (`search/population.py`)."""
import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.recipe import Choice, Interval, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.optimization import Candidate, optional, structures
from steerability.optimization.evaluation.archive import ArchiveEntry, ArchiveView
from steerability.optimization.search.population import initial_population
from tests.optimization.helpers import PAIRS, RAW, Truth, build_study

PROMPT = entry(SystemPrompt, name="prompt", text=Choice(values=("be brief", "be formal", "be kind")))
CAA_ENTRY = entry(
    CAA,
    data=PAIRS,
    train_spec=RAW,
    layer_id=Choice(values=(1, 2, 3)),
    multiplier=Interval(low=0.0, high=3.0),
)


@pytest.fixture()
def study(tmp_path):
    return build_study(
        store=Store.open(root=tmp_path / "store"),
        truth=lambda recipe, item_id: Truth(),
        space=structures(optional(PROMPT), optional(CAA_ENTRY)),
    )


def _entry(study, structure, values, *, fitness=0.1, status="accepted", candidate_id=None):
    if candidate_id is None:
        candidate_id = study.candidate_id(candidate=Candidate(structure=structure, values=values))
    accepted = status == "accepted"
    return ArchiveEntry(
        candidate_id=candidate_id,
        structure=structure,
        values=values,
        status=status,
        stage=5 if accepted else 2,
        reason=None if accepted else "no_behavior_gain",
        fitness=fitness if accepted else 0.0,
        summary={},
    )


def _view(*entries):
    return ArchiveView(entries=entries)


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


def _texts(candidates):
    return [candidate.values["prompt.text"] for candidate in candidates]


def test_accepted_entries_of_a_structure_come_first_by_fitness_and_fallbacks_are_not_fillers(study):
    view = _view(
        _entry(study, "prompt", {"prompt.text": "be brief"}, fitness=0.1),
        _entry(study, "prompt", {"prompt.text": "be kind"}, fitness=0.3),
        _entry(study, "prompt", {"prompt.text": "be formal"}, status="rejected"),
    )
    population = initial_population(study=study, view=view, per_structure=3)
    assert list(population) == ["prompt", "caa", "prompt+caa"]
    assert _texts(population["prompt"]) == ["be kind", "be brief"]
    assert population["caa"] == [study.default_candidate(structure="caa")]
    assert population["prompt+caa"] == [study.default_candidate(structure="prompt+caa")]


def test_initial_candidates_follow_the_accepted_entries_also_when_rejected(study):
    view = _view(
        _entry(study, "prompt", {"prompt.text": "be kind"}, fitness=0.3),
        _entry(study, "prompt", {"prompt.text": "be formal"}, status="rejected"),
    )
    layer = Candidate(structure="caa", values={"caa.layer_id": 2})
    initial = [_prompt("be formal"), layer, _prompt("be kind")]
    population = initial_population(study=study, view=view, initial_candidates=initial, per_structure=3)
    assert _texts(population["prompt"]) == ["be kind", "be formal"]
    assert population["caa"] == [layer]


def test_a_structure_of_two_controls_is_composed_from_the_best_entries_of_other_structures(study):
    view = _view(
        _entry(study, "prompt", {"prompt.text": "be brief"}, fitness=0.1),
        _entry(study, "prompt", {"prompt.text": "be formal"}, fitness=0.4),
        _entry(study, "caa", {"caa.layer_id": 3}, fitness=0.2),
        _entry(study, "caa", {"caa.layer_id": 1}, fitness=0.05),
    )
    population = initial_population(study=study, view=view, per_structure=2, structures=["prompt+caa"])
    composed = Candidate(structure="prompt+caa", values={"prompt.text": "be formal", "caa.layer_id": 3})
    assert population == {"prompt+caa": [composed]}


def test_the_default_candidate_is_the_fallback_when_a_composed_candidate_does_not_bind(study):
    view = _view(
        _entry(study, "caa", {"caa.layer_id": 7}, fitness=0.2, candidate_id="from another space"),
        _entry(study, "prompt", {"prompt.text": "be formal"}, fitness=0.4),
    )
    population = initial_population(study=study, view=view)
    assert _texts(population["prompt"]) == ["be formal"]
    assert population["caa"] == [study.default_candidate(structure="caa")]
    assert population["prompt+caa"] == [study.default_candidate(structure="prompt+caa")]


def test_duplicates_are_taken_once_and_per_structure_caps_a_structure(study):
    view = _view(
        _entry(study, "prompt", {"prompt.text": "be brief"}, fitness=0.1),
        _entry(study, "prompt", {"prompt.text": "be kind"}, fitness=0.3),
        _entry(study, "prompt", {"prompt.text": "be formal"}, fitness=0.2),
    )
    initial = [_prompt("be kind"), _prompt("be kind")]
    population = initial_population(
        study=study,
        view=view,
        initial_candidates=initial,
        per_structure=2,
        structures=["prompt"],
    )
    assert _texts(population["prompt"]) == ["be kind", "be formal"]
    alone = initial_population(
        study=study,
        view=_view(),
        initial_candidates=initial,
        per_structure=3,
        structures=["prompt"],
    )
    assert _texts(alone["prompt"]) == ["be kind"]
    with pytest.raises(ValueError, match="per_structure"):
        initial_population(study=study, view=view, per_structure=0)
