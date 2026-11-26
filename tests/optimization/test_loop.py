"""Tests for the run of a study: initial candidates, phases, and their allowances (`search/loop.py`)."""
import json

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.recipe import RecipeError
from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search.engine import SearchEngine
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.proposers import GridProposer
from steerability.optimization.study import Candidate
from tests.optimization.helpers import FixedProposer, StubFactory, Truth, argument, build_study, prompt_template

TEXTS = tuple(f"text {index}" for index in range(6))


def _truth(recipe, item_id):
    text = argument(recipe, "input_control/system_prompt", "text")
    return Truth(loss=0.01 * TEXTS.index(text), gain=0.02 + 0.02 * TEXTS.index(text))


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


class ViewRecorder(SearchEngine):
    """Records the archive view and the initial candidates it is run with, and proposes nothing."""

    name = "recorder"

    def __init__(self):
        self.views = []
        self.initial = []

    def run(self, *, session, initial_candidates=()):
        self.views.append(session.archive.view())
        self.initial.append(tuple(initial_candidates))


@pytest.fixture()
def store(tmp_path):
    return Store.open(root=tmp_path / "store")


@pytest.fixture()
def study(store):
    return build_study(store=store, truth=_truth, space={"prompt": prompt_template(texts=TEXTS)})


def _run(study, store, phases, **options):
    factory = options.pop("factory", None) or StubFactory(protocol=study.protocol)
    return run_study(study=study, phases=phases, store=store, factory=factory, **options)


def _recorded(store):
    """The text and proposer of every recorded candidate, in recording order."""
    rows = store.execute(sql="SELECT assignment, proposer FROM candidates ORDER BY rowid").fetchall()
    return [(json.loads(assignment)["prompt.text"], proposer) for assignment, proposer in rows]


class TestOnePhase:
    def test_the_run_stops_when_the_session_is_exhausted(self, study, store):
        proposer = FixedProposer(study=study, candidates=[_prompt(text) for text in TEXTS])
        archive = _run(study, store, [SearchPhase(engine=proposer)], max_candidates=3)
        assert len(archive.view()) == 3
        assert [result.reason == "budget" for _, result in proposer.told] == [False, False, False, True]
        assert len(proposer.candidates) == 2

    def test_the_run_stops_when_the_proposer_has_nothing_to_propose(self, study, store):
        proposer = FixedProposer(study=study, candidates=[_prompt("text 0"), _prompt("text 1")])
        archive = _run(study, store, [SearchPhase(engine=proposer)])
        assert len(archive.view()) == 2 and len(proposer.told) == 2

    def test_the_run_stops_after_a_batch_that_records_nothing(self, study, store):
        batch = [_prompt(text) for text in TEXTS[:4]]
        proposer = FixedProposer(study=study, candidates=[*batch, *batch, _prompt("text 5")])
        archive = _run(study, store, [SearchPhase(engine=proposer)])
        assert len(archive.view()) == 4 and len(proposer.told) == 8
        assert proposer.candidates == [_prompt("text 5")]

    def test_a_proposer_refuses_the_session_of_another_study(self, study, store, tmp_path):
        other_store = Store.open(root=tmp_path / "other")
        other = build_study(
            store=other_store,
            truth=_truth,
            space={"prompt": prompt_template(texts=TEXTS)},
            name="other",
        )
        session = StudySession(study=other, store=other_store, factory=StubFactory(protocol=other.protocol))
        session.start()
        with pytest.raises(ValueError, match="study"):
            GridProposer(study=study).run(session=session)


class TestPhases:
    def test_the_first_phase_stops_at_its_allowance_and_the_second_sees_its_results(self, study, store):
        first = FixedProposer(study=study, candidates=[_prompt(text) for text in TEXTS])
        recorder = ViewRecorder()
        _run(study, store, [SearchPhase(engine=first, candidates=2), SearchPhase(engine=recorder)])
        assert _recorded(store) == [("text 0", "fixed"), ("text 1", "fixed")]
        assert [entry.values["prompt.text"] for entry in recorder.views[0]] == ["text 0", "text 1"]

    def test_initial_candidates_are_evaluated_first_and_recorded_as_initial(self, study, store):
        initial = [_prompt("text 3"), _prompt("text 1")]
        proposer = FixedProposer(study=study, candidates=[_prompt("text 1"), _prompt("text 0")])
        recorder = ViewRecorder()
        phases = [SearchPhase(engine=proposer), SearchPhase(engine=recorder)]
        _run(study, store, phases, initial_candidates=initial)
        assert _recorded(store) == [("text 3", "initial"), ("text 1", "initial"), ("text 0", "fixed")]
        assert recorder.initial == [tuple(initial)]

    def test_without_phases_only_the_initial_candidates_are_evaluated(self, study, store):
        archive = _run(study, store, [], initial_candidates=[_prompt("text 2")])
        assert [entry.values["prompt.text"] for entry in archive.view()] == ["text 2"]

    def test_an_initial_candidate_that_does_not_bind_raises_before_any_model_work(self, study, store):
        factory = StubFactory(protocol=study.protocol)
        with pytest.raises(RecipeError, match="outside"):
            _run(study, store, [], initial_candidates=[_prompt("not a text")], factory=factory)
        assert factory.steered_recipes == [] and factory.prepared == []
        assert not any(benchmark.calls for benchmark in study.benchmarks)

    def test_a_bare_engine_and_a_positional_argument_raise_a_type_error(self, study, store):
        engine = GridProposer(study=study)
        with pytest.raises(TypeError, match="sequence of SearchPhase"):
            _run(study, store, engine)
        with pytest.raises(TypeError, match="SearchPhase objects"):
            _run(study, store, [engine])
        with pytest.raises(TypeError):
            run_study(study, [SearchPhase(engine=engine)], store=store)
        with pytest.raises(TypeError, match="SearchEngine"):
            SearchPhase(engine="grid")
        with pytest.raises(ValueError, match="generations"):
            SearchPhase(engine=engine, generations=-1)
