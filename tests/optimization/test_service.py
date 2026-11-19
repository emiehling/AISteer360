"""Tests for the evaluation service and the in-process study loop (`evaluation/service.py`, `search/proposer.py`)."""
import json
from multiprocessing.connection import Client

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.service import EvaluationService, call
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search.enumerate import GridProposer, RandomProposer
from steerability.optimization.search.openevolve import parse, program_text
from steerability.optimization.search.proposer import run_study
from steerability.optimization.study import Candidate
from tests.optimization.helpers import (
    StubFactory,
    Truth,
    argument,
    build_study,
    caa_template,
    multiplier,
    prompt_template,
)

TEXTS = tuple(f"text {index}" for index in range(6))


def _truth(recipe, item_id):
    value = multiplier(recipe)
    if value is not None:
        return Truth(loss=0.0055 * value * value, gain=0.05 * value)
    text = argument(recipe, "input_control/system_prompt", "text")
    return Truth(loss=0.01 * TEXTS.index(text), gain=0.02 + 0.02 * TEXTS.index(text))


def _session(store, **kwargs):
    study = build_study(store, _truth, **kwargs)
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    return session


def _service(session):
    return EvaluationService(session, proposer="service", parse=parse, stage_deadline_seconds=600)


def _comparable(result):
    data = result.to_json()
    data.pop("seconds")
    return data


@pytest.fixture()
def space():
    return {"caa": caa_template(), "prompt": prompt_template(TEXTS)}


def test_stage_one_then_stage_two_equals_full(tmp_path, space):
    staged = _session(Store.open(tmp_path / "staged"), space=space, monotone={"caa.multiplier"})
    full = _session(Store.open(tmp_path / "full"), space=space, monotone={"caa.multiplier"})
    for candidate in (Candidate("caa", {}), Candidate("prompt", {"prompt.text": "text 2"})):
        program = program_text(staged.study, candidate)
        service = _service(staged)
        first = service.handle({"op": "stage1", "program": program})
        assert first["metrics"] == {"stage1_passed": 1.0}
        second = service.handle({"op": "stage2", "program": program})
        whole = _service(full).handle({"op": "full", "program": program})
        assert second == whole
        assert set(whole["metrics"]) == {"combined_score", "gain", "accepted"}
        candidate_id = staged.study.candidate_id(candidate)
        assert _comparable(staged.archive.latest(candidate_id)) == _comparable(full.archive.latest(candidate_id))


def test_a_rejection_in_stage_one_reports_its_final_metrics(tmp_path):
    session = _session(Store.open(tmp_path / "store"), space={"caa": caa_template()})
    session.study.objective.set_truth(lambda recipe, item_id: Truth())
    reply = _service(session).handle({"op": "stage1", "program": program_text(session.study, Candidate("caa", {}))})
    assert reply["metrics"] == {"stage1_passed": 0.0, "combined_score": 0.0, "gain": 0.0, "accepted": 0.0}
    assert (reply["artifacts"]["reason"], reply["artifacts"]["stage"]) == ("no_behavior_gain", "2")


def test_an_unparsed_program_is_invalid_and_not_recorded(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    service = _service(session)
    reply = service.handle({"op": "full", "program": "not a program"})
    assert reply["metrics"]["combined_score"] == 0.0
    assert reply["artifacts"]["reason"] == "invalid" and "block" in reply["artifacts"]["error"]
    unbound = program_text(session.study, Candidate("prompt", {"prompt.text": "nothing"}))
    reply = service.handle({"op": "stage1", "program": unbound})
    assert reply["metrics"]["stage1_passed"] == 0.0 and reply["artifacts"]["reason"] == "invalid"
    assert session.store.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


def test_estimates_read_the_archive(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    service = _service(session)
    program = program_text(session.study, Candidate("prompt", {"prompt.text": "text 1"}))
    assert service.handle({"op": "estimate", "program": program}) == {"known": False, "accepted": False,
                                                                       "fitness": 0.0}
    service.handle({"op": "full", "program": program})
    estimate = service.handle({"op": "estimate", "program": program})
    assert estimate["known"] and estimate["accepted"] and estimate["fitness"] == pytest.approx(0.04)
    assert service.handle({"op": "ping"}) == {}
    assert "error" in service.handle({"op": "unknown"})


def test_messages_over_the_socket_are_json(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    service = _service(session)
    address, authkey = service.start()
    try:
        assert len(authkey) == 32
        assert call(address, authkey, {"op": "ping"}) == {}
        program = program_text(session.study, Candidate("prompt", {"prompt.text": "text 1"}))
        reply = call(address, authkey, {"op": "full", "program": program})
        assert reply["metrics"]["accepted"] == 1.0
        with Client(address, authkey=authkey) as connection:
            connection.send_bytes(b"not json")
            assert "error" in json.loads(connection.recv_bytes().decode("utf-8"))
    finally:
        service.stop()


class FixedProposer:
    """Proposes a fixed list of candidates in order, once."""

    name = "fixed"

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.told = []

    def ask(self, view, n):
        batch, self.candidates = self.candidates[:n], self.candidates[n:]
        return batch

    def tell(self, candidate, result):
        self.told.append((candidate, result))


def _rounds(store):
    return store.execute("SELECT round_index, batches FROM rounds ORDER BY round_index").fetchall()


def _versions(store):
    return store.execute(
        "SELECT candidate_id, version, round_index FROM results ORDER BY candidate_id, version"
    ).fetchall()


def test_run_study_and_the_service_advance_rounds_identically(tmp_path, space):
    candidates = [Candidate("prompt", {"prompt.text": text}) for text in TEXTS[:5]]
    looped = Store.open(tmp_path / "looped")
    study = build_study(looped, _truth, space=space)
    proposer = FixedProposer(candidates)
    run_study(study, proposer, store=looped, factory=StubFactory(study.protocol))
    assert len(proposer.told) == 5
    served = _session(Store.open(tmp_path / "served"), space=space)
    service = _service(served)
    for candidate in candidates:
        service.handle({"op": "full", "program": program_text(served.study, candidate)})
    assert served.round_index == 1
    assert _rounds(looped) == _rounds(served.store)
    assert _versions(looped) == _versions(served.store)


def test_grid_and_random_proposers(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    grid = GridProposer(session.study)
    first = grid.ask(session.archive.view(), 3)
    assert first == [Candidate("caa", {}), Candidate("prompt", {"prompt.text": "text 0"}),
                     Candidate("prompt", {"prompt.text": "text 1"})]
    assert len(grid.ask(session.archive.view(), 10)) == 4
    assert grid.ask(session.archive.view(), 10) == []
    draws = RandomProposer(session.study, seed=3).ask(session.archive.view(), 5)
    assert draws == RandomProposer(session.study, seed=3).ask(session.archive.view(), 5)
    assert len({session.study.candidate_id(candidate) for candidate in draws}) == len(draws)
    session.evaluate(draws[0])
    assert session.study.candidate_id(draws[0]) not in {
        session.study.candidate_id(candidate) for candidate in RandomProposer(session.study, seed=3).ask(
            session.archive.view(), 5)
    }


def test_the_grid_requires_discrete_proposer_axes(tmp_path):
    session = _session(Store.open(tmp_path / "store"), space={"caa": caa_template()}, searched={"caa.multiplier"})
    with pytest.raises(ValueError, match="continuous"):
        GridProposer(session.study)
