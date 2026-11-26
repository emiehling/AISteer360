"""Tests for the evaluation service and the in-process study loop (`evaluation/service.py`, `search/loop.py`)."""
import dataclasses
import json
from multiprocessing.connection import Client

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.service import EvaluationService, call, final_metrics
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import parse, program_text
from steerability.optimization.study import Candidate
from tests.optimization.helpers import (
    FixedProposer,
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
        assert first == {"metrics": {"stage1_passed": 1.0}, "artifacts": {"status": "screened", "stage": "3"}}
        second = service.handle({"op": "stage2", "program": program})
        whole = _service(full).handle({"op": "full", "program": program})
        assert second == whole
        assert set(whole["metrics"]) == {"combined_score", "gain", "accepted", "level"}
        candidate_id = staged.study.candidate_id(candidate)
        assert _comparable(staged.archive.latest(candidate_id)) == _comparable(full.archive.latest(candidate_id))


def test_a_rejection_in_stage_one_reports_its_final_metrics(tmp_path):
    session = _session(Store.open(tmp_path / "store"), space={"caa": caa_template()})
    session.study.objective.set_truth(lambda recipe, item_id: Truth())
    reply = _service(session).handle({"op": "stage1", "program": program_text(session.study, Candidate("caa", {}))})
    assert reply["metrics"] == {"stage1_passed": 0.0, "combined_score": 0.0, "gain": 0.0, "accepted": 0.0,
                                "level": 0.0}
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
    assert (service.programs, service.unparsed, service.unbound) == (2, 1, 1)


def test_estimates_read_the_archive(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    service = _service(session)
    candidate = Candidate(structure="prompt", values={"prompt.text": "text 1"})
    program = program_text(study=session.study, candidate=candidate)
    identity = {"candidate_id": session.study.candidate_id(candidate=candidate), "structure": "prompt"}
    unknown = {"known": False, "accepted": False, "fitness": 0.0}
    assert service.handle(request={"op": "estimate", "program": program}) == {**unknown, **identity}
    service.handle({"op": "full", "program": program})
    estimate = service.handle({"op": "estimate", "program": program})
    assert estimate["known"] and estimate["accepted"] and estimate["fitness"] == pytest.approx(0.04)
    assert {key: estimate[key] for key in identity} == identity
    unbound = program_text(study=session.study, candidate=Candidate(structure="prompt", values={"prompt.text": "x"}))
    for program in ("not a program", unbound):
        assert service.handle(request={"op": "estimate", "program": program}) == {
            **unknown, "candidate_id": None, "structure": None,
        }
    assert service.handle({"op": "ping"}) == {}
    assert "error" in service.handle({"op": "unknown"})


def _full(service, candidate):
    program = program_text(study=service.session.study, candidate=candidate)
    return service.handle(request={"op": "full", "program": program})


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


def test_final_and_invalid_replies_have_a_level_clipped_to_the_unit_interval(tmp_path, space):
    session = _session(Store.open(root=tmp_path / "store"), space=space)
    service = _service(session)
    accepted = _full(service, _prompt("text 2"))
    assert accepted["metrics"]["accepted"] == 1.0 and accepted["metrics"]["level"] == pytest.approx(0.4)
    rejected = _full(service, _prompt("text 5"))
    assert rejected["metrics"]["accepted"] == 0.0 and rejected["metrics"]["level"] == pytest.approx(1.0)
    assert service.handle(request={"op": "full", "program": "not a program"})["metrics"]["level"] == 0.0
    result = session.archive.latest(candidate_id=session.study.candidate_id(candidate=_prompt("text 5")))
    assert final_metrics(result=dataclasses.replace(result, summary={"level": 1.7}))["level"] == 1.0
    assert final_metrics(result=dataclasses.replace(result, summary={"level": -0.3}))["level"] == 0.0
    assert final_metrics(result=dataclasses.replace(result, summary={}))["level"] == 0.0


def test_final_replies_list_the_recent_rejections_of_their_structure_newest_first(tmp_path, space):
    session = _session(Store.open(root=tmp_path / "store"), space=space, monotone={"caa.multiplier"})
    service = EvaluationService(
        session=session,
        proposer="service",
        parse=parse,
        stage_deadline_seconds=600,
        digest=2,
    )
    for text in ("text 5", "text 4", "text 0"):
        reply = _full(service, _prompt(text))
    lines = reply["artifacts"]["recent_rejections"].splitlines()
    assert [line.partition(" -> ")[0] for line in lines] == ['{"prompt.text":"text 4"}', '{"prompt.text":"text 5"}']
    assert lines[0].startswith('{"prompt.text":"text 4"} -> not_confirmed, level 0.8')
    assert "recent_rejections" not in _full(service, Candidate(structure="caa", values={}))["artifacts"]


def test_the_digest_cuts_long_values_and_lists_nothing_when_it_is_zero(tmp_path):
    texts = ("a" * 400, "b", "c")
    session = _session(Store.open(root=tmp_path / "store"), space={"prompt": prompt_template(texts=texts)})
    session.study.objective.set_truth(truth=lambda recipe, item_id: Truth())
    service = _service(session)
    _full(service, _prompt(texts[0]))
    lines = _full(service, _prompt("b"))["artifacts"]["recent_rejections"].splitlines()
    values, _, reason = next(line for line in lines if line.startswith('{"prompt.text":"aa')).partition(" -> ")
    assert len(values) == 300 and values.endswith("...") and reason.startswith("no_behavior_gain")
    silent = EvaluationService(session=session, proposer="service", parse=parse, stage_deadline_seconds=600, digest=0)
    assert "recent_rejections" not in _full(silent, _prompt("c"))["artifacts"]


def test_replies_list_the_last_three_programs_that_could_not_be_evaluated(tmp_path, space):
    session = _session(Store.open(root=tmp_path / "store"), space=space)
    service = _service(session)
    unparsed = service.handle(request={"op": "full", "program": "not a program"})["artifacts"]["recent_unevaluated"]
    assert unparsed.startswith("A program needs exactly one block")
    for index in range(3):
        reply = _full(service, _prompt(f"{index}" + "z" * 300))
    lines = reply["artifacts"]["recent_unevaluated"].splitlines()
    assert len(lines) == 3 and all(len(line) <= 200 for line in lines)
    assert [line[:15] for line in lines] == [f"prompt.text: '{index}" for index in (2, 1, 0)]
    assert "recent_unevaluated" in _full(service, _prompt("text 1"))["artifacts"]


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
    proposer = FixedProposer(study=study, candidates=candidates)
    run_study(
        study=study,
        phases=[SearchPhase(engine=proposer)],
        store=looped,
        factory=StubFactory(protocol=study.protocol),
    )
    assert len(proposer.told) == 5
    served = _session(Store.open(tmp_path / "served"), space=space)
    service = _service(served)
    for candidate in candidates:
        service.handle({"op": "full", "program": program_text(served.study, candidate)})
    assert served.round_index == 1
    assert _rounds(looped) == _rounds(served.store)
    assert _versions(looped) == _versions(served.store)
