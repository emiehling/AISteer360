"""Tests for the study session: start, limits, deadlines, resuming, and finalists (`evaluation/session.py`)."""
import dataclasses
import json
import logging
import time

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certify
from steerability.optimization.study import Candidate, StudyInfeasibleError
from tests.optimization.helpers import SMALL, StubFactory, Truth, argument, build_study, multiplier, prompt_template

CAA = Candidate("caa", {})


def _quadratic(recipe, item_id):
    value = multiplier(recipe)
    if value is None:
        return Truth(gain=0.1)
    return Truth(loss=0.005 * value * value, gain=0.05 * value)


class Interrupted(BaseException):
    """Stops a call partway."""


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _rows(store, table):
    return store.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _started(study, store, **limits):
    session = StudySession(study, store=store, factory=StubFactory(study.protocol), **limits)
    session.start()
    return session


def test_an_infeasible_study_raises_unless_allowed(store, caplog):
    study = build_study(store, _quadratic, space={"prompt": prompt_template(("a",))}, capability_held_out=200)
    with pytest.raises(StudyInfeasibleError, match="guard band"):
        _started(study, store)
    with caplog.at_level(logging.WARNING):
        session = _started(study, store, allow_infeasible=True)
    assert "whatever its baseline shows" in caplog.text and "at its measured baseline" in caplog.text
    assert session.evaluate(Candidate("prompt", {"prompt.text": "a"})).status == "accepted"


def test_a_margin_of_half_a_point_is_refused_before_the_baseline(store):
    study = build_study(store, _quadratic, margin=0.005)
    with pytest.raises(StudyInfeasibleError, match="whatever its baseline shows"):
        _started(study, store)
    assert not any(benchmark.calls for benchmark in study.benchmarks)


def test_start_logs_the_sizes_of_each_constraint(store, caplog):
    study = build_study(store, _quadratic)
    with caplog.at_level(logging.INFO):
        session = _started(study, store)
    assert session.looks(study.constraints[0].benchmark) == (32, 128, 400)
    assert "last look 400, ladder block 100, looks (32, 128, 400)" in caplog.text


def test_a_budget_rejection_writes_no_row_and_a_later_session_evaluates_in_full(store):
    study = build_study(store, _quadratic, monotone={"caa.multiplier"})
    session = _started(study, store, max_generations=100)
    result = session.evaluate(CAA)
    assert (result.status, result.reason) == ("rejected", "budget")
    assert _rows(store, "results") == 0 and _rows(store, "candidates") == 0
    assert session.exhausted()
    assert session.screen(CAA).result.reason == "budget"
    capability = study.constraints[0].benchmark
    earlier = len(capability.calls)
    measured: dict[str, set[str]] = {}
    for resolved, items, _ in capability.calls:
        measured.setdefault(resolved, set()).update(items)
    later = _started(study, store)
    assert later.evaluate(CAA).status == "accepted"
    for resolved, items, _ in capability.calls[earlier:]:
        assert not set(items) & measured.get(resolved, set())


def test_a_timeout_rejection_writes_no_row(store):
    session = _started(build_study(store, _quadratic, monotone={"caa.multiplier"}), store)
    result = session.evaluate(CAA, deadline=time.monotonic() - 1)
    assert (result.status, result.reason, result.stage) == ("rejected", "timeout", 1)
    assert _rows(store, "results") == 0
    assert session.evaluate(CAA).status == "accepted"


def test_the_candidate_limit_counts_recorded_candidates(store):
    texts = ("a", "b", "c")
    study = build_study(store, _quadratic, space={"prompt": prompt_template(texts)})
    session = _started(study, store, max_candidates=2)
    results = [session.evaluate(Candidate("prompt", {"prompt.text": text})) for text in texts]
    assert [result.status for result in results[:2]] == ["accepted", "accepted"]
    assert results[2].reason == "budget"
    assert session.evaluate(Candidate("prompt", {"prompt.text": "a"})).status == "accepted"


def test_a_restarted_session_resumes_at_the_recorded_round(store):
    texts = tuple(f"text {index}" for index in range(8))
    study = build_study(store, _quadratic, space={"prompt": prompt_template(texts)})
    session = _started(study, store)
    for text in texts[:5]:
        session.evaluate(Candidate("prompt", {"prompt.text": text}))
    assert session.round_index == 1
    session.close()
    again = _started(study, store)
    assert again.round_index == 1
    assert again.archive.first_versions(1) == 1
    for text in texts[5:]:
        again.evaluate(Candidate("prompt", {"prompt.text": text}))
    assert again.round_index == 2
    assert sorted(again.rounds) == [0, 1, 2]


def test_the_session_closes_a_factory_it_built_only(store):
    study = build_study(store, _quadratic)
    factory = StubFactory(study.protocol)
    session = StudySession(study, store=store, factory=factory)
    session.close()
    assert not factory.released
    with pytest.raises(RuntimeError, match="start"):
        session.screen(CAA)


def test_finalists_drop_points_below_min_gain_and_are_ordered_by_level(store):
    truths = {"safe": Truth(gain=0.1), "middle": Truth(loss=0.02, gain=0.15), "edge": Truth(loss=0.03, gain=0.2)}
    study = build_study(store, _quadratic, space={"prompt": prompt_template(("low", *truths))},
                        settings=dataclasses.replace(SMALL, round_candidates=16))
    session = _started(study, store)
    ladder = set(session.batches(0)[study.objective.id].ladder)

    def truth(recipe, item_id):
        text = argument(recipe, "input_control/system_prompt", "text")
        if text == "low":
            return Truth(gain=0.1 if item_id in ladder else 0.005)
        return truths[text]

    for benchmark in study.benchmarks:
        benchmark.set_truth(truth)
    results = {text: session.evaluate(Candidate("prompt", {"prompt.text": text})) for text in ("low", *truths)}
    assert all(result.status == "accepted" for result in results.values())
    assert results["low"].fitness == pytest.approx(0.005)
    finalists = session.finalists()
    chosen = {finalist.result.candidate_id: finalist for finalist in finalists}
    ids = {text: result.candidate_id for text, result in results.items()}
    assert ids["low"] not in chosen
    assert list(chosen) == [ids["safe"], ids["edge"]]
    assert [finalist.level for finalist in finalists] == pytest.approx([0.0, 0.6])
    assert all(finalist.fresh for finalist in finalists)


def _finalist_session(store):
    truths = {"safe": Truth(gain=0.1), "middle": Truth(loss=0.02, gain=0.15), "edge": Truth(loss=0.03, gain=0.2)}

    def truth(recipe, item_id):
        return truths[argument(recipe, "input_control/system_prompt", "text")]

    study = build_study(store, truth, space={"prompt": prompt_template(tuple(truths))},
                        settings=dataclasses.replace(SMALL, round_candidates=16))
    session = _started(study, store)
    for text in truths:
        assert session.evaluate(Candidate("prompt", {"prompt.text": text})).status == "accepted"
    return session


def _recorded(store, study):
    row = store.execute("SELECT finalists FROM certifications WHERE study_id = ?", (study.id,)).fetchone()
    return None if row is None else json.loads(row[0])


def test_the_finalist_plan_is_written_before_any_fresh_item_is_measured(store, tmp_path):
    session = _finalist_session(store)
    study = session.study
    capability = study.constraints[0].benchmark
    seen = []
    capability.set_hook(lambda steered, item_ids: seen.append(_recorded(store, study) is not None))
    certify(session, bundle_dir=tmp_path / "bundles")
    assert seen and all(seen)


def test_a_second_call_after_an_interruption_measures_the_planned_items(store, tmp_path):
    session = _finalist_session(store)
    study = session.study
    capability = study.constraints[0].benchmark
    calls = []

    def interrupt(steered, item_ids):
        calls.append(item_ids)
        if len(calls) == 2:
            raise Interrupted()

    capability.set_hook(interrupt)
    with pytest.raises(Interrupted):
        certify(session, bundle_dir=tmp_path / "bundles")
    plan = _recorded(store, study)
    assert plan and all(record["level"] is None for record in plan)
    planned = {item for record in plan for items in record["fresh_items"].values() for item in items}
    search = set(study.pools[capability.id].search)
    capability.set_hook(None)
    earlier = len(capability.calls)
    certify(session, bundle_dir=tmp_path / "bundles")
    measured = {item for _, items, _ in capability.calls[earlier:] for item in items if item in search}
    assert measured and measured <= planned
    final = _recorded(store, study)
    assert all(record["level"] is not None for record in final)
    assert sorted(json.dumps(record["fresh_items"], sort_keys=True) for record in final) == sorted(
        json.dumps(record["fresh_items"], sort_keys=True) for record in plan)


def test_a_study_without_constraints_has_finalists(store, tmp_path):
    study = build_study(store, lambda recipe, item_id: Truth(gain=0.1), space={"prompt": prompt_template(("a",))})
    objective = study.objective
    study = dataclasses.replace(study, constraints=(), pools={objective.id: study.pools[objective.id]})
    session = _started(study, store)
    result = session.evaluate(Candidate("prompt", {"prompt.text": "a"}))
    assert result.status == "accepted" and result.level == 0.0
    finalists = session.finalists()
    assert [finalist.result.candidate_id for finalist in finalists] == [result.candidate_id]
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    assert len(certification.certified) == 1
