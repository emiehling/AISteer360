"""Tests for the study session: limits, deadlines, resuming, and finalists (`optimization/evaluation/session.py`)."""
import dataclasses
import time

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.study import Candidate
from tests.optimization.helpers import SMALL, StubFactory, Truth, argument, build_study, multiplier, prompt_template

CAA = Candidate("caa", {})


def _quadratic(recipe, item_id):
    value = multiplier(recipe)
    if value is None:
        return Truth(gain=0.1)
    return Truth(loss=0.0055 * value * value, gain=0.05 * value)


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _rows(store, table):
    return store.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _started(study, store, **limits):
    session = StudySession(study, store=store, factory=StubFactory(study.protocol), **limits)
    session.start()
    return session


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
