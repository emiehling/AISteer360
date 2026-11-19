"""Tests for the archive of results and its proposer view (`optimization/evaluation/archive.py`)."""
import dataclasses
import json

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store, StoreIntegrityError
from steerability.optimization.evaluation.archive import Archive
from steerability.optimization.evaluation.result import OperatingPoint, Result
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.stats import Paired
from steerability.optimization.study import Candidate
from tests.optimization.helpers import StubFactory, Truth, build_study, prompt_template


def _quiet(recipe, item_id):
    return Truth(gain=0.1)


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _result(study, candidate_id, *, fitness, level, version=1, accepted=True):
    operating = None
    if accepted:
        operating = OperatingPoint(
            position={}, recipe_id="recipe", resolved_id=f"resolved-{candidate_id}",
            capability={"capability": Paired.from_diffs([-level * 0.05] * 10)},
            behavior=Paired.from_diffs([fitness] * 10), cost_ratio=1.0, degenerate_increase=0.0, eligible=True,
        )
    return Result(
        candidate_id=candidate_id, study_id=study.id, version=version, round_index=0,
        status="accepted" if accepted else "rejected", stage=5 if accepted else 4,
        reason=None if accepted else "not_confirmed", fitness=fitness, operating=operating,
        summary={"level": level, "gain": fitness} if accepted else {}, generations=1, seconds=0.5,
    )


def test_versions_are_append_only(store):
    study = build_study(store, _quiet)
    archive = Archive(study, store)
    candidate = Candidate("caa", {})
    first = _result(study, "a", fitness=0.1, level=0.2)
    archive.record(candidate, first, proposer="test")
    archive.record(candidate, first, proposer="test")
    second = dataclasses.replace(first, version=2, fitness=0.05)
    archive.record(candidate, second, proposer="test")
    assert [result.version for result in archive.versions("a")] == [1, 2]
    assert archive.latest("a") == second
    assert archive.versions("a")[0] == first
    with pytest.raises(StoreIntegrityError):
        archive.record(candidate, dataclasses.replace(first, fitness=0.2), proposer="test")


def test_the_envelope_and_best_on_hand_built_results(store):
    study = build_study(store, _quiet)
    archive = Archive(study, store)
    entries = {"a": (0.05, 0.0), "b": (0.10, 0.3), "c": (0.08, 0.5), "d": (0.20, 0.9)}
    for candidate_id, (fitness, level) in entries.items():
        archive.record(Candidate("caa", {}), _result(study, candidate_id, fitness=fitness, level=level),
                       proposer="test")
    archive.record(Candidate("caa", {}), _result(study, "e", fitness=0.0, level=0.0, accepted=False),
                   proposer="test")
    envelope = archive.envelope([-0.1, 0.0, 0.4, 0.6, 1.0])
    assert [None if result is None else result.candidate_id for result in envelope] == [None, "a", "b", "b", "d"]
    assert archive.best().candidate_id == "d"
    assert {result.candidate_id for result in archive.points()} == {"a", "b", "c", "d"}
    assert [result.candidate_id for result in archive.results()] == ["a", "b", "c", "d", "e"]


def test_the_view_exposes_no_item_ids(store):
    study = build_study(store, _quiet, space={"prompt": prompt_template()})
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    for text in ("be brief", "be kind"):
        session.evaluate(Candidate("prompt", {"prompt.text": text}))
    view = session.archive.view()
    assert len(view) == 2 and all(entry.status == "accepted" for entry in view)
    text = json.dumps([dataclasses.asdict(entry) for entry in view])
    for benchmark in study.benchmarks:
        assert not any(item_id in text for item_id in benchmark.item_ids())
    entry = view.entries[0]
    assert set(dataclasses.asdict(entry)) == {"candidate_id", "structure", "values", "status", "stage", "reason",
                                              "fitness", "summary"}
    assert view.get(entry.candidate_id) == entry and entry.candidate_id in view
