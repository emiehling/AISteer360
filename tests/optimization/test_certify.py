"""Tests for certification on the sealed items and the release of bundles (`optimization/release/certify.py`)."""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import noise_floor, paired, usable
from steerability.optimization.measurement.stats import floor_variance
from steerability.optimization.release.certify import certify
from steerability.optimization.study import Candidate, StudyError
from tests.optimization.helpers import SMALL, StubFactory, Truth, argument, benchmarks, build_study, prompt_template

TRUTHS = {"safe": Truth(gain=0.1), "middle": Truth(loss=0.015, gain=0.15), "edge": Truth(loss=0.03, gain=0.2)}


class Interrupted(BaseException):
    """Stops a certification call partway."""


def _truth(recipe, item_id):
    return TRUTHS[argument(recipe, "input_control/system_prompt", "text")]


def _session(store, *, name="synthetic", capability=None, behavior=None):
    study = build_study(store, _truth, space={"prompt": prompt_template(tuple(TRUTHS))}, name=name,
                        settings=dataclasses.replace(SMALL, round_candidates=16), capability=capability,
                        behavior=behavior)
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    for text in TRUTHS:
        session.evaluate(Candidate("prompt", {"prompt.text": text}))
    return session


def _resolved(session, text):
    return session.archive.latest(session.study.candidate_id(Candidate("prompt", {"prompt.text": text}))) \
        .operating.resolved_id


def _sealed(study):
    return {item for split in study.pools.values() for item in split.sealed}


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def test_certified_points_are_written_as_bundles(store, tmp_path):
    session = _session(store)
    study = session.study
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    expected = tuple(_resolved(session, text) for text in ("safe", "middle", "edge"))
    assert certification.certified == expected and certification.failed_at is None
    for index, resolved in enumerate(expected):
        certificate = certification.certificates[resolved]
        assert certificate["schema"] == "certificate/1" and certificate["index"] == index
        assert certificate["study_id"] == study.id and certificate["protocol_id"] == study.protocol.id
        assert certificate["constraints"]["capability"]["n"] == len(study.pools[study.constraints[0].benchmark.id]
                                                                    .sealed)
        assert certificate["behavior"]["n"] == len(study.pools[study.objective.id].sealed)
        assert certificate["behavior"]["lower"] <= certificate["behavior"]["gain"] <= certificate["behavior"]["upper"]
        assert [entry["resolved_id"] for entry in certificate["finalists"]] == list(expected)
        assert certification.bundles[resolved].exists() and certification.bundles[resolved].suffix == ".spipe"
    bundles = [bundle for pipeline in session.factory.pipelines for bundle in pipeline.bundles]
    assert len(bundles) == 3
    for bundle in bundles:
        assert bundle.options["freeze"] is True and bundle.options["protocol"] == study.protocol
        assert bundle.options["certificate"]["schema"] == "certificate/1"
        assert len(bundle.saved) == 1


def test_the_finalist_order_is_recorded_before_any_sealed_item_is_read(store, tmp_path):
    session = _session(store)
    study = session.study
    sealed = _sealed(study)
    seen = []

    def hook(steered, item_ids):
        if set(item_ids) & sealed:
            recorded = store.execute("SELECT finalists FROM certifications WHERE study_id = ?",
                                     (study.id,)).fetchone()
            used = store.execute("SELECT COUNT(*) FROM sealed_uses WHERE resolved_id = ?",
                                 (steered.resolved_id,)).fetchone()[0]
            seen.append(recorded is not None and used > 0)

    for benchmark in study.benchmarks:
        benchmark.set_hook(hook)
    certify(session, bundle_dir=tmp_path / "bundles")
    assert seen and all(seen)


def test_a_study_is_certified_once(store, tmp_path):
    session = _session(store)
    certify(session, bundle_dir=tmp_path / "bundles")
    with pytest.raises(StudyError, match="certified already"):
        certify(session, bundle_dir=tmp_path / "bundles")


def test_a_study_without_finalists_stays_open(store, tmp_path):
    study = build_study(store, _truth, space={"prompt": prompt_template(tuple(TRUTHS))},
                        settings=dataclasses.replace(SMALL, round_candidates=16))
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    early = certify(session, bundle_dir=tmp_path / "bundles")
    assert early.certified == () and early.failed_at is None
    assert store.execute("SELECT COUNT(*) FROM certifications WHERE study_id = ?", (study.id,)).fetchone()[0] == 0
    for text in TRUTHS:
        session.evaluate(Candidate("prompt", {"prompt.text": text}))
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    assert certification.certified == tuple(_resolved(session, text) for text in ("safe", "middle", "edge"))


def test_an_interrupted_call_continues_with_the_recorded_order(store, tmp_path, monkeypatch):
    session = _session(store)
    study = session.study
    sealed = _sealed(study)
    second = _resolved(session, "middle")

    def interrupt(steered, item_ids):
        if steered.resolved_id == second and set(item_ids) & sealed:
            raise Interrupted()

    for benchmark in study.benchmarks:
        benchmark.set_hook(interrupt)
    with pytest.raises(Interrupted):
        certify(session, bundle_dir=tmp_path / "bundles")
    for benchmark in study.benchmarks:
        benchmark.set_hook(None)
    capability = study.constraints[0].benchmark
    first = _resolved(session, "safe")
    sealed_reads = sum(1 for resolved, items, _ in capability.calls if resolved == first and set(items) & sealed)
    monkeypatch.setattr(session, "finalists", lambda: pytest.fail("the recorded finalists must be used"))
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    assert certification.certified == tuple(_resolved(session, text) for text in ("safe", "middle", "edge"))
    assert sealed_reads == sum(1 for resolved, items, _ in capability.calls
                               if resolved == first and set(items) & sealed)


def test_a_study_that_shares_sealed_items_with_a_used_one_is_refused(store, tmp_path):
    capability, behavior = benchmarks(_truth)
    first = _session(store, name="first", capability=capability, behavior=behavior)
    certify(first, bundle_dir=tmp_path / "first")
    second = _session(store, name="second", capability=capability, behavior=behavior)
    with pytest.raises(StudyError, match="already used"):
        certify(second, bundle_dir=tmp_path / "second")
    certification = certify(second, bundle_dir=tmp_path / "second", allow_reused_sealed=True)
    assert len(certification.certified) == 3


def test_a_failing_finalist_stops_the_sequence(store, tmp_path):
    capability, behavior = benchmarks(_truth)
    session = _session(store, capability=capability, behavior=behavior)
    study = session.study
    sealed = _sealed(study)

    def truth(recipe, item_id):
        text = argument(recipe, "input_control/system_prompt", "text")
        if text == "middle" and item_id in sealed:
            return Truth(loss=0.2, gain=0.15)
        return TRUTHS[text]

    capability.set_truth(truth)
    behavior.set_truth(truth)
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    assert certification.certified == (_resolved(session, "safe"),) and certification.failed_at == 1
    edge = _resolved(session, "edge")
    assert not any(resolved == edge and set(items) & sealed for resolved, items, _ in capability.calls)
    outcome = store.execute("SELECT outcome FROM certifications WHERE study_id = ?", (study.id,)).fetchone()[0]
    assert outcome is not None


def test_the_sealed_call_receives_the_whole_sealed_pool_in_pool_order(store, tmp_path):
    probe = build_study(Store.open(tmp_path / "probe"), _truth, space={"prompt": prompt_template(tuple(TRUTHS))})
    dropped = probe.pools[probe.constraints[0].benchmark.id].sealed[3]
    capability, behavior = benchmarks(_truth, capability_omit={dropped})
    session = _session(store, capability=capability, behavior=behavior)
    study = session.study
    split = study.pools[capability.id]
    assert dropped not in usable(study, store, capability, "sealed")
    certification = certify(session, bundle_dir=tmp_path / "bundles")
    for resolved in certification.certified:
        sealed_calls = [items for called, items, _ in capability.calls
                        if called == resolved and set(items) & set(split.sealed)]
        assert sealed_calls[0] == split.sealed
        certificate = certification.certificates[resolved]
        entry = certificate["constraints"]["capability"]
        assert entry["n"] == len(split.sealed) - 1
        table = paired(study, store, resolved, capability, usable(study, store, capability, "sealed"))
        assert entry["lower_bound"] == pytest.approx(table.lower(SMALL.alpha, floor=floor_variance(0.05)))
        assert entry["noise_floor"] == noise_floor(study, store, capability) == 0.0
        assert certificate["behavior"]["noise_floor"] == noise_floor(study, store, behavior)
