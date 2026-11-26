"""Tests for weight changes in a study: the limits on structural fits and the weights of a certificate.

In the synthetic world a structural fit is a weights context other than `"base"` that the stub factory has not
prepared or steered yet. The `LoadLoRA` entries below give these contexts, one structural fit per adapter path.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.core.recipe import Choice, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA
from steerability.optimization import Candidate, optional, structures
from steerability.optimization.evaluation.service import EvaluationService
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certificate_weights, certify
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import parse, program_text
from steerability.spipe.codec import EncodeContext, encode
from tests.optimization.helpers import SMALL, FixedProposer, StubFactory, Truth, argument, build_study

ADAPTERS = ("adapter-a", "adapter-b", "adapter-c")
BASE = "synthetic-model"


def _gain(recipe, item_id):
    return Truth(gain=0.1)


def _space(paths=ADAPTERS):
    lora = entry(LoadLoRA, name="lora", path=Choice(values=tuple(paths)), base_model=BASE)
    prompt = entry(SystemPrompt, name="prompt", text=Choice(values=("be formal", "be brief")))
    return structures(optional(lora), optional(prompt))


def _lora(path):
    return Candidate(structure="lora", values={"lora.path": path})


def _both(path, text):
    return Candidate(structure="lora+prompt", values={"lora.path": path, "prompt.text": text})


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


@pytest.fixture()
def store(tmp_path):
    return Store.open(root=tmp_path / "store")


def _started(study, store, **limits):
    session = StudySession(study=study, store=store, factory=StubFactory(protocol=study.protocol), **limits)
    session.start()
    return session


def test_the_session_lets_through_its_structural_fits_and_stops_a_candidate_that_needs_another(store):
    study = build_study(store=store, truth=_gain, space=_space())
    assert list(study.space) == ["lora", "prompt", "lora+prompt"]
    session = _started(study, store, max_structural_fits=2)
    results = [session.evaluate(candidate=_lora(path)) for path in ADAPTERS]
    assert [result.status for result in results[:2]] == ["accepted", "accepted"]
    assert (results[2].status, results[2].reason, results[2].stage) == ("rejected", "budget", 1)
    assert session.archive.latest(candidate_id=study.candidate_id(candidate=_lora("adapter-c"))) is None
    assert session.structural_fits == 2 and not session.exhausted()
    assert session.evaluate(candidate=_both("adapter-a", "be formal")).status == "accepted"
    assert session.evaluate(candidate=_prompt("be brief")).status == "accepted"
    assert session.evaluate(candidate=_lora("adapter-c")).reason == "budget"


def test_a_candidate_stopped_in_a_batch_is_left_out_of_its_preparation(store):
    study = build_study(store=store, truth=_gain, space=_space(), settings=SMALL)
    session = _started(study, store, max_structural_fits=1)
    session.prepare(candidates=[_lora("adapter-a"), _lora("adapter-b"), _prompt("be formal")])
    prepared = session.factory.prepared[-1]
    recipe_of = {path: study.bind(candidate=_lora(path)).recipe().id for path in ("adapter-a", "adapter-b")}
    assert recipe_of["adapter-a"] in prepared and recipe_of["adapter-b"] not in prepared
    assert session.evaluate(candidate=_lora("adapter-b")).reason == "budget"


def test_a_phase_without_structural_fits_searches_on_the_stored_products(store):
    study = build_study(store=store, truth=_gain, space=_space())
    first = FixedProposer(study=study, candidates=[_lora("adapter-a")])
    second = FixedProposer(
        study=study,
        candidates=[_lora("adapter-b"), _both("adapter-a", "be formal"), _prompt("be brief")],
    )
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=first), SearchPhase(engine=second, structural_fits=0)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
    )
    recorded = archive.view().ids
    assert study.candidate_id(candidate=_lora("adapter-b")) not in recorded
    for candidate in (_lora("adapter-a"), _both("adapter-a", "be formal"), _prompt("be brief")):
        assert study.candidate_id(candidate=candidate) in recorded
    assert [result.reason for _, result in second.told] == ["budget", None, None]


def test_a_run_limit_on_structural_fits_applies_to_every_phase(store):
    study = build_study(store=store, truth=_gain, space=_space())
    proposer = FixedProposer(study=study, candidates=[_lora(path) for path in ADAPTERS])
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=proposer)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
        max_structural_fits=1,
    )
    assert len(archive.view()) == 1
    assert [result.reason for _, result in proposer.told] == [None, "budget", "budget"]


def test_the_service_lists_a_candidate_that_needs_a_structural_fit_beyond_the_limits(store):
    study = build_study(store=store, truth=_gain, space=_space())
    session = _started(study, store)
    service = EvaluationService(session=session, proposer="service", parse=parse, stage_deadline_seconds=600)
    program = program_text(study=study, candidate=_lora("adapter-b"))
    with session.allowance(structural_fits=0):
        reply = service.handle(request={"op": "full", "program": program})
    assert reply["artifacts"]["reason"] == "budget" and reply["metrics"]["combined_score"] == 0.0
    assert "needs a training or merge" in reply["artifacts"]["recent_unevaluated"]


def _by_weights(recipe, item_id):
    path = argument(recipe, "structural_control/load_lora", "path")
    if path is None:
        return Truth(loss=0.01, gain=0.15)
    return Truth(gain=0.1) if path.endswith("adapter-a") else Truth(loss=0.02, gain=0.2)


def test_the_certificate_records_the_weights_of_each_structural_control(tmp_path, store):
    directories = []
    for name in ADAPTERS[:2]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "adapter_config.json").write_text(f'{{"name": "{name}"}}')
        directories.append(str(directory))
    study = build_study(store=store, truth=_by_weights, space=_space(directories))
    session = _started(study, store)
    for candidate in (_lora(directories[0]), _prompt("be formal"), _lora(directories[1])):
        assert session.evaluate(candidate=candidate).status == "accepted"
    certification = certify(session=session, bundle_dir=tmp_path / "bundles")
    assert len(certification.certified) == 3
    identity = {path: encode(LoRAArtifact(path=path, base_model=BASE), EncodeContext(store=None))["$artifact"]
                for path in directories}
    for certificate in certification.certificates.values():
        if certificate["structure"] == "prompt":
            assert certificate["weights"] == []
        else:
            expected = identity[certificate["values"]["lora.path"]]
            assert certificate["weights"] == [{"method": "structural_control/load_lora", "artifact_id": expected}]
    assert len(set(identity.values())) == 2
    disabled = LoadLoRA(path=directories[0], base_model=BASE)
    disabled.enabled = False
    pipeline = SimpleNamespace(structural_controls=[LoadLoRA(path=directories[1], base_model=BASE), disabled])
    assert certificate_weights(pipeline=pipeline) == [
        {"method": "structural_control/load_lora", "artifact_id": identity[directories[1]]},
    ]
