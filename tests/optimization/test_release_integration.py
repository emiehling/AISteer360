"""Certification with a real pipeline factory on a tiny model: the written bundles load with their certificates.

The benchmarks are synthetic (their scores follow a known truth per recipe), so the study's decisions are exact, while
the steered pipelines, their resolved ids, and the bundles are real.
"""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certify
from steerability.optimization.study import Candidate
from steerability.spipe import SPipe
from tests.optimization.helpers import SMALL, Truth, argument, build_study, prompt_template

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TRUTHS = {"safe": Truth(gain=0.1), "edge": Truth(loss=0.03, gain=0.2)}


def _truth(recipe, item_id):
    return TRUTHS[argument(recipe, "input_control/system_prompt", "text")]


def test_certified_bundles_load_with_their_certificates(tmp_path):
    store = Store.open(tmp_path / "store")
    study = build_study(store, _truth, space={"prompt": prompt_template(tuple(TRUTHS))},
                        settings=dataclasses.replace(SMALL, round_candidates=16))
    study = dataclasses.replace(study, protocol=Protocol.create(MODEL, generate_defaults={"temperature": 0}))
    factory = PipelineFactory(study.protocol, store=store, device_map="cpu")
    try:
        with StudySession(study, store=store, factory=factory) as session:
            session.start()
            for text in TRUTHS:
                assert session.evaluate(Candidate("prompt", {"prompt.text": text})).status == "accepted"
            certification = certify(session, bundle_dir=tmp_path / "bundles")
    finally:
        factory.release()
    assert len(certification.certified) == 2 and certification.failed_at is None
    for resolved, path in certification.bundles.items():
        bundle = SPipe.load(path)
        certificate = certification.certificates[resolved]
        assert bundle.manifest["lock"]["certificate"] == certificate
        assert bundle.manifest["lock"]["protocol_id"] == study.protocol.id
        assert bundle.recipe_id == certificate["recipe_id"]
        assert store.get_resolution(study.protocol.id, certificate["recipe_id"]) == resolved
