"""Certification with a real pipeline factory on a tiny model: the written bundles load with their certificates.

The benchmarks are synthetic (their scores follow a known truth per recipe), so the study's decisions are exact, while
the steered pipelines (one of them with a LoRA adapter), their resolved ids, and the bundles are real.
"""
import dataclasses

import pytest
import torch

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Template, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certify
from steerability.optimization.study import Candidate
from steerability.spipe import SPipe
from steerability.spipe.codec import EncodeContext, encode
from tests.optimization.helpers import SMALL, Truth, argument, build_study, prompt_template

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TRUTHS = {"safe": Truth(gain=0.1), "edge": Truth(loss=0.03, gain=0.2)}
ADAPTED = Truth(loss=0.015, gain=0.15)


def _truth(recipe, item_id):
    if argument(recipe, "structural_control/load_lora", "path") is not None:
        return ADAPTED
    return TRUTHS[argument(recipe, "input_control/system_prompt", "text")]


def _adapter(directory):
    """Write a LoRA adapter with random weights for the tiny model."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), config).save_pretrained(directory)
    return str(directory)


def test_certified_bundles_load_with_their_certificates(tmp_path):
    store = Store.open(tmp_path / "store")
    adapter = _adapter(tmp_path / "adapter")
    space = {
        "prompt": prompt_template(texts=tuple(TRUTHS)),
        "lora": Template(entries=(entry(LoadLoRA, name="lora", path=adapter, base_model=MODEL),)),
    }
    settings = dataclasses.replace(SMALL, round_candidates=16)
    study = build_study(store=store, truth=_truth, space=space, settings=settings)
    study = dataclasses.replace(study, protocol=Protocol.create(MODEL, generate_defaults={"temperature": 0}))
    factory = PipelineFactory(study.protocol, store=store, device_map="cpu")
    try:
        with StudySession(study, store=store, factory=factory) as session:
            session.start()
            for text in TRUTHS:
                assert session.evaluate(Candidate("prompt", {"prompt.text": text})).status == "accepted"
            assert session.evaluate(candidate=Candidate(structure="lora", values={})).status == "accepted"
            certification = certify(session, bundle_dir=tmp_path / "bundles")
    finally:
        factory.release()
    assert len(certification.certified) == 3 and certification.failed_at is None
    weights = [certificate["weights"] for certificate in certification.certificates.values()]
    artifact_id = encode(LoRAArtifact(path=adapter, base_model=MODEL), EncodeContext(store=None))["$artifact"]
    loader = {"method": "structural_control/load_lora", "artifact_id": artifact_id}
    assert sorted(weights, key=len) == [[], [], [loader]]
    for resolved, path in certification.bundles.items():
        bundle = SPipe.load(path)
        certificate = certification.certificates[resolved]
        assert bundle.manifest["lock"]["certificate"] == certificate
        assert bundle.manifest["lock"]["protocol_id"] == study.protocol.id
        assert bundle.recipe_id == certificate["recipe_id"]
        assert store.get_resolution(study.protocol.id, certificate["recipe_id"]) == resolved
