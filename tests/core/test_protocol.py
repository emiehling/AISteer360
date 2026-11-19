"""Tests for the measurement protocol (`algorithms/core/protocol.py`)."""
import json

import pytest
import torch

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.protocol import Protocol, ProtocolFacts, ProtocolMismatchError
from steerability.evaluation.provider import ProviderOptions

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _base(**fields) -> Protocol:
    return Protocol.create(MODEL, **fields)


class TestIdentity:
    @pytest.mark.parametrize("changed", [
        {"model": "other/model"},
        {"model_revision": "abc123"},
        {"tokenizer": "other/tokenizer"},
        {"hf_model_kwargs": {"torch_dtype": torch.bfloat16}},
        {"provider": ProviderOptions(max_batch_size=4)},
        {"backend": BackendSpec(kind="vllm-serve", model=MODEL, options={"base_url": "http://x:8000"})},
        {"fit": "in_process"},
    ])
    def test_a_measurement_field_changes_the_id(self, changed):
        model = changed.pop("model", MODEL)
        assert Protocol.create(model, **changed).id != _base().id

    def test_dtypes_differ(self):
        first = _base(hf_model_kwargs={"torch_dtype": torch.bfloat16})
        second = _base(hf_model_kwargs={"torch_dtype": torch.float16})
        assert first.id != second.id

    def test_max_concurrency_does_not_change_the_id(self):
        assert _base(provider=ProviderOptions(max_concurrency=8)).id == _base().id

    def test_a_default_provider_equals_none(self):
        assert _base(provider=ProviderOptions()).id == _base().id

    def test_a_decoding_setting_changes_the_id_but_not_the_fit_context(self):
        greedy = _base(generate_defaults={"temperature": 0})
        sampled = _base(generate_defaults={"temperature": 0.7}, seed=3)
        assert greedy.id != sampled.id
        assert greedy.fit_context_id == sampled.fit_context_id

    def test_versions_change_the_fit_context(self):
        assert _base(versions={"torch": "1"}).fit_context_id != _base(versions={"torch": "2"}).fit_context_id

    def test_credentials_never_appear_in_the_identity(self):
        spec = BackendSpec(
            kind="vllm-serve", model=MODEL,
            options={"base_url": "http://x:8000", "api_key": "secret", "hf_token": "secret", "nested": {"auth_key": "secret"}},
        )
        protocol = _base(backend=spec)
        assert "secret" not in json.dumps(protocol.identity())
        assert protocol.id == _base(backend=BackendSpec(
            kind="vllm-serve", model=MODEL, options={"base_url": "http://x:8000", "nested": {}},
        )).id

    def test_from_identity_round_trips(self):
        protocol = _base(
            model_revision="r1", hf_model_kwargs={"torch_dtype": torch.bfloat16},
            backend=BackendSpec(kind="vllm", model=MODEL, options={"engine_kwargs": {"max_model_len": 512}}),
            provider=ProviderOptions(max_batch_size=2, runtime_kwargs={"substrings": ["x"]}, max_concurrency=4),
            generate_defaults={"temperature": 0}, seed=11,
        )
        rebuilt = Protocol.from_identity(protocol.identity())
        assert rebuilt.id == protocol.id
        assert rebuilt.hf_model_kwargs["torch_dtype"] is torch.bfloat16

    def test_revision_belongs_in_its_own_field(self):
        with pytest.raises(ValueError, match="model_revision"):
            Protocol.create(MODEL, hf_model_kwargs={"revision": "abc"})
        assert _base(model_revision="abc").load_kwargs == {"revision": "abc"}

    def test_from_pipeline(self):
        from steerability.algorithms.core.steering_pipeline import SteeringPipeline

        pipeline = SteeringPipeline(model_name_or_path=MODEL, hf_model_kwargs={"revision": "r2"})
        protocol = Protocol.from_pipeline(pipeline)
        assert protocol.model == MODEL
        assert protocol.model_revision == "r2"
        assert protocol.backend is None


class TestFacts:
    def test_conflicting_facts_raise(self):
        recorded = ProtocolFacts(model_fingerprint="a", torch_dtype="float32")
        with pytest.raises(ProtocolMismatchError, match="model_fingerprint"):
            recorded.merged(ProtocolFacts(model_fingerprint="b"))

    def test_missing_facts_are_filled(self):
        merged = ProtocolFacts(model_fingerprint="a").merged(ProtocolFacts(torch_dtype="float32"))
        assert merged == ProtocolFacts(model_fingerprint="a", torch_dtype="float32")

    def test_observe_reads_a_loaded_model(self):
        from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

        facts = ProtocolFacts.observe(tiny_llama(num_layers=2, hidden=16, heads=2), wordlevel_tokenizer())
        assert facts.model_fingerprint and facts.tokenizer_fingerprint
        assert facts.torch_dtype == "float32"
        assert ProtocolFacts.from_dict(facts.to_dict()) == facts
