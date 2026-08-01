"""Tests for `VLLMServeBackend` and `VLLMServeSession` against a mocked vLLM server, plus the
encoder-decoder spec rejection. No vLLM installation or live server is required."""
import pytest
import torch
from transformers import LlamaConfig, T5Config

from aisteer360.algorithms.core.execution import (
    BackendSpec,
    GenerationItem,
    GenerationParams,
    HookEntry,
    InterventionEntry,
    InterventionSpec,
    PartialBatchError,
    PreparedPrompt,
    ScoringItem,
    TransportError,
    UnsupportedOperationError,
)
from aisteer360.backends.vllm import VLLMServeBackend
from tests.utils.tiny_models import wordlevel_tokenizer


class _FakeServer:
    """Routes `_request_json` calls to canned responses and records requests."""

    def __init__(self, model_id="m", completions=None, version=True, prompt_logprobs=None):
        self.model_id = model_id
        self.completions = completions or {}
        self.version = version
        self.prompt_logprobs = prompt_logprobs
        self.requests: list[tuple[str, dict | None]] = []
        self.fail_prompts: dict[tuple[int, ...], int] = {}

    def handle(self, path, payload):
        self.requests.append((path, payload))
        if path == "/version":
            if self.version:
                return {"version": "0.10.0"}
            raise ValueError("HTTP 404 from /version: not found")
        if path == "/v1/models":
            return {"data": [{"id": self.model_id}]}
        if path == "/v1/completions":
            prompt = tuple(payload["prompt"])
            remaining = self.fail_prompts.get(prompt, 0)
            if remaining > 0:
                self.fail_prompts[prompt] = remaining - 1
                raise TransportError("connection reset")
            if self.prompt_logprobs is not None and "prompt_logprobs" in payload:
                entries = [None] + [
                    {str(token_id): {"logprob": self.prompt_logprobs}}
                    for token_id in prompt[1:]
                ]
                return {"choices": [{
                    "text": "", "finish_reason": "length", "prompt_logprobs": entries,
                }]}
            key = prompt
            choices = self.completions.get(key)
            if choices is None:
                choices = [{
                    "token_ids": [9, 1], "finish_reason": "stop", "stop_reason": None,
                }]
            return {"choices": choices}
        raise ValueError(f"HTTP 404 from {path}: not found")


@pytest.fixture()
def fake_server(monkeypatch):
    server = _FakeServer()

    def fake_request(self, path, payload, expect_json=True):
        return server.handle(path, payload)

    monkeypatch.setattr(VLLMServeBackend, "_request_json", fake_request)
    monkeypatch.setattr(
        "aisteer360.backends.vllm._client_tokenizer",
        lambda source, trust_remote_code=False: wordlevel_tokenizer(),
    )
    monkeypatch.setattr(
        "aisteer360.backends.vllm._config_layout",
        lambda source, trust_remote_code=False: None,
    )
    return server


def _serve_spec(**options):
    merged = {"base_url": "http://server:8000", "retry_backoff": 0.0, **options}
    return BackendSpec(kind="vllm-serve", model="m", options=merged)


class TestServeBackendConstruction:

    def test_constructs_against_vllm_server(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        assert backend._served_model == "m"
        assert ("/version", None) in fake_server.requests

    def test_non_vllm_endpoint_rejected(self, fake_server):
        fake_server.version = False
        with pytest.raises(ValueError, match="version"):
            VLLMServeBackend(_serve_spec())

    def test_base_url_v1_suffix_normalizes(self, fake_server):
        backend = VLLMServeBackend(_serve_spec(base_url="http://server:8000/v1/"))
        assert backend._base_url == "http://server:8000"

    def test_served_model_mismatch_rejected(self, fake_server):
        fake_server.model_id = "other-model"
        with pytest.raises(ValueError, match="other-model"):
            VLLMServeBackend(_serve_spec())

    def test_missing_base_url_rejected(self):
        with pytest.raises(ValueError, match="base_url"):
            VLLMServeBackend(BackendSpec(kind="vllm-serve", model="m"))

    def test_hook_plugin_without_discovery_surface_rejected(self, fake_server):
        with pytest.raises(ValueError, match="hook"):
            VLLMServeBackend(_serve_spec(hook_plugin=True))


class TestServeSessionGenerate:

    def _item(self, ids=(0, 3, 4)):
        return GenerationItem(prompt=PreparedPrompt.from_token_ids(list(ids)))

    def test_token_id_round_trip_and_finish_mapping(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [
            {"token_ids": [5, 6], "finish_reason": "stop", "stop_reason": "sat"},
        ]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams(max_new_tokens=4))
        output = results[0].output
        assert output.output_ids.tolist() == [[5, 6]]
        assert output.adapted_input_ids.tolist() == [[0, 3, 4]]
        assert output.finish_reason == "stop"
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["prompt"] == [0, 3, 4]
        assert body["return_token_ids"] is True
        assert body["max_tokens"] == 4

    def test_eos_maps_from_null_stop_reason(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams())
        assert results[0].output.finish_reason == "eos"

    def test_multiple_candidates_pack_per_item(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [
            {"token_ids": [5, 6, 7], "finish_reason": "length", "stop_reason": None},
            {"token_ids": [8], "finish_reason": "stop", "stop_reason": None},
        ]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams(n=2, max_new_tokens=3))
        output = results[0].output
        assert output.output_ids.shape == (2, 3)
        assert output.finish_reasons == ("length", "eos")

    def test_server_without_token_id_return_rejected(self, fake_server):
        fake_server.completions[(0, 3, 4)] = [{"text": "hi", "finish_reason": "stop"}]
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(PartialBatchError) as excinfo:
                session.generate([self._item()], GenerationParams())
        assert "return_token_ids" in str(excinfo.value)

    def test_transient_transport_failure_retries_to_success(self, fake_server):
        fake_server.fail_prompts[(0, 3, 4)] = 2  # two failures, third attempt succeeds
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            results = session.generate([self._item()], GenerationParams())
        assert len(results) == 1

    def test_persistent_failure_surfaces_partial_batch(self, fake_server):
        fake_server.fail_prompts[(0, 3)] = 99
        backend = VLLMServeBackend(_serve_spec())
        items = [self._item((0, 3, 4)), self._item((0, 3)), self._item((0, 4, 5))]
        with backend.open_session() as session:
            with pytest.raises(PartialBatchError) as excinfo:
                session.generate(items, GenerationParams())
        error = excinfo.value
        assert error.failed_indices == (1,)
        assert len(error.results) == 2
        assert isinstance(error.failures[0][1], TransportError)

    def test_unmapped_extra_key_raises_before_any_request(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        request_count = len(fake_server.requests)
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="num_beams"):
                session.generate([self._item()], GenerationParams(extra={"num_beams": 2}))
        assert len(fake_server.requests) == request_count

    def test_shared_seed_derives_distinct_request_seeds(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        items = [self._item((0, 3, 4)), self._item((0, 3))]
        with backend.open_session() as session:
            session.generate(items, GenerationParams(seed=42))
        seeds = [
            payload["seed"] for path, payload in fake_server.requests
            if path == "/v1/completions"
        ]
        assert len(seeds) == 2
        assert seeds[0] != seeds[1]

    def test_hook_entries_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            state_entries=(HookEntry(hooks={"pre": []}),),
        )
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="huggingface"):
                session.generate([item], GenerationParams())

    def test_intervention_entries_not_implemented(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = GenerationItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]),
            state_entries=(InterventionEntry(spec=InterventionSpec()),),
        )
        with backend.open_session() as session:
            with pytest.raises(NotImplementedError, match="lowering"):
                session.generate([item], GenerationParams())


class TestServeSessionScore:

    def test_prompt_logprob_scoring(self, fake_server):
        fake_server.prompt_logprobs = -1.25
        backend = VLLMServeBackend(_serve_spec())
        items = [
            ScoringItem(
                prompt=PreparedPrompt.from_token_ids([0, 3, 4]),
                ref_output_ids=torch.tensor([[5, 6]]),
            ),
            ScoringItem(
                prompt=PreparedPrompt.from_token_ids([0, 4]),
                ref_output_ids=torch.tensor([[7, 3]]),
            ),
        ]
        with backend.open_session() as session:
            scored = session.score(items, GenerationParams())
        assert scored.shape == (2, 2)
        assert torch.allclose(scored, torch.full((2, 2), -1.25))
        body = next(p for path, p in fake_server.requests if path == "/v1/completions")
        assert body["prompt"] == [0, 3, 4, 5, 6]
        assert body["prompt_logprobs"] == 0

    def test_mismatched_ref_lengths_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        items = [
            ScoringItem(prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5]])),
            ScoringItem(prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5, 6]])),
        ]
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="reference length"):
                session.score(items, GenerationParams())

    def test_forward_kwargs_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids([0, 3]), ref_output_ids=torch.tensor([[5]]),
        )
        with backend.open_session() as session:
            with pytest.raises(ValueError, match="output_attentions"):
                session.score([item], GenerationParams(extra={"output_attentions": True}))


class TestServeSessionLifecycle:

    def test_closed_session_rejected(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        session = backend.open_session()
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.generate([], GenerationParams())

    def test_capture_unsupported(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(UnsupportedOperationError, match="capture"):
                session.capture([PreparedPrompt.from_token_ids([0, 3])], [0], "all_tokens")

    def test_layout_unresolvable_raises_on_access(self, fake_server):
        backend = VLLMServeBackend(_serve_spec())
        with backend.open_session() as session:
            with pytest.raises(RuntimeError, match="layout"):
                _ = session.layout


class TestEncoderDecoderSpecRejection:

    def test_local_encoder_decoder_config_rejected_for_vllm_kinds(self, tmp_path):
        config_dir = tmp_path / "enc-dec"
        T5Config().save_pretrained(config_dir)
        for kind in ("vllm", "vllm-serve"):
            with pytest.raises(ValueError, match="encoder-decoder"):
                BackendSpec(kind=kind, model=str(config_dir))

    def test_huggingface_kind_unaffected(self, tmp_path):
        config_dir = tmp_path / "enc-dec"
        T5Config().save_pretrained(config_dir)
        spec = BackendSpec(kind="huggingface", model=str(config_dir))
        assert spec.model == str(config_dir)

    def test_decoder_only_config_accepted(self, tmp_path):
        config_dir = tmp_path / "decoder"
        LlamaConfig(num_hidden_layers=1, hidden_size=8, num_attention_heads=2).save_pretrained(config_dir)
        spec = BackendSpec(kind="vllm", model=str(config_dir))
        assert spec.kind == "vllm"

    def test_unresolvable_reference_passes(self):
        spec = BackendSpec(kind="vllm", model="m")
        assert spec.model == "m"
