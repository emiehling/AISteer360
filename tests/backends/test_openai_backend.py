"""Semantic tests for `OpenAIBackend` against the fake `plain` server (docs 05, 10)."""
import pytest
import torch

from aisteer360.backends.generation_params import GenerationParams
from aisteer360.core.prompt import PreparedPrompt, Prompt
from aisteer360.core.requirements import Capability
from tests.utils.fake_vllm_hook import FakeVLLMHookServer
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

pytest.importorskip("openai")

from aisteer360.backends.openai_compat.openai import OpenAIBackend  # noqa: E402


@pytest.fixture(scope="module")
def model_tok():
    return tiny_llama(num_layers=2, hidden=32, heads=4, vocab=100), wordlevel_tokenizer()


def _backend(server, tokenizer, **kwargs):
    kwargs.setdefault("max_concurrency", 4)
    kwargs.setdefault("max_retries", 2)
    return OpenAIBackend(
        base_url=server.base_url, model="fake-model", api_key="EMPTY",
        tokenizer_name_or_path=None, **kwargs,
    )


def _text_prompt(text):
    return PreparedPrompt(prompt=Prompt.classify(text), adaptation_level="none")


def test_capabilities_plain_prompting_only(model_tok):
    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        backend = _backend(server, tok)  # no client tokenizer -> no token arrays -> no scoring
        caps = backend.capabilities.capabilities
        assert caps & Capability.MESSAGES
        assert caps & Capability.TEXT
        assert not (caps & Capability.SCORING)


def test_text_generate_returns_output_text(model_tok):
    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        backend = _backend(server, tok)
        session = backend.open_session([], _text_prompt("the cat"), {})
        out = session.generate(_text_prompt("the cat"), GenerationParams.from_gen_kwargs({"max_new_tokens": 4}))
        assert out.output_text is not None and len(out.output_text) == 1
        assert out.output_ids is None  # API backend: text-only
        assert out.metadata["backend"] == "OpenAIBackend"
        assert out.usage and out.usage["prompt_tokens"] > 0


def test_batched_text_generate_order_stable(model_tok):
    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        backend = _backend(server, tok)
        prepared = PreparedPrompt(prompt=Prompt.classify(["a", "b", "c"]), adaptation_level="none")
        out = backend.open_session([], prepared, {}).generate(
            prepared, GenerationParams.from_gen_kwargs({"max_new_tokens": 2})
        )
        assert len(out.output_text) == 3


def test_concurrency_bounded(model_tok):
    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        backend = _backend(server, tok, max_concurrency=4)
        prompts = [f"row {i}" for i in range(32)]
        prepared = PreparedPrompt(prompt=Prompt.classify(prompts), adaptation_level="none")
        backend.open_session([], prepared, {}).generate(
            prepared, GenerationParams.from_gen_kwargs({"max_new_tokens": 1})
        )
        assert server.max_in_flight <= 4
        assert server.max_in_flight >= 2  # genuinely concurrent


def test_retry_then_succeed(model_tok):
    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        server.respond_429_times(1)
        backend = _backend(server, tok, max_retries=3)
        prepared = _text_prompt("retry me")
        out = backend.open_session([], prepared, {}).generate(
            prepared, GenerationParams.from_gen_kwargs({"max_new_tokens": 1})
        )
        assert out.output_text is not None


def test_persistent_500_surfaces_partial_failure(model_tok):
    from aisteer360.backends.errors import BatchPartialFailure

    model, tok = model_tok
    with FakeVLLMHookServer(model, tok, profile="plain") as server:
        server.always_500("/v1/completions")
        backend = _backend(server, tok, max_retries=1)
        prepared = _text_prompt("boom")
        with pytest.raises(BatchPartialFailure) as excinfo:
            backend.open_session([], prepared, {}).generate(
                prepared, GenerationParams.from_gen_kwargs({"max_new_tokens": 1})
            )
        assert 0 in excinfo.value.indices


def test_scoring_parity_with_hf(model_tok):
    """OpenAI scoring (prompt_logprobs) matches HF teacher-forced logprobs on the tiny model."""
    from aisteer360.backends.huggingface.backend import HuggingFaceBackend

    model, tok = model_tok
    prompt_ids = torch.tensor([[3, 4, 5]])
    ref = torch.tensor([[6, 7]])

    with FakeVLLMHookServer(model, tok, profile="plain", support_token_arrays=True) as server:
        # a client tokenizer enables token arrays + scoring
        backend = OpenAIBackend(base_url=server.base_url, model="fake-model", max_concurrency=2)
        backend.tokenizer = tok  # inject the same tokenizer the fake uses
        backend._supports_token_arrays = True
        backend._supports_prompt_logprobs = True
        assert backend.capabilities.capabilities & Capability.SCORING

        prepared = PreparedPrompt(
            prompt=Prompt.classify(prompt_ids),
            adapted_token_ids=prompt_ids,
            adaptation_level="tokens",
        )
        api_logprobs = backend.open_session([], prepared, {}).score(prepared, ref)

    hf = HuggingFaceBackend(lazy_init=True)
    hf.adopt_model(model)
    hf.tokenizer = tok
    hf_prepared = PreparedPrompt(prompt=Prompt.classify(prompt_ids), adaptation_level="none")
    with hf.open_session([], hf_prepared, {}) as session:
        hf_logprobs = session.score(hf_prepared, ref)

    assert api_logprobs.shape == hf_logprobs.shape == (1, 2)
    assert torch.allclose(api_logprobs, hf_logprobs, atol=1e-4)
