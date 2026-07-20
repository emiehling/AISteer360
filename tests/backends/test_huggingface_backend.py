"""Tests for `HuggingFaceBackend` / `HuggingFaceSession` (doc 02 §5, §7).

Exercised directly against the backend on a hub-free tiny model, proving the clean extraction of
generation, scoring, and input normalization from the former pipeline.
"""
import pytest
import torch

from aisteer360.backends.base import StateControlEntry
from aisteer360.backends.generation_params import GenerationParams
from aisteer360.backends.huggingface.backend import HuggingFaceBackend
from aisteer360.core.output import Output
from aisteer360.core.prompt import PreparedPrompt, Prompt
from aisteer360.core.requirements import Capability
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


@pytest.fixture(scope="module")
def backend():
    be = HuggingFaceBackend(lazy_init=True)
    model = tiny_llama(num_layers=2, hidden=32, heads=4, vocab=100)
    be.adopt_model(model)
    be.tokenizer = wordlevel_tokenizer()
    return be


def _tensor_prompt(ids):
    prompt = Prompt.classify(torch.tensor(ids))
    return PreparedPrompt(prompt=prompt, adaptation_level="none")


def test_capabilities_full_set(backend):
    caps = backend.capabilities
    assert caps.capabilities & Capability.RESIDUAL_WRITE
    assert caps.capabilities & Capability.SCORING
    assert caps.capabilities & Capability.RAW_MODEL
    assert caps.accepts_artifacts == frozenset({"model", "checkpoint", "lora"})
    assert caps.max_concurrency == 1


def test_generate_returns_output_with_ids(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    params = GenerationParams.from_gen_kwargs({"max_new_tokens": 4, "do_sample": False})
    with backend.open_session([], prepared, {}) as session:
        output = session.generate(prepared, params)
    assert isinstance(output, Output)
    assert output.output_ids is not None
    assert output.output_ids.size(0) == 1
    assert output.output_ids.size(1) <= 4
    assert output.metadata["backend"] == "HuggingFaceBackend"


def test_generate_full_sequence_includes_prompt(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    params = GenerationParams.from_gen_kwargs(
        {"max_new_tokens": 4, "do_sample": False, "return_full_sequence": True}
    )
    with backend.open_session([], prepared, {}) as session:
        output = session.generate(prepared, params)
    assert output.output_ids.size(1) >= 3  # prompt + continuation


def test_finish_reason_length(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    params = GenerationParams.from_gen_kwargs({"max_new_tokens": 3, "do_sample": False})
    with backend.open_session([], prepared, {}) as session:
        output = session.generate(prepared, params)
    assert output.finish_reason == "length"


def test_score_shape_and_sign(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    ref = torch.tensor([[6, 7]])
    with backend.open_session([], prepared, {}) as session:
        logprobs = session.score(prepared, ref)
    assert logprobs.shape == (1, 2)
    assert (logprobs <= 0).all()


def test_score_empty_ref(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    with backend.open_session([], prepared, {}) as session:
        logprobs = session.score(prepared, torch.zeros((1, 0), dtype=torch.long))
    assert logprobs.shape == (1, 0)


def test_score_broadcasts_single_ref_over_batch(backend):
    prepared = _tensor_prompt([[3, 4, 5], [6, 7, 8]])
    ref = torch.tensor([[9, 10]])
    with backend.open_session([], prepared, {}) as session:
        logprobs = session.score(prepared, ref)
    assert logprobs.shape == (2, 2)


def test_reentrant_enter_raises(backend):
    prepared = _tensor_prompt([[3, 4, 5]])
    session = backend.open_session([], prepared, {})
    with session:
        with pytest.raises(RuntimeError, match="already active"):
            session.__enter__()


def test_session_registers_and_removes_hooks(backend):
    """A hook-level entry registers on enter and is removed on exit."""
    fired = {"count": 0}

    def _hook(module, args, kwargs, output):
        fired["count"] += 1
        return output

    layer_name = "model.layers.0"
    entry = StateControlEntry(
        control_name="probe",
        hooks={"pre": [], "forward": [{"module": layer_name, "hook_func": _hook}], "backward": []},
    )
    prepared = _tensor_prompt([[3, 4, 5]])
    params = GenerationParams.from_gen_kwargs({"max_new_tokens": 2, "do_sample": False})
    with backend.open_session([entry], prepared, {}) as session:
        session.generate(prepared, params)
    assert fired["count"] > 0
    # after exit, another generation without the entry must not fire the hook
    before = fired["count"]
    with backend.open_session([], prepared, {}) as session:
        session.generate(prepared, params)
    assert fired["count"] == before


def test_prepare_tensor_inputs_infers_mask(backend):
    ids, mask = backend.prepare_tensor_inputs([[3, 4, 5]], None)
    assert ids.shape == mask.shape
    assert mask.dtype == ids.dtype
