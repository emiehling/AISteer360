"""Tests for `core.prompt.Prompt` / `PreparedPrompt`.

`Prompt.classify` must reproduce the pipeline's original `_classify_inputs` dispatch 1:1, including
error cases, across all seven input shapes (single vs batched).
"""
import pytest
import torch

from aisteer360.core.prompt import PreparedPrompt, Prompt
from aisteer360.core.steering_pipeline import SteeringPipeline

# (label, raw input) covering every supported shape, single and batched
_CASES = [
    ("str", "hello"),
    ("list_str", ["a", "b"]),
    ("single_chat", [{"role": "user", "content": "hi"}]),
    ("batch_chat", [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]),
    ("tensor_1d", torch.tensor([1, 2, 3])),
    ("tensor_2d", torch.tensor([[1, 2], [3, 4]])),
    ("list_int", [1, 2, 3]),
    ("list_list_int", [[1, 2], [3, 4]]),
]


def _normalized(prompt: Prompt):
    """The normalized payload for a prompt, mirroring `_classify_inputs`'s third return value."""
    if prompt.modality == "chat":
        return prompt.messages
    if prompt.modality == "text":
        return prompt.texts
    return prompt.token_ids


@pytest.mark.parametrize("label, raw", _CASES, ids=[case[0] for case in _CASES])
def test_classify_matches_pipeline_dispatch(label, raw):
    """`Prompt.classify` returns the same (modality, is_single, normalized) as `_classify_inputs`."""
    modality, is_single, normalized = SteeringPipeline._classify_inputs(raw)
    prompt = Prompt.classify(raw)

    assert prompt.modality == modality
    assert prompt.is_single == is_single

    got = _normalized(prompt)
    if isinstance(normalized, torch.Tensor):
        assert torch.equal(got, normalized)
    else:
        assert got == normalized


def test_classify_tensor_3d_raises():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        Prompt.classify(torch.zeros(2, 2, 2))


def test_classify_empty_list_raises():
    with pytest.raises(ValueError, match="Empty input list"):
        Prompt.classify([])


def test_classify_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported input type"):
        Prompt.classify(3.14)


def test_batch_size():
    assert Prompt.classify("hello").batch_size == 1
    assert Prompt.classify(["a", "b", "c"]).batch_size == 3
    assert Prompt.classify(torch.tensor([[1, 2], [3, 4]])).batch_size == 2
    assert Prompt.classify([[{"role": "user", "content": "a"}]]).batch_size == 1


def test_tensor_modality_retains_attention_mask():
    mask = torch.tensor([[1, 1, 0]])
    prompt = Prompt.classify(torch.tensor([[1, 2, 3]]), attention_mask=mask)
    assert prompt.attention_mask is mask


class TestPreparedPrompt:
    def test_mask_shape_must_match_ids(self):
        with pytest.raises(ValueError, match="does not match"):
            PreparedPrompt(
                prompt=Prompt.classify("hi"),
                adapted_token_ids=torch.tensor([[1, 2, 3]]),
                adapted_attention_mask=torch.tensor([[1, 1]]),
                adaptation_level="tokens",
            )

    def test_mask_without_ids_rejected(self):
        with pytest.raises(ValueError, match="adapted_token_ids is None"):
            PreparedPrompt(
                prompt=Prompt.classify("hi"),
                adapted_attention_mask=torch.tensor([[1, 1]]),
                adaptation_level="tokens",
            )

    def test_valid_prepared_prompt(self):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[1, 1, 1]])
        prepared = PreparedPrompt(
            prompt=Prompt.classify("hi"),
            adapted_token_ids=ids,
            adapted_attention_mask=mask,
            adaptation_level="tokens",
        )
        assert prepared.modality == "text"
        assert prepared.is_single is True
        assert prepared.batch_size == 1
