"""Tests for the evolved `core.output.Output` value type (doc 01 §5)."""
import pytest
import torch

from aisteer360.core.output import Output


class _FakeTokenizer:
    """Minimal tokenizer stub exposing `batch_decode`."""

    def batch_decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return [f"decoded:{row.tolist()}" for row in ids]


def test_requires_ids_or_text():
    with pytest.raises(ValueError, match="at least one"):
        Output()


def test_decode_prefers_text():
    output = Output(output_text=["hello", "world"], output_ids=torch.tensor([[9], [9]]))
    # text is authoritative even when ids are present; tokenizer is not consulted
    assert output.decode() == ["hello", "world"]


def test_decode_falls_back_to_ids():
    output = Output(output_ids=torch.tensor([[1, 2], [3, 4]]))
    assert output.decode(_FakeTokenizer()) == ["decoded:[1, 2]", "decoded:[3, 4]"]


def test_decode_ids_without_tokenizer_raises():
    output = Output(output_ids=torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="tokenizer is required"):
        output.decode()


def test_require_ids_returns_ids():
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(Output(output_ids=ids).require_ids(), ids)


def test_require_ids_names_backend_when_absent():
    output = Output(output_text=["hi"], metadata={"backend": "OpenAIBackend"})
    with pytest.raises(ValueError, match="OpenAIBackend"):
        output.require_ids()


def test_require_ids_generic_message_without_backend():
    output = Output(output_text=["hi"])
    with pytest.raises(ValueError, match="the active backend"):
        output.require_ids()
