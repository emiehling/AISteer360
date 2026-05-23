"""Tests for `Encoder` Protocol and `HFEncoder`."""
from __future__ import annotations

import numpy as np
import pytest

from aisteer360.algorithms.input_control.epr.encoders import Encoder, HFEncoder

BERT_TINY = "prajjwal1/bert-tiny"


def _try_load_bert_tiny() -> HFEncoder | None:
    try:
        return HFEncoder(BERT_TINY, pooling="cls", batch_size=2, device="cpu", max_length=32)
    except Exception:
        return None


@pytest.mark.slow
def test_hf_encoder_cls_pooling_shape():
    encoder = _try_load_bert_tiny()
    if encoder is None:
        pytest.skip("bert-tiny not available")
    out = encoder.embed(["hello world", "another sentence"])
    assert out.shape == (2, encoder.hidden_size)
    encoder.cleanup()


@pytest.mark.slow
def test_hf_encoder_mean_pooling_shape():
    try:
        encoder = HFEncoder(BERT_TINY, pooling="mean", batch_size=2, device="cpu", max_length=32)
    except Exception:
        pytest.skip("bert-tiny not available")
    out = encoder.embed(["a", "bb cc"])
    assert out.shape == (2, encoder.hidden_size)
    encoder.cleanup()


@pytest.mark.slow
def test_hf_encoder_save_and_reload(tmp_path):
    encoder = _try_load_bert_tiny()
    if encoder is None:
        pytest.skip("bert-tiny not available")

    out_path = str(tmp_path / "enc")
    encoder.save_pretrained(out_path)
    embedding_before = encoder.embed(["hello"])

    encoder2 = HFEncoder(out_path, pooling="cls", batch_size=2, device="cpu", max_length=32)
    embedding_after = encoder2.embed(["hello"])
    np.testing.assert_allclose(embedding_before, embedding_after, atol=1e-5)
    encoder.cleanup()
    encoder2.cleanup()


@pytest.mark.slow
def test_hf_encoder_trainable_flag():
    try:
        encoder = HFEncoder(BERT_TINY, pooling="cls", trainable=True, device="cpu", max_length=32)
    except Exception:
        pytest.skip("bert-tiny not available")
    assert encoder.model.training is True
    assert any(p.requires_grad for p in encoder.model.parameters())
    encoder.cleanup()


@pytest.mark.slow
def test_hf_encoder_cleanup():
    encoder = _try_load_bert_tiny()
    if encoder is None:
        pytest.skip("bert-tiny not available")
    encoder.cleanup()
    assert encoder.model is None
    assert encoder.tokenizer is None


def test_encoder_protocol_runtime_check():
    class _Toy:
        def embed(self, texts):
            return np.zeros((len(texts), 1), dtype=np.float32)

    assert isinstance(_Toy(), Encoder)
