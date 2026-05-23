"""Tests for the CPO embedder Protocol and `HFMeanPoolEmbedder`."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from aisteer360.algorithms.input_control.cpo.embedder import Embedder, HFMeanPoolEmbedder

from tests.controls.cpo._stubs import StubEmbedder


def _mock_hf_load(hidden_size: int = 8):
    """Patch AutoModel/AutoTokenizer to return deterministic mocks for embedder tests."""

    def fake_tokenize(texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seq_len = max(len(t) for t in texts) if texts else 1
        batch = []
        masks = []
        for t in texts:
            ids = [ord(c) % 100 + 1 for c in t]
            mask = [1] * len(ids)
            ids = ids + [0] * (seq_len - len(ids))
            mask = mask + [0] * (seq_len - len(mask))
            batch.append(ids)
            masks.append(mask)
        encoded = {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

        class _Encoded(dict):
            def to(self, _device):
                return self

        return _Encoded(encoded)

    tokenizer = MagicMock()
    tokenizer.side_effect = fake_tokenize

    def fake_forward(input_ids=None, attention_mask=None, **kwargs):
        # deterministic: hidden state derived from token ids only
        ids = input_ids
        batch, seq_len = ids.shape
        gen = torch.Generator().manual_seed(int(ids.sum().item()))
        hidden = torch.randn(batch, seq_len, hidden_size, generator=gen)
        out = MagicMock()
        out.last_hidden_state = hidden
        return out

    model = MagicMock()
    model.side_effect = fake_forward
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.config = MagicMock()
    model.config.hidden_size = hidden_size
    return tokenizer, model


def test_hf_mean_pool_embedder_shape():
    tokenizer, model = _mock_hf_load(hidden_size=8)
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModel.from_pretrained", return_value=model
    ):
        embedder = HFMeanPoolEmbedder("fake-model")
        emb = embedder.embed(["alpha", "bravo", "charlie"])
    assert emb.shape == (3, 8)
    assert emb.dtype == np.float32


def test_hf_mean_pool_embedder_consistency():
    tokenizer, model = _mock_hf_load(hidden_size=4)
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModel.from_pretrained", return_value=model
    ):
        embedder = HFMeanPoolEmbedder("fake-model")
        emb1 = embedder.embed(["hello"])
        emb2 = embedder.embed(["hello"])
    np.testing.assert_array_equal(emb1, emb2)


def test_hf_mean_pool_embedder_cleanup():
    tokenizer, model = _mock_hf_load(hidden_size=4)
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModel.from_pretrained", return_value=model
    ):
        embedder = HFMeanPoolEmbedder("fake-model")
        assert embedder.model is not None
        embedder.cleanup()
    assert embedder.model is None
    assert embedder.tokenizer is None


def test_embedder_protocol_check():
    stub = StubEmbedder(category_map={"x": np.array([1.0, 0.0], dtype=np.float32)})
    assert isinstance(stub, Embedder)

    class NotAnEmbedder:
        pass

    assert not isinstance(NotAnEmbedder(), Embedder)
