"""Text embedders for CPO.

Defines the `Embedder` Protocol and ships a default `HFMeanPoolEmbedder` backed by any HuggingFace model that exposes
`last_hidden_state`. Users may inject any object satisfying the Protocol (e.g., a `sentence-transformers` wrapper).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Maps a batch of text to a `[batch, dim]` numpy array."""

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HFMeanPoolEmbedder:
    """Default embedder: HF model with mean-pooled hidden states (excluding padding tokens).

    Works with both encoder-only (BERT/RoBERTa) and decoder-only (mean-pool the final layer) models.

    Args:
        model_name_or_path: HF model identifier or local path.
        batch_size: Encoding batch size.
        device: Torch device string (e.g., "cpu", "cuda").
        max_length: Token truncation length.
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.device = device
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            hidden = getattr(self.model.config, "hidden_size", 0)
            return np.zeros((0, hidden), dtype=np.float32)

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                model_out = self.model(**encoded)

            hidden_states = getattr(model_out, "last_hidden_state", None)
            if hidden_states is None:
                raise RuntimeError(
                    f"HFMeanPoolEmbedder: model {self.model_name_or_path!r} does not expose `last_hidden_state`."
                )

            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
            outputs.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)

    def cleanup(self) -> None:
        """Release the underlying model and tokenizer."""
        import torch

        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
