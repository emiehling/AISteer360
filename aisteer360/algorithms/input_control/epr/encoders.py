"""Text encoders for EPR.

Defines the `Encoder` Protocol and ships an `HFEncoder` that wraps any HuggingFace `AutoModel` with `[CLS]` or
attention-mask mean pooling. Used by EPR for both the off-the-shelf dense mode and as the trainable backbone for
contrastive retriever learning.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Encoder(Protocol):
    """Maps a batch of text to a `[batch, dim]` numpy array."""

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HFEncoder:
    """HF model wrapped as an encoder with `[CLS]` or mean pooling.

    `embed` is a non-differentiable, batched encoding loop that returns a numpy array. `forward_torch` is a
    differentiable forward pass returning a `torch.Tensor` for use during contrastive training.

    Args:
        model_name_or_path: HF identifier or local directory.
        pooling: `"cls"` (BERT-family default) or `"mean"` (attention-mask mean pool).
        batch_size: Encoding batch size for `embed`.
        device: Torch device string (e.g. `"cpu"`, `"cuda"`).
        max_length: Token truncation length.
        trainable: If True, the model is left in train mode and gradients are enabled. Used only during the contrastive
            trainer's Stage 2.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling: Literal["cls", "mean"] = "cls",
        batch_size: int = 32,
        device: str = "cpu",
        max_length: int = 256,
        trainable: bool = False,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.pooling = pooling
        self.batch_size = int(batch_size)
        self.device = device
        self.max_length = int(max_length)
        self.trainable = bool(trainable)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(device)

        if not self.trainable:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model.train()

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 0))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batched, non-differentiable encoding to `[N, hidden_size]` float32."""
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        was_training = self.model.training
        self.model.eval()
        outputs: list[np.ndarray] = []
        try:
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
                pooled = self._pool(model_out, encoded["attention_mask"])
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        finally:
            if was_training:
                self.model.train()

        return np.concatenate(outputs, axis=0)

    def forward_torch(self, texts: list[str]) -> torch.Tensor:
        """Differentiable forward pass returning `[N, hidden_size]` on `self.device`.

        Encodes the entire batch in one call (no internal mini-batching). Caller is responsible for choosing a
        manageable batch size.
        """
        if not texts:
            return torch.zeros((0, self.hidden_size), device=self.device)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        model_out = self.model(**encoded)
        pooled = self._pool(model_out, encoded["attention_mask"])
        return pooled

    def _pool(self, model_out, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = getattr(model_out, "last_hidden_state", None)
        if hidden_states is None:
            raise RuntimeError(
                f"HFEncoder: model {self.model_name_or_path!r} does not expose `last_hidden_state`."
            )
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        raise ValueError(f"Unknown pooling {self.pooling!r}")

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def cleanup(self) -> None:
        """Release the underlying model and tokenizer."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
