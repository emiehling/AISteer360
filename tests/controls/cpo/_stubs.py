"""Test scaffolding for CPO unit/integration tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class StubEmbedder:
    """Maps strings to fixed vectors via prefix matching.

    `category_map` is an ordered dict-like of (prefix, vector) pairs. The first prefix matched determines the embedding;
    if no prefix matches, falls back to `default_vector`.
    """

    def __init__(
        self,
        category_map: dict[str, np.ndarray],
        default_vector: np.ndarray | None = None,
    ) -> None:
        if not category_map:
            raise ValueError("category_map must contain at least one entry.")
        first = next(iter(category_map.values()))
        self._dim = int(first.shape[-1])
        for vec in category_map.values():
            if vec.shape[-1] != self._dim:
                raise ValueError("All category vectors must share the same dimensionality.")
        self.category_map = {prefix: np.asarray(v, dtype=np.float32) for prefix, v in category_map.items()}
        self.default_vector = (
            np.asarray(default_vector, dtype=np.float32)
            if default_vector is not None
            else np.zeros(self._dim, dtype=np.float32)
        )
        self.cleanup_called = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for prefix, vec in self.category_map.items():
                if text.startswith(prefix):
                    out[i] = vec
                    break
            else:
                out[i] = self.default_vector
        return out

    def cleanup(self) -> None:
        self.cleanup_called += 1


class StubTokenizer:
    """Char-level tokenizer with no chat template (uses the `f'{template}\\n\\n{user}'` join path)."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            chr(int(i)) for i in ids
            if not (skip_special_tokens and i in (self.pad_token_id, self.eos_token_id))
        )


def make_synthetic_training_data(
    n_per_category: int = 20,
    *,
    prompt_a: str = "PROMPT_A:",
    prompt_b: str = "PROMPT_B:",
    prompt_bad: str = "PROMPT_BAD:",
    query_a_prefix: str = "AAA",
    query_b_prefix: str = "BBB",
) -> tuple[list[dict], list[str]]:
    """Generate observational triples with a known causal structure.

    Two query categories: A (queries starting with `query_a_prefix`) and B. Three prompts: P_A, P_B, P_bad. Outcome is
    1.0 when a category's "best" prompt is paired, 0.0 otherwise. Each query appears with each prompt to give the DML
    fitter overlap across treatments.
    """
    data: list[dict] = []
    for i in range(n_per_category):
        q_a = f"{query_a_prefix}{i:03d}"
        q_b = f"{query_b_prefix}{i:03d}"
        for q, best in [(q_a, prompt_a), (q_b, prompt_b)]:
            for p in (prompt_a, prompt_b, prompt_bad):
                data.append({"query": q, "prompt": p, "outcome": 1.0 if p == best else 0.0})
    pool = [prompt_a, prompt_b, prompt_bad]
    return data, pool


def fixed_query_embedder() -> StubEmbedder:
    """Embeds 'AAA*' → e_A, 'BBB*' → e_B, else zero."""
    return StubEmbedder(
        category_map={
            "AAA": np.array([1.0, 0.0], dtype=np.float32),
            "BBB": np.array([0.0, 1.0], dtype=np.float32),
        },
        default_vector=np.zeros(2, dtype=np.float32),
    )


def fixed_prompt_embedder() -> StubEmbedder:
    """One-hot per prompt template."""
    return StubEmbedder(
        category_map={
            "PROMPT_A:": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "PROMPT_B:": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "PROMPT_BAD:": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        default_vector=np.zeros(3, dtype=np.float32),
    )
