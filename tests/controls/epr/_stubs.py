"""Test scaffolding for EPR unit/integration tests."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch


class StubEncoder:
    """Maps text to deterministic vectors via cluster keyword detection.

    `cluster_map` is `{cluster_name: vector}`. Each text is scanned for any cluster name (substring, case-insensitive);
    the first match assigns that cluster's vector. If no match, falls back to `default_vector`.

    This implementation lets unit tests assert "queries in cluster A retrieve cluster A demos" purely structurally —
    no model load, no training.
    """

    def __init__(
        self,
        cluster_map: dict[str, np.ndarray],
        default_vector: np.ndarray | None = None,
    ) -> None:
        if not cluster_map:
            raise ValueError("cluster_map must contain at least one entry.")
        first = next(iter(cluster_map.values()))
        self._dim = int(first.shape[-1])
        for vec in cluster_map.values():
            if vec.shape[-1] != self._dim:
                raise ValueError("All cluster vectors must share the same dimensionality.")
        self.cluster_map = {k.lower(): np.asarray(v, dtype=np.float32) for k, v in cluster_map.items()}
        self.default_vector = (
            np.asarray(default_vector, dtype=np.float32)
            if default_vector is not None
            else np.zeros(self._dim, dtype=np.float32)
        )
        self.cleanup_called = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            assigned = False
            for keyword, vec in self.cluster_map.items():
                if keyword in lower:
                    out[i] = vec
                    assigned = True
                    break
            if not assigned:
                out[i] = self.default_vector
        return out

    def cleanup(self) -> None:
        self.cleanup_called += 1


class StubTokenizer:
    """Char-level tokenizer that supports decode + encode round-trip with no special tokens."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            chr(int(i)) for i in ids
            if not (skip_special_tokens and i in (self.pad_token_id, self.eos_token_id))
        )


def overlap_score_fn(
    anchor_input: str,
    anchor_output: str,
    candidates: list[dict],
    demo_template: str,
) -> np.ndarray:
    """Stub scoring LM that rewards token overlap between candidate and anchor.

    Uses a simple bag-of-words intersection over `candidate.input + candidate.output` vs.
    `anchor.input + anchor.output`. Candidates whose words overlap more with the anchor get higher scores.
    """
    anchor_words = set(_tokenize(f"{anchor_input} {anchor_output}"))
    scores = np.zeros((len(candidates),), dtype=np.float32)
    for i, cand in enumerate(candidates):
        cand_words = set(_tokenize(f"{cand['input']} {cand['output']}"))
        if not cand_words:
            scores[i] = 0.0
            continue
        scores[i] = float(len(anchor_words & cand_words)) / float(len(cand_words))
    return scores


def cluster_score_fn(
    anchor_input: str,
    anchor_output: str,
    candidates: list[dict],
    demo_template: str,
) -> np.ndarray:
    """Stub scoring LM that rewards same-cluster pairs.

    Cluster is the first whitespace-delimited token (e.g., \"MATH\", \"LANG\"). Same-cluster ⇒ high score; different
    cluster ⇒ low score.
    """
    anchor_cluster = _first_word(anchor_input)
    scores = np.zeros((len(candidates),), dtype=np.float32)
    for i, cand in enumerate(candidates):
        cand_cluster = _first_word(cand["input"])
        scores[i] = 1.0 if cand_cluster == anchor_cluster else 0.0
    return scores


def make_clustered_corpus(
    clusters: dict[str, int],
    seed: int = 0,
) -> list[dict]:
    """Build a synthetic corpus with cluster-tagged examples.

    `clusters` is `{cluster_name: n_examples}`. Each example's input starts with the cluster name and is followed by a
    unique numeric suffix; the output is a cluster-specific token.
    """
    rng = np.random.default_rng(seed)
    corpus: list[dict] = []
    for cluster, n in clusters.items():
        for i in range(n):
            extra = rng.integers(0, 1_000_000)
            corpus.append({
                "input": f"{cluster} item {i:03d} ({extra})",
                "output": f"{cluster.lower()}_label_{i:03d}",
            })
    return corpus


def _tokenize(text: str) -> Iterable[str]:
    return re.findall(r"\w+", text.lower())


def _first_word(text: str) -> str:
    parts = text.strip().split()
    return parts[0] if parts else ""
