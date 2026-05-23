"""Test scaffolding for input_control/common tests.

Internal use only -- not exported from `aisteer360/`.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from aisteer360.evaluation.metrics.base import Metric


class StubTokenizer:
    """Minimal tokenizer: char-level encode/decode of integer code points."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids if not (skip_special_tokens and i in (self.pad_token_id, self.eos_token_id)))


class StubModel:
    """Minimal model with a `generate` method that appends a fixed suffix.

    Generates `suffix_ids` after the input. Honors `device` for the returned tensor.
    """

    def __init__(self, suffix_ids: list[int] | None = None, device: torch.device | None = None) -> None:
        self.suffix_ids = suffix_ids if suffix_ids is not None else [ord("X"), ord("Y")]
        self.device = device or torch.device("cpu")

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        suffix = torch.tensor(self.suffix_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
        suffix = suffix.expand(input_ids.size(0), -1)
        return torch.cat([input_ids, suffix], dim=1)


class StubMetric(Metric):
    """Length-based scoring metric returning a single-key dict."""

    def __init__(self, key: str = "score") -> None:
        super().__init__()
        self.key = key

    def compute(
        self,
        responses: list[Any],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {self.key: [float(len(r)) for r in responses]}


def make_canned_lm(responses: list[str]) -> Callable[[str], str]:
    """Return a callable that emits `responses` in order, ignoring its prompt argument.

    Raises StopIteration-style IndexError if called more times than there are responses.
    """
    state = {"i": 0}

    def lm(prompt: str) -> str:
        i = state["i"]
        if i >= len(responses):
            raise IndexError(f"canned lm exhausted after {len(responses)} calls")
        state["i"] = i + 1
        return responses[i]

    return lm


def make_data(prompts: list[str], expected: list[str] | None = None) -> list[dict]:
    """Build a list of `{"input_ids": [...], "expected": ...}` dicts via StubTokenizer encoding."""
    tok = StubTokenizer()
    out = []
    for i, p in enumerate(prompts):
        d = {"input_ids": tok.encode(p)}
        if expected is not None:
            d["expected"] = expected[i]
        out.append(d)
    return out
