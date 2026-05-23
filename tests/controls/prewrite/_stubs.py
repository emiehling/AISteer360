"""Test scaffolding for PRewrite unit tests."""
from __future__ import annotations

from typing import Any

import torch

from aisteer360.evaluation.metrics.base import Metric


class _StubBatch(dict):
    """Dict subclass that mirrors HF's `BatchEncoding.to(device)`."""

    def to(self, device):
        new = _StubBatch()
        for k, v in self.items():
            new[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return new


class StubTokenizer:
    """Char-level tokenizer with no chat template."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return _StubBatch(input_ids=torch.tensor([ids], dtype=torch.long))
        return ids

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return _StubBatch(input_ids=torch.tensor([ids], dtype=torch.long))
        return _StubBatch(input_ids=ids)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            chr(int(i)) for i in ids
            if not (skip_special_tokens and i in (self.pad_token_id, self.eos_token_id))
        )


class StubFeedbackMetric(Metric):
    """Returns a configurable score per call. Records the calls it received for assertion."""

    def __init__(self, score: float | list[float] = 0.5) -> None:
        super().__init__()
        self.score = score
        self.calls: list[dict] = []
        self._call_index = 0

    def compute(
        self,
        responses: list[Any],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"responses": responses, "kwargs": kwargs})
        if isinstance(self.score, list):
            value = self.score[self._call_index % len(self.score)]
            self._call_index += 1
        else:
            value = self.score
        return {"score": value}


class StubRewriter:
    """Instrumented rewriter exposed where a PreTrainedModel is expected.

    Records each `generate` call. Returns a deterministic prefix-based "rewrite".
    """

    def __init__(self, response_text: str = "REWRITTEN") -> None:
        self.response_text = response_text
        self.generate_calls: list[dict] = []
        self._device = torch.device("cpu")
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        # synthesize response token ids by encoding self.response_text as ints
        response_ids = torch.tensor(
            [[ord(c) for c in self.response_text]], dtype=input_ids.dtype, device=input_ids.device,
        )
        return torch.cat([input_ids, response_ids], dim=1)


class StubTaskLM:
    """Frozen task-LM stub with deterministic response."""

    def __init__(self, response_text: str = "TASK_RESPONSE") -> None:
        self.response_text = response_text
        self.generate_calls: list[dict] = []
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        response_ids = torch.tensor(
            [[ord(c) for c in self.response_text]], dtype=input_ids.dtype, device=input_ids.device,
        )
        return torch.cat([input_ids, response_ids], dim=1)


class StubPRewriteTrainer:
    """Bypasses real PPO; records args and returns a fake trained rewriter.

    Constructed with the same kwargs the real trainer accepts.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.train_called = False
        self.training_data: list[dict] | None = None
        self.rewriter_model = kwargs["rewriter_model"]
        self.rewriter_tokenizer = kwargs["rewriter_tokenizer"]
        self.mode = kwargs["mode"]

    def train(self, training_data):
        self.train_called = True
        self.training_data = list(training_data)
        return self.rewriter_model

    def _generate_one_rewrite(self):
        return "STATIC_REWRITTEN_INSTRUCTION"
