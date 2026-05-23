"""End-to-end integration test for the GEPA control."""
from __future__ import annotations

import torch

from aisteer360.algorithms.core.registry import REGISTRY
from aisteer360.algorithms.input_control.gepa import GEPA, GEPAArgs


TARGET = "target"
PROBE_LEN = len(TARGET)


class _InstructionEchoModel:
    """Returns the FIRST `PROBE_LEN` tokens of the input as new tokens.

    Combined with `_NoChatTokenizer` and an adapter that prepends the instruction, the model output reflects the
    instruction prefix -- letting the test's similarity metric measure how close the candidate instruction is to TARGET.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        head = input_ids[:, :PROBE_LEN]
        return torch.cat([input_ids, head], dim=1)


class _NoChatTokenizer:
    """Char-level tokenizer with no chat template (uses `f'{instruction}\\n\\n{user}'` join)."""

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


def _similarity_to_target(s: str) -> float:
    """Position-wise match against TARGET, normalized to [0, 1]."""
    if not s:
        return 0.0
    matches = sum(1 for i, c in enumerate(TARGET) if i < len(s) and s[i] == c)
    return matches / len(TARGET)


class _SimilarityFeedbackMetric:
    def compute_with_feedback(self, responses, references=None, prompts=None):
        out = []
        for r in responses:
            score = _similarity_to_target(r[:PROBE_LEN])
            out.append({
                "score": score,
                "feedback": f"You said {r[:PROBE_LEN]!r}, expected {TARGET!r}.",
            })
        return out


def _make_step_closer_lm():
    """Reflection LM that returns a string one char closer to TARGET than the parent's instruction."""

    def lm(prompt: str) -> str:
        marker = "Current instruction:\n"
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\n", start)
        current = prompt[start:end].strip()
        if current == TARGET:
            return TARGET
        chars = list(current)
        for i, ch in enumerate(TARGET):
            if i >= len(chars):
                chars.append(ch)
                break
            if chars[i] != ch:
                chars[i] = ch
                break
        else:
            chars = chars[: len(TARGET)]
        return "".join(chars[: len(TARGET)])

    return lm


def test_gepa_registry_pickup():
    assert "gepa" in REGISTRY["input_control"]
    entry = REGISTRY["input_control"]["gepa"]
    assert entry.control_cls is GEPA
    assert entry.args_cls is GEPAArgs


def test_gepa_end_to_end_converges():
    tokenizer = _NoChatTokenizer()
    model = _InstructionEchoModel()

    train_data = [{"input_ids": tokenizer.encode("q1"), "id": "t1"}]
    val_data = [
        {"input_ids": tokenizer.encode("v1"), "id": "v1"},
        {"input_ids": tokenizer.encode("v2"), "id": "v2"},
    ]

    args = GEPAArgs(
        seed_instruction="start!",  # length matches TARGET; first char wrong
        feedback_metric=_SimilarityFeedbackMetric(),
        reflection_lm=_make_step_closer_lm(),
        max_metric_calls=200,
        train_data=train_data,
        val_data=val_data,
        reflection_minibatch_size=1,
        use_merge=False,
        skip_perfect_score=True,
        seed=0,
    )
    gepa = GEPA(args)
    gepa.steer(model=model, tokenizer=tokenizer)

    assert gepa.memory is not None
    assert gepa.memory.instruction == TARGET
    assert gepa._reflection_lm is None  # cleanup ran


def test_gepa_cleanup_idempotent():
    tokenizer = _NoChatTokenizer()
    model = _InstructionEchoModel()
    args = GEPAArgs(
        seed_instruction="start!",
        feedback_metric=_SimilarityFeedbackMetric(),
        reflection_lm=_make_step_closer_lm(),
        max_metric_calls=20,
        train_data=[{"input_ids": tokenizer.encode("q"), "id": "t"}],
        val_data=[{"input_ids": tokenizer.encode("v"), "id": "v"}],
        reflection_minibatch_size=1,
        use_merge=False,
    )
    gepa = GEPA(args)
    gepa.steer(model=model, tokenizer=tokenizer)
    gepa.cleanup()  # second call should not raise
    assert gepa._reflection_lm is None
