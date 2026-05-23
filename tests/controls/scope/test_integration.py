"""End-to-end integration test for SCOPE.

Validates Phase 1's `is_stateful=True` path against a real method:

  1. Build a `SteeringPipeline` with a stub task LM, stub tokenizer, and SCOPE as input control.
  2. Run `generate(prompt_1)`; verify a rule appears in SCOPE's memory after.
  3. Run `generate(prompt_2)`; verify the steered input passed to the task LM contains the rule from step 2.
  4. Verify `pipeline._last_output` is set.
  5. Verify `reset_session()` clears tactical and preserves strategic.
"""
from __future__ import annotations

import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.scope import SCOPE, SCOPEArgs
from aisteer360.algorithms.input_control.scope.memory import Rule

from tests.controls.scope._stubs import StubReflectionLM


class _RecordingTaskLM(torch.nn.Module):
    """Returns the input plus a fixed echo suffix; records every input it sees."""

    PROBE_LEN = 4

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.seen_inputs: list[torch.Tensor] = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.seen_inputs.append(input_ids.clone())
        head = input_ids[:, : self.PROBE_LEN]
        return torch.cat([input_ids, head], dim=1)


class _CharTokenizer:
    """Char-level tokenizer with no chat template — uses the `f'{system}\\n\\n{user}'` join path."""

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

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **kwargs):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]


def _classifier_response(stream: str, confidence: float) -> str:
    return '{"category": "%s", "confidence": %s}' % (stream, confidence)


def test_scope_end_to_end_with_pipeline():
    tokenizer = _CharTokenizer()
    task_model = _RecordingTaskLM()

    reflection_lm = StubReflectionLM([
        # turn 1: generator + classifier (n_candidates=1, selector skipped)
        "always start with HEY",
        _classifier_response("strategic", 0.95),
        # turn 2 outputs (in case observe runs again)
        "be very concise",
        _classifier_response("strategic", 0.95),
    ])

    args = SCOPEArgs(
        reflection_lm=reflection_lm,
        n_candidates=1,
        confidence_threshold=0.85,
    )
    scope = SCOPE(args)

    pipeline = SteeringPipeline(controls=[scope], lazy_init=True)
    pipeline.model = task_model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    input_ids_1 = torch.tensor([tokenizer.encode("first prompt")], dtype=torch.long)
    pipeline.generate(input_ids_1, max_new_tokens=4)

    assert pipeline._last_output is not None
    assert len(scope.memory.strategic) == 1
    assert scope.memory.strategic[0].text == "always start with HEY"

    input_ids_2 = torch.tensor([tokenizer.encode("second prompt")], dtype=torch.long)
    pipeline.generate(input_ids_2, max_new_tokens=4)

    steered_turn_2 = task_model.seen_inputs[1]
    decoded_turn_2 = tokenizer.decode(steered_turn_2[0])
    assert "always start with HEY" in decoded_turn_2
    assert "second prompt" in decoded_turn_2

    scope.memory.tactical.append(
        Rule(
            text="ephemeral",
            confidence=0.5,
            stream="tactical",
            created_at=0.0,
        )
    )
    assert scope.memory.tactical
    scope.reset_session()
    assert scope.memory.tactical == []
    assert len(scope.memory.strategic) >= 1


def test_scope_registry_pickup():
    """Verify the STEERING_METHOD export shape that the registry crawler reads.

    Note: the global REGISTRY crawler in `aisteer360.algorithms.core.registry` is currently broken (doubled-path
    import error, shared by all input controls — see GEPA's matching test). We assert the export shape directly so
    Phase 5 doesn't depend on that pre-existing bug. When the crawler is fixed, this test can be replaced with a
    lookup via `REGISTRY["input_control"]["scope"]`.
    """
    from aisteer360.algorithms.input_control.scope import STEERING_METHOD

    assert STEERING_METHOD["category"] == "input_control"
    assert STEERING_METHOD["name"] == "scope"
    assert STEERING_METHOD["control"] is SCOPE
    assert STEERING_METHOD["args"] is SCOPEArgs
