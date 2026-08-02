"""Tests for the evaluation generation utilities on the unified pipeline path.

Covers `generate_on_pipeline` / `batch_retry_generate` returning aligned `(texts, outputs)` on both the
batched and per-example branches, the `batch_retry_generate` return-shape matrix and retry alignment of
`outputs`, override-column resolution against prompt rows (aligned under retry and expansion, conflict
and missing-column rules), that message-level input controls fire without a bypass warning, that the
adapted prompt reflects a single chat template, bare-model wrapping, the template-less `TypeError`,
left-padding after uneven batches, and `output_record_fields`.
"""
import json
import warnings

import pytest
import torch

from aisteer360.algorithms.core.output import Output
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.utils.rendering import has_chat_template
from aisteer360.evaluation.utils.generation_utils import (
    batch_retry_generate,
    generate_on_pipeline,
    output_record_fields,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
GEN_KWARGS = {"max_new_tokens": 4, "do_sample": False}

_MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _ensure_chat_template(tokenizer):
    """Assign a minimal chat template when the CI tokenizer lacks one (no-op otherwise)."""
    if not has_chat_template(tokenizer):
        tokenizer.chat_template = _MINIMAL_CHAT_TEMPLATE
    return tokenizer


class _NonBatchingInputControl(InputControl):
    """Enabled, prompt-preserving input control that is not batch-safe (forces the fallback branch)."""

    supports_batching = False

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


class _MessageLevelControl(InputControl):
    """Message-level input control; prepends a system turn (engages `adapt_messages`)."""

    supports_batching = True

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def adapt_messages(self, messages, runtime_kwargs=None):
        return [[{"role": "system", "content": "be helpful"}] + list(chat) for chat in messages]


class _RecordingControl(InputControl):
    """Batch-safe input control that records each call's `runtime_kwargs` (for override alignment)."""

    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_runtime_kwargs = []

    def adapt(self, input_ids, runtime_kwargs=None):
        self.seen_runtime_kwargs.append(runtime_kwargs)
        return input_ids


class _RecordingControlB(_RecordingControl):
    """A second recording-control class name, for two-control override-routing tests."""


@pytest.fixture(scope="module")
def batching_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
    pipeline.steer()
    _ensure_chat_template(pipeline.tokenizer)
    return pipeline


@pytest.fixture(scope="module")
def fallback_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_NonBatchingInputControl()])
    pipeline.steer()
    _ensure_chat_template(pipeline.tokenizer)
    return pipeline


@pytest.fixture(scope="module")
def tokenizer(batching_pipeline):
    return batching_pipeline.tokenizer


def _prompt_batch(n: int) -> list[dict]:
    return [{"prompt": f"question {i}"} for i in range(n)]


class TestGenerateOnPipeline:
    """Both branches return aligned `(texts, outputs)` carrying the steered prompt."""

    def test_batched_branch_aligned(self, batching_pipeline):
        assert batching_pipeline.supports_batching
        texts, outputs = generate_on_pipeline(
            batch=_prompt_batch(3), pipeline=batching_pipeline, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        assert len(texts) == len(outputs) == 3
        assert all(isinstance(text, str) for text in texts)
        assert all(isinstance(out, Output) for out in outputs)
        assert all(out.adapted_input_ids is not None for out in outputs)

    def test_fallback_branch_aligned(self, fallback_pipeline):
        assert not fallback_pipeline.supports_batching
        texts, outputs = generate_on_pipeline(
            batch=_prompt_batch(3), pipeline=fallback_pipeline, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        assert len(texts) == len(outputs) == 3
        assert all(isinstance(out, Output) for out in outputs)


class TestNoBypassWarning:
    """A message-level input control fires on the benchmark path with no bypass warning."""

    def test_no_adapt_messages_bypass_warning(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_MessageLevelControl()])
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            batch_retry_generate(
                prompt_data=_prompt_batch(2), model_or_pipeline=pipeline,
                tokenizer=pipeline.tokenizer, gen_kwargs=GEN_KWARGS, return_outputs=True, batch_size=8,
            )
        assert not [w for w in recorded if "adapt_messages" in str(w.message)]

    def test_adapted_prompt_has_single_template(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_MessageLevelControl()])
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)

        _, outputs = generate_on_pipeline(
            batch=_prompt_batch(2), pipeline=pipeline, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        # the message control's injected system content appears exactly once (no re-templating round-trip),
        # and the prompt begins with a single BOS (no double-BOS from re-tokenizing a rendered string)
        bos = pipeline.tokenizer.bos_token
        for output in outputs:
            fields = output_record_fields(output, pipeline.tokenizer)
            assert fields["adapted_prompt"].count("be helpful") == 1
            if bos:
                assert not fields["adapted_prompt"].startswith(bos + bos)


class _CountingParse:
    """parse_fn that returns None on the first call for a target text, then succeeds thereafter."""

    def __init__(self, fail_first_for: str):
        self.fail_first_for = fail_first_for
        self.seen: dict[str, int] = {}

    def __call__(self, text):
        self.seen[text] = self.seen.get(text, 0) + 1
        if text == self.fail_first_for and self.seen[text] == 1:
            return None
        return f"parsed:{text}"


class TestBatchRetryGenerate:
    """Return-shape matrix over (return_raw, return_outputs) and retry alignment of `outputs`."""

    @pytest.mark.parametrize(
        "return_raw,return_outputs,expected_len",
        [
            (False, False, None),  # plain list
            (True, False, 2),
            (False, True, 3),
            (True, True, 3),  # return_outputs wins regardless of return_raw
        ],
    )
    def test_return_shape_matrix(self, batching_pipeline, tokenizer, return_raw, return_outputs, expected_len):
        result = batch_retry_generate(
            prompt_data=_prompt_batch(2),
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            return_raw=return_raw,
            return_outputs=return_outputs,
            batch_size=8,
        )
        if expected_len is None:
            assert isinstance(result, list)
            assert not isinstance(result, tuple)
        else:
            assert isinstance(result, tuple)
            assert len(result) == expected_len
            if return_outputs:
                parsed, raw, outputs = result
                assert len(parsed) == len(raw) == len(outputs) == 2
                assert all(isinstance(out, Output) for out in outputs)

    def test_retry_aligns_outputs_with_final_response(self, batching_pipeline, tokenizer):
        batch = _prompt_batch(3)
        first_texts, _ = generate_on_pipeline(
            batch=batch, pipeline=batching_pipeline, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        parse_fn = _CountingParse(fail_first_for=first_texts[1])

        parsed, raw, outputs = batch_retry_generate(
            prompt_data=batch,
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            parse_fn=parse_fn,
            max_retries=2,
            return_outputs=True,
            batch_size=8,
        )

        assert parse_fn.seen[first_texts[1]] >= 2
        assert len(outputs) == 3
        for index in range(3):
            assert isinstance(outputs[index], Output)
            assert parsed[index] == f"parsed:{raw[index]}"

    def test_bare_model_is_wrapped_and_records_adapted_ids(self, batching_pipeline, tokenizer):
        parsed, raw, outputs = batch_retry_generate(
            prompt_data=_prompt_batch(2),
            model_or_pipeline=batching_pipeline.model,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            return_outputs=True,
            batch_size=8,
        )
        assert len(outputs) == 2
        assert all(out.adapted_input_ids is not None for out in outputs)


class TestOverrideAlignment:
    """Override columns resolve against prompt rows: aligned under retry and expansion."""

    def _pipeline_with_recorder(self):
        recorder = _RecordingControl()
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[recorder])
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)
        return pipeline, recorder

    def test_retry_row_carries_its_own_override(self):
        pipeline, recorder = self._pipeline_with_recorder()
        rows = [{"prompt": "q0", "mark": "A"}, {"prompt": "q1", "mark": "B"}]

        # fail exactly the second row on the first pass, so the retry batch is the single row 1
        seen = {"count": 0}

        def parser(text):
            seen["count"] += 1
            if seen["count"] == 2:  # first pass: row 0 parses, row 1 fails once
                return None
            return f"ok:{text}"

        recorder.seen_runtime_kwargs.clear()
        batch_retry_generate(
            prompt_data=rows,
            model_or_pipeline=pipeline,
            tokenizer=pipeline.tokenizer,
            gen_kwargs=GEN_KWARGS,
            runtime_overrides={"_RecordingControl": {"marks": "mark"}},
            parse_fn=parser,
            max_retries=1,
            batch_size=8,
        )
        # last recorded call is the retry of row 1; its marks must be ["B"], not ["A"]
        assert recorder.seen_runtime_kwargs[-1] == {"marks": ["B"]}

    def test_expansion_maps_per_row(self):
        pipeline, recorder = self._pipeline_with_recorder()
        # a batch longer than any external source; per-row columns map correctly with nothing external consulted
        rows = [{"prompt": f"q{i}", "mark": f"m{i}"} for i in range(5)]
        recorder.seen_runtime_kwargs.clear()
        generate_on_pipeline(
            batch=rows,
            pipeline=pipeline,
            gen_kwargs=GEN_KWARGS,
            runtime_overrides={"_RecordingControl": {"marks": "mark"}},
            batch_size=8,
        )
        # one batched call over five rows: the marks list is the five per-row values in order
        assert recorder.seen_runtime_kwargs[-1] == {"marks": [f"m{i}" for i in range(5)]}

    def test_missing_column_from_every_row_raises(self):
        pipeline, _ = self._pipeline_with_recorder()
        with pytest.raises(ValueError, match="missing from every prompt row"):
            generate_on_pipeline(
                batch=[{"prompt": "q0"}, {"prompt": "q1"}],
                pipeline=pipeline,
                gen_kwargs=GEN_KWARGS,
                runtime_overrides={"_RecordingControl": {"marks": "absent"}},
                batch_size=8,
            )

    def test_missing_from_some_rows_substitutes_empty(self):
        pipeline, recorder = self._pipeline_with_recorder()
        rows = [{"prompt": "q0", "mark": "A"}, {"prompt": "q1"}]  # second row lacks the column
        recorder.seen_runtime_kwargs.clear()
        generate_on_pipeline(
            batch=rows,
            pipeline=pipeline,
            gen_kwargs=GEN_KWARGS,
            runtime_overrides={"_RecordingControl": {"marks": "mark"}},
            batch_size=8,
        )
        assert recorder.seen_runtime_kwargs[-1] == {"marks": ["A", []]}

    def test_same_variable_same_spec_two_controls_accepted(self):
        # two distinct control classes mapping one variable to the same column share the value stream
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_RecordingControl(), _RecordingControlB()])
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)
        generate_on_pipeline(
            batch=[{"prompt": "q0", "mark": "A"}],
            pipeline=pipeline,
            gen_kwargs=GEN_KWARGS,
            runtime_overrides={"_RecordingControl": {"marks": "mark"}, "_RecordingControlB": {"marks": "mark"}},
            batch_size=8,
        )

    def test_same_variable_different_spec_raises(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_RecordingControl(), _RecordingControlB()])
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)
        rows = [{"prompt": "q0", "mark_a": "A", "mark_b": "B"}]
        with pytest.raises(ValueError, match="cannot hold two value streams"):
            generate_on_pipeline(
                batch=rows,
                pipeline=pipeline,
                gen_kwargs=GEN_KWARGS,
                runtime_overrides={
                    "_RecordingControl": {"marks": "mark_a"},
                    "_RecordingControlB": {"marks": "mark_b"},
                },
                batch_size=8,
            )


class TestTemplateLessTokenizer:
    """A message-list prompt with a template-less tokenizer raises `TypeError`."""

    def test_message_list_raises(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
        pipeline.steer()
        pipeline.tokenizer.chat_template = None  # strip any template
        assert not has_chat_template(pipeline.tokenizer)
        with pytest.raises(TypeError, match="no chat"):
            generate_on_pipeline(
                batch=[{"prompt": [{"role": "user", "content": "hi"}]}],
                pipeline=pipeline,
                gen_kwargs=GEN_KWARGS,
                batch_size=8,
            )


class TestLeftPaddingAfterUnevenBatch:
    """A batched run over uneven-length prompts leaves the tokenizer left-padded, with no warning."""

    def test_padding_side_left_and_no_right_pad_warning(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
        pipeline.steer()
        _ensure_chat_template(pipeline.tokenizer)
        pipeline.tokenizer.padding_side = "right"  # start on the wrong side

        rows = [{"prompt": "short"}, {"prompt": "a considerably longer prompt than the first one"}]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            generate_on_pipeline(batch=rows, pipeline=pipeline, gen_kwargs=GEN_KWARGS, batch_size=8)

        assert pipeline.tokenizer.padding_side == "left"
        right_pad_warnings = [
            w for w in recorded
            if "right-padding" in str(w.message) or "right padding" in str(w.message)
        ]
        assert not right_pad_warnings


class TestOutputRecordFields:
    """`output_record_fields` contributes finish_reason always and adapted_prompt only when present."""

    def test_none_output(self, tokenizer):
        assert output_record_fields(None, tokenizer) == {"finish_reason": None}

    def test_pipeline_output_has_adapted_prompt(self, tokenizer):
        out = Output(
            output_ids=torch.tensor([[5, 6]]),
            adapted_input_ids=torch.tensor([[7, 8]]),
            finish_reason="length",
        )
        fields = output_record_fields(out, tokenizer)
        assert fields["finish_reason"] == "length"
        assert isinstance(fields["adapted_prompt"], str)

    def test_raw_model_output_omits_adapted_prompt(self, tokenizer):
        out = Output(output_ids=torch.tensor([[5, 6]]), adapted_input_ids=None, finish_reason="eos")
        fields = output_record_fields(out, tokenizer)
        assert fields["finish_reason"] == "eos"
        assert "adapted_prompt" not in fields


class TestUseCaseSurfacing:
    """End-to-end: instruction_following generation dicts and export carry the new keys."""

    @pytest.fixture
    def use_case_data(self):
        return [
            {
                "prompt": f"Write about topic {i}.",
                "instructions": ["be concise"],
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{}],
            }
            for i in range(2)
        ]

    def _use_case(self, use_case_data):
        from aisteer360.evaluation.use_cases.instruction_following.use_case import InstructionFollowing

        use_case = InstructionFollowing.__new__(InstructionFollowing)
        use_case.evaluation_data = use_case_data
        use_case.evaluation_metrics = []
        return use_case

    def test_generation_dicts_have_new_keys(self, batching_pipeline, tokenizer, use_case_data):
        use_case = self._use_case(use_case_data)
        generations = use_case.generate(
            model_or_pipeline=batching_pipeline, tokenizer=tokenizer, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        assert len(generations) == 2
        for gen in generations:
            assert gen["finish_reason"] in ("eos", "length", "stop", None)
            assert isinstance(gen["adapted_prompt"], str)  # pipeline path always carries the steered prompt

    def test_export_round_trips_new_keys(self, tmp_path, batching_pipeline, tokenizer, use_case_data):
        use_case = self._use_case(use_case_data)
        generations = use_case.generate(
            model_or_pipeline=batching_pipeline, tokenizer=tokenizer, gen_kwargs=GEN_KWARGS, batch_size=8,
        )
        evaluations = {"StrictInstruction": {"follow_all_instructions": [True] * len(generations)}}
        profiles = {"steered": [{"trial_id": 0, "generations": generations, "evaluations": evaluations, "params": {}}]}
        use_case.export(profiles, str(tmp_path))

        with open(tmp_path / "responses.json") as f:
            rows = json.load(f)
        assert len(rows) == 2
        for row in rows:
            assert "steered_finish_reason" in row
            assert "steered_adapted_prompt" in row
