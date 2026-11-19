"""Pipeline-seam padding alignment on the batched generate path.

`SteeringPipeline._execute_generation` left-packs the steered prompt tensors after the input
chain, matching the layout the Hugging Face session executes every batched forward in. These
tests drive ragged (unequal-length) batches through the full pipeline-plus-session path, the
only place a right-padded build-time layout and the left-packed forward-time layout can
diverge.

Runs hub-free across the tiny CI models and the `device` fixture; forces `padding_side="right"`
so the divergence appears regardless of a model's tokenizer default.
"""
import torch

from steerability.algorithms.core.internals.model_layout import text_config
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.act_add.control import ActAdd
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.utils.tokenization import to_left_pad


def _cast_control(hidden: int, condition_layer: int) -> CAST:
    """A conditional CAST with a precomputed condition vector and a manual condition point.

    The behavior direction is tiny so generation stays close to the base model; the condition
    layer scores the prompt and gates the (broadcast) behavior addition per row.
    """
    torch.manual_seed(0)
    condition_vector = SteeringVector(
        model_type="test", directions={condition_layer: torch.randn(1, hidden)},
    )
    behavior_vector = SteeringVector(
        model_type="test", directions={0: 0.01 * torch.randn(1, hidden)},
    )
    return CAST(
        behavior_vector=behavior_vector,
        behavior_layer_ids=[0],
        condition_vector=condition_vector,
        condition_point={
            "layer_ids": [condition_layer],
            "threshold": 0.0,
            "comparator": "ge",
            "comparison_mode": "mean",
        },
    )


def test_batched_gate_scores_match_single(model_and_tokenizer, device):
    """Per-row condition scores from a ragged batch match the single-prompt scores.

    The condition hook pools the prefill hidden states over the stored prompt mask. When the
    build-time layout (right-padded) and the forward-time layout (left-packed) disagree, the
    pooled mean for every row shorter than the batch maximum covers pad positions, so the
    batched scores drift from the single-prompt scores for the same prompts.
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        cfg = text_config(model)
        condition_layer = cfg.num_hidden_layers - 1
        control = _cast_control(cfg.hidden_size, condition_layer)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        prompts = ["a b c d e f g h", "a b c", "a b c d e"]  # ragged lengths
        gen = {"max_new_tokens": 2, "do_sample": False}

        single_scores = []
        for prompt in prompts:
            pipeline.generate(text=prompt, **gen)
            single_scores.append(float(control.latest_decision.scores[condition_layer]))

        pipeline.generate(text=prompts, **gen)
        batched_scores = [
            float(score) for score in control.latest_decision.scores_per_row[condition_layer]
        ]

        torch.testing.assert_close(
            torch.tensor(batched_scores), torch.tensor(single_scores), atol=2e-3, rtol=1e-3,
        )
    finally:
        tokenizer.padding_side = original_padding_side


def test_full_sequence_return_has_no_interior_pads(model_and_tokenizer, device):
    """`return_full_sequence=True` returns the left-packed prompt, so short rows carry no interior pads.

    The prompt slice of the returned ids equals `to_left_pad(input_ids, attention_mask)` row for
    row, the layout the continuation was generated from (`[pads, prompt, continuation]`).
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        control = _cast_control(text_config(model).hidden_size, text_config(model).num_hidden_layers - 1)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()

        encoded = tokenizer(["a b c d e f g h", "a b c", "a b c d e"], return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.size(1)

        returned = pipeline.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            do_sample=False,
            return_full_sequence=True,
        )

        expected_prompt, _ = to_left_pad(input_ids, attention_mask)
        assert torch.equal(returned[:, :prompt_len].cpu(), expected_prompt.cpu())
    finally:
        tokenizer.padding_side = original_padding_side


RAGGED_PROMPTS = ["a b c d e f g h", "a b c"]


def _ragged_batch(tokenizer, device):
    """A right-padded two-row batch of unequal lengths, with each row's unpadded ids."""
    encoded = tokenizer(RAGGED_PROMPTS, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    rows = [input_ids[i][attention_mask[i].bool()].unsqueeze(0) for i in range(input_ids.size(0))]
    return input_ids, attention_mask, rows


def _assert_rows_match_singles(batched: torch.Tensor, singles: list[torch.Tensor]) -> None:
    for index, single in enumerate(singles):
        width = single.size(1)
        assert torch.equal(batched[index, :width].cpu(), single[0].cpu()), f"row {index} differs"


def test_from_position_batch_matches_single_rows(model_and_tokenizer, device):
    """A ragged batch steered from an absolute position generates as each row does alone.

    Absolute positions count from each row's first real token, so the short row (left-packed
    behind pads) is not steered at its first real token when `from_position=1`.
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        torch.manual_seed(0)
        vector = SteeringVector(
            model_type="test", directions={0: 8.0 * torch.randn(1, text_config(model).hidden_size)},
        )
        control = CAA(steering_vector=vector, layer_id=0, token_scope="from_position", from_position=1)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        assert pipeline.supports_batching

        input_ids, attention_mask, rows = _ragged_batch(tokenizer, device)
        gen = {"max_new_tokens": 4, "do_sample": False}
        batched = pipeline.generate(input_ids=input_ids, attention_mask=attention_mask, **gen)
        singles = [pipeline.generate(input_ids=row, **gen) for row in rows]
        _assert_rows_match_singles(batched, singles)

        reference = rows[0][:, :3]
        batched_logprobs = pipeline.compute_logprobs(
            input_ids, attention_mask=attention_mask, ref_output_ids=reference,
        )
        for index, row in enumerate(rows):
            single_logprobs = pipeline.compute_logprobs(row, ref_output_ids=reference)
            torch.testing.assert_close(
                batched_logprobs[index].cpu(), single_logprobs[0].cpu(), atol=1e-4, rtol=1e-4,
            )
    finally:
        tokenizer.padding_side = original_padding_side


def test_positional_injection_batch_matches_single_rows(model_and_tokenizer, device):
    """A ragged batch with an aligned positional vector generates as each row does alone.

    The pipeline batches only batch-safe controls, so the test checks the batched path directly.
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        torch.manual_seed(0)
        vector = SteeringVector(
            model_type="test", directions={0: 8.0 * torch.randn(2, text_config(model).hidden_size)},
        )
        control = ActAdd(steering_vector=vector, layer_id=0, alignment=1)
        control.supports_batching = True
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        assert pipeline.supports_batching

        input_ids, attention_mask, rows = _ragged_batch(tokenizer, device)
        gen = {"max_new_tokens": 4, "do_sample": False}
        batched = pipeline.generate(input_ids=input_ids, attention_mask=attention_mask, **gen)
        singles = [pipeline.generate(input_ids=row, **gen) for row in rows]
        _assert_rows_match_singles(batched, singles)
    finally:
        tokenizer.padding_side = original_padding_side


def test_batch_on_a_pipeline_that_is_not_batch_safe_splits_into_single_rows(model_and_tokenizer, device):
    """A batch reaching a control without batch safety generates one prompt per call, in order."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    torch.manual_seed(0)
    vector = SteeringVector(
        model_type="test", directions={0: 8.0 * torch.randn(2, text_config(model).hidden_size)},
    )
    control = ActAdd(steering_vector=vector, layer_id=0, alignment=1)
    control.supports_batching = False
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    gen = {"max_new_tokens": 3, "do_sample": False}
    batched = pipeline.generate(text=RAGGED_PROMPTS, **gen)
    assert batched == [pipeline.generate(text=prompt, **gen) for prompt in RAGGED_PROMPTS]
    outputs = pipeline.generate(text=RAGGED_PROMPTS, return_output=True, **gen)
    assert [output.decode(tokenizer)[0] for output in outputs] == batched
