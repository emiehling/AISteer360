"""Tests for input control base classes (real `aisteer360` implementations)."""
import warnings

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import NoInputControl
from aisteer360.algorithms.input_control.common.memory import TextMemory
from aisteer360.algorithms.input_control.few_shot.control import FewShot

POS_POOL = [
    {"input": "The service was excellent.", "label": "Positive"},
    {"input": "I had a wonderful time!", "label": "Positive"},
]
NEG_POOL = [
    {"input": "Terrible support; I regret this.", "label": "Negative"},
]


def test_no_input_control_is_identity():
    """`NoInputControl.adapt` returns input_ids unchanged for any prior value."""
    control = NoInputControl()

    input_ids = torch.tensor([[1, 2, 3, 4]])

    # no prior
    result = control.adapt(input_ids, prior=None, runtime_kwargs={})
    assert result is input_ids

    # with prior
    prior = Output(output_ids=torch.tensor([[5, 6]]), runtime_kwargs=None)
    result = control.adapt(input_ids, prior=prior, runtime_kwargs={"k": "v"})
    assert result is input_ids

    # list input
    list_input = [1, 2, 3]
    result = control.adapt(list_input, prior=None, runtime_kwargs=None)
    assert result is list_input


def test_no_input_control_observe_is_no_op():
    """The default `observe` is safely callable and returns None."""
    control = NoInputControl()
    output = Output(output_ids=torch.tensor([[1, 2, 3]]), runtime_kwargs=None)
    result = control.observe(
        input_ids=torch.tensor([[4, 5, 6]]),
        output=output,
        runtime_kwargs=None,
    )
    assert result is None


def test_few_shot_memory_is_none_before_steer():
    """A freshly constructed FewShot has memory is None."""
    fewshot = FewShot(
        directive="Test directive",
        positive_example_pool=POS_POOL,
        k_positive=1,
    )
    assert fewshot.memory is None


def test_few_shot_populates_memory_at_steer(model_and_tokenizer):
    """After pipeline.steer(), FewShot.memory is a TextMemory whose instruction matches directive and whose
    demonstrations is the labeled union of the pools."""
    _, tokenizer = model_and_tokenizer

    fewshot = FewShot(
        directive="Be precise.",
        positive_example_pool=POS_POOL,
        negative_example_pool=NEG_POOL,
        k_positive=1,
        k_negative=1,
    )
    pipeline = SteeringPipeline(controls=[fewshot], lazy_init=True)
    pipeline.tokenizer = tokenizer
    pipeline.model = object()  # unused for this assertion path
    fewshot.steer(model=None, tokenizer=tokenizer)

    assert isinstance(fewshot.memory, TextMemory)
    assert fewshot.memory.instruction == "Be precise."

    demos = fewshot.memory.demonstrations
    assert demos is not None
    assert len(demos) == len(POS_POOL) + len(NEG_POOL)

    positive_count = sum(1 for d in demos if d.get("_label") == "positive")
    negative_count = sum(1 for d in demos if d.get("_label") == "negative")
    assert positive_count == len(POS_POOL)
    assert negative_count == len(NEG_POOL)


def test_few_shot_adapt_reads_from_memory(model_and_tokenizer, device):
    """Replacing control.memory after steer() with a different TextMemory changes adapt()'s output."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    fewshot = FewShot(
        directive="Original directive",
        positive_example_pool=POS_POOL,
        k_positive=1,
    )
    pipeline = SteeringPipeline(controls=[fewshot], lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    prompt_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)

    # baseline adapt with the steer-built memory
    baseline = fewshot.adapt(prompt_ids, runtime_kwargs={})
    baseline_text = tokenizer.decode(
        baseline[0].tolist() if baseline.ndim == 2 else baseline.tolist(),
        skip_special_tokens=True,
    )

    # swap memory in-place: change instruction and demonstrations
    fewshot.memory = TextMemory(
        instruction="Completely different directive xyzzy",
        demonstrations=[
            {"input": "FOOBAR", "label": "Positive", "_label": "positive"},
        ],
        template=fewshot.memory.template,
    )

    swapped = fewshot.adapt(prompt_ids, runtime_kwargs={})
    swapped_text = tokenizer.decode(
        swapped[0].tolist() if swapped.ndim == 2 else swapped.tolist(),
        skip_special_tokens=True,
    )

    assert "xyzzy" in swapped_text or "FOOBAR" in swapped_text
    assert baseline_text != swapped_text


def test_few_shot_no_pools_no_demos(model_and_tokenizer):
    """FewShot constructed without any pools produces a memory.demonstrations of None and adapt returns input
    unchanged with the existing UserWarning."""
    _, tokenizer = model_and_tokenizer

    fewshot = FewShot(directive="Just a directive, no examples")
    fewshot.steer(model=None, tokenizer=tokenizer)

    assert isinstance(fewshot.memory, TextMemory)
    assert fewshot.memory.demonstrations is None
    assert fewshot.memory.instruction == "Just a directive, no examples"

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fewshot.adapt(input_ids, runtime_kwargs={})

    assert result is input_ids
    assert any("No examples provided" in str(w.message) for w in caught)
