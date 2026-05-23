"""Integration test for PRewrite — runs the full training loop on tiny models.

This test is marked slow because it instantiates real HF causal LMs and runs (a small number of) PPO-style training
steps. Run via `pytest -m slow tests/controls/prewrite/`.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from aisteer360.algorithms.input_control.prewrite import PRewrite, PRewriteArgs
from aisteer360.algorithms.input_control.prewrite.memory import ModelMemory
from aisteer360.evaluation.metrics.base import Metric


pytestmark = pytest.mark.slow


_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class KeywordReward(Metric):
    """Reward = 1.0 if `response` contains the target token, else 0.0."""

    def __init__(self, target: str = "yes") -> None:
        super().__init__()
        self.target = target

    def compute(self, responses: list[Any], prompts: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        scores = [1.0 if self.target in r.lower() else 0.0 for r in responses]
        return {"score": scores}


def _load_tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(_TINY, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(_TINY, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load {_TINY}: {exc}")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_prewrite_end_to_end_per_query(tmp_path):
    task_model, task_tokenizer = _load_tiny()

    training_data = [
        {"input": "Is the sky blue?"},
        {"input": "Is fire hot?"},
        {"input": "Is grass green?"},
        {"input": "Is water wet?"},
    ]

    args = PRewriteArgs(
        initial_prompt="respond.",
        rewriter_model_name_or_path=_TINY,
        training_data=training_data,
        feedback_metric=KeywordReward(target="yes"),
        mode="per_query",
        n_steps=2,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=1,
        learning_rate=1e-5,
        kl_coef=0.05,
        rewriter_gen_kwargs={"max_new_tokens": 8},
        task_gen_kwargs={"max_new_tokens": 8},
        use_peft=True,
        lora_kwargs={
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        },
        seed=0,
    )

    control = PRewrite(args)
    control.steer(model=task_model, tokenizer=task_tokenizer)

    assert isinstance(control.memory, ModelMemory)

    ids = task_tokenizer.encode("hello", add_special_tokens=False)
    out = control.adapt(ids)
    assert len(out) > len(ids)

    save_path = str(tmp_path / "prewrite_memory")
    control.memory.save(save_path)
    loaded = ModelMemory.load(save_path)
    assert loaded.model is not None
    assert loaded.tokenizer is not None
