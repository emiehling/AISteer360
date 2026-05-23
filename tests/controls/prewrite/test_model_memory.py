"""Tests for `ModelMemory`."""
from __future__ import annotations

import json
import os

import pytest
import torch

from aisteer360.algorithms.input_control.common.memory.base import Memory
from aisteer360.algorithms.input_control.prewrite.memory import ModelMemory


_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load {_TINY_MODEL}: {exc}")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_model_memory_defaults():
    memory = ModelMemory(model_name_or_path="dummy")
    assert memory.model_type == "model"
    assert memory.model is None
    assert memory.tokenizer is None
    assert memory.extras == {}


def test_model_memory_save_directory_structure(tmp_path):
    model, tokenizer = _load_tiny_model()
    memory = ModelMemory(
        model_name_or_path=_TINY_MODEL,
        model=model,
        tokenizer=tokenizer,
        extras={"mode": "per_query", "use_peft": False},
    )
    base = str(tmp_path / "memory")
    memory.save(base)

    saved_dir = base + ".mmem"
    assert os.path.isdir(saved_dir)
    assert os.path.isdir(os.path.join(saved_dir, "model"))
    assert os.path.isdir(os.path.join(saved_dir, "tokenizer"))
    assert os.path.isfile(os.path.join(saved_dir, "meta.json"))

    with open(os.path.join(saved_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_type"] == "model"
    assert meta["model_name_or_path"] == _TINY_MODEL
    assert meta["extras"]["mode"] == "per_query"


def test_model_memory_round_trip_full(tmp_path):
    model, tokenizer = _load_tiny_model()
    memory = ModelMemory(
        model_name_or_path=_TINY_MODEL,
        model=model,
        tokenizer=tokenizer,
        extras={"use_peft": False},
    )
    base = str(tmp_path / "rt")
    memory.save(base)

    encoded = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        before = model(**encoded).logits

    loaded = ModelMemory.load(base)
    assert loaded.model is not None
    assert loaded.tokenizer is not None

    with torch.no_grad():
        after = loaded.model(**encoded).logits

    torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)


def test_model_memory_round_trip_peft(tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    model, tokenizer = _load_tiny_model()
    lora_cfg = LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.eval()

    memory = ModelMemory(
        model_name_or_path=_TINY_MODEL,
        model=peft_model,
        tokenizer=tokenizer,
        extras={"use_peft": True},
    )
    base = str(tmp_path / "peft_rt")
    memory.save(base)

    saved_dir = base + ".mmem"
    model_files = os.listdir(os.path.join(saved_dir, "model"))
    assert any("adapter" in f for f in model_files)

    encoded = tokenizer("hello", return_tensors="pt")
    with torch.no_grad():
        before = peft_model(**encoded).logits

    loaded = ModelMemory.load(base)
    assert loaded.model is not None
    with torch.no_grad():
        after = loaded.model(**encoded).logits
    torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)


def test_model_memory_load_rejects_wrong_type(tmp_path):
    model, tokenizer = _load_tiny_model()
    memory = ModelMemory(
        model_name_or_path=_TINY_MODEL, model=model, tokenizer=tokenizer,
    )
    base = str(tmp_path / "wrong")
    memory.save(base)
    saved_dir = base + ".mmem"

    meta_path = os.path.join(saved_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["model_type"] = "text"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="model_type"):
        ModelMemory.load(base)


def test_model_memory_satisfies_memory_protocol():
    memory = ModelMemory(model_name_or_path="dummy")
    assert isinstance(memory, Memory)


def test_model_memory_cleanup():
    model, tokenizer = _load_tiny_model()
    memory = ModelMemory(
        model_name_or_path=_TINY_MODEL, model=model, tokenizer=tokenizer,
    )
    memory.cleanup()
    assert memory.model is None
    assert memory.tokenizer is None
