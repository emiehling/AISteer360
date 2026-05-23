"""Tests for TextMemory dataclass and serialization."""
import json
import os

import pytest

from aisteer360.algorithms.input_control.common.memory import Memory, TextMemory


def test_text_memory_defaults():
    """A TextMemory() with no args has expected default values."""
    memory = TextMemory()
    assert memory.model_type == "text"
    assert memory.instruction is None
    assert memory.demonstrations is None
    assert memory.template is None
    assert memory.extras == {}


def test_text_memory_round_trip(tmp_path):
    """save then load produces an equal TextMemory (all fields, including model_type)."""
    original = TextMemory(
        instruction="Be helpful.",
        demonstrations=[{"input": "x", "output": "y"}],
        template="{directive}\n{example_blocks}",
        extras={"version": 1, "tag": "test"},
    )
    path = str(tmp_path / "memory.tmem")
    original.save(path)

    loaded = TextMemory.load(path)
    assert loaded.model_type == original.model_type
    assert loaded.instruction == original.instruction
    assert loaded.demonstrations == original.demonstrations
    assert loaded.template == original.template
    assert loaded.extras == original.extras


def test_text_memory_extension_appended(tmp_path):
    """save("memory") writes to "memory.tmem"; load("memory") finds it."""
    memory = TextMemory(instruction="test")
    base_path = str(tmp_path / "memory")
    memory.save(base_path)

    assert os.path.exists(base_path + ".tmem")
    assert not os.path.exists(base_path)

    loaded = TextMemory.load(base_path)
    assert loaded.instruction == "test"


def test_text_memory_load_rejects_wrong_model_type(tmp_path):
    """Loading a JSON file whose model_type != 'text' raises ValueError."""
    bad_path = str(tmp_path / "bad.tmem")
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump({"model_type": "model", "instruction": "x"}, f)

    with pytest.raises(ValueError, match="model_type"):
        TextMemory.load(bad_path)


def test_text_memory_with_demonstrations_preserves_labels(tmp_path):
    """Round-trip preserves _label keys in demonstrations."""
    original = TextMemory(
        demonstrations=[
            {"input": "good", "_label": "positive"},
            {"input": "bad", "_label": "negative"},
        ],
    )
    path = str(tmp_path / "labeled")
    original.save(path)
    loaded = TextMemory.load(path)

    assert loaded.demonstrations is not None
    assert loaded.demonstrations[0]["_label"] == "positive"
    assert loaded.demonstrations[1]["_label"] == "negative"


def test_text_memory_model_type_is_immutable():
    """model_type cannot be set via the constructor (it's init=False)."""
    with pytest.raises(TypeError):
        TextMemory(model_type="something_else")


def test_text_memory_is_a_memory():
    """isinstance(TextMemory(), Memory) is True (Protocol check)."""
    assert isinstance(TextMemory(), Memory)
