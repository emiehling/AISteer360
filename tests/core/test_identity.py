"""Tests for content identity and seeds (`algorithms/core/identity.py`) and the encoder they rest on.

Covers the canonical form of recipes (key order, defaults, names, disabled entries, cross-category
order, int and float), content-addressed artifacts, fail-closed encoding, the equality of identity
and save mode for every artifact type, versioned callables, and `derive_seed`.
"""
import functools

import pytest
import torch

from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
from steerability.algorithms.core.identity import canonical_json, content_id, derive_seed, json_id, versioned
from steerability.algorithms.core.internals.probes.probe import Probe
from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
from steerability.algorithms.core.recipe import Recipe, RecipeError, entry
from steerability.algorithms.input_control.common.memory.pool import PoolMemory
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.gepa.control import GEPA
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.input_control.user_prefix.control import UserPrefix
from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import AdditiveTransform
from steerability.spipe.codec import CodeRef, DecodeContext, EncodeContext, decode, encode
from steerability.spipe.errors import SpipeCodeRefError, SpipeSaveError
from steerability.spipe.store import ArtifactStore

PAIRS = {"positives": ["yes"], "negatives": ["no"]}


@versioned("1")
def scored(response, row):
    return 1.0


def plain(response, row):
    return 0.0


class _Unknown:
    pass


def _vector(scale: float = 1.0) -> SteeringVector:
    return SteeringVector(model_type="llama", directions={1: scale * torch.ones(1, 4)})


class TestRecipeIdentity:
    def test_key_order_defaults_names_disabled_entries_and_order_do_not_change_the_id(self):
        base = Recipe((entry(SystemPrompt, text="a"), entry(CAA, data=PAIRS, layer_id=1)))
        same = [
            Recipe((entry(SystemPrompt, text="a", mode="prepend"), entry(CAA, layer_id=1, data=PAIRS))),
            Recipe((entry(SystemPrompt, name="prompt", text="a"), entry(CAA, data=PAIRS, layer_id=1))),
            Recipe((entry(UserPrefix, text="x", enabled=False), entry(SystemPrompt, text="a"),
                    entry(CAA, data=PAIRS, layer_id=1))),
            Recipe((entry(CAA, data=PAIRS, layer_id=1), entry(SystemPrompt, text="a"))),
        ]
        assert {recipe.id for recipe in same} == {base.id}

    def test_one_and_one_point_zero_give_one_id(self):
        as_int = Recipe((entry(CAA, data=PAIRS, multiplier=1),))
        assert as_int.id == Recipe((entry(CAA, data=PAIRS, multiplier=1.0),)).id

    def test_a_transform_strength_changes_the_id(self):
        def adapter(strength):
            return Recipe((entry(ActivationAdapter, transform=AdditiveTransform(_vector(), strength=strength),
                                 layer_ids=[1]),))

        assert adapter(1.0).id != adapter(2.0).id

    def test_the_text_of_a_text_memory_changes_the_id(self):
        def gepa(text):
            return Recipe((entry(GEPA, memory=TextMemory(slots={"instruction": text})),))

        assert gepa("be brief").id != gepa("be kind").id

    @pytest.mark.parametrize("value", [
        lambda response, row: 1.0,
        functools.partial(plain),
        _Unknown().__init__,
        _Unknown(),
    ], ids=["lambda", "partial", "bound_method", "unknown_object"])
    def test_unnamed_callables_and_unknown_objects_raise(self, value):
        with pytest.raises(SpipeSaveError):
            canonical_json(value)

    def test_a_closure_raises(self):
        def make():
            def inner(response, row):
                return 1.0
            return inner

        with pytest.raises(SpipeSaveError):
            content_id(make())

    def test_a_recipe_with_an_unencodable_value_raises(self):
        with pytest.raises(RecipeError):
            Recipe((entry(CAA, steering_vector=_Unknown(), layer_id=1),)).id


def _probe() -> Probe:
    torch.manual_seed(0)
    return Probe(
        model_type="llama", location="layer_output", pooling="last", layer_ids=[1], weights={1: torch.randn(4)},
        bias=0.1,
    )


def _artifacts(tmp_path) -> dict:
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_model.safetensors").write_bytes(b"weights")
    (lora / "adapter_config.json").write_text("{}")
    (lora / "training_args.bin").write_bytes(b"trainer state")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "sub").mkdir(parents=True)
    (checkpoint / "sub" / "model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer state")
    pool = PoolMemory[dict]()
    pool.add({"prompt": "a"}, polarity="pos")
    return {
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "steering_vector": _vector(),
        "probe": _probe(),
        "probe_set": ProbeSet({"p": _probe()}),
        "text_memory": TextMemory(slots={"instruction": "be brief"}),
        "pool_memory": pool,
        "lora": LoRAArtifact(path=str(lora), base_model="m"),
        "checkpoint": CheckpointArtifact(path=str(checkpoint)),
    }


def test_identity_mode_equals_save_mode_for_every_artifact_type(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    for name, value in _artifacts(tmp_path).items():
        identity = encode(value, EncodeContext(store=None))
        saved = encode(value, EncodeContext(store=store))
        assert identity == saved, name
        assert "$artifact" in identity, name


def test_identity_mode_rehashes_a_vector_after_to():
    vector = _vector()
    before = content_id(vector)
    vector.to("cpu", torch.float64)
    vector.directions[1] = vector.directions[1] * 2
    assert content_id(vector) != before


class TestVersionedCallables:
    def test_the_version_enters_the_encoding_and_the_id(self, monkeypatch):
        encoded = encode(scored, EncodeContext(store=None))
        assert encoded == {"$ref": f"{__name__}:scored", "version": "1"}
        first = content_id(scored)
        monkeypatch.setattr(scored, "__steerability_version__", "2")
        assert content_id(scored) != first

    def test_decoding_a_different_version_raises(self):
        encoded = {"$ref": f"{__name__}:scored", "version": "0"}
        with pytest.raises(SpipeCodeRefError, match="version"):
            decode(encoded, DecodeContext(allow_code=True))
        assert decode({"$ref": f"{__name__}:scored", "version": "1"}, DecodeContext(allow_code=True)) is scored

    def test_a_code_ref_keeps_its_version(self):
        ref = decode({"$ref": f"{__name__}:scored", "version": "1"}, DecodeContext(code_mode="sentinel"))
        assert ref == CodeRef(f"{__name__}:scored", version="1")
        assert content_id(ref) == content_id(scored)

    def test_versioned_rejects_an_empty_version(self):
        with pytest.raises(TypeError):
            versioned("")


class TestContentId:
    def test_json_id_matches_content_id(self):
        value = {"a": [1, 2.0], "b": _vector()}
        assert json_id(encode(value, EncodeContext(store=None))) == content_id(value)

    def test_length_and_nan(self):
        assert len(content_id({"a": 1})) == 16
        assert len(content_id({"a": 1}, length=64)) == 64
        with pytest.raises(ValueError):
            canonical_json({"a": float("nan")})


class TestDeriveSeed:
    def test_pinned_value(self):
        assert derive_seed(7, "op", 3) == 763721040572809648

    def test_range_and_distinctness(self):
        seeds = {derive_seed(1, "op", index) for index in range(64)}
        assert len(seeds) == 64
        assert all(0 <= seed < 2**63 for seed in seeds)
