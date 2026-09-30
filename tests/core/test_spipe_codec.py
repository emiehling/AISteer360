"""Codec round-trips and security gates for the `.spipe` value codec."""
import io
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.payloads import ConstraintSource
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.spipe.codec import CodeRef, DataRef, DecodeContext, EncodeContext, decode, digest_of, encode
from steerability.spipe.errors import (
    SpipeCodeRefError,
    SpipeFormatError,
    SpipeIntegrityError,
    SpipeSaveError,
    SpipeStaleError,
)
from steerability.spipe.store import ArtifactStore

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def module_level_scorer(response, row):
    return float(len(response))


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def roundtrip(value, store, *, allow_code=False):
    ctx = EncodeContext(store=store)
    encoded = encode(value, ctx)
    return decode(encoded, DecodeContext(store=store, allow_code=allow_code)), encoded


def test_plain_values_pass_through(store):
    value = {"a": 1, "b": [1.5, "x", None, True], "c": {"nested": [1, 2]}}
    decoded, encoded = roundtrip(value, store)
    assert decoded == value
    assert encoded == value


def test_nonstring_keys_roundtrip_via_map(store):
    value = {1: [0, 2], 5: [1]}
    decoded, encoded = roundtrip(value, store)
    assert decoded == value
    assert "$map" in encoded


def test_map_encoding_and_digest_are_key_order_independent():
    forward = encode({10: 1.0, 5: 2.0}, EncodeContext(store=None))
    backward = encode({5: 2.0, 10: 1.0}, EncodeContext(store=None))
    assert forward == backward
    assert digest_of({10: 1.0, 5: 2.0}) == digest_of({5: 2.0, 10: 1.0})


def test_dataclass_roundtrip(store):
    pairs = ContrastivePairs(positives=["a", "b"], negatives=["c", "d"], prompts=["p", "q"])
    decoded, encoded = roundtrip(pairs, store)
    assert encoded["$dc"].endswith("ContrastivePairs")
    assert decoded == pairs


def test_enum_and_dtype_roundtrip(store):
    from peft import PeftType

    from steerability.algorithms.core.execution.contracts import Capability

    decoded, _ = roundtrip(PeftType.LORA, store)
    assert decoded is PeftType.LORA
    decoded, _ = roundtrip(Capability.IN_PROCESS_TORCH, store)
    assert decoded is Capability.IN_PROCESS_TORCH
    decoded, _ = roundtrip(torch.bfloat16, store)
    assert decoded is torch.bfloat16


def test_tensor_roundtrip_via_store(store):
    tensor = torch.randn(3, 4)
    decoded, encoded = roundtrip(tensor, store)
    assert "$artifact" in encoded
    assert torch.allclose(decoded, tensor)


def test_steering_vector_roundtrip(store):
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    vector = SteeringVector(
        model_type="llama",
        directions={1: torch.randn(1, 8), 3: torch.randn(1, 8)},
        explained_variances={1: 0.5, 3: 0.25},
        meta={"location": "layer_output"},
    )
    ctx = EncodeContext(store=store)
    encoded = encode(vector, ctx)
    decoded = decode(encoded, DecodeContext(store=store))
    assert decoded.model_type == "llama"
    assert sorted(decoded.directions) == [1, 3]
    assert torch.allclose(decoded.directions[3], vector.directions[3])
    assert decoded.explained_variances == {1: 0.5, 3: 0.25}
    assert decoded.meta == {"location": "layer_output"}


def test_direction_class_artifacts_decode_verified(store):
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    vector = SteeringVector(model_type="llama", directions={1: torch.randn(1, 8)})
    ctx = EncodeContext(store=store)
    ctx.artifact_fields = {"artifact_class": "direction", "source": "ContrastiveFit", "fit_digest": "0" * 12}
    encoded = encode(vector, ctx)
    decoded = decode(encoded, DecodeContext(store=store, verify="strict"))
    assert isinstance(decoded, VerifiedPrecomputed)
    assert decoded.artifact_class == "direction"


def test_reserved_dollar_key_rejected(store):
    with pytest.raises(SpipeSaveError, match="reserved"):
        encode({"$evil": 1}, EncodeContext(store=store))


def test_lambda_rejected_with_naming_hint(store):
    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode(lambda x: x, EncodeContext(store=store))


def test_partial_and_bound_method_rejected(store):
    import functools

    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode(functools.partial(module_level_scorer, "x"), EncodeContext(store=store))
    with pytest.raises(SpipeSaveError, match="module-level name"):
        encode("abc".upper, EncodeContext(store=store))


def test_ref_gating_both_directions(store):
    ctx = EncodeContext(store=store)
    encoded = encode(module_level_scorer, ctx)
    assert encoded == {"$ref": f"{__name__}:module_level_scorer"}
    assert ctx.code_refs

    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        decode(encoded, DecodeContext(store=store, allow_code=False))
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True))
    assert decoded is module_level_scorer
    sentinel = decode(encoded, DecodeContext(store=store, allow_code=False, code_mode="sentinel"))
    assert isinstance(sentinel, CodeRef)
    with pytest.raises(SpipeCodeRefError):
        sentinel("x", {})


def test_sentinel_ref_never_imports_under_allow_code(store):
    # the digest-only decode path (code_mode="sentinel") must not import the target even when
    # allow_code is granted, so a $ref to an absent module still yields the inert CodeRef
    encoded = {"$ref": "steerability_absent_module_xyz:some_fn"}
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True, code_mode="sentinel"))
    assert decoded == CodeRef("steerability_absent_module_xyz:some_fn")
    assert encode(decoded, EncodeContext(store=None)) == encoded


def test_dc_import_gating(store):
    encoded = {"$dc": "os.path.sep", "fields": {}}
    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        decode(encoded, DecodeContext(store=store, allow_code=False))


@pytest.mark.parametrize("allow_code", [False, True])
def test_dc_refuses_toolkit_callables(store, monkeypatch, allow_code):
    import steerability.utils.optional as optional

    calls = []
    monkeypatch.setattr(optional, "require", lambda *args, **kwargs: calls.append((args, kwargs)))
    encoded = {"$dc": "steerability.utils.optional.require", "fields": {"module_name": "os"}}
    with pytest.raises(SpipeFormatError, match="steerability.utils.optional.require"):
        decode(encoded, DecodeContext(store=store, allow_code=allow_code))
    assert not calls

    store_class = {"$dc": "steerability.spipe.store.ArtifactStore", "fields": {"root": "unused"}}
    with pytest.raises(SpipeFormatError, match="dataclass"):
        decode(store_class, DecodeContext(store=store, allow_code=allow_code))


def test_dc_check_applies_to_trusted_names(store, monkeypatch):
    import steerability.spipe.codec as codec

    monkeypatch.setattr(codec, "_trusted_dc_qualnames", lambda: frozenset({"collections.OrderedDict"}))
    with pytest.raises(SpipeFormatError, match="collections.OrderedDict"):
        decode({"$dc": "collections.OrderedDict", "fields": {}}, DecodeContext(store=store))


def test_decodable_dataclasses_are_toolkit_dataclasses():
    from dataclasses import is_dataclass
    from importlib import import_module

    from steerability.spipe.codec import DECODABLE_DATACLASSES

    for qualname in DECODABLE_DATACLASSES:
        module_name, _, name = qualname.rpartition(".")
        cls = getattr(import_module(module_name), name)
        assert is_dataclass(cls)
        assert f"{cls.__module__}.{cls.__qualname__}" == qualname


# toolkit dataclasses whose construction can run code; `$dc` constructs them only under `allow_code`
CODE_GATED_DATACLASSES = frozenset({
    # resolving the model config honors a manifest-supplied `trust_remote_code`
    "steerability.algorithms.core.execution.spec.BackendSpec",
})


def test_decodable_dataclasses_cover_the_registered_args():
    import ast
    import inspect
    import typing
    from collections.abc import Callable
    from dataclasses import fields, is_dataclass

    from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
    from steerability.algorithms.core.internals.probes import Probe, ProbeSet
    from steerability.algorithms.core.registry import REGISTRY
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated
    from steerability.algorithms.output_control.routed_decoding.actions import Generate, Prefix, Respond
    from steerability.algorithms.output_control.routed_decoding.routing import Predicate, Route, Router
    from steerability.algorithms.state_control.common import sources
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.selectors.base import BaseSelector
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector
    from steerability.algorithms.state_control.common.transforms.base import BaseTransform
    from steerability.spipe.codec import DECODABLE_DATACLASSES

    def type_checking_names(module) -> dict:
        """The names `module` imports under `if TYPE_CHECKING:`, for resolving its string annotations."""
        names: dict = {}
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.If) and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
                imports = [statement for statement in node.body if isinstance(statement, (ast.Import, ast.ImportFrom))]
                exec(compile(ast.Module(imports, type_ignores=[]), module.__file__, "exec"), dict(vars(module)), names)
        return names

    def field_hints(cls: type) -> dict:
        return typing.get_type_hints(cls, localns=type_checking_names(inspect.getmodule(cls)))

    # types the encoder writes as `$artifact`, `$component`, `$data`, or `$ref` rather than `$dc`
    non_dc_types = (
        CheckpointArtifact, LoRAArtifact, PoolMemory, Probe, ProbeSet, SteeringVector, TextMemory,
        BaseSelector, BaseTransform, Gate, Predicate, Route, Router, CodeRef, DataRef,
    )
    # dataclasses that reach the codec through slots typed `Any` or as a protocol: route actions
    # (`Route.action`, `Router.default_action`) and the artifact sources of transforms
    untyped_roots = (
        Fixed, Generated, Generate, Prefix, Respond,
        sources.ConditionPointSearch, sources.ContrastiveFit, sources.LayerFilteredFit, sources.SinglePairFit,
    )
    found: set = set()

    def collect_dc_encoded(annotation) -> None:
        """Collect the toolkit dataclasses that a value of type `annotation` can contain and that
        encode as `$dc`, following unions, containers, and dataclass fields."""
        if typing.get_origin(annotation) in (typing.Literal, Callable):
            return
        if isinstance(annotation, type) and is_dataclass(annotation):
            toolkit = annotation.__module__.startswith("steerability.")
            if not toolkit or issubclass(annotation, non_dc_types) or annotation in found:
                return
            found.add(annotation)
            hints = field_hints(annotation)
            for dataclass_field in fields(annotation):
                if dataclass_field.init:
                    collect_dc_encoded(hints[dataclass_field.name])
            return
        for argument in typing.get_args(annotation):
            collect_dc_encoded(argument)

    for methods in REGISTRY.values():
        for method in methods.values():
            if method.args_cls is None:
                continue
            hints = field_hints(method.args_cls)
            for dataclass_field in fields(method.args_cls):
                if dataclass_field.init:
                    collect_dc_encoded(hints[dataclass_field.name])
    for root in untyped_roots:
        collect_dc_encoded(root)

    reached = {f"{cls.__module__}.{cls.__qualname__}" for cls in found}
    assert {"steerability.algorithms.core.execution.payloads.ConstraintSource",
            "steerability.algorithms.state_control.common.fit_specs.VectorTrainSpec"} <= reached
    assert sorted(reached - DECODABLE_DATACLASSES - CODE_GATED_DATACLASSES) == []
    assert not CODE_GATED_DATACLASSES & DECODABLE_DATACLASSES


def phase_text(prompt_text, params):
    return "x"


def saved_and_loaded(directory, control, *, freeze=False, **load_kwargs):
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.spipe import SPipe

    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[control]).to_spipe(freeze=freeze)
    return SPipe.load(spipe.save(directory / "bundle.spipe"), **load_kwargs)


@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("source", [
    ConstraintSource(kind="regex", value="a|b"),
    {"kind": "choice", "value": ["a", "b"]},
], ids=["object", "mapping"])
def test_constraint_sources_load_without_allow_code(tmp_path, source, freeze):
    from steerability.algorithms.core.execution.payloads import as_constraint_source
    from steerability.algorithms.output_control.constrained_decoding import ConstrainedDecoding

    loaded = saved_and_loaded(tmp_path, ConstrainedDecoding(source=source), freeze=freeze)
    (control,) = loaded.pipeline().output_controls
    assert control.source == as_constraint_source(source)
    (inspected,) = loaded.instantiate_entry(0, lenient=True)
    assert inspected.source == control.source


def test_raw_phase_plan_routes_load_without_allow_code(tmp_path):
    from steerability.algorithms.core.internals.probes import Probe, ProbeSet
    from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated
    from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router, generate

    weights = torch.randn(16, generator=torch.Generator().manual_seed(0))
    probes = ProbeSet({"always": Probe(
        model_type="llama", location="layer_input", pooling="mean", layer_ids=[1],
        weights={1: weights / weights.norm()}, bias=1e9, meta={},
    )})

    def routed(plan):
        rules = Router([Route("raw", when=P("always"), action=plan)], default_action=generate())
        return RoutedDecoding(probes=probes, rules=rules)

    plan = [Generated(budget=2, until_token_ids=(3,)), Fixed("x"), Generated()]
    loaded = saved_and_loaded(tmp_path, routed(plan))
    (control,) = loaded.pipeline().output_controls
    assert list(control.rules.routes[0].action) == plan
    (inspected,) = loaded.instantiate_entry(0, lenient=True)
    assert list(inspected.rules.routes[0].action) == plan

    gated = saved_and_loaded(tmp_path / "callable", routed([Fixed(phase_text), Generated()]))
    with pytest.raises(SpipeCodeRefError, match="phase_text"):
        gated.pipeline()


def test_dc_trusts_the_resolved_class_rather_than_the_target(store):
    from peft import LoraConfig

    reexport = {
        "$dc": "steerability.algorithms.structural_control.wrappers.trl.base_mixin.LoraConfig",
        "fields": {"r": 4},
    }
    with pytest.raises(SpipeCodeRefError, match="peft.tuners"):
        decode(reexport, DecodeContext(store=store, allow_code=False))
    decoded = decode(reexport, DecodeContext(store=store, allow_code=True))
    assert isinstance(decoded, LoraConfig) and decoded.r == 4


def remote_code_model_dir(root, marker):
    """A model directory whose config maps `AutoConfig` to a module that writes `marker` on import."""
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "marker",
        "auto_map": {"AutoConfig": "configuration_marker.MarkerConfig"},
    }))
    (root / "configuration_marker.py").write_text(
        "from pathlib import Path\n\n"
        "from transformers import PretrainedConfig\n\n"
        f"Path({str(marker)!r}).write_text('imported')\n\n\n"
        "class MarkerConfig(PretrainedConfig):\n"
        "    model_type = 'marker'\n"
    )
    return root


def test_dc_refuses_toolkit_dataclasses_that_run_code(store, tmp_path, monkeypatch):
    import transformers.dynamic_module_utils as dynamic_module_utils

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    marker = tmp_path / "marker"
    record = store.put_tree(remote_code_model_dir(tmp_path / "model", marker), {"type": "CheckpointArtifact"})
    backend = {"$dc": "steerability.algorithms.core.execution.spec.BackendSpec", "fields": {
        "kind": "vllm",
        "model": {"$artifact": record.id, "as": "path"},
        "options": {"trust_remote_code": True},
    }}
    with pytest.raises(SpipeCodeRefError, match="BackendSpec"):
        decode(backend, DecodeContext(store=store, allow_code=False))
    spipe = recipe_spipe([recipe_entry("input_control/system_prompt", {"text": backend})], store=store)
    for lenient in (True, False):
        with pytest.raises(SpipeCodeRefError, match="BackendSpec"):
            spipe.instantiate_entry(0, lenient=lenient)
    assert not marker.exists()


CONTRASTIVE_PAIRS = "steerability.algorithms.core.internals.data.ContrastivePairs"
MODEL_ACCESS = "steerability.algorithms.core.execution.access.ModelAccess"
DECISION = {"$component": "decision", "params": {"name": "a"}}
PROBE_SET_FIT = "steerability.algorithms.core.internals.probes.probe_set.ProbeSetFit"

MALFORMED_VALUES = {
    "dc_target_not_a_string": {"$dc": 5},
    "dc_missing_module": {"$dc": "steerability.no_such_module.Thing", "fields": {}},
    "dc_missing_attribute": {"$dc": "steerability.spipe.codec.NoSuchClass", "fields": {}},
    "dc_fields_not_an_object": {"$dc": CONTRASTIVE_PAIRS, "fields": [1, 2]},
    "dc_unknown_field": {"$dc": CONTRASTIVE_PAIRS, "fields": {"unknown": 1}},
    "dc_rejected_field": {"$dc": CONTRASTIVE_PAIRS, "fields": {"positives": 5, "negatives": 5}},
    "dc_post_init_attribute_error": {"$dc": PROBE_SET_FIT, "fields": {"data": {"a": 1}, "spec": "x"}},
    "enum_unknown_member": {"$dc": MODEL_ACCESS, "value": "NOPE"},
    "enum_dunder_member": {"$dc": MODEL_ACCESS, "value": "__class__"},
    "enum_list_member": {"$dc": MODEL_ACCESS, "value": ["FACTS"]},
    "enum_missing_member": {"$dc": MODEL_ACCESS},
    "dtype_list": {"$dc": "torch.dtype", "value": ["float32"]},
    "map_not_a_list": {"$map": 5},
    "map_short_entry": {"$map": [[1]]},
    "map_unhashable_key": {"$map": [[{"a": 1}, 1]]},
    "component_kind_not_a_string": {"$component": 5},
    "component_params_not_an_object": {"$component": "fixed_layer", "params": [1]},
    "component_missing_param": {"$component": "route", "params": {"when": DECISION}},
    "and_of_strings": {"$component": "and", "params": {"left": "a", "right": "b"}},
    "or_of_a_predicate_and_an_int": {"$component": "or", "params": {"left": DECISION, "right": 3}},
    "not_of_an_int": {"$component": "not", "params": {"operand": 3}},
    "gate_artifact_not_an_object": {"$component": "gate", "params": {}, "artifact": "x"},
    "ref_not_a_string": {"$ref": 5},
    "data_not_an_object": {"$data": 5},
    "data_value_not_a_string": {"$data": {"kind": "hf", "repo_id": ["org/data"]}},
}


@pytest.mark.parametrize("value", MALFORMED_VALUES.values(), ids=MALFORMED_VALUES.keys())
def test_malformed_values_raise_format_errors(store, value):
    for allow_code in (False, True):
        with pytest.raises(SpipeFormatError, match=r"args\.text"):
            decode(value, DecodeContext(store=store, allow_code=allow_code), "args.text")
    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}),
        recipe_entry("input_control/system_prompt", {"text": value}),
    ])
    assert spipe.instantiate_entry(0)
    for lenient in (False, True):
        with pytest.raises(SpipeFormatError, match=r"args\.text"):
            spipe.instantiate_entry(1, lenient=lenient)


def test_control_constructor_lookup_errors_are_reported_per_entry():
    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}),
        recipe_entry("output_control/constrained_decoding", {"source": {"value": "a"}}),
    ])
    assert spipe.instantiate_entry(0)
    for lenient in (False, True):
        with pytest.raises(SpipeFormatError, match=r"controls\[1\].*do not construct"):
            spipe.instantiate_entry(1, lenient=lenient)


def test_live_model_refused(store):
    import torch.nn as nn

    with pytest.raises(SpipeSaveError, match="name_or_path"):
        encode(nn.Linear(2, 2), EncodeContext(store=store))


def test_data_ref_roundtrip_kept(store):
    ref = DataRef(kind="hf", repo_id="org/data", split="train")
    ctx = EncodeContext(store=store)
    encoded = encode(ref, ctx)
    assert encoded == {"$data": {"kind": "hf", "repo_id": "org/data", "split": "train"}}
    kept = decode(encoded, DecodeContext(store=store, data_mode="keep"))
    assert kept == ref


def test_hf_dataset_encodes_opaque(store):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": ["a", "b"]})
    encoded = encode(ds, EncodeContext(store=store))
    assert encoded["$data"]["kind"] == "opaque"
    assert encoded["$data"]["fingerprint"] == ds._fingerprint


def test_component_transform_roundtrip(store):
    from steerability.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

    transform = NormPreservingTransform(AdditiveTransform({1: torch.randn(1, 8)}, strength=2.5))
    ctx = EncodeContext(store=store)
    encoded = encode(transform, ctx)
    assert encoded["$component"] == "norm_preserving"
    assert encoded["inner"]["$component"] == "additive"
    decoded = decode(encoded, DecodeContext(store=store))
    assert isinstance(decoded, NormPreservingTransform)
    assert decoded.inner.strength == 2.5
    assert torch.allclose(decoded.inner.directions[1], transform.inner.directions[1])


def test_component_gate_roundtrip(store):
    from steerability.algorithms.state_control.common.gating import (
        Evidence,
        Gate,
        PerKeyThreshold,
        ProjectedCosineReadout,
    )

    directions = {1: torch.randn(8)}
    gate = Gate(
        Evidence((1,), ProjectedCosineReadout(directions), pooling="mean"),
        PerKeyThreshold(threshold=0.4, comparator="ge", aggregate="any"),
    )
    ctx = EncodeContext(store=store)
    encoded = encode(gate, ctx)
    assert encoded["$component"] == "gate"
    decoded = decode(encoded, DecodeContext(store=store))
    assert isinstance(decoded, Gate)
    assert decoded.evidence.layer_ids == (1,)
    assert decoded.rule.threshold == 0.4
    pooled = torch.randn(2, 8)
    assert torch.allclose(decoded.evidence.readout(pooled, 1), gate.evidence.readout(pooled, 1))


def test_callable_readout_gate_refused(store):
    from steerability.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate, SumThreshold

    gate = Gate(Evidence((1,), CallableReadout(lambda pooled, lid: pooled[:, 0])), SumThreshold())
    with pytest.raises(ValueError, match="CallableReadout"):
        encode(gate, EncodeContext(store=store))


def test_selector_roundtrip(store):
    from steerability.algorithms.state_control.common.selectors import FractionalDepthSelector

    decoded, encoded = roundtrip(FractionalDepthSelector(fraction=0.4, minimum=1), store)
    assert encoded["$component"] == "fractional_depth"
    assert decoded.fraction == 0.4 and decoded.minimum == 1


def test_as_path_decodes_to_payload_path(store, tmp_path):
    src = tmp_path / "product"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"abc")
    from steerability.algorithms.core.execution.payloads import CheckpointArtifact

    encoded = encode(CheckpointArtifact(path=str(src)), EncodeContext(store=store))
    assert encoded.get("as") == "path"
    decoded = decode(encoded, DecodeContext(store=store))
    assert (pytest.importorskip("pathlib").Path(decoded) / "weights.bin").read_bytes() == b"abc"


def test_pickle_backed_memory_gated_behind_allow_code(store):
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory

    pool = PoolMemory(items=["a", "b"], metadata={"score": [1.0, 2.0]})
    encoded = encode(pool, EncodeContext(store=store))
    with pytest.raises(SpipeCodeRefError, match="pickled"):
        decode(encoded, DecodeContext(store=store, allow_code=False))
    decoded = decode(encoded, DecodeContext(store=store, allow_code=True))
    assert decoded.items == ["a", "b"]


# content gate for tree artifacts
def torch_saved(obj, **kwargs) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer, **kwargs)
    return buffer.getvalue()


def write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def payload_files(payload: str) -> list[str]:
    root = Path(payload)
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_relabelled_pickle_tree_is_refused_without_allow_code(store, tmp_path):
    source = write_tree(tmp_path / "memory", {"scorer.pkl": pickle.dumps({"w": [1.0]}), "cache.json": b"{}"})
    record = store.put_tree(source, {"type": "CheckpointArtifact"})
    ref = {"$artifact": record.id, "as": "path"}
    with pytest.raises(SpipeCodeRefError, match="scorer.pkl"):
        decode(ref, DecodeContext(store=store, allow_code=False))
    assert payload_files(decode(ref, DecodeContext(store=store, allow_code=True))) == ["cache.json", "scorer.pkl"]


@pytest.mark.parametrize("name, legacy", [
    ("pytorch_model.bin", False),
    ("pytorch_model.bin", True),
    ("optimizer.pt", False),
    ("scorer.joblib", False),
    ("nested/memory.pkl.gz", False),
])
def test_pickle_bearing_files_are_refused(store, tmp_path, name, legacy):
    data = torch_saved({"w": torch.ones(2)}, _use_new_zipfile_serialization=not legacy)
    source = write_tree(tmp_path / "product", {"config.json": b"{}", name: data})
    record = store.put_tree(source, {"type": "CheckpointArtifact"})
    with pytest.raises(SpipeCodeRefError, match=name):
        decode({"$artifact": record.id, "as": "path"}, DecodeContext(store=store))


def test_lora_tree_decodes_without_allow_code(store, tmp_path):
    from safetensors.torch import save

    from steerability.algorithms.core.execution.payloads import LoRAArtifact

    adapter = write_tree(tmp_path / "adapter", {
        "adapter_model.safetensors": save({"w": torch.ones(2)}),
        "adapter_config.json": b"{}",
        "weights.bin": b"not a pickle",
    })
    encoded = encode(LoRAArtifact(path=str(adapter), base_model=TINY_MODEL), EncodeContext(store=store))
    decoded = decode(encoded, DecodeContext(store=store, allow_code=False))
    assert payload_files(decoded) == ["adapter_config.json", "adapter_model.safetensors", "weights.bin"]


def test_trainer_state_is_left_out_of_frozen_trees(store, tmp_path):
    from safetensors.torch import save

    from steerability.algorithms.core.execution.payloads import LoRAArtifact

    weights = save({"w": torch.ones(2)})
    trainer_state = torch_saved({"lr": 1e-4})
    adapter = write_tree(tmp_path / "adapter", {
        "adapter_model.safetensors": weights,
        "adapter_config.json": b"{}",
        "training_args.bin": trainer_state,
        "checkpoint-2/adapter_model.safetensors": weights,
        "checkpoint-2/optimizer.pt": trainer_state,
        "checkpoint-2/rng_state.pth": trainer_state,
        "checkpoint-2/scheduler.pt": trainer_state,
        "checkpoint-2/trainer_state.json": b"{}",
        "checkpoint-2/training_args.bin": trainer_state,
    })
    encoded = encode(LoRAArtifact(path=str(adapter), base_model=TINY_MODEL), EncodeContext(store=store))
    assert payload_files(decode(encoded, DecodeContext(store=store, allow_code=False))) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint-2/adapter_model.safetensors",
        "checkpoint-2/trainer_state.json",
    ]


def test_frozen_pickled_weights_warn_and_need_allow_code(store, tmp_path, caplog):
    from steerability.algorithms.core.execution.payloads import CheckpointArtifact

    checkpoint = write_tree(tmp_path / "checkpoint", {
        "config.json": b"{}",
        "pytorch_model.bin": torch_saved({"w": torch.ones(2)}),
    })
    with caplog.at_level(logging.WARNING, logger="steerability.spipe.codec"):
        encoded = encode(CheckpointArtifact(path=str(checkpoint)), EncodeContext(store=store))
    assert "pytorch_model.bin" in caplog.text and "allow_code" in caplog.text
    with pytest.raises(SpipeCodeRefError, match="pytorch_model.bin"):
        decode(encoded, DecodeContext(store=store))
    assert "pytorch_model.bin" in payload_files(decode(encoded, DecodeContext(store=store, allow_code=True)))


def test_sidecar_must_match_the_manifest_record(store, tmp_path):
    source = write_tree(tmp_path / "memory", {"scorer.pkl": pickle.dumps({"w": [1.0]}), "cache.json": b"{}"})
    record = store.put_tree(source, {"type": "CPOMemory"})
    sidecar_path = store.root / record.id.replace(":", "-", 1) / "artifact.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["type"] = "CheckpointArtifact"
    sidecar_path.write_text(json.dumps(sidecar))

    ctx = DecodeContext(store=store, allow_code=True, manifest_records={record.id: record})
    with pytest.raises(SpipeIntegrityError, match="CPOMemory"):
        decode({"$artifact": record.id, "as": "path"}, ctx)


def test_artifact_references_need_well_formed_ids(store):
    traversal = "sha256:x/../../outside"
    for value in (
        {"$artifact": traversal},
        {"$artifact": 5},
        {"$component": "gate", "params": {}, "artifact": {"$artifact": traversal}},
        {"$component": "gate", "params": {}, "artifact": traversal},
    ):
        with pytest.raises(SpipeFormatError, match="artifact"):
            decode(value, DecodeContext(store=store))
    thin_gate = {"$component": "gate", "params": {}, "artifact": {"$artifact": "sha256:" + "0" * 64}}
    with pytest.raises(SpipeIntegrityError, match="thin bundle"):
        decode(thin_gate, DecodeContext(store=None))


def test_directory_bundles_read_artifacts_only_inside_the_bundle(tmp_path):
    from steerability.spipe import SPipe
    from steerability.spipe.format import write_manifest
    from steerability.spipe.store import tree_id_for

    outside = write_tree(tmp_path / "outside", {"memory.json": b'{"instruction": "outside the bundle"}'})
    tree_id = tree_id_for(outside)
    linked = tmp_path / "linked"
    artifact = linked / "artifacts" / tree_id.replace(":", "-", 1)
    artifact.mkdir(parents=True)
    (artifact / "artifact.json").write_text(json.dumps({"id": tree_id, "encoding": "tree", "type": "TextMemory"}))
    (artifact / "payload").symlink_to(outside)
    prewrite = recipe_entry("input_control/prewrite", {"memory": {"$artifact": tree_id}})
    write_manifest(recipe_spipe([prewrite]).manifest, linked)
    with pytest.raises(SpipeFormatError, match="symlink"):
        SPipe.load(linked)

    # a symlinked `artifacts/` directory is rejected even when its artifacts match their ids
    external = ArtifactStore(tmp_path / "external")
    assert external.put_tree(outside, {"type": "TextMemory"}).id == tree_id
    linked_root = tmp_path / "linked_root"
    linked_root.mkdir()
    write_manifest(recipe_spipe([prewrite]).manifest, linked_root)
    (linked_root / "artifacts").symlink_to(external.root)
    with pytest.raises(SpipeFormatError, match="symlink"):
        SPipe.load(linked_root)

    # the planted sidecar is not read because a traversal id is never joined onto the store root
    (tmp_path / "planted").mkdir()
    (tmp_path / "planted" / "artifact.json").write_text("not json")
    reference = {"$artifact": "sha256:x/../../../planted"}
    bundle = recipe_spipe([recipe_entry("input_control/system_prompt", {"text": reference})]).save(tmp_path / "bundle")
    (bundle / "artifacts" / "sha256-x").mkdir(parents=True)
    loaded = SPipe.load(bundle)
    with pytest.raises(SpipeFormatError, match="not an artifact id"):
        loaded.instantiate_entry(0)


def test_digest_mode_stable_across_roundtrip(store):
    pairs = ContrastivePairs(positives=["a", "b"], negatives=["c", "d"])
    fit_input = {"data": pairs, "scorer": module_level_scorer, "tensor": torch.ones(2, 2, dtype=torch.bfloat16)}
    before = digest_of(fit_input)

    ctx = EncodeContext(store=store)
    encoded = encode(fit_input, ctx)
    decoded = decode(encoded, DecodeContext(store=store, allow_code=False, code_mode="sentinel"))
    assert before == digest_of(decoded)


def test_unhandled_object_raises_in_strict_mode(store):
    class Opaque:
        pass

    with pytest.raises(SpipeSaveError, match="no serialized form"):
        encode(Opaque(), EncodeContext(store=store))
    # digest mode reduces to a type name instead
    assert digest_of(Opaque()) == digest_of(Opaque())


@pytest.fixture(scope="module")
def frozen_caa_spipe():
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.algorithms.state_control.caa.control import CAA

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    caa = CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    return pipeline.to_spipe()


def test_manifest_record_governs_verification_wrap(tmp_path, frozen_caa_spipe):
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed
    from steerability.spipe import SPipe

    thin = frozen_caa_spipe.save(tmp_path / "thin_dir", artifacts="thin")
    external = frozen_caa_spipe.save(tmp_path / "fat_dir") / "artifacts"
    # a shared external store may hold a sidecar written by another bundle for the same content
    (record,) = (
        record
        for entry in frozen_caa_spipe.manifest["controls"]
        for record in entry["resolved"]["artifacts"].values()
    )
    assert record["artifact_class"] == "direction"
    sidecar_path = external / record["id"].replace(":", "-", 1) / "artifact.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.update(artifact_class="opaque", source=None, fit_digest=None)
    sidecar_path.write_text(json.dumps(sidecar))

    rebuilt = SPipe.load(thin, artifact_store=external).pipeline()
    source = rebuilt.state_controls[0].steering_vector
    assert isinstance(source, VerifiedPrecomputed)
    assert source.artifact_class == "direction"


def test_relabelled_sidecar_fails_integrity_checks(tmp_path, frozen_caa_spipe):
    from steerability.spipe import SPipe

    fat = frozen_caa_spipe.save(tmp_path / "fat_dir")
    loaded = SPipe.load(fat)
    (record,) = frozen_caa_spipe.manifest["controls"][0]["resolved"]["artifacts"].values()
    sidecar_path = fat / "artifacts" / record["id"].replace(":", "-", 1) / "artifact.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["type"] = "Tensor"
    sidecar_path.write_text(json.dumps(sidecar))

    report = loaded.verify()
    assert not report.ok
    assert any("SteeringVector" in error for error in report.errors)
    with pytest.raises(SpipeIntegrityError, match="SteeringVector"):
        loaded.pipeline()
    with pytest.raises(SpipeIntegrityError, match="SteeringVector"):
        loaded.instantiate_entry(0)
    with pytest.raises(SpipeIntegrityError, match="SteeringVector"):
        SPipe.load(fat)


def test_verify_and_describe_report_a_symlinked_artifact(tmp_path, frozen_caa_spipe):
    from steerability.spipe import SPipe

    bundle = frozen_caa_spipe.save(tmp_path / "bundle")
    loaded = SPipe.load(bundle)
    assert loaded.verify().ok
    for tensor_file in (bundle / "artifacts").glob("*/artifact.safetensors"):
        copy = tmp_path / tensor_file.parent.name
        tensor_file.rename(copy)
        tensor_file.symlink_to(copy)

    report = loaded.verify()
    assert not report.ok
    assert any("symlink" in error for error in report.errors)
    assert "unreadable" in loaded.describe()


def test_external_artifact_stores_are_read_as_given(tmp_path, frozen_caa_spipe):
    from steerability.spipe import SPipe

    fat = frozen_caa_spipe.save(tmp_path / "fat")
    thin = frozen_caa_spipe.save(tmp_path / "thin", artifacts="thin")
    linked_store = tmp_path / "linked_store"
    linked_store.mkdir()
    for artifact in (fat / "artifacts").iterdir():
        (linked_store / artifact.name).symlink_to(artifact)

    for artifact_store in (linked_store, lambda artifact_id: linked_store / artifact_id.replace(":", "-", 1)):
        loaded = SPipe.load(thin, artifact_store=artifact_store)
        assert loaded.verify().ok
        assert loaded.instantiate_entry(0)


# per-entry instantiation
def recipe_entry(method, args, *, enabled=True):
    return {"method": method, "enabled": enabled, "args": args, "resolved": None}


def recipe_spipe(controls, *, allow_code=False, store=None):
    from steerability.spipe import SPipe

    manifest = {
        "format": "spipe/1",
        "created_at": "2026-08-25T00:00:00Z",
        "toolkit_version": "0.5.2",
        "code_dependent": False,
        "model": {"ref": TINY_MODEL, "revision": None},
        "controls": controls,
        "lock": None,
    }
    return SPipe(manifest, store=store, base_dir=None, allow_code=allow_code)


def test_entries_enumerate_the_manifest():
    from steerability.spipe import SpipeEntry

    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}),
        recipe_entry("input_control/external_marker", {"marker": "[m]"}, enabled=False),
    ])
    entries = spipe.entries
    assert all(isinstance(entry, SpipeEntry) for entry in entries)
    assert [(entry.index, entry.method, entry.enabled, entry.resolved) for entry in entries] == [
        (0, "input_control/system_prompt", True, False),
        (1, "input_control/external_marker", False, False),
    ]
    entries[0].args["text"] = "edited"
    assert spipe.entries[0].args == {"text": "Be brief."}


def test_frozen_entry_reports_its_resolution(frozen_caa_spipe):
    (entry,) = frozen_caa_spipe.entries
    assert entry.method == "state_control/caa"
    assert entry.resolved


def test_instantiate_entry_isolates_an_unregistered_method():
    from steerability.algorithms.input_control.system_prompt.control import SystemPrompt

    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}, enabled=False),
        recipe_entry("input_control/external_marker", {"marker": "[m]"}),
    ])
    (control,) = spipe.instantiate_entry(0)
    assert isinstance(control, SystemPrompt)
    assert control.enabled is False
    with pytest.raises(SpipeFormatError, match="external_marker"):
        spipe.instantiate_entry(1)
    with pytest.raises(SpipeFormatError, match="external_marker"):
        spipe.pipeline()
    with pytest.raises(IndexError):
        spipe.instantiate_entry(2)


def test_load_leaves_an_unregistered_frozen_entry_to_instantiate_entry(tmp_path, frozen_caa_spipe):
    from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
    from steerability.spipe import SPipe
    from steerability.spipe.format import MANIFEST_NAME

    bundle = frozen_caa_spipe.save(tmp_path / "fat_dir")
    manifest = json.loads((bundle / MANIFEST_NAME).read_text())
    manifest["controls"][0]["method"] = "state_control/external_caa"
    manifest["controls"].insert(0, recipe_entry("input_control/system_prompt", {"text": "Be brief."}))
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest))

    loaded = SPipe.load(bundle)
    (control,) = loaded.instantiate_entry(0)
    assert isinstance(control, SystemPrompt)
    for prefer in ("frozen", "recipe"):
        with pytest.raises(SpipeFormatError, match="external_caa"):
            loaded.instantiate_entry(1, prefer=prefer)
    with pytest.raises(SpipeFormatError, match="external_caa"):
        loaded.pipeline()
    report = loaded.verify()
    assert report.ok
    assert any("staleness check could not run" in warning and "external_caa" in warning for warning in report.warnings)


def edited_bundle(tmp_path, frozen_spipe, edit):
    from steerability.spipe.format import MANIFEST_NAME

    bundle = frozen_spipe.save(tmp_path / "fat_dir")
    manifest = json.loads((bundle / MANIFEST_NAME).read_text())
    edit(manifest)
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest))
    return bundle


@pytest.mark.parametrize("name, value, match", [
    ("train_spec", {"$dc": "steerability.no_such_module.Spec", "fields": {}}, "no_such_module"),
    ("unknown_field", 1, "unknown_field"),
], ids=["malformed_value", "rejected_by_the_constructor"])
def test_load_leaves_a_malformed_frozen_entry_to_instantiate_entry(tmp_path, frozen_caa_spipe, name, value, match):
    from steerability.spipe import SPipe

    def edit(manifest):
        manifest["controls"][0]["args"][name] = value

    bundle = edited_bundle(tmp_path, frozen_caa_spipe, edit)
    loaded = SPipe.load(bundle)
    for prefer in ("frozen", "recipe"):
        with pytest.raises(SpipeFormatError, match=match):
            loaded.instantiate_entry(0, prefer=prefer)
    report = loaded.verify()
    assert report.ok
    assert any("could not run" in warning and match in warning for warning in report.warnings)
    assert SPipe.load(bundle, allow_stale=True).instantiate_entry(0)


def test_a_failing_fit_identity_makes_a_frozen_entry_stale(tmp_path, frozen_caa_spipe, monkeypatch):
    from steerability.algorithms.state_control.caa.control import CAA
    from steerability.spipe import SPipe

    bundle = frozen_caa_spipe.save(tmp_path / "fat_dir")

    def failing_fit_identity(self):
        raise RuntimeError("fit identity unavailable")

    monkeypatch.setattr(CAA, "fit_identity", failing_fit_identity)
    with pytest.raises(SpipeStaleError, match="fit identity unavailable"):
        SPipe.load(bundle)
    assert SPipe.load(bundle, allow_stale=True).instantiate_entry(0)


def test_instantiate_entry_requires_allow_code_per_entry():
    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}),
        recipe_entry("input_control/system_prompt", {"text": {"$ref": f"{__name__}:module_level_scorer"}}),
    ])
    assert spipe.instantiate_entry(0)
    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        spipe.instantiate_entry(1)


def test_lenient_instantiation_keeps_data_references(tmp_path, monkeypatch):
    from steerability.algorithms.core.execution.access import ModelAccess
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.algorithms.state_control.caa.control import CAA

    def refuse_load(self):
        raise AssertionError("lenient instantiation must not load datasets")

    monkeypatch.setattr(DataRef, "load", refuse_load)
    caa = CAA(data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]}, layer_id=1)
    recipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[caa]).to_spipe(freeze=False)
    hub_data = {"$data": {"kind": "hf", "repo_id": "org/data", "split": "train"}}
    spipe = recipe_spipe([
        recipe.manifest["controls"][0],
        recipe_entry("structural_control/sft", {"train_dataset": hub_data, "output_dir": str(tmp_path / "sft")}),
    ])

    (caa_control,) = spipe.instantiate_entry(0, lenient=True)
    assert caa_control.steer_access() == ModelAccess.CAPTURE
    (sft_control,) = spipe.instantiate_entry(1, lenient=True, verify="strict")
    assert sft_control.train_dataset == DataRef(kind="hf", repo_id="org/data", split="train")
    with pytest.raises(AssertionError, match="must not load"):
        spipe.instantiate_entry(1)


def test_lenient_instantiation_runs_no_bundle_code_under_allow_code(tmp_path, monkeypatch, store):
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory

    marker = tmp_path / "imported"
    (tmp_path / "bundle_marker_module.py").write_text(
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n"
        f"Path({str(marker)!r}).write_text('imported')\n\n\n"
        "@dataclass\n"
        "class Thing:\n"
        "    label: str = ''\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    bundle_class = {"$dc": "bundle_marker_module.Thing", "fields": {"label": "x"}}
    pool = encode(PoolMemory(items=["a"], metadata={"score": [1.0]}), EncodeContext(store=store))
    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": bundle_class}),
        recipe_entry("input_control/system_prompt", {"text": pool}),
    ], allow_code=True, store=store)

    for index in (0, 1):
        with pytest.raises(SpipeCodeRefError, match="lenient"):
            spipe.instantiate_entry(index, lenient=True)
    assert not marker.exists()
    assert "bundle_marker_module" not in sys.modules
    with pytest.raises(SpipeFormatError, match="text"):
        spipe.instantiate_entry(0)
    assert marker.exists()
    sys.modules.pop("bundle_marker_module", None)


@pytest.mark.parametrize("lenient", [False, True])
def test_constructor_errors_are_reported_per_entry(lenient):
    spipe = recipe_spipe([
        recipe_entry("input_control/system_prompt", {"text": "Be brief."}),
        recipe_entry("input_control/system_prompt", {"text": 3}),
        recipe_entry("output_control/constrained_decoding", {}),
    ])
    assert spipe.instantiate_entry(0, lenient=lenient)
    with pytest.raises(SpipeFormatError, match=r"controls\[1\] \(input_control/system_prompt\).*TypeError"):
        spipe.instantiate_entry(1, lenient=lenient)
    with pytest.raises(SpipeFormatError, match=r"controls\[2\] \(output_control/constrained_decoding\).*ValueError"):
        spipe.instantiate_entry(2, lenient=lenient)


def test_instantiate_entry_matches_pipeline(frozen_caa_spipe):
    from steerability.algorithms.core.execution.access import ModelAccess
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed

    (frozen,) = frozen_caa_spipe.instantiate_entry(0)
    (recipe,) = frozen_caa_spipe.instantiate_entry(0, prefer="recipe")
    assert isinstance(frozen.steering_vector, VerifiedPrecomputed)
    assert frozen.steer_access() == ModelAccess.FACTS
    assert recipe.data is not None
    assert recipe.steer_access() == ModelAccess.CAPTURE
    (piped,) = frozen_caa_spipe.pipeline(verify="off").state_controls
    assert type(piped) is type(frozen)
    assert piped.steering_vector.policy == frozen.steering_vector.policy == "off"
    with pytest.raises(ValueError, match="prefer"):
        frozen_caa_spipe.instantiate_entry(0, prefer="latest")


@pytest.mark.parametrize("allow_code", [False, True])
def test_manifest_dc_naming_a_function_fails_to_decode(allow_code):
    call = {"$dc": "steerability.utils.optional.require", "fields": {"module_name": "os"}}
    spipe = recipe_spipe([recipe_entry("input_control/system_prompt", {"text": call})], allow_code=allow_code)
    for lenient in (False, True):
        with pytest.raises(SpipeFormatError, match="steerability.utils.optional.require"):
            spipe.instantiate_entry(0, lenient=lenient)
    with pytest.raises(SpipeFormatError, match="steerability.utils.optional.require"):
        spipe.pipeline()
