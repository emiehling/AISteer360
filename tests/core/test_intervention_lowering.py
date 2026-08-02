"""Tests for the intervention-spec lowering surface: canonicalization and salt derivation
byte-aligned with `vllm_hook_plugins.core.canonical`, and artifact-id collection on the seam
type."""
import pytest
import torch
from vllm_hook_plugins.core.canonical import canonical_bytes, request_salt, spec_hash

from aisteer360.algorithms.core.execution import InterventionSpec

_VECTOR_ID = "sha256:" + "ab" * 32
_PROBE_ID = "sha256:" + "cd" * 32
_MODIFIER_ID = "sha256:" + "ef" * 32


def _spec() -> InterventionSpec:
    return InterventionSpec(ops=(
        {
            "layers": (13,),
            "transform": {
                "kind": "additive",
                "strength": 2.0,
                "modifiers": ({"kind": "alignment_adaptive", "artifact": _MODIFIER_ID},),
                "artifact": _VECTOR_ID,
            },
            "scope": {"kind": "after_prompt"},
            "gate": {
                "kind": "cache_once",
                "inner": {
                    "kind": "probe_sum",
                    "condition_layers": (6,),
                    "pooling": "mean",
                    "artifact": _PROBE_ID,
                },
            },
        },
    ))


class TestCanonicalAlignment:

    def test_canonical_byte_equals_plugin_canonical_bytes(self):
        spec = _spec()
        assert spec.canonical().encode("utf-8") == canonical_bytes(spec.to_wire())

    def test_canonical_uses_compact_separators_and_sorted_keys(self):
        spec = InterventionSpec(ops=({"layers": (1,), "transform": {"kind": "additive"}, "scope": {"kind": "all"}, "gate": None},))
        canonical = spec.canonical()
        assert ": " not in canonical and ", " not in canonical
        assert canonical.index('"gate"') < canonical.index('"layers"') < canonical.index('"scope"')

    def test_to_wire_converts_tuples_to_lists(self):
        wire = _spec().to_wire()
        assert isinstance(wire["ops"], list)
        assert wire["ops"][0]["layers"] == [13]
        assert isinstance(wire["ops"][0]["transform"]["modifiers"], list)

    def test_salt_matches_reference_derivation(self):
        spec = _spec()
        assert spec.salt() == request_salt(spec.to_wire(), list(spec.artifact_ids()))
        assert spec.salt() == request_salt(spec.to_wire(), [_PROBE_ID, _VECTOR_ID, _MODIFIER_ID])

    def test_salt_differs_from_spec_hash_and_covers_artifacts(self):
        spec = _spec()
        assert spec.salt() != spec_hash(spec.to_wire())
        bare = InterventionSpec(ops=spec.ops)
        assert bare.salt() == spec.salt()

    def test_artifact_ids_collects_transform_modifier_and_nested_gate(self):
        assert _spec().artifact_ids() == tuple(sorted({_VECTOR_ID, _PROBE_ID, _MODIFIER_ID}))

    def test_inline_tensor_raises_type_error(self):
        spec = InterventionSpec(ops=(
            {"layers": (0,), "transform": {"kind": "additive", "vector": torch.ones(4)}, "scope": {"kind": "all"}, "gate": None},
        ))
        with pytest.raises(TypeError):
            spec.canonical()


class TestRequiredKinds:

    def test_collects_kinds_across_ops_and_nested_gates(self):
        required = _spec().required_kinds()
        assert required.transforms == frozenset({"additive"})
        assert required.modifiers == frozenset({"alignment_adaptive"})
        assert required.scopes == frozenset({"after_prompt"})
        assert required.gates == frozenset({"cache_once", "probe_sum"})


class TestEntrySelection:

    @staticmethod
    def _steered_pipeline(control):
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        pipeline.model = tiny_llama(num_layers=4, hidden=16, heads=2)
        pipeline.tokenizer = wordlevel_tokenizer()
        pipeline.steer()
        return pipeline

    @staticmethod
    def _caa(**kwargs):
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA

        vector = SteeringVector(
            model_type="llama", directions={1: torch.ones(1, 16)},
        )
        return CAA(steering_vector=vector, layer_id=1, **kwargs)

    @staticmethod
    def _capabilities(**kind_overrides):
        from aisteer360.algorithms.core.execution import (
            BackendCapabilities,
            Capability,
            InterventionKinds,
        )

        kinds = {
            "transforms": frozenset({"additive", "directional_ablation", "rotation", "head_additive"}),
            "modifiers": frozenset({"norm_preserving", "alignment_adaptive"}),
            "scopes": frozenset({"all", "after_prompt", "last_k", "from_position"}),
            "gates": frozenset({"null", "cache_once", "probe_sum", "multi_key_threshold"}),
        }
        kinds.update(kind_overrides)
        return BackendCapabilities(
            atoms=frozenset({Capability.INTERVENTION_SPECS}),
            intervention_kinds=InterventionKinds(**kinds),
        )

    def test_intervention_entries_built_for_exportable_control(self):
        from aisteer360.algorithms.core.execution import InterventionEntry

        pipeline = self._steered_pipeline(self._caa())
        control = pipeline.state_controls[0]
        entry = pipeline._lower_control(
            control, self._capabilities().intervention_kinds, {}, {},
        )
        assert isinstance(entry, InterventionEntry)
        assert entry.spec.ops[0]["transform"]["kind"] == "additive"

    def test_stale_kind_server_yields_verdict_naming_kind(self):
        from aisteer360.algorithms.core.execution import UnsupportedOperationError

        pipeline = self._steered_pipeline(self._caa())
        control = pipeline.state_controls[0]
        narrowed = self._capabilities(transforms=frozenset({"rotation"}))
        with pytest.raises(UnsupportedOperationError, match="additive"):
            pipeline._lower_control(control, narrowed.intervention_kinds, {}, {})

    def test_hook_only_control_yields_verdict(self):
        from aisteer360.algorithms.core.execution import UnsupportedOperationError
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA

        positional = CAA(
            steering_vector=SteeringVector(model_type="llama", directions={1: torch.ones(3, 16)}),
            layer_id=1,
        )
        pipeline = self._steered_pipeline(positional)
        control = pipeline.state_controls[0]
        with pytest.raises(UnsupportedOperationError, match="no intervention-spec form"):
            pipeline._lower_control(control, self._capabilities().intervention_kinds, {}, {})


class TestVerdictStrings:

    def test_positional_caa_names_the_gap(self):
        from aisteer360.algorithms.core.execution import BackendSpec
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA

        control = CAA(
            steering_vector=SteeringVector(model_type="llama", directions={1: torch.ones(3, 16)}),
            layer_id=1,
        )
        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        report = pipeline.check(inference_backend=BackendSpec(
            kind="vllm", model="m", options={"hook_plugin": True},
        ))
        (failure,) = report.failures_for("generate")
        assert failure.message == (
            "CAA is unsupported at generate on backend kind 'vllm': missing IN_PROCESS_TORCH; "
            "positional directions have no intervention-spec form; run on the huggingface backend."
        )

    def test_cast_names_the_missing_gate_kind(self):
        from aisteer360.algorithms.core.execution import BackendSpec
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from aisteer360.algorithms.state_control.cast.control import CAST

        control = CAST(behavior_vector=None, behavior_data={"positives": ["a"], "negatives": ["b"]})
        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        report = pipeline.check(inference_backend=BackendSpec(
            kind="vllm", model="m", options={"hook_plugin": True},
        ))
        messages = [failure.message for failure in report.failures_for("generate")]
        assert any("projected-cosine condition has no intervention-spec gate kind" in m for m in messages)

    def test_exportable_caa_is_supported_on_plugin_backend(self):
        from aisteer360.algorithms.core.execution import BackendSpec
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA

        control = CAA(
            steering_vector=SteeringVector(model_type="llama", directions={1: torch.ones(1, 16)}),
            layer_id=1,
        )
        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        report = pipeline.check(inference_backend=BackendSpec(
            kind="vllm", model="m", options={"hook_plugin": True},
        ))
        assert report.supported("generate")
        bare = pipeline.check(inference_backend=BackendSpec(kind="vllm", model="m"))
        assert not bare.supported("generate")


class TestDiscoveryIntersection:

    def test_negotiated_kinds_narrow_static_tables(self):
        from aisteer360.algorithms.core.execution import BackendSpec, capabilities_for_spec
        from aisteer360.backends import vllm as vllm_backend

        spec = BackendSpec(kind="vllm", model="intersect-test", options={"hook_plugin": True})
        static = capabilities_for_spec(spec)
        assert "rotation" in static.intervention_kinds.transforms

        payload = {
            "intervention_kinds": {
                "transforms": ["additive", "directional_ablation", "head_additive"],
                "modifiers": ["norm_preserving", "alignment_adaptive"],
                "scopes": ["all", "after_prompt", "last_k", "from_position"],
                "gates": ["null", "cache_once", "probe_sum", "multi_key_threshold"],
                "constraints": {"head_additive": "tensor_parallel_size==1"},
            },
            "processor_kinds": {"processors": []},
            "capture_kinds": {"kinds": ["residual"], "locations": ["layer_output"], "modes": ["all_tokens"]},
        }
        vllm_backend._DISCOVERY_CACHE[spec.spec_hash] = payload
        try:
            negotiated = capabilities_for_spec(spec)
            assert "rotation" not in negotiated.intervention_kinds.transforms
            assert "additive" in negotiated.intervention_kinds.transforms
            assert negotiated.processor_kinds is None
            assert negotiated.capture_kinds.locations == frozenset({"layer_output"})
            assert negotiated.atoms == static.atoms
        finally:
            vllm_backend._DISCOVERY_CACHE.pop(spec.spec_hash, None)
