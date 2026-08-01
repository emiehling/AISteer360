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
