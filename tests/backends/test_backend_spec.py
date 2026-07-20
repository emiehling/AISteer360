"""Tests for `BackendSpec` build/hash (doc 02 §3, §7)."""
from immutabledict import immutabledict

from aisteer360.backends.specs import BackendSpec


def test_stable_hash_invariant_under_kwarg_ordering():
    spec_a = BackendSpec(kind="openai", base_url="http://x", model="m", kwargs=immutabledict({"a": 1, "b": 2}))
    spec_b = BackendSpec(kind="openai", base_url="http://x", model="m", kwargs=immutabledict({"b": 2, "a": 1}))
    assert spec_a.stable_hash() == spec_b.stable_hash()


def test_stable_hash_distinguishes_fields():
    base = BackendSpec(kind="huggingface", model="m1")
    assert base.stable_hash() != BackendSpec(kind="huggingface", model="m2").stable_hash()
    assert base.stable_hash() != BackendSpec(kind="openai", model="m1", base_url="u").stable_hash()


def test_build_resolves_huggingface(monkeypatch):
    # build() must resolve the class through the registry without loading a model
    built = {}

    class _Fake:
        @classmethod
        def from_spec(cls, spec):
            built["spec"] = spec
            return "backend-instance"

    import aisteer360.core.registry as registry

    monkeypatch.setattr(registry, "resolve_backend", lambda kind: _Fake)
    spec = BackendSpec(kind="huggingface", model="tiny")
    assert spec.build() == "backend-instance"
    assert built["spec"] is spec
