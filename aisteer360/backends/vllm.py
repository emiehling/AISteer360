"""Capability advertisement for the vLLM backend kinds.

This module imports cleanly without vLLM installed. The capability tables are static data used
by `check()`. Constructing a backend instance requires the `vllm` optional dependency and
raises `NotImplementedError`, since vLLM execution is not implemented.
"""
from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.capabilities import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.session import SteeringSession
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.utils.optional import require

_PLUGIN_INTERVENTION_KINDS = InterventionKinds(
    transforms=frozenset({"additive", "directional_ablation", "rotation", "head_additive"}),
    modifiers=frozenset({"norm_preserving", "alignment_adaptive"}),
    scopes=frozenset({"all", "after_prompt", "last_k", "from_position"}),
    gates=frozenset({"null", "cache_once", "probe_sum", "multi_key_threshold"}),
    constraints={"head_additive": "tensor_parallel_size==1"},
)

_PLUGIN_PROCESSOR_KINDS = ProcessorKinds(processors=frozenset({"constraint"}))

_PLUGIN_CAPTURE_KINDS = CaptureKinds(
    kinds=frozenset({"residual"}),
    locations=frozenset({"layer_output", "layer_input"}),
    modes=frozenset({"all_tokens", "last_token"}),
)

VLLM_BASELINE_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({Capability.SERVE_CHECKPOINT, Capability.SERVE_LORA}),
)


def _vllm_capabilities(spec: BackendSpec, *, offline: bool) -> BackendCapabilities:
    """Capabilities implied by a vLLM spec: the plugin-free baseline, extended when the spec
    declares the vLLM-Hook plugin active. Hidden capture is advertised on the offline engine
    only, since serve-mode capture needs a bulk-tensor return path."""
    if not spec.get_option("hook_plugin"):
        return VLLM_BASELINE_CAPABILITIES
    atoms = VLLM_BASELINE_CAPABILITIES.atoms | {
        Capability.INTERVENTION_SPECS,
        Capability.PER_STEP_LOGIT_SPECS,
    }
    capture_kinds = None
    if offline:
        atoms = atoms | {Capability.HIDDEN_CAPTURE}
        capture_kinds = _PLUGIN_CAPTURE_KINDS
    return BackendCapabilities(
        atoms=frozenset(atoms),
        intervention_kinds=_PLUGIN_INTERVENTION_KINDS,
        processor_kinds=_PLUGIN_PROCESSOR_KINDS,
        capture_kinds=capture_kinds,
    )


class VLLMBackend(Backend):
    """The offline vLLM engine backend.

    Capability advertisement is available through `capabilities_for_spec` without constructing
    the backend. Construction requires the `vllm` optional dependency and raises
    `NotImplementedError`, since engine execution is not implemented.
    """

    def __init__(self, spec: BackendSpec) -> None:
        if spec.kind != "vllm":
            raise ValueError(f"VLLMBackend requires a 'vllm' spec; got kind {spec.kind!r}.")
        self.spec = spec
        require("vllm")
        raise NotImplementedError(
            "vLLM offline-engine execution is not yet implemented; capability reporting for "
            "vLLM specs is available through SteeringPipeline.check()."
        )

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=True)

    def open_session(self) -> SteeringSession:
        """Unreachable, since `__init__` raises `NotImplementedError`."""
        raise NotImplementedError


class VLLMServeBackend(Backend):
    """The vLLM OpenAI-compatible server backend.

    Targets a vLLM server rather than an arbitrary OpenAI-compatible endpoint. Capability
    advertisement is available through `capabilities_for_spec` without constructing the backend.
    Construction requires the `vllm` optional dependency and raises `NotImplementedError`,
    since request execution is not implemented.
    """

    def __init__(self, spec: BackendSpec) -> None:
        if spec.kind != "vllm-serve":
            raise ValueError(f"VLLMServeBackend requires a 'vllm-serve' spec; got kind {spec.kind!r}.")
        self.spec = spec
        require("vllm")
        raise NotImplementedError(
            "vLLM serve execution is not yet implemented; capability reporting for vLLM specs "
            "is available through SteeringPipeline.check()."
        )

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=False)

    def open_session(self) -> SteeringSession:
        """Unreachable, since `__init__` raises `NotImplementedError`."""
        raise NotImplementedError
