"""Capability atoms and negotiated kind sets for backend capability advertisement.

A capability atom marks a mechanism that some control requirement can fail on; facts true of
every backend belong to the session protocol contract instead. The kind sets state which
activation edits, per-step logit processors, and capture forms a capable backend executes, and
are advertised alongside the corresponding atoms.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class Capability(Enum):
    """Distinguishing capability atoms advertised by backends.

    Attributes:
        IN_PROCESS_TORCH: The backend exposes the model as a live `torch.nn.Module` in the client
            process, so torch hooks, live logits processors, and direct weight access are
            available. The name refers to this mechanism rather than to process locality.
        INTERVENTION_SPECS: The backend executes activation interventions submitted as
            `InterventionSpec` payloads. The Hugging Face backend does not advertise this atom,
            since torch hooks cover every intervention a spec expresses; requirements state the
            relationship as alternatives.
        PER_STEP_LOGIT_SPECS: The backend hosts per-step logit math submitted as `ProcessorSpec`
            payloads.
        HIDDEN_CAPTURE: The backend serves hidden-state capture through `SteeringSession.capture`.
        BEAM_PROPOSALS: The backend implements beam-search proposal semantics (`num_beams` with
            multiple returned sequences).
        WEIGHT_TRAINING: The backend supports weight updates against the pipeline model.
        MODEL_ADOPTION: The backend can adopt an in-memory model produced by a structural control.
        SERVE_CHECKPOINT: The backend can serve a checkpoint directory produced elsewhere.
        SERVE_LORA: The backend can serve a LoRA adapter produced elsewhere.
    """

    IN_PROCESS_TORCH = "in_process_torch"
    INTERVENTION_SPECS = "intervention_specs"
    PER_STEP_LOGIT_SPECS = "per_step_logit_specs"
    HIDDEN_CAPTURE = "hidden_capture"
    BEAM_PROPOSALS = "beam_proposals"
    WEIGHT_TRAINING = "weight_training"
    MODEL_ADOPTION = "model_adoption"
    SERVE_CHECKPOINT = "serve_checkpoint"
    SERVE_LORA = "serve_lora"


@dataclass(frozen=True, slots=True)
class InterventionKinds:
    """Activation-intervention kinds a backend executes, by permanent wire name.

    Wire names mirror toolkit class names (`AdditiveTransform` serializes as `"additive"`,
    `CacheOnceGate` as `"cache_once"`), so the mapping is definitional rather than maintained.
    Kind names are permanent and their meanings never change; new behavior is a new kind.
    Compatibility is set containment on kind names.

    Attributes:
        transforms: Transform kinds, e.g. `{"additive", "directional_ablation", "rotation",
            "head_additive"}`.
        modifiers: Wrapper-transform kinds, e.g. `{"norm_preserving", "alignment_adaptive"}`.
        scopes: Token-scope kinds, e.g. `{"all", "after_prompt", "last_k", "from_position"}`.
        gates: Gate kinds; an always-open gate is the `"null"` kind.
        constraints: Per-kind execution constraints, e.g.
            `{"head_additive": "tensor_parallel_size==1"}`. Informational; containment checks
            ignore this field.
    """

    transforms: frozenset[str] = frozenset()
    modifiers: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    gates: frozenset[str] = frozenset()
    constraints: Mapping[str, str] = field(default_factory=dict)

    def contains(self, required: "InterventionKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return (
            required.transforms <= self.transforms
            and required.modifiers <= self.modifiers
            and required.scopes <= self.scopes
            and required.gates <= self.gates
        )


@dataclass(frozen=True, slots=True)
class ProcessorKinds:
    """Engine-hosted logit-processor kinds a backend executes, by permanent wire name.

    Attributes:
        processors: Processor kinds, e.g. `{"constraint"}`.
    """

    processors: frozenset[str] = frozenset()

    def contains(self, required: "ProcessorKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return required.processors <= self.processors


@dataclass(frozen=True, slots=True)
class CaptureKinds:
    """Hidden-state capture forms a backend serves, by permanent wire name.

    Attributes:
        kinds: Capture kinds, e.g. `{"residual"}`.
        locations: Capture locations, e.g. `{"layer_output", "layer_input"}`.
        modes: Capture modes, e.g. `{"all_tokens", "last_token"}`.
    """

    kinds: frozenset[str] = frozenset()
    locations: frozenset[str] = frozenset()
    modes: frozenset[str] = frozenset()

    def contains(self, required: "CaptureKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return (
            required.kinds <= self.kinds
            and required.locations <= self.locations
            and required.modes <= self.modes
        )


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """A backend's full capability advertisement: atoms plus negotiated kind sets.

    Attributes:
        atoms: The advertised `Capability` atoms.
        intervention_kinds: Advertised intervention kinds, present when
            `Capability.INTERVENTION_SPECS` is among the atoms.
        processor_kinds: Advertised processor kinds, present when
            `Capability.PER_STEP_LOGIT_SPECS` is among the atoms.
        capture_kinds: Advertised capture kinds, present when `Capability.HIDDEN_CAPTURE` is
            among the atoms.
    """

    atoms: frozenset[Capability] = frozenset()
    intervention_kinds: InterventionKinds | None = None
    processor_kinds: ProcessorKinds | None = None
    capture_kinds: CaptureKinds | None = None
