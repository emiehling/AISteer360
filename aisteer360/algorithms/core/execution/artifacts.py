"""Typed artifacts that cross the steering/inference role boundary, with provenance."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Identity of the side that produced an artifact.

    Attributes:
        backend_spec_hash: `BackendSpec.spec_hash` of the producing backend.
        model_fingerprint: Fingerprint of the producing model.
        tokenizer_fingerprint: Fingerprint of the producing tokenizer and chat template.
    """

    backend_spec_hash: str | None = None
    model_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ModelArtifact:
    """An in-memory model handed across the role boundary; consuming it requires
    `Capability.MODEL_ADOPTION`.

    Attributes:
        model: The loaded model.
        provenance: Identity of the producing side.
    """

    model: Any
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """A checkpoint directory handed across the role boundary; consuming it requires
    `Capability.SERVE_CHECKPOINT`.

    Attributes:
        path: Checkpoint directory path.
        provenance: Identity of the producing side.
    """

    path: str
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


@dataclass(frozen=True, slots=True)
class LoRAArtifact:
    """A LoRA adapter handed across the role boundary; consuming it requires
    `Capability.SERVE_LORA`.

    Attributes:
        path: Adapter directory path.
        base_model: Model reference the adapter applies to.
        provenance: Identity of the producing side.
    """

    path: str
    base_model: str
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)


Artifact = ModelArtifact | CheckpointArtifact | LoRAArtifact
