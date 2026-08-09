"""Execution seam types for multi-backend steering.

A `Backend` owns identity, capability advertisement, and session creation; a `SteeringSession`
is the scope within which steering is in force and the unit of concurrency. The pipeline
interacts with backends through these two interfaces. Backend implementations live in
`aisteer360.backends`; this package holds every seam type and imports nothing from
`aisteer360.backends` at module level.
"""
from aisteer360.algorithms.core.execution.access import (
    ModelAccess,
    PlannedFit,
    PlannedStep,
    SteerPlan,
)
from aisteer360.algorithms.core.execution.payloads import (
    Artifact,
    ArtifactProvenance,
    CheckpointArtifact,
    LoRAArtifact,
    ModelArtifact,
)
from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.payloads import ConstraintSource, as_constraint_source
from aisteer360.algorithms.core.execution.contracts import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    ConstraintKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.fanout import (
    PartialBatchError,
    TransportError,
    derive_item_seed,
    run_bounded,
    with_transport_retries,
)
from aisteer360.algorithms.core.execution.payloads import (
    InterventionSpec,
    ProcessorSpec,
)
from aisteer360.algorithms.core.execution.payloads import (
    ConstraintEntry,
    CaptureResult,
    GenerationItem,
    HookEntry,
    InterventionEntry,
    ItemResult,
    OutputControlEntry,
    ProcessorSpecEntry,
    ScoringItem,
    StackEntry,
    StateControlEntry,
)
from aisteer360.algorithms.core.execution.payloads import ModelFacts
from aisteer360.algorithms.core.execution.params import (
    GenerationParams,
    merge_lowered_params,
)
from aisteer360.algorithms.core.execution.payloads import PreparedPrompt
from aisteer360.algorithms.core.execution.backend import (
    capabilities_for_spec,
    resolve_backend_class,
)
from aisteer360.algorithms.core.execution.contracts import (
    Alternative,
    Requirements,
    SpecConstraint,
    any_of,
    needs,
)
from aisteer360.algorithms.core.execution.backend import SteeringSession
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.execution.contracts import (
    SupportFailure,
    SupportReport,
    UnsupportedOperationError,
    UnsupportedPipelineError,
    evaluate_support,
)

__all__ = [
    "Artifact",
    "ArtifactProvenance",
    "Backend",
    "BackendCapabilities",
    "BackendSpec",
    "Capability",
    "CaptureKinds",
    "ConstraintEntry",
    "ConstraintKinds",
    "ConstraintSource",
    "as_constraint_source",
    "CaptureResult",
    "CheckpointArtifact",
    "GenerationItem",
    "GenerationParams",
    "HookEntry",
    "InterventionEntry",
    "InterventionKinds",
    "InterventionSpec",
    "ItemResult",
    "LoRAArtifact",
    "ModelAccess",
    "ModelArtifact",
    "ModelFacts",
    "OutputControlEntry",
    "PlannedFit",
    "PlannedStep",
    "PreparedPrompt",
    "ProcessorKinds",
    "ProcessorSpec",
    "ProcessorSpecEntry",
    "Requirements",
    "Alternative",
    "ScoringItem",
    "SpecConstraint",
    "StackEntry",
    "StateControlEntry",
    "SteerPlan",
    "SteeringSession",
    "SupportFailure",
    "SupportReport",
    "UnsupportedOperationError",
    "UnsupportedPipelineError",
    "PartialBatchError",
    "TransportError",
    "any_of",
    "capabilities_for_spec",
    "derive_item_seed",
    "evaluate_support",
    "merge_lowered_params",
    "needs",
    "resolve_backend_class",
    "run_bounded",
    "with_transport_retries",
]
