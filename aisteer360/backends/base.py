"""Backend and session abstract base classes plus their shared value types.

A `Backend` owns model/endpoint identity and capability negotiation; a `SteeringSession` is the
scope within which steering is in force for one or more generation calls. Sessions are the unit of
concurrency: an in-process `HuggingFaceSession` is exclusive (its hooks mutate one shared module
graph, so only one generation may be in flight at a time), while API-backed sessions are stateless
request builders that are safe to drive from many tasks at once (`concurrency_safe = True`).

The dependency rule (doc 00): backends may import `core/` and the compile-relevant types under
`algorithms/state_control/_common/`, but `algorithms/*` never imports `backends/*`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from aisteer360.core.output import Output
from aisteer360.core.requirements import Capability

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from aisteer360.backends.generation_params import GenerationParams
    from aisteer360.backends.specs import BackendSpec
    from aisteer360.core.prompt import PreparedPrompt


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do, as a capability set plus concurrency and artifact metadata.

    Attributes:
        capabilities: The granted `Capability` set (static ∪ handshake-derived).
        max_concurrency: Maximum in-flight generations the backend supports (1 for exclusive
            in-process backends).
        accepts_artifacts: The structural-control artifact kinds the backend can deploy
            (subset of `{"model", "checkpoint", "lora"}`).
        notes: Per-capability semantic notes surfaced as `degraded` verdicts during validation
            (e.g. a chunked-prefill gating caveat).
    """

    capabilities: Capability
    max_concurrency: int = 1
    accepts_artifacts: frozenset[str] = frozenset()
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """A steer-time product handed from a structural control to a backend.

    Attributes:
        kind: One of `"model"`, `"checkpoint"`, `"lora"`.
        ref: The artifact reference (a path, repo id, or adapter name).
        metadata: Optional extra descriptors (base model id, dtype, ...).
    """

    kind: str
    ref: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateControlEntry:
    """One state control's contribution to a session, in pipeline list order.

    Exactly one of `plan` / `hooks` is set: declarative controls contribute an `InterventionPlan`
    (compiled by the backend), hook-level controls contribute a ready hook dict (HF sessions only).

    Attributes:
        control_name: The control's display name (for diagnostics).
        plan: The declarative intervention plan, or `None` for hook-level controls.
        hooks: A ready `{"pre": [...], "forward": [...], "backward": [...]}` hook dict, or `None`
            for declarative controls.
    """

    control_name: str
    plan: Any | None = None          # InterventionPlan | None (typed Any until doc 03 lands)
    hooks: dict | None = None


class SteeringSession(ABC):
    """A scope within which state steering is in force, as a context manager.

    Steering is active for the lifetime of the session: every internal forward an output control
    performs runs inside `with session:`, so hooks live exactly for the duration of decoding.

    Attributes:
        model: The underlying `PreTrainedModel` on in-process backends (behind
            `Capability.RAW_MODEL`); `None` on API backends.
        concurrency_safe: `True` for stateless API sessions that may be driven concurrently;
            `False` for exclusive in-process sessions.
    """

    model: "PreTrainedModel | None" = None
    concurrency_safe: bool = False

    @abstractmethod
    def generate(self, prepared: "PreparedPrompt", params: "GenerationParams") -> Output:
        """Generate under the session's active steering.

        Args:
            prepared: The adapted prompt to generate from.
            params: Normalized generation parameters.

        Returns:
            The generation `Output`.
        """

    def score(self, prepared: "PreparedPrompt", ref_output_ids: torch.Tensor) -> torch.Tensor | None:
        """Return per-token log-probabilities of `ref_output_ids` under the session's steering.

        Only backends granting `Capability.SCORING` implement this; others return `None`.

        Args:
            prepared: The adapted prompt to score against.
            ref_output_ids: Reference continuation token ids, `[batch, ref_len]` (single-row refs
                broadcast across the prompt batch).

        Returns:
            A `[batch, ref_len]` tensor of per-token log-probabilities, or `None` if unsupported.
        """
        return None

    def __enter__(self) -> "SteeringSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class Backend(ABC):
    """A target that a steering pipeline runs inference (or fitting) against.

    Attributes:
        spec: The declarative `BackendSpec` that produced this backend (always populated; backends
            constructed directly synthesize one) so runs can be checkpointed and cached by
            `spec.stable_hash()`.
        tokenizer: A client-side HF tokenizer, or `None` when the backend does not tokenize.
        model_identity: The repo id / served-model name, or `None` if unknown.
    """

    spec: "BackendSpec"
    tokenizer: "PreTrainedTokenizerBase | None" = None
    model_identity: str | None = None

    @classmethod
    def from_spec(cls, spec: "BackendSpec") -> "Backend":
        """Build a backend instance from a declarative `BackendSpec`.

        Each backend maps the spec's `model` / `base_url` / `kwargs` onto its own constructor. The
        default forwards `spec.kwargs` and sets `.spec`; subclasses override to translate field
        names (e.g. `model` → `model_name_or_path`).

        Args:
            spec: The declarative backend description.

        Returns:
            A constructed backend whose `.spec` is `spec`.
        """
        backend = cls(**dict(spec.kwargs))
        backend.spec = spec
        return backend

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """The backend's capabilities (may consult a cached handshake)."""

    @property
    def executes_hooks_in_process(self) -> bool:
        """Whether this backend realizes state controls as in-process forward hooks.

        In-process backends (HuggingFace) want ready hook dicts in each `StateControlEntry`; server
        backends want the declarative `InterventionPlan` to compile to their wire schema. The
        pipeline consults this to decide which representation to build. Default `False`.
        """
        return False

    @abstractmethod
    def open_session(
        self,
        entries: list[StateControlEntry],
        prompt_ctx: "PreparedPrompt",
        runtime_kwargs: dict,
    ) -> SteeringSession:
        """Open a steering session for the given state-control entries.

        Args:
            entries: State-control contributions in pipeline list order.
            prompt_ctx: The adapted prompt the session will generate from.
            runtime_kwargs: Per-call control parameters.

        Returns:
            A `SteeringSession` ready to generate/score under the requested steering.
        """

    def accept_artifact(self, artifact: Artifact) -> None:
        """Deploy a structural-control artifact to this backend.

        Args:
            artifact: The artifact to deploy.

        Raises:
            ArtifactNotDeployable: If this backend cannot deploy the given artifact kind.
        """
        from aisteer360.backends.errors import ArtifactNotDeployable

        raise ArtifactNotDeployable(
            f"The {type(self).__name__} cannot deploy a {artifact.kind!r} artifact.",
            model=self.model_identity,
        )

    def close(self) -> None:
        """Release any resources the backend holds (models, clients). Default is a no-op."""
        return None
