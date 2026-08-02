"""The `Backend` base class: identity, capability advertisement, and session creation."""
from abc import ABC, abstractmethod

from aisteer360.algorithms.core.execution.capabilities import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.session import SteeringSession
from aisteer360.algorithms.core.execution.spec import BackendSpec


class Backend(ABC):
    """A backend owns a loaded model, engine, or connection pool and its lifecycle, advertises
    capabilities, and creates sessions.

    Long-lived consumers hold backends in a cache keyed by `BackendSpec` so configurations
    differing only in per-request steering share one resource.

    Attributes:
        spec: The frozen identity of this backend configuration.
    """

    spec: BackendSpec

    @classmethod
    @abstractmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`, computable without constructing the
        backend. Constructed backends advertise the same sets, verified against the live
        resource where a discovery surface exists."""

    @abstractmethod
    def open_session(self) -> SteeringSession:
        """Open a session for one logical operation."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The advertised capability atoms."""
        return self.capabilities_for_spec(self.spec).atoms

    @property
    def intervention_kinds(self) -> InterventionKinds | None:
        """The advertised intervention kinds, when `Capability.INTERVENTION_SPECS` is present."""
        return self.capabilities_for_spec(self.spec).intervention_kinds

    @property
    def processor_kinds(self) -> ProcessorKinds | None:
        """The advertised processor kinds, when `Capability.PER_STEP_LOGIT_SPECS` is present."""
        return self.capabilities_for_spec(self.spec).processor_kinds

    @property
    def capture_kinds(self) -> CaptureKinds | None:
        """The advertised capture kinds, when `Capability.HIDDEN_CAPTURE` is present."""
        return self.capabilities_for_spec(self.spec).capture_kinds

    def stage_artifacts(self, payloads) -> None:
        """Make each content-addressed artifact available to the execution side.

        Called by the pipeline at the end of `steer()` with the tensor payloads of every
        lowered intervention spec, keyed by content-addressed artifact id. Staging is
        idempotent: an artifact that already exists at the destination is success. The
        in-process backend keeps live tensors and needs no staging; engine-backed backends
        write into the registry the serving engine reads.

        Args:
            payloads: Mapping from artifact id to a name-to-tensor mapping.
        """
        return None
