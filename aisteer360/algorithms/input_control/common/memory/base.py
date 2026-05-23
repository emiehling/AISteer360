"""Memory: typed containers for input-control artifacts."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Structural type for input-control memory.

    A `Memory` is whatever artifact an input control reads in `adapt()` and optionally writes to in `steer()` and
    `observe()`. The Protocol captures only the minimal serialization contract; concrete subclasses define their own
    fields and load semantics.

    Attributes:
        model_type: Short tag identifying the concrete memory subclass. Mirrors the convention used by `SteeringVector`
            in `state_control/common/`. Useful for polymorphic dispatch in future tooling (Benchmark resume, debugging).
    """

    model_type: str

    def save(self, path: str) -> None:
        """Persist the memory to `path`."""
        ...
