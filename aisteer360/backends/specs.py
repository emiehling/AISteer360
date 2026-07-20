"""`BackendSpec`: a declarative, hashable description of a backend.

Mirrors `ControlSpec`. A spec resolves its backend class through the registry (avoiding import
cycles) and yields a stable content hash so runs can be checkpointed and cached by backend identity
(doc 09). Every `Backend` carries a `spec` — backends constructed directly synthesize one — so the
cache key is always available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

from immutabledict import immutabledict

if TYPE_CHECKING:
    from aisteer360.backends.base import Backend

BackendKind = Literal["huggingface", "openai", "vllm_hook"]


@dataclass(frozen=True)
class BackendSpec:
    """A declarative description of a backend.

    Attributes:
        kind: The backend kind (`"huggingface"`, `"openai"`, `"vllm_hook"`).
        model: HF repo/path or served-model name.
        base_url: The endpoint for API backends.
        kwargs: Extra constructor kwargs (immutable so the spec is hashable).
    """

    kind: BackendKind
    model: str | None = None
    base_url: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=immutabledict)

    def build(self) -> "Backend":
        """Resolve the backend class through the registry and construct it from this spec.

        Returns:
            A constructed backend whose `.spec` is this spec.
        """
        from aisteer360.core.registry import resolve_backend

        backend_cls = resolve_backend(self.kind)
        return backend_cls.from_spec(self)

    def stable_hash(self) -> str:
        """Return a short content hash, invariant under `kwargs` dict ordering.

        Returns:
            An 8-character hash of `(kind, model, base_url, kwargs)`.
        """
        from aisteer360.evaluation.utils.data_utils import _hash_params

        payload = {
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "kwargs": dict(self.kwargs),
        }
        return _hash_params(payload)
