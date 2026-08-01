"""Explicit registry resolving `BackendSpec` kinds to backend classes.

The registry is a fixed mapping over the core-owned backend kinds. Backend modules are
imported on first resolution, so `core` carries no module-level dependency on
`aisteer360.backends`.
"""
from importlib import import_module
from typing import TYPE_CHECKING

from aisteer360.algorithms.core.execution.capabilities import BackendCapabilities
from aisteer360.algorithms.core.execution.spec import BackendSpec

if TYPE_CHECKING:
    from aisteer360.algorithms.core.execution.backend import Backend

_BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "huggingface": ("aisteer360.backends.huggingface", "HFBackend"),
    "vllm": ("aisteer360.backends.vllm", "VLLMBackend"),
    "vllm-serve": ("aisteer360.backends.vllm", "VLLMServeBackend"),
}


def resolve_backend_class(spec: BackendSpec) -> "type[Backend]":
    """The backend class registered for `spec.kind`.

    Args:
        spec: The backend spec to resolve.

    Returns:
        The backend class. Importing the class does not require the backend's optional
        dependencies; constructing an instance may.

    Raises:
        ValueError: If no backend class is registered for the spec's kind.
    """
    entry = _BACKEND_CLASSES.get(spec.kind)
    if entry is None:
        raise ValueError(f"No backend class is registered for kind {spec.kind!r}.")
    module_name, attribute = entry
    return getattr(import_module(module_name), attribute)


def capabilities_for_spec(spec: BackendSpec) -> BackendCapabilities:
    """The capability advertisement implied by `spec`, without constructing a backend."""
    return resolve_backend_class(spec).capabilities_for_spec(spec)
