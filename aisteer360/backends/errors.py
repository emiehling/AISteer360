"""Backend error taxonomy.

A single `BackendError` hierarchy so callers can distinguish connection failures, unmet capability
probes, unsupported generation parameters, deployment mismatches, and partial batch failures. Every
error carries the `base_url` / `model` context of the backend that raised it where available.
"""
from __future__ import annotations


class BackendError(Exception):
    """Base class for all backend errors.

    Args:
        message: Human-readable description.
        base_url: The backend's endpoint, when applicable.
        model: The served model / repo id, when applicable.
    """

    def __init__(self, message: str, *, base_url: str | None = None, model: str | None = None) -> None:
        context = []
        if model is not None:
            context.append(f"model={model!r}")
        if base_url is not None:
            context.append(f"base_url={base_url!r}")
        if context:
            message = f"{message} ({', '.join(context)})"
        super().__init__(message)
        self.base_url = base_url
        self.model = model


class BackendConnectionError(BackendError):
    """The backend endpoint could not be reached, or the request timed out."""


class BackendCapabilityError(BackendError):
    """A required capability probe failed, or a requested capability is unavailable."""


class UnsupportedGenerationParam(BackendError):
    """A generation parameter is not supported by the target backend.

    Args:
        param: The offending parameter name.
        backend_kind: The backend kind that rejected it.
    """

    def __init__(self, param: str, backend_kind: str, *, base_url: str | None = None, model: str | None = None) -> None:
        super().__init__(
            f"Generation parameter {param!r} is not supported by the {backend_kind} backend.",
            base_url=base_url,
            model=model,
        )
        self.param = param
        self.backend_kind = backend_kind


class ArtifactNotDeployable(BackendError):
    """A structural-control artifact cannot be deployed to this backend.

    Raised, for example, when a full checkpoint is handed to an API backend that can only accept a
    LoRA adapter; the message carries the exact serve command to run instead.
    """


class BatchPartialFailure(BackendError):
    """One or more rows of a batched request failed while others succeeded.

    Args:
        indices: The batch indices that failed.
        causes: The per-index exceptions, aligned with `indices`.
    """

    def __init__(
        self,
        indices: list[int],
        causes: list[BaseException],
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            f"{len(indices)} of the batched requests failed at indices {indices}.",
            base_url=base_url,
            model=model,
        )
        self.indices = indices
        self.causes = causes
