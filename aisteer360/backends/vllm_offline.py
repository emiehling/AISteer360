"""Placeholder for an offline, in-process vLLM backend (deferred, decision D4).

The serve-only path (`OpenAIBackend` / `VLLMHookBackend` against `vllm serve`) is the supported way
to run steering over vLLM. An offline in-process vLLM engine is deferred; this module marks the seam
so the design decision is discoverable from the code. See `design_docs/02-backends-foundation.md`
(§8) and `design_docs/00-overview.md` (D4).
"""
from __future__ import annotations

from aisteer360.backends.base import Backend


class VLLMOfflineBackend(Backend):
    """Not implemented: an offline in-process vLLM backend (deferred, D4)."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        raise NotImplementedError(
            "An offline in-process vLLM backend is deferred (decision D4). Use OpenAIBackend or "
            "VLLMHookBackend against a running `vllm serve`. See design_docs/02-backends-foundation.md §8."
        )

    @property
    def capabilities(self):  # pragma: no cover - unreachable; __init__ raises
        raise NotImplementedError

    def open_session(self, entries, prompt_ctx, runtime_kwargs):  # pragma: no cover - unreachable
        raise NotImplementedError
