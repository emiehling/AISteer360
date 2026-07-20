"""`VLLMHookBackend`: server-side steering over an OpenAI-compatible `vllm serve` + hook plugin.

The handshake, capability derivation, and the `InterventionPlan → vllm_xargs` compiler are specified
in `design_docs/06-vllm-hook-compiler.md` and land in the integration phase (P2), co-designed with
the engine (`design_docs/07-vllm-hook-engine.md`, a separate repo). This module extends
`OpenAIBackend` and, like it, never imports `vllm`.
"""
from __future__ import annotations

from aisteer360.backends.openai_compat.openai import OpenAIBackend
from aisteer360.backends.specs import BackendSpec


class VLLMHookBackend(OpenAIBackend):
    """Server-side activation steering over vLLM-Hook (compiler pending; see doc 06)."""

    @classmethod
    def from_spec(cls, spec: BackendSpec) -> "VLLMHookBackend":
        backend = cls(base_url=spec.base_url, model=spec.model, **dict(spec.kwargs))
        backend.spec = spec
        return backend

    def open_session(self, entries, prompt_ctx, runtime_kwargs):
        active = [entry for entry in entries if entry.plan is not None or entry.hooks is not None]
        if active:
            raise NotImplementedError(
                "VLLMHookBackend server-side steering (the plan→vllm_xargs compiler) is implemented in "
                "the integration phase; see design_docs/06-vllm-hook-compiler.md. Prompting/scoring "
                "work via the OpenAIBackend base."
            )
        return super().open_session(entries, prompt_ctx, runtime_kwargs)
