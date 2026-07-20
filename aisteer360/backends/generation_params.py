"""Normalized generation parameters with per-backend rendering.

HF-style `**gen_kwargs` remain the user-facing vocabulary throughout the toolkit (pipeline, use
cases, `Benchmark.gen_kwargs`); `GenerationParams.from_gen_kwargs` is the single parsing point.

`to_hf_kwargs()` reproduces today's pass-through exactly — unknown keys flow through untouched so
the HF path never becomes stricter. `to_openai_kwargs(strict)` renders the canonical fields onto the
OpenAI vocabulary and routes vLLM extensions through `extra_body`; unknown or unsupported keys raise
`UnsupportedGenerationParam` when `strict` (the default) and are dropped with a single warning
otherwise.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from aisteer360.backends.errors import UnsupportedGenerationParam

logger = logging.getLogger(__name__)

# gen_kwargs keys rendered onto same-named OpenAI fields
_OPENAI_PASSTHROUGH = ("temperature", "top_p", "n", "seed", "stop")

# gen_kwargs keys routed through the vLLM `extra_body` extension
_OPENAI_EXTRA_BODY = ("top_k", "min_p", "repetition_penalty", "stop_token_ids", "logprobs", "prompt_logprobs")

# HF-only control flags consumed by the pipeline/backend, not forwarded as sampler params to any server
_HF_CONTROL_FLAGS = ("return_full_sequence", "output_hidden_states", "output_attentions")

# gen_kwargs with no OpenAI equivalent — rejected (strict) or dropped (non-strict) on API backends
_OPENAI_UNSUPPORTED = ("num_beams", "bad_words_ids", "constraints", "num_return_sequences")


@dataclass(slots=True)
class GenerationParams:
    """Backend-agnostic generation parameters parsed from HF-style `gen_kwargs`.

    The original kwargs are retained verbatim in `raw` so `to_hf_kwargs()` is an exact pass-through;
    the canonical fields drive the OpenAI rendering.

    Attributes:
        raw: The original gen_kwargs, unmodified (source of truth for the HF path).
        max_new_tokens: Maximum tokens to generate, if set.
        temperature, top_p, n, seed, stop: Standard sampling controls (same-named on OpenAI).
        greedy: Whether decoding is greedy (`do_sample=False`); renders to `temperature=0` on OpenAI.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    seed: int | None = None
    stop: list[str] | str | None = None
    greedy: bool = False

    @classmethod
    def from_gen_kwargs(cls, gen_kwargs: dict[str, Any] | None) -> "GenerationParams":
        """Parse HF-style `gen_kwargs` into a `GenerationParams`.

        Args:
            gen_kwargs: The user-facing generation kwargs (may be `None`).

        Returns:
            A `GenerationParams` retaining the raw kwargs and canonical fields.

        Raises:
            UnsupportedGenerationParam: If `max_length` is supplied (use `max_new_tokens`).
        """
        raw = dict(gen_kwargs or {})
        if "max_length" in raw:
            raise UnsupportedGenerationParam("max_length", "any", model=None)

        greedy = raw.get("do_sample") is False
        n = raw.get("n", raw.get("num_return_sequences"))

        return cls(
            raw=raw,
            max_new_tokens=raw.get("max_new_tokens"),
            temperature=raw.get("temperature"),
            top_p=raw.get("top_p"),
            n=n,
            seed=raw.get("seed"),
            stop=raw.get("stop"),
            greedy=greedy,
        )

    def replace(self, **changes: Any) -> "GenerationParams":
        """Return a copy with canonical fields (and mirrored `raw` entries) updated.

        Changes are applied to both the canonical field and the corresponding `raw` key so the HF
        pass-through stays consistent (used by output controls that mutate e.g. `max_new_tokens`).
        """
        raw = dict(self.raw)
        for key, value in changes.items():
            if key == "max_new_tokens":
                raw["max_new_tokens"] = value
            elif key in _OPENAI_PASSTHROUGH:
                raw[key] = value
        return replace(self, raw=raw, **changes)

    def to_hf_kwargs(self) -> dict[str, Any]:
        """Return the gen_kwargs for `model.generate`, an exact pass-through of the original kwargs.

        Unknown keys flow through untouched — the HF path must not become stricter than it is today.
        """
        return dict(self.raw)

    def to_openai_kwargs(self, strict: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        """Render onto the OpenAI vocabulary plus a vLLM `extra_body`.

        Args:
            strict: When `True` (default), unknown or unsupported keys raise
                `UnsupportedGenerationParam`. When `False`, they are dropped with a single warning.

        Returns:
            A `(kwargs, extra_body)` pair: `kwargs` for the OpenAI client call, `extra_body` for
            vLLM-specific extensions.

        Raises:
            UnsupportedGenerationParam: If `strict` and an unknown/unsupported key is present.
        """
        kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        dropped: list[str] = []

        def _reject(key: str) -> None:
            if strict:
                raise UnsupportedGenerationParam(key, "openai")
            dropped.append(key)

        for key, value in self.raw.items():
            if key == "max_new_tokens":
                kwargs["max_tokens"] = value
            elif key == "do_sample":
                if value is False:
                    kwargs["temperature"] = 0.0
                # do_sample=True is the OpenAI default; no explicit param needed
            elif key == "num_return_sequences":
                kwargs["n"] = value
            elif key in _OPENAI_PASSTHROUGH:
                kwargs[key] = value
            elif key in _OPENAI_EXTRA_BODY:
                extra_body[key] = value
            elif key in _HF_CONTROL_FLAGS:
                continue  # consumed elsewhere; never a server sampler param
            elif key in _OPENAI_UNSUPPORTED:
                _reject(key)
            else:
                _reject(key)

        # a greedy request with no explicit temperature still renders to temperature=0
        if self.greedy and "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        if dropped:
            warnings.warn(
                f"Dropping generation parameter(s) unsupported by the OpenAI backend: {sorted(set(dropped))}.",
                UserWarning,
            )
        return kwargs, extra_body
