"""Anthropic API generator + judge providers.

Both providers are thin sync adapters over the `anthropic` SDK. We import the SDK lazily so an
agent that doesn't select anthropic doesn't need it installed.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from aisteer360.evaluation.metrics.base_judge import build_structured_parser

from .base import GenerationProvider, JudgeProvider

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENCY = 8


def _load_anthropic():
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Install it with: pip install anthropic"
        ) from exc
    return anthropic


class AnthropicGenerationProvider(GenerationProvider):
    """Generate contrastive-pair responses via the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ):
        anthropic = _load_anthropic()
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_concurrency = max(1, max_concurrency)

    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[str]:
        if not user_prompts:
            return []
        results: list[str | None] = [None] * len(user_prompts)

        def one(idx: int, prompt: str) -> None:
            if cancel_check is not None and cancel_check():
                results[idx] = ""
                return
            resp = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            results[idx] = _extract_text(resp)

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = [pool.submit(one, i, p) for i, p in enumerate(user_prompts)]
            for _ in as_completed(futures):
                if cancel_check is not None and cancel_check():
                    for f in futures:
                        f.cancel()
                    return []
        if any(r is None for r in results):
            return []
        return [r for r in results if r is not None]


class AnthropicJudgeProvider(JudgeProvider):
    """Score responses with the Anthropic Messages API using the shared rubric parser."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ):
        anthropic = _load_anthropic()
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_concurrency = max(1, max_concurrency)

    def score(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        template: str,
        scale: tuple[float, float],
    ) -> dict[str, Any]:
        fmt, parse = build_structured_parser(scale)
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must be the same length")
        scores: list[float] = [float("nan")] * len(responses)

        def one(idx: int) -> None:
            rubric = template.format(
                response=responses[idx],
                prompt=prompts[idx],
                lower_bound=scale[0],
                upper_bound=scale[1],
            ) if "{prompt}" in template else template.format(
                response=responses[idx],
                lower_bound=scale[0],
                upper_bound=scale[1],
            )
            user_text = rubric + "\n\n" + fmt
            try:
                resp = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=128,
                    temperature=0.0,
                    messages=[{"role": "user", "content": user_text}],
                )
                text = _extract_text(resp)
                scores[idx] = parse(text, scale)
            except Exception as exc:
                logger.warning("anthropic judge failed for idx=%d: %s", idx, exc)

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            list(pool.map(one, range(len(responses))))

        finite = [s for s in scores if s == s]  # exclude NaN
        mean = sum(finite) / len(finite) if finite else float("nan")
        return {"mean_score": mean, "scores": scores, "raw_scores": [[s] for s in scores]}


def _extract_text(resp: Any) -> str:
    """Extract the textual content from an Anthropic message response."""
    parts = getattr(resp, "content", None) or []
    out: list[str] = []
    for part in parts:
        if getattr(part, "type", None) == "text":
            out.append(getattr(part, "text", ""))
    return "".join(out).strip()
