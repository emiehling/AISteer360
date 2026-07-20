"""`OpenAIBackend`: a backend for any OpenAI-compatible endpoint.

No steering here — the vLLM-hook steering subclass is `vllm_hook.py` (doc 06). Capabilities are
`MESSAGES | TEXT`, extended by lazy feature probes to `TOKEN_IDS` (token-array prompts) and
`SCORING` (`prompt_logprobs`). The module depends only on `openai` (extra `aisteer360[openai]`) and
never imports `vllm`.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from aisteer360.backends.base import Backend, BackendCapabilities, SteeringSession, StateControlEntry
from aisteer360.backends.errors import (
    BackendConnectionError,
    BatchPartialFailure,
)
from aisteer360.backends.generation_params import GenerationParams
from aisteer360.backends.openai_compat._async import run_coros
from aisteer360.backends.specs import BackendSpec
from aisteer360.core.output import Output
from aisteer360.core.requirements import Capability

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from aisteer360.core.prompt import PreparedPrompt

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _import_openai():
    """Import the async/sync OpenAI clients, with an actionable message when the extra is missing."""
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415

        return AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAIBackend requires the `openai` package. Install the extra: "
            "`pip install aisteer360[openai]` (or `uv pip install -e .[openai]`)."
        ) from exc


class OpenAIBackend(Backend):
    """Backend for an OpenAI-compatible server (prompting and scoring; no steering).

    Args:
        base_url: The endpoint base URL.
        model: The served model name.
        api_key: Bearer key (default `"EMPTY"` for local servers).
        tokenizer_name_or_path: Optional client-side HF tokenizer; `None` disables token-level
            features (token-array prompts and scoring).
        max_concurrency: Maximum in-flight requests.
        timeout_s: Per-request timeout in seconds.
        max_retries: Retry budget for transient failures (429/5xx/connection).
        strict_generation_params: When True (default), unsupported gen params raise; else they are
            dropped with a warning.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        tokenizer_name_or_path: str | None = None,
        max_concurrency: int = 8,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        strict_generation_params: bool = True,
    ) -> None:
        async_client_cls = _import_openai()
        self.base_url = base_url
        self.model_identity = model
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.strict_generation_params = strict_generation_params
        self.spec = BackendSpec(kind="openai", model=model, base_url=base_url)

        self._client = async_client_cls(base_url=base_url, api_key=api_key, timeout=timeout_s)

        self.tokenizer: PreTrainedTokenizerBase | None = None
        if tokenizer_name_or_path is not None:
            from transformers import AutoTokenizer

            from aisteer360.utils.tokenization import ensure_pad_token

            self.tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(tokenizer_name_or_path))

        # lazy, cached feature-probe results and the derived capability note
        self._supports_token_arrays: bool | None = None
        self._supports_prompt_logprobs: bool | None = None
        self._capability_notes: dict[Any, str] = {}
        self._warned_template_mismatch = False

    @classmethod
    def from_spec(cls, spec: BackendSpec) -> "OpenAIBackend":
        """Build from a spec, mapping `spec.base_url` / `spec.model` onto the constructor."""
        backend = cls(base_url=spec.base_url, model=spec.model, **dict(spec.kwargs))
        backend.spec = spec
        return backend

    # capability negotiation

    @property
    def capabilities(self) -> BackendCapabilities:
        """`MESSAGES | TEXT`, extended by probes to `TOKEN_IDS` and `SCORING` when confirmed."""
        capabilities = Capability.MESSAGES | Capability.TEXT
        if self._probe_token_arrays():
            capabilities |= Capability.TOKEN_IDS
            if self._probe_prompt_logprobs():
                capabilities |= Capability.SCORING
        elif self._probe_prompt_logprobs():
            # prompt_logprobs works but scoring needs token arrays (safe prompt/ref seam)
            self._capability_notes["scoring"] = (
                "SCORING withheld: the server accepts `prompt_logprobs` but not token-array prompts, "
                "which are required to score at the prompt/ref boundary safely."
            )
        return BackendCapabilities(
            capabilities=capabilities,
            max_concurrency=self.max_concurrency,
            accepts_artifacts=frozenset(),
            notes=dict(self._capability_notes),
        )

    def _probe_token_arrays(self) -> bool:
        """Confirm (once) whether the server accepts a token-array completion prompt."""
        if self._supports_token_arrays is None:
            if self.tokenizer is None:
                self._supports_token_arrays = False
            else:
                try:
                    self._run_single(
                        lambda: self._client.completions.create(
                            model=self.model_identity, prompt=[[0]], max_tokens=1
                        )
                    )
                    self._supports_token_arrays = True
                except Exception:
                    self._supports_token_arrays = False
        return self._supports_token_arrays

    def _probe_prompt_logprobs(self) -> bool:
        """Confirm (once) whether the server returns `prompt_logprobs`."""
        if self._supports_prompt_logprobs is None:
            try:
                response = self._run_single(
                    lambda: self._client.completions.create(
                        model=self.model_identity, prompt="probe", max_tokens=1,
                        extra_body={"prompt_logprobs": 0},
                    )
                )
                choice = response.choices[0]
                self._supports_prompt_logprobs = getattr(choice, "prompt_logprobs", None) is not None or (
                    getattr(response, "prompt_logprobs", None) is not None
                )
            except Exception:
                self._supports_prompt_logprobs = False
        return self._supports_prompt_logprobs

    # request driving with retries

    def _run_single(self, factory):
        """Run one coroutine factory with retry/backoff; raise on terminal failure.

        `factory` must be a zero-arg callable returning a *fresh* coroutine each call (so a retry can
        re-issue the request rather than re-await a spent coroutine).
        """
        results = self._run_with_retry([factory])
        result = results[0]
        if isinstance(result, BaseException):
            raise result
        return result

    def _run_with_retry(self, factories):
        """Run coroutine factories with bounded concurrency and per-slot retry on transient errors."""
        def _wrap(factory):
            async def _attempt():
                delay = 0.5
                last_exc = None
                for _ in range(self.max_retries + 1):
                    try:
                        return await factory()
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_retryable(exc):
                            raise
                        last_exc = exc
                        retry_after = self._retry_after(exc)
                        await _sleep(retry_after if retry_after is not None else delay)
                        delay = min(delay * 2, 8.0)
                raise last_exc
            return _attempt

        import asyncio

        async def _sleep(seconds):
            await asyncio.sleep(seconds)

        return run_coros([_wrap(factory) for factory in factories], self.max_concurrency)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status in _RETRYABLE_STATUS:
            return True
        name = type(exc).__name__
        return "Connection" in name or "Timeout" in name

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("Retry-After") or headers.get("retry-after")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    # session lifecycle

    def open_session(
        self,
        entries: list[StateControlEntry],
        prompt_ctx: "PreparedPrompt",
        runtime_kwargs: dict,
    ) -> "OpenAISession":
        """Open a concurrency-safe API session. Rejects state-control entries (no steering here)."""
        active = [entry for entry in entries if entry.plan is not None or entry.hooks is not None]
        if active:
            names = ", ".join(entry.control_name for entry in active)
            raise RuntimeError(
                f"OpenAIBackend does not apply state controls ({names}); use VLLMHookBackend for "
                "server-side steering, or HuggingFaceBackend in-process."
            )
        return OpenAISession(self)

    def close(self) -> None:
        """Close the underlying async client's transport."""
        import contextlib

        with contextlib.suppress(Exception):
            self._run_single(lambda: self._client.close())


class OpenAISession(SteeringSession):
    """A stateless, concurrency-safe session over an `OpenAIBackend`.

    Batched `generate` fans out one request per row and reassembles order-stably; per-row failures
    raise an aggregate `BatchPartialFailure` naming the failed indices — never a silent partial.
    """

    concurrency_safe = True
    model = None

    def __init__(self, backend: OpenAIBackend) -> None:
        self._backend = backend

    def generate(self, prepared: "PreparedPrompt", params: GenerationParams) -> Output:
        """Generate one `Output` per batch row via the server, reassembled in order.

        Args:
            prepared: The adapted prompt.
            params: Normalized generation parameters.

        Returns:
            A single `Output` whose fields are batched across rows (text list, per-row usage merged).

        Raises:
            BatchPartialFailure: If any row fails after retries.
            UnsupportedGenerationParam: If a param is unsupported and strict mode is on.
        """
        rows = self._rows(prepared)
        kwargs, extra_body = params.to_openai_kwargs(strict=self._backend.strict_generation_params)
        factories = [self._request_factory(row, kwargs, extra_body) for row in rows]
        results = self._backend._run_with_retry(factories)

        failures = [i for i, r in enumerate(results) if isinstance(r, BaseException)]
        if failures:
            raise BatchPartialFailure(
                failures,
                [results[i] for i in failures],
                base_url=self._backend.base_url,
                model=self._backend.model_identity,
            )

        texts: list[str] = []
        finish_reasons: list[str | None] = []
        usages: list[dict] = []
        for response in results:
            choice = response.choices[0]
            texts.append(self._choice_text(choice))
            finish_reasons.append(_map_finish_reason(getattr(choice, "finish_reason", None)))
            usages.append(_usage_dict(getattr(response, "usage", None)))

        return Output(
            output_text=texts,
            finish_reason=finish_reasons[0] if len(finish_reasons) == 1 else None,
            usage=_merge_usages(usages),
            metadata={"backend": "OpenAIBackend", "finish_reasons": finish_reasons},
        )

    def _rows(self, prepared: "PreparedPrompt") -> list[Any]:
        """Split a prepared prompt into per-row request payloads (messages / text / token ids)."""
        prompt = prepared.prompt
        if prepared.adaptation_level == "messages" and prepared.adapted_messages is not None:
            return list(prepared.adapted_messages)
        if prepared.adaptation_level == "tokens" and prepared.adapted_token_ids is not None:
            return [ids.tolist() for ids in prepared.adapted_token_ids]
        if prompt.modality == "chat":
            return list(prompt.messages)
        if prompt.modality == "text":
            return list(prompt.texts)
        return [ids.tolist() for ids in prompt.token_ids]

    def _request_factory(self, row, kwargs, extra_body):
        """Build a zero-arg coroutine factory for one row (chat / completion request)."""
        client = self._backend._client
        model = self._backend.model_identity
        call_kwargs = dict(kwargs)
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        if isinstance(row, list) and row and isinstance(row[0], dict):
            return lambda: client.chat.completions.create(model=model, messages=row, **call_kwargs)
        return lambda: client.completions.create(model=model, prompt=row, **call_kwargs)

    @staticmethod
    def _choice_text(choice) -> str:
        """Extract text from a chat or completion choice."""
        message = getattr(choice, "message", None)
        if message is not None:
            return message.content or ""
        return getattr(choice, "text", "") or ""

    def score(self, prepared: "PreparedPrompt", ref_output_ids: torch.Tensor) -> torch.Tensor | None:
        """Per-token log-probabilities of `ref_output_ids` via `prompt_logprobs` (token-ID prompts).

        Resolves prompt ids client-side, concatenates `prompt_ids + ref_ids`, requests
        `prompt_logprobs`, and slices positions `len(prompt_ids) .. len(concat) - 1`.

        Returns:
            A `[batch, ref_len]` tensor, or `None` if the backend does not grant `SCORING`.
        """
        if not (self._backend.capabilities.capabilities & Capability.SCORING):
            return None
        if self._backend.tokenizer is None:
            return None

        prompt_ids_rows = self._resolve_prompt_ids(prepared)
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)

        batch = len(prompt_ids_rows)
        if ref_output_ids.size(0) == 1 and batch > 1:
            ref_output_ids = ref_output_ids.expand(batch, -1)

        factories = []
        seams = []
        for i, prompt_ids in enumerate(prompt_ids_rows):
            ref_ids = ref_output_ids[i].tolist()
            concat = list(prompt_ids) + ref_ids
            seams.append((len(prompt_ids), len(concat)))
            factories.append(
                lambda concat=concat: self._backend._client.completions.create(
                    model=self._backend.model_identity, prompt=concat, max_tokens=0,
                    extra_body={"prompt_logprobs": 0},
                )
            )
        results = self._backend._run_with_retry(factories)

        rows = []
        for (start, end), result in zip(seams, results):
            if isinstance(result, BaseException):
                raise result
            prompt_logprobs = self._extract_prompt_logprobs(result)
            row = [self._logprob_at(prompt_logprobs, pos) for pos in range(start, end)]
            rows.append(torch.tensor(row, dtype=torch.float32))
        return torch.stack(rows, dim=0)

    def _resolve_prompt_ids(self, prepared: "PreparedPrompt") -> list[list[int]]:
        """Resolve each row's prompt to token ids client-side (tensor as-is; text/chat rendered)."""
        tokenizer = self._backend.tokenizer
        prompt = prepared.prompt
        if prepared.adaptation_level == "tokens" and prepared.adapted_token_ids is not None:
            return [ids.tolist() for ids in prepared.adapted_token_ids]
        if prompt.modality == "tensor":
            return [ids.tolist() for ids in prompt.token_ids]
        if prompt.modality == "chat" or prepared.adaptation_level == "messages":
            messages_batch = prepared.adapted_messages or prompt.messages
            rendered = tokenizer.apply_chat_template(
                messages_batch, add_generation_prompt=True, tokenize=True
            )
            return [list(row) for row in rendered] if isinstance(rendered[0], list) else [list(rendered)]
        return [tokenizer(text, add_special_tokens=True)["input_ids"] for text in prompt.texts]

    @staticmethod
    def _extract_prompt_logprobs(response) -> list:
        """Pull the `prompt_logprobs` array from a completion response (choice- or top-level)."""
        choice = response.choices[0]
        value = getattr(choice, "prompt_logprobs", None)
        if value is None:
            value = getattr(response, "prompt_logprobs", None)
        return value or []

    @staticmethod
    def _logprob_at(prompt_logprobs: list, position: int) -> float:
        """Return the logprob of the realized token at an absolute prompt position."""
        if position >= len(prompt_logprobs):
            return 0.0
        entry = prompt_logprobs[position]
        if entry is None:
            return 0.0
        # entry maps token-id (str) -> {logprob, rank, ...}; take the single realized token
        if isinstance(entry, dict):
            first = next(iter(entry.values()))
            return float(first["logprob"] if isinstance(first, dict) else first)
        return float(entry)


def _map_finish_reason(reason: str | None) -> str | None:
    """Map an OpenAI finish reason onto the toolkit vocabulary."""
    if reason in (None, ""):
        return None
    return {"stop": "eos", "length": "length", "stop_sequence": "stop"}.get(reason, reason)


def _usage_dict(usage) -> dict:
    """Coerce a response `usage` object into a plain dict."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _merge_usages(usages: list[dict]) -> dict:
    """Sum per-row usage counts into one dict (None-safe)."""
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = False
    for usage in usages:
        for key in merged:
            value = usage.get(key)
            if value is not None:
                merged[key] += value
                seen = True
    return merged if seen else {}
