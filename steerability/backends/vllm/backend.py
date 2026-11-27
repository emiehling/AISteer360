"""The vLLM backends: the offline engine (`"vllm"`) and the OpenAI-compatible server
(`"vllm-serve"`).

This module holds the backend-side helpers (the artifact uploader, layout and tokenizer
resolution, discovery reconciliation, the engine request coroutines) and the two backend classes.
It imports cleanly without vLLM installed: constructing `VLLMBackend` requires the `vllm` optional
dependency (it boots an engine); `VLLMServeBackend` needs only a reachable vLLM server. Every
`from vllm ...` and `from vllm_hook_plugins ...` import stays inside a function or method body.
"""
import concurrent.futures
import gc
import hashlib
import itertools
import json
import logging
import os
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from steerability.algorithms.core.execution.backend import Backend
from steerability.algorithms.core.execution.contracts import BackendCapabilities
from steerability.algorithms.core.execution.fanout import TransportError
from steerability.algorithms.core.execution.payloads import (
    Artifact,
    InterventionSpec,
    LoRAArtifact,
    ModelFacts,
)
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.execution.staging import split_artifacts
from steerability.algorithms.core.internals.fingerprint import config_digest_dict, is_absent_chat_template_fingerprint
from steerability.algorithms.core.internals.model_layout import config_head_geometry, text_config
from steerability.backends.vllm.capabilities import _intersect_with_discovery, _reconcile_discovery, _vllm_capabilities
from steerability.backends.vllm.engine_thread import EngineThread
from steerability.backends.vllm.environment import engine_boot_environment, engine_environment
from steerability.backends.vllm.rendering import raise_for_spec_rejection
from steerability.backends.vllm.session import VLLMOfflineSession, VLLMServeSession
from steerability.utils.optional import require
from steerability.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 120.0
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_MAX_ATTEMPTS = 3


class _ArtifactUploader:
    """Materializes spec tensor payloads into the registry root the serving engine reads."""

    def __init__(self, root: str | None):
        self._root = root
        self._registry = None
        self._written: set[str] = set()

    def upload(self, spec: InterventionSpec) -> None:
        if spec.artifacts:
            self.upload_payloads(spec.artifacts)

    def upload_payloads(self, payloads) -> None:
        """Write content-addressed payloads into the registry, verifying each id."""
        if not payloads:
            return
        if self._registry is None:
            artifacts_module = require("vllm_hook_plugins.core.artifacts")
            self._registry = artifacts_module.ArtifactRegistry(self._root)
        for artifact_id, tensors in payloads.items():
            if artifact_id in self._written:
                continue
            written_id = self._registry.write(dict(tensors))
            if written_id != artifact_id:
                raise ValueError(
                    f"Artifact registry wrote {written_id} for a payload the spec references as "
                    f"{artifact_id}; the client and registry disagree on content addressing."
                )
            self._written.add(artifact_id)


def _reject_encoder_decoder(model_ref: str, trust_remote_code: bool = False) -> None:
    """Reject encoder-decoder models for vLLM execution (in-process only per the seam)."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError(
            f"Model {model_ref!r} is an encoder-decoder model; encoder-decoder execution is "
            "in-process only. Run this pipeline on the huggingface backend."
        )


def _config_layout(model_ref: str, trust_remote_code: bool = False) -> ModelFacts | None:
    """A client-side `ModelFacts` from the model config, or None when the config cannot be loaded.

    A loaded config lacking a structural fact (`hidden_size`, `num_hidden_layers`) raises
    `AttributeError`. The fingerprint hashes the config fields that `model_fingerprint` hashes
    (`config_digest_dict`, i.e., the diff dict without `transformers_version`). It identifies the
    architecture and configuration rather than the weights.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    facts = text_config(config)
    hidden_size = facts.hidden_size
    num_heads, head_dim = config_head_geometry(facts)
    dtype = getattr(config, "dtype", None)
    digest = hashlib.sha256(
        json.dumps(config_digest_dict(config), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return ModelFacts(
        num_layers=facts.num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=str(dtype).removeprefix("torch.") if dtype is not None else "unknown",
        model_fingerprint=digest,
        model_type=getattr(config, "model_type", None),
        model_ref=model_ref,
        fingerprint_grade="config",
    )


def _client_tokenizer(source: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    return ensure_pad_token(tokenizer)


async def _construct_engine(model_ref: str, engine_kwargs: dict):
    """An `AsyncLLM` built on the running loop, which the engine's own tasks then belong to.

    The `LLM_CLASS` usage context is required for the offline scheduler defaults (`max_num_seqs`,
    `max_num_batched_tokens`); `AsyncLLM.from_engine_args` otherwise uses `ENGINE_CONTEXT`, for
    which vLLM applies its smaller generic `SchedulerConfig` defaults.
    """
    from vllm import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(model=model_ref, **engine_kwargs), usage_context=UsageContext.LLM_CLASS,
    )


def adapter_identity(*, adapter: LoRAArtifact) -> tuple[str, int]:
    """The name and the integer id under which a vLLM engine or server serves a LoRA adapter.

    Both derive from the adapter's path and the model fingerprint of its provenance. A retrained adapter at the same
    path is therefore served as a new adapter, and two adapters on one engine never share an id (vLLM keys its loaded
    adapters and its cached prefixes by the id).

    Args:
        adapter: The adapter.

    Returns:
        The name (`"steered-"` and eight hex digits) and an integer id from 1 to 2**28.
    """
    identity = f"{adapter.path}:{adapter.provenance.model_fingerprint or ''}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"steered-{digest[:8]}", int(digest[:7], 16) + 1


def adapter_rank(*, path: str) -> int | None:
    """The largest rank of a PEFT LoRA adapter directory, from the `r` and `rank_pattern` of its `adapter_config.json`.

    Args:
        path: The adapter directory.

    Returns:
        The rank, or None when the file is missing or gives none.
    """
    config = Path(path) / "adapter_config.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ranks = [data.get("r"), *(data.get("rank_pattern") or {}).values()]
    ranks = [int(rank) for rank in ranks if isinstance(rank, int) and not isinstance(rank, bool)]
    return max(ranks) if ranks else None


_LORA_WEIGHT_MARKERS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def adapter_modules(*, path: str) -> frozenset[str] | None:
    """The names of the modules that a PEFT LoRA adapter directory changes.

    The names are read from the tensor names of `adapter_model.safetensors` (the name before `lora_A` or `lora_B`),
    or from a `target_modules` list in `adapter_config.json` when the directory has no safetensors file.

    Args:
        path: The adapter directory.

    Returns:
        The module names, or None when neither file gives them.
    """
    weights = Path(path) / "adapter_model.safetensors"
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt") as file:
            names = [key.split(".") for key in file.keys()]
        return frozenset(
            parts[parts.index(marker) - 1]
            for parts in names
            for marker in _LORA_WEIGHT_MARKERS
            if marker in parts[1:]
        )
    try:
        data = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    targets = data.get("target_modules")
    return frozenset(targets) if isinstance(targets, list) else None


async def _final_output(engine, prompt, sampling_params, request_id: str, lora_request=None):
    """The finished `RequestOutput` of one `AsyncLLM.generate` call."""
    final = None
    async for output in engine.generate(prompt, sampling_params, request_id, lora_request=lora_request):
        final = output
    return final


async def _collective_rpc(engine, method: str):
    """The per-worker replies of one collective RPC on the engine."""
    return await engine.collective_rpc(method, None, (), None)


async def _shutdown(engine) -> None:
    """Shut the engine down on the loop its tasks belong to."""
    engine.shutdown()


@dataclass(frozen=True, slots=True)
class RequestHandle:
    """One engine request submitted through the backend's engine thread.

    Attributes:
        request_id: The request id the engine was given.
        future: The future that resolves with the request's finished `RequestOutput`.
    """

    request_id: str
    future: concurrent.futures.Future

    def result(self, timeout: float | None = None) -> Any:
        """Block until the request finishes and return its `RequestOutput`."""
        return self.future.result(timeout)

    def done(self) -> bool:
        """Whether the request finished, failed, or was cancelled."""
        return self.future.done()

    def cancel(self) -> bool:
        """Cancel the request, which aborts it on the engine; False when it already completed."""
        return self.future.cancel()


class _DiscoveryNegotiation:
    """The negotiated capabilities of a vLLM backend that may hold a discovery payload."""

    _discovery: dict | None

    def negotiated_capabilities(self) -> BackendCapabilities:
        """The spec's advertisement, narrowed to this backend's discovery payload when it fetched one."""
        static = self.capabilities_for_spec(self.spec)
        if self._discovery is None:
            return static
        return _intersect_with_discovery(static, self._discovery)


class VLLMBackend(_DiscoveryNegotiation, Backend):
    """The offline vLLM engine backend.

    Boots one vLLM `AsyncLLM` engine per backend instance from the spec (`engine_kwargs` option
    forwarded to `AsyncEngineArgs`); requires the `vllm` optional dependency. The engine runs on an
    event loop that the backend owns on a daemon thread, and every engine operation is a coroutine
    submitted to that loop. The engine's scheduler admits requests from any thread as they
    arrive, and concurrent session calls therefore share one running batch. On a
    `hook_plugin` engine, the plugin's patch of `AsyncLLM.generate` stages each request's
    intervention and capture specs on the workers, attaches its capture payload, and clears its
    state when the request finishes or is aborted.

    The last `CheckpointArtifact` overrides the served model reference. A LoRA adapter is not a
    property of the backend: each session serves the adapter it is opened with (`open_session`) as a
    LoRA request on each of its requests, and one engine therefore serves the pipelines of several
    adapters. The engine serves adapters only when the spec's `engine_kwargs` enable them
    (`BackendSpec.lora_rank_limit`). When the spec declares `hook_plugin`, the
    unified worker is selected via `VLLM_HOOK_WORKER=unified`, and the discovery payload is
    fetched from the engine at construction and kept on the backend. `negotiated_capabilities()`
    narrows the spec's advertisement to it, while `capabilities_for_spec` gives the spec's
    advertisement without constructing the backend.

    The engine is built under vLLM's `LLM_CLASS` usage context, which keeps the scheduler defaults
    (`max_num_seqs`, `max_num_batched_tokens`) of an offline `vllm.LLM` engine. Without it vLLM
    applies its smaller generic scheduler defaults. Per-request lifecycle logging is off by default
    (`enable_log_requests=False`), and the engine's periodic stats logging keeps vLLM's default
    (`disable_log_stats=False`), which reports preemptions. An explicit `enable_log_requests` or
    `disable_log_stats` entry in `engine_kwargs` takes precedence.
    """

    def __init__(self, spec: BackendSpec, artifacts: Sequence[Artifact] = ()) -> None:
        if spec.kind != "vllm":
            raise ValueError(f"VLLMBackend requires a 'vllm' spec; got kind {spec.kind!r}.")
        self.spec = spec
        self._released = False
        self._engine = None
        self._engine_thread: EngineThread | None = None
        self._lora_requests: dict[str, Any] = {}
        require("vllm")

        checkpoint, _ = split_artifacts(artifacts=artifacts)
        model_ref = checkpoint.path if checkpoint is not None else spec.model
        if model_ref is None:
            raise ValueError("VLLMBackend needs a model reference on the spec or a checkpoint artifact.")
        trust_remote_code = bool(spec.get_option("trust_remote_code", default=False))
        _reject_encoder_decoder(model_ref, trust_remote_code)

        engine_kwargs = dict(spec.get_option("engine_kwargs", default={}) or {})
        # default to a compact grammar so json constraints match the in-process automaton
        # (disable_any_whitespace needs an explicit backend); caller kwargs win
        engine_kwargs.setdefault("structured_outputs_config", {"disable_any_whitespace": True, "backend": "xgrammar"})
        if trust_remote_code:
            engine_kwargs.setdefault("trust_remote_code", True)
        if spec.get_option("hook_plugin"):
            # worker hooks do not run under CUDA-graph replay; spec construction rejects an
            # explicit False, so this only fills the default
            engine_kwargs.setdefault("enforce_eager", True)
        # per-request lifecycle logging defaults off; stats logging keeps vLLM's default; caller kwargs win
        engine_kwargs.setdefault("enable_log_requests", False)

        self._request_counter = itertools.count()
        # the pipeline records a backend only when the constructor returns, so a failure from
        # here on must release the engine thread and any engine before propagating
        try:
            self._engine_thread = EngineThread(name=f"vllm-{spec.spec_hash[:8]}")
            self._engine_thread.start()
            # the boot environment is scoped to this engine's construction and restored
            # afterwards, so a later plugin-free engine in the same process is unaffected
            forced, defaults = engine_boot_environment(bool(spec.get_option("hook_plugin")))
            with engine_environment(forced, defaults):
                self._engine = self._engine_thread.run(_construct_engine(model_ref, engine_kwargs))

            tokenizer_source = (
                spec.get_option("tokenizer_name_or_path")
                or model_ref
            )
            self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
            self._layout = _config_layout(model_ref, trust_remote_code)
            self._plain_salt = uuid.uuid4().hex
            self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
            self._discovery: dict | None = None
            if spec.get_option("hook_plugin"):
                self._discovery = self._fetch_discovery()
        except Exception:
            self.release()
            raise

    def stage_artifacts(self, payloads) -> None:
        """Write each content-addressed artifact into the plugin registry the engine reads.

        The offline engine shares the process's filesystem, so staging is a registry write
        (idempotent, verified against the content address).
        """
        self._artifact_uploader.upload_payloads(payloads)

    def _fetch_discovery(self) -> dict | None:
        try:
            replies = self._engine_thread.run(_collective_rpc(self._engine, "hook_capabilities"))
        except Exception as error:
            logger.warning("vLLM-Hook discovery failed: %s", error)
            return None
        payload = next((reply for reply in replies or () if reply), None)
        if payload is None:
            logger.warning(
                "vLLM-Hook discovery returned no payload; is VLLM_HOOK_WORKER=unified active?"
            )
            return None
        _reconcile_discovery(self.spec, self.capabilities_for_spec(self.spec), payload)
        return payload

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=True)

    def open_session(self, *, adapter: LoRAArtifact | None = None) -> "VLLMOfflineSession":
        """Open a request session over the shared engine.

        Args:
            adapter: The LoRA adapter that every request of the session is served under (`lora_request`), or None.

        Raises:
            RuntimeError: If the backend has been released.
            ValueError: If the engine cannot serve the adapter (see `lora_request`).
        """
        self._require_engine()
        return VLLMOfflineSession(backend=self, lora_request=self.lora_request(adapter=adapter))

    def lora_request(self, *, adapter: LoRAArtifact | None) -> Any:
        """The vLLM `LoRARequest` that serves an adapter on this engine, or None without an adapter.

        The request's name and id come from `adapter_identity`, and the request is built once per adapter.

        Args:
            adapter: The adapter, or None.

        Returns:
            The request, or None.

        Raises:
            RuntimeError: If the backend has been released.
            ValueError: If the spec's `engine_kwargs` do not set `enable_lora`, the engine's dtype is not float16 or
                bfloat16 (vLLM applies adapters in these only), the adapter's rank (from its `adapter_config.json`) is
                above the engine's `max_lora_rank`, or the adapter changes a module that the engine's
                `lora_target_modules` leave out (vLLM would apply the adapter in part). The message gives the option
                to set.
        """
        if adapter is None:
            return None
        name, lora_id = adapter_identity(adapter=adapter)
        if name in self._lora_requests:
            return self._lora_requests[name]
        dtype = self._require_engine().model_config.dtype
        if not self.spec.get_option("engine_kwargs", "enable_lora"):
            raise ValueError(
                f"The engine serves no LoRA adapter, and a session asked for the one at {adapter.path!r}; set "
                "engine_kwargs={'enable_lora': True} in the backend spec's options.",
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"The engine runs in {str(dtype).removeprefix('torch.')}, and vLLM applies a LoRA adapter in float16 "
                f"or bfloat16 only; set engine_kwargs={{'dtype': 'bfloat16'}} in the backend spec's options to serve "
                f"the adapter at {adapter.path!r}.",
            )
        limit = self.spec.lora_rank_limit()
        rank = adapter_rank(path=str(adapter.path))
        if rank is not None and rank > limit:
            raise ValueError(
                f"The LoRA adapter at {adapter.path!r} has rank {rank}, above the engine's max_lora_rank of {limit}; "
                f"set a max_lora_rank of at least {rank} in the engine_kwargs of the backend spec's options.",
            )
        modules = adapter_modules(path=str(adapter.path))
        outside = [] if modules is None else self.spec.lora_modules_left_out(modules=modules)
        if outside:
            targets = list(self.spec.get_option("engine_kwargs", "lora_target_modules"))
            raise ValueError(
                f"The LoRA adapter at {adapter.path!r} changes the modules {outside}, which the engine's "
                f"lora_target_modules {targets} leave out, and vLLM would apply the adapter in part; add them to "
                "lora_target_modules in the engine_kwargs of the backend spec's options.",
            )
        from vllm.lora.request import LoRARequest

        request = LoRARequest(lora_name=name, lora_int_id=lora_id, lora_path=str(adapter.path))
        self._lora_requests[name] = request
        return request

    def _require_engine(self):
        """The loaded engine, or a `RuntimeError` when the backend has been released."""
        if self._engine is None:
            raise RuntimeError(
                "This VLLMBackend was released; construct a new backend (or a new "
                "SteeringPipeline operation, which reconstructs backends automatically)."
            )
        return self._engine

    def _submit_request(self, prompt, sampling_params, *, lora_request=None) -> RequestHandle:
        """Submit one engine request through the engine thread and return its handle.

        The request's output kind is set to final-only, which makes the engine return one output
        when the request finishes. Request ids combine the spec hash with a per-backend counter.

        Args:
            prompt: The engine prompt (a `TokensPrompt`, with its `cache_salt` when salted).
            sampling_params: The request's `SamplingParams`; its `output_kind` is overwritten.
            lora_request: The LoRA request to serve the request under, or None.

        Returns:
            The `RequestHandle` whose future resolves with the finished `RequestOutput`.

        Raises:
            RuntimeError: If the backend has been released.
            EngineClosedError: If the engine thread closes before the request is submitted.
        """
        engine = self._require_engine()
        from vllm.sampling_params import RequestOutputKind

        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        request_id = f"{self.spec.spec_hash[:8]}-{next(self._request_counter)}"
        future = self._engine_thread.submit(
            _final_output(engine, prompt, sampling_params, request_id, lora_request=lora_request)
        )
        return RequestHandle(request_id=request_id, future=future)

    def release(self) -> None:
        """Shut the engine down explicitly and mark the backend unusable.

        In-flight requests are cancelled first, and their waiters raise `EngineClosedError`. The
        engine is then shut down on its event loop, and the engine thread is stopped and joined.
        Release is idempotent; after it, a new backend must be constructed. The distributed-state
        teardown is process-global, so release assumes no other loaded vLLM engine in the process.
        Ray-based executors are out of scope. Engine-touching calls on any still-open session raise
        `RuntimeError` after release.
        """
        if self._released:
            return
        self._released = True
        engine, self._engine = self._engine, None
        engine_thread = self._engine_thread
        self._lora_requests = {}

        if engine_thread is not None:
            engine_thread.cancel_pending()
            if engine is not None:
                try:
                    engine_thread.run(_shutdown(engine))
                except Exception:
                    logger.warning("vLLM engine shutdown failed; continuing.", exc_info=True)
            engine_thread.close()

        del engine
        gc.collect()

        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            logger.warning("vLLM distributed-state teardown failed; continuing.", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMServeBackend(_DiscoveryNegotiation, Backend):
    """The vLLM OpenAI-compatible server backend.

    Targets a vLLM server rather than an arbitrary OpenAI-compatible endpoint: construction
    verifies the server's version surface (`GET /version`), fetches the plugin discovery payload
    (`GET /v1/hook/capabilities`) when the spec declares `hook_plugin` and keeps it on the backend
    (`negotiated_capabilities()` narrows the spec's advertisement to it), and checks the served
    model id against the spec (or serves the pipeline's structural artifacts). Prompts submit as
    token ids on the completions endpoint with the token-id return option; the chat endpoint is
    not used. Requires no local vLLM installation.

    A LoRA adapter is selected per session (`open_session`). A session loads its adapter onto the
    server on first use (`/v1/load_lora_adapter`, which needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING` on
    the server) under the name `adapter_identity` gives, and its requests name that adapter as their
    model. An adapter is loaded once per backend.

    Spec options: `base_url` (required, the server root), `api_key`, `request_timeout`,
    `max_concurrency`, `max_retries`, `retry_backoff`, `tokenizer_name_or_path`,
    `trust_remote_code`, `hook_plugin`.
    """

    def __init__(self, spec: BackendSpec, artifacts: Sequence[Artifact] = ()) -> None:
        if spec.kind != "vllm-serve":
            raise ValueError(f"VLLMServeBackend requires a 'vllm-serve' spec; got kind {spec.kind!r}.")
        self.spec = spec
        base_url = spec.get_option("base_url")
        if not base_url:
            raise ValueError("VLLMServeBackend requires a 'base_url' option on the spec.")
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._api_key = spec.get_option("api_key")
        self._timeout = float(spec.get_option("request_timeout", default=_DEFAULT_REQUEST_TIMEOUT))
        self.max_concurrency = int(spec.get_option("max_concurrency", default=_DEFAULT_MAX_CONCURRENCY))
        self.max_attempts = int(spec.get_option("max_retries", default=_DEFAULT_MAX_ATTEMPTS))
        self.backoff_base = float(spec.get_option("retry_backoff", default=0.5))
        trust_remote_code = bool(spec.get_option("trust_remote_code", default=False))

        version = self._get_json("/version")
        if not isinstance(version, dict) or "version" not in version:
            raise ValueError(
                f"The endpoint at {self._base_url} does not expose the vLLM version surface; "
                "only vLLM servers are supported."
            )

        self._discovery: dict | None = None
        if spec.get_option("hook_plugin"):
            try:
                self._discovery = self._get_json("/v1/hook/capabilities")
            except (TransportError, ValueError) as error:
                raise ValueError(
                    f"The spec declares hook_plugin but {self._base_url} serves no "
                    f"/v1/hook/capabilities discovery surface: {error}"
                ) from error
            _reconcile_discovery(spec, self.capabilities_for_spec(spec), self._discovery)

        checkpoint, _ = split_artifacts(artifacts=artifacts)
        expected_model = checkpoint.path if checkpoint is not None else spec.model
        served = self._served_model_ids()
        if expected_model is None:
            if len(served) != 1:
                raise ValueError(
                    f"The spec names no model and the server serves {served}; set the "
                    "spec's model to disambiguate.",
                )
            self._served_model = served[0]
        elif expected_model in served:
            self._served_model = expected_model
        else:
            raise ValueError(
                f"The server at {self._base_url} serves {served}, not the configured "
                f"model {expected_model!r}.",
            )
        self._adapters: set[str] = set()
        self._adapter_lock = threading.Lock()

        tokenizer_source = (
            spec.get_option("tokenizer_name_or_path")
            or (checkpoint.path if checkpoint is not None else None)
            or spec.model
        )
        if tokenizer_source is None:
            tokenizer_source = self._served_model
        self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
        self._layout = _config_layout(
            (checkpoint.path if checkpoint is not None else None) or spec.model or self._served_model,
            trust_remote_code,
        )
        self._plain_salt = uuid.uuid4().hex
        # spec artifacts write to the registry root the server reads; shared_fs visibility
        # is verified against the server after writing (see stage_artifacts)
        self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
        if self._discovery is not None:
            self._verify_fingerprints()

    def _served_model_ids(self) -> list[str]:
        payload = self._get_json("/v1/models")
        return [entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)]

    def stage_artifacts(self, payloads) -> None:
        """Make each content-addressed artifact available to the serving engine.

        With an `artifact_dir` option the payloads are written into that registry root, which
        must be the server's registry directory (its `VLLM_HOOK_REGISTRY_DIR`) on a shared
        filesystem; visibility is verified through the server's artifact route when the
        discovery payload advertises `artifact_registry_root`. Otherwise each payload is PUT
        to the plugin's artifact route (`/v1/hook/artifacts/{id}`, body safetensors bytes, id
        verified server-side); already-exists is success.
        """
        if not payloads:
            return
        if self.spec.get_option("artifact_dir"):
            self._artifact_uploader.upload_payloads(payloads)
            self._verify_shared_fs_visibility(payloads)
            return
        import safetensors.torch

        for artifact_id, tensors in payloads.items():
            if artifact_id in self._artifact_uploader._written:
                continue
            data = safetensors.torch.save({name: tensors[name] for name in sorted(tensors)})
            self._put_bytes(f"/v1/hook/artifacts/{artifact_id}", data)
            self._artifact_uploader._written.add(artifact_id)

    def _verify_shared_fs_visibility(self, payloads) -> None:
        """Probe that shared_fs artifacts are visible to the server's registry.

        Gated on the discovery payload advertising `artifact_registry_root` (servers that
        advertise it also serve `HEAD /v1/hook/artifacts/{id}`); older servers skip the
        probe and keep the write-and-trust behavior.
        """
        server_root = (self._discovery or {}).get("artifact_registry_root")
        if not server_root:
            return
        client_root = os.path.abspath(self.spec.get_option("artifact_dir"))
        for artifact_id in payloads:
            if self._head_ok(f"/v1/hook/artifacts/{artifact_id}"):
                continue
            raise ValueError(
                f"artifact {artifact_id} written under {client_root} is not visible to the "
                f"server's registry ({server_root}); the shared_fs transport requires "
                "artifact_dir and the server's VLLM_HOOK_REGISTRY_DIR to name the same "
                "directory. Set VLLM_HOOK_REGISTRY_DIR on the server, point artifact_dir at "
                "the server's registry root, or drop artifact_dir to use the HTTP artifact "
                "route."
            )

    def _head_ok(self, path: str) -> bool:
        """HEAD a server path; True on 200, False on 404, raise otherwise."""
        request = urllib.request.Request(f"{self._base_url}{path}", method="HEAD")
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            body = error.read().decode("utf-8", errors="replace")
            raise ValueError(f"HTTP {error.code} from {self._base_url}{path}: {body}") from error

    def _put_bytes(self, path: str, data: bytes) -> None:
        """PUT raw bytes to the server, mapping a missing route to a configuration error."""
        import urllib.error
        import urllib.request

        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, data=data, method="PUT")
        request.add_header("Content-Type", "application/octet-stream")
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                return
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in (404, 405):
                raise ValueError(
                    f"{self._base_url} serves no artifact route ({error.code}); update the "
                    "server's vllm_hook_plugins, or configure artifact_dir on a filesystem "
                    "shared with the server."
                ) from error
            raise_for_spec_rejection(body)
            raise ValueError(f"HTTP {error.code} from {url}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(f"artifact upload to {url} failed: {error}") from error

    def model_for(self, *, adapter: LoRAArtifact | None) -> str:
        """The model name that a request names to be served under an adapter (the served base model without one).

        An adapter that the backend has not loaded yet is loaded onto the server first (`_load_lora_adapter`).

        Args:
            adapter: The adapter, or None.

        Returns:
            The model name.

        Raises:
            ValueError: If the server does not serve the adapter's base model or refuses the adapter.
        """
        if adapter is None:
            return self._served_model
        name, _ = adapter_identity(adapter=adapter)
        with self._adapter_lock:
            if name not in self._adapters:
                self._load_lora_adapter(lora=adapter)
                self._adapters.add(name)
        return name

    def _load_lora_adapter(self, lora: LoRAArtifact) -> str:
        served = self._served_model_ids()
        base = lora.base_model or self.spec.model
        if base and base not in served:
            raise ValueError(
                f"The server at {self._base_url} serves {served}, not the adapter's base "
                f"model {base!r}."
            )
        adapter_name, _ = adapter_identity(adapter=lora)
        if adapter_name in served:
            return adapter_name
        try:
            self._post_json(
                "/v1/load_lora_adapter",
                {"lora_name": adapter_name, "lora_path": lora.path},
                expect_json=False,
            )
        except (TransportError, ValueError) as error:
            raise ValueError(
                f"Could not load the LoRA artifact at {lora.path!r} onto the server "
                f"(dynamic adapter loading requires VLLM_ALLOW_RUNTIME_LORA_UPDATING): {error}"
            ) from error
        return adapter_name

    def _verify_fingerprints(self) -> None:
        """Verify the client tokenizer against the discovery payload's fingerprint recipes.

        Uses the plugin's engine-free `core.fingerprints` when the `vllm_hook_plugins` package
        is installed; mismatches warn rather than raise. Without the package, verification is
        skipped with a warning. A served fingerprint equal to the absent-template digest means
        the server exposes no chat template, so the comparison is skipped since a mismatch
        against it would reflect exposure rather than divergence.
        """
        model_block = (self._discovery or {}).get("model", {})
        remote_chat = model_block.get("chat_template_fingerprint")
        if remote_chat is None:
            return
        if is_absent_chat_template_fingerprint(remote_chat):
            logger.debug(
                "The server at %s does not expose a chat template (fingerprint %s is the "
                "absent-template digest); skipping the chat template comparison.",
                self._base_url, remote_chat,
            )
            return
        try:
            from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint
        except ImportError:
            logger.warning(
                "Install vllm-hook-plugins to verify the client tokenizer against the server's "
                "fingerprints; skipping verification."
            )
            return
        local_chat = chat_template_fingerprint(getattr(self.tokenizer, "chat_template", None))
        if local_chat != remote_chat:
            logger.warning(
                "Client chat template (fingerprint %s) differs from the served one (%s); "
                "templated prompts may diverge from server-side expectations.",
                local_chat, remote_chat,
            )

    def _request_json(self, path: str, payload: dict | None, expect_json: bool = True) -> dict:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = ""
            try:
                body = error.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            # 5xx, timeouts, and rate limiting are transport-level and safe to retry
            if error.code >= 500 or error.code in (408, 429):
                raise TransportError(f"HTTP {error.code} from {url}: {body}") from error
            # admission rejections carry the plugin's E_* code and JSON path verbatim
            raise_for_spec_rejection(body)
            raise ValueError(f"HTTP {error.code} from {url}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(f"Request to {url} failed: {error}") from error
        if not expect_json:
            return {"text": body}
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError(f"Non-JSON response from {url}: {error}") from error

    def _get_json(self, path: str) -> dict:
        return self._request_json(path, None)

    def _post_json(self, path: str, payload: dict, expect_json: bool = True) -> dict:
        return self._request_json(path, payload, expect_json=expect_json)

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=False)

    def open_session(self, *, adapter: LoRAArtifact | None = None) -> "VLLMServeSession":
        """Open a request session over the shared connection.

        Args:
            adapter: The LoRA adapter that the session's requests name as their model (`model_for`), or None.
        """
        return VLLMServeSession(backend=self, adapter=adapter)
