"""The vLLM backends: the offline engine (`"vllm"`) and the OpenAI-compatible server
(`"vllm-serve"`).

This module imports cleanly without vLLM installed. The capability tables are static data used
by `check()`; the strict parameter-rendering table and the request/response mapping helpers are
plain functions. Constructing `VLLMBackend` requires the `vllm` optional dependency (it boots an
engine); `VLLMServeBackend` needs only a reachable vLLM server.
"""
import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Literal

import torch

from aisteer360.algorithms.core.execution.artifacts import (
    Artifact,
    CheckpointArtifact,
    LoRAArtifact,
)
from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.capabilities import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.fanout import (
    PartialBatchError,
    TransportError,
    derive_item_seed,
    run_bounded,
    with_transport_retries,
)
from aisteer360.algorithms.core.execution.items import (
    CaptureResult,
    GenerationItem,
    HookEntry,
    InterventionEntry,
    ItemResult,
    ProcessorSpecEntry,
    ScoringItem,
    StackEntry,
)
from aisteer360.algorithms.core.execution.layout import ModelLayout
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.prompts import PreparedPrompt
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.execution.support import UnsupportedOperationError
from aisteer360.algorithms.core.output import Output
from aisteer360.utils.optional import require
from aisteer360.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)

_PLUGIN_INTERVENTION_KINDS = InterventionKinds(
    transforms=frozenset({"additive", "directional_ablation", "rotation", "head_additive"}),
    modifiers=frozenset({"norm_preserving", "alignment_adaptive"}),
    scopes=frozenset({"all", "after_prompt", "last_k", "from_position"}),
    gates=frozenset({"null", "cache_once", "probe_sum", "multi_key_threshold"}),
    constraints={"head_additive": "tensor_parallel_size==1"},
)

_PLUGIN_PROCESSOR_KINDS = ProcessorKinds(processors=frozenset({"constraint"}))

_PLUGIN_CAPTURE_KINDS = CaptureKinds(
    kinds=frozenset({"residual"}),
    locations=frozenset({"layer_output", "layer_input"}),
    modes=frozenset({"all_tokens", "last_token"}),
)

VLLM_BASELINE_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({Capability.SERVE_CHECKPOINT, Capability.SERVE_LORA}),
)

_DISCOVERY_CACHE: dict[str, dict] = {}

_DEFAULT_REQUEST_TIMEOUT = 120.0
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_MAX_ATTEMPTS = 3


def _vllm_capabilities(spec: BackendSpec, *, offline: bool) -> BackendCapabilities:
    """Capabilities implied by a vLLM spec: the plugin-free baseline, extended when the spec
    declares the vLLM-Hook plugin active. Hidden capture is advertised on the offline engine
    only, since serve-mode capture needs a bulk-tensor return path.

    Once a backend for the spec has fetched discovery, the advertised kind sets are the
    intersection of the static tables and the discovery payload, so a server missing a kind
    stops advertising it."""
    if not spec.get_option("hook_plugin"):
        return VLLM_BASELINE_CAPABILITIES
    atoms = VLLM_BASELINE_CAPABILITIES.atoms | {
        Capability.INTERVENTION_SPECS,
        Capability.PER_STEP_LOGIT_SPECS,
    }
    capture_kinds = None
    if offline:
        atoms = atoms | {Capability.HIDDEN_CAPTURE}
        capture_kinds = _PLUGIN_CAPTURE_KINDS
    capabilities = BackendCapabilities(
        atoms=frozenset(atoms),
        intervention_kinds=_PLUGIN_INTERVENTION_KINDS,
        processor_kinds=_PLUGIN_PROCESSOR_KINDS,
        capture_kinds=capture_kinds,
    )
    payload = _DISCOVERY_CACHE.get(spec.spec_hash)
    if payload is not None:
        capabilities = _intersect_with_discovery(capabilities, payload)
    return capabilities


def _intersect_with_discovery(capabilities: BackendCapabilities, payload: dict) -> BackendCapabilities:
    """The static capability tables narrowed to what the discovery payload confirms."""
    remote_interventions = payload.get("intervention_kinds") or {}
    intervention_kinds = capabilities.intervention_kinds
    if intervention_kinds is not None:
        intervention_kinds = InterventionKinds(
            transforms=intervention_kinds.transforms & frozenset(remote_interventions.get("transforms", ())),
            modifiers=intervention_kinds.modifiers & frozenset(remote_interventions.get("modifiers", ())),
            scopes=intervention_kinds.scopes & frozenset(remote_interventions.get("scopes", ())),
            gates=intervention_kinds.gates & frozenset(remote_interventions.get("gates", ())),
            constraints=dict(remote_interventions.get("constraints", {}) or intervention_kinds.constraints),
        )
    remote_processors = payload.get("processor_kinds") or {}
    processor_kinds = capabilities.processor_kinds
    if processor_kinds is not None:
        processor_kinds = ProcessorKinds(
            processors=processor_kinds.processors & frozenset(remote_processors.get("processors", ())),
        )
    remote_capture = payload.get("capture_kinds") or {}
    capture_kinds = capabilities.capture_kinds
    if capture_kinds is not None:
        capture_kinds = CaptureKinds(
            kinds=capture_kinds.kinds & frozenset(remote_capture.get("kinds", ())),
            locations=capture_kinds.locations & frozenset(remote_capture.get("locations", ())),
            modes=capture_kinds.modes & frozenset(remote_capture.get("modes", ())),
        )
    return BackendCapabilities(
        atoms=capabilities.atoms,
        intervention_kinds=intervention_kinds,
        processor_kinds=processor_kinds,
        capture_kinds=capture_kinds,
    )


def render_vllm_sampling_args(params: GenerationParams) -> dict[str, Any]:
    """Render normalized generation parameters onto vLLM sampling-parameter names.

    The table is exhaustive on this arm. Every normalized field maps to its vLLM name
    (`max_new_tokens` to `max_tokens`, `min_new_tokens` to `min_tokens`, `greedy=True` to
    `temperature=0.0`, `n` to `n`, stop strings to `stop` with
    `include_stop_str_in_output=True`, extra stop ids to `stop_token_ids`), and any key left in
    `extra` raises rather than being dropped. `seed` is not rendered here; sessions derive and
    attach per-item seeds.

    Args:
        params: The normalized parameters.

    Returns:
        Keyword arguments for `vllm.SamplingParams` (also valid as vLLM completions-request
        fields).

    Raises:
        ValueError: If `params.extra` is non-empty; the message names the unmapped keys.
        ValueError: If `params.greedy` is True while a non-zero `temperature` is also set.
    """
    if params.extra:
        raise ValueError(
            f"Generation parameter(s) {sorted(params.extra)} have no vLLM rendering; the vLLM "
            "table is exhaustive and unmapped parameters are rejected rather than dropped."
        )
    args: dict[str, Any] = {}
    if params.max_new_tokens is not None:
        args["max_tokens"] = params.max_new_tokens
    if params.min_new_tokens is not None:
        args["min_tokens"] = params.min_new_tokens
    if params.temperature is not None:
        args["temperature"] = params.temperature
    if params.top_p is not None:
        args["top_p"] = params.top_p
    if params.top_k is not None:
        args["top_k"] = params.top_k
    if params.repetition_penalty is not None:
        args["repetition_penalty"] = params.repetition_penalty
    if params.n is not None:
        args["n"] = params.n
    if params.greedy is True:
        if params.temperature not in (None, 0.0):
            raise ValueError(
                "greedy decoding conflicts with a non-zero temperature; drop one of the two."
            )
        args["temperature"] = 0.0
    if params.stop_strings:
        args["stop"] = list(params.stop_strings)
        args["include_stop_str_in_output"] = True
    if params.stop_token_ids:
        args["stop_token_ids"] = list(params.stop_token_ids)
    return args


def map_vllm_finish_reason(finish_reason: str | None, stop_reason: Any) -> str | None:
    """Map a vLLM candidate's finish reason onto the toolkit vocabulary.

    vLLM reports `"stop"` for EOS, stop strings, and stop token ids alike, with `stop_reason`
    None for EOS and the matched string or token id otherwise; `"length"` maps through
    unchanged, and anything else (e.g. `"abort"`) maps to None.

    Args:
        finish_reason: The vLLM candidate's finish reason.
        stop_reason: The vLLM candidate's stop reason.

    Returns:
        One of `"stop"`, `"eos"`, `"length"`, or None.
    """
    if finish_reason == "stop":
        return "eos" if stop_reason is None else "stop"
    if finish_reason == "length":
        return "length"
    return None


def extract_ref_logprobs(prompt_logprobs: Sequence | None, ref_ids: Sequence[int]) -> list[float]:
    """Pull the reference tokens' log-probabilities from a prompt-logprobs structure.

    Accepts both the offline shape (per-position mappings from token id to an object with a
    `logprob` attribute) and the serve JSON shape (string token-id keys mapping to dicts with a
    `"logprob"` entry). The reference occupies the last `len(ref_ids)` prompt positions.

    Args:
        prompt_logprobs: The per-prompt-position logprob entries, aligned with the submitted
            prompt tokens (position 0 is None).
        ref_ids: The reference token ids.

    Returns:
        One log-probability per reference token.

    Raises:
        ValueError: If the structure is missing or a reference position lacks its token's entry.
    """
    if prompt_logprobs is None:
        raise ValueError(
            "The response carries no prompt_logprobs; scoring requires prompt_logprobs=0 support."
        )
    if len(prompt_logprobs) < len(ref_ids):
        raise ValueError(
            f"prompt_logprobs has {len(prompt_logprobs)} positions for {len(ref_ids)} reference tokens."
        )
    values: list[float] = []
    offset = len(prompt_logprobs) - len(ref_ids)
    for position, token_id in enumerate(ref_ids):
        entry = prompt_logprobs[offset + position]
        if entry is None:
            raise ValueError(f"No logprob entry at reference position {position}.")
        record = entry.get(token_id, entry.get(str(token_id))) if hasattr(entry, "get") else None
        if record is None:
            raise ValueError(f"Token {token_id} missing from the logprob entry at position {position}.")
        if hasattr(record, "logprob"):
            values.append(float(record.logprob))
        elif isinstance(record, dict):
            values.append(float(record["logprob"]))
        else:
            values.append(float(record))
    return values


def _reject_entries(items: Sequence[GenerationItem | ScoringItem], backend_name: str) -> None:
    """Refuse control entries a request session cannot execute, naming the gap."""
    for item in items:
        for entry in (*item.state_entries, *item.output_entries):
            if isinstance(entry, HookEntry):
                raise UnsupportedOperationError(
                    f"HookEntry requires in-process torch hooks; the {backend_name} session "
                    "executes no client-side hooks. Run this pipeline on the huggingface backend."
                )
            if isinstance(entry, StackEntry):
                if entry.logits_processors or entry.stopping_criteria:
                    raise UnsupportedOperationError(
                        f"StackEntry carries live processor or criteria objects, which the "
                        f"{backend_name} session cannot execute; run this pipeline on the "
                        "huggingface backend."
                    )
            elif isinstance(entry, InterventionEntry):
                raise NotImplementedError(
                    "InterventionEntry lowering to the vLLM-Hook worker is not implemented in "
                    "this toolkit version."
                )
            elif isinstance(entry, ProcessorSpecEntry):
                raise NotImplementedError(
                    "ProcessorSpecEntry lowering is not implemented; the plugin serves no "
                    "processor kinds yet."
                )


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


def _config_layout(model_ref: str, trust_remote_code: bool = False) -> ModelLayout | None:
    """A client-side `ModelLayout` from the model config, or None when unresolvable.

    The fingerprint hashes the config JSON (volatile name/version fields removed), so it
    identifies the architecture and configuration rather than the weights.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    dtype = getattr(config, "torch_dtype", None)
    config_dict = {
        key: value for key, value in config.to_dict().items()
        if key not in ("_name_or_path", "transformers_version")
    }
    digest = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return ModelLayout(
        num_layers=getattr(config, "num_hidden_layers", 0),
        hidden_size=hidden_size or 0,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=str(dtype).removeprefix("torch.") if dtype is not None else "unknown",
        model_fingerprint=digest,
    )


def _client_tokenizer(source: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    return ensure_pad_token(tokenizer)


def _split_artifacts(artifacts: Sequence[Artifact]) -> tuple[CheckpointArtifact | None, LoRAArtifact | None]:
    checkpoint = next((a for a in artifacts if isinstance(a, CheckpointArtifact)), None)
    lora = next((a for a in artifacts if isinstance(a, LoRAArtifact)), None)
    return checkpoint, lora


def _reconcile_discovery(spec: BackendSpec, static: BackendCapabilities, payload: dict) -> None:
    """Warn when the discovery payload disagrees with the static advertisement.

    The static tables are the spec-implied advertisement; the discovery payload is the runtime
    authority. Kind-set gating consumes the intersection when spec lowering lands; at this
    phase a mismatch is surfaced as a warning.
    """
    discovered = payload.get("intervention_kinds", {})
    static_kinds = static.intervention_kinds
    if static_kinds is not None:
        for field_name, advertised in (
            ("transforms", static_kinds.transforms),
            ("modifiers", static_kinds.modifiers),
            ("scopes", static_kinds.scopes),
            ("gates", static_kinds.gates),
        ):
            remote = set(discovered.get(field_name, []))
            missing = advertised - remote
            if missing:
                logger.warning(
                    "vLLM-Hook discovery for spec %s lacks advertised %s %s; the intersection "
                    "governs spec execution.",
                    spec.spec_hash, field_name, sorted(missing),
                )


class VLLMBackend(Backend):
    """The offline vLLM engine backend.

    Boots one engine per backend instance from the spec (`engine_kwargs` option forwarded to
    `vllm.LLM`); requires the `vllm` optional dependency. A `CheckpointArtifact` overrides the
    served model reference and a `LoRAArtifact` attaches as a LoRA request on every generation.
    When the spec declares `hook_plugin`, the unified worker is selected via
    `VLLM_HOOK_WORKER=unified` and the discovery payload is fetched once and cached by spec
    hash. Capability advertisement is available through `capabilities_for_spec` without
    constructing the backend.
    """

    def __init__(self, spec: BackendSpec, artifacts: Sequence[Artifact] = ()) -> None:
        if spec.kind != "vllm":
            raise ValueError(f"VLLMBackend requires a 'vllm' spec; got kind {spec.kind!r}.")
        self.spec = spec
        require("vllm")
        import os

        from vllm import LLM

        checkpoint, lora = _split_artifacts(artifacts)
        model_ref = checkpoint.path if checkpoint is not None else spec.model
        if model_ref is None:
            raise ValueError("VLLMBackend needs a model reference on the spec or a checkpoint artifact.")
        trust_remote_code = bool(spec.get_option("trust_remote_code", default=False))
        _reject_encoder_decoder(model_ref, trust_remote_code)

        engine_kwargs = dict(spec.get_option("engine_kwargs", default={}) or {})
        if lora is not None:
            engine_kwargs.setdefault("enable_lora", True)
        if trust_remote_code:
            engine_kwargs.setdefault("trust_remote_code", True)

        # the worker-selection variable is scoped to this engine's boot so a later plugin-free
        # engine in the same process is unaffected
        previous_worker = os.environ.get("VLLM_HOOK_WORKER")
        if spec.get_option("hook_plugin"):
            os.environ["VLLM_HOOK_WORKER"] = "unified"
        try:
            self._llm = LLM(model=model_ref, **engine_kwargs)
        finally:
            if spec.get_option("hook_plugin"):
                if previous_worker is None:
                    os.environ.pop("VLLM_HOOK_WORKER", None)
                else:
                    os.environ["VLLM_HOOK_WORKER"] = previous_worker
        self._lora_request = None
        if lora is not None:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("steered", 1, lora.path)

        tokenizer_source = (
            spec.get_option("tokenizer_name_or_path")
            or model_ref
        )
        self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
        self._layout = _config_layout(model_ref, trust_remote_code)
        self._discovery: dict | None = None
        if spec.get_option("hook_plugin"):
            self._discovery = self._fetch_discovery()

    def _fetch_discovery(self) -> dict | None:
        cached = _DISCOVERY_CACHE.get(self.spec.spec_hash)
        if cached is not None:
            return cached
        payload = None
        for target in (self._llm, getattr(self._llm, "llm_engine", None)):
            rpc = getattr(target, "collective_rpc", None)
            if callable(rpc):
                try:
                    replies = rpc("hook_capabilities")
                except Exception as error:
                    logger.warning("vLLM-Hook discovery failed: %s", error)
                    return None
                payload = next((reply for reply in replies if reply), None)
                break
        if payload is None:
            logger.warning(
                "vLLM-Hook discovery returned no payload; is VLLM_HOOK_WORKER=unified active?"
            )
            return None
        _DISCOVERY_CACHE[self.spec.spec_hash] = payload
        _reconcile_discovery(self.spec, self.capabilities_for_spec(self.spec), payload)
        return payload

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The capability advertisement implied by `spec`."""
        return _vllm_capabilities(spec, offline=True)

    def open_session(self) -> "VLLMOfflineSession":
        """Open a request session over the shared engine."""
        return VLLMOfflineSession(self)


class _RequestSessionBase:
    """Lifecycle and layout shared by the vLLM request sessions."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self._closed = False
        self._generate_count = 0

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    def close(self) -> None:
        """Close the session; further use raises `RuntimeError`."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This session is closed; open a new session on the backend.")

    @property
    def tokenizer(self):
        """The backend's client-side tokenizer."""
        self._ensure_open()
        return self._backend.tokenizer

    @property
    def layout(self) -> ModelLayout:
        """Structural facts from the model config (client-side).

        Raises:
            RuntimeError: If the model config could not be resolved.
        """
        self._ensure_open()
        layout = self._backend._layout
        if layout is None:
            raise RuntimeError(
                "The model config could not be resolved client-side, so no layout is available."
            )
        return layout

    def _item_seed(self, item: GenerationItem, params: GenerationParams, index: int) -> int | None:
        if item.seed is not None:
            return item.seed
        if params.seed is not None:
            return derive_item_seed(params.seed, f"generate-{self._generate_count}", index)
        return None

    def _resolve_item_ids(self, item: GenerationItem | ScoringItem) -> list[int]:
        """The prompt's real token ids, with padding positions dropped per the attention mask,
        since a padded batch row would otherwise submit its pad tokens as prompt content."""
        resolved = item.prompt.resolve_token_ids(self.tokenizer)
        ids = resolved.token_ids[0]
        if resolved.attention_mask is not None:
            ids = ids[resolved.attention_mask[0].bool()]
        return ids.tolist()

    def _pack_output(
        self, index: int, prompt_ids: list[int], candidates: list[tuple[list[int], str | None]],
    ) -> ItemResult:
        """Build one `ItemResult` from per-candidate token ids and mapped finish reasons."""
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max((len(ids) for ids, _ in candidates), default=0)
        rows = torch.full((len(candidates), max_len), pad_token_id, dtype=torch.long)
        reasons: list[str | None] = []
        for row, (ids, reason) in enumerate(candidates):
            if ids:
                rows[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            reasons.append(reason)
        return ItemResult(
            index=index,
            output=Output(
                output_ids=rows,
                adapted_input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                finish_reason=reasons[0] if reasons else None,
                finish_reasons=tuple(reasons),
            ),
        )

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Hidden-state capture over the plugin is not implemented in this toolkit version."""
        raise UnsupportedOperationError(
            "Hidden-state capture on vLLM backends is not implemented in this toolkit version."
        )


class VLLMOfflineSession(_RequestSessionBase):
    """Request session over the offline engine.

    Token-id prompts submit as `TokensPrompt`s in one engine call with per-item sampling
    parameters; the engine schedules the batch internally, so no client-side fan-out is needed.
    """

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item through the engine.

        Args:
            items: The generation items; entries must be empty (no client-side hooks or live
                processors execute here).
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise.

        Returns:
            One `ItemResult` per item, in item order.
        """
        self._ensure_open()
        if not items:
            return []
        _reject_entries(items, "vllm")
        base_args = render_vllm_sampling_args(params)

        from vllm import SamplingParams, TokensPrompt

        prompts = []
        sampling = []
        prompt_ids_per_item: list[list[int]] = []
        for index, item in enumerate(items):
            ids = self._resolve_item_ids(item)
            prompt_ids_per_item.append(ids)
            args = dict(base_args)
            seed = self._item_seed(item, params, index)
            if seed is not None:
                args["seed"] = seed
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            sampling.append(SamplingParams(**args))
        self._generate_count += 1

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._backend._lora_request is not None:
            generate_kwargs["lora_request"] = self._backend._lora_request
        request_outputs = self._backend._llm.generate(prompts, sampling, **generate_kwargs)

        results: list[ItemResult] = []
        for index, request_output in enumerate(request_outputs):
            candidates = [
                (
                    list(candidate.token_ids),
                    map_vllm_finish_reason(
                        candidate.finish_reason, getattr(candidate, "stop_reason", None),
                    ),
                )
                for candidate in request_output.outputs
            ]
            results.append(self._pack_output(index, prompt_ids_per_item[index], candidates))
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        Each item's prompt and reference concatenate into one token-id prompt submitted with
        `prompt_logprobs=0`, and the reference positions' log-probabilities are read back.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys; forward keyword arguments have no remote
                rendering.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
        """
        self._ensure_open()
        if params.extra:
            raise ValueError(
                f"Scoring parameter(s) {sorted(params.extra)} have no vLLM rendering; remote "
                "scoring accepts no forward keyword arguments."
            )
        if not items:
            return torch.zeros((0, 0), dtype=torch.float32)
        _reject_entries(items, "vllm")
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")
        ref_len = ref_lens.pop()
        if ref_len == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32)

        from vllm import SamplingParams, TokensPrompt

        prompts = []
        ref_ids_per_item: list[list[int]] = []
        for item in items:
            prompt_ids = self._resolve_item_ids(item)
            ref_ids = item.ref_output_ids.reshape(-1).tolist()
            ref_ids_per_item.append(ref_ids)
            prompts.append(TokensPrompt(prompt_token_ids=[*prompt_ids, *ref_ids]))
        sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._backend._lora_request is not None:
            generate_kwargs["lora_request"] = self._backend._lora_request
        request_outputs = self._backend._llm.generate(prompts, sampling, **generate_kwargs)
        rows = [
            extract_ref_logprobs(request_output.prompt_logprobs, ref_ids)
            for request_output, ref_ids in zip(request_outputs, ref_ids_per_item)
        ]
        return torch.tensor(rows, dtype=torch.float32)


class VLLMServeBackend(Backend):
    """The vLLM OpenAI-compatible server backend.

    Targets a vLLM server rather than an arbitrary OpenAI-compatible endpoint: construction
    verifies the server's version surface (`GET /version`), fetches the plugin discovery payload
    (`GET /v1/hook/capabilities`) when the spec declares `hook_plugin`, and checks the served
    model id against the spec (or serves the pipeline's structural artifacts). Prompts submit as
    token ids on the completions endpoint with the token-id return option; the chat endpoint is
    not used. Requires no local vLLM installation.

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
            self._discovery = _DISCOVERY_CACHE.get(spec.spec_hash)
            if self._discovery is None:
                try:
                    self._discovery = self._get_json("/v1/hook/capabilities")
                except (TransportError, ValueError) as error:
                    raise ValueError(
                        f"The spec declares hook_plugin but {self._base_url} serves no "
                        f"/v1/hook/capabilities discovery surface: {error}"
                    ) from error
                _DISCOVERY_CACHE[spec.spec_hash] = self._discovery
                _reconcile_discovery(spec, self.capabilities_for_spec(spec), self._discovery)

        checkpoint, lora = _split_artifacts(artifacts)
        expected_model = checkpoint.path if checkpoint is not None else spec.model
        if lora is not None:
            self._served_model = self._load_lora_adapter(lora)
        else:
            served = self._served_model_ids()
            if expected_model is None:
                if len(served) != 1:
                    raise ValueError(
                        f"The spec names no model and the server serves {served}; set the "
                        "spec's model to disambiguate."
                    )
                self._served_model = served[0]
            elif expected_model in served:
                self._served_model = expected_model
            else:
                raise ValueError(
                    f"The server at {self._base_url} serves {served}, not the configured "
                    f"model {expected_model!r}."
                )

        tokenizer_source = (
            spec.get_option("tokenizer_name_or_path")
            or (checkpoint.path if checkpoint is not None else None)
            or (lora.base_model if lora is not None else None)
            or spec.model
        )
        if tokenizer_source is None:
            tokenizer_source = self._served_model
        self.tokenizer = _client_tokenizer(tokenizer_source, trust_remote_code)
        self._layout = _config_layout(
            (checkpoint.path if checkpoint is not None else None) or spec.model or self._served_model,
            trust_remote_code,
        )
        if self._discovery is not None:
            self._verify_fingerprints(tokenizer_source)

    def _served_model_ids(self) -> list[str]:
        payload = self._get_json("/v1/models")
        return [entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)]

    def _load_lora_adapter(self, lora: LoRAArtifact) -> str:
        served = self._served_model_ids()
        base = lora.base_model or self.spec.model
        if base and base not in served:
            raise ValueError(
                f"The server at {self._base_url} serves {served}, not the adapter's base "
                f"model {base!r}."
            )
        # the adapter name keys on path plus provenance, so a retrained adapter at the same
        # path loads as a new server-side adapter rather than reusing stale weights
        identity = f"{lora.path}:{lora.provenance.model_fingerprint or ''}"
        adapter_name = f"steered-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:8]}"
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

    def _verify_fingerprints(self, tokenizer_source: str) -> None:
        """Verify the client tokenizer against the discovery payload's fingerprint recipes.

        Uses the plugin's engine-free `core.fingerprints` when the `vllm_hook_plugins` package
        is installed; mismatches warn rather than raise. Without the package, verification is
        skipped with a warning.
        """
        model_block = (self._discovery or {}).get("model", {})
        remote_chat = model_block.get("chat_template_fingerprint")
        if remote_chat is None:
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

    def open_session(self) -> "VLLMServeSession":
        """Open a request session over the shared connection."""
        return VLLMServeSession(self)


class VLLMServeSession(_RequestSessionBase):
    """Request session over a vLLM server's completions endpoint.

    Items fan out concurrently under the backend's `max_concurrency`; transport failures retry
    with exponential backoff, and a batch whose items partially fail raises `PartialBatchError`
    carrying the successes and the re-issuable failures.
    """

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item through the completions endpoint.

        Args:
            items: The generation items; entries must be empty.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise.

        Returns:
            One `ItemResult` per item, in item order.

        Raises:
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if not items:
            return []
        _reject_entries(items, "vllm-serve")
        base_args = render_vllm_sampling_args(params)
        backend = self._backend

        item_ids = [self._resolve_item_ids(item) for item in items]
        seeds = [self._item_seed(item, params, index) for index, item in enumerate(items)]
        self._generate_count += 1

        def make_task(index: int):
            def task() -> ItemResult:
                body: dict[str, Any] = {
                    "model": backend._served_model,
                    "prompt": item_ids[index],
                    "return_token_ids": True,
                    **base_args,
                }
                if seeds[index] is not None:
                    body["seed"] = seeds[index]
                payload = with_transport_retries(
                    lambda: backend._post_json("/v1/completions", body),
                    max_attempts=backend.max_attempts,
                    backoff_base=backend.backoff_base,
                )
                choices = payload.get("choices", [])
                if not choices:
                    raise ValueError("The completions response carries no choices.")
                candidates = []
                for choice in choices:
                    token_ids = choice.get("token_ids")
                    if token_ids is None:
                        raise ValueError(
                            "The completions response carries no token_ids; the server does "
                            "not support the token-id return option (return_token_ids)."
                        )
                    candidates.append((
                        list(token_ids),
                        map_vllm_finish_reason(choice.get("finish_reason"), choice.get("stop_reason")),
                    ))
                return self._pack_output(index, item_ids[index], candidates)
            return task

        outcomes = run_bounded([make_task(i) for i in range(len(items))], backend.max_concurrency)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        results = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
        if failures:
            raise PartialBatchError(results, failures)
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens via prompt logprobs.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: Must carry no `extra` keys.

        Returns:
            Log probabilities of shape `[num_items, ref_len]` on CPU.

        Raises:
            ValueError: If items carry differing reference lengths or `params.extra` is
                non-empty.
            PartialBatchError: If some items failed after transport retries while others
                succeeded.
        """
        self._ensure_open()
        if params.extra:
            raise ValueError(
                f"Scoring parameter(s) {sorted(params.extra)} have no vLLM rendering; remote "
                "scoring accepts no forward keyword arguments."
            )
        if not items:
            return torch.zeros((0, 0), dtype=torch.float32)
        _reject_entries(items, "vllm-serve")
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")
        ref_len = ref_lens.pop()
        if ref_len == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32)
        backend = self._backend

        prompt_ids = [self._resolve_item_ids(item) for item in items]
        ref_ids = [item.ref_output_ids.reshape(-1).tolist() for item in items]

        def make_task(index: int):
            def task() -> list[float]:
                body = {
                    "model": backend._served_model,
                    "prompt": [*prompt_ids[index], *ref_ids[index]],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 0,
                }
                payload = with_transport_retries(
                    lambda: backend._post_json("/v1/completions", body),
                    max_attempts=backend.max_attempts,
                    backoff_base=backend.backoff_base,
                )
                choices = payload.get("choices", [])
                if not choices:
                    raise ValueError("The completions response carries no choices.")
                prompt_logprobs = choices[0].get("prompt_logprobs")
                return extract_ref_logprobs(prompt_logprobs, ref_ids[index])
            return task

        outcomes = run_bounded([make_task(i) for i in range(len(items))], backend.max_concurrency)
        failures = [(i, outcome) for i, outcome in enumerate(outcomes) if isinstance(outcome, Exception)]
        if failures:
            successes = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
            raise PartialBatchError(successes, failures)
        return torch.tensor(outcomes, dtype=torch.float32)
