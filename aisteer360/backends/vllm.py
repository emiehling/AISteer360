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
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from typing import Any, Literal

import torch

from aisteer360.algorithms.core.execution.payloads import (
    Artifact,
    CheckpointArtifact,
    LoRAArtifact,
)
from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.contracts import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    ConstraintKinds,
    InterventionKinds,
    ProcessorKinds,
)
from aisteer360.algorithms.core.execution.payloads import ConstraintSource
from aisteer360.algorithms.core.execution.fanout import (
    PartialBatchError,
    TransportError,
    derive_item_seed,
    run_bounded,
    with_transport_retries,
)
from aisteer360.algorithms.core.execution.payloads import (
    CaptureResult,
    ConstraintEntry,
    GenerationItem,
    HookEntry,
    InterventionEntry,
    ItemResult,
    ProcessorSpecEntry,
    ScoringItem,
    StackEntry,
)
from aisteer360.algorithms.core.execution.payloads import InterventionSpec
from aisteer360.algorithms.core.execution.payloads import ModelFacts
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.payloads import PreparedPrompt
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.execution.contracts import UnsupportedOperationError
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


_PLUGIN_CAPTURE_KINDS = CaptureKinds(
    kinds=frozenset({"residual"}),
    locations=frozenset({"layer_output", "layer_input"}),
    modes=frozenset({"all_tokens", "last_token"}),
)

_VLLM_CONSTRAINT_KINDS = ConstraintKinds(
    constraints=frozenset({"json_schema", "regex", "grammar", "choice"}),
)
VLLM_BASELINE_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({
        Capability.SERVE_CHECKPOINT,
        Capability.SERVE_LORA,
        Capability.GUIDED_DECODING,
    }),
    constraint_kinds=_VLLM_CONSTRAINT_KINDS,
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
    }
    capture_kinds = None
    if offline:
        atoms = atoms | {Capability.HIDDEN_CAPTURE}
        capture_kinds = _PLUGIN_CAPTURE_KINDS
    capabilities = BackendCapabilities(
        atoms=frozenset(atoms),
        intervention_kinds=_PLUGIN_INTERVENTION_KINDS,
        capture_kinds=capture_kinds,
        constraint_kinds=_VLLM_CONSTRAINT_KINDS,
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
        constraint_kinds=capabilities.constraint_kinds,
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


def _split_item_entries(
    items: Sequence[GenerationItem | ScoringItem],
    backend_name: str,
    *,
    plugin_active: bool,
    allow_constraints: bool = True,
) -> tuple[list[InterventionSpec | None], list[ConstraintSource | None]]:
    """Per-item intervention spec and constraint source after refusing unservable entries.

    `InterventionEntry` contributions are merged per item (ops concatenated in entry order,
    tensor payloads unioned); an item without spec entries yields None. A `ConstraintEntry`
    renders onto the engine's native structured-output parameters, one per item. Hook and
    live-processor entries name the in-process gap; intervention entries on a plugin-free
    backend name the `hook_plugin` fix.
    """
    specs: list[InterventionSpec | None] = []
    constraints: list[ConstraintSource | None] = []
    for item in items:
        item_specs: list[InterventionSpec] = []
        item_constraint: ConstraintSource | None = None
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
                if not plugin_active:
                    raise UnsupportedOperationError(
                        f"InterventionEntry requires the vLLM-Hook plugin; declare "
                        f"hook_plugin=True on the {backend_name} backend spec, or run this "
                        "pipeline on the huggingface backend."
                    )
                item_specs.append(entry.spec)
            elif isinstance(entry, ConstraintEntry):
                if not allow_constraints:
                    raise UnsupportedOperationError(
                        "Structured outputs do not apply to prompt logprobs; scoring with an "
                        "enabled constraint control requires the huggingface backend or "
                        "include_in_scoring=False."
                    )
                if item_constraint is not None:
                    raise UnsupportedOperationError(
                        "The engine hosts one structured-output constraint per request; compose "
                        "constraints into one source or run this pipeline on the huggingface "
                        "backend."
                    )
                item_constraint = entry.source
            elif isinstance(entry, ProcessorSpecEntry):
                raise UnsupportedOperationError(
                    f"ProcessorSpecEntry requires engine-hosted processor kinds, which the "
                    f"{backend_name} backend does not serve; run this pipeline on the "
                    "huggingface backend."
                )
        specs.append(merge_intervention_specs(item_specs) if item_specs else None)
        constraints.append(item_constraint)
    return specs, constraints


def render_guided_decoding_field(source: ConstraintSource) -> tuple[str, Any]:
    """The vLLM structured-output parameter name and payload for a constraint source."""
    if source.kind == "json_schema":
        value = source.value if isinstance(source.value, str) else dict(source.value)
        return "json", value
    if source.kind == "regex":
        return "regex", source.value
    if source.kind == "grammar":
        return "grammar", source.value
    return "choice", list(source.value)


def merge_intervention_specs(specs: Sequence[InterventionSpec]) -> InterventionSpec:
    """One spec carrying every op of `specs`, in order, with tensor payloads unioned."""
    if len(specs) == 1:
        return specs[0]
    ops: list = []
    artifacts: dict = {}
    for spec in specs:
        ops.extend(spec.ops)
        artifacts.update(spec.artifacts)
    return InterventionSpec(ops=tuple(ops), artifacts=artifacts)


def _load_safetensors_bytes(data: bytes) -> dict[str, torch.Tensor]:
    import safetensors.torch

    return safetensors.torch.load(data)


def remap_spec_for_scoring(spec: InterventionSpec, prompt_len: int) -> InterventionSpec:
    """A scoring copy of `spec` with `after_prompt` scopes rewritten to `from_position`.

    The teacher-forced reference is part of the server-side prompt, so the worker's "after the
    prompt" would select nothing; the rewrite anchors the scope at the original prompt length,
    the position of the first reference token in the submitted ids.
    """
    ops = []
    changed = False
    for op in spec.to_wire()["ops"]:
        if op.get("scope", {}).get("kind") == "after_prompt":
            op = {**op, "scope": {"kind": "from_position", "position": int(prompt_len)}}
            changed = True
        ops.append(op)
    if not changed:
        return spec
    return InterventionSpec(ops=tuple(ops), artifacts=spec.artifacts)


# spec-rejection codes that are support facts (a capability or constraint the backend lacks)
# rather than malformed payloads
_SUPPORT_FACT_CODES = ("E_UNKNOWN_KIND", "E_CONSTRAINT")
_SPEC_ERROR_RE = re.compile(r"\bE_[A-Z_]+ at \S+:")


def raise_for_spec_rejection(message: str) -> None:
    """Raise the toolkit error for a server-side spec rejection message carrying an `E_*` code.

    Kind and constraint gaps (`E_UNKNOWN_KIND`, `E_CONSTRAINT`) are support facts a stale
    client missed and raise `UnsupportedOperationError`; every other `E_*` rejection is a
    malformed spec and raises `ValueError`. The code and JSON path are preserved verbatim.
    A message without an `E_*` code returns without raising.
    """
    if not _SPEC_ERROR_RE.search(message):
        return
    if any(code in message for code in _SUPPORT_FACT_CODES):
        raise UnsupportedOperationError(message)
    raise ValueError(message)


def _refuse_by_engine_facts(discovery: dict | None, operation: str) -> None:
    """Refuse intervention or capture submission when discovery reports incompatible engine facts."""
    engine = (discovery or {}).get("engine", {})
    if engine.get("speculative_decoding"):
        raise UnsupportedOperationError(
            f"The serving engine runs speculative decoding, so {operation} requests are refused: "
            "draft-model forwards are unhooked and verification passes break the worker's "
            "position accounting. Disable speculative decoding on the engine."
        )
    if engine.get("enforce_eager") is False:
        raise UnsupportedOperationError(
            f"The serving engine compiles CUDA graphs, so {operation} requests are refused: "
            "worker hooks do not run under CUDA-graph replay. Start the engine with "
            "enforce_eager=True / --enforce-eager."
        )


def _refuse_by_constraints(
    specs: Sequence[InterventionSpec | None],
    discovery: dict | None,
    advertised: InterventionKinds | None,
) -> None:
    """Refuse specs whose kinds violate an advertised engine constraint, naming the fix.

    The only shipped constraint is `head_additive: tensor_parallel_size==1`; the check reads
    the constraint table from the negotiated kinds and the live value from discovery's engine
    facts, so the refusal matches what server-side staging would reject with `E_CONSTRAINT`.
    """
    constraints = dict(advertised.constraints) if advertised is not None else {}
    if not constraints or discovery is None:
        return
    tensor_parallel_size = (discovery.get("engine") or {}).get("tensor_parallel_size", 1)
    if tensor_parallel_size == 1:
        return
    for spec in specs:
        if spec is None:
            continue
        constrained = spec.required_kinds().transforms & set(constraints)
        if constrained:
            kind = sorted(constrained)[0]
            raise UnsupportedOperationError(
                f"Intervention kind {kind!r} requires {constraints[kind]}, but the serving engine "
                f"reports tensor_parallel_size={tensor_parallel_size}; serve the model with "
                "tensor_parallel_size=1 or run this pipeline on the huggingface backend."
            )


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
    """A client-side `ModelFacts` from the model config, or None when unresolvable.

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
    return ModelFacts(
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
        if spec.get_option("hook_plugin"):
            # worker hooks do not run under CUDA-graph replay; spec construction rejects an
            # explicit False, so this only fills the default
            engine_kwargs.setdefault("enforce_eager", True)

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
        self._plain_salt = uuid.uuid4().hex
        self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
        self._discovery: dict | None = None
        if spec.get_option("hook_plugin"):
            self._discovery = self._fetch_discovery()

    def stage_artifacts(self, payloads) -> None:
        """Write each content-addressed artifact into the plugin registry the engine reads.

        The offline engine shares the process's filesystem, so staging is a registry write
        (idempotent, verified against the content address).
        """
        self._artifact_uploader.upload_payloads(payloads)

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
    def layout(self) -> ModelFacts:
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

    def _prepare_spec_submission(
        self,
        items: Sequence[GenerationItem | ScoringItem],
        backend_name: str,
        allow_constraints: bool = True,
    ) -> tuple[list[InterventionSpec | None], list[ConstraintSource | None], list[str] | None]:
        """Per-item intervention specs, constraint sources, and cache salts for a batch.

        Spec-bearing items salt with the reference derivation over the spec and its artifact
        ids; spec-free items through a plugin-active backend salt with the backend's constant
        salt (structural KV isolation; the worker cannot police requests that carry no
        new-surface keys). Engine-fact refusals and constraint checks run before any artifact
        is written; artifact payloads are then materialized into the registry root the engine
        reads.
        """
        backend = self._backend
        plugin_active = bool(backend.spec.get_option("hook_plugin"))
        specs, constraints = _split_item_entries(
            items, backend_name, plugin_active=plugin_active, allow_constraints=allow_constraints,
        )
        if any(spec is not None for spec in specs):
            discovery = getattr(backend, "_discovery", None)
            _refuse_by_engine_facts(discovery, "intervention")
            _refuse_by_constraints(specs, discovery, backend.intervention_kinds)
            for spec in specs:
                if spec is not None:
                    backend._artifact_uploader.upload(spec)
        salts: list[str] | None = None
        if plugin_active:
            salts = [
                spec.salt() if spec is not None else backend._plain_salt for spec in specs
            ]
        return specs, constraints, salts

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

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Hidden-state capture over the plugin's capture surface.

        One request per prompt carries a `capture` spec and a fresh random `cache_salt`
        (a prefix-cache hit skips forward passes, so capture cannot tolerate reused salts) with
        `max_tokens=1`; the surplus decode position is truncated by the plugin. Per-layer
        tensors are stacked and right-padded to the batch's longest prompt.

        Args:
            prompts: The prompts to capture over.
            layers: 0-based decoder-layer indices to capture.
            mode: `"all_tokens"` for every prompt position, `"last_token"` for the final real
                position per row.
            location: The residual-stream boundary, `"layer_output"` or `"layer_input"`.

        Returns:
            The capture result: `[N, T, H]` per layer for `"all_tokens"` or `[N, H]` for
            `"last_token"`, on CPU in the engine's native dtype, with the derived `[N, T]`
            attention mask.

        Raises:
            UnsupportedOperationError: If the spec declares no `hook_plugin`, the negotiated
                capture kinds lack the requested mode or location, or the engine facts refuse
                capture (speculative decoding, non-eager execution).
            ValueError: If `prompts` is empty, a layer id is out of range, or the engine
                returned no capture payload.
        """
        self._ensure_open()
        backend = self._backend
        if not backend.spec.get_option("hook_plugin"):
            raise UnsupportedOperationError(
                "Hidden-state capture requires the vLLM-Hook plugin; declare hook_plugin=True "
                "on the vllm backend spec, or run capture on the huggingface backend."
            )
        capture_kinds = backend.capture_kinds
        required = CaptureKinds(
            kinds=frozenset({"residual"}),
            locations=frozenset({location}),
            modes=frozenset({mode}),
        )
        if capture_kinds is None or not capture_kinds.contains(required):
            raise UnsupportedOperationError(
                f"The serving backend does not advertise capture mode {mode!r} at location "
                f"{location!r}; update the server's vllm_hook_plugins or run capture on the "
                "huggingface backend."
            )
        _refuse_by_engine_facts(backend._discovery, "capture")
        if not prompts:
            raise ValueError("capture() requires at least one prompt.")
        num_layers = self.layout.num_layers
        missing = sorted(int(layer) for layer in layers if not 0 <= int(layer) < num_layers)
        if missing:
            raise ValueError(
                f"Requested layer ids {missing} are out of range; the model has {num_layers} layers."
            )

        from vllm import SamplingParams, TokensPrompt

        layer_ids = [int(layer) for layer in layers]
        capture_spec = {"layers": layer_ids, "mode": mode, "location": location}
        engine_prompts = []
        prompt_lens: list[int] = []
        for prompt in prompts:
            resolved = prompt.resolve_token_ids(self.tokenizer)
            ids = resolved.token_ids[0]
            if resolved.attention_mask is not None:
                ids = ids[resolved.attention_mask[0].bool()]
            ids = ids.tolist()
            prompt_lens.append(len(ids))
            engine_prompt = TokensPrompt(prompt_token_ids=ids)
            engine_prompt["cache_salt"] = uuid.uuid4().hex
            engine_prompts.append(engine_prompt)
        sampling = SamplingParams(max_tokens=1, temperature=0.0, extra_args={"capture": capture_spec})

        request_outputs = self._backend._llm.generate(engine_prompts, sampling, use_tqdm=False)

        rows_per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_ids}
        for index, request_output in enumerate(request_outputs):
            payload = getattr(request_output, "captures", None)
            if payload is None:
                raise ValueError(
                    "The engine returned no capture payload; is the vLLM-Hook unified worker "
                    "active on this engine?"
                )
            manifest_json, data = payload
            manifest = json.loads(manifest_json)
            tensors = _load_safetensors_bytes(data)
            for layer in layer_ids:
                stacked = tensors.get(f"layer_{layer}")
                if stacked is None or stacked.size(0) < prompt_lens[index]:
                    raise ValueError(
                        f"The capture payload covers layer {layer} at "
                        f"{0 if stacked is None else stacked.size(0)} of {prompt_lens[index]} "
                        f"prompt positions for prompt {index}; positions recorded: "
                        f"{manifest.get('positions', {}).get(str(layer))}."
                    )
                rows_per_layer[layer].append(stacked[: prompt_lens[index]])

        max_len = max(prompt_lens)
        attention_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
        for index, length in enumerate(prompt_lens):
            attention_mask[index, :length] = 1

        hidden: dict[int, torch.Tensor] = {}
        for layer, rows in rows_per_layer.items():
            if mode == "last_token":
                hidden[layer] = torch.stack([row[-1] for row in rows])
            else:
                padded = torch.zeros(len(rows), max_len, rows[0].size(-1), dtype=rows[0].dtype)
                for index, row in enumerate(rows):
                    padded[index, : row.size(0)] = row
                hidden[layer] = padded
        return CaptureResult(hidden=hidden, attention_mask=attention_mask, mode=mode, location=location)

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item through the engine.

        Args:
            items: The generation items; state entries lower as intervention specs on
                plugin-active backends, and no client-side hooks or live processors execute
                here.
            params: Normalized generation parameters shared by all items; unmapped `extra` keys
                raise.

        Returns:
            One `ItemResult` per item, in item order.
        """
        self._ensure_open()
        if not items:
            return []
        item_specs, item_constraints, item_salts = self._prepare_spec_submission(items, "vllm")
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
            if item_constraints[index] is not None:
                from vllm.sampling_params import GuidedDecodingParams

                field, value = render_guided_decoding_field(item_constraints[index])
                args["guided_decoding"] = GuidedDecodingParams(**{field: value})
            if item_specs[index] is not None:
                args["extra_args"] = {"intervention_spec": item_specs[index].to_wire()}
            prompt = TokensPrompt(prompt_token_ids=ids)
            if item_salts is not None:
                prompt["cache_salt"] = item_salts[index]
            prompts.append(prompt)
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
        item_specs, _, item_salts = self._prepare_spec_submission(
            items, "vllm", allow_constraints=False,
        )
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")
        ref_len = ref_lens.pop()
        if ref_len == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32)

        from vllm import SamplingParams, TokensPrompt

        prompts = []
        sampling = []
        ref_ids_per_item: list[list[int]] = []
        for index, item in enumerate(items):
            prompt_ids = self._resolve_item_ids(item)
            ref_ids = item.ref_output_ids.reshape(-1).tolist()
            ref_ids_per_item.append(ref_ids)
            prompt = TokensPrompt(prompt_token_ids=[*prompt_ids, *ref_ids])
            args: dict[str, Any] = {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
            if item_specs[index] is not None:
                scoring_spec = remap_spec_for_scoring(item_specs[index], len(prompt_ids))
                args["extra_args"] = {"intervention_spec": scoring_spec.to_wire()}
                if item_salts is not None:
                    item_salts[index] = scoring_spec.salt()
            if item_salts is not None:
                prompt["cache_salt"] = item_salts[index]
            prompts.append(prompt)
            sampling.append(SamplingParams(**args))

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
        self._plain_salt = uuid.uuid4().hex
        # spec artifacts write to the registry root the server reads; the shared_fs transport
        # assumes this filesystem is shared with the server
        self._artifact_uploader = _ArtifactUploader(spec.get_option("artifact_dir"))
        if self._discovery is not None:
            self._verify_fingerprints(tokenizer_source)

    def _served_model_ids(self) -> list[str]:
        payload = self._get_json("/v1/models")
        return [entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)]

    def stage_artifacts(self, payloads) -> None:
        """Make each content-addressed artifact available to the serving engine.

        With an `artifact_dir` option the payloads are written into that registry root (a
        filesystem shared with the server). Otherwise each payload is PUT to the plugin's
        artifact route (`/v1/hook/artifacts/{id}`, body safetensors bytes, id verified
        server-side); already-exists is success.
        """
        if not payloads:
            return
        if self.spec.get_option("artifact_dir"):
            self._artifact_uploader.upload_payloads(payloads)
            return
        import safetensors.torch

        for artifact_id, tensors in payloads.items():
            if artifact_id in self._artifact_uploader._written:
                continue
            data = safetensors.torch.save({name: tensors[name] for name in sorted(tensors)})
            self._put_bytes(f"/v1/hook/artifacts/{artifact_id}", data)
            self._artifact_uploader._written.add(artifact_id)

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
        item_specs, item_constraints, item_salts = self._prepare_spec_submission(items, "vllm-serve")
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
                if item_constraints[index] is not None:
                    field, value = render_guided_decoding_field(item_constraints[index])
                    body[f"guided_{field}"] = value
                if item_specs[index] is not None:
                    # vllm_xargs is scalar-only, so nested specs travel as JSON strings
                    body["vllm_xargs"] = {
                        "intervention_spec": item_specs[index].canonical(),
                    }
                if item_salts is not None:
                    body["cache_salt"] = item_salts[index]
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
        item_specs, _, item_salts = self._prepare_spec_submission(
            items, "vllm-serve", allow_constraints=False,
        )
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
                if item_specs[index] is not None:
                    scoring_spec = remap_spec_for_scoring(item_specs[index], len(prompt_ids[index]))
                    body["vllm_xargs"] = {"intervention_spec": scoring_spec.canonical()}
                    body["cache_salt"] = scoring_spec.salt()
                elif item_salts is not None:
                    body["cache_salt"] = item_salts[index]
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
