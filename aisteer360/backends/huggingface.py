"""The in-process Hugging Face backend and its exclusive session."""
import contextlib
from collections.abc import Callable, Sequence
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.capabilities import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
)
from aisteer360.algorithms.core.execution.items import (
    CaptureResult,
    GenerationItem,
    HookEntry,
    ItemResult,
    ScoringItem,
    StackEntry,
)
from aisteer360.algorithms.core.execution.layout import ModelLayout
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.prompts import PreparedPrompt
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.execution.support import UnsupportedOperationError
from aisteer360.algorithms.core.output import Output, infer_finish_reasons
from aisteer360.algorithms.output_control.base import stack_generate_kwargs
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.utils.tokenization import (
    ensure_pad_token,
    infer_attention_mask_from_ids,
    to_left_pad,
)

HF_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({
        Capability.IN_PROCESS_TORCH,
        Capability.HIDDEN_CAPTURE,
        Capability.BEAM_PROPOSALS,
        Capability.WEIGHT_TRAINING,
        Capability.MODEL_ADOPTION,
    }),
    capture_kinds=CaptureKinds(
        kinds=frozenset({"residual"}),
        locations=frozenset({"layer_output", "layer_input"}),
        modes=frozenset({"all_tokens", "last_token"}),
    ),
)

_CAPTURE_BATCH_SIZE = 8


def render_hf_gen_kwargs(params: GenerationParams) -> dict:
    """Render normalized generation parameters onto `model.generate` keyword arguments.

    Keys in `params.extra` pass through untouched; a normalized field always takes precedence
    over a same-named extra key. `seed` is not rendered here, since the session applies it as a
    `fork_rng`-scoped `manual_seed` around the item's decode.

    Args:
        params: The normalized parameters.

    Returns:
        Keyword arguments for `model.generate`.
    """
    gen_kwargs = dict(params.extra)
    if params.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = params.max_new_tokens
    if params.min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = params.min_new_tokens
    if params.temperature is not None:
        gen_kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        gen_kwargs["top_p"] = params.top_p
    if params.top_k is not None:
        gen_kwargs["top_k"] = params.top_k
    if params.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = params.repetition_penalty
    if params.greedy is not None:
        gen_kwargs["do_sample"] = not params.greedy
    if params.n is not None:
        gen_kwargs["num_return_sequences"] = params.n
    return gen_kwargs


def register_hook_specs(model: PreTrainedModel, hooks) -> list:
    """Attach hook specifications to `model`, returning the removable handles.

    Pre and forward hooks register with `with_kwargs=True`; backward hooks register as full
    backward hooks. If registration fails partway, handles already attached are removed before
    re-raising.

    Args:
        model: The model to hook.
        hooks: Hook specifications keyed by phase (`"pre"`, `"forward"`, `"backward"`).

    Returns:
        The registered `RemovableHandle`s.
    """
    handles: list = []
    try:
        for phase in ("pre", "forward", "backward"):
            for spec in hooks.get(phase, []):
                module = model.get_submodule(spec["module"])
                if phase == "pre":
                    handle = module.register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
                elif phase == "forward":
                    handle = module.register_forward_hook(spec["hook_func"], with_kwargs=True)
                else:
                    handle = module.register_full_backward_hook(spec["hook_func"])
                handles.append(handle)
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


class HFBackend(Backend):
    """The in-process Hugging Face backend.

    Owns a loaded model and tokenizer, either loaded from a spec or adopted from a caller that
    already holds them. At most one session may be open per backend at a time, so the backend
    runs one generation at a time.
    """

    def __init__(
        self,
        spec: BackendSpec,
        *,
        model_provider: Callable[[], PreTrainedModel | None] | None = None,
        tokenizer_provider: Callable[[], object | None] | None = None,
    ) -> None:
        """Construct the backend, loading the model from `spec` unless providers are given.

        Loading reads the options `hf_model_kwargs`, `device_map`, `tokenizer_name_or_path`,
        and `trust_remote_code`. Option values must be plain data, since spec canonicalization
        renders live objects (e.g. a quantization config instance) as strings that
        `from_pretrained` cannot consume. A `device_map` key inside `hf_model_kwargs` is used
        when the spec carries no top-level `device_map` option.

        Args:
            spec: The backend spec.
            model_provider: Callable returning the adopted model; used with
                `tokenizer_provider` instead of loading.
            tokenizer_provider: Callable returning the adopted tokenizer.

        Raises:
            ValueError: If `spec.kind` is not `"huggingface"`, or no model reference is
                available to load from.
        """
        if spec.kind != "huggingface":
            raise ValueError(f"HFBackend requires a 'huggingface' spec; got kind {spec.kind!r}.")
        self.spec = spec
        self._open_session: ExclusiveSession | None = None

        if model_provider is not None:
            self._model_provider = model_provider
            self._tokenizer_provider = tokenizer_provider or (lambda: None)
            return

        if spec.model is None:
            raise ValueError(
                "HFBackend needs a model reference on the spec, or model_provider/"
                "tokenizer_provider for an already-loaded model."
            )
        hf_model_kwargs = dict(spec.get_option("hf_model_kwargs", default={}))
        device_map = spec.get_option("device_map", default=hf_model_kwargs.pop("device_map", "auto"))
        model = AutoModelForCausalLM.from_pretrained(
            spec.model,
            device_map=device_map,
            **hf_model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.get_option("tokenizer_name_or_path") or spec.model,
            trust_remote_code=bool(spec.get_option("trust_remote_code", default=False)),
        )
        tokenizer = ensure_pad_token(tokenizer)
        self._model_provider = lambda: model
        self._tokenizer_provider = lambda: tokenizer

    @classmethod
    def adopt(
        cls,
        spec: BackendSpec,
        model_provider: Callable[[], PreTrainedModel | None],
        tokenizer_provider: Callable[[], object | None],
    ) -> "HFBackend":
        """Wrap an already-loaded model and tokenizer without loading anything.

        Providers are read on every access, so a caller whose model is replaced mid-steer (a
        structural control returning a new model) always exposes the current one to sessions.

        Args:
            spec: The backend spec identifying this configuration.
            model_provider: Callable returning the current model (may return None before one
                exists).
            tokenizer_provider: Callable returning the current tokenizer.

        Returns:
            The adopting backend.
        """
        return cls(spec, model_provider=model_provider, tokenizer_provider=tokenizer_provider)

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The static Hugging Face capability advertisement (spec-independent)."""
        return HF_CAPABILITIES

    def open_session(self) -> "ExclusiveSession":
        """Open the backend's one exclusive session.

        Returns:
            The session, usable as a context manager.

        Raises:
            RuntimeError: If an exclusive session is already open on this backend.
        """
        if self._open_session is not None and not self._open_session.closed:
            raise RuntimeError(
                "An exclusive session is already open on this backend; close it before opening "
                "another."
            )
        self._open_session = ExclusiveSession(self)
        return self._open_session


class ExclusiveSession:
    """The in-process session: direct model access, hook scopes, and the default decode loop.

    Exposes `.model` for components whose requirements include `Capability.IN_PROCESS_TORCH`.
    The default decode delegates to `model.generate`. Items execute serially, each under its own
    hook registrations, which preserves in-process semantics for every entry combination.
    """

    def __init__(self, backend: HFBackend) -> None:
        self._backend = backend
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    def close(self) -> None:
        """Close the session; further use raises `RuntimeError`."""
        self._closed = True

    def __enter__(self) -> "ExclusiveSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This session is closed; open a new session on the backend.")

    @property
    def model(self) -> PreTrainedModel:
        """The live model.

        Raises:
            RuntimeError: If the session is closed or no model is available yet.
        """
        self._ensure_open()
        model = self._backend._model_provider()
        if model is None:
            raise RuntimeError("No model is available on this session.")
        return model

    @property
    def tokenizer(self):
        """The tokenizer, or None when the adopting caller has not resolved one yet."""
        self._ensure_open()
        return self._backend._tokenizer_provider()

    @property
    def layout(self) -> ModelLayout:
        """Structural facts derived from the loaded model, computed on every access so weight
        edits and model replacements are always reflected.

        `num_layers` comes from the resolved decoder layer list, `hidden_size` and
        `num_attention_heads` from the model config, `head_dim` from the config with a
        `hidden_size // num_attention_heads` fallback, `dtype` from the model, and
        `model_fingerprint` from the weight/config fingerprint.
        """
        model = self.model

        from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint

        _, layer_names = get_model_layer_list(model)
        config = model.config
        hidden_size = config.hidden_size
        num_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and num_heads:
            head_dim = hidden_size // num_heads
        return ModelLayout(
            num_layers=len(layer_names),
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            dtype=str(model.dtype).removeprefix("torch."),
            model_fingerprint=model_fingerprint(model),
        )

    def _resolve_prompt_tensors(self, prompt: PreparedPrompt) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids and attention mask for one prompt, on the model device."""
        resolved = prompt.resolve_token_ids(self.tokenizer)
        device = self.model.device
        input_ids = resolved.token_ids.to(device)
        attention_mask = resolved.attention_mask
        if attention_mask is None:
            tokenizer = self.tokenizer
            if tokenizer is not None and tokenizer.pad_token_id is not None:
                attention_mask = infer_attention_mask_from_ids(input_ids, tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask = attention_mask.to(dtype=input_ids.dtype, device=device)
        return input_ids, attention_mask

    def _compose_entry_stacks(
        self, output_entries, extra_processors=(), extra_criteria=(),
    ) -> tuple[LogitsProcessorList, StoppingCriteriaList]:
        """Compose the items' stack entries, appending caller extras after entry contributions."""
        processors: list = []
        criteria: list = []
        for entry in output_entries:
            if not isinstance(entry, StackEntry):
                raise UnsupportedOperationError(
                    f"{type(entry).__name__} requires an engine-hosted processor path; the "
                    "in-process session consumes StackEntry contributions."
                )
            processors.extend(entry.logits_processors)
            criteria.extend(entry.stopping_criteria)
        processors.extend(extra_processors)
        criteria.extend(extra_criteria)
        return LogitsProcessorList(processors), StoppingCriteriaList(criteria)

    def _register_state_entries(self, model: PreTrainedModel, state_entries) -> list:
        handles: list = []
        try:
            for entry in state_entries:
                if not isinstance(entry, HookEntry):
                    raise UnsupportedOperationError(
                        f"{type(entry).__name__} requires an intervention-capable backend; the "
                        "in-process session consumes HookEntry contributions."
                    )
                handles.extend(register_hook_specs(model, entry.hooks))
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def _seeded(self, seed: int | None):
        """A context that snapshots and restores RNG state around a seeded decode.

        The CPU generator is always covered. On CUDA models every CUDA device generator is
        covered, since sharded models may sample on a device other than the first parameter's.
        On MPS models the MPS generator is covered.
        """
        if seed is None:
            return contextlib.nullcontext()
        device = self.model.device
        if device.type == "cuda":
            return torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
        if device.type == "mps":
            return _mps_rng_fork()
        return torch.random.fork_rng(devices=[])

    def _apply_seed(self, seed: int) -> None:
        """Seed the generators covered by `_seeded` for this decode."""
        torch.default_generator.manual_seed(seed)
        device = self.model.device
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)

    def generate(
        self,
        items: Sequence[GenerationItem],
        params: GenerationParams,
    ) -> list[ItemResult]:
        """Generate one result per item, serially, each under its own hook registrations.

        Every item's decode delegates to `model.generate` with the item's composed processor and
        criteria stacks; caller-supplied `logits_processor` and `stopping_criteria` entries in
        `params.extra` append after the item's own contributions. A seeded item decodes inside a
        seeded RNG fork, so seeded runs are reproducible and the covered generator state (CPU,
        plus the model device's) is restored afterwards.

        Args:
            items: The generation items.
            params: Normalized generation parameters shared by all items.

        Returns:
            One `ItemResult` per item, in item order.
        """
        self._ensure_open()
        model = self.model
        tokenizer = self.tokenizer
        gen_kwargs = render_hf_gen_kwargs(params)
        user_processors = tuple(gen_kwargs.pop("logits_processor", None) or ())
        user_criteria = tuple(gen_kwargs.pop("stopping_criteria", None) or ())

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        results: list[ItemResult] = []
        for index, item in enumerate(items):
            input_ids, attention_mask = self._resolve_prompt_tensors(item.prompt)
            processors, criteria = self._compose_entry_stacks(
                item.output_entries, extra_processors=user_processors, extra_criteria=user_criteria,
            )
            stacks = stack_generate_kwargs(processors, criteria)
            handles = self._register_state_entries(model, item.state_entries)
            try:
                seed = item.seed if item.seed is not None else params.seed
                with self._seeded(seed):
                    if seed is not None:
                        self._apply_seed(seed)
                    full_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **stacks,
                        **gen_kwargs,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            new_tokens = full_ids[:, input_ids.size(1):]
            reasons = infer_finish_reasons(
                new_tokens, gen_kwargs, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            )
            results.append(ItemResult(
                index=index,
                output=Output(
                    output_ids=new_tokens,
                    adapted_input_ids=input_ids,
                    finish_reason=reasons[0],
                ),
            ))
        return results

    def score(
        self,
        items: Sequence[ScoringItem],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of each item's reference tokens.

        For each item, the prompt left-packs (pad positions move before the real tokens) so the
        reference follows the prompt's last real token, then prompt and reference concatenate
        into one causal forward pass under the item's hook registrations; the item's logits
        processors replay position-by-position with the same `(prefix_ids, scores)` view they
        receive during generation. Stopping criteria never apply. Decoder-only models only; the
        pipeline's `compute_logprobs` serves encoder-decoder models in-process.

        Args:
            items: The scoring items. Every item must carry the same reference length.
            params: `params.extra` passes through as forward keyword arguments.

        Returns:
            Log probabilities of shape `[num_items, ref_len]`.

        Raises:
            ValueError: If items carry differing reference lengths.
            UnsupportedOperationError: If the model is an encoder-decoder model.
        """
        self._ensure_open()
        model = self.model
        if getattr(model.config, "is_encoder_decoder", False):
            raise UnsupportedOperationError(
                "Session scoring supports decoder-only models; encoder-decoder scoring runs "
                "through SteeringPipeline.compute_logprobs."
            )
        device = model.device
        forward_kwargs = dict(params.extra)

        if not items:
            return torch.zeros((0, 0), device=device, dtype=torch.float32)
        ref_lens = {item.ref_output_ids.shape[-1] for item in items}
        if len(ref_lens) > 1:
            raise ValueError(f"All scoring items must share one reference length; got {sorted(ref_lens)}.")

        all_logprobs: list[torch.Tensor] = []
        for item in items:
            input_ids, attention_mask = self._resolve_prompt_tensors(item.prompt)
            input_ids, attention_mask = to_left_pad(input_ids, attention_mask)
            ref_output_ids = item.ref_output_ids
            if ref_output_ids.ndim == 1:
                ref_output_ids = ref_output_ids.unsqueeze(0)
            ref_output_ids = ref_output_ids.to(device)
            ref_len = ref_output_ids.size(1)
            if ref_len == 0:
                all_logprobs.append(torch.zeros((1, 0), device=device, dtype=torch.float32))
                continue

            processors, _ = self._compose_entry_stacks(item.output_entries)
            handles = self._register_state_entries(model, item.state_entries)
            try:
                with torch.no_grad():
                    combined_ids = torch.cat([input_ids, ref_output_ids], dim=1)
                    combined_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, ref_len, device=device, dtype=attention_mask.dtype),
                    ], dim=1)
                    outputs = model(
                        input_ids=combined_ids,
                        attention_mask=combined_mask,
                        **forward_kwargs,
                    )
                    input_len = input_ids.size(1)
                    logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                    if len(processors):
                        for t in range(logits.size(1)):
                            prefix = torch.cat([input_ids, ref_output_ids[:, :t]], dim=1)
                            logits[:, t, :] = processors(prefix, logits[:, t, :])
                    logprobs = torch.log_softmax(logits, dim=-1)
                    all_logprobs.append(
                        logprobs.gather(dim=-1, index=ref_output_ids.unsqueeze(-1)).squeeze(-1)
                    )
            finally:
                for handle in handles:
                    handle.remove()

        return torch.cat(all_logprobs, dim=0)

    def capture(
        self,
        prompts: list[PreparedPrompt],
        layers: list[int],
        mode: Literal["all_tokens", "last_token"],
        location: Literal["layer_output", "layer_input"] = "layer_output",
    ) -> CaptureResult:
        """Capture residual-stream hidden states for `prompts` at `layers`.

        Prompts resolve to token ids, right-pad into one batch, and run through the shared
        layerwise extraction. In `"all_tokens"` mode each layer's tensor is `[N, T, H]`; in
        `"last_token"` mode the last real (non-pad) position of each row is selected, giving
        `[N, H]`.

        Args:
            prompts: The prompts to capture.
            layers: 0-based layer ids to keep.
            mode: `"all_tokens"` or `"last_token"`.
            location: `"layer_output"` (a layer's output boundary) or `"layer_input"` (the
                boundary a forward pre-hook observes).

        Returns:
            The captured tensors and the batch attention mask, on CPU.

        Raises:
            ValueError: If `mode` is unknown or a requested layer id is out of range.
        """
        self._ensure_open()
        if mode not in ("all_tokens", "last_token"):
            raise ValueError(f"Unknown capture mode {mode!r}; modes are 'all_tokens', 'last_token'.")
        if not prompts:
            raise ValueError("capture() requires at least one prompt.")

        from aisteer360.algorithms.core.internals.capture import (
            layerwise_tokenwise_hidden,
        )
        from aisteer360.algorithms.core.internals.pooling import (
            aggregate_condition_hidden,
        )

        model = self.model
        device = model.device
        tokenizer = self.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

        rows = [self._resolve_prompt_tensors(prompt) for prompt in prompts]
        max_len = max(ids.size(1) for ids, _ in rows)
        input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
        for row, (ids, mask) in enumerate(rows):
            length = ids.size(1)
            input_ids[row, :length] = ids[0]
            attention_mask[row, :length] = mask[0]

        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
        hidden = layerwise_tokenwise_hidden(
            model, enc, batch_size=_CAPTURE_BATCH_SIZE, location=location,
        )
        missing = [layer for layer in layers if layer not in hidden]
        if missing:
            raise ValueError(
                f"Requested layer ids {missing} are out of range; the model has {len(hidden)} layers."
            )

        mask_cpu = attention_mask.cpu()
        selected = {layer: hidden[layer] for layer in layers}
        if mode == "last_token":
            selected = {
                layer: aggregate_condition_hidden(tensor, "last", mask_cpu)
                for layer, tensor in selected.items()
            }
        return CaptureResult(
            hidden=selected, attention_mask=mask_cpu, mode=mode, location=location,
        )


@contextlib.contextmanager
def _mps_rng_fork():
    """Snapshot and restore the CPU and MPS generator states around a seeded decode."""
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        torch.mps.set_rng_state(mps_state)
