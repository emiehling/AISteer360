"""
Core steering pipeline for composing and applying multiple LLM control methods across backends.
"""
import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, overload

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.backends.base import Artifact, Backend, StateControlEntry
from aisteer360.backends.generation_params import GenerationParams
from aisteer360.backends.huggingface.backend import HuggingFaceBackend
from aisteer360.backends.specs import BackendSpec
from aisteer360.core.output import Output
from aisteer360.core.prompt import PreparedPrompt, Prompt
from aisteer360.core.requirements import Capability, ControlVerdict, ValidationReport
from aisteer360.core.utils.controls import merge_controls, warn_if_adapt_messages_bypassed
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl

logger = logging.getLogger(__name__)

# static fix hints for unmet capabilities, rendered into unsupported verdicts
_CAPABILITY_FIXES = {
    Capability.FORWARD_HOOKS: "run this pipeline on HuggingFaceBackend (in-process hooks)",
    Capability.ATTENTION_WRITE: "run this pipeline on HuggingFaceBackend (attention-mask steering is in-process only)",
    Capability.RAW_MODEL: "run this pipeline on HuggingFaceBackend (this control needs direct model access)",
    Capability.WEIGHT_TRAINING: "run steering on HuggingFaceBackend (weight training is in-process only)",
    Capability.STEPWISE_LOGITS: "run this pipeline on HuggingFaceBackend (stepwise-logits decoding is in-process only)",
    Capability.TOKEN_IDS: "attach a client-side tokenizer, or run on a backend that accepts token-array prompts",
    Capability.SCORING: "run on a backend that supports reference-continuation scoring",
    Capability.RESIDUAL_WRITE: "run on a backend that supports residual-stream steering",
    Capability.SERVER_GATING: "run on a backend that evaluates conditions server-side",
    Capability.HIDDEN_READ: "run on a backend that exposes hidden states",
}


def _render_fix(missing: Capability) -> str:
    """Compose an actionable fix string from the missing capabilities."""
    parts = [hint for capability, hint in _CAPABILITY_FIXES.items() if missing & capability]
    return "; ".join(dict.fromkeys(parts)) if parts else "run on a backend that grants the missing capabilities"


@dataclass(slots=True)
class SteeringPipeline:
    """Compose and apply structural, state, input, and output controls over a pluggable backend.

    Controls are prepared once (`steer()`) in a fixed order, then used together during generation.
    Inference runs against a `backend`; fitting/training may use a separate `steering_backend`.

    Workflow:

    1. Instantiate with control objects and an inference `backend` (a `Backend` or a `BackendSpec`).
    2. Call `steer()` once to prepare all controls in order (structural → input → state → output).
    3. Use `generate()` for inference (polymorphic across str / list[str] / chat / tensor), or
        `compute_logprobs()` for teacher-forced scoring.

    Args:
        controls (Sequence[StructuralControl | StateControl | InputControl | OutputControl], optional):
            Controls for the pipeline. The state category accepts any number of controls (applied in
            list order); the input, structural, and output categories accept at most one each.
            Omitted categories fall back to no-op controls.
        backend (Backend | BackendSpec, optional): The inference backend. Required by `steer()`.
            Model/tokenizer/device arguments live on the backend (e.g. `HuggingFaceBackend`).
        steering_backend (Backend | BackendSpec, optional): The backend used for fitting/training
            (structural controls, vector estimators). Defaults to the inference backend when it
            grants the steer-phase capabilities.

    Raises:
        RuntimeError: If `generate()`/`compute_logprobs()` is called before `steer()`, or no model
            is available after steering on an in-process backend.
        ValueError: If multiple controls are supplied for a single-instance category.

    State-control ordering contract:

    For the state category, list order in `controls` is the single composition surface: list order =
    `steer()` order = hook registration order = execution order for hooks on the same module.
    Non-commuting combinations (e.g. ablate∘add vs. add∘ablate) are order-sensitive by design; a
    gated/condition-scoring control placed after another observes activations already edited at
    earlier layers.
    """

    controls: Sequence[StructuralControl | StateControl | InputControl | OutputControl] = ()
    backend: Backend | BackendSpec | None = None
    steering_backend: Backend | BackendSpec | None = None

    structural_control: StructuralControl = field(init=False)
    input_control: InputControl = field(init=False)
    state_controls: list[StateControl] = field(init=False)
    output_control: OutputControl = field(init=False)

    _backend: Backend | None = field(default=None, init=False, repr=False)
    _steering_backend: Backend | None = field(default=None, init=False, repr=False)
    _is_steered: bool = field(default=False, init=False, repr=False)
    _warned_tensor_with_adapt_messages: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        controls_merged = merge_controls(self.controls)
        self.structural_control = controls_merged["structural_control"]
        self.input_control = controls_merged["input_control"]
        self.state_controls = controls_merged["state_controls"]
        self.output_control = controls_merged["output_control"]

        self._backend = self._resolve_backend(self.backend)
        self._steering_backend = None

        self._inject_tokenizer()

    @staticmethod
    def _resolve_backend(backend: Backend | BackendSpec | None) -> Backend | None:
        """Build a `BackendSpec` into a `Backend`, or pass a `Backend` through."""
        if backend is None:
            return None
        if isinstance(backend, BackendSpec):
            return backend.build()
        return backend

    def _inject_tokenizer(self) -> None:
        """Late-inject the inference backend's tokenizer into controls that accept one."""
        tokenizer = self.tokenizer
        if tokenizer is None:
            return
        for control in (self.structural_control, self.input_control, *self.state_controls, self.output_control):
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer") is None:
                setattr(control, "tokenizer", tokenizer)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """The inference backend's tokenizer, or `None`."""
        return getattr(self._backend, "tokenizer", None) if self._backend is not None else None

    @property
    def model(self) -> PreTrainedModel:
        """The inference backend's model (in-process backends only).

        Raises:
            RuntimeError: If no backend is set, or the backend exposes no model (API backends).
        """
        if self._backend is None:
            raise RuntimeError("No backend is set on this pipeline.")
        model = getattr(self._backend, "model", None)
        if model is None:
            raise RuntimeError(
                f"{type(self._backend).__name__} does not expose a model; `.model` is only available "
                "on in-process backends (HuggingFaceBackend)."
            )
        return model

    @property
    def device(self) -> torch.device | str | None:
        """The inference backend's device, or `None`."""
        return getattr(self._backend, "device", None) if self._backend is not None else None

    @property
    def supports_batching(self) -> bool:
        """True if all enabled controls in this pipeline are batch-safe."""
        controls = (self.structural_control, self.input_control, *self.state_controls, self.output_control)
        return all(
            getattr(control, "supports_batching", False)
            for control in controls
            if getattr(control, "enabled", True)
        )

    def _all_controls(self) -> tuple:
        """Every control in steer/execution order (structural → input → state → output)."""
        return (self.structural_control, self.input_control, *self.state_controls, self.output_control)

    def validate(self, backend: Backend | None = None) -> ValidationReport:
        """Return a per-control runnability report against the given (or inference) backend.

        For each enabled control, its `Requirements` are intersected with the backend's granted
        capabilities. Steer-phase requirements validate against the steering backend; generate-phase
        requirements against the inference backend.

        Args:
            backend: The inference backend to validate against; defaults to this pipeline's backend.

        Returns:
            A `ValidationReport` with one verdict per enabled control.
        """
        inference_backend = backend or self._backend
        steering_backend = self._steering_backend or inference_backend

        verdicts: list[ControlVerdict] = []
        for control in self._all_controls():
            if not getattr(control, "enabled", True):
                continue
            requirements = control.requires()
            if not requirements.capabilities:
                continue
            target = steering_backend if requirements.phase == "steer" else inference_backend
            verdicts.append(self._verdict_for(control, requirements, target))
        return ValidationReport(verdicts=verdicts)

    @staticmethod
    def _verdict_for(control, requirements, backend: Backend | None) -> ControlVerdict:
        """Build one control's verdict against a backend's capabilities."""
        name = type(control).__name__
        if backend is None:
            return ControlVerdict(
                control=name,
                status="unsupported",
                missing=requirements.capabilities,
                fix="attach a backend that grants the required capabilities",
            )
        granted = backend.capabilities.capabilities
        missing = requirements.capabilities & ~granted
        if missing:
            return ControlVerdict(
                control=name, status="unsupported", missing=missing, fix=_render_fix(missing)
            )
        notes = getattr(backend.capabilities, "notes", {})
        for note_key, note in notes.items():
            if isinstance(note_key, Capability) and (requirements.capabilities & note_key):
                return ControlVerdict(control=name, status="degraded", note=note)
        return ControlVerdict(control=name, status="supported")

    def steer(self, **steer_kwargs) -> "SteeringPipeline":
        """Prepare all controls in order (structural → input → state → output).

        Resolves the steering and inference backends, validates runnability, runs the structural
        control (adopting its model or artifact into the inference backend), then prepares the input,
        state, and output controls. Steering runs at most once.

        Args:
            **steer_kwargs: Extra keyword arguments forwarded to each control's `steer()`.

        Returns:
            This pipeline (steered).

        Raises:
            RuntimeError: If no backend is set, or no model is available after steering on an
                in-process backend.
        """
        if self._is_steered:
            return self
        if self._backend is None:
            raise RuntimeError("No inference backend set; pass `backend=` to construct the pipeline.")

        # resolve the steering backend: explicit > inference backend (when it can fit) > inference
        self._steering_backend = self._resolve_backend(self.steering_backend) or self._backend

        self.validate().raise_if_failed()

        steering_model = getattr(self._steering_backend, "model", None)
        steering_tokenizer = getattr(self._steering_backend, "tokenizer", None)

        # structural control may return a model or an Artifact
        if getattr(self.structural_control, "enabled", True):
            produced = self.structural_control.steer(
                model=steering_model, tokenizer=steering_tokenizer, backend=self._steering_backend, **steer_kwargs
            )
            self._adopt_structural_result(produced)
            steering_model = getattr(self._steering_backend, "model", None)

        # input / state / output controls prepare against the (steering) backend's model
        for control in (self.input_control, *self.state_controls, self.output_control):
            steer_fn = getattr(control, "steer", None)
            if not callable(steer_fn):
                continue
            maybe_model = steer_fn(
                steering_model, tokenizer=steering_tokenizer, backend=self._steering_backend, **steer_kwargs
            )
            if isinstance(maybe_model, nn.Module):
                self._adopt_model(maybe_model)

        # post-steer tokenizer resolution on the inference backend
        if isinstance(self._backend, HuggingFaceBackend):
            if self.model is None:
                raise RuntimeError(
                    "No model is available after steering. Provide a backend model or ensure a "
                    "StructuralControl returns one."
                )
            self._backend.resolve_tokenizer_fallback(
                structural_out_path=getattr(getattr(self.structural_control, "args", None), "out_path", None)
            )

        self._inject_tokenizer()
        self._is_steered = True
        return self

    def _adopt_structural_result(self, produced) -> None:
        """Adopt a structural control's product (a model or an `Artifact`) into the inference backend."""
        if produced is None:
            return
        if isinstance(produced, Artifact):
            self._backend.accept_artifact(produced)
            if self._steering_backend is not self._backend:
                self._steering_backend.accept_artifact(produced)
            return
        if isinstance(produced, nn.Module):
            self._adopt_model(produced)

    def _adopt_model(self, model: PreTrainedModel) -> None:
        """Adopt a model into the inference (and steering) backend when in-process."""
        for backend in {id(self._backend): self._backend, id(self._steering_backend): self._steering_backend}.values():
            if isinstance(backend, HuggingFaceBackend):
                backend.adopt_model(model)

    def _prepare_prompt(self, prompt: Prompt, runtime_kwargs: dict) -> PreparedPrompt:
        """Apply input-control adaptation and return a `PreparedPrompt`.

        Implements the adaptation-level dispatch: chat input with an `adapt_messages` override adapts
        at the message level (token-level `adapt` skipped); other inputs adapt at the token level
        after tokenization by the backend. The applied-exactly-once rule is preserved.
        """
        control = self.input_control
        enabled = getattr(control, "enabled", True)
        overrides_messages = type(control).adapt_messages is not InputControl.adapt_messages

        if prompt.modality == "chat":
            if enabled and overrides_messages:
                adapted = control.adapt_messages(prompt.messages, runtime_kwargs=runtime_kwargs)
                if adapted is not None:
                    return PreparedPrompt(prompt=prompt, adapted_messages=adapted, adaptation_level="messages")
            # chat with only token-level adapt: the backend tokenizes, then adapt() runs on ids
            if enabled and type(control).adapt is not InputControl.adapt:
                return self._token_level_adapt(prompt, runtime_kwargs)
            return PreparedPrompt(prompt=prompt, adaptation_level="none")

        # text / tensor
        self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
            control, self._warned_tensor_with_adapt_messages
        )
        if enabled and type(control).adapt is not InputControl.adapt:
            return self._token_level_adapt(prompt, runtime_kwargs)
        return PreparedPrompt(prompt=prompt, adaptation_level="none")

    def _token_level_adapt(self, prompt: Prompt, runtime_kwargs: dict) -> PreparedPrompt:
        """Tokenize (via the backend) then apply the input control's token-level `adapt`.

        The pipeline drops a length-mismatched mask at construction (the `PreparedPrompt`
        invariant); the backend re-infers when the mask is `None`.
        """
        input_ids, attention_mask = self._backend.resolve_prompt_tensors(
            PreparedPrompt(prompt=prompt, adaptation_level="none")
        )
        adapted_ids = self.input_control.adapt(input_ids, runtime_kwargs=runtime_kwargs)
        if isinstance(adapted_ids, list):
            adapted_ids = torch.tensor(adapted_ids, dtype=torch.long)
        if adapted_ids.ndim == 1:
            adapted_ids = adapted_ids.unsqueeze(0)
        adapted_ids = adapted_ids.to(input_ids.device)
        mask = attention_mask if attention_mask is not None and attention_mask.shape == adapted_ids.shape else None
        return PreparedPrompt(
            prompt=prompt,
            adapted_token_ids=adapted_ids,
            adapted_attention_mask=mask,
            adaptation_level="tokens",
        )

    def _build_entries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict,
        gen_kwargs: dict,
    ) -> list[StateControlEntry]:
        """Build one `StateControlEntry` per enabled state control, in pipeline list order.

        On in-process backends every control contributes ready `hooks` (declarative controls compile
        their plan through `get_hooks` → `compile_plan_to_hooks`). On server backends, declarative
        controls contribute the portable `plan` and hook-level controls remain unsupported (rejected
        at validation). Each control is reset before its entry is built.
        """
        in_process = getattr(self._backend, "executes_hooks_in_process", False)
        entries: list[StateControlEntry] = []
        for control in self.state_controls:
            if not getattr(control, "enabled", True):
                continue
            control.reset()
            plan = None
            hooks = None
            declarative = type(control).plan is not StateControl.plan
            if in_process or not declarative:
                hooks = control.get_hooks(input_ids, runtime_kwargs, attention_mask=attention_mask, **gen_kwargs)
            else:
                from aisteer360.algorithms.state_control._common.intervention import PromptContext

                pad_token_id = getattr(getattr(control, "tokenizer", None), "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = getattr(control, "_pad_token_id", None)
                prompt_ctx = PromptContext.from_ids(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
                plan = control.plan(prompt_ctx, runtime_kwargs)
            entries.append(StateControlEntry(control_name=type(control).__name__, plan=plan, hooks=hooks))
        return entries

    @overload
    def generate(self, inputs: str, attention_mask: torch.Tensor | None = ...,
                 runtime_kwargs: dict | None = ..., return_output: Literal[False] = ...,
                 **gen_kwargs: Any) -> str: ...
    @overload
    def generate(self, inputs: list[str], attention_mask: torch.Tensor | None = ...,
                 runtime_kwargs: dict | None = ..., return_output: Literal[False] = ...,
                 **gen_kwargs: Any) -> list[str]: ...
    @overload
    def generate(self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = ...,
                 runtime_kwargs: dict | None = ..., return_output: Literal[False] = ...,
                 **gen_kwargs: Any) -> torch.Tensor: ...
    @overload
    def generate(self, inputs: Any, attention_mask: torch.Tensor | None = ...,
                 runtime_kwargs: dict | None = ..., return_output: Literal[True] = ...,
                 **gen_kwargs: Any) -> Output | list[Output]: ...

    def generate(
        self,
        inputs: str | list[str] | list[dict] | list[list[dict]] | torch.Tensor | list[int] | list[list[int]] | None = None,
        attention_mask: torch.Tensor | None = None,
        runtime_kwargs: dict | None = None,
        return_output: bool = False,
        *,
        input_ids: Any = None,
        **gen_kwargs,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Polymorphic generation across text, chat, and tensor inputs.

        | Input type | Default return type |
        | --- | --- |
        | `str` | `str` |
        | `list[str]` | `list[str]` |
        | `list[dict]` (one chat) | `str` |
        | `list[list[dict]]` (batch of chats) | `list[str]` |
        | `torch.Tensor` (tokenized) | `torch.Tensor` (new tokens only) |

        With `return_output=True`, always returns `Output` (single) or `list[Output]` (batched).

        NOTE: returned token ids EXCLUDE the prompt by default; pass `return_full_sequence=True`
        for HF-style prompt+continuation ids. `attention_mask` is meaningful only for tensor inputs.

        Args:
            inputs: One of the modalities above.
            attention_mask: Optional, tensor-only.
            runtime_kwargs: Per-generation control parameters (e.g., `{"substrings": [...]}`).
            return_output: If True, return `Output` object(s) instead of decoded text / token ids.
            input_ids: Keyword alias for `inputs` (tokenized-prompt callers).
            **gen_kwargs: Generation parameters (HF vocabulary); may include `return_full_sequence`.

        Returns:
            See the dispatch table.

        Raises:
            RuntimeError: If `steer()` has not been called.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.generate()`.")

        if inputs is None and input_ids is not None:
            inputs = input_ids
        elif inputs is not None and input_ids is not None:
            raise TypeError("Pass either `inputs` or `input_ids`, not both.")
        if inputs is None:
            raise TypeError("`generate()` requires `inputs` (or the `input_ids` keyword).")

        runtime_kwargs = runtime_kwargs or {}
        params = GenerationParams.from_gen_kwargs(gen_kwargs)
        return_full_sequence = bool(gen_kwargs.get("return_full_sequence", False))

        prompt = Prompt.classify(inputs)
        if prompt.modality == "tensor":
            prompt.attention_mask = attention_mask  # meaningful only for tensor prompts
        elif attention_mask is not None:
            warnings.warn(
                f"`attention_mask` is ignored for {prompt.modality} input; it is rebuilt after tokenization.",
                UserWarning,
            )

        prepared = self._prepare_prompt(prompt, runtime_kwargs)
        steered_input_ids, steered_attention_mask = self._backend.resolve_prompt_tensors(prepared)

        # `return_full_sequence` is a pipeline-level flag, not an HF `generate` param
        hf_kwargs = params.to_hf_kwargs()
        hf_kwargs.pop("return_full_sequence", None)

        entries = self._build_entries(steered_input_ids, steered_attention_mask, runtime_kwargs, hf_kwargs)
        session = self._backend.open_session(entries, prepared, runtime_kwargs)

        with session:
            full_output_ids = self.output_control.generate(
                input_ids=steered_input_ids,
                attention_mask=steered_attention_mask,
                runtime_kwargs=runtime_kwargs,
                model=session.model,
                **hf_kwargs,
            )

        from aisteer360.core.utils.generation import infer_finish_reason

        prompt_len = steered_input_ids.size(1)
        new_tokens = full_output_ids[:, prompt_len:]
        returned_ids = full_output_ids if return_full_sequence else new_tokens
        finish_reason = infer_finish_reason(new_tokens, hf_kwargs)

        if return_output:
            return self._shape_outputs(returned_ids, steered_input_ids, runtime_kwargs, finish_reason, prompt.is_single)

        if prompt.modality == "tensor":
            return returned_ids

        decoded = self.tokenizer.batch_decode(returned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded[0] if prompt.is_single else decoded

    @staticmethod
    def _shape_outputs(returned_ids, adapted_input_ids, runtime_kwargs, finish_reason, is_single):
        """Build `Output` object(s) for the `return_output=True` path."""
        base = Output(
            output_ids=returned_ids,
            adapted_input_ids=adapted_input_ids,
            runtime_kwargs=runtime_kwargs or None,
            finish_reason=finish_reason,
            metadata={"backend": "HuggingFaceBackend"},
        )
        if is_single:
            return base
        return [
            Output(
                output_ids=base.output_ids[i:i + 1],
                adapted_input_ids=base.adapted_input_ids[i:i + 1] if base.adapted_input_ids is not None else None,
                runtime_kwargs=base.runtime_kwargs,
                finish_reason=base.finish_reason,
                metadata=base.metadata,
            )
            for i in range(base.output_ids.size(0))
        ]

    def compute_logprobs(
        self,
        input_ids: list[int] | torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        ref_output_ids: list[int] | torch.LongTensor = None,
        runtime_kwargs: dict | None = None,
        **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Per-token log-probabilities of `ref_output_ids` under structural, input, and state steering.

        Output controls are not applied (scoring, not generation). When all controls are batch-safe,
        a single batched forward is used (left-padded internally); otherwise a sequential fallback
        runs per item. Delegates the forward + logit-slice to the backend session's `score_tensors`.

        Args:
            input_ids: Prompt token ids `[seq_len]` or `[batch, seq_len]` (or a list).
            attention_mask: Optional matching attention mask.
            ref_output_ids: Reference tokens to score `[ref_len]` or `[batch, ref_len]`.
            runtime_kwargs: Per-call control parameters.
            **forward_kwargs: Extra kwargs forwarded to the model forward pass.

        Returns:
            A `[batch, ref_len]` tensor (decoder-only) or `[batch, ref_len - 1]` (encoder-decoder).

        Raises:
            RuntimeError: If `steer()` has not been called.
            ValueError: If `ref_output_ids` is None.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.compute_logprobs()`.")
        if ref_output_ids is None:
            raise ValueError("`ref_output_ids` is required for `compute_logprobs()`.")

        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)

        if self.supports_batching:
            prepared = self._prepare_prompt(
                Prompt.classify(input_ids, attention_mask=attention_mask), runtime_kwargs
            )
            steered_ids, steered_mask = self._backend.resolve_prompt_tensors(prepared)
            entries = self._build_entries(steered_ids, steered_mask, runtime_kwargs, forward_kwargs)
            session = self._backend.open_session(entries, prepared, runtime_kwargs)
            with session:
                return session.score_tensors(steered_ids, steered_mask, ref_output_ids, **forward_kwargs)

        # sequential fallback (one or more controls are not batch-safe)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device)

        num_inputs = input_ids.size(0)
        ref_len = ref_output_ids.size(1)
        if ref_output_ids.size(0) == 1 and num_inputs > 1:
            ref_output_ids = ref_output_ids.expand(num_inputs, -1)
        if ref_len == 0:
            return torch.zeros((num_inputs, 0), device=device, dtype=torch.float32)

        all_logprobs = []
        for i in range(num_inputs):
            single_mask = attention_mask[i:i + 1] if attention_mask is not None else None
            prepared = self._prepare_prompt(
                Prompt.classify(input_ids[i:i + 1], attention_mask=single_mask), runtime_kwargs
            )
            steered_ids, steered_mask = self._backend.resolve_prompt_tensors(prepared)
            entries = self._build_entries(steered_ids, steered_mask, runtime_kwargs, forward_kwargs)
            session = self._backend.open_session(entries, prepared, runtime_kwargs)
            with session:
                row = session.score_tensors(steered_ids, steered_mask, ref_output_ids[i:i + 1], **forward_kwargs)
            all_logprobs.append(row)
        return torch.cat(all_logprobs, dim=0)

    def cleanup(self) -> None:
        """Release control resources and close the steering backend."""
        for control in self._all_controls():
            cleanup = getattr(control, "cleanup", None)
            if callable(cleanup):
                with contextlib.suppress(Exception):
                    cleanup()
        if self._steering_backend is not None and self._steering_backend is not self._backend:
            self._steering_backend.close()

    @staticmethod
    def _classify_inputs(inputs: Any) -> tuple[Literal["text", "chat", "tensor"], bool, Any]:
        """Classify an input into `(modality, is_single, normalized)` (delegates to `Prompt.classify`)."""
        prompt = Prompt.classify(inputs)
        if prompt.modality == "chat":
            normalized = prompt.messages
        elif prompt.modality == "text":
            normalized = prompt.texts
        else:
            normalized = prompt.token_ids
        return prompt.modality, prompt.is_single, normalized
