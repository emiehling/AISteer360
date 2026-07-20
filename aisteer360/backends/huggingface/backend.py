"""`HuggingFaceBackend`: the in-process Hugging Face execution backend.

This is code motion from `SteeringPipeline`, not a redesign. The backend owns model/tokenizer
loading, the tensor-normalization half of the old `_prepare_inputs` (2-D/device normalization,
pad-mask inference, duplicate-BOS warning), text/chat tokenization, and hands generation/scoring to
`HuggingFaceSession`. The input-control `adapt()` call stays pipeline-side and feeds `PreparedPrompt`
(doc 04); the pipeline drops a length-mismatched mask at construction and the backend infers a mask
whenever it is `None`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.backends.base import Backend, BackendCapabilities, StateControlEntry
from aisteer360.backends.huggingface.session import HuggingFaceSession
from aisteer360.backends.specs import BackendSpec
from aisteer360.core.prompt import PreparedPrompt
from aisteer360.core.requirements import Capability
from aisteer360.utils.tokenization import (
    ensure_pad_token,
    infer_attention_mask_from_ids,
    warn_if_duplicate_bos,
)

logger = logging.getLogger(__name__)

_HF_CAPABILITIES = (
    Capability.MESSAGES
    | Capability.TEXT
    | Capability.TOKEN_IDS
    | Capability.SCORING
    | Capability.RESIDUAL_WRITE
    | Capability.HIDDEN_READ
    | Capability.SERVER_GATING
    | Capability.ATTENTION_WRITE
    | Capability.FORWARD_HOOKS
    | Capability.RAW_MODEL
    | Capability.WEIGHT_TRAINING
    | Capability.STEPWISE_LOGITS
)


class HuggingFaceBackend(Backend):
    """In-process backend running a `PreTrainedModel` locally.

    Absorbs model/tokenizer loading, input normalization, and scoring from the former
    `SteeringPipeline`. Grants the full in-process capability set and accepts model / checkpoint /
    LoRA artifacts.

    Args:
        model_name_or_path: HuggingFace model hub name or local directory. Required unless
            `lazy_init=True` (a structural control will supply the model at steer time).
        tokenizer_name_or_path: Tokenizer location; defaults to `model_name_or_path`.
        device_map: Device map passed to `from_pretrained`. Defaults to `"auto"`. Cannot be combined
            with `device`.
        device: Device passed to the model's `.to()`. When set, `device_map` must remain `"auto"`.
        hf_model_kwargs: Extra kwargs forwarded to `from_pretrained`.
        trust_remote_code: Trust remote code when loading the tokenizer. Defaults to `False`. To
            trust remote code for the model, pass `trust_remote_code=True` via `hf_model_kwargs`.
        lazy_init: When `True`, defer loading the base model until it is supplied via
            `adopt_model` / `steer()`. Useful when a `StructuralControl` builds the final weights
            (e.g. MergeKit). Defaults to `False`.
    """

    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        *,
        tokenizer_name_or_path: str | None = None,
        device_map: str | dict[str, int] | int | torch.device | None = "auto",
        device: torch.device | str | None = None,
        hf_model_kwargs: dict | None = None,
        trust_remote_code: bool = False,
        lazy_init: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device_map = device_map
        self.device = device
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.trust_remote_code = trust_remote_code
        self.lazy_init = lazy_init

        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model_identity: str | None = str(model_name_or_path) if model_name_or_path is not None else None
        self.spec = BackendSpec(kind="huggingface", model=self.model_identity)

        # warn-once state relocated from the pipeline
        self._warned_duplicate_bos = False

        if not self.lazy_init:
            self._load()
        elif isinstance(self.tokenizer_name_or_path, (str, Path)):
            self.tokenizer = ensure_pad_token(
                AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=self.trust_remote_code)
            )

    @classmethod
    def from_spec(cls, spec: BackendSpec) -> "HuggingFaceBackend":
        """Build from a spec, mapping `spec.model` onto `model_name_or_path`."""
        backend = cls(model_name_or_path=spec.model, **dict(spec.kwargs))
        backend.spec = spec
        return backend

    def _load(self) -> None:
        """Load the model and tokenizer per the device/device_map contract."""
        if self.model_name_or_path is None:
            raise ValueError("`model_name_or_path` must be provided when lazy_init=False")
        if self.device is not None and self.device_map != "auto":
            raise ValueError("Cannot specify both `device` and `device_map`.")

        if self.device is not None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **self.hf_model_kwargs)
            self.model = self.model.to(self.device)
            self.device = self.model.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, device_map=self.device_map, **self.hf_model_kwargs
            )
            self.device = self.model.device

        self.tokenizer = ensure_pad_token(
            AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path or self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
        )

    def adopt_model(self, model: PreTrainedModel) -> None:
        """Adopt a model produced by a structural control at steer time.

        Does not resolve the tokenizer; the pipeline sets it explicitly or calls
        `resolve_tokenizer_fallback` once steering is complete.

        Args:
            model: The model to run inference against.
        """
        self.model = model
        self.device = model.device
        self.model_identity = getattr(model, "name_or_path", None) or self.model_identity

    def resolve_tokenizer_fallback(self, structural_out_path: str | None = None) -> None:
        """Resolve the tokenizer post-steer when it was not loaded up front.

        Args:
            structural_out_path: A fallback path (e.g. a structural control's output dir) to load
                the tokenizer from when the model carries no `name_or_path`.

        Raises:
            RuntimeError: If the tokenizer cannot be resolved.
        """
        if self.tokenizer is not None:
            return
        repo = getattr(self.model, "name_or_path", None) if self.model is not None else None
        try:
            self.tokenizer = ensure_pad_token(
                AutoTokenizer.from_pretrained(
                    repo or Path(structural_out_path or ""), trust_remote_code=self.trust_remote_code
                )
            )
        except Exception as exception:
            raise RuntimeError("Failed to resolve tokenizer post-steer.") from exception

    @property
    def capabilities(self) -> BackendCapabilities:
        """The full in-process capability set."""
        return BackendCapabilities(
            capabilities=_HF_CAPABILITIES,
            max_concurrency=1,
            accepts_artifacts=frozenset({"model", "checkpoint", "lora"}),
        )

    @property
    def executes_hooks_in_process(self) -> bool:
        """The HF backend realizes state controls as in-process forward hooks."""
        return True

    def prepare_tensor_inputs(
        self,
        input_ids: list[int] | torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize token ids/mask to 2-D device tensors, inferring the mask when absent.

        This is the normalization half of the former `SteeringPipeline._prepare_inputs`: 2-D shaping,
        device placement, pad-mask inference (pad-token-based, else all ones), and the once-per-backend
        duplicate-BOS warning. Input-control `adapt()` is applied pipeline-side before this call.

        Args:
            input_ids: Token ids, `[seq_len]` / `[batch, seq_len]` or a (nested) list.
            attention_mask: Optional mask matching `input_ids`, or `None`.

        Returns:
            A `(input_ids, attention_mask)` pair of 2-D tensors on the model device.
        """
        device = self.model.device

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
            if attention_mask.shape[-1] != input_ids.shape[-1]:
                attention_mask = None

        if attention_mask is None:
            if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
                attention_mask = infer_attention_mask_from_ids(input_ids, self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        attention_mask = attention_mask.to(dtype=input_ids.dtype, device=device)

        self._warned_duplicate_bos = warn_if_duplicate_bos(
            input_ids, attention_mask, self.tokenizer, self._warned_duplicate_bos
        )
        return input_ids, attention_mask

    def resolve_prompt_tensors(self, prepared: PreparedPrompt) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize/normalize a `PreparedPrompt` into `(input_ids, attention_mask)` on device.

        Tokenization is late: chat prompts run through `apply_chat_template(..., return_dict=True)`,
        text prompts through the tokenizer with padding, and adapted-token / tensor prompts through
        `prepare_tensor_inputs`.

        Args:
            prepared: The adapted prompt.

        Returns:
            A `(input_ids, attention_mask)` pair of 2-D device tensors.
        """
        level = prepared.adaptation_level

        if level == "tokens" and prepared.adapted_token_ids is not None:
            return self.prepare_tensor_inputs(prepared.adapted_token_ids, prepared.adapted_attention_mask)

        if level == "messages" and prepared.adapted_messages is not None:
            input_ids, attention_mask = self._tokenize_chat(prepared.adapted_messages)
            return self.prepare_tensor_inputs(input_ids, attention_mask)

        prompt = prepared.prompt
        if prompt.modality == "chat":
            input_ids, attention_mask = self._tokenize_chat(prompt.messages)
            return self.prepare_tensor_inputs(input_ids, attention_mask)
        if prompt.modality == "text":
            tokenized = self.tokenizer(list(prompt.texts), return_tensors="pt", padding=True)
            return self.prepare_tensor_inputs(tokenized["input_ids"], tokenized.get("attention_mask"))
        # tensor modality
        return self.prepare_tensor_inputs(prompt.token_ids, prompt.attention_mask)

    def _tokenize_chat(self, messages_batch: list[list[dict]]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the chat template with a generation prompt, returning ids and mask."""
        encoded = self.tokenizer.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        return input_ids, attention_mask

    def open_session(
        self,
        entries: list[StateControlEntry],
        prompt_ctx: PreparedPrompt,
        runtime_kwargs: dict,
    ) -> HuggingFaceSession:
        """Open an exclusive in-process session registering the entries' hooks.

        Args:
            entries: State-control contributions in pipeline list order.
            prompt_ctx: The adapted prompt (unused by the HF session; hooks were built against it
                upstream).
            runtime_kwargs: Per-call control parameters (unused here).

        Returns:
            A `HuggingFaceSession` bound to this backend's model.
        """
        if self.model is None:
            raise RuntimeError("HuggingFaceBackend has no model; load it or adopt one before opening a session.")
        return HuggingFaceSession(self, entries)

    def accept_artifact(self, artifact) -> None:
        """Load a model/checkpoint/LoRA artifact into the backend.

        Args:
            artifact: The structural-control artifact to deploy.

        Raises:
            ArtifactNotDeployable: If the artifact kind is unknown to this backend.
        """
        from aisteer360.backends.errors import ArtifactNotDeployable

        if artifact.kind in ("model", "checkpoint"):
            self.model = AutoModelForCausalLM.from_pretrained(
                artifact.ref, device_map=self.device_map, **self.hf_model_kwargs
            )
            self.device = self.model.device
            self.model_identity = artifact.ref
            self.resolve_tokenizer_fallback(structural_out_path=artifact.ref)
            return
        raise ArtifactNotDeployable(
            f"HuggingFaceBackend cannot deploy a {artifact.kind!r} artifact.", model=self.model_identity
        )

    def close(self) -> None:
        """Drop the model reference (frees GPU memory when the pipeline calls `cleanup`)."""
        self.model = None
