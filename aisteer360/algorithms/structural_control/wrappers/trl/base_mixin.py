import inspect
from dataclasses import fields, is_dataclass
from typing import Any

from peft import PeftType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from aisteer360.algorithms.core.execution.payloads import (
    Artifact,
    CheckpointArtifact,
    LoRAArtifact,
)
from aisteer360.algorithms.core.execution.contracts import Capability


class TRLMixin:
    """
    Small shared helpers for TRL-based structural controls.
    """

    # populated from Args by subclasses
    base_model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    hf_model_kwargs: dict[str, Any] = {}
    trust_remote_code: bool = False

    training_args: dict[str, Any] = {}
    output_dir: str | None = None
    resume_from_checkpoint: str | None = None

    use_peft: bool = False
    peft_type: Any = None
    lora_kwargs: dict[str, Any] = {}
    adapter_name: str | None = None

    merge_lora_after_train: bool = False
    merged_output_dir: str | None = None

    # resolved at runtime
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    device = None

    def _resolve_model_tokenizer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        if model is None:
            if not self.base_model_name_or_path:
                raise ValueError("TRLMixin: model is None and `base_model_name_or_path` was not provided.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **(self.hf_model_kwargs or {}),
            )
        else:
            self.model = model

        if tokenizer is None:
            path = (
                self.tokenizer_name_or_path
                or getattr(self.model, "name_or_path", None)
                or self.base_model_name_or_path
            )
            if not path:
                raise ValueError("TRLMixin: could not resolve tokenizer path.")
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=self.trust_remote_code)
        else:
            self.tokenizer = tokenizer

        self.device = next(self.model.parameters()).device
        return self.model, self.tokenizer

    @staticmethod
    def _filter_kwargs_for_class_or_callable(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only kwargs accepted by a dataclass or callable."""
        if is_dataclass(target):
            allowed = {f.name for f in fields(target)}
        else:
            try:
                allowed = set(inspect.signature(target).parameters.keys())
            except (TypeError, ValueError):
                allowed = set(kwargs.keys())
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    def _post_train_freeze(self) -> PreTrainedModel:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self.model

    def _maybe_save_trained_artifacts(self, trainer) -> None:
        output_dir = self.training_args.get("output_dir") or self.output_dir
        if output_dir:
            trainer.save_model(output_dir)
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass

    def _resolved_output_dir(self) -> str | None:
        return self.training_args.get("output_dir") or self.output_dir

    def artifact_capability(self) -> Capability | None:
        """The serve capability implied by this training configuration.

        Training runs only when `train_dataset` is set, so a configuration without one produces
        no artifact. A LoRA run without merge-back saves an adapter to the output directory
        (`Capability.SERVE_LORA`); a full fine-tune saves a checkpoint there
        (`Capability.SERVE_CHECKPOINT`); a merged LoRA run yields a checkpoint only when
        `merged_output_dir` is set.
        """
        if getattr(self, "train_dataset", None) is None:
            return None
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        if is_lora and not self.merge_lora_after_train:
            return Capability.SERVE_LORA if self._resolved_output_dir() else None
        if is_lora and self.merge_lora_after_train:
            return Capability.SERVE_CHECKPOINT if self.merged_output_dir else None
        return Capability.SERVE_CHECKPOINT if self._resolved_output_dir() else None

    def export_artifact(self) -> Artifact | None:
        """The on-disk product of this configuration's `steer()`, matching
        `artifact_capability()`."""
        capability = self.artifact_capability()
        if capability is None:
            return None
        if capability == Capability.SERVE_LORA:
            base = (
                self.base_model_name_or_path
                or getattr(self.model, "name_or_path", None)
                or ""
            )
            return LoRAArtifact(path=str(self._resolved_output_dir()), base_model=str(base))
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        path = self.merged_output_dir if (is_lora and self.merge_lora_after_train) else self._resolved_output_dir()
        return CheckpointArtifact(path=str(path))

    def _maybe_merge_lora_in_place(self) -> None:
        """Optionally merge LoRA into the base weights."""
        if not (self.use_peft and self.merge_lora_after_train):
            return

        # trainer often returns a PEFT-wrapped model; merge if possible
        if hasattr(self.model, "merge_and_unload"):
            merged_model = self.model.merge_and_unload()
            self.model = merged_model
            self.device = next(self.model.parameters()).device

            # save if requested
            if self.merged_output_dir:
                self.model.save_pretrained(self.merged_output_dir)
                try:
                    self.tokenizer.save_pretrained(self.merged_output_dir)
                except Exception:
                    pass
