"""ModelMemory: container for trained-model artifacts."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelMemory:
    """Memory wrapping a trained neural module.

    For methods whose memory artifact is a trained model rather than text or rules. Used by PRewrite at Phase 7; may
    be promoted to `common/memory/` if a second method adopts the shape.

    The model and tokenizer are runtime fields (not serialized as objects). `save` writes the HF model directory;
    `load` reconstructs the model and tokenizer via `AutoModelForCausalLM` / `AutoTokenizer`.

    Attributes:
        model_name_or_path: HF identifier or local path used at training time. Stored as metadata for round-trip
            identification (also used to locate the base model when only a LoRA adapter is saved).
        model: The instantiated `PreTrainedModel`. None until loaded or assigned.
        tokenizer: The associated tokenizer. None until loaded or assigned.
        extras: Free-form. Used by PRewrite for things like:

            - `"mode"`: `"per_query"` | `"static"`
            - `"use_peft"`: bool
            - `"peft_adapter_name"`: str (when LoRA)
            - `"training_config"`: dict of PPO hyperparams used
    """

    model_name_or_path: str
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    model_type: str = field(default="model", init=False)

    _EXTENSION = ".mmem"

    def save(self, path: str) -> None:
        """Save to a directory `<path>.mmem/`.

        Layout:

            - `model/` — HF `save_pretrained` (full model OR LoRA adapter, depending on `extras['use_peft']`).
            - `tokenizer/` — tokenizer `save_pretrained`.
            - `meta.json` — `model_type`, `model_name_or_path`, and `extras`.

        Args:
            path: Output path (directory). `.mmem` extension appended if not present.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ModelMemory.save requires both `model` and `tokenizer` to be set.")
        if not path.endswith(self._EXTENSION):
            path += self._EXTENSION
        os.makedirs(path, exist_ok=True)

        model_dir = os.path.join(path, "model")
        tokenizer_dir = os.path.join(path, "tokenizer")

        # for PEFT models, save_pretrained writes only the adapter
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)

        meta = {
            "model_type": self.model_type,
            "model_name_or_path": self.model_name_or_path,
            "extras": self._jsonable(self.extras),
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "ModelMemory":
        """Load a `ModelMemory` from a directory.

        If `extras['use_peft']` is True, the saved `model/` directory holds a LoRA adapter; the base model is loaded
        from `model_name_or_path` and the adapter is applied via `PeftModel.from_pretrained`.

        Args:
            path: Directory path. `.mmem` extension appended if not present.
            device: Optional device string for model placement.

        Returns:
            `ModelMemory` with `model` and `tokenizer` populated.

        Raises:
            ValueError: If the meta `model_type` does not match this class.
        """
        if not path.endswith(cls._EXTENSION):
            path += cls._EXTENSION

        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_type") != "model":
            raise ValueError(
                f"Cannot load ModelMemory: meta model_type is "
                f"{meta.get('model_type')!r}, expected 'model'."
            )

        model_name_or_path = meta.get("model_name_or_path")
        extras = meta.get("extras", {}) or {}
        use_peft = bool(extras.get("use_peft", False))

        model_dir = os.path.join(path, "model")
        tokenizer_dir = os.path.join(path, "tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        if use_peft:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = PeftModel.from_pretrained(base, model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

        if device is not None:
            model = model.to(device)

        return cls(
            model_name_or_path=model_name_or_path,
            model=model,
            tokenizer=tokenizer,
            extras=extras,
        )

    def cleanup(self) -> None:
        """Release model and tokenizer references; empty CUDA cache."""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        """Best-effort coercion of `extras` to JSON-safe primitives."""
        if isinstance(obj, dict):
            return {str(k): ModelMemory._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ModelMemory._jsonable(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)
