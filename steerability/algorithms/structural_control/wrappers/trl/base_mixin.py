import difflib
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import trl
from peft import LoraConfig, PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.base_control import NotFreezableError
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import Artifact, CheckpointArtifact, LoRAArtifact
from steerability.algorithms.core.fits import product_is_stored, resolve_product, stored_product
from steerability.algorithms.core.internals import lora_target_pattern

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession


def materialized_dataset(*, dataset: Any) -> Any:
    """A dataset given as a `DataRef` loaded, or any other value unchanged.

    A Hub reference loads with `DataRef.load`, and a reference to a local file loads with the `datasets` reader of
    its suffix (`.json` and `.jsonl` as JSON, `.csv`, or `.parquet`) as one `train` split. A dataset given by reference
    can be rebuilt in another process, e.g., the child process of a time-limited training.

    Args:
        dataset: A dataset, a `DataRef`, or None.

    Returns:
        The dataset.

    Raises:
        ValueError: If a local file has another suffix.
    """
    from steerability.spipe.codec import DataRef

    if not isinstance(dataset, DataRef):
        return dataset
    loaded = dataset.load()
    if dataset.kind != "path":
        return loaded
    readers = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
    reader = readers.get(Path(loaded).suffix.lower())
    if reader is None:
        raise ValueError(
            f"A DataRef to a local file reads a JSON, JSON Lines, CSV, or Parquet file; got {loaded!r}.",
        )
    from datasets import load_dataset

    return load_dataset(reader, data_files=str(loaded), split="train")


def resolve_config_kwargs(config_cls: type, training_args: dict[str, Any]) -> dict[str, Any]:
    """The kwargs to construct `config_cls` from `training_args`.

    Every key must be a dataclass field of `config_cls`; `None` values are dropped so the config's
    own default applies. The toolkit forwards `training_args` verbatim to TRL, so a key the config
    does not declare would otherwise be passed to a constructor that does not accept it.

    Args:
        config_cls: A TRL config dataclass (`DPOConfig`, `SFTConfig`, `GRPOConfig`).
        training_args: The composed training arguments.

    Returns:
        The accepted kwargs, in `training_args` order, without `None` values.

    Raises:
        ValueError: If any key is not a field of `config_cls`. The message names the config class,
            the installed TRL version, and the unknown keys, with a closest-field suggestion when
            one is close.
    """
    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(key for key in training_args if key not in allowed)
    if unknown:
        details = []
        for key in unknown:
            close = difflib.get_close_matches(key, allowed, n=1)
            details.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
        raise ValueError(
            f"{config_cls.__name__} (trl {trl.__version__}) does not accept training_args key(s): "
            f"{', '.join(details)}. training_args is forwarded verbatim to TRL; remove these keys or "
            f"use a field the installed {config_cls.__name__} declares."
        )
    return {key: value for key, value in training_args.items() if value is not None}


class TRLMixin:
    """The shared steer step, artifacts, and freezing of the TRL-based structural controls.

    A configuration with a `train_dataset` trains, and its product (the LoRA adapter, or the checkpoint of a full
    fine-tune or a merged LoRA run) is its one fit. Under an active fit cache, `steer()` resolves the product through
    the cache (`resolve_product`). A stored product is installed as the frozen form does (a `load_lora` or
    `load_checkpoint` steer), and nothing trains. While the cache contains the product, the steer step declares
    `ModelAccess.FACTS` and no fit. After `steer()`, the control refers to the stored product, which `export_artifact()`
    and `export_state()` report as an artifact object. Without a cache, `steer()` trains every time and the product
    is the output directory. A subclass implements `_train(model, tokenizer, **kwargs)`, which trains (when a dataset
    is set) and returns the model.
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
    tokenizer: PreTrainedTokenizerBase | None = None
    device = None
    _resolved_base_ref: str | None = None
    _transient_output_dir: str | None = None
    _product: Artifact | None = None
    _installed: bool = False

    def _train(self, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase | None = None,
               **kwargs) -> PreTrainedModel:
        raise NotImplementedError

    def steer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session: "SteeringSession | None" = None,
        **kwargs: Any,
    ) -> PreTrainedModel | None:
        """Train the model, or install the stored product of an earlier training, and return the model.

        A configuration without a `train_dataset` trains nothing and returns the resolved model, frozen, and a merged
        LoRA run without `merged_output_dir` trains every time, since it leaves no product. Otherwise the product is
        resolved through the active fit cache (`resolve_product`). On a hit the stored product is installed
        on the incoming model as the frozen form does, and nothing trains. On a miss the model is trained, the product
        is stored, and the trained model is returned, and a temporary output directory that the control created is
        removed once the product is stored elsewhere. On a hit as on a miss, the control then refers to the stored
        product (`export_artifact`).

        Args:
            model: The incoming pipeline model, or None (the model is then loaded from `base_model_name_or_path`).
            tokenizer: The pipeline tokenizer, or None.
            session: The steering session. On a session that serves no in-process model (an engine backend), a
                stored product is not installed and the engine serves it.
            **kwargs: Further arguments of the training (e.g., `ref_model` for DPO).

        Returns:
            The trained or adapted model, or None on a session that serves no in-process model.
        """
        if self.fit_identity() is None or self.artifact_capability() is None:
            return self._train(model=model, tokenizer=tokenizer, session=session, **kwargs)
        trained: list[PreTrainedModel] = []

        def fit() -> dict[str, Any]:
            trained.append(self._train(model=model, tokenizer=tokenizer, session=session, **kwargs))
            return self.export_state()

        if self._product is None:
            self._product = self._artifact_of(state=resolve_product(control=self, fit=fit))
            self._remove_transient_output_dir()
        if trained:
            return trained[0]
        return self._install(model=model, tokenizer=tokenizer, session=session)

    def _artifact_of(self, *, state: Mapping[str, Any]) -> Artifact:
        """The product of a state, as this configuration's `artifact_capability()` decides.

        The state's `"artifact"` is an artifact object (a state the control exported) or a path (a state decoded from
        the fit cache), and a LoRA state gives its base model reference under `"base_model"`.
        """
        artifact = state["artifact"]
        path = str(getattr(artifact, "path", artifact))
        if self.artifact_capability() == Capability.SERVE_LORA:
            return LoRAArtifact(path=path, base_model=str(state["base_model"]))
        return CheckpointArtifact(path=path)

    def _remove_transient_output_dir(self) -> None:
        """Remove the temporary output directory once the product is stored in another place."""
        transient = self._transient_output_dir
        if transient is None or self._product is None or Path(self._product.path) == Path(transient):
            return
        shutil.rmtree(transient, ignore_errors=True)
        self._transient_output_dir = None

    def _install(self, *, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase | None,
                 session) -> PreTrainedModel | None:
        """Install the product on the incoming model with the loader of the frozen form."""
        from steerability.algorithms.core.registry import resolve_method_key

        product = self._product
        state = {"artifact": product.path}
        if isinstance(product, LoRAArtifact):
            state["base_model"] = product.base_model
            if model is None and getattr(session, "in_process", True):
                model, tokenizer = self._resolve_model_tokenizer(model=model, tokenizer=tokenizer)
        method, loader_args = self.frozen_form(state=state)
        loader = resolve_method_key(key=method).control_cls(**loader_args)
        self._installed = True
        return loader.steer(model=model, tokenizer=tokenizer, session=session)

    def restore(self, model: PreTrainedModel | None) -> PreTrainedModel | None:
        """Remove an installed adapter and return the incoming model, without merging the adapter.

        Args:
            model: The adapted pipeline model.

        Returns:
            The model without the adapter (PEFT's `unload()`), or None unless `steer()` installed a stored LoRA
            product (rather than training) on `model`, a PEFT model.
        """
        from peft import PeftModel

        if not self._installed or not isinstance(self._product, LoRAArtifact) or not isinstance(model, PeftModel):
            return None
        return model.unload()

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.FACTS` while the active fit cache contains this configuration's product, else `MODULE`.

        A stored product is installed rather than trained, which needs no staged model on an engine backend.
        """
        return ModelAccess.FACTS if product_is_stored(control=self) else ModelAccess.MODULE

    def _resolve_model_tokenizer(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Resolve the model and tokenizer, returning both as locals.

        Loads the model from `base_model_name_or_path` when `model` is None, and the tokenizer from
        `tokenizer_name_or_path`, the model's `name_or_path`, or `base_model_name_or_path` when
        `tokenizer` is None. Records the base model reference on `self._resolved_base_ref` (for
        `export_artifact`) and the model's device on `self.device`. The model is not stored on the
        instance; it is returned for the caller to thread.

        Returns:
            The resolved `(model, tokenizer)` pair.

        Raises:
            ValueError: If `model` is None and `base_model_name_or_path` is unset, or the tokenizer
                path cannot be resolved.
        """
        if model is None:
            if not self.base_model_name_or_path:
                raise ValueError("TRLMixin: model is None and `base_model_name_or_path` was not provided.")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **(self.hf_model_kwargs or {}),
            )

        self._resolved_base_ref = self.base_model_name_or_path or getattr(model, "name_or_path", None)

        if tokenizer is None:
            path = (
                self.tokenizer_name_or_path
                or getattr(model, "name_or_path", None)
                or self.base_model_name_or_path
            )
            if not path:
                raise ValueError("TRLMixin: could not resolve tokenizer path.")
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=self.trust_remote_code)
        else:
            self.tokenizer = tokenizer

        self.device = next(model.parameters()).device
        return model, self.tokenizer

    def _peft_config(self, model: nn.Module) -> LoraConfig | None:
        """The `LoraConfig` for this run, or None when PEFT is off or not LoRA.

        Built at steer time from a copy of `lora_kwargs`, with a suffix-list `target_modules` scoped
        to the decoder stack of `model` through `lora_target_pattern`. The declared `target_modules`
        on `self.args` is not modified, so `fit_identity()` and the recipe contain the list as the
        caller wrote it.

        Args:
            model: The resolved model the trainer will wrap.

        Returns:
            The config, or None.
        """
        if not (self.use_peft and self.peft_type == PeftType.LORA):
            return None
        kwargs = dict(self.lora_kwargs)
        kwargs["target_modules"] = lora_target_pattern(kwargs.get("target_modules"), model)
        return LoraConfig(**kwargs)

    def _post_train_freeze(self, model: PreTrainedModel) -> PreTrainedModel:
        """Put `model` in eval mode, freeze its parameters, and return it.

        The model is passed in and returned; it is not stored on the instance.
        """
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _maybe_save_trained_artifacts(self, trainer) -> None:
        output_dir = self._resolved_output_dir()
        trainer.save_model(output_dir)
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

    def _resolved_output_dir(self) -> str:
        """Return the configured output directory (`training_args["output_dir"]`, else `output_dir`).

        Without a configured directory, a temporary directory is created on the first call and
        returned on every later call of the same control.
        """
        configured = self.training_args.get("output_dir") or self.output_dir
        if configured:
            return str(configured)
        if self._transient_output_dir is None:
            self._transient_output_dir = tempfile.mkdtemp(prefix="steerability-trainer-")
        return self._transient_output_dir

    def _config_kwargs(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of `config_kwargs` whose `output_dir` is the resolved output directory."""
        return {**config_kwargs, "output_dir": self._resolved_output_dir()}

    def artifact_capability(self) -> Capability | None:
        """The serve capability implied by this training configuration.

        Training runs only when `train_dataset` is set, so a configuration without one produces
        no artifact. A LoRA run without merge-back saves an adapter to the output directory
        (`Capability.SERVE_LORA`); a full fine-tune saves a checkpoint there
        (`Capability.SERVE_CHECKPOINT`). The output directory is the configured one or a
        temporary directory. A merged LoRA run yields a checkpoint only when `merged_output_dir`
        is set.
        """
        if getattr(self, "train_dataset", None) is None:
            return None
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        if is_lora and not self.merge_lora_after_train:
            return Capability.SERVE_LORA
        if is_lora and self.merge_lora_after_train:
            return Capability.SERVE_CHECKPOINT if self.merged_output_dir else None
        return Capability.SERVE_CHECKPOINT

    def adapter_rank(self) -> int | None:
        """The LoRA rank `r` when this configuration's product is an adapter (`Capability.SERVE_LORA`), else None."""
        return int(self.r) if self.artifact_capability() == Capability.SERVE_LORA else None

    def adapter_target_modules(self) -> list[str] | None:
        """The `target_modules` list when this configuration's product is an adapter, else None (also for a regex)."""
        if self.artifact_capability() != Capability.SERVE_LORA or not isinstance(self.target_modules, list):
            return None
        return list(self.target_modules)

    def export_artifact(self) -> Artifact | None:
        """The on-disk product of this configuration's `steer()`, matching `artifact_capability()`.

        After `steer()`, and before it while the active fit cache contains the product, this is the stored product.
        Otherwise it is the configured output directory (`merged_output_dir` for a merged LoRA run).
        """
        capability = self.artifact_capability()
        if capability is None:
            return None
        if self._product is None and product_is_stored(control=self):
            self._product = self._artifact_of(state=stored_product(control=self))
        if self._product is not None:
            return self._product
        if capability == Capability.SERVE_LORA:
            base = self.base_model_name_or_path or self._resolved_base_ref or ""
            return LoRAArtifact(path=str(self._resolved_output_dir()), base_model=str(base))
        is_lora = bool(self.use_peft) and self.peft_type == PeftType.LORA
        path = self.merged_output_dir if (is_lora and self.merge_lora_after_train) else self._resolved_output_dir()
        return CheckpointArtifact(path=str(path))

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """One `("TrainedWeights", "direction")` fit when the configuration trains (`train_dataset` is set) and the
        active fit cache does not contain its product, else none."""
        if getattr(self, "train_dataset", None) is None or product_is_stored(control=self):
            return ()
        return (("TrainedWeights", "direction"),)

    def export_state(self) -> dict[str, Any]:
        """The trained product under the `"artifact"` key, for freezing and for the fit cache.

        The artifact is the one `export_artifact()` reports, and a LoRA product also gives its base model reference
        under `"base_model"`, since a stored adapter decodes to a path. Returns an empty mapping when the
        configuration trains nothing (`train_dataset` is None), so an inert wrapper's recipe is its frozen form.

        Raises:
            NotFreezableError: If the configuration trains but produces no on-disk product
                (a merged LoRA run without `merged_output_dir`).
        """
        if getattr(self, "train_dataset", None) is None:
            return {}
        artifact = self.export_artifact()
        if artifact is None:
            raise NotFreezableError(
                f"{type(self).__name__} merges its LoRA adapter after training but writes no on-disk "
                "product; set merged_output_dir to make the result freezable."
            )
        state: dict[str, Any] = {"artifact": artifact}
        if isinstance(artifact, LoRAArtifact):
            state["base_model"] = artifact.base_model
        return state

    def frozen_form(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The product as a `load_lora` or `load_checkpoint` entry, as this configuration's `artifact_capability()`
        decides.

        `state["artifact"]` is an artifact object, which a bundle stores by value, or a path.
        """
        if self.artifact_capability() == Capability.SERVE_LORA:
            return "structural_control/load_lora", {
                "path": state["artifact"],
                "base_model": state["base_model"],
                "merge": False,
            }
        return "structural_control/load_checkpoint", {"path": state["artifact"]}

    def fit_identity(self) -> Any | None:
        """The training-relevant args projection, or None when nothing trains.

        Output locations (`output_dir`, `merged_output_dir`, `resume_from_checkpoint`, and
        `training_args["output_dir"]`) are excluded, since where a product is saved does not
        change what was trained.
        """
        if getattr(self, "train_dataset", None) is None:
            return None
        args = getattr(self, "args", None)
        if args is None:
            return None
        excluded = {"output_dir", "merged_output_dir", "resume_from_checkpoint"}
        payload: dict[str, Any] = {}
        for f in fields(args):
            if not f.init or f.name in excluded:
                continue
            value = getattr(args, f.name)
            if f.name == "training_args" and isinstance(value, dict):
                value = {k: v for k, v in value.items() if k not in excluded}
            payload[f.name] = value
        return payload

    def _maybe_merge_lora_in_place(self, model: PreTrainedModel) -> PreTrainedModel:
        """Optionally merge LoRA into the base weights, returning the (possibly merged) model.

        When `use_peft` and `merge_lora_after_train` are set and `model` exposes `merge_and_unload`,
        merges the adapter, refreshes `self.device` from the merged model, and saves the merged model
        and tokenizer to `merged_output_dir` when set. Returns the merged model, or `model` unchanged
        otherwise. The model is not stored on the instance.
        """
        if not (self.use_peft and self.merge_lora_after_train):
            return model

        # trainer often returns a PEFT-wrapped model; merge if possible
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            self.device = next(model.parameters()).device

            # save if requested
            if self.merged_output_dir:
                model.save_pretrained(self.merged_output_dir)
                try:
                    self.tokenizer.save_pretrained(self.merged_output_dir)
                except Exception:
                    pass

        return model
