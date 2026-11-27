"""Structural control base class.

This module provides `StructuralControl`, the abstract base class for methods that make persistent changes to the
model by updating its weights or modifying its architecture. A subclass implements `steer()`, which returns the new or
modified model.

A structural control transforms the base parameters θ into θ'. Generations from the steered model then follow
y ~ p_θ'(x).

Examples of structural controls:

- Fine-tuning (full or parameter-efficient like LoRA)
- Model merging (e.g., via MergeKit)
- Direct Preference Optimization (DPO)
- Adapter layers and modules
- Weight interpolation and averaging

See Also:

- `steerability.algorithms.structural_control`: Implementations of structural control methods
- `steerability.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import abstractmethod
from collections.abc import Sequence

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability, Requirements, SpecConstraint, any_of, needs
from steerability.algorithms.core.execution.payloads import Artifact
from steerability.algorithms.core.execution.spec import BackendSpec


def lora_capacity(*, rank: int | None) -> SpecConstraint:
    """The backend configuration that an offline vLLM engine needs to serve a control's LoRA adapter.

    An offline engine serves an adapter only when its spec's `engine_kwargs` set `enable_lora` and do not set the
    `dtype` to float32, and an adapter of a given rank only when they also allow that rank
    (`BackendSpec.lora_rank_limit`). The constraint holds on the other backend kinds, since the in-process backend
    attaches the adapter to its model and a vLLM server's capacity is set when the server starts.

    Args:
        rank: The adapter's rank, or None when the configuration does not give it (the engine's limit is then
            checked when the adapter is served).

    Returns:
        The constraint on the spec of the generate phase.
    """

    def serves(spec: BackendSpec) -> bool:
        if spec.kind != "vllm":
            return True
        limit = spec.lora_rank_limit()
        return limit is not None and (rank is None or rank <= limit)

    needed = "`enable_lora`" if rank is None else f"`enable_lora` and a `max_lora_rank` of at least {rank}"
    return SpecConstraint(
        description=(
            f"an offline engine serves a LoRA adapter{'' if rank is None else f' of rank {rank}'} only when the "
            f"engine_kwargs of the backend spec's options set {needed} and do not set `dtype` to float32 (vLLM "
            "applies adapters in float16 or bfloat16)."
        ),
        predicate=serves,
    )


def lora_coverage(*, modules: Sequence[str]) -> SpecConstraint:
    """The backend configuration that an offline vLLM engine needs to apply all of a control's LoRA adapter.

    An engine whose spec's `engine_kwargs` set `lora_target_modules` gives LoRA layers only to those modules, and vLLM
    applies an adapter's weights only on them (`BackendSpec.lora_modules_left_out`). The constraint holds on the other
    backend kinds and on an engine without `lora_target_modules`.

    Args:
        modules: The modules the adapter changes.

    Returns:
        The constraint on the spec of the generate phase.
    """

    def covers(spec: BackendSpec) -> bool:
        return spec.kind != "vllm" or not spec.lora_modules_left_out(modules=modules)

    return SpecConstraint(
        description=(
            f"an offline engine applies all of a LoRA adapter that changes {sorted(set(modules))} only when its "
            "`lora_target_modules` cover these modules; add them to lora_target_modules in the engine_kwargs of the "
            "backend spec's options."
        ),
        predicate=covers,
    )


def lora_scoring() -> SpecConstraint:
    """The backend configuration that an offline vLLM engine needs to score under a control's LoRA adapter.

    A scoring request reads the log-probabilities of its prompt. A batch of such requests under an adapter can stop an
    engine whose LoRA layers include the language-model head (vLLM 0.32). The `lora_target_modules` option of the spec's
    `engine_kwargs` restricts the engine's LoRA layers to the modules it lists, and the engine checks at session open
    that it covers the modules an adapter changes. The constraint holds on the other backend kinds, since the in-process
    backend attaches the adapter to its model and a vLLM server's LoRA layers are set when the server starts.

    Returns:
        The constraint on the spec of the score phase.
    """

    def scores(spec: BackendSpec) -> bool:
        if spec.kind != "vllm":
            return True
        modules = spec.get_option("engine_kwargs", "lora_target_modules")
        return bool(modules) and "lm_head" not in modules

    return SpecConstraint(
        description=(
            "an offline engine scores under a LoRA adapter only when its LoRA layers leave out lm_head; set "
            "`lora_target_modules` in the engine_kwargs of the backend spec's options to the modules the adapter "
            "changes, e.g., [\"q_proj\", \"v_proj\"]."
        ),
        predicate=scores,
        phases=("score",),
    )


class StructuralControl(BaseControl):
    """Abstract base class for structural control steering methods.

    Modifies model parameters or architecture persistently, returning a new model instance with transformed weights.

    Methods:
        steer(model, tokenizer, **kwargs) -> PreTrainedModel: Training logic (required)
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = True

    @abstractmethod
    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase = None,
            session=None,
            **kwargs
    ) -> PreTrainedModel:
        """Required steering/preparation.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def artifact_capability(self) -> Capability | None:
        """The serve capability implied by this configuration's steer-time artifact, or None.

        Controls whose `steer()` writes a servable product to disk return
        `Capability.SERVE_CHECKPOINT` for a full-weights checkpoint or `Capability.SERVE_LORA`
        for an adapter, so the generate phase gains a serving alternative. The default returns
        None (no on-disk artifact), which keeps the generate phase in-process only.

        Returns:
            The capability, or None.
        """
        return None

    def export_artifact(self) -> Artifact | None:
        """The steer-time artifact this control produced, or None.

        Called by the pipeline after `steer()` completes. The returned artifact must exist on
        disk and correspond to `artifact_capability()` (a `CheckpointArtifact` for
        `Capability.SERVE_CHECKPOINT`, a `LoRAArtifact` for `Capability.SERVE_LORA`). The
        default returns None.

        Returns:
            The artifact, or None.
        """
        return None

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The generate phase requires `Capability.IN_PROCESS_TORCH` for in-process adoption of
        the returned model; when the configuration produces an on-disk artifact
        (`artifact_capability()`), serving that artifact is an alternative, so a backend
        advertising the matching serve capability also supports the generate phase.

        A configuration that serves a LoRA adapter also needs an offline vLLM engine that serves it
        (`lora_capacity`, with the rank of `adapter_rank()`), that applies all of it (`lora_coverage`, when
        `adapter_target_modules()` gives its modules), and that scores under it (`lora_scoring`).

        Returns:
            The control's phase-keyed requirements.
        """
        generate = needs(Capability.IN_PROCESS_TORCH)
        capability = self.artifact_capability()
        constraints: tuple[SpecConstraint, ...] = ()
        if capability is not None:
            generate = any_of(
                generate,
                needs(capability, hint="serve the steer-time artifact on a vLLM backend"),
            )
        if capability is Capability.SERVE_LORA:
            constraints = (lora_capacity(rank=self.adapter_rank()), lora_scoring())
            modules = self.adapter_target_modules()
            if modules is not None:
                constraints += (lora_coverage(modules=modules),)
        return Requirements(generate=generate, spec_constraints=constraints)

    def adapter_rank(self) -> int | None:
        """The rank of the LoRA adapter this configuration serves, when the configuration gives it.

        The default returns None, e.g., for an adapter loaded from a directory, whose rank is in its configuration
        file.

        Returns:
            The rank, or None.
        """
        return None

    def adapter_target_modules(self) -> list[str] | None:
        """The names of the modules that the LoRA adapter of this configuration changes, when the configuration gives
        them.

        The default returns None, e.g., for an adapter loaded from a directory, whose modules are in its weights.

        Returns:
            The module names, or None.
        """
        return None

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; training happens on live weights."""
        return ModelAccess.MODULE

    def restore(self, model: PreTrainedModel | None) -> PreTrainedModel | None:
        """Undo this control's change to the model it steered and return the model as it was before `steer()`.

        A control whose change can be undone in place (an unmerged LoRA adapter attached to the incoming model)
        removes it and returns the restored model. A `PipelineFactory` steers a recipe whose structural controls all
        install an adapter from disk on its shared base model, and it calls this method, last control first, when the
        recipe's pipeline is released. The default returns None, i.e., the change cannot be undone.

        Args:
            model: The pipeline model after this control's `steer()` (and after the restore of the controls that
                follow it).

        Returns:
            The restored model, or None when the change cannot be undone.
        """
        return None
