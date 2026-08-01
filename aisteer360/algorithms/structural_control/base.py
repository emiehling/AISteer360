"""Structural control base classes.

This module provides the abstract base class for methods that create persistent changes to the model, either through
weight updates or architectural changes.

Two base classes are provided:

- `StructuralControl`: Base class for all structural control methods.
- `NoStructuralControl`: Identity (null) control; used when no structural control is defined in steering pipeline.

Structural controls implement steering through model weight or architecture modifications, transforming base parameters
θ to θ', resulting in generations following y ~ p_θ'(x).

Examples of structural controls:

- Fine-tuning (full or parameter-efficient like LoRA)
- Model merging (e.g., via MergeKit)
- Direct Preference Optimization (DPO)
- Adapter layers and modules
- Weight interpolation and averaging

See Also:

- `aisteer360.algorithms.structural_control`: Implementations of structural control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import abstractmethod

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.base_control import BaseControl
from aisteer360.algorithms.core.execution.capabilities import Capability
from aisteer360.algorithms.core.execution.requirements import Requirements, needs


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
            tokenizer: PreTrainedTokenizer = None,
            session=None,
            **kwargs
    ) -> PreTrainedModel:
        """Required steering/preparation.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        Structural controls train against the live model, so the steer phase requires
        `Capability.IN_PROCESS_TORCH` and `Capability.WEIGHT_TRAINING`. The generate phase
        requires `Capability.IN_PROCESS_TORCH`, since the returned model is adopted in process.

        Returns:
            The control's phase-keyed requirements.
        """
        return Requirements(
            steer=needs(Capability.IN_PROCESS_TORCH, Capability.WEIGHT_TRAINING),
            generate=needs(Capability.IN_PROCESS_TORCH),
        )


class NoStructuralControl(StructuralControl):
    """Identity structural control.

    Used as the default when no structural control is needed. Passes the model through unchanged.
    """
    enabled: bool = False

    def steer(self, model: PreTrainedModel, **__) -> PreTrainedModel:
        """Null steer operation; returns model."""
        return model
