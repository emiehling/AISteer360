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
from abc import ABC, abstractmethod
from dataclasses import fields

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.core.base_args import BaseArgs
from aisteer360.core.requirements import Capability, Requirements


class StructuralControl(ABC):
    """Abstract base class for structural control steering methods.

    Modifies model parameters or architecture persistently, returning a new model instance with transformed weights.

    Methods:
        steer(model, tokenizer, **kwargs) -> PreTrainedModel: Training logic (required)
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = True

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes
        for field in fields(self.args):
            setattr(self, field.name, getattr(self.args, field.name))

    @abstractmethod
    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer = None,
            **kwargs
    ) -> PreTrainedModel:
        """Required steering/preparation."""
        pass

    def cleanup(self) -> None:
        """Release resources allocated during steer().

        Override this method in subclasses that allocate GPU memory or other resources
        during steering to ensure proper cleanup.
        """
        pass

    def requires(self) -> Requirements:
        """Return the backend capabilities this control needs at steer time.

        Structural controls train or transform weights in process, requiring `WEIGHT_TRAINING` and
        direct model access (`RAW_MODEL`) on the steering backend. Disabled controls require nothing.

        Returns:
            The control's `Requirements` (phase `"steer"`).
        """
        if not getattr(self, "enabled", True):
            return Requirements()
        return Requirements(
            capabilities=Capability.WEIGHT_TRAINING | Capability.RAW_MODEL,
            phase="steer",
        )


class NoStructuralControl(StructuralControl):
    """Identity structural control.

    Used as the default when no structural control is needed. Passes the model through unchanged.
    """
    enabled: bool = False

    def steer(self, model: PreTrainedModel, **__) -> PreTrainedModel:
        """Null steer operation; returns model."""
        return model
