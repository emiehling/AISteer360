"""Input control base classes.

This module provides the abstract base class for methods that modify prompts before they reach the model.

Two base classes are provided:

- `InputControl`: Base class for all input control methods.
- `NoInputControl`: Identity (null) control; used when no input control is defined in steering pipeline.

Input controls implement steering through prompt transformation σ(x), enabling behavior modification without altering
model parameters or architecture. These methods transform inputs before they reach the model, resulting in generations
following y ~ p_θ(σ(x)).

Examples of input controls:

- Few-shot learning (prepending examples)
- Prompt templates and formatting
- Soft prompts and prompt tuning
- Chain-of-thought prompting
- Iterative prompt refinement

See Also:

- `aisteer360.algorithms.input_control`: Implementations of input control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Type

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.types import Output


class InputControl(ABC):
    """Abstract base class for input control steering methods.

    Transforms prompts before model processing. May maintain internal memory that is updated either offline (in
    `steer()`) or online (in `observe()`).

    Subclasses that maintain artifacts (instructions, demonstrations, learned weights, rule streams, ...) should expose
    them via a `memory` attribute. The framework treats this attribute as opaque but reserves it as the recognized
    location for serialization tooling, debugging, and Benchmark checkpoint integration. The expected type is anything
    satisfying the `Memory` Protocol (`input_control.common.memory.Memory`); see `TextMemory` for the canonical reusable
    shape.

    Methods:
        adapt(input_ids, prior, runtime_kwargs) -> input_ids: Transform prompt token IDs (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
        observe(input_ids, output, runtime_kwargs) -> None: Post-generation memory update (optional)
        cleanup() -> None: Release resources allocated during steer (optional)
    """

    Args: Type[BaseArgs] | None = None

    enabled: bool = True
    supports_batching: bool = False
    is_stateful: bool = False

    memory: "Memory | None" = None  # subclasses populate in steer() or observe()

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
    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Transform `input_ids` into a steered prompt.

        May read instance state (e.g. `self.memory`) that was populated by `steer()` or `observe()`.

        Args:
            input_ids: The user's prompt token IDs.
            prior: The most recent `Output` produced by this pipeline, or None if no prior output exists OR if
                `is_stateful` is False. The pipeline is responsible for not passing `prior` to stateless controls.
            runtime_kwargs: Per-call parameters.

        Returns:
            The transformed token IDs.
        """

    def steer(
        self,
        model=None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        """Optional offline preparation. Default is no-op."""
        pass

    def observe(
        self,
        input_ids: torch.Tensor,
        output: Output,
        runtime_kwargs: dict | None = None,
    ) -> None:
        """Optional post-generation callback. Default is no-op.

        Invoked by `SteeringPipeline.generate()` after every call IF and ONLY IF `is_stateful` is True. Stateful
        controls override this to update internal memory based on the model's response.

        Args:
            input_ids: The user's pre-adapt prompt (2D tensor).
            output: The model's response, wrapped in an `Output`.
            runtime_kwargs: Per-call parameters.
        """
        pass

    def cleanup(self) -> None:
        """Release resources allocated during `steer()`.

        Override this method in subclasses that allocate GPU memory or other resources during steering to ensure proper
        cleanup.
        """
        pass


class NoInputControl(InputControl):
    """Identity input control.

    Used as the default when no input control is needed. Returns input_ids unchanged.
    """
    enabled: bool = False
    supports_batching: bool = True
    is_stateful: bool = False
    tokenizer: PreTrainedTokenizerBase | None = None

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Identity adapter; returns input_ids unchanged."""
        return input_ids

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> None:
        """Null steer operation; attaches tokenizer."""
        self.tokenizer = tokenizer
