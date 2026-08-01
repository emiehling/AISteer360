"""Output control base classes.

Output controls participate in decoding through two mechanisms:

- Logits-processor composition: each control's `get_logits_processors()` results are gathered in
  pipeline `controls` list order and composed by `LogitsProcessorList`.
- Stopping-criteria composition: each control's `get_stopping_criteria()` results are gathered the
  same way; generation stops when any criterion fires.
- The decode loop is exclusive: it is implemented by exactly one `DecodingDriver`, which receives
  the composed stacks as explicit parameters and must apply them at every scoring step of every
  forward pass it issues.

Examples of output controls:

- Reward-augmented decoding (a step-level control)
- Self-disciplined autoregressive sampling (a step-level control)
- Decoding-time alignment / lookahead search (a decoding driver)
- Phase splicing / thinking intervention (a decoding driver)

See Also:

- `aisteer360.algorithms.output_control`: Implementations of output control methods
- `aisteer360.algorithms.output_control._common`: Shared component library
- `aisteer360.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Type

import torch
from transformers import LogitsProcessorList, PreTrainedModel, StoppingCriteriaList

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.base_control import BaseControl
from aisteer360.algorithms.core.execution.capabilities import Capability
from aisteer360.algorithms.core.execution.items import GenerationItem
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.prompts import PreparedPrompt
from aisteer360.algorithms.core.execution.requirements import Requirements, needs


def stack_generate_kwargs(logits_processors, stopping_criteria) -> dict:
    """Build the `model.generate` kwargs for the composed stacks, each included only when non-empty.

    Shared by every driver that delegates to `model.generate` (the default driver, and the segment
    and phase drivers per rollout or per phase) so the "pass the stack only when non-empty" rule
    lives in one place.
    """
    extra: dict = {}
    if logits_processors is not None and len(logits_processors):
        extra["logits_processor"] = logits_processors
    if stopping_criteria is not None and len(stopping_criteria):
        extra["stopping_criteria"] = stopping_criteria
    return extra


def session_generate(session, input_ids, attention_mask=None, **gen_kwargs) -> torch.Tensor:
    """Run one generate call through a `SteeringSession`, returning full sequences.

    Drop-in replacement for `model.generate(input_ids=..., attention_mask=..., **gen_kwargs)`
    inside driver rollouts. Each row of `input_ids` becomes one `GenerationItem`; the keyword
    arguments normalize through `GenerationParams.from_gen_kwargs`, so live `logits_processor`
    and `stopping_criteria` stacks travel in `extra` (consumable in process only). The returned
    tensor holds the prompt plus continuation per candidate row, right-padded to a common
    length with the session tokenizer's pad token.

    Args:
        session: The `SteeringSession` to generate on.
        input_ids: Prompt token ids of shape `[batch, seq_len]`.
        attention_mask: Attention mask matching `input_ids`, or None.
        **gen_kwargs: Generation keyword arguments in `model.generate` vocabulary.

    Returns:
        Full sequences of shape `[batch * n, seq_len + gen_len]`.
    """
    params = GenerationParams.from_gen_kwargs(**gen_kwargs)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    items = []
    for row in range(input_ids.size(0)):
        mask_row = attention_mask[row:row + 1] if attention_mask is not None else None
        items.append(GenerationItem(
            prompt=PreparedPrompt.from_token_ids(input_ids[row:row + 1], mask_row),
        ))
    results = session.generate(items, params)

    tokenizer = getattr(session, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None) or 0

    full_rows: list[torch.Tensor] = []
    for result in results:
        prompt_ids = result.output.adapted_input_ids
        out_ids = result.output.output_ids.to(prompt_ids.device)
        repeated = prompt_ids.expand(out_ids.size(0), -1)
        full_rows.append(torch.cat([repeated, out_ids], dim=1))
    max_len = max(row.size(1) for row in full_rows)
    padded = [
        torch.nn.functional.pad(row, (0, max_len - row.size(1)), value=pad_token_id)
        for row in full_rows
    ]
    return torch.cat(padded, dim=0)


def resolve_generate_callable(model, runtime_kwargs: dict | None, session=None):
    """Resolve the generate callable a driver rolls out with.

    A `runtime_kwargs["base_generate"]` override is honored with a `DeprecationWarning` (pass a
    session instead); otherwise the session's generate is used when a session is available, and
    `model.generate` as the in-process fallback.

    Args:
        model: The pipeline model, or None on backends without a live model.
        runtime_kwargs: Per-call parameters, possibly carrying the deprecated override.
        session: The `SteeringSession` for this generation, or None.

    Returns:
        A callable with the `model.generate` calling convention returning full sequences.

    Raises:
        ValueError: If no generate callable can be resolved.
    """
    runtime_kwargs = runtime_kwargs or {}
    override = runtime_kwargs.get("base_generate")
    if override is not None:
        warnings.warn(
            "runtime_kwargs['base_generate'] is deprecated; drivers generate through the "
            "pipeline's session. The override is honored for this call.",
            DeprecationWarning,
            stacklevel=3,
        )
        if not callable(override):
            raise ValueError("'base_generate' must be callable.")
        return override
    if session is not None:
        def _generate(input_ids, attention_mask=None, **gen_kwargs):
            return session_generate(session, input_ids, attention_mask, **gen_kwargs)
        return _generate
    if model is not None:
        return model.generate
    raise ValueError("No generate callable available: the driver received neither a session nor a model.")


class OutputControl(BaseControl):
    """Base class for output-control steering methods.

    An `OutputControl` participates in decoding through the composable mechanisms above.
    Controls that implement a decoding procedure subclass `DecodingDriver` instead.

    Class attributes:
        include_in_scoring: Whether this control's logits processors also apply during
            `SteeringPipeline.compute_logprobs()` (per-position, teacher-forced). Defaults
            to True. Set False when the processors are too expensive to evaluate per reference
            position (see `BaseCandidateValue.scoring_cost`).
        same_model_forwards: Whether this component issues additional forward passes through the
            pipeline's own model during decoding. Such passes must be wrapped in
            `auxiliary_pass()` (see `aisteer360.algorithms.core.utils.auxiliary_pass`), which
            keeps them out of state-control condition scoring, gate updates, and fallback
            position counting. Defaults to False; the flag is declarative metadata and is not
            read by the pipeline.
    """

    Args: Type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False
    include_in_scoring: bool = True
    same_model_forwards: bool = False

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """The control's logits processors for the current generation.

        Called once per `generate()` / `compute_logprobs()` call, after input and state
        controls have prepared the prompt (mirrors `StateControl.get_hooks`). `**kwargs`
        carries `attention_mask` and the caller's generation kwargs. Returned objects
        follow the HF `LogitsProcessor` convention; in-list order is preserved by the composition.

        A processor must behave as a function of `(prefix_ids, scores)`. Internal state is
        permitted only as memoization keyed on the prefix and must re-derive on a prefix mismatch,
        since drivers may restart, rewind, or reorder sequences, and scoring replays prefixes
        teacher-forced (subclass `_common.processors.base.PrefixKeyedProcessor` to satisfy this
        mechanically). Return fresh processor instances from this hook; it is invoked once per call
        precisely so that per-generation state is isolated.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.
            **kwargs: Carries `attention_mask` and the caller's generation kwargs.

        Returns:
            A list of HF `LogitsProcessor`-style objects.
        """
        return []

    def get_stopping_criteria(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """The control's stopping criteria.

        Not applied during scoring (there is no loop to stop). Same call convention as
        `get_logits_processors`.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.
            **kwargs: Carries `attention_mask` and the caller's generation kwargs.

        Returns:
            A list of HF `StoppingCriteria`-style objects.
        """
        return []

    def export_generation_params(self, runtime_kwargs: dict | None = None) -> Mapping[str, Any] | None:
        """The control's sampling-expressible contribution, or None.

        A control whose behavior is expressible as normalized generation parameters returns a
        mapping over a subset of `stop_strings`, `stop_token_ids`, `max_new_tokens`, and
        `min_new_tokens`; the pipeline merges it into the call's `GenerationParams` (stop rules
        union with the caller's; token bounds only tighten) and does not additionally collect
        the control's live processors and criteria for that call, so the control executes on
        every backend through the session's composed stop rules. The default returns None, which
        keeps the control on the live processor/criteria mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The parameter contribution, or None.
        """
        return None

    def export_constraint(self, runtime_kwargs: dict | None = None):
        """The control's declarative constrained-decoding source, or None.

        A control whose per-step masking compiles from a declarative source returns a
        `ConstraintSource`; on a backend advertising `Capability.GUIDED_DECODING` the pipeline
        renders it onto the engine's native structured-output parameters in place of the
        control's live processor. The default returns None, which keeps the control on the live
        processor mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The constraint source, or None.
        """
        return None

    def steer(self, model: PreTrainedModel, tokenizer=None, session=None, **kwargs) -> None:
        """Optional one-time preparation (e.g., load a reward model, fit a probe).

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The default requires `Capability.IN_PROCESS_TORCH` at generate and, when
        `include_in_scoring` is True, at score as well, since remote prompt-logprob computation
        applies neither live processors nor engine-registered sampling processors to prefill
        logits. Setting `include_in_scoring=False` removes the score-phase requirement.

        Returns:
            The control's phase-keyed requirements.
        """
        score = needs(Capability.IN_PROCESS_TORCH) if self.include_in_scoring else ()
        return Requirements(generate=needs(Capability.IN_PROCESS_TORCH), score=score)


class DecodingDriver(OutputControl):
    """An output control that implements the decoding procedure.

    Exactly one enabled driver may exist per pipeline (the decode loop does not compose).
    Driver contract: `logits_processors` and `stopping_criteria` are the composed,
    authoritative stacks for this generation; the driver applies them at every scoring
    step of every forward pass it issues. Delegating to `model.generate(...,
    logits_processor=..., stopping_criteria=...)` satisfies the contract; hand-rolled
    loops apply them explicitly.

    A driver is also an `OutputControl`: it may additionally contribute processors or
    criteria of its own via the `get_*` hooks, which the pipeline composes like any other
    control's.

    The pipeline passes `session=`, the `SteeringSession` for this generation. Drivers issue
    their rollouts through it (`resolve_generate_callable` returns the right callable), so a
    driver runs on any backend whose session serves its rollout parameters; `model` is None on
    backends without a live model.
    """

    @abstractmethod
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: PreTrainedModel | None,
        logits_processors: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        runtime_kwargs: dict | None,
        session=None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the decoding procedure; return full sequence ids (prompt + continuation)."""


class HFGenerateDriver(DecodingDriver):
    """Default decoding driver: delegate the loop to the model's own `generate`."""

    supports_batching: bool = True

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        extra = stack_generate_kwargs(logits_processors, stopping_criteria)
        return model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **extra, **gen_kwargs
        )
