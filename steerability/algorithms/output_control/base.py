"""Base classes for output controls.

Output controls take part in decoding through three mechanisms:

- **Logits processors**: the pipeline gathers the results of each control's `get_logits_processors()` in the
  order of the pipeline's `controls` list and composes them into one `LogitsProcessorList`.
- **Stopping criteria**: the pipeline gathers the results of each control's `get_stopping_criteria()` in the
  same order. Generation stops when any criterion is met.
- **Decode loop**: the loop does not compose. At most one enabled `DecodingDriver` implements it, and the
  pipeline's default decode loop runs when no driver is enabled. A `DecodingDriver` receives the composed
  logits processors and stopping criteria as arguments and must apply them at every scoring step of every
  forward pass it runs.

Examples of output controls:

- Reward-augmented decoding (a step-level control)
- Self-disciplined autoregressive sampling (a step-level control)
- Decoding-time alignment and lookahead search (decoding drivers)
- Phased decoding and thinking intervention (decoding drivers)

See Also:

- `steerability.algorithms.output_control`: Implementations of output control methods
- `steerability.algorithms.output_control.common`: Shared component library
- `steerability.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Type

import torch
from transformers import LogitsProcessorList, PreTrainedModel, StoppingCriteriaList

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl
from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.core.execution.session_utils import session_generate

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.payloads import ConstraintSource, ProcessorSpec


def stack_generate_kwargs(logits_processors, stopping_criteria) -> dict:
    """Build the `model.generate` kwargs that pass the composed logits processors and stopping criteria.

    Args:
        logits_processors: The composed logits processors, or None.
        stopping_criteria: The composed stopping criteria, or None.

    Returns:
        A dict with `logits_processor` when `logits_processors` is non-empty and `stopping_criteria` when
        `stopping_criteria` is non-empty. An empty or None argument contributes no key.
    """
    extra: dict = {}
    if logits_processors is not None and len(logits_processors):
        extra["logits_processor"] = logits_processors
    if stopping_criteria is not None and len(stopping_criteria):
        extra["stopping_criteria"] = stopping_criteria
    return extra


def resolve_generate_callable(
    model: PreTrainedModel | None, runtime_kwargs: dict | None, session: Any = None
) -> Callable[..., torch.Tensor]:
    """Resolve the generate callable a driver rolls out with.

    Drivers generate through the pipeline's session (a `SteeredSession` carrying this
    generation's control entries), so a driver runs on any backend whose session serves its
    rollout parameters.

    Args:
        model: The pipeline model, or None on backends without a live model; unused.
        runtime_kwargs: Per-call parameters; unused.
        session: The `SteeringSession` for this generation.

    Returns:
        A callable with the `model.generate` calling convention returning full sequences.

    Raises:
        ValueError: If no session was provided.
    """
    if session is None:
        raise ValueError("No generate callable available: the driver received no session.")

    def _generate(input_ids, attention_mask=None, **gen_kwargs):
        return session_generate(session, input_ids, attention_mask, **gen_kwargs)

    return _generate


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
            `auxiliary_pass()` (see `steerability.algorithms.core.utils.auxiliary_pass`), which
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

    def get_logits_processors(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> list:
        """The control's logits processors for the current generation.

        Called once per `generate()` / `compute_logprobs()` call, after input and state
        controls have prepared the prompt (mirrors `StateControl.get_hooks`). `**kwargs`
        carries `attention_mask` and the caller's generation kwargs. Returned objects
        follow the HF `LogitsProcessor` convention; in-list order is preserved by the composition.

        A processor must behave as a function of `(prefix_ids, scores)`. Internal state is
        permitted only as memoization keyed on the prefix and must re-derive on a prefix mismatch,
        since drivers may restart, rewind, or reorder sequences, and scoring replays prefixes
        teacher-forced (subclass `common.processors.base.PrefixKeyedProcessor` to satisfy this
        mechanically). Return fresh processor instances from this hook; it is invoked once per call
        precisely so that per-generation state is isolated.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            A list of HF `LogitsProcessor`-style objects.
        """
        return []

    def get_stopping_criteria(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> list:
        """The control's stopping criteria.

        Not applied during scoring (there is no loop to stop). Same call convention as
        `get_logits_processors`.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.

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

    def export_processor_spec(self, runtime_kwargs: dict | None = None) -> ProcessorSpec | None:
        """The control's engine-hosted processor form, or None.

        A control whose per-step logit math is expressible in an engine's served processor
        vocabulary returns a `ProcessorSpec`; on a backend advertising
        `Capability.PER_STEP_LOGIT_SPECS` with the spec's kind, the pipeline submits it as a
        `ProcessorSpecEntry` in place of the control's live processor. The default returns
        None, which keeps the control on the live processor mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The processor spec, or None.
        """
        return None

    def export_constraint(self, runtime_kwargs: dict | None = None) -> ConstraintSource | None:
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
    """An output control that implements the decode loop.

    A subclass implements `decode()`, which receives the prompt batch, the composed logits processors
    and stopping criteria, and a session, and returns the full sequences. A pipeline may contain at
    most one enabled `DecodingDriver`, and `SteeringPipeline` raises `ValueError` at construction when
    it contains more. A driver is also an `OutputControl`. The pipeline composes the results of its
    `get_logits_processors()` and `get_stopping_criteria()` with those of the other output controls.

    The driver must apply the composed logits processors and stopping criteria at every scoring step
    of every forward pass it runs. Passing them to `model.generate` as `logits_processor` and
    `stopping_criteria` meets this requirement, and a custom decoding loop applies them explicitly.
    The driver issues its rollouts through the session it receives (`resolve_generate_callable`
    returns a generate callable for it). A driver written this way runs on any backend whose session
    supports its generation parameters. `model` is None on backends without a loaded model.

    In-process state hooks are built once per generation for the whole prompt batch and index their
    state by batch row. A driver whose session calls cover only some of the rows, or rows at
    different stream lengths, must set `whole_batch_rollouts` to False. With such a driver, the
    pipeline raises `ValueError` for a batch of more than one row when enabled state controls run as
    in-process hooks, and `SteeringPipeline.supports_batching` is False for that combination. Under a
    seed with `seed_scope="item"`, the in-process session decodes each session call one row at a time.
    The pipeline then raises `ValueError` for such a batch with any driver, unless the call passes
    `seed_scope="dispatch"`.

    Attributes:
        whole_batch_rollouts: Whether every session call the driver issues covers all rows of the
            batch, in row order, at a common stream length. Defaults to True.
    """

    whole_batch_rollouts: bool = True

    def max_rollouts_per_query(self) -> int | None:
        """Return an upper bound on the rollouts the driver requests for one candidate of one row.

        Every sequence the driver requests through the session counts, including proposals it
        discards. A session call over `F` rows with `num_return_sequences=n` counts as `F * n`
        rollouts. A `decode()` call with `num_return_sequences=k` requests up to `k` times the bound
        for each row. The bound lets a caller budget or refuse a configuration before running it.
        The default returns None.

        Returns:
            The bound, or None when the configuration has no static bound.
        """
        return None

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
        """Run the decode loop and return the full sequences.

        Args:
            input_ids: Prompt token ids of shape `[B, T]`, left-padded by the pipeline.
            attention_mask: Prompt attention mask with the same shape as `input_ids`.
            model: The loaded model, or None on backends without a loaded model.
            logits_processors: The composed `LogitsProcessorList`, with any processors the caller
                passed after those of the output controls. The driver applies it at every scoring
                step.
            stopping_criteria: The composed `StoppingCriteriaList`, with any criteria the caller
                passed after those of the output controls. The driver applies it at every scoring
                step.
            runtime_kwargs: Per-call parameters passed to `generate()`, or None.
            session: The `SteeredSession` through which the driver issues its rollouts.
            **gen_kwargs: Generation keyword arguments, e.g., `num_return_sequences` (`n`, 1 when
                absent) and `max_new_tokens`. They contain no logits processors or stopping criteria,
                since the pipeline passes those only through `logits_processors` and
                `stopping_criteria`.

        Returns:
            A tensor of shape `[B * n, T + L]`, where `L` is the length of the longest continuation.
            Rows are ordered by prompt row and then by candidate. Each row is the padded prompt row of
            `input_ids` followed by one candidate's continuation, right-padded. The pipeline reads the
            columns after the first `T` as the continuations.
        """
