"""Constrained decoding: declarative structured outputs rendered per execution arm."""
from __future__ import annotations

import torch

from steerability.algorithms.core.execution.contracts import Capability, ConstraintKinds, Requirements, any_of, needs
from steerability.algorithms.core.execution.payloads import ConstraintSource
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.processors.constraint import ConstraintProcessor

from .args import ConstrainedDecodingArgs


class ConstrainedDecoding(OutputControl):
    """Output control that restricts generated text to a JSON schema, regular expression, grammar, or set of choices.

    `ConstrainedDecoding` is configured with exactly one constraint. The `json_schema`, `regex`, `grammar` (EBNF), and
    `choice` arguments each give a declarative constraint of that kind. The `source` argument accepts the same
    constraints as a `ConstraintSource` or as a mapping with `kind` and `value` keys. The `automaton` argument accepts
    an in-memory object that implements the `ConstraintAutomaton` protocol (`reset(prefix_ids)` and
    `allowed(prefix_ids)`) instead.

    A declarative constraint is applied in one of two ways, depending on the backend:

    - **In process**: the constraint is compiled with xgrammar the first time it is applied, using the pipeline's
      tokenizer. A `ConstraintProcessor` sets the logit of every token the grammar does not permit to -inf.
    - **vLLM backends**: the constraint is passed to the engine as its native structured-output parameters, and
      nothing is compiled in the client. This requires a pipeline without a decoding driver.

    Both forms are built from the same source. They differ in the implementation that computes the mask (the
    client-side automaton or the engine's grammar backend).

    In process, each row of a batch has its own grammar state. Rows can be batched prompts, the candidates of
    `num_return_sequences`, or the beams of beam search. Each generation starts with new grammar states over the
    compiled grammar. The constrained output of a row begins at the end of its prompt, after any text that a decoding
    driver inserts before the first generated token. After a token the grammar rejects (e.g., text that a decoding
    driver inserts after generated tokens), or once the grammar has terminated, only the stop tokens are permitted
    for that row.

    An automaton object runs in process only. `supports_batching` is False for it, since the object may return one
    permitted set for the whole batch. Every generation uses the same object. A control constructed with an automaton
    object therefore serves one generation at a time.

    With `include_in_scoring=True` (the default), the constraint also applies in `compute_logprobs`. Scoring then
    requires the in-process backend, since structured outputs do not apply to prompt logprobs. With
    `include_in_scoring=False`, scores do not reflect the constraint.

    Attributes:
        tokenizer: The tokenizer attached by the pipeline and used to compile a declarative constraint, or None
            before the pipeline attaches it.
    """

    Args = ConstrainedDecodingArgs

    def _configure(self) -> None:
        self.tokenizer = None
        self._compiled_automaton = None
        # a user-supplied automaton may return one allowed set for the whole batch
        self.supports_batching = self.automaton is None

    def requirements(self) -> Requirements:
        """In-process compilation or engine-native structured outputs at generate."""
        score = needs(Capability.IN_PROCESS_TORCH) if self.include_in_scoring else ()
        if self.source is None:
            return Requirements(
                generate=needs(
                    Capability.IN_PROCESS_TORCH,
                    hint=(
                        "a live automaton object has no declarative form; construct the control "
                        "with a ConstraintSource (or json_schema/regex/grammar/choice) or run "
                        "this pipeline on the huggingface backend"
                    ),
                ),
                score=score,
            )
        return Requirements(
            generate=any_of(
                needs(Capability.IN_PROCESS_TORCH),
                needs(
                    Capability.GUIDED_DECODING,
                    kinds=ConstraintKinds(constraints=frozenset({self.source.kind})),
                ),
            ),
            score=score,
        )

    def export_constraint(self, runtime_kwargs: dict | None = None) -> ConstraintSource | None:
        """The declarative source, or None for automaton-object configurations."""
        return self.source

    def _automaton(self, prompt_ids: torch.Tensor | None):
        if self.automaton is not None:
            return self.automaton
        if self._compiled_automaton is None:
            if self.tokenizer is None:
                raise RuntimeError(
                    "ConstrainedDecoding requires a tokenizer to compile its constraint; "
                    "steer() must run first."
                )
            from .utils.automaton import compile_constraint_automaton

            self._compiled_automaton = compile_constraint_automaton(self.source, self.tokenizer)
        return self._compiled_automaton.fresh(prompt_ids)

    def get_logits_processors(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> list:
        """Return a `ConstraintProcessor` that applies the constraint in process.

        With an automaton object, the processor drives that object. Otherwise, the processor drives a new automaton
        over the compiled grammar that treats the rows of `input_ids` as the prompts of the generation. The grammar
        is compiled on the first call and reused by later calls.

        Args:
            input_ids: The steered prompt token ids of shape `[B, T]`.
            runtime_kwargs: Per-call parameters (unused).
            **kwargs: The `attention_mask` and the generation keyword arguments (unused).

        Returns:
            A list with one `ConstraintProcessor`.

        Raises:
            RuntimeError: If a declarative constraint must be compiled and no tokenizer is attached, e.g., because
                `steer()` has not run.
        """
        return [ConstraintProcessor(self._automaton(input_ids))]
