from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.budget_forcing.args import BudgetForcingArgs
from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PhasedDriver


class BudgetForcing(PhasedDriver):
    """Implementation of Budget Forcing (s1) from Muennighoff et al., 2025.

    Budget Forcing controls the length of a reasoning model's thinking at test time. It limits each
    thinking segment to a token budget and forces the closing think tag before the answer. It can
    also lengthen the thinking by appending an extension text such as `"Wait"` and generating another
    thinking segment. The constructor arguments configure the plan:

    - `max_thinking_tokens` (default 512): the token budget of each thinking segment.
    - `end_think` (default `"</think>"`): the closing think tag, which ends a thinking segment and is
      appended before the answer.
    - `end_think_token_ids` (default empty): token ids that also end a thinking segment, used for a
      closing tag that tokenizes to a special token.
    - `num_extensions` (default 0): the number of extension rounds.
    - `extension_text` (default `"Wait"`): the text appended at the start of each extension round.

    The plan runs in three steps:

    1. **Thinking**: a `Generated` phase runs until `end_think`, a token in `end_think_token_ids`, or
       `max_thinking_tokens` tokens, whichever comes first.
    2. **Extensions**: each of the `num_extensions` rounds appends `extension_text` and generates
       another thinking segment with the same boundaries.
    3. **Answer**: `end_think` is appended, followed by a `Generated` phase with no budget of its own.

    When a thinking segment ends on the closing tag or token, the tag stays in the stream, and the
    extension text or the forced closing tag is appended after it. Every `Generated` phase applies
    the composed logits processors and stopping criteria, and a step-level control in the pipeline
    steers each phase. The caller's `max_new_tokens` limits each candidate's thinking and answer
    together. The full thinking and answer are returned as the continuation.

    Reference:

    - "s1: Simple test-time scaling"
      Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
      Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
      [https://arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)
    """

    Args = BudgetForcingArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def _configure(self) -> None:
        """Budget forcing keeps the full thinking + answer stream (no extract rule)."""
        self.extract_after = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to splice phase boundaries."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def max_rollouts_per_query(self) -> int:
        """Return `num_extensions + 2`, the number of `Generated` phases in the plan.

        The plan has the initial thinking phase, one thinking phase per extension round, and the
        answer phase. Each phase requests one rollout per candidate.

        Returns:
            The bound on the rollouts for one candidate of one row.
        """
        return self.num_extensions + 2

    def plan(self, prompt_text: str, params: dict) -> list:
        """Build the thinking-budget plan: bounded thinking, optional extensions, forced tag, answer.

        Each thinking phase ends at the `end_think` string, any token in `end_think_token_ids`, or
        `max_thinking_tokens`; the forced closing tag before the answer is the `end_think` text.
        """
        thinking = lambda: Generated(
            until=self.end_think, until_token_ids=self.end_think_token_ids, budget=self.max_thinking_tokens,
        )
        plan = [thinking()]
        for _ in range(self.num_extensions):
            plan.append(Fixed(self.extension_text))
            plan.append(thinking())
        plan.append(Fixed(self.end_think))
        plan.append(Generated())
        return plan
