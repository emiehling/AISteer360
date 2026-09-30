"""
Few-shot learning control for prompt adaptation.
"""
import hashlib
import json
import warnings
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.core.identity import canonical_value
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.common.formatters.few_shot_block import FewShotBlockFormatter
from steerability.algorithms.input_control.common.memory.pool import PoolMemory
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.common.selectors.base import BaseSelector
from steerability.algorithms.input_control.few_shot.args import FewShotArgs
from steerability.algorithms.input_control.few_shot.selectors import selector_from_arg
from steerability.utils.rendering import has_chat_template, render_messages


class FewShot(InputControl):
    """Input control that adds labeled positive and negative examples to each prompt.

    `FewShot` renders a `directive` and a set of examples as one block and adds the block to each
    prompt. The examples come from one of two sources:

    1. **Pool-based sampling**: `steer()` stores `positive_example_pool` and `negative_example_pool`,
        and each prompt receives `k_positive` and `k_negative` examples chosen by the `selector` (uniform
        random sampling by default, or a retriever such as `EPRSelector`). Examples are chosen
        separately for each prompt row, with the prompt as the selector's query.

    2. **Runtime examples**: the `positive_examples` and `negative_examples` runtime kwargs give the
        examples for one call directly. When either key is present, pool-based selection is skipped
        for both polarities.

    The block contains the `directive` (when set) followed by each example under the header
    `"### Positive example (behavior to follow)"` or `"### Negative example (behavior to avoid)"`. Each
    example field is rendered as a `Key: value` line with the key in title case. Keys with a leading
    underscore are reserved by the toolkit (e.g., `_polarity`) and are not rendered. User fields should
    not start with `_`.

    On chat input the block is combined with the leading system message according to `system_mode`.
    `"append"` places the block after the existing content, `"prepend"` places it before, and `"insert"`
    adds it as a separate second system message. Under `"append"` and `"prepend"` every chat input
    yields exactly one leading system message. Under `"insert"` a chat that already has a leading system
    message yields two, which some chat templates (Qwen3) reject. When the chat has no leading system
    message, all three modes insert one containing the block. On token input with a tokenizer that has
    a chat template, the prompt is decoded and wrapped as a user message, and the block is placed in a
    new system message before the chat template is applied. With a tokenizer that has no chat template,
    the block text is prepended to the token stream and `system_mode` does not apply. When there are no
    examples and no `directive`, the input is returned unchanged (and `adapt()` emits a `UserWarning`).

    The following `runtime_kwargs` are accepted:

    - `"positive_examples"`: A list of example dicts used as the positive examples for every prompt row
        in the call, in place of pool-based selection.
    - `"negative_examples"`: A list of example dicts used as the negative examples for every prompt row
        in the call, in place of pool-based selection.

    Args:
        directive: Text placed at the start of the block, before the examples.
        positive_example_pool: Positive examples to select from. Requires `k_positive`.
        negative_example_pool: Negative examples to select from. Requires `k_negative`.
        k_positive: Number of positive examples selected for each prompt.
        k_negative: Number of negative examples selected for each prompt.
        selector: Selector that picks examples from the pools, given as a `BaseSelector` instance, a
            registered name (e.g., `"random"`), or None for `RandomSelector`.
        selector_seed: Seed for pool selection when `selector` is a registered name or None. Each draw is
            seeded from `selector_seed`, the pool polarity, and the query content. The examples a query
            receives are therefore independent of call order. `steer()` raises `ValueError` when the
            selector has no `reseed()` method. Not accepted with a selector instance.
        system_mode: How the rendered block combines with an existing leading system message on chat input
            (`"append"` (default), `"prepend"`, or `"insert"`). Ignored when the chat has no leading system
            message.
        separator: String inserted between the existing system content and the block for `"append"` and
            `"prepend"`. Empty string allowed.
        formatter: Formatter that renders the block. When given, the formatter determines the placement,
            and `system_mode` and `separator` are not used.

    Attributes:
        pool: The `PoolMemory` of pool examples tagged by polarity, built by `steer()`.
        tokenizer: The tokenizer attached by `steer()`.
    """

    Args = FewShotArgs

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "positive_examples",
            "type": "list[dict]",
            "scope": "call",
            "help": "Positive examples for this call, applied to every prompt row; overrides pool-based selection.",
        },
        {
            "name": "negative_examples",
            "type": "list[dict]",
            "scope": "call",
            "help": "Negative examples for this call, applied to every prompt row; overrides pool-based selection.",
        },
    ]

    supports_batching: bool = True

    # placeholders (dataclass attrs from FewShotArgs override these at __init__ time)
    tokenizer: PreTrainedTokenizerBase | None = None
    directive: str | None = None
    positive_example_pool: Sequence[dict] | None = None
    negative_example_pool: Sequence[dict] | None = None
    k_positive: int | None = None
    k_negative: int | None = None
    system_mode: str = "append"
    separator: str = "\n\n"
    selector: Any = None  # str | BaseSelector | None — resolved in steer()
    selector_seed: int | None = None
    formatter: Any = None  # BaseFormatter | None — resolved in steer()

    # method-owned state populated in steer()
    pool: PoolMemory[dict] | None = None
    _selector: BaseSelector | None = None
    _formatter: FewShotBlockFormatter | None = None

    def steer(
            self,
            model=None,
            tokenizer: PreTrainedTokenizerBase | None = None,
            **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

        # build the example pool
        self.pool = PoolMemory[dict]()
        for example in self.positive_example_pool or []:
            self.pool.add(example, polarity="pos")
        for example in self.negative_example_pool or []:
            self.pool.add(example, polarity="neg")

        # resolve selector argument (instance | name | None) into a BaseSelector[dict]
        self._selector = selector_from_arg(self.selector)
        if self.selector_seed is not None and not callable(getattr(self._selector, "reseed", None)):
            raise ValueError(
                f"selector_seed requires a selector with a reseed() method; {type(self._selector).__name__} has none."
            )

        # selectors that need offline preparation (e.g. EPR) get a chance here
        prepare = getattr(self._selector, "prepare", None)
        if callable(prepare):
            prepare(model=model, tokenizer=tokenizer, data=self.pool)

        # formatter is shared between adapt and adapt_messages
        self._formatter = self.formatter or FewShotBlockFormatter(mode=self.system_mode, separator=self.separator)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Add few-shot examples to the model's prompt and return adapted token ids.

        Both the chat-template path and the no-template fallback route through the resolved
        `BaseFormatter`, which renders the example block from the pool.

        Assumes `input_ids` represents the user's prompt before any chat templating. Pre-templated
        input will be re-templated and produce malformed output; use `adapt_messages` for chat input.

        Args:
            input_ids: The user's prompt token IDs.
            runtime_kwargs: May contain `positive_examples` / `negative_examples` to override pool sampling.

        Returns:
            The transformed token IDs.

        Raises:
            RuntimeError: If tokenizer is not set (requires calling `steer()` first)

        Warnings:
            UserWarning: Issued when no examples are configured or none remain after selection.
        """
        if self.tokenizer is None:
            raise RuntimeError("FewShot needs a tokenizer; call .steer() first.")

        # infer mode from arguments
        using_runtime_examples = (
            runtime_kwargs
            and ("positive_examples" in runtime_kwargs or "negative_examples" in runtime_kwargs)
        )
        using_pool_mode = self.positive_example_pool is not None or self.negative_example_pool is not None

        using_directive = bool(self.directive)
        if not (using_runtime_examples or using_pool_mode or using_directive):
            warnings.warn(
                "FewShot: nothing to inject (no examples, no directive). Returning input unchanged.",
                UserWarning,
            )
            return input_ids

        # determine input format
        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        # normalize to 2D list format [batch_size, seq_len]
        if is_tensor:
            if input_ids.ndim == 1:
                batch_input_ids = [input_ids.tolist()]
                single_sequence = True
            else:
                batch_input_ids = input_ids.tolist()
                single_sequence = False
        else:
            if isinstance(input_ids[0], int):
                batch_input_ids = [input_ids]
                single_sequence = True
            else:
                batch_input_ids = input_ids
                single_sequence = False

        use_chat_template = has_chat_template(self.tokenizer)

        adapted_batch: list[list[int]] = []
        for input_ids_single in batch_input_ids:
            original_text = self.tokenizer.decode(input_ids_single, skip_special_tokens=True)

            # sample or gather examples independently per item
            if using_runtime_examples:
                examples = self._gather_runtime_examples(runtime_kwargs)
            else:
                examples = self._sample_from_pools(query=original_text)

            if not examples and not self.directive:
                warnings.warn(
                    "FewShot: nothing to inject for this item. Returning input unchanged.",
                    UserWarning,
                )
                adapted_batch.append(list(input_ids_single))
                continue

            slot_memory = TextMemory(slots={
                "examples": examples,
                "directive": self.directive or "",
            })

            if use_chat_template:
                chat = [{"role": "user", "content": original_text}]
                adapted_chat = self._formatter.apply_to_messages([chat], slot_memory)[0]
                rendered = render_messages(self.tokenizer, adapted_chat, add_generation_prompt=True)
                adapted_tokens = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            else:
                input_tensor = torch.tensor(input_ids_single, dtype=torch.long).unsqueeze(0)
                adapted_tensor = self._formatter.apply_to_ids(input_tensor, slot_memory, self.tokenizer)
                adapted_tokens = adapted_tensor[0].tolist()

            adapted_batch.append(adapted_tokens)

        # pad to uniform length for batched output
        max_len = max(len(seq) for seq in adapted_batch)
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError(
                "FewShot: tokenizer has no pad_token_id; cannot pad batch sequences. "
                "Set a pad token before using FewShot with batched inputs."
            )
        pad_id = self.tokenizer.pad_token_id

        padded_batch = [seq + [pad_id] * (max_len - len(seq)) for seq in adapted_batch]

        # convert back to original format
        if is_tensor:
            result = torch.tensor(padded_batch, dtype=original_dtype, device=original_device)
            if single_sequence:
                result = result.squeeze(0)
            return result
        else:
            if single_sequence:
                return padded_batch[0]
            return padded_batch

    def _select_with_polarity(self, polarity: str, k: int, query: Any = None) -> list[dict]:
        """Run the resolved selector against the pool subset matching `polarity`."""
        if self.pool is None or self._selector is None:
            return []
        polarities = self.pool.metadata.get("polarity", [])
        items = [item for item, pol in zip(self.pool.items, polarities) if pol == polarity]
        if not items or k <= 0:
            return []
        if self.selector_seed is not None:
            self._selector.reseed(self._selection_seed(polarity, query))
        return self._selector.select(items, query=query, k=k)

    def _selection_seed(self, polarity: str, query: Any) -> int:
        """Return the seed for one pool draw.

        The seed is the first 8 bytes, read as a big-endian integer, of a SHA-256 digest of
        `selector_seed`, the pool polarity, and the canonical JSON form of `query`.

        Args:
            polarity: The pool polarity, `"pos"` or `"neg"`.
            query: The query the draw is made for (the decoded prompt text or the chat messages).

        Returns:
            A non-negative integer below `2**64`.
        """
        payload = json.dumps(canonical_value(query), sort_keys=True)
        digest = hashlib.sha256(f"{self.selector_seed}:{polarity}:{payload}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def _sample_from_pools(self, query: Any = None) -> list[dict[str, Any]]:
        """Sample examples from the pools, attaching polarity labels for downstream formatting."""
        all_examples: list[dict[str, Any]] = []
        if self.positive_example_pool and self.k_positive and self.k_positive > 0:
            for example in self._select_with_polarity("pos", self.k_positive, query=query):
                all_examples.append({**example, "_polarity": "positive"})
        if self.negative_example_pool and self.k_negative and self.k_negative > 0:
            for example in self._select_with_polarity("neg", self.k_negative, query=query):
                all_examples.append({**example, "_polarity": "negative"})
        return all_examples

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        """Merge the directive and labeled example blocks into the leading system message of each chat.

        Placement follows `system_mode`. Under `"append"` and `"prepend"` each chat ends up with exactly one
        leading system message, while `"insert"` adds a separate second one. A chat with no leading system
        message gains one containing the block.

        Runtime examples (`positive_examples` / `negative_examples` in `runtime_kwargs`) take precedence
        over pool-based selection. If there are no examples and no directive, returns None (no change).
        """
        runtime_kwargs = runtime_kwargs or {}
        using_runtime = "positive_examples" in runtime_kwargs or "negative_examples" in runtime_kwargs
        using_pools = self.positive_example_pool is not None or self.negative_example_pool is not None
        if not (using_runtime or using_pools or self.directive):
            return None

        out: list[list[dict]] = []
        for chat in messages:
            if using_runtime:
                examples = self._gather_runtime_examples(runtime_kwargs)
            else:
                examples = self._sample_from_pools(query=chat)
            if not examples and not self.directive:
                out.append(list(chat))
                continue
            slot_memory = TextMemory(slots={
                "examples": examples,
                "directive": self.directive or "",
            })
            adapted_batch = self._formatter.apply_to_messages([chat], slot_memory)
            out.append(adapted_batch[0])
        return out

    @staticmethod
    def _gather_runtime_examples(runtime_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Gather examples from runtime_kwargs."""
        examples = []
        if "positive_examples" in runtime_kwargs:
            for example in runtime_kwargs["positive_examples"]:
                examples.append({**example, "_polarity": "positive"})
        if "negative_examples" in runtime_kwargs:
            for example in runtime_kwargs["negative_examples"]:
                examples.append({**example, "_polarity": "negative"})
        return examples
