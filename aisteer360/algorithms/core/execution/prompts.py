"""One prompt in message, text, or token form, tokenized as late as possible."""
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPrompt:
    """A sum type over `messages | text | token_ids` for one prompt, plus metadata.

    Exactly one of `text`, `messages`, or `token_ids` is set at construction. Tokenization is
    forced only when a consumer needs token ids (`resolve_token_ids`); the in-process resolution
    reproduces the pipeline's tokenization calls, so resolved ids match the early-tokenized path.

    Attributes:
        text: A plain-text prompt, or None.
        messages: One conversation as a tuple of message mappings, or None.
        token_ids: Token ids of shape `[1, seq_len]`, or None until resolved.
        attention_mask: Attention mask matching `token_ids`, or None.
        is_single: Whether the originating call passed a single (non-batched) prompt.
        message_handled: `id()`s of input controls whose `adapt_messages` already performed the
            adaptation for this prompt.
    """

    text: str | None = None
    messages: tuple[Mapping, ...] | None = None
    token_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    is_single: bool = True
    message_handled: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        sources = [
            name for name, value in (
                ("text", self.text), ("messages", self.messages), ("token_ids", self.token_ids),
            ) if value is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                f"PreparedPrompt requires exactly one of text, messages, or token_ids; got "
                f"{', '.join(sources) or 'none'}."
            )

    @classmethod
    def from_text(cls, text: str) -> "PreparedPrompt":
        """Build a text-form prompt."""
        return cls(text=text)

    @classmethod
    def from_messages(cls, messages: list[Mapping] | tuple[Mapping, ...]) -> "PreparedPrompt":
        """Build a message-form prompt from one conversation."""
        return cls(messages=tuple(messages))

    @classmethod
    def from_token_ids(
        cls,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | None = None,
    ) -> "PreparedPrompt":
        """Build a token-form prompt from a 1-D or `[1, seq_len]` tensor or a `list[int]`.

        Raises:
            ValueError: If `token_ids` carries more than one row; a prompt is one row.
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.size(0) != 1:
            raise ValueError(
                f"A PreparedPrompt holds one prompt row; got shape {tuple(token_ids.shape)}."
            )
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return cls(token_ids=token_ids, attention_mask=attention_mask)

    def resolve_token_ids(self, tokenizer) -> "PreparedPrompt":
        """Return a token-form copy of this prompt, tokenizing text or messages when needed.

        Text prompts tokenize via `tokenizer(...)`; message prompts via
        `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Both match the
        pipeline's own tokenization calls. A prompt already in token form is returned unchanged.

        Args:
            tokenizer: The pipeline tokenizer.

        Returns:
            A `PreparedPrompt` with `token_ids` (and, when available, `attention_mask`) set.

        Raises:
            ValueError: If tokenization is required but `tokenizer` is None.
        """
        if self.token_ids is not None:
            return self
        if tokenizer is None:
            raise ValueError("A tokenizer is required to resolve this prompt to token ids.")

        if self.text is not None:
            encoded = tokenizer([self.text], return_tensors="pt", padding=True)
            return replace(
                self,
                text=None,
                token_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )

        encoded = tokenizer.apply_chat_template(
            [list(self.messages)],
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
        return replace(self, messages=None, token_ids=input_ids, attention_mask=attention_mask)
