"""Modality-tagged prompt value types consumed by the pipeline and backends.

`Prompt` formalizes the input classification the pipeline previously did inline
(`SteeringPipeline._classify_inputs`): it tags the caller's input by modality (chat / text / tensor)
and normalizes it to a batched form without tokenizing. `PreparedPrompt` is the post-input-control,
pre-backend form; tokenization happens late, at the backend boundary, rather than at the top of
`generate()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

PromptModality = Literal["chat", "text", "tensor"]


@dataclass(slots=True)
class Prompt:
    """A caller's prompt, tagged by modality and normalized to a batched form.

    Exactly one of `messages` / `texts` / `token_ids` is populated, selected by `modality`. The
    input is normalized to batched shape (`list[list[dict]]` for chat, `list[str]` for text, a
    2-D `[B, T]` tensor for tensor) while `is_single` records whether the caller passed a single
    (non-batched) input so the pipeline can shape the return symmetrically.

    Attributes:
        modality: One of `"chat"`, `"text"`, `"tensor"`.
        is_single: `True` when the caller passed a single (non-batched) input.
        messages: Batched chats (`list[list[dict]]`) when `modality == "chat"`, else `None`.
        texts: Batched prompt strings (`list[str]`) when `modality == "text"`, else `None`.
        token_ids: A 2-D `[B, T]` tensor when `modality == "tensor"`, else `None`.
        attention_mask: An optional `[B, T]` mask, meaningful only for the tensor modality.
    """

    modality: PromptModality
    is_single: bool
    messages: list[list[dict]] | None = None
    texts: list[str] | None = None
    token_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @classmethod
    def classify(cls, inputs: Any, attention_mask: torch.Tensor | None = None) -> "Prompt":
        """Classify the input modality and normalize to a batched `Prompt`.

        Reproduces the seven-shape dispatch of the pipeline's original `_classify_inputs`
        (`str` / `list[str]` / `list[dict]` / `list[list[dict]]` / tensor / `list[int]` /
        `list[list[int]]`), including its error cases.

        Args:
            inputs: One of the seven supported input shapes.
            attention_mask: Optional attention mask; retained only for the tensor modality (it is
                meaningless for chat/text, which are tokenized later at the backend).

        Returns:
            A `Prompt` with the input normalized to batched form.

        Raises:
            ValueError: If a tensor input is neither 1-D nor 2-D, or an input list is empty.
            TypeError: If the input type is unsupported.
        """
        if isinstance(inputs, str):
            return cls(modality="text", is_single=True, texts=[inputs])

        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 1:
                return cls(
                    modality="tensor",
                    is_single=True,
                    token_ids=inputs.unsqueeze(0),
                    attention_mask=attention_mask,
                )
            if inputs.ndim == 2:
                return cls(
                    modality="tensor",
                    is_single=False,
                    token_ids=inputs,
                    attention_mask=attention_mask,
                )
            raise ValueError(f"Tensor input must be 1-D or 2-D; got {inputs.ndim}-D.")

        if isinstance(inputs, list):
            if len(inputs) == 0:
                raise ValueError("Empty input list.")
            first = inputs[0]
            if isinstance(first, str):
                return cls(modality="text", is_single=False, texts=list(inputs))
            if isinstance(first, dict):
                # one chat (list of messages)
                return cls(modality="chat", is_single=True, messages=[list(inputs)])
            if isinstance(first, list) and first and isinstance(first[0], dict):
                # batch of chats (list of list of messages)
                return cls(modality="chat", is_single=False, messages=[list(chat) for chat in inputs])
            if isinstance(first, int):
                # 1-D token id list
                return cls(
                    modality="tensor",
                    is_single=True,
                    token_ids=torch.tensor([list(inputs)], dtype=torch.long),
                    attention_mask=attention_mask,
                )
            if isinstance(first, list) and first and isinstance(first[0], int):
                # 2-D token id list-of-lists
                return cls(
                    modality="tensor",
                    is_single=False,
                    token_ids=torch.tensor([list(seq) for seq in inputs], dtype=torch.long),
                    attention_mask=attention_mask,
                )

        raise TypeError(f"Unsupported input type: {type(inputs).__name__}.")

    @property
    def batch_size(self) -> int:
        """Number of prompts in the batch."""
        if self.modality == "chat":
            return len(self.messages)
        if self.modality == "text":
            return len(self.texts)
        return int(self.token_ids.size(0))


@dataclass(slots=True)
class PreparedPrompt:
    """Post-input-control, pre-backend prompt form.

    The pipeline applies input-control adaptation and records the richest representation the chosen
    adaptation level produced. Exactly one adapted representation is set, indicated by
    `adaptation_level`:

    - `"messages"`: `adapted_messages` is set (message-level adaptation on chat input).
    - `"tokens"`: `adapted_token_ids` (and optionally `adapted_attention_mask`) is set
        (token-level adaptation, or a raw tensor prompt).
    - `"none"`: no adaptation ran; the backend consumes `prompt` directly.

    Tokenization for the `"messages"` and text/chat `"none"` levels is deferred to the backend.

    Invariant (enforced in `__post_init__`): when `adapted_attention_mask` is set, its shape matches
    `adapted_token_ids`. The pipeline drops a mask invalidated by a length-changing `adapt()` at
    construction rather than passing it through; backends infer a mask whenever it is `None`
    (pad-token-based, else all ones).

    Attributes:
        prompt: The classified, unadapted `Prompt`.
        adapted_messages: Message-level adapted chats, or `None`.
        adapted_texts: Text-level adapted prompt strings, or `None`.
        adapted_token_ids: Token-level adapted ids as a `[B, T]` tensor, or `None`.
        adapted_attention_mask: A `[B, T]` mask matching `adapted_token_ids`, or `None`.
        adaptation_level: Which representation is authoritative (`"messages"`, `"tokens"`, `"none"`).
    """

    prompt: Prompt
    adapted_messages: list[list[dict]] | None = None
    adapted_texts: list[str] | None = None
    adapted_token_ids: torch.Tensor | None = None
    adapted_attention_mask: torch.Tensor | None = None
    adaptation_level: Literal["messages", "tokens", "none"] = "none"

    def __post_init__(self) -> None:
        if self.adapted_attention_mask is not None:
            if self.adapted_token_ids is None:
                raise ValueError("adapted_attention_mask is set but adapted_token_ids is None.")
            if self.adapted_attention_mask.shape != self.adapted_token_ids.shape:
                raise ValueError(
                    f"adapted_attention_mask shape {tuple(self.adapted_attention_mask.shape)} does "
                    f"not match adapted_token_ids shape {tuple(self.adapted_token_ids.shape)}."
                )

    @property
    def modality(self) -> PromptModality:
        """The underlying prompt modality."""
        return self.prompt.modality

    @property
    def is_single(self) -> bool:
        """Whether the caller passed a single (non-batched) input."""
        return self.prompt.is_single

    @property
    def batch_size(self) -> int:
        """Number of prompts in the batch."""
        return self.prompt.batch_size
