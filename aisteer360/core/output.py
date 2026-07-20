"""The `Output` value type returned by generation across all backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class Output:
    """The result of one generation call.

    An `Output` is text-native: token ids are available on in-process backends but optional on API
    backends, where the server returns text directly. At least one of `output_ids` / `output_text`
    is always set (enforced in `__post_init__`), and every consumer routes through `decode`, which
    prefers text when present and otherwise decodes ids.

    Attributes:
        output_ids: Generated token IDs as a `[batch, seq]` tensor, excluding the prompt (the same
            slice the pipeline returns to the caller by default). `None` on API backends that report
            only text.
        output_text: Generated text, one string per batch row. Authoritative on API backends;
            `None` on in-process backends that report only ids.
        adapted_input_ids: The `input_ids` actually fed to the model after all input-control
            transformations. Useful for inspection/debugging (e.g., to see the steered prompt).
            `None` when not provided (e.g., chat-path introspection routes through
            `adapted_messages` instead, or the backend never tokenizes client-side).
        adapted_messages: The chat messages actually sent, for chat-path introspection on backends
            that never tokenize client-side. `None` otherwise.
        runtime_kwargs: The `runtime_kwargs` that produced this output. `None` if no runtime
            overrides were used.
        finish_reason: How generation ended; one of `"eos"`, `"length"`, `"stop"`, or `None` if not
            tracked.
        usage: Prompt/completion token counts when the backend reports them (e.g. API `usage`);
            `None` otherwise.
        metadata: Open-ended dict for backend-attached extras (resolved gen_kwargs, timing, choices,
            the originating backend kind). Keys are not part of the stable contract.
    """

    output_ids: torch.Tensor | None = None
    output_text: list[str] | None = None
    adapted_input_ids: torch.Tensor | None = None
    adapted_messages: list[list[dict]] | None = None
    runtime_kwargs: dict | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.output_ids is None and self.output_text is None:
            raise ValueError("Output requires at least one of `output_ids` or `output_text`.")

    def decode(
        self,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Return the generated text, one string per batch row.

        Prefers `output_text` when it is set (the authoritative form on API backends); otherwise
        decodes `output_ids`, which requires a tokenizer.

        Args:
            tokenizer: Tokenizer used to decode `output_ids`. Ignored when `output_text` is set;
                required otherwise.
            skip_special_tokens: Forwarded to `tokenizer.batch_decode` when decoding ids.

        Returns:
            One decoded string per batch row.

        Raises:
            ValueError: If `output_text` is unset and no tokenizer is supplied to decode ids.
        """
        if self.output_text is not None:
            return self.output_text
        if tokenizer is None:
            raise ValueError(
                "Output has no `output_text`; a tokenizer is required to decode `output_ids`."
            )
        return tokenizer.batch_decode(self.output_ids, skip_special_tokens=skip_special_tokens)

    def require_ids(self) -> torch.Tensor:
        """Return `output_ids`, raising a targeted error when they are unavailable.

        Returns:
            The `[batch, seq]` generated token ids.

        Raises:
            ValueError: If `output_ids` is `None` (the case on API backends), naming the originating
                backend from `metadata["backend"]` when present.
        """
        if self.output_ids is not None:
            return self.output_ids
        backend = (self.metadata or {}).get("backend", "the active backend")
        raise ValueError(
            f"Output has no token ids: {backend} returns text only. Use `decode()` / `output_text`, "
            "or run this pipeline on a backend that reports token ids (e.g. HuggingFaceBackend)."
        )
