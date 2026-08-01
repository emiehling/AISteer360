"""Normalized generation parameters with one rendering rule per backend family."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

NORMALIZED_PARAM_NAMES: tuple[str, ...] = (
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "greedy",
    "n",
    "repetition_penalty",
    "seed",
)


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """The sampling-facing subset of generation parameters, normalized across backends.

    Each backend family owns one rendering rule. In-process, the normalized fields render onto
    `model.generate` names and every key in `extra` passes through untouched. On API backends
    the normalized table is exhaustive and unmapped parameters raise, so `extra` is rejected
    there.

    Attributes:
        max_new_tokens: Maximum number of new tokens.
        min_new_tokens: Minimum number of new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        top_k: Top-k sampling cutoff.
        greedy: True forces greedy decoding, False forces sampling, None leaves the backend
            default.
        n: Number of returned candidates per prompt.
        repetition_penalty: Repetition penalty.
        seed: Sampling seed. In-process it renders as a `fork_rng`-scoped `manual_seed` around
            the item's decode; on vLLM it maps to the request seed.
        extra: Additional keyword arguments passed through unmapped on the in-process arm. A
            normalized field always takes precedence over a same-named key in `extra`.
    """

    max_new_tokens: int | None = None
    min_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    greedy: bool | None = None
    n: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_gen_kwargs(cls, **gen_kwargs: Any) -> "GenerationParams":
        """Split keyword arguments into normalized fields and pass-through extras.

        `do_sample` maps onto `greedy` (inverted) and `num_return_sequences` onto `n`; keys named
        exactly like a normalized field bind to it; everything else lands in `extra`.

        Args:
            **gen_kwargs: Generation keyword arguments in `model.generate` vocabulary.

        Returns:
            The normalized `GenerationParams`.
        """
        normalized: dict[str, Any] = {}
        if "do_sample" in gen_kwargs:
            normalized["greedy"] = not gen_kwargs.pop("do_sample")
        if "num_return_sequences" in gen_kwargs:
            normalized["n"] = gen_kwargs.pop("num_return_sequences")
        for name in NORMALIZED_PARAM_NAMES:
            if name in gen_kwargs:
                normalized[name] = gen_kwargs.pop(name)
        return cls(**normalized, extra=gen_kwargs)
