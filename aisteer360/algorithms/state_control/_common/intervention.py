"""Intervention IR: the declarative form a state control emits from `plan()`.

A control describes *what* to steer as pure data — an `InterventionPlan` (a list of `Intervention`s
plus their optional `ConditionSpec` gating) — and the shared machinery compiles it: to in-process
hooks via `compile_plan_to_hooks` (`runtime.py`), or to the wire schema via the vLLM-Hook compiler
(doc 06). A plan is pure data plus component references (bound transforms, scorers): no model
references, no per-generation mutable state.

`ArtifactHandle` lives here rather than in `sources.py`: an `ArtifactSource` is a pre-binding fit
recipe, whereas an `ArtifactHandle` wraps an already-bound tensor awaiting wire encoding — different
lifecycle stages. Keeping the whole IR vocabulary in one module lets the wire compiler import it from
a single place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from aisteer360.algorithms.state_control._common.specs import Comparator, CompMode, HiddenStateLocation
from aisteer360.algorithms.state_control._common.token_scope import TokenScope

if TYPE_CHECKING:
    from aisteer360.algorithms.state_control._common.gates.base import BaseGate
    from aisteer360.algorithms.state_control._common.runtime import ConditionScorer
    from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform


@dataclass(slots=True, frozen=True)
class ArtifactHandle:
    """A placeholder for an already-bound tensor awaiting wire encoding (doc 06).

    Produced by a component's `export_payload` in place of a raw tensor so the wire compiler can
    encode it (inline base64 or content-hash ref) without the component knowing the transport.

    Attributes:
        tensor: The bound tensor to encode.
        role: A short label describing what the tensor is (e.g. `"direction"`, `"vector"`).
    """

    tensor: torch.Tensor
    role: str = "tensor"


@dataclass(slots=True)
class HookTarget:
    """One module to hook for an intervention.

    The compiler resolves `module` (a dotted submodule path) and uses `layer_id` for the global
    pass-opener ordering. Most interventions hook the decoder layer at `layer_id`; ITI hooks the
    `o_proj` submodule and AngularSteering hooks per-layer norm sub-modules, so the module path is
    resolved by the control and carried here explicitly.

    Attributes:
        module: The dotted submodule path to hook.
        layer_id: The layer index the module belongs to (indexes per-layer transform artifacts and
            orders the pass opener).
    """

    module: str
    layer_id: int


@dataclass(slots=True)
class ConditionSpec:
    """How condition hooks score the prompt and feed a gate.

    Two forms, distinguished by whether `threshold` is set:

    - **Threshold (portable)**: `threshold` and `comparator` are set. `compile_plan_to_hooks` builds
        the `CacheOnceGate(MultiKeyThresholdGate)` stack from these fields; this is the declarative,
        wire-portable form of standard gating (CAST).
    - **Custom gate (in-process only)**: `threshold` is `None`; the owning `Intervention` supplies a
        pre-built `gate`, and this spec only wires the `scorer` to that gate at the condition layers
        (ActivationAdapter with a user scorer and user gate).

    When set on an `Intervention` (and the intervention is not a follower), condition hooks are
    emitted at `targets` feeding the resolved gate. Neither `condition` nor `gate` set ⇒ ungated.

    Attributes:
        targets: The modules whose hidden states are scored (one per condition layer).
        scorer: The per-row condition scorer (exports via `export_payload` when portable).
        threshold: The gate threshold for the portable form, or `None` for the custom-gate form.
        comparator: The canonical comparator (`normalize_comparator` output), or `None`.
        comp_mode: Prompt-token aggregation for scoring (`"mean"` / `"last"`).
        cache: `"prompt_once"` (score the prompt once, freeze) or `"dynamic"` (rescore each pass).
        location: The residual-stream boundary the condition scores at.
        aggregate: How multiple condition layers combine (`"any"` / `"all"`).
    """

    targets: list[HookTarget]
    scorer: "ConditionScorer"
    threshold: float | None = None
    comparator: Comparator | None = None
    comp_mode: CompMode = "mean"
    cache: str = "prompt_once"
    location: HiddenStateLocation = "layer_input"
    aggregate: str = "any"

    @property
    def is_threshold(self) -> bool:
        """Whether this is the portable threshold form (a gate can be built from it)."""
        return self.threshold is not None and self.comparator is not None

    @property
    def layer_ids(self) -> list[int]:
        """The condition layer ids, in target order."""
        return [target.layer_id for target in self.targets]


@dataclass(slots=True)
class Intervention:
    """One behavior transform applied at a set of layers, optionally gated.

    A plan is a list of these, applied in list order (composition is non-commuting; order is the
    contract). Transforms must be **bound** before entering a plan.

    Attributes:
        targets: The modules the transform is applied at (one per behavior layer).
        hook_point: The residual-stream boundary (`"layer_output"` builds forward hooks;
            `"layer_input"` builds forward pre-hooks).
        transform: The bound transform to apply (carries its own artifact).
        scope: Which token positions to steer.
        scope_params: Extra scope parameters (`last_k`, `from_position`).
        gate: A custom (non-declarative) gate — the in-process-only path; `None` when using
            `condition` or when ungated.
        condition: The declarative gating spec; when set, the gate stack is built from it.
        gate_driven_externally: When `True`, this intervention reads a gate driven by another
            intervention/control (the driver/follower pattern) and contributes no condition hooks.
    """

    targets: list[HookTarget]
    hook_point: HiddenStateLocation
    transform: "BaseTransform"
    scope: TokenScope = "after_prompt"
    scope_params: dict = field(default_factory=dict)
    gate: "BaseGate | None" = None
    condition: ConditionSpec | None = None
    gate_driven_externally: bool = False

    @property
    def layer_ids(self) -> list[int]:
        """The behavior layer ids, in target order."""
        return [target.layer_id for target in self.targets]


# a plan is a list of interventions; application order == list order
InterventionPlan = list[Intervention]


@dataclass(slots=True)
class PromptContext:
    """Minimal per-generation prompt context passed to `plan()`.

    A small struct so `plan()` never needs the tokenizer at generation time — just the prompt ids,
    the pad-aware attention mask, per-row prompt lengths, and the pad token id.

    Attributes:
        input_ids: The prompt token ids `[B, T]`.
        attention_mask: The pad-aware attention mask `[B, T]`, or `None`.
        prompt_lens: Per-row prompt lengths `[B]`.
        pad_token_id: The tokenizer's pad token id, or `None`.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    prompt_lens: torch.Tensor
    pad_token_id: int | None

    @classmethod
    def from_ids(
        cls,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ) -> "PromptContext":
        """Build a `PromptContext` from prompt ids, coercing to 2-D and computing prompt lengths.

        Args:
            input_ids: Prompt token ids `[T]` or `[B, T]`.
            attention_mask: Optional matching attention mask.
            pad_token_id: The tokenizer's pad token id, or `None`.

        Returns:
            A populated `PromptContext`.
        """
        from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
        from aisteer360.utils.tokenization import infer_attention_mask_from_ids

        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if attention_mask is not None:
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        elif pad_token_id is not None:
            # infer a pad-aware mask from the ids (leading/trailing pad runs only) so condition
            # scorers align with real prompt positions without masking interior pad==eos tokens
            mask = infer_attention_mask_from_ids(ids, pad_token_id)
        else:
            mask = None
        prompt_lens = compute_prompt_lens(ids, pad_token_id)
        return cls(input_ids=ids, attention_mask=mask, prompt_lens=prompt_lens, pad_token_id=pad_token_id)
