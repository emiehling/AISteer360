"""Shared specification dataclasses for state control components, and the intervention IR.

The intervention IR (`TokenScope`, `Condition`, `Intervention`) is the single declarative
statement of a residual-stream state control's behavior. Both compilers read it: `build_hooks`
turns a bound intervention tuple into torch hooks for one generation, and `lower_interventions`
turns it into an `InterventionSpec` for intervention-capable backends. Components describe
their own wire form (`WireForm` via each component's `export`), so no layer of the system
re-derives another layer's configuration by introspection.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, Protocol, Sequence, runtime_checkable

import torch

from aisteer360.algorithms.core.execution.capabilities import InterventionKinds
from aisteer360.algorithms.core.internals.capture import HiddenStateLocation
from aisteer360.utils.rendering import PromptFormat

if TYPE_CHECKING:
    from aisteer360.algorithms.core.execution.interventions import InterventionSpec

    from .condition_scorers import ConditionScorer
    from .gates.base import BaseGate
    from .selectors.base import BaseSelector
    from .transforms.base import BaseTransform

Comparator = Literal["larger", "smaller"]
ComparatorInput = Literal["larger", "smaller", "score_above", "score_below"]
CompMode = Literal["mean", "last"]

_COMPARATOR_ALIASES: dict[str, Comparator] = {
    "larger": "larger", "score_above": "larger",
    "smaller": "smaller", "score_below": "smaller",
}


def normalize_comparator(value: str) -> Comparator:
    """Map user-facing comparator names to the canonical internal values.

    Canonical semantics in this toolkit: "larger" opens the gate when score >= threshold, and
    "smaller" opens it when score <= threshold.

    This convention is inverted relative to the CAST reference implementation
    (github.com/IBM/activation-steering), where "larger" means the threshold is larger and fires
    when similarity < threshold. Settings copied from the paper or reference repo must flip the
    comparator. Prefer the unambiguous aliases "score_above" / "score_below".

    Args:
        value: One of "larger", "smaller", "score_above", "score_below".

    Returns:
        The canonical comparator ("larger" or "smaller").

    Raises:
        ValueError: If `value` is not a recognized comparator name.
    """
    try:
        return _COMPARATOR_ALIASES[value]
    except KeyError:
        raise ValueError(
            f"Unknown comparator {value!r}; expected one of {sorted(_COMPARATOR_ALIASES)}."
        ) from None


@dataclass(frozen=True)
class VectorTrainSpec:
    """Configuration for how to train/extract direction vectors.

    Attributes:
        method: Extraction algorithm.
            "pca_pairwise" uses PCA on paired differences of hidden states.
            "pca_center" uses PCA on all positive/negative hidden states centered
                by their grand mean (the CAST extraction from the paper).
            "mean_diff" uses the mean difference of hidden states (CAA method).
        accumulate: How to select hidden state spans for aggregation.
            "all" uses the full sequence.
            "suffix-only" uses only the portion after the shared prompt.
            "last_token" uses only the final non-pad token position.
        batch_size: Batch size for hidden state extraction forward passes.
        prompt_format: How to render contrastive examples into model-ready text
            (via `render_for_model`); the rendered string is tokenized with
            `add_special_tokens=False`.
            "chat_completion" renders `prompts` as user turns and appends
            positives/negatives as completions (prompt+answer pairs, e.g. CAA);
            falls back to "raw" when no `prompts` are provided.
            "chat_prompt" renders each positive/negative as a standalone user turn
            (standalone-prompt contrasts, e.g. the CAST condition); matches the
            inference rendering exactly.
            "raw" concatenates `prompts` + text verbatim with no chat template
            (base-model methods and standalone statements).
        location: Residual-stream boundary each layer key maps to. `outputs.hidden_states` is a
            tuple of `num_layers + 1` tensors: index 0 is the embedding output (the input to layer
            0) and index `i` is the output of layer `i - 1`.
            "layer_output" (default): key `l` maps to the output of layer `l`
            (`hidden_states[l + 1]`), the boundary hooked by controls that intervene on the layer
            output.
            "layer_input": key `l` maps to the input of layer `l`, i.e. the output of layer `l - 1`
            (`hidden_states[l]`), the boundary observed by layer pre-hooks.
            A vector fit at one boundary is a distinct artifact from one fit at the other, so fit it
            at the boundary where the consuming control scores or applies it.
    """

    method: Literal["pca_pairwise", "pca_center", "mean_diff"] = "pca_pairwise"
    accumulate: Literal["all", "suffix-only", "last_token"] = "all"
    batch_size: int = 8
    prompt_format: PromptFormat = "chat_completion"
    location: HiddenStateLocation = "layer_output"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.prompt_format not in ("raw", "chat_completion", "chat_prompt"):
            raise ValueError(
                f"prompt_format must be one of raw/chat_completion/chat_prompt, got {self.prompt_format!r}."
            )
        if self.location not in ("layer_output", "layer_input"):
            raise ValueError(
                f"location must be 'layer_output' or 'layer_input', got {self.location!r}."
            )


@dataclass(frozen=True)
class ConditionSearchSpec:
    """Configuration for automatic condition point search.

    Attributes:
        auto_find: If True, run the search during steer(). If False, the
            user must provide condition_layer_ids and threshold manually.
        candidate_layers: Explicit layer ids to search over. If None, use
            layer_range.
        layer_range: 0-based (start, end) half-open range of layers to consider. Ignored if
            candidate_layers is set. Defaults to all layers.
        threshold_range: (min, max) for the threshold grid search (half-open, step-exact).
        threshold_step: Step size for the threshold grid.
    """

    auto_find: bool = True
    candidate_layers: Sequence[int] | None = None
    layer_range: tuple[int, int] | None = None
    threshold_range: tuple[float, float] = (0.0, 1.0)
    threshold_step: float = 0.01

    def __post_init__(self):
        lo, hi = self.threshold_range
        if lo >= hi:
            raise ValueError(f"threshold_range ({lo}, {hi}): min must be < max.")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be > 0.")


Boundary = Literal["layer_output", "layer_input"]
Site = Literal["decoder_layer", "o_proj", "norm_input"]
ScopeKindLiteral = Literal["all", "after_prompt", "last_k", "from_position"]


@dataclass(frozen=True, slots=True)
class WireForm:
    """One component's form on the wire: the kind name, scalar params, and named tensors.

    `params` follow the plugin's `KIND_PARAMS` table for the kind; `tensors` follow its
    `ARTIFACT_TENSORS` table.

    Attributes:
        kind: The permanent wire kind name.
        params: Scalar parameters, inlined next to `kind` on the wire.
        tensors: Named tensor payloads, materialized as one content-addressed artifact.
    """

    kind: str
    params: Mapping[str, float | int | bool | str | tuple[int, ...] | list[int]] = field(default_factory=dict)
    tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenScope:
    """A token-position selector with its parameters.

    Attributes:
        kind: One of `"all"`, `"after_prompt"`, `"last_k"`, `"from_position"`.
        last_k: Number of trailing positions, required when `kind == "last_k"`.
        from_position: Absolute start position (inclusive), required when
            `kind == "from_position"`.
    """

    kind: ScopeKindLiteral
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        if self.kind not in ("all", "after_prompt", "last_k", "from_position"):
            raise ValueError(f"Unknown token scope kind {self.kind!r}.")
        if self.kind == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when kind is 'last_k'.")
        if self.kind == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when kind is 'from_position'.")

    def export(self) -> WireForm:
        """The scope's wire form. Total, since every scope kind is a wire kind."""
        if self.kind == "last_k":
            return WireForm(kind="last_k", params={"k": int(self.last_k)})
        if self.kind == "from_position":
            return WireForm(kind="from_position", params={"position": int(self.from_position)})
        return WireForm(kind=self.kind)


@dataclass(frozen=True, slots=True)
class Condition:
    """Where a gated intervention reads evidence and how the evidence is scored.

    In process, the scorer runs in condition hooks at `layer_ids`; on the wire, its
    exportable content merges into the gate's wire form.

    Attributes:
        layer_ids: Condition layers (0-based decoder-layer indices at the intervention's
            boundary).
        scorer: Per-row condition scorer feeding the intervention's gate.
    """

    layer_ids: tuple[int, ...]
    scorer: "ConditionScorer"

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", tuple(int(lid) for lid in self.layer_ids))
        if not self.layer_ids:
            raise ValueError("Condition requires at least one condition layer.")


@runtime_checkable
class GateConditionSource(Protocol):
    """A recipe that resolves to a gate (and optionally a condition) for a given model.

    Occupies an `Intervention`'s gate slot (and, when it also produces a condition, its
    condition slot as the same object). `Intervention.bind` resolves it once. The declared
    wire gate kinds are a class-level fact so `Intervention.wire_kinds()` can run before
    binding; None marks the resolved gating hook-only.
    """

    wire_gate_kinds: ClassVar[frozenset[str] | None]

    def resolve_gate_condition(
        self, model, tokenizer, *, layout=None, session=None
    ) -> tuple["BaseGate", Condition | None]:
        """Return the resolved gate and condition (None when unconditional)."""
        ...


class CoveredLayers:
    """Layer selector resolving to the bound transform's covered layers.

    Used when the behavior layers are a fact of the artifact rather than of the model, e.g. a
    steering plane supplied for a subset of layers. `Intervention.bind` binds the transform
    first and takes its `covered_layer_ids` (intersected with the model's layer range, and
    with `within` when given) as the behavior layers, raising when none remain.

    Args:
        within: Optional base selection the covered layers are intersected with, as explicit
            layer ids or a selector resolved against the model's layer count.
    """

    def __init__(self, within: "Sequence[int] | BaseSelector | None" = None):
        self.within = tuple(int(lid) for lid in within) if isinstance(within, (list, tuple)) else within

    def resolve(self, covered, num_layers: int) -> tuple[int, ...]:
        """The covered layers intersected with the model range and the base selection.

        Raises:
            ValueError: If no layer survives the intersection.
        """
        layer_ids = {int(lid) for lid in covered if 0 <= int(lid) < num_layers}
        if self.within is not None:
            if isinstance(self.within, tuple):
                requested = set(self.within)
            else:
                selected = self.within.select(num_layers=num_layers)
                requested = (
                    {int(lid) for lid in selected}
                    if isinstance(selected, (list, tuple, set, frozenset))
                    else {int(selected)}
                )
            layer_ids &= requested
            if not layer_ids:
                raise ValueError(
                    f"No target layer has a direction in the steering artifact "
                    f"(requested {sorted(requested)}, available {sorted(int(lid) for lid in covered)})."
                )
        if not layer_ids:
            raise ValueError("No active layers for this intervention after filtering.")
        return tuple(sorted(layer_ids))


def _default_gate() -> "BaseGate":
    from .gates.base import AlwaysOpenGate

    return AlwaysOpenGate()


def _default_scope() -> TokenScope:
    return TokenScope("after_prompt")


@dataclass(frozen=True, slots=True)
class Intervention:
    """One activation edit: apply `transform` at `layers`, at `scope` positions, on the
    `boundary` side of the layer, whenever `gate` is open.

    Declared unbound at control construction: `layers` may be a layer selector, the transform
    may carry an `ArtifactSource` (or be a factory over a `TransformContext`), and the gate or
    condition may be given as a `GateConditionSource`. `bind(model, tokenizer, layout=...)`
    returns the resolved form with layer coverage validated. Kind identity (`wire_kinds`) is
    readable on the unbound form, which is what lets `check()` run before `steer()`.

    Interventions are generation-invariant: prompt lengths, pad masks, and position offsets
    are runtime facts consumed by `build_hooks` in process and resolved per request by the
    worker on the wire. Nothing prompt-dependent appears here.

    IR dataclasses never use instance defaults for object-valued fields, since a shared
    default gate would carry sized state across every intervention in the process;
    object-valued defaults use `default_factory` only.

    Attributes:
        layers: Behavior layers (0-based decoder-layer indices), a selector resolved at bind
            time, or `CoveredLayers` to take the bound transform's covered layers.
        transform: The transform applied at masked positions of open rows.
        scope: Token positions to steer.
        gate: Per-row gate consulted at apply time, or a source resolving to one.
        condition: Where and how gate evidence is computed, or None for unconditional gates.
            When the gate slot holds a `GateConditionSource` producing a condition, this slot
            holds the same source object or None.
        boundary: Which side of the hooked module the edit applies at. `"layer_output"`
            builds forward hooks; `"layer_input"` builds forward pre-hooks.
        site: The hooked module family. None derives it from the transform kind
            (`head_additive` targets the attention output projection, everything else the
            decoder layer); `"norm_input"` targets each layer's normalization sub-modules
            and has no wire form.
        require_coverage: When True (default), `bind` raises if the resolved transform lacks
            a direction for any behavior layer; when False, uncovered layers are hooked and
            pass through unchanged.
    """

    layers: tuple[int, ...] | "BaseSelector" | CoveredLayers
    transform: "BaseTransform"
    scope: TokenScope = field(default_factory=_default_scope)
    gate: "BaseGate | GateConditionSource" = field(default_factory=_default_gate)
    condition: "Condition | GateConditionSource | None" = None
    boundary: Boundary = "layer_output"
    site: Site | None = None
    require_coverage: bool = True

    def __post_init__(self):
        if self.boundary not in ("layer_output", "layer_input"):
            raise ValueError(f"boundary must be 'layer_output' or 'layer_input'; got {self.boundary!r}.")
        if self.site not in (None, "decoder_layer", "o_proj", "norm_input"):
            raise ValueError(f"Unknown site {self.site!r}.")
        if isinstance(self.layers, (list, tuple)):
            object.__setattr__(self, "layers", tuple(int(lid) for lid in self.layers))

    @property
    def is_unbound(self) -> bool:
        """True when binding must run model-side work: a layer selector to resolve, a
        transform source or factory to fit, or a gate/condition source to search."""
        from .transforms.base import BaseTransform

        if not isinstance(self.layers, tuple):
            return True
        if not isinstance(self.transform, BaseTransform) or not self.transform.is_bound:
            return True
        if isinstance(self.gate, GateConditionSource) and not _is_gate(self.gate):
            return True
        return False

    def resolved_site(self) -> Site:
        """The module family this intervention hooks, deriving None from the transform kind."""
        if self.site is not None:
            return self.site
        from .transforms.base import BaseTransform, unwrap_modifiers

        if isinstance(self.transform, BaseTransform):
            core, _ = unwrap_modifiers(self.transform)
            if type(core).wire_kind == "head_additive":
                return "o_proj"
        return "decoder_layer"

    def bind(self, model, tokenizer, *, layout=None, session=None) -> "Intervention":
        """Resolve every declared element against `model` (or a session `layout`).

        Resolves the layer selector, binds the transform (fitting artifact sources and
        invoking factories), resolves gate/condition sources, validates layer coverage and
        scorer compatibility, and returns the bound intervention. Never mutates `self`.

        Args:
            model: The live model, or None for concrete-artifact configurations bound
                against a session layout.
            tokenizer: Tokenizer used when fitting sources.
            layout: Structural facts (`ModelLayout`) used when `model` is None.
            session: Optional `SteeringSession` forwarded to sources for capture-backed
                fitting and searching.

        Returns:
            The bound intervention.

        Raises:
            ValueError: If a layer is out of range, the transform lacks coverage for a
                behavior layer, or a condition scorer is incompatible with the boundary or
                model.
        """
        from .layout_facts import resolve_layout
        from .transforms.context import resolve_transform_slot

        layout = layout if layout is not None else resolve_layout(model, session)
        num_layers = layout.num_layers

        transform = self.transform
        if isinstance(self.layers, CoveredLayers):
            transform = resolve_transform_slot(
                transform, model, tokenizer, [], layout=layout,
                require_coverage=False, session=session,
            )
            covered = transform.covered_layer_ids
            if not covered:
                raise ValueError("No active layers for this intervention after filtering.")
            layer_ids = self.layers.resolve(covered, num_layers)
        elif isinstance(self.layers, tuple):
            layer_ids = self.layers
        else:
            selected = self.layers.select(num_layers=num_layers)
            if isinstance(selected, (list, tuple, set, frozenset)):
                layer_ids = tuple(sorted(int(lid) for lid in selected))
            else:
                layer_ids = (int(selected),)
        for lid in layer_ids:
            if not 0 <= lid < num_layers:
                raise ValueError(f"layer_id {lid} out of range [0, {num_layers}).")

        gate = self.gate
        condition = self.condition
        if isinstance(gate, GateConditionSource) and not _is_gate(gate):
            if condition is not None and condition is not gate:
                raise ValueError(
                    "When the gate slot holds a GateConditionSource, the condition slot must "
                    "be None or the same source object."
                )
            gate, condition = gate.resolve_gate_condition(
                model, tokenizer, layout=layout, session=session,
            )
        if condition is not None and not isinstance(condition, Condition):
            raise ValueError(
                f"condition must resolve to a Condition or None; got {type(condition).__name__}."
            )
        if condition is not None:
            for lid in condition.layer_ids:
                if not 0 <= lid < num_layers:
                    raise ValueError(f"condition_layer_id {lid} out of range [0, {num_layers}).")
            self._validate_scorer(condition.scorer, layout)

        if not isinstance(self.layers, CoveredLayers):
            transform = resolve_transform_slot(
                transform, model, tokenizer, list(layer_ids), layout=layout,
                require_coverage=self.require_coverage, session=session,
            )

        bound = replace(
            self, layers=layer_ids, transform=transform, gate=gate, condition=condition,
        )
        unbound_kinds = self.wire_kinds()
        bound_kinds = bound.wire_kinds()
        # binding may replace parameter values and tensors, never kinds; narrowing to None is
        # the artifact-dependent case caught by eager steer-time lowering
        assert unbound_kinds is None or bound_kinds is None or bound_kinds == unbound_kinds, (
            f"binding changed wire kinds from {unbound_kinds} to {bound_kinds}"
        )
        return bound

    def _validate_scorer(self, scorer, layout) -> None:
        """Check an optional scorer's declared boundary and model identity against this
        intervention."""
        scorer_location = getattr(scorer, "location", None)
        if scorer_location is not None and scorer_location != self.boundary:
            raise ValueError(
                f"Condition scorer expects features at '{scorer_location}' but this "
                f"intervention hooks '{self.boundary}'. Declare the intervention with "
                f"boundary='{scorer_location}', or refit the probe with "
                f"location='{self.boundary}'."
            )
        scorer_fingerprint = getattr(scorer, "model_fingerprint", None)
        if scorer_fingerprint is not None and layout is not None:
            live_fingerprint = getattr(layout, "model_fingerprint", None)
            if live_fingerprint is not None and scorer_fingerprint != live_fingerprint:
                raise ValueError(
                    f"Condition scorer was fitted on a different model (fingerprint "
                    f"{scorer_fingerprint!r} vs {live_fingerprint!r}). Refit the probe on "
                    "this model, or disarm the check with allow_model_mismatch=True on "
                    "probe_condition() or Probe.as_condition()."
                )

    def wire_kinds(self) -> InterventionKinds | None:
        """The wire kind names this configuration lowers to, or None when hook-only.

        Readable on the unbound form: sources and components declare kind identity at
        construction, so `check()` consults this before `steer()`. Artifact-dependent
        inexpressibility (e.g. a positional direction behind a broadcast-declared source) is
        undetectable here and is caught by eager steer-time lowering.
        """
        from .transforms.base import BaseTransform, unwrap_modifiers

        if self.resolved_site() == "norm_input":
            return None
        if not isinstance(self.transform, BaseTransform):
            return None  # a factory slot is unknown before binding
        core, wrappers = unwrap_modifiers(self.transform)
        kind = core.wire_plan()
        if kind is None:
            return None
        modifiers: set[str] = set()
        for wrapper in wrappers:
            modifier_kind = wrapper.modifier_wire_kind(kind)
            if modifier_kind is None:
                return None
            modifiers.add(modifier_kind)
        if self.boundary == "layer_input" and isinstance(self.layers, tuple) and 0 in self.layers:
            return None  # layer 0 input edits precede the first wire boundary
        gates = _gate_wire_kinds(self.gate, self.condition)
        if gates is None:
            return None
        return InterventionKinds(
            transforms=frozenset({kind}),
            modifiers=frozenset(modifiers),
            scopes=frozenset({self.scope.kind}),
            gates=gates,
        )


def _is_gate(obj) -> bool:
    from .gates.base import BaseGate

    return isinstance(obj, BaseGate)


def _gate_wire_kinds(gate, condition) -> frozenset[str] | None:
    """Wire gate kinds for a gate/condition pair; None marks the gating hook-only.

    Probe-backed gating is the only conditional configuration with a wire form: the gate must
    be a `ProbeSumGate` (bare or `cache_once`-wrapped) and the condition's scorer must be the
    `ProbeContributionScorer` over the same probe with condition layers matching the probe's
    layers, since the wire gate computes the scorer's affine evidence from the probe weights
    itself. A bare probe gate still plans `cache_once`, the wire form of the
    prompt-scored-once convention.
    """
    from .condition_scorers import ProbeContributionScorer
    from .gates.base import AlwaysOpenGate, BaseGate
    from .gates.cache_once import CacheOnceGate
    from .gates.probe_sum import ProbeSumGate

    if not isinstance(gate, BaseGate):
        return getattr(type(gate), "wire_gate_kinds", None)
    if isinstance(gate, AlwaysOpenGate):
        return frozenset()
    inner = gate.inner if isinstance(gate, CacheOnceGate) else gate
    if not isinstance(inner, ProbeSumGate):
        return None
    if condition is None:
        return None
    scorer = condition.scorer
    if not isinstance(scorer, ProbeContributionScorer):
        return None
    if scorer.probe is not inner.probe:
        return None
    if set(condition.layer_ids) != set(inner.probe.layer_ids):
        return None
    return frozenset({"cache_once", "probe_sum"})


def combine_kinds(kind_sets) -> InterventionKinds | None:
    """Union `InterventionKinds` across an iterable, propagating None (hook-only)."""
    transforms: set[str] = set()
    modifiers: set[str] = set()
    scopes: set[str] = set()
    gates: set[str] = set()
    empty = True
    for kinds in kind_sets:
        if kinds is None:
            return None
        empty = False
        transforms |= kinds.transforms
        modifiers |= kinds.modifiers
        scopes |= kinds.scopes
        gates |= kinds.gates
    if empty:
        return InterventionKinds()
    return InterventionKinds(
        transforms=frozenset(transforms),
        modifiers=frozenset(modifiers),
        scopes=frozenset(scopes),
        gates=frozenset(gates),
    )


def artifact_id_for(tensors: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, torch.Tensor]]:
    """The content-addressed artifact id and prepared tensors for a tensor payload.

    Tensors are prepared as float32, contiguous, CPU copies (cloned before the cast, so the
    live steering artifacts are never mutated or aliased), and the id is the SHA-256 over the
    safetensors serialization with sorted tensor names, matching the plugin registry's `write`
    byte-for-byte. Identical logical content therefore yields identical ids regardless of the
    producing device or dtype.

    Args:
        tensors: Mapping from tensor name to tensor.

    Returns:
        The `sha256:<hex>` id and the prepared name-to-tensor mapping.
    """
    import safetensors.torch

    prepared = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        for name, tensor in tensors.items()
    }
    data = safetensors.torch.save({name: prepared[name] for name in sorted(prepared)})
    return "sha256:" + hashlib.sha256(data).hexdigest(), prepared


def _map_behavior_layer(layer_id: int, boundary: Boundary, site: Site, num_layers: int) -> int | None:
    """Map a toolkit behavior layer onto its wire layer index, or None when unmappable.

    A wire op applies at the residual-stream boundary after decoder layer `N`. A
    `"layer_output"` hook at layer `l` is wire layer `l`; a `"layer_input"` hook at layer `l`
    is wire layer `l - 1` (layer 0 has no wire form); the `"o_proj"` site keeps its layer
    index, matching the wire `head_additive` placement.
    """
    if site == "o_proj":
        mapped = layer_id
    elif boundary == "layer_input":
        mapped = layer_id - 1
    else:
        mapped = layer_id
    return mapped if 0 <= mapped < num_layers else None


def _map_condition_layers(
    layer_ids: Sequence[int], boundary: Boundary, num_layers: int
) -> list[int] | None:
    """Map toolkit condition layers onto wire layer indices, or None when unmappable.

    A wire gate's condition layers read the materialized input of decoder layer `N`, so
    layers read at `"layer_output"` shift to `l + 1`.
    """
    offset = 1 if boundary == "layer_output" else 0
    mapped = [int(layer_id) + offset for layer_id in layer_ids]
    if all(0 <= layer_id < num_layers for layer_id in mapped):
        return mapped
    return None


def _merge_gate_condition(
    gate,
    condition: Condition | None,
    boundary: Boundary,
    num_layers: int,
    register,
) -> dict[str, Any] | None | type(...):
    """The wire gate for a gate/condition pair, folding the toolkit's gate/scorer/condition
    split into the wire `GateSpec`.

    The wire gate's params are the gate's exported params plus `condition_layers` from
    `condition.layer_ids` plus the scorer form's params; the wire gate's artifact is the
    gate's exported tensors if any, else the scorer form's tensors. Both sides exporting
    tensors, or exporting conflicting param values, is a compile error. Returns the Ellipsis
    sentinel for an ungated op (always-open), None when the configuration has no wire form.
    """
    from .gates.base import AlwaysOpenGate, BaseGate
    from .gates.cache_once import CacheOnceGate

    if gate is None or not isinstance(gate, BaseGate):
        return None
    if isinstance(gate, AlwaysOpenGate):
        return ...
    if isinstance(gate, CacheOnceGate):
        inner = _merge_gate_condition(gate.inner, condition, boundary, num_layers, register)
        if inner is None or inner is ...:
            return None
        return {"kind": "cache_once", "inner": inner}

    form = gate.export()
    if form is None:
        return None
    if form.kind == "null":
        return ...
    if condition is None:
        return None  # a conditional wire gate reads evidence at declared condition layers
    from .gates.probe_sum import ProbeSumGate

    if isinstance(gate, ProbeSumGate) and tuple(condition.layer_ids) != tuple(gate.probe.layer_ids):
        raise ValueError(
            "Condition layers must match the probe's layer order exactly; the wire gate's "
            f"weight rows align with condition_layers. Got {tuple(condition.layer_ids)} vs "
            f"probe layers {tuple(gate.probe.layer_ids)}."
        )
    params = dict(form.params)
    tensors = dict(form.tensors)
    if condition is not None:
        mapped = _map_condition_layers(condition.layer_ids, boundary, num_layers)
        if mapped is None:
            return None
        params["condition_layers"] = mapped
        scorer_export = getattr(condition.scorer, "export", None)
        scorer_form = scorer_export() if callable(scorer_export) else None
        if scorer_form is None:
            return None
        for name, value in scorer_form.params.items():
            if name in params and params[name] != value:
                raise ValueError(
                    f"Gate and scorer disagree on wire param {name!r}: "
                    f"{params[name]!r} vs {value!r}."
                )
            params[name] = value
        if scorer_form.tensors:
            if tensors:
                raise ValueError(
                    "Both the gate and the condition scorer export tensors; exactly one may "
                    "own the wire artifact."
                )
            tensors = dict(scorer_form.tensors)
    wire: dict[str, Any] = {"kind": form.kind, **params}
    if tensors:
        wire["artifact"] = register(tensors)
    return wire


def lower_interventions(
    interventions: Sequence[Intervention],
    *,
    num_layers: int,
    allowed_gates: frozenset[str] | None = None,
) -> "InterventionSpec | None":
    """Lower bound interventions to an `InterventionSpec`, or None when any element has no
    wire form.

    Folds each component's `export`, `unwrap_modifiers`, the scope export, and the
    gate/condition merge. One wire op is emitted per (intervention, layer), in intervention
    order then ascending layer order; artifact ids are content hashes, so layers sharing a
    tensor share one artifact. Bare probe gates are wrapped in `cache_once`, the wire form of
    the prompt-scored-once convention. The assembled spec is pre-flight validated with the
    plugin's `parse_intervention_spec`, so a malformed spec fails here with the same `E_*`
    code and JSON path the server would return.

    Args:
        interventions: Bound interventions, in application order.
        num_layers: Decoder layer count from the model layout.
        allowed_gates: Gate kinds negotiated with the serving backend; defaults to the full
            wire gate table.

    Returns:
        The validated spec with tensor payloads attached, or None.

    Raises:
        ValueError: If the assembled spec fails pre-flight validation (a toolkit-side
            serialization bug; the message carries the `E_*` code and JSON path), or the
            gate/condition merge is ambiguous.
        ModuleNotFoundError: If `vllm_hook_plugins` is not installed.
    """
    from aisteer360.algorithms.core.execution.interventions import InterventionSpec
    from aisteer360.utils.optional import require

    from .gates.probe_sum import ProbeSumGate
    from .transforms.base import unwrap_modifiers

    kinds = require("vllm_hook_plugins.core.kinds")
    schema = require("vllm_hook_plugins.core.schema")

    artifacts: dict[str, dict[str, torch.Tensor]] = {}

    def register(tensors: Mapping[str, torch.Tensor]) -> str:
        artifact_id, prepared = artifact_id_for(tensors)
        artifacts.setdefault(artifact_id, prepared)
        return artifact_id

    ops: list[dict[str, Any]] = []
    for intervention in interventions:
        if not isinstance(intervention.layers, tuple):
            raise ValueError("lower_interventions requires bound interventions; call bind() first.")
        site = intervention.resolved_site()
        if site == "norm_input":
            return None
        scope_wire = intervention.scope.export()
        scope: dict[str, Any] = {"kind": scope_wire.kind, **scope_wire.params}

        gate = intervention.gate
        merged = _merge_gate_condition(
            gate, intervention.condition, intervention.boundary, num_layers, register,
        )
        if merged is None:
            return None
        if merged is ...:
            gate_wire = None
        elif isinstance(gate, ProbeSumGate):
            gate_wire = {"kind": "cache_once", "inner": merged}
        else:
            gate_wire = merged

        core, wrappers = unwrap_modifiers(intervention.transform)
        for layer_id in sorted(intervention.layers):
            form = core.export(layer_id)
            if form is None:
                return None
            wire_layer = _map_behavior_layer(layer_id, intervention.boundary, site, num_layers)
            if wire_layer is None:
                return None
            transform_wire: dict[str, Any] = {"kind": form.kind, **form.params}
            modifier_wires: list[dict[str, Any]] = []
            for wrapper in wrappers:
                if wrapper.modifier_wire_kind(form.kind) is None:
                    return None
                modifier_form = wrapper.export_modifier(layer_id)
                if modifier_form is None:
                    continue  # this wrapper contributes no modifier at this layer
                modifier_wire: dict[str, Any] = {"kind": modifier_form.kind, **modifier_form.params}
                if modifier_form.tensors:
                    modifier_wire["artifact"] = register(modifier_form.tensors)
                modifier_wires.append(modifier_wire)
            transform_wire["modifiers"] = modifier_wires
            if form.tensors:
                transform_wire["artifact"] = register(form.tensors)
            ops.append({
                "layers": [wire_layer],
                "transform": transform_wire,
                "scope": dict(scope),
                "gate": gate_wire,
            })

    if not ops:
        return None

    spec = InterventionSpec(ops=tuple(ops), artifacts=artifacts)
    schema.parse_intervention_spec(
        spec.to_wire(),
        num_layers=num_layers,
        allowed_gates=allowed_gates if allowed_gates is not None else kinds.GATE_KINDS,
    )
    return spec
