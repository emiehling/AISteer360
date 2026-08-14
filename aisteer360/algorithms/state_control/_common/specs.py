"""The intervention IR for state control components.

The intervention IR (`TokenScope`, `Condition`, `Intervention`) is the single declarative
statement of a residual-stream state control's behavior. Both compilers read it:
`runtime.build_hooks` turns a bound intervention tuple into torch hooks for one generation,
and `lowering.lower_interventions` turns it into an `InterventionSpec` for
intervention-capable backends. Components describe their own wire form (`WireForm` via each
component's `export`), so no layer of the system re-derives another layer's configuration by
introspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Protocol, Sequence, get_args, runtime_checkable

import torch

from aisteer360.algorithms.core.execution.contracts import InterventionKinds

if TYPE_CHECKING:
    from .condition_scorers import ConditionScorer
    from .gates.base import BaseGate
    from .selectors.base import BaseSelector
    from .transforms.base import BaseTransform

Boundary = Literal["layer_output", "layer_input"]
Site = Literal["decoder_layer", "o_proj", "norm_input"]
ScopeKind = Literal["all", "after_prompt", "last_k", "from_position"]


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

    kind: ScopeKind
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        if self.kind not in get_args(ScopeKind):
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
            layout: Structural facts (`ModelFacts`) used when `model` is None.
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
        if (
            self.boundary == "layer_input"
            and self.resolved_site() == "decoder_layer"
            and isinstance(self.layers, tuple)
            and 0 in self.layers
        ):
            # layer 0 input edits precede the first wire boundary; the o_proj site keeps its
            # layer index on the wire, so layer 0 stays expressible there
            return None
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
    be a `ProbeSumGate` (bare or `cache_once`-wrapped), since the wire gate computes the
    scorer's affine evidence from the probe weights itself. With a condition, its scorer must
    be the `ProbeContributionScorer` over the same probe with condition layers matching the
    probe's layers. Without a condition (the follower half of a shared-gate composition), the
    probe itself supplies the evidence layers, so the gating still lowers. A bare probe gate
    plans `cache_once`, the wire form of the prompt-scored-once convention.
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
    if condition is not None:
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
