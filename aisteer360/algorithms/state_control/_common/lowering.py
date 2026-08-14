"""The wire compiler from bound interventions to intervention-spec payloads.

`lower_interventions` compiles a bound `Intervention` tuple into an `InterventionSpec` for
intervention-capable backends, folding each component's wire form and content-addressing
tensor payloads via `artifact_id_for`. Its counterpart, `build_hooks` in `runtime.py`,
compiles the same IR into torch hooks for one generation.
"""
from __future__ import annotations

import hashlib
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from .specs import Boundary, Condition, Intervention, Site

if TYPE_CHECKING:
    from aisteer360.algorithms.core.execution.payloads import InterventionSpec


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
) -> "dict[str, Any] | None | EllipsisType":
    """The wire gate for a gate/condition pair, folding the toolkit's gate/scorer/condition
    split into the wire `GateSpec`.

    The wire gate's params are the gate's exported params plus `condition_layers` plus the
    scorer form's params; the wire gate's artifact is the gate's exported tensors if any, else
    the scorer form's tensors. Both sides exporting tensors, or exporting conflicting param
    values, is a compile error. A probe gate's evidence layers follow the probe's own layer
    order (the exported weight rows align with it) at the probe's fitted boundary; a
    condition, when present, must cover the same layer set. Without a condition (the follower
    half of a shared-gate composition), the probe alone supplies the evidence layers. Returns
    the Ellipsis sentinel for an ungated op (always-open), None when the configuration has no
    wire form.
    """
    from .gates.base import AlwaysOpenGate, BaseGate
    from .gates.cache_once import CacheOnceGate
    from .gates.probe_sum import ProbeSumGate

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

    params = dict(form.params)
    tensors = dict(form.tensors)

    if isinstance(gate, ProbeSumGate):
        if condition is not None and set(condition.layer_ids) != set(gate.probe.layer_ids):
            raise ValueError(
                "Condition layers must cover the probe's layers exactly; the wire gate's "
                f"weight rows align with the probe. Got {tuple(condition.layer_ids)} vs "
                f"probe layers {tuple(gate.probe.layer_ids)}."
            )
        # the probe owns the evidence layers and their order (weight rows align with them),
        # read at the probe's fitted boundary
        condition_layers = [int(layer_id) for layer_id in gate.probe.layer_ids]
        condition_boundary = gate.probe.location
    elif condition is not None:
        condition_layers = list(condition.layer_ids)
        condition_boundary = boundary
    else:
        return None  # a conditional wire gate reads evidence at declared condition layers

    mapped = _map_condition_layers(condition_layers, condition_boundary, num_layers)
    if mapped is None:
        return None
    params["condition_layers"] = mapped

    if condition is not None:
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
    from aisteer360.algorithms.core.execution.payloads import InterventionSpec
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
