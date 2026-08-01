"""Serialization of the runtime tuple (transform, layers, token scope, gate) into an
`InterventionSpec`.

The exported spec is the second serialization of the same objects the torch hooks close over:
transforms and gates contribute their own wire payloads (`to_intervention_op_payload`,
`to_intervention_gate`), and this module assembles ops, materializes tensor payloads as
content-addressed artifacts, maps hook placements onto wire layer indices, and pre-flight
validates the result against the plugin schema. A configuration any step cannot serialize
exactly yields None, which marks it hook-only.

Wire layer semantics: an intervention op applies at the residual-stream boundary after decoder
layer `N`, and a gate's condition layers read the materialized input of decoder layer `N`. Hook
placements map accordingly: `"layer_output"` at layer `l` is wire layer `l`; `"layer_input"` at
layer `l` is wire layer `l - 1` (layer 0 has no wire form); `"o_proj"` (per-head attention
outputs entering the output projection) keeps its layer index, matching the wire
`head_additive` placement. Condition layers read at `"layer_output"` shift to `l + 1`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

import safetensors.torch
import torch

from aisteer360.algorithms.core.execution.capabilities import Capability, InterventionKinds
from aisteer360.algorithms.core.execution.interventions import InterventionSpec
from aisteer360.algorithms.core.execution.requirements import Alternative, any_of, needs
from aisteer360.utils.optional import require

from .gates.base import AlwaysOpenGate, BaseGate
from .transforms.base import BaseTransform

logger = logging.getLogger(__name__)

PLACEMENTS = ("layer_output", "layer_input", "o_proj")


def intervention_generate_requirement(plan: InterventionKinds | None) -> tuple[Alternative, ...]:
    """The generate-phase requirement for a state control with the given kind plan.

    A configuration with a wire form runs in-process or on any backend advertising
    `INTERVENTION_SPECS` with the planned kinds; a configuration without one (`plan` is None)
    keeps the conservative in-process requirement.

    Args:
        plan: The kind names the configuration serializes to, or None when hook-only.

    Returns:
        The requirement alternatives.
    """
    if plan is None:
        return needs(Capability.IN_PROCESS_TORCH)
    return any_of(
        needs(Capability.IN_PROCESS_TORCH),
        needs(Capability.INTERVENTION_SPECS, kinds=plan),
    )


def artifact_id_for(tensors: dict[str, torch.Tensor]) -> tuple[str, dict[str, torch.Tensor]]:
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
    prepared = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        for name, tensor in tensors.items()
    }
    data = safetensors.torch.save({name: prepared[name] for name in sorted(prepared)})
    return "sha256:" + hashlib.sha256(data).hexdigest(), prepared


def _map_behavior_layer(layer_id: int, placement: str, num_layers: int) -> int | None:
    if placement == "layer_input":
        mapped = layer_id - 1
    else:
        mapped = layer_id
    if 0 <= mapped < num_layers:
        return mapped
    return None


def _map_condition_layers(layer_ids: Sequence[int], placement: str, num_layers: int) -> list[int] | None:
    offset = 1 if placement == "layer_output" else 0
    mapped = [int(layer_id) + offset for layer_id in layer_ids]
    if all(0 <= layer_id < num_layers for layer_id in mapped):
        return mapped
    return None


def intervention_spec_from_runtime_config(
    *,
    transform: BaseTransform,
    layer_ids: Sequence[int],
    token_scope: str,
    gate: BaseGate | None = None,
    num_layers: int,
    placement: str = "layer_output",
    condition_placement: str | None = None,
    last_k: int | None = None,
    from_position: int | None = None,
    allowed_gates: frozenset[str] | None = None,
    runtime_kwargs: dict | None = None,
) -> InterventionSpec | None:
    """Assemble an `InterventionSpec` from a control's runtime tuple, or None when hook-only.

    Ops are built per behavior layer from the transform's wire payloads; layers whose payloads
    match exactly (same kind, scalar params, modifiers, and tensor content) share one op with a
    grouped `layers` list, and distinct layers sharing one tensor share one artifact. The gate
    payload is shared across ops; probe-backed gates always travel wrapped in `cache_once`, the
    wire form of the prompt-scored-once convention. The assembled spec is pre-flight validated
    with the plugin's `parse_intervention_spec`, so a malformed spec fails here with the same
    `E_*` code and JSON path the server would return.

    Args:
        transform: The live transform (possibly wrapper-chained) the hooks apply.
        layer_ids: The behavior layers, as toolkit layer indices at `placement`.
        token_scope: The token scope kind (`"all"`, `"after_prompt"`, `"last_k"`,
            `"from_position"`).
        gate: The live gate, or None for ungated application.
        num_layers: Decoder layer count from the model layout.
        placement: Where the hooks intervene (`"layer_output"`, `"layer_input"`, `"o_proj"`).
        condition_placement: Where condition hooks read; defaults to `placement`.
        last_k: Scope parameter, required when `token_scope == "last_k"`.
        from_position: Scope parameter, required when `token_scope == "from_position"`.
        allowed_gates: Gate kinds negotiated with the serving backend; defaults to the full
            wire gate table.
        runtime_kwargs: Per-call parameters, unused by the shared assembly and accepted so the
            export signature parallels `get_hooks`.

    Returns:
        The validated spec with tensor payloads attached, or None when any element of the
        configuration has no wire form.

    Raises:
        ValueError: If `placement` is unknown, or the assembled spec fails pre-flight
            validation (a toolkit-side serialization bug; the message carries the `E_*` code
            and JSON path).
        ModuleNotFoundError: If `vllm_hook_plugins` is not installed.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement {placement!r}; placements are {', '.join(PLACEMENTS)}.")
    condition_placement = condition_placement or placement

    kinds = require("vllm_hook_plugins.core.kinds")
    schema = require("vllm_hook_plugins.core.schema")

    artifacts: dict[str, dict[str, torch.Tensor]] = {}

    def register(tensors: dict[str, torch.Tensor]) -> str:
        artifact_id, prepared = artifact_id_for(tensors)
        artifacts.setdefault(artifact_id, prepared)
        return artifact_id

    # scope payload
    scope: dict[str, Any] = {"kind": token_scope}
    if token_scope == "last_k":
        scope["k"] = int(last_k) if last_k is not None else None
    elif token_scope == "from_position":
        scope["position"] = int(from_position) if from_position is not None else None
    if None in scope.values():
        return None

    # gate payload, shared across ops
    gate_wire: dict[str, Any] | None = None
    if gate is not None and not isinstance(gate, AlwaysOpenGate):
        payload = gate.to_intervention_gate()
        if payload is None:
            return None
        if payload.get("kind") == "probe_sum":
            payload = {"kind": "cache_once", "params": {}, "tensors": {}, "inner": payload}
        if payload.get("kind") != "null":
            gate_wire = _gate_wire(payload, condition_placement, num_layers, register)
            if gate_wire is None:
                return None

    # transform payloads per behavior layer, grouped by identical wire content
    grouped: dict[str, dict[str, Any]] = {}
    for layer_id in sorted(int(layer_id) for layer_id in layer_ids):
        payload = transform.to_intervention_op_payload(layer_id)
        if payload is None:
            return None
        wire_layer = _map_behavior_layer(layer_id, placement, num_layers)
        if wire_layer is None:
            return None

        transform_wire: dict[str, Any] = {"kind": payload["kind"], **payload["params"]}
        modifier_wires = []
        for modifier in payload["modifiers"]:
            modifier_wire = {"kind": modifier["kind"], **modifier["params"]}
            if modifier["tensors"]:
                modifier_wire["artifact"] = register(modifier["tensors"])
            modifier_wires.append(modifier_wire)
        transform_wire["modifiers"] = modifier_wires
        if payload["tensors"]:
            transform_wire["artifact"] = register(payload["tensors"])

        signature = json.dumps(transform_wire, sort_keys=True, default=str)
        group = grouped.setdefault(signature, {"layers": [], "transform": transform_wire})
        group["layers"].append(wire_layer)

    if not grouped:
        return None

    ops = tuple(
        {
            "layers": sorted(group["layers"]),
            "transform": group["transform"],
            "scope": dict(scope),
            "gate": gate_wire,
        }
        for group in grouped.values()
    )

    spec = InterventionSpec(ops=ops, artifacts=artifacts)
    schema.parse_intervention_spec(
        spec.to_wire(),
        num_layers=num_layers,
        allowed_gates=allowed_gates if allowed_gates is not None else kinds.GATE_KINDS,
    )
    return spec


def _gate_wire(
    payload: dict[str, Any],
    condition_placement: str,
    num_layers: int,
    register,
) -> dict[str, Any] | None:
    """The wire form of a gate payload, with condition layers mapped and tensors registered.

    A payload naming its own `"condition_placement"` (e.g. a probe gate carrying the probe's
    fitted location) overrides the caller's placement for its condition layers.
    """
    placement = payload.get("condition_placement", condition_placement)
    params = dict(payload.get("params", {}))
    condition_layers = params.get("condition_layers")
    if condition_layers is not None:
        if placement == "o_proj":
            return None
        mapped = _map_condition_layers(condition_layers, placement, num_layers)
        if mapped is None:
            return None
        params["condition_layers"] = mapped
    wire: dict[str, Any] = {"kind": payload["kind"], **params}
    if payload.get("tensors"):
        wire["artifact"] = register(payload["tensors"])
    inner = payload.get("inner")
    if inner is not None:
        inner_wire = _gate_wire(inner, condition_placement, num_layers, register)
        if inner_wire is None:
            return None
        wire["inner"] = inner_wire
    return wire
