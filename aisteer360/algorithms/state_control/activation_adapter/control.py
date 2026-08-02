"""ActivationAdapter: assemble an activation-steering recipe from `_common` components."""
from __future__ import annotations

import logging

from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate, CacheOnceGate, ProbeSumGate
from aisteer360.algorithms.state_control._common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control._common.specs import Condition, Intervention, TokenScope
from aisteer360.algorithms.state_control.base import InterventionControl

from .args import ActivationAdapterArgs

logger = logging.getLogger(__name__)


class ActivationAdapter(InterventionControl):
    """Composable activation-steering control (single-behavior atom).

    `ActivationAdapter` wires together the `state_control/_common` component families (a transform
    that carries its own steering artifact, a selector, a gate, and a token scope) so an
    activation-steering recipe can be assembled directly without writing a new control class. It
    edits the residual stream at one or more layers during generation, applying the transform at
    masked positions whenever its gate is open.

    The transform is the sole artifact carrier. It holds a concrete `SteeringVector` / directions
    mapping (bound at construction), or an `ArtifactSource` such as `ContrastiveFit(data=...)` that
    is resolved once at `steer()` time. The adapter has no artifact slots and never sees a
    `SteeringVector` directly.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    (transform, layers, scope, gate, and condition), and the base class binds it at `steer()`,
    verifying the transform covers every behavior layer.

    Steering multiple behaviors is done by placing multiple adapters in a pipeline's `controls` list
    (each adapter owns exactly one transform chain / gate / token scope). Joint conditioning is
    achieved by sharing one gate instance across adapters. One driver declares the condition path
    (`condition_layer_ids` + `score_fn`) and updates the gate; N followers pass the same gate
    instance with `gate_driven_externally=True` and read its decision. Gate reads are
    side-effect-free and gate reset is idempotent, so the shared instance is reset harmlessly once
    per adapter when hooks are built.

    Within a forward pass, a follower's behavior hook at layer L reads `is_open()` when L forwards,
    so it observes driver evidence only from condition layers `< L`. Evidence from layers `>= L`
    takes effect on the next pass. When a driver and follower hook the same layer index, place the
    driver before the follower in the pipeline's `controls` list (registration order = execution
    order).

    Batching is native (`supports_batching = True`); gates are row-vectorized, so a gated adapter
    scores and gates each prompt of a batch independently. The gate rejects scalar scores for
    multi-row batches, so a mis-specified scorer fails loudly rather than silently applying one
    decision batch-wide.
    """

    Args = ActivationAdapterArgs
    supports_batching = True

    def _configure(self):
        if self.layer_ids is not None:
            layers = tuple(sorted(set(int(lid) for lid in self.layer_ids)))
        else:
            if isinstance(self.layer_selector, ConditionPointSelector):
                raise ValueError(
                    "ConditionPointSelector returns a ConditionPoint for gating, not a behavior layer; "
                    "supply layer_ids or a layer selector that returns layer id(s)."
                )
            layers = self.layer_selector

        condition = None
        if self.condition_layer_ids:
            condition = Condition(
                layer_ids=tuple(sorted(set(int(lid) for lid in self.condition_layer_ids))),
                scorer=self.score_fn,
            )

        self._template = (Intervention(
            layers=layers,
            transform=self.transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            gate=self.gate if self.gate is not None else AlwaysOpenGate(),
            condition=condition,
            boundary=self.hook_point,
        ),)

    @property
    def hook_only_hint(self) -> str:
        gate = self.gate
        inner = gate.inner if isinstance(gate, CacheOnceGate) else gate
        if gate is not None and not isinstance(gate, AlwaysOpenGate) and not isinstance(inner, ProbeSumGate):
            return (
                "this gate configuration has no intervention-spec serialization (probe-backed "
                "gating lowers; MultiKeyThresholdGate and custom scorers do not); run on the "
                "huggingface backend"
            )
        return (
            "this transform configuration has no intervention-spec form; run on the "
            "huggingface backend"
        )

    @property
    def _layer_ids(self) -> list[int]:
        """The resolved behavior layers (empty before `steer()`)."""
        return list(self.interventions[0].layers) if self.interventions else []

    @property
    def _condition_layer_ids(self) -> list[int]:
        """The condition layers (empty when ungated)."""
        if self.interventions and self.interventions[0].condition is not None:
            return list(self.interventions[0].condition.layer_ids)
        return list(self.condition_layer_ids or [])

    def cleanup(self) -> None:
        """Drop references to the bound interventions."""
        self.interventions = ()
