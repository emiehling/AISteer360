"""Gate that sums a probe's per-layer contributions and decides at the calibrated bias.

The probe condition path is decomposed so a scorer streams each condition layer's affine
contribution (`w_l . x_l`, without the bias) and this gate sums the contributions and applies
the bias once at decision time; see `condition_scorers` for the scorer and the
`probe_condition()` factory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from aisteer360.algorithms.core.internals.probes.probe import Probe

from .base import BaseGate

if TYPE_CHECKING:
    from ..specs import WireForm


class ProbeSumGate(BaseGate):
    """Gate that sums a probe's per-layer contributions and decides at the calibrated bias.

    Each `update(scores, key=layer_id)` records one condition layer's per-row contribution;
    `is_ready()` reports True once every layer in `probe.layer_ids` has reported for the
    current generation, and `open_rows()` returns `(sum of contributions + probe.bias) >= 0`
    per row (all-closed before any evidence). Row semantics (logical rows, beam collapse)
    follow `BaseGate`.

    Args:
        probe: The probe whose layers and bias define the decision.
    """

    wire_kind: ClassVar[str | None] = "probe_sum"

    def __init__(self, probe: Probe):
        self.probe = probe
        self.expected_keys: set[int] = set(probe.layer_ids)
        self.bias: float = float(probe.bias)
        self._contributions: dict[int, torch.Tensor] = {}

    def reset(self, num_rows: int = 1) -> None:
        """Clear all stored contributions and size the gate to the logical batch."""
        super().reset(num_rows)
        self._contributions.clear()

    def update(self, scores: torch.Tensor | float, *, key: int | None = None) -> None:
        """Record one condition layer's per-row contribution.

        Args:
            scores: Per-row contributions, shape `[num_rows]` (float allowed when
                `num_rows == 1`).
            key: The condition layer id the contribution belongs to.
        """
        rows = self._coerce_scores(scores)
        self._contributions[key if key is not None else 0] = rows

    def open_rows(self) -> torch.BoolTensor:
        """Per-row decision at the calibrated bias; all-closed before any evidence."""
        if not self._contributions:
            return torch.zeros(self.num_rows, dtype=torch.bool)
        total = torch.stack(list(self._contributions.values()), dim=0).sum(dim=0)
        return total + self.bias >= 0

    def is_ready(self) -> bool:
        """True once every expected condition layer has reported."""
        return self.expected_keys <= self._contributions.keys()

    def export(self) -> "WireForm | None":
        """The `probe_sum` wire form built from the probe.

        The `weights` tensor stacks the probe's per-layer weight vectors row-aligned with the
        probe's layer order, and the calibrated bias travels as the artifact's scalar `bias`
        tensor. The intervention's `Condition` supplies the wire `condition_layers`, merged in
        by the lowering.
        """
        from ..specs import WireForm

        weights = torch.stack(
            [self.probe.weights[layer_id].to(torch.float32) for layer_id in self.probe.layer_ids]
        )
        return WireForm(
            kind="probe_sum",
            params={"pooling": self.probe.pooling},
            tensors={"weights": weights, "bias": torch.tensor(float(self.probe.bias))},
        )

    def to_intervention_gate(self) -> dict | None:
        """The `probe_sum` wire payload built from the probe.

        The `weights` tensor stacks the probe's per-layer weight vectors row-aligned with the
        `condition_layers` order, and the calibrated bias travels as the artifact's scalar
        `bias` tensor. Condition layers are the probe's layer ids at the probe's fitted
        location; the exporter maps them onto wire layer-input indices.
        """
        weights = torch.stack(
            [self.probe.weights[layer_id].to(torch.float32) for layer_id in self.probe.layer_ids]
        )
        return {
            "kind": "probe_sum",
            "params": {
                "condition_layers": [int(layer_id) for layer_id in self.probe.layer_ids],
                "pooling": self.probe.pooling,
            },
            "tensors": {"weights": weights, "bias": torch.tensor(float(self.probe.bias))},
            "condition_placement": self.probe.location,
        }
