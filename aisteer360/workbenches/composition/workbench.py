"""Composition workbench: build a `SteeringPipeline` from a `PipelineDefinition`.

Lives entirely on the agent side. The server only stores opaque session config; per-request
pipeline definitions are sent inline by the browser and consumed here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from .interface.schemas import ControlNode, PipelineDefinition

logger = logging.getLogger(__name__)


def hash_pipeline(definition: PipelineDefinition) -> str:
    """Stable hash of a `PipelineDefinition`, used to detect when steer() must rerun.

    Position is intentionally excluded — moving a node on the canvas does not change its
    semantics and should not invalidate the cached pipeline.
    """
    payload = {
        "model": definition.model_name_or_path,
        "nodes": sorted(
            [
                {
                    "id": node.id,
                    "category": node.category,
                    "method": node.method,
                    "args": node.args,
                }
                for node in definition.nodes
            ],
            key=lambda n: (n["category"], n["method"], n["id"]),
        ),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _resolve_control(node: ControlNode):
    """Look up the control class for a node and instantiate it from `node.args`.

    Raises ValueError if the (category, method) pair is not registered.
    """
    from aisteer360.algorithms.core.registry import REGISTRY
    bucket = REGISTRY.get(node.category)
    if not bucket:
        raise ValueError(
            f"Unknown control category '{node.category}'. "
            f"Known: {sorted(REGISTRY.keys())}"
        )
    method = bucket.get(node.method)
    if method is None:
        raise ValueError(
            f"Unknown method '{node.method}' for category '{node.category}'. "
            f"Known: {sorted(bucket.keys())}"
        )
    return method.control_cls(**(node.args or {}))


class CompositionWorkbench:
    """Agent-side helper for turning `PipelineDefinition`s into `SteeringPipeline`s."""

    def build_pipeline(
        self,
        definition: PipelineDefinition,
        *,
        device_map: str | dict[str, int] = "auto",
        hf_model_kwargs: dict[str, Any] | None = None,
    ):
        """Instantiate controls from `definition` and return a fresh `SteeringPipeline`.

        The caller is responsible for invoking `.steer()` on the returned pipeline.
        """
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

        controls = [_resolve_control(node) for node in definition.nodes]
        return SteeringPipeline(
            model_name_or_path=definition.model_name_or_path,
            controls=controls,
            device_map=device_map,
            hf_model_kwargs=dict(hf_model_kwargs or {}),
        )

    def definition_changed(
        self,
        old: PipelineDefinition | None,
        new: PipelineDefinition,
    ) -> bool:
        if old is None:
            return True
        return hash_pipeline(old) != hash_pipeline(new)

    def model_changed(
        self,
        old: PipelineDefinition | None,
        new: PipelineDefinition,
    ) -> bool:
        if old is None:
            return True
        return old.model_name_or_path != new.model_name_or_path


__all__ = ["CompositionWorkbench", "hash_pipeline"]
