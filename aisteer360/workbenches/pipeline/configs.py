"""Pipeline workbench configs.

Stub. Today, the only persisted config a session needs is the model name and the load-time
kwargs; the per-request `PipelineDefinition` is sent inline with each inference request and is
not persisted server-side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionConfig:
    """Per-session configuration, persisted as opaque JSON in `sessions.config_json`.

    The `PipelineDefinition` is intentionally NOT persisted at this level — pipelines are sent
    per-request and may change many times across a single session.
    """

    model_name_or_path: str
    hf_model_kwargs: dict[str, Any] = field(default_factory=dict)
    device_map: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "hf_model_kwargs": dict(self.hf_model_kwargs),
            "device_map": self.device_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        return cls(
            model_name_or_path=str(data["model_name_or_path"]),
            hf_model_kwargs=dict(data.get("hf_model_kwargs") or {}),
            device_map=str(data.get("device_map") or "auto"),
        )


__all__ = ["SessionConfig"]
