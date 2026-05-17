"""Pydantic models for the composition workbench API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ControlNode(BaseModel):
    """A single control in the composition canvas.

    `id` is a client-generated UUID used for edge connections and for diffing across edits.
    `category` is one of the four steering categories (`input_control`, `structural_control`,
    `state_control`, `output_control`). `method` is the registry key for the control class.
    `args` are the constructor kwargs that will be passed to the control. `position` is the
    canvas position used by the UI; the agent ignores it.
    """

    id: str
    category: str
    method: str
    args: dict[str, Any] = Field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)


class PipelineDefinition(BaseModel):
    """Full composition state — what the agent needs to build a SteeringPipeline."""

    model_config = {"protected_namespaces": ()}

    model_name_or_path: str
    nodes: list[ControlNode] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    """Body for POST /api/sessions."""

    model_config = {"protected_namespaces": ()}

    model_name_or_path: str
    hf_model_kwargs: dict[str, Any] = Field(default_factory=dict)
    device_map: str = "auto"
    idle_timeout_s: float = 600.0


class AgentCommand(BaseModel):
    command: str
    server: str
    session_id: str
    agent_token: str


class SessionSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    model_name: str
    status: str
    model_info: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    stale: bool = False
    created_at: float
    updated_at: float
    last_heartbeat: float | None = None
    idle_timeout_s: float


class SessionDetail(SessionSummary):
    config: dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    session: SessionDetail
    agent_token: str
    agent_command: AgentCommand
    dispatch_status: Literal["local", "ssh", "manual", "failed"] = "manual"
    dispatch_error: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class InferenceRequest(BaseModel):
    """Body for POST /api/sessions/{id}/infer."""

    pipeline: PipelineDefinition
    prompt: str
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class InferenceAcceptedResponse(BaseModel):
    request_id: str


class InferenceResultEvent(BaseModel):
    """WS event published when the agent completes one inference request."""

    event: Literal["inference_result"] = "inference_result"
    request_id: str
    generated_text: str
    elapsed_ms: float
    pipeline_hash: str


# ── agent-facing schemas ─────────────────────────────────────────

class SessionClaimResponse(BaseModel):
    """Returned to the agent on successful claim of a session."""

    model_config = {"protected_namespaces": ()}

    session_id: str
    model_name_or_path: str
    config: dict[str, Any]
    provider_keys: dict[str, str | None] = Field(default_factory=dict)
    idle_timeout_s: float


class SessionReadyPost(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_info: dict[str, Any] = Field(default_factory=dict)


class SessionResultPost(BaseModel):
    request_id: str
    generated_text: str
    elapsed_ms: float
    pipeline_hash: str
    error: str | None = None


class SessionPollResponse(BaseModel):
    """Response of GET /api/agent/sessions/{id}/poll.

    Exactly one of `request` / `close` is set on a non-empty response. `request` is the queued
    `InferenceRequest` body augmented with a `request_id`; `close` signals that the browser asked
    for graceful shutdown.
    """

    request: dict[str, Any] | None = None
    close: bool = False


class SessionErrorPost(BaseModel):
    message: str


__all__ = [
    "ControlNode",
    "PipelineDefinition",
    "SessionCreateRequest",
    "SessionCreateResponse",
    "SessionSummary",
    "SessionDetail",
    "SessionListResponse",
    "InferenceRequest",
    "InferenceAcceptedResponse",
    "InferenceResultEvent",
    "AgentCommand",
    "SessionClaimResponse",
    "SessionReadyPost",
    "SessionResultPost",
    "SessionPollResponse",
    "SessionErrorPost",
]
