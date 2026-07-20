"""Execution backends for steering pipelines.

A `Backend` runs inference (or fitting) against a target — an in-process Hugging Face model, or an
OpenAI-compatible server — and negotiates capabilities so a pipeline can answer *is this control
runnable here?* per control. The dependency rule: backends import `core/` and the compile-relevant
types under `algorithms/state_control/_common/`, never the controls themselves.
"""
from aisteer360.backends.base import (
    Artifact,
    Backend,
    BackendCapabilities,
    StateControlEntry,
    SteeringSession,
)
from aisteer360.backends.generation_params import GenerationParams
from aisteer360.backends.specs import BackendSpec

__all__ = [
    "Artifact",
    "Backend",
    "BackendCapabilities",
    "BackendSpec",
    "GenerationParams",
    "StateControlEntry",
    "SteeringSession",
]
