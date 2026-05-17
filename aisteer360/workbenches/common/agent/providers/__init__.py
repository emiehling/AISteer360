"""LLM provider abstractions for the agent.

The steered model is always local HuggingFace (we need hidden-state access for extraction and
hook-based calibration). These providers apply only to the generator role (stage 1) and the
judge role (stage 3).
"""
from .base import (
    GenerationProvider,
    JudgeProvider,
    ProviderKeys,
    build_generation_provider,
    build_judge_provider,
)

__all__ = [
    "GenerationProvider",
    "JudgeProvider",
    "ProviderKeys",
    "build_generation_provider",
    "build_judge_provider",
]
