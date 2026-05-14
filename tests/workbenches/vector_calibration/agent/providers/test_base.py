"""Tests for the provider factory."""
from __future__ import annotations

import sys
import types

import pytest

from aisteer360.workbenches.vector_calibration.agent.providers.base import (
    ProviderKeys,
    build_generation_provider,
    build_judge_provider,
)


def _install_anthropic_stub(monkeypatch):
    mod = types.ModuleType("anthropic")
    class _Client:
        def __init__(self, api_key):
            self.api_key = api_key
    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def _install_openai_stub(monkeypatch):
    mod = types.ModuleType("openai")
    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    mod.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", mod)


def test_build_anthropic_generation_requires_key(monkeypatch) -> None:
    _install_anthropic_stub(monkeypatch)
    cfg = {"generation": {"generator_model": "claude", "generator_provider": "anthropic"}}
    with pytest.raises(ValueError, match="Anthropic generator"):
        build_generation_provider(cfg, ProviderKeys())


def test_build_openai_generation_passes_base_url(monkeypatch) -> None:
    _install_openai_stub(monkeypatch)
    cfg = {
        "generation": {
            "generator_model": "gpt",
            "generator_provider": "openai",
            "generator_base_url": "http://vllm",
        },
    }
    gen = build_generation_provider(cfg, ProviderKeys(openai_key="sk-x"))
    assert gen._client.kwargs["base_url"] == "http://vllm"
    assert gen._client.kwargs["api_key"] == "sk-x"


def test_build_anthropic_judge_requires_key(monkeypatch) -> None:
    _install_anthropic_stub(monkeypatch)
    cfg = {"calibration": {"judge": {"model": "claude", "provider": "anthropic"}}}
    with pytest.raises(ValueError, match="Anthropic judge"):
        build_judge_provider(cfg, ProviderKeys())


def test_judge_and_generation_are_independent(monkeypatch) -> None:
    """Building one provider should not require the other's keys."""
    _install_openai_stub(monkeypatch)
    cfg = {
        "generation": {"generator_model": "gpt", "generator_provider": "openai"},
        "calibration": {"judge": {"model": "irrelevant", "provider": "anthropic"}},
    }
    gen = build_generation_provider(cfg, ProviderKeys(openai_key="sk-x"))
    assert gen._client.kwargs["api_key"] == "sk-x"
