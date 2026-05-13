"""Tests for the provider factory."""
from __future__ import annotations

import sys
import types

import pytest

from aisteer360.workbenches.vector_calibration.agent.providers.base import (
    ProviderKeys,
    build_from_config,
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


def test_build_anthropic_requires_key(monkeypatch) -> None:
    _install_anthropic_stub(monkeypatch)
    cfg = {
        "generation": {"generator_model": "claude", "generator_provider": "anthropic"},
        "calibration": {"judge": {"model": "claude", "provider": "hf"}},
    }
    # no anthropic key present
    with pytest.raises(ValueError, match="Anthropic generator"):
        # HFJudgeProvider would try to load a real model, so we stub it too
        import aisteer360.workbenches.vector_calibration.agent.providers.hf_local as hf
        monkeypatch.setattr(hf, "HFJudgeProvider", lambda **_: object())
        build_from_config(cfg, ProviderKeys())


def test_build_openai_passes_base_url(monkeypatch) -> None:
    _install_openai_stub(monkeypatch)
    import aisteer360.workbenches.vector_calibration.agent.providers.hf_local as hf
    monkeypatch.setattr(hf, "HFJudgeProvider", lambda **_: object())
    cfg = {
        "generation": {
            "generator_model": "gpt",
            "generator_provider": "openai",
            "generator_base_url": "http://vllm",
        },
        "calibration": {"judge": {"model": "gpt", "provider": "hf"}},
    }
    gen, _ = build_from_config(cfg, ProviderKeys(openai_key="sk-x"))
    assert gen._client.kwargs["base_url"] == "http://vllm"
    assert gen._client.kwargs["api_key"] == "sk-x"
