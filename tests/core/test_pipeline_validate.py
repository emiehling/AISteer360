"""Tests for `SteeringPipeline.validate()` verdicts and report rendering (doc 04 §2, §6)."""
import pytest

from aisteer360.backends.base import Backend, BackendCapabilities
from aisteer360.backends.specs import BackendSpec
from aisteer360.core.requirements import Capability
from aisteer360.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.base import StateControl


class _FakeBackend(Backend):
    """A backend granting an arbitrary capability set, with no model."""

    def __init__(self, capabilities: Capability, notes=None):
        self._caps = BackendCapabilities(capabilities=capabilities, notes=notes or {})
        self.spec = BackendSpec(kind="openai", model="fake")
        self.tokenizer = None
        self.model_identity = "fake"

    @property
    def capabilities(self):
        return self._caps

    def open_session(self, entries, prompt_ctx, runtime_kwargs):
        raise NotImplementedError


class _PlanControl(StateControl):
    Args = None

    def plan(self, prompt_ctx, runtime_kwargs=None):
        return []


class _HookControl(StateControl):
    Args = None

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        return {"pre": [], "forward": [], "backward": []}


def test_plan_control_supported_on_residual_write_backend():
    backend = _FakeBackend(Capability.RESIDUAL_WRITE)
    pipeline = SteeringPipeline(controls=[_PlanControl()], backend=backend)
    report = pipeline.validate()
    assert report.ok
    verdict = next(v for v in report.verdicts if v.control == "_PlanControl")
    assert verdict.status == "supported"


def test_hook_control_unsupported_on_api_backend():
    backend = _FakeBackend(Capability.MESSAGES | Capability.TEXT)
    pipeline = SteeringPipeline(controls=[_HookControl()], backend=backend)
    report = pipeline.validate()
    assert not report.ok
    verdict = next(v for v in report.verdicts if v.control == "_HookControl")
    assert verdict.status == "unsupported"
    assert verdict.missing & Capability.FORWARD_HOOKS
    assert "HuggingFaceBackend" in verdict.fix


def test_raise_if_failed_message_is_actionable():
    backend = _FakeBackend(Capability.MESSAGES)
    pipeline = SteeringPipeline(controls=[_HookControl()], backend=backend)
    with pytest.raises(RuntimeError, match="FORWARD_HOOKS"):
        pipeline.validate().raise_if_failed()


def test_degraded_when_backend_attaches_note():
    backend = _FakeBackend(
        Capability.RESIDUAL_WRITE, notes={Capability.RESIDUAL_WRITE: "chunked prefill: gated prompt steering approximate"}
    )
    pipeline = SteeringPipeline(controls=[_PlanControl()], backend=backend)
    report = pipeline.validate()
    verdict = next(v for v in report.verdicts if v.control == "_PlanControl")
    assert verdict.status == "degraded"
    assert "chunked prefill" in verdict.note


def test_report_str_is_readable():
    backend = _FakeBackend(Capability.MESSAGES)
    pipeline = SteeringPipeline(controls=[_HookControl()], backend=backend)
    text = str(pipeline.validate())
    assert "_HookControl" in text
    assert "unsupported" in text
