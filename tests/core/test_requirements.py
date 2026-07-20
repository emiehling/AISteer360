"""Tests for `core.requirements` and the base-control `requires()` defaults (doc 01 §6)."""
import json

from aisteer360.core.requirements import (
    Capability,
    ControlVerdict,
    Requirements,
    ValidationReport,
)
from aisteer360.evaluation.utils.data_utils import to_jsonable


def test_capability_combines_and_tests_membership():
    combined = Capability.MESSAGES | Capability.TEXT
    assert combined & Capability.MESSAGES
    assert combined & Capability.TEXT
    assert not (combined & Capability.SCORING)


def test_requirements_and_capability_are_jsonable():
    req = Requirements(capabilities=Capability.RESIDUAL_WRITE | Capability.HIDDEN_READ, phase="generate")
    encoded = to_jsonable({"req": req, "cap": Capability.SCORING})
    # must round-trip through json without raising (values are string reprs)
    json.dumps(encoded)
    assert isinstance(encoded["cap"], str)


def test_validation_report_ok_and_str():
    report = ValidationReport(
        verdicts=[
            ControlVerdict(control="FewShot", status="supported"),
            ControlVerdict(control="CAA", status="degraded", note="quantized o_proj"),
        ]
    )
    assert report.ok is True
    text = str(report)
    assert "FewShot" in text and "CAA" in text and "quantized o_proj" in text


def test_validation_report_raise_if_failed():
    report = ValidationReport(
        verdicts=[
            ControlVerdict(
                control="PASTA",
                status="unsupported",
                missing=Capability.FORWARD_HOOKS,
                fix="run on HuggingFaceBackend",
            )
        ]
    )
    assert report.ok is False
    try:
        report.raise_if_failed()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        message = str(exc)
        assert "PASTA" in message
        assert "FORWARD_HOOKS" in message
        assert "HuggingFaceBackend" in message


class TestBaseControlRequires:
    def test_input_control_defaults(self):
        from aisteer360.algorithms.input_control.base import InputControl, NoInputControl

        class TokenOnly(InputControl):
            Args = None

            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

        class MessageLevel(InputControl):
            Args = None

            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

            def adapt_messages(self, messages, runtime_kwargs=None):
                return messages

        assert TokenOnly().requires().capabilities == Capability.TOKEN_IDS
        assert MessageLevel().requires().capabilities == Capability.MESSAGES
        # disabled no-op requires nothing
        assert NoInputControl().requires().capabilities == Capability(0)

    def test_structural_control_defaults(self):
        from aisteer360.algorithms.structural_control.base import NoStructuralControl

        assert NoStructuralControl().requires().capabilities == Capability(0)

    def test_state_control_defaults(self):
        from aisteer360.algorithms.state_control.base import NoStateControl, StateControl

        class Hooked(StateControl):
            Args = None

            def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
                return {"pre": [], "forward": [], "backward": []}

        req = Hooked().requires()
        assert req.capabilities == Capability.FORWARD_HOOKS
        assert req.phase == "generate"
        assert NoStateControl().requires().capabilities == Capability(0)

    def test_output_control_defaults(self):
        from aisteer360.algorithms.output_control.base import NoOutputControl

        assert NoOutputControl().requires().capabilities == Capability(0)
