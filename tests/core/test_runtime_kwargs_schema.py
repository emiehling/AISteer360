"""Tests for runtime-kwargs scope declarations: merging, defaults, conflicts, and steer-time
enforcement on the pipeline."""
import pytest

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.utils.controls import runtime_kwargs_schema
from tests.conftest import MockInputControl, MockStateControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _RowInput(MockInputControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "spans", "type": "list[str]", "required": True, "scope": "row"},
    ]


class _RowState(MockStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "spans", "type": "list[str]", "required": False, "scope": "row"},
    ]


class _UnscopedState(MockStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "spans", "type": "list[str]"},
    ]


class _CallState(MockStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "gate_inputs", "type": "dict", "scope": "call"},
    ]


class _BadScopeState(MockStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "spans", "type": "list[str]", "scope": "per_row"},
    ]


class _OtherTypeState(MockStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "spans", "type": "dict", "scope": "row"},
    ]


class TestRuntimeKwargsSchema:
    def test_missing_scope_defaults_to_call(self):
        merged = runtime_kwargs_schema([_UnscopedState()])
        assert merged["spans"]["scope"] == "call"

    def test_declared_scopes_pass_through(self):
        merged = runtime_kwargs_schema([_RowInput(), _CallState()])
        assert merged["spans"]["scope"] == "row"
        assert merged["gate_inputs"]["scope"] == "call"

    def test_invalid_scope_raises_naming_control_and_entry(self):
        with pytest.raises(ValueError, match="_BadScopeState.*'spans'.*'per_row'"):
            runtime_kwargs_schema([_BadScopeState()])

    def test_agreeing_shared_names_merge(self):
        merged = runtime_kwargs_schema([_RowInput(), _RowState()])
        assert merged["spans"]["scope"] == "row"
        # the first declaration's other fields are kept
        assert merged["spans"]["required"] is True

    def test_scope_conflict_raises_naming_both_controls(self):
        with pytest.raises(ValueError, match="_RowInput and _UnscopedState.*different\\s+scopes"):
            runtime_kwargs_schema([_RowInput(), _UnscopedState()])

    def test_type_conflict_raises_naming_both_controls(self):
        with pytest.raises(ValueError, match="_RowInput and _OtherTypeState.*different\\s+types"):
            runtime_kwargs_schema([_RowInput(), _OtherTypeState()])

    def test_disabled_controls_are_excluded(self):
        conflicting = _UnscopedState()
        conflicting.enabled = False
        merged = runtime_kwargs_schema([_RowInput(), conflicting])
        assert merged["spans"]["scope"] == "row"


class TestSteerTimeEnforcement:
    def _pipeline(self, controls) -> SteeringPipeline:
        return SteeringPipeline(controls=controls, model=tiny_llama(), tokenizer=wordlevel_tokenizer())

    def test_conflicting_declarations_raise_at_steer(self):
        pipeline = self._pipeline([_RowInput(), _UnscopedState()])
        with pytest.raises(ValueError, match="different\\s+scopes"):
            pipeline.steer()

    def test_agreeing_shared_declarations_keep_the_sharing_warning(self):
        pipeline = self._pipeline([_RowInput(), _RowState()])
        with pytest.warns(UserWarning, match="share"):
            pipeline.steer()
