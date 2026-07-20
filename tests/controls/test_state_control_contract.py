"""Tests for the `StateControl` plan/get_hooks contract (doc 03 §3, §7)."""
import pytest

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.core.requirements import Capability


def test_neither_entry_point_fails_at_class_definition():
    with pytest.raises(TypeError, match="override.*plan|plan.*get_hooks"):

        class Neither(StateControl):
            Args = None


def test_both_entry_points_fails_at_class_definition():
    with pytest.raises(TypeError, match="both"):

        class Both(StateControl):
            Args = None

            def plan(self, prompt_ctx, runtime_kwargs=None):
                return []

            def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
                return {"pre": [], "forward": [], "backward": []}


def test_plan_only_ok_and_declares_residual_write():
    class PlanOnly(StateControl):
        Args = None

        def plan(self, prompt_ctx, runtime_kwargs=None):
            return []

    assert PlanOnly().requires().capabilities == Capability.RESIDUAL_WRITE


def test_get_hooks_only_ok_and_declares_forward_hooks():
    class HooksOnly(StateControl):
        Args = None

        def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
            return {"pre": [], "forward": [], "backward": []}

    assert HooksOnly().requires().capabilities == Capability.FORWARD_HOOKS


def test_migrated_controls_use_plan_pasta_uses_hooks():
    """The migrated declarative controls override `plan`; PASTA stays hook-level."""
    from aisteer360.algorithms.state_control.act_add.control import ActAdd
    from aisteer360.algorithms.state_control.caa.control import CAA
    from aisteer360.algorithms.state_control.cast.control import CAST
    from aisteer360.algorithms.state_control.directional_ablation.control import DirectionalAblation
    from aisteer360.algorithms.state_control.iti.control import ITI
    from aisteer360.algorithms.state_control.pasta.control import PASTA

    for control_cls in (CAA, ActAdd, CAST, DirectionalAblation, ITI):
        assert control_cls.plan is not StateControl.plan, f"{control_cls.__name__} should override plan()"
        assert control_cls.get_hooks is StateControl.get_hooks, f"{control_cls.__name__} should inherit get_hooks"

    assert PASTA.get_hooks is not StateControl.get_hooks
    assert PASTA.plan is StateControl.plan
    pasta_caps = PASTA(substrings=[], head_config=[0], alpha=1.0).requires().capabilities
    assert pasta_caps & Capability.ATTENTION_WRITE
    assert pasta_caps & Capability.FORWARD_HOOKS
