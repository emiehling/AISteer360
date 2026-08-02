"""Shared helpers for `TransformHookRuntime` tests.

`RecordingTransform` records the token mask seen at each apply and adds a constant, so a
mis-positioned mask changes hidden states and therefore greedy outputs. `strip_clock` wraps a
runtime hook to drop the `cache_position` kwarg, forcing the pass-counting fallback path.
"""
import torch

from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform


class RecordingTransform(BaseTransform):
    """Records the token mask seen at each apply; adds a constant."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.masks: list[torch.BoolTensor] = []

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        self.masks.append(token_mask.detach().clone())
        return hidden_states + self.value


def strip_clock(hook):
    """Wrap a runtime hook so it never sees `cache_position`, forcing the pass-counting fallback.

    Only the wrapped hook is blinded; when a pre-hook returns replacement inputs, the kwarg is
    restored so the module's real call still receives it.
    """

    def stripped(module, args, kwargs, *rest):
        seen = {k: v for k, v in kwargs.items() if k != "cache_position"}
        result = hook(module, args, seen, *rest)
        if (
            not rest  # pre-hook shape; a returned (args, kwargs) pair replaces the module inputs
            and isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
            and "cache_position" in kwargs
            and "cache_position" not in result[1]
        ):
            return result[0], {**result[1], "cache_position": kwargs["cache_position"]}
        return result

    return stripped


class RuntimeCapture:
    """Captures each `TransformHookRuntime` that `build_hooks` constructs.

    `build_hooks` creates one fresh runtime per logical generation and discards its reference
    once the hook closures own it; tests asserting position or opener state install this via
    `capture_built_runtimes` and read `.last`.
    """

    def __init__(self):
        self.runtimes = []

    @property
    def last(self):
        return self.runtimes[-1] if self.runtimes else None


def capture_built_runtimes(monkeypatch) -> RuntimeCapture:
    """Patch the runtime module so every runtime built by `build_hooks` is recorded."""
    import aisteer360.algorithms.state_control._common.runtime as runtime_module

    capture = RuntimeCapture()
    original = runtime_module.TransformHookRuntime

    class _Recording(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            capture.runtimes.append(self)

    monkeypatch.setattr(runtime_module, "TransformHookRuntime", _Recording)
    return capture
