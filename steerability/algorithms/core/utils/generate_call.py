"""Context marker for the `model.generate` calls that an in-process session issues.

The in-process session runs each `model.generate` call inside `generate_call()`, which assigns the
call an ordinal. Ordinals increase over the life of the process. `current_generate_call()` returns
the ordinal of the call in progress. Each thread and each asyncio task sees its own value, since the
value is stored in a context variable.

The state-control hook runtime reads the ordinal to restart its fallback position count at the
first pass of each call, since each call prefills its stream from position 0. The restart also
applies to hooks that stay registered across several calls, such as the hooks under which a
decoding driver's rollouts run.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools

_ORDINALS = itertools.count(1)

_CURRENT: contextvars.ContextVar[int | None] = contextvars.ContextVar("generate_call", default=None)


@contextlib.contextmanager
def generate_call():
    """Mark the model forwards inside the block as the passes of one `model.generate` call.

    Each entry assigns a new ordinal. The previous value is restored when the block exits.
    """
    token = _CURRENT.set(next(_ORDINALS))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_generate_call() -> int | None:
    """Return the ordinal of the marked `model.generate` call in progress.

    Returns:
        The ordinal of the innermost enclosing `generate_call()` block, or None outside a marked
        call.
    """
    return _CURRENT.get()
