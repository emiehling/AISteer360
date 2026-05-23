"""Local pytest config for the EPR test suite.

Registers the `slow` marker and skips slow tests by default unless explicitly opted in via `-m slow`.
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (real model load + training; deselect with `-m \"not slow\"`).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = config.getoption("-m")
    if selected and "slow" in selected:
        return
    skip_slow = pytest.mark.skip(reason="slow test; pass `-m slow` to run.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
