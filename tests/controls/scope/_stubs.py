"""Test-only scaffolding for SCOPE meta-agent and control tests."""
from __future__ import annotations

from typing import Callable


class StubReflectionLM:
    """Callable returning canned responses in cyclic order; records prompts for inspection."""

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("StubReflectionLM requires at least one response.")
        self.responses = list(responses)
        self.prompts: list[str] = []
        self._idx = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return response


def make_classifier_lm(stream: str, confidence: float) -> Callable[[str], str]:
    """Convenience factory: returns a callable that emits a classifier-shaped JSON response."""
    payload = '{"category": "%s", "confidence": %s}' % (stream, confidence)

    def lm(prompt: str) -> str:
        return payload

    return lm
