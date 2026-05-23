"""SCOPE meta-agents: Generator, Selector, Classifier, Optimizer.

Each class wraps a prompt template and a reflection-LM callable. Output parsing is best-effort with safe fallbacks so a
single noisy LM response never breaks the whole `observe()` call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Literal

from aisteer360.algorithms.input_control.scope.memory import Rule, RuleStreamMemory
from aisteer360.algorithms.input_control.scope.templates import (
    DEFAULT_CLASSIFIER_TEMPLATE,
    DEFAULT_GENERATOR_TEMPLATE,
    DEFAULT_OPTIMIZER_CONFLICT_TEMPLATE,
    DEFAULT_OPTIMIZER_CONSOLIDATION_TEMPLATE,
    DEFAULT_OPTIMIZER_SUBSUMPTION_TEMPLATE,
    DEFAULT_SELECTOR_TEMPLATE,
)

logger = logging.getLogger(__name__)

ReflectionLM = Callable[[str], str]


def _format_rules(memory: RuleStreamMemory) -> str:
    """Render the dual-stream memory as a labeled bulleted list for prompt placeholders."""
    lines: list[str] = []
    if memory.strategic:
        lines.append("Strategic:")
        for rule in memory.strategic:
            lines.append(f"- {rule.text}")
    if memory.tactical:
        lines.append("Tactical:")
        for rule in memory.tactical:
            lines.append(f"- {rule.text}")
    return "\n".join(lines) if lines else "(none)"


def _format_rule_list(rules: list[Rule]) -> str:
    if not rules:
        return "(none)"
    return "\n".join(f"- {rule.text}" for rule in rules)


class GuidelineGenerator:
    """Synthesize N candidate guidelines from `(input, response, current_memory)`.

    Calls the reflection LM `n_candidates` times with the synthesis template and returns the parsed guideline strings.
    Empty responses are dropped.
    """

    def __init__(
        self,
        lm: ReflectionLM,
        n_candidates: int = 2,
        template: str | None = None,
    ) -> None:
        self.lm = lm
        self.n_candidates = max(1, int(n_candidates))
        self.template = template or DEFAULT_GENERATOR_TEMPLATE

    def synthesize(
        self,
        input_text: str,
        response_text: str,
        current_memory: RuleStreamMemory,
    ) -> list[str]:
        prompt = self.template.format(
            current_rules=_format_rules(current_memory),
            input_text=input_text,
            response_text=response_text,
        )

        candidates: list[str] = []
        for _ in range(self.n_candidates):
            try:
                raw = self.lm(prompt)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Generator LM call failed: %s", exc)
                continue
            text = (raw or "").strip()
            if text:
                candidates.append(text)
        return candidates


class GuidelineSelector:
    """Best-of-N selection over candidate guidelines.

    Single LM call. Parses an integer index from the LM output. Falls back to the first candidate if parsing fails or
    the index is out of range.
    """

    def __init__(self, lm: ReflectionLM, template: str | None = None) -> None:
        self.lm = lm
        self.template = template or DEFAULT_SELECTOR_TEMPLATE

    def select(
        self,
        candidates: list[str],
        current_memory: RuleStreamMemory,
        input_text: str,
        response_text: str,
    ) -> str:
        if not candidates:
            raise ValueError("Selector requires at least one candidate.")
        if len(candidates) == 1:
            return candidates[0]

        rendered = "\n".join(f"{i}: {c}" for i, c in enumerate(candidates))
        prompt = self.template.format(
            current_rules=_format_rules(current_memory),
            input_text=input_text,
            response_text=response_text,
            candidates=rendered,
        )

        try:
            raw = self.lm(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Selector LM call failed: %s", exc)
            return candidates[0]

        match = re.search(r"-?\d+", raw or "")
        if match is None:
            return candidates[0]
        try:
            idx = int(match.group(0))
        except ValueError:
            return candidates[0]
        if 0 <= idx < len(candidates):
            return candidates[idx]
        return candidates[0]


class GuidelineClassifier:
    """Route a guideline to "strategic" or "tactical" with confidence in [0, 1].

    Single LM call. Parses a JSON object of the form `{"category": ..., "confidence": ...}`. Falls back to
    `("tactical", 0.0)` on parse failure.
    """

    def __init__(self, lm: ReflectionLM, template: str | None = None) -> None:
        self.lm = lm
        self.template = template or DEFAULT_CLASSIFIER_TEMPLATE

    def classify(
        self,
        guideline: str,
        current_memory: RuleStreamMemory,
    ) -> tuple[Literal["strategic", "tactical"], float]:
        prompt = self.template.format(
            current_rules=_format_rules(current_memory),
            guideline=guideline,
        )

        try:
            raw = self.lm(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Classifier LM call failed: %s", exc)
            return ("tactical", 0.0)

        return _parse_classifier_output(raw or "")


def _parse_classifier_output(raw: str) -> tuple[Literal["strategic", "tactical"], float]:
    """Find the first JSON object in `raw` and read `category` + `confidence`."""
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match is None:
        return ("tactical", 0.0)
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return ("tactical", 0.0)

    category = data.get("category")
    if category not in ("strategic", "tactical"):
        return ("tactical", 0.0)

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return (category, confidence)


class MemoryOptimizer:
    """Consolidate the strategic stream when it exceeds capacity.

    Three-step pipeline:

      1. Conflict Resolution: merge contradictory rules.
      2. Subsumption Pruning: drop specific rules covered by general ones.
      3. Consolidation: merge similar rules into comprehensive ones.

    Each step is a separate LM call; output is parsed into bullet lines. The optimizer returns a new list of strategic
    `Rule`s with length <= the original input length.
    """

    def __init__(
        self,
        lm: ReflectionLM,
        conflict_template: str | None = None,
        subsumption_template: str | None = None,
        consolidation_template: str | None = None,
    ) -> None:
        self.lm = lm
        self.conflict_template = conflict_template or DEFAULT_OPTIMIZER_CONFLICT_TEMPLATE
        self.subsumption_template = subsumption_template or DEFAULT_OPTIMIZER_SUBSUMPTION_TEMPLATE
        self.consolidation_template = consolidation_template or DEFAULT_OPTIMIZER_CONSOLIDATION_TEMPLATE

    def consolidate(self, strategic_rules: list[Rule]) -> list[Rule]:
        if len(strategic_rules) <= 1:
            return list(strategic_rules)

        original_count = len(strategic_rules)
        rules = list(strategic_rules)

        rules = self._step(rules, self.conflict_template)
        rules = self._step(rules, self.subsumption_template)
        rules = self._step(rules, self.consolidation_template)

        if len(rules) > original_count:
            rules = rules[:original_count]
        return rules

    def _step(self, rules: list[Rule], template: str) -> list[Rule]:
        if not rules:
            return rules
        prompt = template.format(rules=_format_rule_list(rules))
        try:
            raw = self.lm(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Optimizer LM call failed: %s", exc)
            return rules

        new_texts = _parse_bullet_lines(raw or "")
        if not new_texts:
            return rules

        parent_ids = [id(rule) for rule in rules]
        timestamp = max(rule.created_at for rule in rules)

        new_rules: list[Rule] = []
        for text in new_texts:
            new_rules.append(
                Rule(
                    text=text,
                    confidence=max(rule.confidence for rule in rules),
                    stream="strategic",
                    created_at=timestamp,
                    metadata={"parent_rule_ids": parent_ids},
                )
            )
        return new_rules


def _parse_bullet_lines(raw: str) -> list[str]:
    """Extract one bullet/line per guideline; tolerant of `-`, `*`, and `1.` style prefixes."""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[\-\*•]\s*", "", stripped)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
        if cleaned:
            lines.append(cleaned)
    return lines
