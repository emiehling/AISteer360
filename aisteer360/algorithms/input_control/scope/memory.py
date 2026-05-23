"""RuleStreamMemory: dual-stream memory of timestamped, confidence-scored rules."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Rule:
    """One guideline in a SCOPE memory stream.

    Attributes:
        text: The natural-language guideline content.
        confidence: Classifier's confidence in the rule's generality (used to gate strategic-vs-tactical routing). In
            [0, 1].
        stream: "strategic" or "tactical"; which stream this rule lives in.
        created_at: Unix timestamp at synthesis.
        metadata: Free-form. Conventions:

            - "source_input_text": pre-adapt input string when synthesized
            - "source_response_text": model output that triggered synthesis
            - "synthesis_mode": "corrective" | "enhancement" | "unified"
            - "parent_rule_ids": for rules produced by Optimizer consolidation steps
    """

    text: str
    confidence: float
    stream: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleStreamMemory:
    """Dual-stream memory of timestamped, confidence-scored rules.

    The strategic stream persists across sessions; the tactical stream is reset by `SCOPE.reset_session()`.

    Used only by SCOPE in Phase 5. May be promoted to `common/memory/` if a second method adopts the shape.
    """

    strategic: list[Rule] = field(default_factory=list)
    tactical: list[Rule] = field(default_factory=list)

    model_type: str = field(default="rule_stream", init=False)

    _EXTENSION = ".rsm"

    def save(self, path: str) -> None:
        """Save the memory to a JSON file.

        Args:
            path: Path to save to. ".rsm" extension appended if not present.
        """
        if not path.endswith(self._EXTENSION):
            path += self._EXTENSION
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            "model_type": self.model_type,
            "strategic": [asdict(rule) for rule in self.strategic],
            "tactical": [asdict(rule) for rule in self.tactical],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "RuleStreamMemory":
        """Load a memory from a JSON file.

        Args:
            path: Path to load from. ".rsm" extension appended if not present.

        Returns:
            Loaded `RuleStreamMemory` instance.

        Raises:
            ValueError: If the file's `model_type` does not match this class.
        """
        if not path.endswith(cls._EXTENSION):
            path += cls._EXTENSION
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_type") != "rule_stream":
            raise ValueError(
                f"Cannot load RuleStreamMemory: file model_type is "
                f"{data.get('model_type')!r}, expected 'rule_stream'."
            )

        memory = cls()
        for rule_dict in data.get("strategic", []):
            memory.strategic.append(Rule(**rule_dict))
        for rule_dict in data.get("tactical", []):
            memory.tactical.append(Rule(**rule_dict))
        return memory

    def reset_tactical(self) -> None:
        """Clear the tactical stream; strategic is preserved."""
        self.tactical.clear()

    def all_rules(self) -> list[Rule]:
        """Strategic followed by tactical (paper-recommended ordering)."""
        return list(self.strategic) + list(self.tactical)
