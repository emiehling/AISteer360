"""TextMemory: container for text-based prompt artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextMemory:
    """Memory for methods that carry textual prompt artifacts.

    Used by methods that optimize or hold fixed text: FewShot (fixed memory, no optimization), GEPA, MIPROv2 (optimized
    instruction + demos), TextGrad (text variables at graph nodes, typically with the user-facing instruction at the
    input node), etc.

    Attributes:
        instruction: Single instruction string prepended to the user query in `adapt()`. None if the method doesn't
            carry an instruction.
        demonstrations: Pool of example dicts. Each entry is a free-form mapping; conventions like
            `_label: "positive" | "negative"` are used by methods that need to distinguish example classes (e.g.
            FewShot). Methods that do not need labels simply omit the field.
        template: Optional template string for assembling the system prompt from `instruction` and `demonstrations`.
            None means the consumer uses its own default.
        extras: Method-specific metadata that doesn't fit the above shape. Use sparingly; prefer subclassing TextMemory
            if a field is structurally important.
    """

    instruction: str | None = None
    demonstrations: list[dict[str, Any]] | None = None
    template: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    model_type: str = field(default="text", init=False)

    _EXTENSION = ".tmem"

    def save(self, path: str) -> None:
        """Save the TextMemory to a JSON file.

        Args:
            path: Path to save to. ".tmem" extension appended if not present.
        """
        if not path.endswith(self._EXTENSION):
            path += self._EXTENSION
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            "model_type": self.model_type,
            "instruction": self.instruction,
            "demonstrations": self.demonstrations,
            "template": self.template,
            "extras": self.extras,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TextMemory":
        """Load a TextMemory from a JSON file.

        Args:
            path: Path to load from. ".tmem" extension appended if not present.

        Returns:
            Loaded TextMemory instance.

        Raises:
            ValueError: If the file's `model_type` does not match this class.
        """
        if not path.endswith(cls._EXTENSION):
            path += cls._EXTENSION
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_type") != "text":
            raise ValueError(
                f"Cannot load TextMemory: file model_type is "
                f"{data.get('model_type')!r}, expected 'text'."
            )

        return cls(
            instruction=data.get("instruction"),
            demonstrations=data.get("demonstrations"),
            template=data.get("template"),
            extras=data.get("extras", {}),
        )
