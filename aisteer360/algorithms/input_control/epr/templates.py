"""Default templates for EPR prompt assembly."""
from __future__ import annotations

DEFAULT_DEMO_TEMPLATE = "Input: {input}\nOutput: {output}"
DEFAULT_DEMO_SEPARATOR = "\n\n"
DEFAULT_FINAL_TEMPLATE = "{demonstrations}\n\nInput: {query}\nOutput:"
