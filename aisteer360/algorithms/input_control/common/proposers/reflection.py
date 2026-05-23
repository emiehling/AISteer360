"""ReflectionProposer: LLM-based critique-and-revise."""
from __future__ import annotations

from typing import Any, Callable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.trace import Trace

ReflectionLM = Callable[[str], str]


class ReflectionProposer:
    """Generate revised candidates by asking a reflection LM to critique and rewrite.

    The default prompt template assumes a `TextMemory` parent (the most common case at Phase 3). Methods with other
    memory shapes should subclass and override `_format_reflection_prompt` and `_parse_response`.

    Args:
        reflection_lm: Callable taking a prompt string and returning a response string. Methods load whatever LM they
            want behind this interface; tests pass canned callables.
        n_candidates: Number of children to produce per call.
        template: Optional override for the reflection prompt template. Defaults to a built-in template with
            placeholders for current instruction, observed traces, and revision instructions.
    """

    DEFAULT_TEMPLATE = (
        "You are improving a system prompt for a language model.\n\n"
        "Current instruction:\n{instruction}\n\n"
        "Recent execution traces (input -> output -> score):\n{traces}\n\n"
        "Propose a single revised instruction that would score higher. "
        "Respond with just the revised instruction, no preamble.\n"
    )

    def __init__(
        self,
        reflection_lm: ReflectionLM,
        n_candidates: int = 1,
        template: str | None = None,
    ) -> None:
        self.reflection_lm = reflection_lm
        self.n_candidates = n_candidates
        self.template = template or self.DEFAULT_TEMPLATE

    def propose(
        self,
        parent: Candidate,
        archive: Any,
        traces: list[Trace] | None = None,
    ) -> list[Candidate]:
        if not isinstance(parent.memory, TextMemory):
            raise TypeError(
                f"ReflectionProposer expects a TextMemory parent; got "
                f"{type(parent.memory).__name__}. Subclass to support other shapes."
            )

        effective_traces = traces if traces is not None else archive.traces_for(parent)
        prompt = self._format_reflection_prompt(parent.memory, effective_traces)

        candidates: list[Candidate] = []
        for _ in range(self.n_candidates):
            response = self.reflection_lm(prompt)
            new_memory = self._parse_response(response, parent.memory)
            candidates.append(Candidate(
                memory=new_memory,
                metadata={
                    "parent_id": parent.id,
                    "proposer": "ReflectionProposer",
                },
            ))
        return candidates

    def _format_reflection_prompt(
        self,
        memory: TextMemory,
        traces: list,
    ) -> str:
        """Format the reflection prompt; override for custom templates."""
        traces_text = "\n".join(
            f"  ({i}) score={t.score}: {t.metadata.get('raw_response', '<no response>')}"
            for i, t in enumerate(traces[:5])
        ) or "  (no traces available)"
        return self.template.format(
            instruction=memory.instruction or "(no instruction)",
            traces=traces_text,
        )

    def _parse_response(self, response: str, parent_memory: TextMemory) -> TextMemory:
        """Parse the reflection LM's response into a new TextMemory.

        Default behavior: the response is the new `instruction`; demonstrations and template carry over from the
        parent.
        """
        return TextMemory(
            instruction=response.strip(),
            demonstrations=parent_memory.demonstrations,
            template=parent_memory.template,
            extras=dict(parent_memory.extras),
        )
