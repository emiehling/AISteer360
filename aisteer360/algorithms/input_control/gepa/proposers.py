"""GEPA-specific proposers: GEPAReflectionProposer and MergeProposer."""
from __future__ import annotations

import random
from typing import Any, Callable

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.proposers.reflection import ReflectionProposer
from aisteer360.algorithms.input_control.common.trace import Trace


class GEPAReflectionProposer(ReflectionProposer):
    """Reflection proposer that uses textual feedback in its prompt.

    Subclass of `common.proposers.ReflectionProposer`. Same `propose()` shape; the difference is the prompt template
    (includes per-trace feedback) and the assumption that traces are passed explicitly by the caller (GEPA's loop).
    Falls back to `archive.traces_for(parent)` only when `traces` is None.
    """

    DEFAULT_TEMPLATE = (
        "You are improving an instruction for a language model.\n\n"
        "Current instruction:\n{instruction}\n\n"
        "The current instruction was applied to the following examples, "
        "producing these responses, scores, and feedback:\n\n{traces}\n\n"
        "Propose a single revised instruction that addresses the feedback "
        "above. Respond with ONLY the revised instruction, with no "
        "preamble or explanation.\n"
    )

    def _format_reflection_prompt(self, memory: TextMemory, traces: list[Trace]) -> str:
        trace_blocks = []
        for i, t in enumerate(traces):
            response = t.metadata.get("raw_response", "<no response>") if t.metadata else "<no response>"
            feedback = t.feedback or "(no feedback)"
            trace_blocks.append(
                f"Example {i + 1}:\n"
                f"  Response: {response}\n"
                f"  Score: {t.score}\n"
                f"  Feedback: {feedback}\n"
            )
        return self.template.format(
            instruction=memory.instruction or "(no instruction)",
            traces="\n".join(trace_blocks) or "(no traces)",
        )


class MergeProposer:
    """Combine two TextMemory candidates into a third via the reflection LM.

    Selects the parent and one other archive member at random; asks the reflection LM to combine their complementary
    strengths into a single new instruction. Returns one new Candidate per call.

    If the archive has fewer than 2 members, returns `[]`. The caller's loop is responsible for handling the empty
    case.

    Args:
        reflection_lm: Callable taking a prompt string and returning a response string.
        template: Optional override for the merge prompt template.
        rng_seed: Optional seed for reproducibility of the partner-candidate selection.
    """

    DEFAULT_TEMPLATE = (
        "You are combining two candidate instructions for a language model. "
        "Each was evolved separately; your task is to produce a single "
        "unified instruction that inherits the complementary strengths of "
        "both.\n\n"
        "Instruction A:\n{instruction_a}\n\n"
        "Instruction B:\n{instruction_b}\n\n"
        "Respond with ONLY the merged instruction, with no preamble or "
        "explanation.\n"
    )

    def __init__(
        self,
        reflection_lm: Callable[[str], str],
        template: str | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.reflection_lm = reflection_lm
        self.template = template or self.DEFAULT_TEMPLATE
        self._rng = random.Random(rng_seed)

    def propose(
        self,
        parent: Candidate,
        archive: Any,
        traces: list[Trace] | None = None,
    ) -> list[Candidate]:
        if not isinstance(parent.memory, TextMemory):
            raise TypeError(
                f"MergeProposer expects a TextMemory parent; got {type(parent.memory).__name__}."
            )

        members = [m for m in archive.members() if m.id != parent.id]
        if not members:
            return []

        other = self._rng.choice(members)
        if not isinstance(other.memory, TextMemory):
            raise TypeError(
                f"MergeProposer expects a TextMemory partner; got {type(other.memory).__name__}."
            )

        prompt = self.template.format(
            instruction_a=parent.memory.instruction or "",
            instruction_b=other.memory.instruction or "",
        )
        response = self.reflection_lm(prompt)

        new_memory = TextMemory(
            instruction=response.strip(),
            demonstrations=parent.memory.demonstrations,
            template=parent.memory.template,
            extras=dict(parent.memory.extras),
        )
        return [Candidate(
            memory=new_memory,
            metadata={
                "merge_parents": [parent.id, other.id],
                "proposer": "MergeProposer",
            },
        )]
