"""End-to-end integration test for the optimize() loop."""
from __future__ import annotations

import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.archives import ParetoArchive
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.loop import optimize
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.proposers import ReflectionProposer
from aisteer360.algorithms.input_control.common.trace import Trace


TARGET = "target"


def _similarity(s: str) -> float:
    """Shared character count with TARGET, normalized to [0, 1]."""
    if not s:
        return 0.0
    target_chars = list(TARGET)
    matches = 0
    for c in s:
        if c in target_chars:
            target_chars.remove(c)
            matches += 1
    return matches / len(TARGET)


class _SimilarityScorer:
    """Stub scorer: scores TextMemory.instruction by similarity to TARGET."""

    def score(self, candidates, data):
        results = []
        dummy = torch.tensor([[0]])
        for cand in candidates:
            instruction = cand.memory.instruction or ""
            sim = 1.0 if instruction == TARGET else _similarity(instruction)
            trace = Trace(
                input_ids=dummy,
                steered_input_ids=dummy,
                output=Output(output_ids=dummy),
                score={"similarity": sim},
                metadata={"raw_response": instruction},
            )
            results.append([trace])
        return results


def _make_step_closer_lm():
    """Reflection LM that returns a string one character closer to TARGET than the parent's instruction.

    Strategy: read the parent's current instruction from the prompt (after "Current instruction:\n"), then return a
    string with one extra TARGET character appended (or replaced) until it equals TARGET.
    """

    def lm(prompt: str) -> str:
        marker = "Current instruction:\n"
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\n", start)
        current = prompt[start:end].strip()

        if current == TARGET:
            return TARGET

        # find the first index where current diverges from TARGET; correct that one position.
        chars = list(current)
        for i, ch in enumerate(TARGET):
            if i >= len(chars):
                chars.append(ch)
                break
            if chars[i] != ch:
                chars[i] = ch
                break
        else:
            # current is a prefix or exact match handled above; truncate excess
            chars = chars[: len(TARGET)]
        return "".join(chars[: len(TARGET)])

    return lm


def test_optimize_converges_to_target():
    initial = Candidate(memory=TextMemory(instruction="start"))
    scorer = _SimilarityScorer()
    proposer = ReflectionProposer(reflection_lm=_make_step_closer_lm(), n_candidates=1)
    archive = ParetoArchive(score_keys=["similarity"], rng_seed=0)

    best = optimize(
        initial_candidate=initial,
        scorer=scorer,
        proposer=proposer,
        archive=archive,
        data=[],
        budget=10,
    )

    assert isinstance(best.memory, TextMemory)
    assert best.memory.instruction == TARGET
