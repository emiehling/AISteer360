"""Tests for GEPAReflectionProposer and MergeProposer."""
from __future__ import annotations

import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.archives import LatestArchive
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.trace import Trace
from aisteer360.algorithms.input_control.gepa.archive import PerInstanceParetoArchive
from aisteer360.algorithms.input_control.gepa.proposers import GEPAReflectionProposer, MergeProposer


def _trace(score: float, raw: str, feedback: str | None, example: dict) -> Trace:
    t = torch.tensor([[1]])
    return Trace(
        input_ids=t,
        steered_input_ids=t,
        output=Output(output_ids=t),
        score=score,
        feedback=feedback,
        metadata={"raw_response": raw, "task_example": example},
    )


def test_gepa_reflection_uses_feedback_in_prompt():
    seen: list[str] = []

    def lm(prompt: str) -> str:
        seen.append(prompt)
        return "revised instruction"

    prop = GEPAReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory=TextMemory(instruction="orig"))
    archive = LatestArchive()
    archive.ingest([parent], [[]])

    traces = [
        _trace(0.5, "RESPONSE_A", "FEEDBACK_A", {"id": "ex1"}),
        _trace(0.2, "RESPONSE_B", "FEEDBACK_B", {"id": "ex2"}),
    ]
    children = prop.propose(parent, archive, traces=traces)

    assert len(children) == 1
    assert children[0].memory.instruction == "revised instruction"
    assert "FEEDBACK_A" in seen[0]
    assert "FEEDBACK_B" in seen[0]
    assert "RESPONSE_A" in seen[0]


def test_gepa_reflection_falls_back_to_archive():
    seen: list[str] = []

    def lm(prompt: str) -> str:
        seen.append(prompt)
        return "revised"

    prop = GEPAReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory=TextMemory(instruction="orig"))
    archive = LatestArchive()
    archive.ingest([parent], [[_trace(0.7, "ARCHIVE_RESP", "ARCHIVE_FB", {"id": "x"})]])

    prop.propose(parent, archive)
    assert "ARCHIVE_FB" in seen[0]


def test_merge_proposer_combines_two():
    def lm(prompt: str) -> str:
        return "merged instruction"

    prop = MergeProposer(reflection_lm=lm, rng_seed=0)
    arc = PerInstanceParetoArchive()
    parent = Candidate(memory=TextMemory(instruction="A"), id="parent")
    other = Candidate(memory=TextMemory(instruction="B"), id="other")
    arc.ingest(
        [parent, other],
        [
            [_trace(1.0, "x", None, {"id": "i1"})],
            [_trace(1.0, "y", None, {"id": "i2"})],
        ],
    )
    children = prop.propose(parent, arc)
    assert len(children) == 1
    assert children[0].memory.instruction == "merged instruction"
    assert set(children[0].metadata["merge_parents"]) == {"parent", "other"}
    assert children[0].metadata["proposer"] == "MergeProposer"


def test_merge_proposer_empty_when_lt_two():
    def lm(prompt: str) -> str:
        return "should not be called"

    prop = MergeProposer(reflection_lm=lm, rng_seed=0)
    arc = PerInstanceParetoArchive()
    parent = Candidate(memory=TextMemory(instruction="A"), id="parent")
    arc.ingest([parent], [[_trace(1.0, "x", None, {"id": "i1"})]])
    assert prop.propose(parent, arc) == []


def test_merge_proposer_excludes_parent():
    chosen: list[str] = []

    def lm(prompt: str) -> str:
        # capture the B-side instruction; if it ever equals "PARENT" we know parent was selected
        b_marker = prompt.split("Instruction B:\n", 1)[1].split("\n\n", 1)[0]
        chosen.append(b_marker)
        return "merged"

    prop = MergeProposer(reflection_lm=lm, rng_seed=0)
    arc = PerInstanceParetoArchive()
    parent = Candidate(memory=TextMemory(instruction="PARENT"), id="parent")
    other1 = Candidate(memory=TextMemory(instruction="OTHER1"), id="other1")
    other2 = Candidate(memory=TextMemory(instruction="OTHER2"), id="other2")
    arc.ingest(
        [parent, other1, other2],
        [
            [_trace(1.0, "p", None, {"id": "i1"})],
            [_trace(1.0, "o1", None, {"id": "i2"})],
            [_trace(1.0, "o2", None, {"id": "i3"})],
        ],
    )
    for _ in range(10):
        prop.propose(parent, arc)
    assert "PARENT" not in chosen
