"""Tests for the Proposer Protocol and ReflectionProposer."""
from __future__ import annotations

import pytest
import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.archives import LatestArchive
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.common.proposers import Proposer, ReflectionProposer
from aisteer360.algorithms.input_control.common.trace import Trace

from ._stubs import make_canned_lm


def _make_trace(score: float, raw_response: str) -> Trace:
    t = torch.tensor([[1]])
    return Trace(
        input_ids=t,
        steered_input_ids=t,
        output=Output(output_ids=t),
        score=score,
        metadata={"raw_response": raw_response},
    )


class _ProtocolProposer:
    def propose(self, parent, archive):
        return []


def test_proposer_protocol_check():
    assert isinstance(_ProtocolProposer(), Proposer)


def test_reflection_proposer_returns_n_candidates():
    lm = make_canned_lm(["new instruction A", "new instruction B", "new instruction C"])
    prop = ReflectionProposer(reflection_lm=lm, n_candidates=3)
    parent = Candidate(memory=TextMemory(instruction="old"))
    archive = LatestArchive()
    archive.ingest([parent], [[]])
    children = prop.propose(parent, archive)
    assert len(children) == 3
    assert [c.memory.instruction for c in children] == [
        "new instruction A",
        "new instruction B",
        "new instruction C",
    ]


def test_reflection_proposer_carries_demonstrations():
    lm = make_canned_lm(["revised"])
    prop = ReflectionProposer(reflection_lm=lm)
    parent_mem = TextMemory(
        instruction="old",
        demonstrations=[{"q": "x", "a": "y"}],
        template="TEMPL",
        extras={"k": "v"},
    )
    parent = Candidate(memory=parent_mem)
    archive = LatestArchive()
    archive.ingest([parent], [[]])
    children = prop.propose(parent, archive)
    child_mem = children[0].memory
    assert isinstance(child_mem, TextMemory)
    assert child_mem.instruction == "revised"
    assert child_mem.demonstrations == [{"q": "x", "a": "y"}]
    assert child_mem.template == "TEMPL"
    assert child_mem.extras == {"k": "v"}


def test_reflection_proposer_rejects_non_text_memory():
    lm = make_canned_lm(["x"])
    prop = ReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory={"not": "text-memory"})
    archive = LatestArchive()
    with pytest.raises(TypeError):
        prop.propose(parent, archive)


def test_reflection_proposer_records_parent_id():
    lm = make_canned_lm(["revised"])
    prop = ReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory=TextMemory(instruction="old"), id="parent-123")
    archive = LatestArchive()
    archive.ingest([parent], [[]])
    children = prop.propose(parent, archive)
    assert children[0].metadata["parent_id"] == "parent-123"
    assert children[0].metadata["proposer"] == "ReflectionProposer"


def test_reflection_proposer_uses_traces_argument():
    seen: list[str] = []

    def lm(prompt: str) -> str:
        seen.append(prompt)
        return "revised"

    prop = ReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory=TextMemory(instruction="old"))
    archive = LatestArchive()
    # archive contains an "archive_only" trace; explicit traces should override
    archive.ingest([parent], [[_make_trace(0.0, "ARCHIVE_ONLY")]])
    explicit = [_make_trace(0.9, "EXPLICIT_TRACE")]
    prop.propose(parent, archive, traces=explicit)

    assert "EXPLICIT_TRACE" in seen[0]
    assert "ARCHIVE_ONLY" not in seen[0]


def test_reflection_proposer_backward_compat_no_traces():
    seen: list[str] = []

    def lm(prompt: str) -> str:
        seen.append(prompt)
        return "revised"

    prop = ReflectionProposer(reflection_lm=lm)
    parent = Candidate(memory=TextMemory(instruction="old"))
    archive = LatestArchive()
    archive.ingest([parent], [[_make_trace(0.0, "FROM_ARCHIVE")]])
    prop.propose(parent, archive)

    assert "FROM_ARCHIVE" in seen[0]
