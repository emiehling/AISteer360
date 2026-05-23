"""optimize: orchestrator that ties Scorer, Proposer, Archive together."""
from __future__ import annotations

from aisteer360.algorithms.input_control.common.archives.base import Archive
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.proposers.base import Proposer
from aisteer360.algorithms.input_control.common.scorers.base import Scorer


def optimize(
    initial_candidate: Candidate,
    scorer: Scorer,
    proposer: Proposer,
    archive: Archive,
    data: list[dict],
    budget: int,
) -> Candidate:
    """Run the candidate optimization loop until `budget` iterations elapse.

    Sequence:

        1. Score the initial candidate; ingest it into the archive.
        2. For each of `budget` iterations:

            a. Pick a parent via `archive.select_for_mutation()`.
            b. Ask the proposer for children.
            c. Score the children.
            d. Ingest into the archive.

        3. Return `archive.best()`.

    `budget` counts proposal iterations; the initial scoring is outside the budget. A budget of 0 returns the initial
    candidate after one scoring pass. Empty proposer output is permissible -- the loop skips ingest and moves on.

    Args:
        initial_candidate: Seed candidate (typically constructed from the method's Args / a default Memory).
        scorer: Train-time scoring component.
        proposer: Candidate-generation component.
        archive: Storage + selection component.
        data: Task examples for the Scorer.
        budget: Number of proposal iterations.

    Returns:
        The archive's best candidate after the loop completes.
    """
    initial_traces = scorer.score([initial_candidate], data)
    archive.ingest([initial_candidate], initial_traces)

    for _ in range(budget):
        parent = archive.select_for_mutation()
        children = proposer.propose(parent, archive)
        if not children:
            continue
        children_traces = scorer.score(children, data)
        archive.ingest(children, children_traces)

    return archive.best()
