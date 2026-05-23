"""Candidate: a memory snapshot with identity and provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Candidate:
    """A snapshot of a Memory plus identity and provenance.

    A Candidate is the unit that flows through the optimization loop: Proposers produce them, Scorers score them,
    Archives store and select them. The Memory is the thing being optimized; everything else on the Candidate is
    metadata for the loop's bookkeeping.

    Attributes:
        memory: The memory snapshot (anything satisfying the Memory Protocol).
        id: Stable identifier, auto-generated UUID by default. Used by archives to track lineage and identify
            candidates without relying on object equality.
        metadata: Free-form attachments. Conventions used by specific proposers and archives:

            - `"parent_id"`: id of the candidate this one was derived from
            - `"proposer"`: name of the proposer that produced this candidate
            - `"step"`: optimization iteration index
            - `"merge_parents"`: list of ids when this is a merge product

            The framework treats this opaquely.
    """

    memory: Any
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
