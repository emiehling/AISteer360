"""The record of a study's results.

The archive stores candidates and their results in the store (`candidates` and `results` tables) and answers queries
on them. It generates nothing. Results are append-only versions, and a search engine's population is a working copy
of the archive, which is authoritative. `ArchiveView` is the part a proposer may see, i.e., per candidate its id,
structure, values, status, stage, reason, fitness, and summary, and nothing item-level, together with whether a new
structural fit may run (`fits_open`).
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.store import StoreIntegrityError
from steerability.optimization.evaluation.result import Result, jsonable
from steerability.optimization.study import Candidate, dumps, ensure_tables, now

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.optimization.study import Study


@dataclass(frozen=True)
class ArchiveEntry:
    """What a proposer may see of one candidate: its latest result without item-level data.

    Attributes:
        candidate_id: The candidate's id.
        structure: The candidate's structure.
        values: The candidate's proposer-axis values.
        status: `"accepted"` or `"rejected"`.
        stage: The stage that decided the result.
        reason: The rejection reason, or None.
        fitness: The fitness.
        summary: The result's summary.
    """

    candidate_id: str
    structure: str
    values: Mapping[str, Any]
    status: str
    stage: int
    reason: str | None
    fitness: float
    summary: Mapping[str, float | str]


@dataclass(frozen=True)
class ArchiveView:
    """The proposer's view of an archive, one entry per candidate in recording order.

    Attributes:
        entries: The entries.
        fits_open: Whether a new structural fit (a training or a merge) may run now. A view from
            `StudySession.view` sets it from the session's limits on structural fits. When it is false, a candidate
            that needs a new structural fit comes back with the reason `budget` and is not recorded.
    """

    entries: tuple[ArchiveEntry, ...]
    fits_open: bool = True

    def __iter__(self) -> Iterator[ArchiveEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, candidate_id: object) -> bool:
        return any(entry.candidate_id == candidate_id for entry in self.entries)

    @property
    def ids(self) -> frozenset[str]:
        """The ids of the recorded candidates."""
        return frozenset(entry.candidate_id for entry in self.entries)

    def get(self, candidate_id: str) -> ArchiveEntry | None:
        """The entry of a candidate, or None."""
        for entry in self.entries:
            if entry.candidate_id == candidate_id:
                return entry
        return None

    def accepted(self) -> list[ArchiveEntry]:
        """The accepted entries, by fitness, highest first."""
        return sorted((entry for entry in self.entries if entry.status == "accepted"),
                      key=lambda entry: (-entry.fitness, entry.candidate_id))


class Archive:
    """The stored candidates and results of one study.

    Args:
        study: The study.
        store: The store the archive reads and writes.
    """

    def __init__(self, study: Study, store: Store) -> None:
        self.study = study
        self.store = store
        self._candidates: dict[str, Candidate] = {}
        ensure_tables(store)

    def record(self, candidate: Candidate, result: Result, *, proposer: str) -> None:
        """Record a candidate (once) and a version of its result.

        Recording the same version again with the same content does nothing.

        Args:
            candidate: The candidate.
            result: The result; its `version` is the next version of the candidate.
            proposer: The name of the proposer that proposed the candidate.

        Raises:
            StoreIntegrityError: If the version is already recorded with a different result.
        """
        self._candidates[result.candidate_id] = candidate
        self.store.execute(
            """INSERT OR IGNORE INTO candidates (study_id, candidate_id, structure, assignment, proposer, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (self.study.id, result.candidate_id, candidate.structure, dumps(jsonable(dict(candidate.values))),
             proposer, now()),
        )
        text = dumps(result.to_json())
        self.store.execute(
            """INSERT OR IGNORE INTO results (study_id, candidate_id, version, round_index, result, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (self.study.id, result.candidate_id, result.version, result.round_index, text, now()),
        )
        stored = self.store.execute(
            "SELECT result FROM results WHERE study_id = ? AND candidate_id = ? AND version = ?",
            (self.study.id, result.candidate_id, result.version),
        ).fetchone()
        if stored[0] != text:
            raise StoreIntegrityError(
                f"Version {result.version} of candidate {result.candidate_id} is recorded with a different result."
            )

    def candidate(self, candidate_id: str) -> Candidate | None:
        """The candidate with an id, from this process or from its recorded structure and values, or None."""
        if candidate_id in self._candidates:
            return self._candidates[candidate_id]
        row = self.store.execute(
            "SELECT structure, assignment FROM candidates WHERE study_id = ? AND candidate_id = ?",
            (self.study.id, candidate_id),
        ).fetchone()
        return None if row is None else Candidate(row[0], json.loads(row[1]))

    def remember(self, candidate_id: str, candidate: Candidate) -> None:
        """Keep a candidate object in this process (its values then need not be read back from their JSON form)."""
        self._candidates[candidate_id] = candidate

    def versions(self, candidate_id: str) -> list[Result]:
        """Every recorded version of a candidate's result, oldest first."""
        rows = self.store.execute(
            "SELECT result FROM results WHERE study_id = ? AND candidate_id = ? ORDER BY version",
            (self.study.id, candidate_id),
        ).fetchall()
        return [Result.from_json(json.loads(row[0])) for row in rows]

    def latest(self, candidate_id: str) -> Result | None:
        """The latest version of a candidate's result, or None."""
        row = self.store.execute(
            "SELECT result FROM results WHERE study_id = ? AND candidate_id = ? ORDER BY version DESC LIMIT 1",
            (self.study.id, candidate_id),
        ).fetchone()
        return None if row is None else Result.from_json(json.loads(row[0]))

    def results(self) -> list[Result]:
        """The latest version of every candidate's result, in recording order."""
        rows = self.store.execute(
            """SELECT r.result FROM results r JOIN (SELECT candidate_id, MAX(version) AS version FROM results
                WHERE study_id = ? GROUP BY candidate_id) m ON r.candidate_id = m.candidate_id AND r.version = m.version
                JOIN candidates c ON c.study_id = r.study_id AND c.candidate_id = r.candidate_id
                WHERE r.study_id = ? ORDER BY c.rowid""",
            (self.study.id, self.study.id),
        ).fetchall()
        return [Result.from_json(json.loads(row[0])) for row in rows]

    def first_versions(self, round_index: int) -> int:
        """The number of candidates whose first result version was recorded in a round."""
        row = self.store.execute(
            "SELECT COUNT(*) FROM results WHERE study_id = ? AND version = 1 AND round_index = ?",
            (self.study.id, round_index),
        ).fetchone()
        return int(row[0])

    def points(self) -> list[Result]:
        """The latest versions that are accepted."""
        return [result for result in self.results() if result.accepted]

    def best(self) -> Result | None:
        """The accepted result with the highest fitness, or None."""
        points = self.points()
        return max(points, key=lambda result: (result.fitness, result.candidate_id)) if points else None

    def envelope(self, levels: Sequence[float], *, among: Sequence[Result] | None = None) -> list[Result | None]:
        """For each level, the accepted result with the highest fitness among those at or below that level.

        Args:
            levels: The levels.
            among: The accepted results to choose from, or None for every accepted result.

        Returns:
            One result (or None) per level.
        """
        points = self.points() if among is None else list(among)
        chosen: list[Result | None] = []
        for level in levels:
            inside = [result for result in points if result.level is not None and result.level <= level + 1e-12]
            chosen.append(max(inside, key=lambda result: (result.fitness, -result.level, result.candidate_id))
                          if inside else None)
        return chosen

    def view(self) -> ArchiveView:
        """The proposer's view of the archive."""
        rows = self.store.execute(
            """SELECT c.candidate_id, c.structure, c.assignment, r.result FROM candidates c JOIN results r
                ON r.study_id = c.study_id AND r.candidate_id = c.candidate_id
                JOIN (SELECT candidate_id, MAX(version) AS version FROM results WHERE study_id = ?
                GROUP BY candidate_id) m ON r.candidate_id = m.candidate_id AND r.version = m.version
                WHERE c.study_id = ? ORDER BY c.rowid""",
            (self.study.id, self.study.id),
        ).fetchall()
        entries = []
        for candidate_id, structure, assignment, text in rows:
            result = json.loads(text)
            entries.append(ArchiveEntry(
                candidate_id=candidate_id, structure=structure, values=json.loads(assignment),
                status=result["status"], stage=int(result["stage"]), reason=result.get("reason"),
                fitness=float(result["fitness"]), summary=dict(result.get("summary") or {}),
            ))
        return ArchiveView(tuple(entries))


__all__ = ["Archive", "ArchiveEntry", "ArchiveView"]
