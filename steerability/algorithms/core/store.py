"""The persistent, content-addressed record of fits, rollouts, scores, and per-item evidence.

A store is a directory:

```
<root>/
  store.json        {"schema": "store/1"}
  artifacts/        an ArtifactStore (spipe/store.py)
  index.sqlite      SQLite, WAL journal by default, busy timeout 30000 ms
  logs/             Inspect logs written by benchmark calls; not authoritative
```

Every row is written once (`INSERT OR IGNORE`) and never updated or deleted, with one exception:
a protocol's observed facts are recorded by the first load that observes them. Each `put_*`
returns the row stored afterwards, which is the earlier row when the insert was ignored. For
recipes and resolutions a differing earlier row raises `StoreIntegrityError`. For the other
tables the first writer wins, since repeated generation and fitting need not be bitwise equal.
Several threads and processes may use one store; each thread opens its own connection.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from steerability.algorithms.core.protocol import ProtocolFacts, ProtocolMismatchError

if TYPE_CHECKING:
    from steerability.algorithms.core.protocol import Protocol
    from steerability.algorithms.core.recipe import Recipe
    from steerability.spipe.store import ArtifactStore

STORE_SCHEMA = "store/1"
STORE_FILE = "store.json"
INDEX_FILE = "index.sqlite"
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"
BUSY_TIMEOUT_MS = 30000

_TABLES = (
    """CREATE TABLE IF NOT EXISTS protocols (protocol_id TEXT PRIMARY KEY, identity TEXT NOT NULL, observed TEXT,
        created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS recipes (recipe_id TEXT PRIMARY KEY, canonical BLOB NOT NULL)",
    """CREATE TABLE IF NOT EXISTS fits (fit_key TEXT PRIMARY KEY, fit_context_id TEXT NOT NULL,
        weights_context TEXT NOT NULL, fit_digest TEXT NOT NULL, value TEXT NOT NULL, seconds REAL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS resolutions (protocol_id TEXT NOT NULL, recipe_id TEXT NOT NULL,
        resolved_id TEXT NOT NULL, PRIMARY KEY (protocol_id, recipe_id))""",
    """CREATE TABLE IF NOT EXISTS items (benchmark_id TEXT NOT NULL, item_id TEXT NOT NULL, group_id TEXT,
        payload TEXT NOT NULL, PRIMARY KEY (benchmark_id, item_id))""",
    """CREATE TABLE IF NOT EXISTS rollouts (rollout_key TEXT PRIMARY KEY, protocol_id TEXT NOT NULL,
        resolved_id TEXT NOT NULL, completion TEXT NOT NULL, finish_reason TEXT, tokens INTEGER,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS scores (score_key TEXT PRIMARY KEY, scorer_id TEXT NOT NULL, value REAL NOT NULL,
        detail TEXT)""",
    """CREATE TABLE IF NOT EXISTS evidence (protocol_id TEXT NOT NULL, resolved_id TEXT NOT NULL,
        benchmark_id TEXT NOT NULL, item_id TEXT NOT NULL, replicate INTEGER NOT NULL DEFAULT 0,
        value REAL NOT NULL, status TEXT NOT NULL, tokens INTEGER, rollout_key TEXT, detail TEXT,
        created_at TEXT NOT NULL, PRIMARY KEY (protocol_id, resolved_id, benchmark_id, item_id, replicate))""",
    "CREATE TABLE IF NOT EXISTS schemas (name TEXT PRIMARY KEY, created_at TEXT NOT NULL)",
)


class StoreIntegrityError(RuntimeError):
    """A write-once row would be replaced by a different row, or the store directory is not a store."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


@dataclass(frozen=True)
class ProtocolRow:
    """A stored protocol: its identity and the facts observed on its first load (or None)."""

    protocol_id: str
    identity: dict
    observed: dict | None
    created_at: str


@dataclass(frozen=True)
class FitRow:
    """A stored fit.

    Attributes:
        fit_key: The key of the fit (protocol fit context, weights, and fit digest).
        fit_context_id: The protocol's fit context id.
        weights_context: The recipe's weights context.
        fit_digest: The fit's digest.
        value: The fitted value's encoded JSON, whose artifacts are in the store's `artifacts`.
        seconds: The duration of the fit.
        created_at: When the row was written.
    """

    fit_key: str
    fit_context_id: str
    weights_context: str
    fit_digest: str
    value: str
    seconds: float | None
    created_at: str


@dataclass(frozen=True)
class ItemRow:
    """A stored benchmark item.

    Attributes:
        benchmark_id: The benchmark's id.
        item_id: The item's id as a string.
        group_id: The id of the item's group, or None.
        payload: The item's original id, input, target, and metadata.
    """

    benchmark_id: str
    item_id: str
    group_id: str | None
    payload: dict


@dataclass(frozen=True)
class RolloutRow:
    """A stored generation: the text a scorer sees, with its finish reason and token count."""

    rollout_key: str
    protocol_id: str
    resolved_id: str
    completion: str
    finish_reason: str | None
    tokens: int | None
    created_at: str


@dataclass(frozen=True)
class ScoreRow:
    """A stored score of one completion by one scorer."""

    score_key: str
    scorer_id: str
    value: float
    detail: Any


@dataclass(frozen=True)
class EvidenceRow:
    """A stored measurement of one item under one resolved recipe and protocol.

    Attributes:
        protocol_id: The protocol's id.
        resolved_id: The resolved recipe's id.
        benchmark_id: The benchmark's id.
        item_id: The item's id.
        replicate: The replicate index.
        value: The chosen scorer's value.
        status: `"ok"`, or `"error"` for an item whose generation or scoring failed.
        tokens: The tokens generated for the item, or None.
        rollout_key: The key of the stored rollout, or None for a row produced through Inspect.
        detail: Every scorer's raw value under `"scores"`, and the completion under
            `"completion"` when `rollout_key` is None.
        created_at: When the row was written.
    """

    protocol_id: str
    resolved_id: str
    benchmark_id: str
    item_id: str
    replicate: int
    value: float
    status: str
    tokens: int | None
    rollout_key: str | None
    detail: Any
    created_at: str


class Store:
    """The persistent store; build with `Store.open`.

    Args:
        root: The store directory.
        journal_mode: The SQLite journal mode.
    """

    def __init__(self, root: Path, journal_mode: Literal["WAL", "DELETE"]) -> None:
        self.root = Path(root)
        self.journal_mode = journal_mode
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._artifacts: ArtifactStore | None = None

    @classmethod
    def open(cls, root: str | Path, *, journal_mode: Literal["WAL", "DELETE"] = "WAL") -> Store:
        """Open the store at `root`, creating it when the directory is new or empty.

        On a network filesystem whose locking SQLite cannot rely on, open the store on local disk
        or use `journal_mode="DELETE"` with a single writing process.

        Args:
            root: The store directory.
            journal_mode: `"WAL"` (default) or `"DELETE"`.

        Returns:
            The store.

        Raises:
            StoreIntegrityError: If `root` contains a `store.json` of another schema.
            ValueError: If `journal_mode` is not recognized.
        """
        if journal_mode not in ("WAL", "DELETE"):
            raise ValueError(f"journal_mode must be 'WAL' or 'DELETE'; got {journal_mode!r}.")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        marker = root / STORE_FILE
        if marker.exists():
            schema = _loads(marker.read_text(encoding="utf-8")).get("schema")
            if schema != STORE_SCHEMA:
                raise StoreIntegrityError(
                    f"{root} is a store of schema {schema!r}; this toolkit reads {STORE_SCHEMA!r}."
                )
        else:
            temporary = root / f"{STORE_FILE}.tmp"
            temporary.write_text(json.dumps({"schema": STORE_SCHEMA}) + "\n", encoding="utf-8")
            temporary.replace(marker)
        (root / ARTIFACTS_DIR).mkdir(exist_ok=True)
        (root / LOGS_DIR).mkdir(exist_ok=True)
        store = cls(root, journal_mode)
        connection = store._connection()
        for statement in _TABLES:
            connection.execute(statement)
        return store

    @property
    def artifacts(self) -> ArtifactStore:
        """The content-addressed artifact store under `artifacts/`."""
        if self._artifacts is None:
            from steerability.spipe.store import ArtifactStore

            self._artifacts = ArtifactStore(self.root / ARTIFACTS_DIR)
        return self._artifacts

    @property
    def logs(self) -> Path:
        """The directory Inspect logs are written under."""
        return self.root / LOGS_DIR

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self.root / INDEX_FILE, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None, check_same_thread=False,
            )
            connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            connection.execute(f"PRAGMA journal_mode={self.journal_mode}")
            self._local.connection = connection
            with self._lock:
                self._connections.append(connection)
        return connection

    def close(self) -> None:
        """Close every connection the store opened. The store reopens connections on next use."""
        with self._lock:
            connections, self._connections = self._connections, []
        for connection in connections:
            connection.close()
        self._local = threading.local()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def ensure_schema(self, name: str, statements: Sequence[str]) -> None:
        """Create extension tables once, recording `name` as applied.

        Args:
            name: The extension schema name, e.g. `"direct/1"`.
            statements: Idempotent `CREATE TABLE IF NOT EXISTS` statements.
        """
        connection = self._connection()
        for statement in statements:
            connection.execute(statement)
        connection.execute("INSERT OR IGNORE INTO schemas (name, created_at) VALUES (?, ?)", (name, _now()))

    def execute(self, sql: str, parameters: Sequence = ()) -> sqlite3.Cursor:
        """Run one statement on the index, for the tables an extension created with `ensure_schema`.

        Args:
            sql: The statement.
            parameters: Its parameters.

        Returns:
            The cursor.
        """
        return self._connection().execute(sql, parameters)

    def _execute(self, sql: str, parameters: Sequence = ()) -> sqlite3.Cursor:
        return self._connection().execute(sql, parameters)

    # protocols

    def put_protocol(self, protocol: Protocol, facts: ProtocolFacts | None = None) -> ProtocolRow:
        """Record a protocol's identity, and its observed facts when none are recorded yet.

        Args:
            protocol: The protocol.
            facts: Facts observed on a model load, or None.

        Returns:
            The stored row.

        Raises:
            ProtocolMismatchError: If `facts` conflicts with the recorded facts.
        """
        self._execute(
            "INSERT OR IGNORE INTO protocols (protocol_id, identity, observed, created_at) VALUES (?, ?, ?, ?)",
            (protocol.id, _dumps(protocol.identity()), None, _now()),
        )
        if facts is not None:
            stored = self.get_protocol(protocol.id)
            if stored.observed is None:
                self._execute(
                    "UPDATE protocols SET observed = ? WHERE protocol_id = ? AND observed IS NULL",
                    (_dumps(facts.to_dict()), protocol.id),
                )
            stored = self.get_protocol(protocol.id)
            conflicts = ProtocolFacts.from_dict(stored.observed).conflicts(facts)
            if conflicts:
                raise ProtocolMismatchError(
                    f"The loaded model does not match the facts recorded for protocol {protocol.id}: "
                    + "; ".join(conflicts) + "."
                )
        return self.get_protocol(protocol.id)

    def get_protocol(self, protocol_id: str) -> ProtocolRow | None:
        """The stored protocol row, or None."""
        row = self._execute(
            "SELECT protocol_id, identity, observed, created_at FROM protocols WHERE protocol_id = ?", (protocol_id,),
        ).fetchone()
        if row is None:
            return None
        return ProtocolRow(row[0], _loads(row[1]), _loads(row[2]), row[3])

    # recipes

    def put_recipe(self, recipe: Recipe) -> dict:
        """Record a recipe's canonical form.

        Returns:
            The stored canonical form.

        Raises:
            StoreIntegrityError: If a different canonical form is stored under the recipe's id.
        """
        canonical = recipe.canonical()
        self._execute(
            "INSERT OR IGNORE INTO recipes (recipe_id, canonical) VALUES (?, ?)",
            (recipe.id, zlib.compress(_dumps(canonical).encode("utf-8"))),
        )
        stored = self.get_recipe(recipe.id)
        if stored != json.loads(_dumps(canonical)):
            raise StoreIntegrityError(f"Recipe {recipe.id} is stored with a different canonical form.")
        return stored

    def get_recipe(self, recipe_id: str) -> dict | None:
        """The stored canonical form of a recipe, or None."""
        row = self._execute("SELECT canonical FROM recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()
        return None if row is None else json.loads(zlib.decompress(row[0]).decode("utf-8"))

    # fits

    def put_fit(self, row: FitRow) -> FitRow:
        """Record a fit; the first writer of a key wins.

        Returns:
            The stored row.
        """
        self._execute(
            """INSERT OR IGNORE INTO fits (fit_key, fit_context_id, weights_context, fit_digest, value, seconds,
                created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row.fit_key, row.fit_context_id, row.weights_context, row.fit_digest, row.value, row.seconds,
             row.created_at),
        )
        return self.get_fit(row.fit_key)

    def get_fit(self, fit_key: str) -> FitRow | None:
        """The stored fit row, or None."""
        row = self._execute(
            """SELECT fit_key, fit_context_id, weights_context, fit_digest, value, seconds, created_at FROM fits
                WHERE fit_key = ?""",
            (fit_key,),
        ).fetchone()
        return None if row is None else FitRow(*row)

    # resolutions

    def put_resolution(self, protocol_id: str, recipe_id: str, resolved_id: str) -> str:
        """Record the resolved id of a recipe under a protocol.

        Returns:
            The stored resolved id.

        Raises:
            StoreIntegrityError: If the recipe already resolves to a different id under the protocol.
        """
        self._execute(
            "INSERT OR IGNORE INTO resolutions (protocol_id, recipe_id, resolved_id) VALUES (?, ?, ?)",
            (protocol_id, recipe_id, resolved_id),
        )
        stored = self.get_resolution(protocol_id, recipe_id)
        if stored != resolved_id:
            raise StoreIntegrityError(
                f"Recipe {recipe_id} resolves to {stored} under protocol {protocol_id}, not {resolved_id}; a fit "
                "was replaced or the stored fits are inconsistent."
            )
        return stored

    def get_resolution(self, protocol_id: str, recipe_id: str) -> str | None:
        """The stored resolved id of a recipe under a protocol, or None."""
        row = self._execute(
            "SELECT resolved_id FROM resolutions WHERE protocol_id = ? AND recipe_id = ?", (protocol_id, recipe_id),
        ).fetchone()
        return None if row is None else row[0]

    # items

    def put_items(self, rows: Iterable[ItemRow]) -> None:
        """Record benchmark items; an item already stored is kept."""
        connection = self._connection()
        connection.execute("BEGIN")
        try:
            connection.executemany(
                "INSERT OR IGNORE INTO items (benchmark_id, item_id, group_id, payload) VALUES (?, ?, ?, ?)",
                [(row.benchmark_id, row.item_id, row.group_id, _dumps(row.payload)) for row in rows],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def get_items(self, benchmark_id: str) -> list[ItemRow]:
        """The stored items of a benchmark, in insertion order."""
        rows = self._execute(
            "SELECT benchmark_id, item_id, group_id, payload FROM items WHERE benchmark_id = ? ORDER BY rowid",
            (benchmark_id,),
        ).fetchall()
        return [ItemRow(row[0], row[1], row[2], _loads(row[3])) for row in rows]

    def has_items(self, benchmark_id: str) -> bool:
        """Whether any item of the benchmark is stored."""
        row = self._execute("SELECT 1 FROM items WHERE benchmark_id = ? LIMIT 1", (benchmark_id,)).fetchone()
        return row is not None

    # rollouts

    def put_rollout(self, row: RolloutRow) -> RolloutRow:
        """Record a rollout; the first writer of a key wins.

        Returns:
            The stored row.
        """
        self._execute(
            """INSERT OR IGNORE INTO rollouts (rollout_key, protocol_id, resolved_id, completion, finish_reason, tokens,
                created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row.rollout_key, row.protocol_id, row.resolved_id, row.completion, row.finish_reason, row.tokens,
             row.created_at),
        )
        return self.get_rollout(row.rollout_key)

    def get_rollout(self, rollout_key: str) -> RolloutRow | None:
        """The stored rollout, or None."""
        row = self._execute(
            """SELECT rollout_key, protocol_id, resolved_id, completion, finish_reason, tokens, created_at FROM rollouts
                WHERE rollout_key = ?""",
            (rollout_key,),
        ).fetchone()
        return None if row is None else RolloutRow(*row)

    # scores

    def put_score(self, row: ScoreRow) -> ScoreRow:
        """Record a score; the first writer of a key wins.

        Returns:
            The stored row.
        """
        self._execute(
            "INSERT OR IGNORE INTO scores (score_key, scorer_id, value, detail) VALUES (?, ?, ?, ?)",
            (row.score_key, row.scorer_id, float(row.value), None if row.detail is None else _dumps(row.detail)),
        )
        return self.get_score(row.score_key)

    def get_score(self, score_key: str) -> ScoreRow | None:
        """The stored score, or None."""
        row = self._execute(
            "SELECT score_key, scorer_id, value, detail FROM scores WHERE score_key = ?", (score_key,),
        ).fetchone()
        return None if row is None else ScoreRow(row[0], row[1], row[2], _loads(row[3]))

    # evidence

    def put_evidence(self, row: EvidenceRow) -> EvidenceRow:
        """Record one evidence row; the first writer of a key wins.

        Returns:
            The stored row.
        """
        self._execute(
            """INSERT OR IGNORE INTO evidence (protocol_id, resolved_id, benchmark_id, item_id, replicate, value,
                status, tokens, rollout_key, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.protocol_id, row.resolved_id, row.benchmark_id, row.item_id, int(row.replicate), float(row.value),
             row.status, row.tokens, row.rollout_key, None if row.detail is None else _dumps(row.detail),
             row.created_at),
        )
        return self.get_evidence(row.protocol_id, row.resolved_id, row.benchmark_id, [row.item_id], row.replicate)[
            row.item_id
        ]

    def get_evidence(
        self,
        protocol_id: str,
        resolved_id: str,
        benchmark_id: str,
        item_ids: Sequence[str] | None = None,
        replicate: int | None = None,
    ) -> dict[str, EvidenceRow]:
        """The stored evidence rows of a resolved recipe on a benchmark, keyed by item id.

        Args:
            protocol_id: The protocol's id.
            resolved_id: The resolved recipe's id.
            benchmark_id: The benchmark's id.
            item_ids: The items to read, or None for every item.
            replicate: The replicate to read, or None for every replicate (the last row read per
                item is returned).

        Returns:
            Mapping from item id to its row.
        """
        sql = """SELECT protocol_id, resolved_id, benchmark_id, item_id, replicate, value, status, tokens, rollout_key,
            detail, created_at FROM evidence WHERE protocol_id = ? AND resolved_id = ? AND benchmark_id = ?"""
        parameters: list[Any] = [protocol_id, resolved_id, benchmark_id]
        if replicate is not None:
            sql += " AND replicate = ?"
            parameters.append(int(replicate))
        rows = self._execute(sql, parameters).fetchall()
        wanted = None if item_ids is None else set(item_ids)
        result = {}
        for row in rows:
            if wanted is not None and row[3] not in wanted:
                continue
            result[row[3]] = EvidenceRow(*row[:9], _loads(row[9]), row[10])
        return result

    def evidence_rows(self, protocol_id: str, resolved_id: str, benchmark_id: str) -> list[EvidenceRow]:
        """Every stored evidence row of a resolved recipe on a benchmark, ordered by replicate and item."""
        rows = self._execute(
            """SELECT protocol_id, resolved_id, benchmark_id, item_id, replicate, value, status, tokens, rollout_key,
                detail, created_at FROM evidence WHERE protocol_id = ? AND resolved_id = ? AND benchmark_id = ?
                ORDER BY replicate, item_id""",
            (protocol_id, resolved_id, benchmark_id),
        ).fetchall()
        return [EvidenceRow(*row[:9], _loads(row[9]), row[10]) for row in rows]


__all__ = [
    "EvidenceRow",
    "FitRow",
    "ItemRow",
    "ProtocolRow",
    "RolloutRow",
    "ScoreRow",
    "Store",
    "StoreIntegrityError",
]
