"""SQLite persistence layer for the workbench server.

All run state (config, status, progress, model info, tokens) lives here. Artefacts live on disk
under each run's `run_dir`; the database only stores the path. The server never loads a model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import bcrypt

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  behavior TEXT NOT NULL,
  steered_model TEXT NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL,
  phase TEXT,
  progress_json TEXT,
  model_info_json TEXT,
  owner_token_hash TEXT NOT NULL,
  agent_token_hash TEXT NOT NULL,
  run_dir TEXT NOT NULL,
  error TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  last_heartbeat REAL,
  claimed_at REAL,
  completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_token_hash);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""

STATUS_CREATED = "created"
STATUS_CLAIMED = "claimed"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

ACTIVE_STATUSES = (STATUS_CLAIMED, STATUS_RUNNING)
STALE_THRESHOLD_S = 300.0


@dataclass
class Run:
    """In-memory representation of a run row."""

    id: str
    behavior: str
    steered_model: str
    config_json: str
    status: str
    phase: str | None
    progress_json: str | None
    model_info_json: str | None
    owner_token_hash: str
    agent_token_hash: str
    run_dir: str
    error: str | None
    cancel_requested: bool
    created_at: float
    updated_at: float
    last_heartbeat: float | None
    claimed_at: float | None
    completed_at: float | None

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    @property
    def progress(self) -> dict[str, Any]:
        return json.loads(self.progress_json) if self.progress_json else {}

    @property
    def model_info(self) -> dict[str, Any]:
        return json.loads(self.model_info_json) if self.model_info_json else {}

    def is_stale(self, now: float | None = None, threshold: float = STALE_THRESHOLD_S) -> bool:
        if self.status not in ACTIVE_STATUSES:
            return False
        if self.last_heartbeat is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.last_heartbeat) > threshold

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "behavior": self.behavior,
            "steered_model": self.steered_model,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "model_info": self.model_info,
            "error": self.error,
            "stale": self.is_stale(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_heartbeat": self.last_heartbeat,
            "claimed_at": self.claimed_at,
            "completed_at": self.completed_at,
        }

    def to_detail(self) -> dict[str, Any]:
        data = self.to_summary()
        data["config"] = self.config
        data["run_dir"] = self.run_dir
        return data


# ── hashing helpers ──────────────────────────────────────────────

def sha256_hex(token: str) -> str:
    """SHA-256 hex digest of a token string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_agent_token(token: str) -> str:
    """bcrypt hash an agent token. Returns a utf-8 string suitable for storage."""
    return bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_agent_token(token: str, hashed: str) -> bool:
    """Constant-time bcrypt check for an agent token."""
    try:
        return bcrypt.checkpw(token.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def mint_owner_token() -> str:
    return f"dt-{uuid.uuid4().hex}"


def mint_agent_token() -> str:
    return f"sk-run-{secrets.token_hex(24)}"


# ── database wrapper ─────────────────────────────────────────────

class Database:
    """Thin async wrapper around aiosqlite tailored to the workbench server.

    One long-lived connection. SQLite's writer is single-threaded; at the scale of a dev tool this
    is fine. Callers should use the CRUD methods below rather than executing raw SQL.
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected.")
        return self._conn

    # ── writes ───────────────────────────────────────────────────

    async def create_run(
        self,
        *,
        run_id: str,
        behavior: str,
        steered_model: str,
        config: dict[str, Any],
        owner_token_hash: str,
        agent_token_hash: str,
        run_dir: Path,
    ) -> Run:
        now = time.time()
        await self.conn.execute(
            """
            INSERT INTO runs (id, behavior, steered_model, config_json, status,
                              owner_token_hash, agent_token_hash, run_dir,
                              created_at, updated_at, cancel_requested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run_id, behavior, steered_model, json.dumps(config), STATUS_CREATED,
                owner_token_hash, agent_token_hash, str(run_dir), now, now,
            ),
        )
        await self.conn.commit()
        row = await self.get_run(run_id)
        assert row is not None
        return row

    async def update_config(self, run_id: str, config: dict[str, Any]) -> None:
        await self._update_one(run_id, config_json=json.dumps(config))

    async def update_status(
        self,
        run_id: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        error: str | None = None,
        claimed_at: float | None = None,
        completed_at: float | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if phase is not None:
            fields["phase"] = phase
        if error is not None:
            fields["error"] = error
        if claimed_at is not None:
            fields["claimed_at"] = claimed_at
        if completed_at is not None:
            fields["completed_at"] = completed_at
        if fields:
            await self._update_one(run_id, **fields)

    async def update_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        await self._update_one(run_id, progress_json=json.dumps(progress))

    async def update_model_info(self, run_id: str, model_info: dict[str, Any]) -> None:
        await self._update_one(run_id, model_info_json=json.dumps(model_info))

    async def set_cancel(self, run_id: str, flag: bool = True) -> None:
        await self._update_one(run_id, cancel_requested=1 if flag else 0)

    async def heartbeat(self, run_id: str) -> None:
        await self._update_one(run_id, last_heartbeat=time.time())

    async def claim(self, run_id: str) -> None:
        now = time.time()
        await self._update_one(
            run_id,
            status=STATUS_CLAIMED,
            claimed_at=now,
            last_heartbeat=now,
        )

    async def regenerate_agent_token(self, run_id: str) -> str:
        """Mint a new agent token, rewrite the stored hash, return the plaintext token."""
        token = mint_agent_token()
        await self._update_one(run_id, agent_token_hash=hash_agent_token(token))
        return token

    async def _update_one(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [run_id]
        await self.conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", params)
        await self.conn.commit()

    # ── reads ────────────────────────────────────────────────────

    async def get_run(self, run_id: str) -> Run | None:
        cur = await self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cur.fetchone()
        await cur.close()
        return _row_to_run(row) if row else None

    async def list_runs_for_owner(self, owner_token_hash: str) -> list[Run]:
        cur = await self.conn.execute(
            "SELECT * FROM runs WHERE owner_token_hash = ? ORDER BY created_at DESC",
            (owner_token_hash,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [_row_to_run(r) for r in rows]


def _row_to_run(row: aiosqlite.Row) -> Run:
    data = dict(row)
    data["cancel_requested"] = bool(data.get("cancel_requested", 0))
    return Run(**data)


# ── run-dir helpers ──────────────────────────────────────────────

def build_run_id(behavior: str) -> str:
    """Build a behavior-timestamped run id compatible with the current workbench naming."""
    import datetime
    safe_behavior = (behavior or "run").strip().replace("/", "_").replace(" ", "_") or "run"
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{safe_behavior}_{ts}_{suffix}"


def resolve_data_root(data_root: str | Path | None) -> Path:
    """Resolve the artefact root directory."""
    import os
    if data_root is not None:
        return Path(data_root).resolve()
    env = os.environ.get("AISTEER_WORKBENCH_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return Path("./runs").resolve()


def ensure_run_dir(data_root: Path, run_id: str) -> Path:
    run_dir = data_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


__all__ = [
    "Run",
    "Database",
    "STATUS_CREATED",
    "STATUS_CLAIMED",
    "STATUS_RUNNING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_CANCELLED",
    "ACTIVE_STATUSES",
    "STALE_THRESHOLD_S",
    "sha256_hex",
    "hash_agent_token",
    "verify_agent_token",
    "mint_owner_token",
    "mint_agent_token",
    "build_run_id",
    "resolve_data_root",
    "ensure_run_dir",
]
