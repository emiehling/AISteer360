"""A `StudySession` served over a local socket, for search engines that evaluate in worker processes.

Since a loaded model cannot be shared with an engine's worker processes, the process that owns the GPU serves the
session. The transport is `multiprocessing.connection` with a random 32-byte authkey, and every message is one UTF-8
JSON object (never a pickled object). The operations are:

- `{"op": "stage1", "program": <text>}`: `session.screen`, with the partial evaluation kept by candidate id.
- `{"op": "stage2", "program": <text>}`: `session.finish` on the kept partial (`full` for an unknown candidate).
- `{"op": "full", "program": <text>}`: `session.evaluate`.
- `{"op": "estimate", "program": <text>}`: the archive's latest result of the candidate, as
  `{"known", "accepted", "fitness", "candidate_id", "structure"}` (the last two are None for a program that does not
  parse or bind).
- `{"op": "ping"}`: an empty reply.

Evaluation operations reply `{"metrics": {...}, "artifacts": {...}}`. A final result (accepted or rejected) has the
metrics `combined_score` (the fitness), `gain` (the operating point's gain, or 0.0), `accepted` (1.0 or 0.0), and
`level` (the result's level clipped to [0, 1], or 0.0 when it was not measured). A candidate that passes `stage1`
has exactly the metrics `{"stage1_passed": 1.0}`, since OpenEvolve decides whether to run the second stage from them,
and one rejected there adds `stage1_passed` of 0.0 to the final metrics. The artifacts are the status, stage, and
reason, and the entries of the result's summary, as text. Every final reply (and the reply to a program that does
not parse or bind) also lists what failed before, since OpenEvolve stores the artifacts of admitted programs only:

- `recent_rejections`: the latest recorded rejections of the candidate's structure, newest first, one line each
  with the candidate's values, the reason, and the level and gain when the summary has them.
- `recent_unevaluated`: the messages of the latest programs that could not be evaluated, newest first, i.e.,
  programs that did not parse or bind.

Both contain candidate values, reason names, and search-pool aggregates only.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client, Listener
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import RecipeError
from steerability.optimization.evaluation.result import compact

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveEntry
    from steerability.optimization.evaluation.evaluator import Partial
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.study import Candidate

logger = logging.getLogger(__name__)

HANDLER_THREADS = 4
UNEVALUATED_MESSAGES = 3
VALUES_CHARS = 300
MESSAGE_CHARS = 200


def call(address: str, authkey: bytes, request: Mapping[str, Any]) -> dict:
    """Send one request to an evaluation service and return its reply.

    Args:
        address: The service's address.
        authkey: The service's authkey.
        request: The JSON request.

    Returns:
        The JSON reply.
    """
    with Client(address, authkey=authkey) as connection:
        connection.send_bytes(json.dumps(dict(request)).encode("utf-8"))
        return json.loads(connection.recv_bytes().decode("utf-8"))


def _text(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def final_metrics(result: Result) -> dict[str, float]:
    """The metrics of a final result: `combined_score`, `gain`, `accepted`, and `level`.

    `level` is the result's level clipped to [0, 1], or 0.0 when it was not measured.
    """
    gain = result.operating.gain if result.operating is not None else 0.0
    level = 0.0 if result.level is None else min(max(float(result.level), 0.0), 1.0)
    return {
        "combined_score": float(result.fitness),
        "gain": float(gain),
        "accepted": 1.0 if result.accepted else 0.0,
        "level": level,
    }


def result_artifacts(result: Result, error: str | None = None) -> dict[str, str]:
    """The artifacts of a result: status, stage, reason, the summary entries, and an error message when given."""
    artifacts = {"status": result.status, "stage": str(result.stage), "reason": result.reason or ""}
    artifacts.update({key: _text(value) for key, value in result.summary.items()})
    if error:
        artifacts["error"] = error
    return artifacts


def _cut(*, text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _rejection_line(*, entry: ArchiveEntry) -> str:
    """One line on a rejected archive entry: its values (at most 300 characters), reason, level, and gain."""
    line = f"{_cut(text=compact(value=dict(entry.values)), limit=VALUES_CHARS)} -> {entry.reason}"
    for key in ("level", "gain"):
        value = entry.summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            line += f", {key} {value:.3g}"
    return line


class EvaluationService:
    """A study session served over a local socket.

    Evaluations run one at a time (a lock is held around each), and each operation passes the session the deadline
    `time.monotonic() + stage_deadline_seconds`. Estimates and pings do not wait for a running evaluation. A program
    that does not parse, or a candidate that does not bind, gets a `combined_score` of 0.0 with the reason `invalid`
    and the error message as an artifact, and nothing is recorded. The service keeps the messages of the last three
    programs that could not be evaluated (each cut at 200 characters), since they are not in the archive.

    Args:
        session: The started session.
        proposer: The proposer name recorded with evaluated candidates.
        parse: The function that turns a program into a `Candidate`, raising `ValueError` for a malformed program.
        stage_deadline_seconds: The time each operation may take.
        digest: The largest number of recent rejections a reply lists (`recent_rejections`); 0 lists none.

    Attributes:
        programs: The number of `stage1` and `full` operations received. A `stage2` operation is not counted,
            since its program was counted at `stage1`.
        unparsed: The counted programs that did not parse.
        unbound: The counted programs whose candidate did not bind.
    """

    def __init__(self, session: StudySession, *, proposer: str, parse: Callable[[str], Candidate],
                 stage_deadline_seconds: float, digest: int = 5) -> None:
        if isinstance(digest, bool) or not isinstance(digest, int) or digest < 0:
            raise ValueError(f"digest must be a non-negative integer; got {digest!r}.")
        self.session = session
        self.proposer = proposer
        self.parse = parse
        self.stage_deadline_seconds = float(stage_deadline_seconds)
        self.digest = digest
        self._lock = threading.Lock()
        self.programs = 0
        self.unparsed = 0
        self.unbound = 0
        self._partials: dict[str, Partial] = {}
        self._unevaluated: deque[str] = deque(maxlen=UNEVALUATED_MESSAGES)
        self._listener: Listener | None = None
        self._authkey: bytes | None = None
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._stopping = False

    @property
    def address(self) -> str | None:
        """The address the service listens on, or None before `start`."""
        return None if self._listener is None else self._listener.address

    def start(self) -> tuple[str, bytes]:
        """Bind a listener at an address the platform chooses and serve requests in a thread.

        Returns:
            The address and the authkey.

        Raises:
            RuntimeError: If the service is already started.
        """
        if self._listener is not None:
            raise RuntimeError("The evaluation service is already started.")
        self._authkey = secrets.token_bytes(32)
        self._listener = Listener(authkey=self._authkey)
        self._stopping = False
        self._executor = ThreadPoolExecutor(max_workers=HANDLER_THREADS, thread_name_prefix="steerability-service")
        self._thread = threading.Thread(target=self._serve, name="steerability-service", daemon=True)
        self._thread.start()
        return self._listener.address, self._authkey

    def _serve(self) -> None:
        while not self._stopping:
            try:
                connection = self._listener.accept()
            except Exception:
                if self._stopping:
                    break
                logger.warning("The evaluation service refused a connection.", exc_info=True)
                continue
            if self._stopping:
                connection.close()
                break
            self._executor.submit(self._answer, connection)

    def _answer(self, connection) -> None:
        with connection:
            try:
                request = json.loads(connection.recv_bytes().decode("utf-8"))
                reply = self.handle(request)
            except Exception as error:
                logger.warning("The evaluation service failed on a request.", exc_info=True)
                reply = {"error": f"{type(error).__name__}: {error}"}
            try:
                connection.send_bytes(json.dumps(reply).encode("utf-8"))
            except (OSError, EOFError):
                logger.info("The client closed its connection before the reply.")

    def stop(self) -> None:
        """Stop serving, wait for the requests in progress, and close the listener."""
        if self._listener is None:
            return
        self._stopping = True
        try:
            Client(self._listener.address, authkey=self._authkey).close()
        except Exception:
            logger.debug("Waking the listener failed.", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=30)
        self._listener.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._listener = None
        self._thread = None
        self._executor = None

    def handle(self, request: Mapping[str, Any]) -> dict:
        """Answer one request (the operations in the module docstring).

        Args:
            request: The JSON request.

        Returns:
            The JSON reply.
        """
        operation = request.get("op")
        if operation == "ping":
            return {}
        if operation == "estimate":
            return self._estimate(str(request.get("program", "")))
        if operation in ("stage1", "stage2", "full"):
            with self._lock:
                return self._evaluate(operation, str(request.get("program", "")))
        return {"error": f"Unknown operation {operation!r}."}

    def _estimate(self, program: str) -> dict:
        unknown = {"known": False, "accepted": False, "fitness": 0.0, "candidate_id": None, "structure": None}
        try:
            candidate = self.parse(program)
            candidate_id = self.session.study.candidate_id(candidate=candidate)
        except (ValueError, RecipeError):
            return unknown
        identity = {"candidate_id": candidate_id, "structure": candidate.structure}
        latest = self.session.archive.latest(candidate_id=candidate_id)
        if latest is None:
            return {**unknown, **identity}
        return {"known": True, "accepted": latest.accepted, "fitness": float(latest.fitness), **identity}

    def _history(self, *, structure: str | None) -> dict[str, str]:
        """The `recent_rejections` of a structure (left out when it is unknown) and the `recent_unevaluated`."""
        artifacts: dict[str, str] = {}
        if structure is not None and self.digest:
            lines = []
            for entry in reversed(self.session.archive.view().entries):
                if entry.status == "rejected" and entry.structure == structure:
                    lines.append(_rejection_line(entry=entry))
                    if len(lines) >= self.digest:
                        break
            if lines:
                artifacts["recent_rejections"] = "\n".join(lines)
        if self._unevaluated:
            artifacts["recent_unevaluated"] = "\n".join(reversed(self._unevaluated))
        return artifacts

    def _invalid(self, *, message: str, stage1: bool, structure: str | None) -> dict:
        self._unevaluated.append(_cut(text=" ".join(message.split()), limit=MESSAGE_CHARS))
        metrics = {"combined_score": 0.0, "gain": 0.0, "accepted": 0.0, "level": 0.0}
        if stage1:
            metrics = {"stage1_passed": 0.0, **metrics}
        artifacts = {"status": "rejected", "stage": "0", "reason": "invalid", "error": message}
        return {"metrics": metrics, "artifacts": {**artifacts, **self._history(structure=structure)}}

    def _final(self, *, result: Result, partial: Partial, stage1: bool = False) -> dict:
        candidate = partial.candidate
        metrics = final_metrics(result=result)
        if stage1:
            metrics = {"stage1_passed": 0.0, **metrics}
        artifacts = result_artifacts(result=result, error=partial.error)
        return {"metrics": metrics, "artifacts": {**artifacts, **self._history(structure=candidate.structure)}}

    def _evaluate(self, operation: str, program: str) -> dict:
        deadline = time.monotonic() + self.stage_deadline_seconds
        counted = int(operation != "stage2")
        self.programs += counted
        try:
            candidate = self.parse(program)
        except (ValueError, RecipeError) as error:
            self.unparsed += counted
            return self._invalid(message=str(error), stage1=operation == "stage1", structure=None)
        session = self.session
        if operation == "stage2":
            try:
                candidate_id = session.study.candidate_id(candidate=candidate)
            except RecipeError as error:
                return self._invalid(message=str(error), stage1=False, structure=candidate.structure)
            partial = self._partials.pop(candidate_id, None)
            if partial is not None:
                return self._final(result=session.finish(partial=partial, deadline=deadline), partial=partial)
        partial = session.screen(candidate=candidate, proposer=self.proposer, deadline=deadline)
        if partial.candidate_id is None:
            self.unbound += counted
            return self._invalid(
                message=partial.error or "The candidate does not bind.",
                stage1=operation == "stage1",
                structure=candidate.structure,
            )
        if operation == "stage1":
            result = partial.result
            if result is None or result.accepted:
                self._partials[partial.candidate_id] = partial
                return {"metrics": {"stage1_passed": 1.0}, "artifacts": {"status": "screened", "stage": "3"}}
            return self._final(result=result, partial=partial, stage1=True)
        return self._final(result=session.finish(partial=partial, deadline=deadline), partial=partial)


__all__ = ["EvaluationService", "call", "final_metrics", "result_artifacts"]
