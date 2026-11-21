"""A `StudySession` served over a local socket, for search engines that evaluate in worker processes.

Since a loaded model cannot be shared with an engine's worker processes, the process that owns the GPU serves the
session. The transport is `multiprocessing.connection` with a random 32-byte authkey, and every message is one UTF-8
JSON object (never a pickled object). The operations are:

- `{"op": "stage1", "program": <text>}`: `session.screen`, with the partial evaluation kept by candidate id.
- `{"op": "stage2", "program": <text>}`: `session.finish` on the kept partial (`full` for an unknown candidate).
- `{"op": "full", "program": <text>}`: `session.evaluate`.
- `{"op": "estimate", "program": <text>}`: the archive's latest result of the candidate, as
  `{"known", "accepted", "fitness"}`.
- `{"op": "ping"}`: an empty reply.

Evaluation operations reply `{"metrics": {...}, "artifacts": {...}}`. A final result (accepted or rejected) has the
metrics `combined_score` (the fitness), `gain` (the operating point's gain, or 0.0), and `accepted` (1.0 or 0.0). A
candidate that passes `stage1` has the metrics `{"stage1_passed": 1.0}` and no `combined_score`, and one rejected
there adds `stage1_passed` of 0.0 to the final metrics. The artifacts are the status, stage, and reason, and the
entries of the result's summary, as text.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client, Listener
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import RecipeError

if TYPE_CHECKING:
    from steerability.optimization.evaluation.evaluator import Partial
    from steerability.optimization.evaluation.result import Result
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.study import Candidate

logger = logging.getLogger(__name__)

HANDLER_THREADS = 4


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
    """The metrics of a final result: `combined_score`, `gain`, and `accepted`."""
    gain = result.operating.gain if result.operating is not None else 0.0
    return {"combined_score": float(result.fitness), "gain": float(gain), "accepted": 1.0 if result.accepted else 0.0}


def result_artifacts(result: Result, error: str | None = None) -> dict[str, str]:
    """The artifacts of a result: status, stage, reason, the summary entries, and an error message when given."""
    artifacts = {"status": result.status, "stage": str(result.stage), "reason": result.reason or ""}
    artifacts.update({key: _text(value) for key, value in result.summary.items()})
    if error:
        artifacts["error"] = error
    return artifacts


class EvaluationService:
    """A study session served over a local socket.

    Evaluations run one at a time (a lock is held around each), and each operation passes the session the deadline
    `time.monotonic() + stage_deadline_seconds`. Estimates and pings do not wait for a running evaluation. A program
    that does not parse, or a candidate that does not bind, gets a `combined_score` of 0.0 with the reason `invalid`
    and the error message as an artifact, and nothing is recorded.

    Args:
        session: The started session.
        proposer: The proposer name recorded with evaluated candidates.
        parse: The function that turns a program into a `Candidate`, raising `ValueError` for a malformed program.
        stage_deadline_seconds: The time each operation may take.

    Attributes:
        programs: The number of `stage1` and `full` operations received. A `stage2` operation is not counted,
            since its program was counted at `stage1`.
        unparsed: The counted programs that did not parse.
        unbound: The counted programs whose candidate did not bind.
    """

    def __init__(self, session: StudySession, *, proposer: str, parse: Callable[[str], Candidate],
                 stage_deadline_seconds: float) -> None:
        self.session = session
        self.proposer = proposer
        self.parse = parse
        self.stage_deadline_seconds = float(stage_deadline_seconds)
        self._lock = threading.Lock()
        self.programs = 0
        self.unparsed = 0
        self.unbound = 0
        self._partials: dict[str, Partial] = {}
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
        try:
            candidate = self.parse(program)
            candidate_id = self.session.study.candidate_id(candidate)
        except (ValueError, RecipeError):
            return {"known": False, "accepted": False, "fitness": 0.0}
        latest = self.session.archive.latest(candidate_id)
        if latest is None:
            return {"known": False, "accepted": False, "fitness": 0.0}
        return {"known": True, "accepted": latest.accepted, "fitness": float(latest.fitness)}

    @staticmethod
    def _invalid(message: str, *, stage1: bool) -> dict:
        metrics = {"combined_score": 0.0, "gain": 0.0, "accepted": 0.0}
        if stage1:
            metrics = {"stage1_passed": 0.0, **metrics}
        return {"metrics": metrics, "artifacts": {"status": "rejected", "stage": "0", "reason": "invalid",
                                                  "error": message}}

    def _evaluate(self, operation: str, program: str) -> dict:
        deadline = time.monotonic() + self.stage_deadline_seconds
        counted = int(operation != "stage2")
        self.programs += counted
        try:
            candidate = self.parse(program)
        except (ValueError, RecipeError) as error:
            self.unparsed += counted
            return self._invalid(str(error), stage1=operation == "stage1")
        session = self.session
        if operation == "stage2":
            try:
                candidate_id = session.study.candidate_id(candidate)
            except RecipeError as error:
                return self._invalid(str(error), stage1=False)
            partial = self._partials.pop(candidate_id, None)
            if partial is not None:
                result = session.finish(partial, deadline=deadline)
                return {"metrics": final_metrics(result), "artifacts": result_artifacts(result, partial.error)}
        partial = session.screen(candidate, proposer=self.proposer, deadline=deadline)
        if partial.candidate_id is None:
            self.unbound += counted
            return self._invalid(partial.error or "The candidate does not bind.", stage1=operation == "stage1")
        if operation == "stage1":
            result = partial.result
            if result is None or result.accepted:
                self._partials[partial.candidate_id] = partial
                return {"metrics": {"stage1_passed": 1.0}, "artifacts": {"status": "screened", "stage": "3"}}
            return {"metrics": {"stage1_passed": 0.0, **final_metrics(result)},
                    "artifacts": result_artifacts(result, partial.error)}
        result = session.finish(partial, deadline=deadline)
        return {"metrics": final_metrics(result), "artifacts": result_artifacts(result, partial.error)}


__all__ = ["EvaluationService", "call", "final_metrics", "result_artifacts"]
