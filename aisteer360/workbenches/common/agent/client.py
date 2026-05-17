"""HTTP client wrapping the agent-facing API surface.

A thin synchronous httpx client; matches the synchronous nature of the per-stage / per-request
agent calls. The agent keeps one long-lived client and reuses its connection pool.

Both run-driven workbenches (vector calibration) and session-driven workbenches (composition) use
the same client. The run-aware methods POST to `/api/agent/runs/{run_id}/*`; the session-aware
methods POST to `/api/agent/sessions/{session_id}/*`. Whether you use one set or the other
depends on which runner is driving the client.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AgentServerError(RuntimeError):
    """Raised when the server returns a non-2xx response for an agent request."""

    def __init__(self, status_code: int, body: str, path: str):
        super().__init__(f"{path} -> {status_code}: {body}")
        self.status_code = status_code
        self.body = body
        self.path = path


class ServerClient:
    """HTTP client for the agent-facing endpoints.

    The client is parameterized by `run_id` (legacy positional arg, used by the VC workbench).
    For session-driven workbenches, `session_id` aliases to the same field — only the endpoint
    paths differ.
    """

    def __init__(
        self,
        base_url: str,
        run_id: str,
        agent_token: str,
        *,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {agent_token}"},
            timeout=timeout,
        )

    @property
    def session_id(self) -> str:
        return self.run_id

    # ── lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ServerClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── run-driven API (vector calibration) ──────────────────────

    def claim(self) -> dict[str, Any]:
        return self._post(f"/api/agent/runs/{self.run_id}/claim").json()

    def get_config(self) -> dict[str, Any]:
        return self._get(f"/api/agent/runs/{self.run_id}/config").json()

    def check_cancel(self) -> bool:
        data = self._get(f"/api/agent/runs/{self.run_id}/cancel-check").json()
        return bool(data.get("cancel_requested", False))

    def post_progress(
        self,
        phase: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._post(
            f"/api/agent/runs/{self.run_id}/progress",
            json={
                "phase": phase,
                "completed": completed,
                "total": total,
                "payload": payload or {},
            },
        )

    def post_model_info(self, info: dict[str, Any]) -> None:
        self._post(f"/api/agent/runs/{self.run_id}/model-info", json=info)

    def stage_start(self, stage: str) -> None:
        self._post(f"/api/agent/runs/{self.run_id}/stage/{stage}/start")

    def stage_complete(self, stage: str, *, notes: str | None = None) -> None:
        self._post(
            f"/api/agent/runs/{self.run_id}/stage/{stage}/complete",
            json={"notes": notes},
        )

    def upload_artifact(self, name: str, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as f:
            self._post(
                f"/api/agent/runs/{self.run_id}/artifacts/{name}",
                files={"file": (path.name, f, "application/octet-stream")},
            )

    def complete(self) -> None:
        self._post(f"/api/agent/runs/{self.run_id}/complete")

    def error(self, message: str) -> None:
        self._post(f"/api/agent/runs/{self.run_id}/error", json={"message": message})

    def post_logs(self, lines: list[str]) -> None:
        self._post(f"/api/agent/runs/{self.run_id}/logs", json={"lines": lines})

    # ── session-driven API (composition workbench) ───────────────

    def session_claim(self) -> dict[str, Any]:
        return self._post(f"/api/agent/sessions/{self.session_id}/claim").json()

    def session_ready(self, model_info: dict[str, Any] | None = None) -> None:
        self._post(
            f"/api/agent/sessions/{self.session_id}/ready",
            json={"model_info": model_info or {}},
        )

    def session_poll(self, timeout_s: float = 30.0) -> dict[str, Any] | None:
        """Long-poll for the next request or close signal.

        Returns the raw payload from the server, which has shape `{"request": {...}}` for an
        inference request, `{"close": true}` for a graceful shutdown signal, or
        `{"request": null, "close": false}` on timeout (Python None).
        """
        resp = self._get(
            f"/api/agent/sessions/{self.session_id}/poll",
            params={"timeout": timeout_s},
            timeout=timeout_s + 5,
        )
        data = resp.json()
        if not data.get("request") and not data.get("close"):
            return None
        return data

    def session_result(self, request_id: str, result: dict[str, Any]) -> None:
        self._post(
            f"/api/agent/sessions/{self.session_id}/result",
            json={"request_id": request_id, **result},
        )

    def session_heartbeat(self) -> None:
        self._post(f"/api/agent/sessions/{self.session_id}/heartbeat")

    def session_error(self, message: str) -> None:
        self._post(
            f"/api/agent/sessions/{self.session_id}/error",
            json={"message": message},
        )

    def session_close(self) -> None:
        self._post(f"/api/agent/sessions/{self.session_id}/close")

    # ── low-level helpers ────────────────────────────────────────

    def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise_for_status(self._client.get(path, **kwargs), path)

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise_for_status(self._client.post(path, **kwargs), path)

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> httpx.Response:
        if resp.status_code >= 400:
            raise AgentServerError(resp.status_code, resp.text, path)
        return resp
