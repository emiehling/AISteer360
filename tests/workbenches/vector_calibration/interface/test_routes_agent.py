"""Round-trip tests for /api/agent/runs/{id}/* endpoints."""
from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi.testclient import TestClient

from ..conftest import run_body


def _create_run(
    client: TestClient, owner_header: dict, stages: list[str] | None = None
) -> tuple[str, str]:
    resp = client.post(
        "/api/runs", json=run_body(stages=stages), headers=owner_header,
    ).json()
    return resp["run"]["id"], resp["agent_token"]


def test_claim_requires_agent_token(client: TestClient, owner_header: dict) -> None:
    rid, _ = _create_run(client, owner_header)
    # no auth
    assert client.post(f"/api/agent/runs/{rid}/claim").status_code == 401
    # wrong token
    bad = {"Authorization": "Bearer sk-run-wrong"}
    assert client.post(f"/api/agent/runs/{rid}/claim", headers=bad).status_code == 401


def test_claim_then_progress_then_complete(
    client: TestClient, owner_header: dict, tmp_data_root: Path
) -> None:
    rid, agent_token = _create_run(client, owner_header)
    ah = {"Authorization": f"Bearer {agent_token}"}

    claim = client.post(f"/api/agent/runs/{rid}/claim", headers=ah)
    assert claim.status_code == 200
    body = claim.json()
    assert body["run_id"] == rid
    assert body["stages"] == ["generation", "extraction", "calibration"]

    assert client.post(
        f"/api/agent/runs/{rid}/progress",
        json={"phase": "generation", "completed": 3, "total": 10, "payload": {}},
        headers=ah,
    ).status_code == 200

    # model info
    assert client.post(
        f"/api/agent/runs/{rid}/model-info",
        json={"model_name": "x", "num_layers": 12},
        headers=ah,
    ).status_code == 200

    # artifact upload
    files = {"file": ("pairs.jsonl", io.BytesIO(b'{"a":1}\n'), "application/x-jsonlines")}
    r = client.post(f"/api/agent/runs/{rid}/artifacts/pairs", files=files, headers=ah)
    assert r.status_code == 200
    run_dir = Path(tmp_data_root) / rid
    assert (run_dir / "pairs.jsonl").exists()

    # complete
    assert client.post(f"/api/agent/runs/{rid}/complete", headers=ah).status_code == 200

    # browser sees the new state
    detail = client.get(f"/api/runs/{rid}", headers=owner_header).json()
    assert detail["status"] == "completed"
    assert detail["progress"]["completed"] == 3
    assert detail["model_info"]["num_layers"] == 12


def test_claim_returns_requested_stages_only(client: TestClient, owner_header: dict) -> None:
    rid, agent_token = _create_run(client, owner_header, stages=["generation"])
    ah = {"Authorization": f"Bearer {agent_token}"}
    body = client.post(f"/api/agent/runs/{rid}/claim", headers=ah).json()
    assert body["stages"] == ["generation"]


def test_cancel_check_reflects_owner_cancel(client: TestClient, owner_header: dict) -> None:
    rid, agent_token = _create_run(client, owner_header)
    ah = {"Authorization": f"Bearer {agent_token}"}
    client.post(f"/api/agent/runs/{rid}/claim", headers=ah)

    data = client.get(f"/api/agent/runs/{rid}/cancel-check", headers=ah).json()
    assert data["cancel_requested"] is False

    client.post(f"/api/runs/{rid}/cancel", headers=owner_header)
    data = client.get(f"/api/agent/runs/{rid}/cancel-check", headers=ah).json()
    assert data["cancel_requested"] is True


def test_heartbeat_and_stale_flag(client: TestClient, owner_header: dict) -> None:
    rid, agent_token = _create_run(client, owner_header)
    ah = {"Authorization": f"Bearer {agent_token}"}
    client.post(f"/api/agent/runs/{rid}/claim", headers=ah)
    client.post(
        f"/api/agent/runs/{rid}/progress",
        json={"phase": "generation", "completed": 1, "total": 10},
        headers=ah,
    )
    detail = client.get(f"/api/runs/{rid}", headers=owner_header).json()
    assert detail["status"] == "running"
    assert detail["stale"] is False

    # rewrite last_heartbeat to 600s in the past and re-read
    import sqlite3
    import os
    db_path = [p for p in os.listdir(client.app.state.data_root) if p.endswith(".db")][0]
    with sqlite3.connect(str(Path(client.app.state.data_root) / db_path)) as conn:
        conn.execute("UPDATE runs SET last_heartbeat = ? WHERE id = ?", (time.time() - 600, rid))
    detail = client.get(f"/api/runs/{rid}", headers=owner_header).json()
    assert detail["stale"] is True
