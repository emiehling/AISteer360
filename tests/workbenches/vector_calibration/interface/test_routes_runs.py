"""Tests for browser-facing /api/runs/* routes."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ..conftest import ALL_STAGES, minimal_config, run_body


def _auth(owner_header: dict[str, str]) -> dict[str, str]:
    return owner_header


def test_create_run_returns_agent_command(client: TestClient, owner_header: dict) -> None:
    resp = client.post("/api/runs", json=run_body(), headers=owner_header)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["run"]["status"] == "created"
    assert data["run"]["stages"] == ALL_STAGES
    assert data["agent_token"].startswith("sk-run-")
    assert "aisteer360-agent" in data["agent_command"]["command"]
    assert data["agent_command"]["run_id"] == data["run"]["id"]


def test_create_run_rejects_missing_stages(client: TestClient, owner_header: dict) -> None:
    resp = client.post(
        "/api/runs", json={"config": minimal_config()}, headers=owner_header
    )
    assert resp.status_code == 422


def test_create_generation_only_skips_target_and_judge_checks(
    client: TestClient, owner_header: dict
) -> None:
    cfg = minimal_config()
    cfg["steered_model"] = ""
    cfg["calibration"]["judge"]["model"] = ""
    resp = client.post(
        "/api/runs",
        json=run_body(config=cfg, stages=["generation"]),
        headers=owner_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["run"]["stages"] == ["generation"]


def test_list_is_scoped_to_owner(client: TestClient) -> None:
    alice = {"Authorization": "Bearer dt-alice"}
    bob = {"Authorization": "Bearer dt-bob"}
    client.post("/api/runs", json=run_body(behavior="a"), headers=alice)
    client.post("/api/runs", json=run_body(behavior="b"), headers=bob)

    alice_runs = client.get("/api/runs", headers=alice).json()["runs"]
    bob_runs = client.get("/api/runs", headers=bob).json()["runs"]
    assert len(alice_runs) == 1 and alice_runs[0]["behavior"] == "a"
    assert len(bob_runs) == 1 and bob_runs[0]["behavior"] == "b"


def test_get_run_404_for_other_owner(client: TestClient) -> None:
    alice = {"Authorization": "Bearer dt-alice"}
    mallory = {"Authorization": "Bearer dt-mallory"}
    run_id = client.post("/api/runs", json=run_body(), headers=alice).json()["run"]["id"]
    assert client.get(f"/api/runs/{run_id}", headers=mallory).status_code == 404
    assert client.get(f"/api/runs/{run_id}", headers=alice).status_code == 200


def test_missing_token_is_401(client: TestClient) -> None:
    assert client.get("/api/runs").status_code == 401


def test_cancel_created_run_sets_cancelled(client: TestClient, owner_header: dict) -> None:
    rid = client.post("/api/runs", json=run_body(), headers=owner_header).json()["run"]["id"]
    r = client.post(f"/api/runs/{rid}/cancel", headers=owner_header)
    assert r.status_code == 200
    detail = client.get(f"/api/runs/{rid}", headers=owner_header).json()
    assert detail["status"] == "cancelled"


def test_regenerate_token_rejects_non_created(client: TestClient, owner_header: dict) -> None:
    rid = client.post("/api/runs", json=run_body(), headers=owner_header).json()["run"]["id"]
    r1 = client.post(f"/api/runs/{rid}/regenerate-token", headers=owner_header)
    assert r1.status_code == 200
    client.post(f"/api/runs/{rid}/cancel", headers=owner_header)
    r2 = client.post(f"/api/runs/{rid}/regenerate-token", headers=owner_header)
    assert r2.status_code == 409


# ── continue endpoint ────────────────────────────────────────────

def _force_terminal(client: TestClient, owner_header: dict, run_id: str) -> None:
    """Cancel the run so it lands in a continue-eligible terminal state."""
    client.post(f"/api/runs/{run_id}/cancel", headers=owner_header)


def test_continue_rejects_active_run(client: TestClient, owner_header: dict) -> None:
    rid = client.post(
        "/api/runs", json=run_body(stages=["generation"]), headers=owner_header
    ).json()["run"]["id"]
    resp = client.post(
        f"/api/runs/{rid}/continue",
        json=run_body(stages=["extraction"]),
        headers=owner_header,
    )
    assert resp.status_code == 409


def test_continue_requires_pairs_for_extraction(
    client: TestClient, owner_header: dict
) -> None:
    rid = client.post(
        "/api/runs", json=run_body(stages=["generation"]), headers=owner_header
    ).json()["run"]["id"]
    _force_terminal(client, owner_header, rid)

    resp = client.post(
        f"/api/runs/{rid}/continue",
        json=run_body(stages=["extraction"]),
        headers=owner_header,
    )
    assert resp.status_code == 400
    assert "pairs.jsonl" in resp.text


def test_continue_round_trip(client: TestClient, owner_header: dict, tmp_data_root: Path) -> None:
    rid = client.post(
        "/api/runs", json=run_body(stages=["generation"]), headers=owner_header
    ).json()["run"]["id"]
    _force_terminal(client, owner_header, rid)

    run_dir = tmp_data_root / rid
    (run_dir / "pairs.jsonl").write_text(
        json.dumps({"prompt": "p", "positive": "P", "negative": "N", "behavior": "warmth"}) + "\n"
    )

    resp = client.post(
        f"/api/runs/{rid}/continue",
        json=run_body(stages=["extraction"]),
        headers=owner_header,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["run"]["id"] == rid
    assert data["run"]["status"] == "created"
    assert data["run"]["stages"] == ["extraction"]
    assert data["run"]["error"] is None
    assert data["agent_token"].startswith("sk-run-")


def test_continue_requires_svec_for_calibration(
    client: TestClient, owner_header: dict, tmp_data_root: Path
) -> None:
    rid = client.post(
        "/api/runs", json=run_body(stages=["generation"]), headers=owner_header
    ).json()["run"]["id"]
    _force_terminal(client, owner_header, rid)

    resp = client.post(
        f"/api/runs/{rid}/continue",
        json=run_body(stages=["calibration"]),
        headers=owner_header,
    )
    assert resp.status_code == 400
    assert "warmth.svec" in resp.text or ".svec" in resp.text
