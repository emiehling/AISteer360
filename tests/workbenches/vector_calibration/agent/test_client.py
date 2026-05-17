"""ServerClient tests against a live uvicorn server on localhost."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from aisteer360.workbenches.common.agent.client import (
    AgentServerError,
    ServerClient,
)
from aisteer360.workbenches.vector_calibration.interface.app import create_app

from ..conftest import minimal_config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_data_root: Path):
    import httpx
    port = _free_port()
    app = create_app(data_root=tmp_data_root)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if server.started:
            try:
                # /api/catalog requires auth (returns 401) but reaching the 401 proves routes are mounted
                r = httpx.get(base_url + "/api/catalog", timeout=1.0)
                if r.status_code in (200, 401):
                    break
            except httpx.HTTPError:
                pass
        time.sleep(0.05)
    yield base_url, app
    server.should_exit = True
    thread.join(timeout=5.0)


def _create_run(base_url: str, owner_header: dict) -> tuple[str, str]:
    import httpx
    from ..conftest import run_body
    resp = httpx.post(
        base_url + "/api/runs",
        json=run_body(),
        headers=owner_header,
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["run"]["id"], data["agent_token"]


def test_full_agent_flow(live_server, owner_header: dict, tmp_path: Path) -> None:
    base_url, app = live_server
    rid, agent_token = _create_run(base_url, owner_header)

    with ServerClient(base_url, rid, agent_token) as sc:
        claim = sc.claim()
        assert claim["run_id"] == rid
        cfg = sc.get_config()
        assert cfg["steered_model"] == minimal_config()["steered_model"]

        assert sc.check_cancel() is False
        sc.post_progress("generation", completed=1, total=3)
        sc.post_model_info({"num_layers": 5})

        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text('{"a":1}\n')
        sc.upload_artifact("pairs", pairs)

        sc.stage_start("generation")
        sc.stage_complete("generation")
        sc.complete()

    import httpx
    detail = httpx.get(
        f"{base_url}/api/runs/{rid}", headers=owner_header, timeout=10.0
    ).json()
    assert detail["status"] == "completed"
    assert detail["progress"]["completed"] == 1
    assert detail["model_info"]["num_layers"] == 5


def test_wrong_token_raises_401(live_server, owner_header: dict) -> None:
    base_url, _ = live_server
    rid, _ = _create_run(base_url, owner_header)
    with ServerClient(base_url, rid, "sk-run-wrong") as sc:
        try:
            sc.claim()
        except AgentServerError as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError("expected 401")
