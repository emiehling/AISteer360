"""Agent runner orchestration test with a stubbed workbench + live uvicorn."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from aisteer360.workbenches.vector_calibration.agent.client import ServerClient
from aisteer360.workbenches.vector_calibration.agent.providers.base import ProviderKeys
from aisteer360.workbenches.vector_calibration.agent.runner import AgentRunner
from aisteer360.workbenches.vector_calibration.interface.app import create_app

from ..conftest import minimal_config


class _FakeWorkbench:
    def __init__(self, config):
        self.config = config
        self._run_dir: Path | None = None
        self._model = None
        self._tokenizer = None

    def run_generation(self, on_progress=None, run_dir=None, cancel_check=None, generation_provider=None):
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "pairs.jsonl").write_text(
            '{"prompt":"p","positive":"P","negative":"N","behavior":"warmth"}\n'
        )
        if on_progress:
            on_progress(1, 1)
        from aisteer360.algorithms.state_control.common.specs import ContrastivePairs
        from aisteer360.workbenches.vector_calibration.results import GenerationResult
        return GenerationResult(
            pairs=ContrastivePairs(positives=["P"], negatives=["N"], prompts=["p"]),
            seed_prompts_used=["p"],
            config={},
        )

    def run_extraction(self, pairs=None, run_dir=None):
        (Path(run_dir) / f"{self.config.generation.behavior}.svec").write_text("{}")
        return object()

    def run_calibration(self, steering_vector=None, on_progress=None, run_dir=None, judge_provider=None):
        (Path(run_dir) / "calibration_result.json").write_text("{}")
        if on_progress:
            on_progress({"completed": 1, "total": 1})
        return object()

    def cleanup(self):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_data_root: Path):
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
                r = httpx.get(base_url + "/api/catalog", timeout=1.0)
                if r.status_code in (200, 401):
                    break
            except httpx.HTTPError:
                pass
        time.sleep(0.05)
    yield base_url
    server.should_exit = True
    thread.join(timeout=5.0)


def test_runner_round_trip(monkeypatch, live_server: str, owner_header: dict) -> None:
    resp = httpx.post(
        live_server + "/api/runs",
        json={"config": minimal_config()},
        headers=owner_header,
        timeout=10.0,
    ).json()
    rid, agent_token = resp["run"]["id"], resp["agent_token"]

    import aisteer360.workbenches.vector_calibration.agent.runner as runner_mod
    monkeypatch.setattr(runner_mod, "VectorCalibrationWorkbench", _FakeWorkbench)
    monkeypatch.setattr(
        runner_mod, "build_from_config", lambda cfg, keys: (None, None)
    )

    with ServerClient(live_server, rid, agent_token) as sc:
        runner = AgentRunner(sc, ProviderKeys())
        runner.run()

    detail = httpx.get(
        f"{live_server}/api/runs/{rid}", headers=owner_header, timeout=10.0
    ).json()
    assert detail["status"] == "completed"
