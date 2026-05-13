"""Solo-dev convenience wrapper.

`python -m aisteer360.workbenches.vector_calibration.interface --model <id>` starts the
coordination server, creates one run under an auto-minted owner token, spawns an
`aisteer360-agent` subprocess to drive it, and prints the browser URL (with the owner token
embedded so the SPA picks it up). Ctrl-C cancels the run, terminates the agent, and stops the
server.

The solo path exercises the same server + agent code as the production multi-user flow — there
is no second "simple" mode to maintain.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from aisteer360.workbenches.vector_calibration import (
    CalibrationBuilderConfig,
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
)

from .app import create_app
from .catalog import ALL_ROLES, CatalogEntry, load_catalog, save_catalog
from .db import mint_owner_token, resolve_data_root, sha256_hex

logger = logging.getLogger(__name__)


def _default_config(model_id: str) -> dict:
    cfg = CalibrationBuilderConfig(
        steered_model=model_id,
        generation=GenerationConfig(
            generator_model=model_id,
            behavior="",
            positive_prompt="",
            negative_prompt="",
        ),
        extraction=ExtractionConfig(),
        calibration=CalibrationConfig(
            judge=JudgeConfig(model=model_id, criteria=""),
        ),
    )
    from dataclasses import asdict
    data = asdict(cfg)
    data["generation"]["generator_provider"] = "hf"
    data["calibration"]["judge"]["provider"] = "hf"
    return data


def _ensure_catalog_entry(model_id: str) -> None:
    catalog = load_catalog()
    if any(e.model_id == model_id for e in catalog):
        return
    catalog.append(
        CatalogEntry(
            label=model_id.split("/")[-1],
            model_id=model_id,
            provider="hf",
            roles=list(ALL_ROLES),
        )
    )
    save_catalog(catalog)


def _run_uvicorn(app, host: str, port: int, log_level: str) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("uvicorn failed to start within 10s")


def _wait_for_health(base_url: str, timeout_s: float = 10.0) -> None:
    """Poll until the server accepts connections (even a 401 confirms it's up)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            httpx.get(base_url + "/api/catalog", timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    logger.warning("Health check did not complete cleanly; continuing anyway.")


def _create_solo_run(base_url: str, owner_token: str, model_id: str) -> tuple[str, str, str]:
    """POST /api/runs with a placeholder config. Returns (run_id, agent_token, agent_command)."""
    resp = httpx.post(
        base_url + "/api/runs",
        json={"config": _default_config(model_id)},
        headers={"Authorization": f"Bearer {owner_token}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["run"]["id"], data["agent_token"], data["agent_command"]["command"]


def _spawn_agent(command: str) -> subprocess.Popen:
    logger.info("Spawning agent: %s", command)
    return subprocess.Popen(
        command,
        shell=True,  # the command is constructed with shlex.quote on the server side
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration dashboard (solo-dev wrapper)")
    parser.add_argument("--model", required=True, help="Steered model name or path")
    parser.add_argument("--save-dir", default=None, help="Artefact directory (defaults to ./runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8360)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Start the server only; do not spawn a local agent. Useful for connecting a remote agent.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = resolve_data_root(args.save_dir)
    _ensure_catalog_entry(args.model)

    app = create_app(data_root=data_root)
    server = _run_uvicorn(app, args.host, args.port, args.log_level)
    base_url = f"http://{args.host}:{args.port}"
    _wait_for_health(base_url)

    owner_token = mint_owner_token()
    logger.info("Owner token: %s (hash %s)", owner_token, sha256_hex(owner_token)[:12])

    agent_proc: subprocess.Popen | None = None
    run_id: str | None = None

    if not args.no_agent:
        run_id, agent_token, agent_command = _create_solo_run(
            base_url, owner_token, args.model
        )
        browser_url = f"{base_url}/?owner_token={owner_token}"
        print(f"\n[dashboard] {browser_url}", flush=True)
        print(f"[agent]    {agent_command}\n", flush=True)
        agent_proc = _spawn_agent(agent_command)
    else:
        browser_url = f"{base_url}/?owner_token={owner_token}"
        print(f"\n[dashboard] {browser_url}", flush=True)
        print("[agent]    (skipped; pass --no-agent=false to auto-spawn)\n", flush=True)

    stop_requested = threading.Event()

    def _handle_sigint(_sig, _frame):
        if stop_requested.is_set():
            return
        stop_requested.set()
        logger.info("Interrupt received — cancelling run and shutting down.")
        if run_id is not None:
            try:
                httpx.post(
                    f"{base_url}/api/runs/{run_id}/cancel",
                    headers={"Authorization": f"Bearer {owner_token}"},
                    timeout=5.0,
                )
            except httpx.HTTPError as exc:
                logger.debug("cancel POST failed: %s", exc)
        if agent_proc is not None and agent_proc.poll() is None:
            try:
                agent_proc.terminate()
            except Exception:
                pass
        if run_id is not None and not args.no_agent:
            print(
                "\nTo reattach a new agent to this run after restart:\n"
                f"  {agent_command}\n",
                flush=True,
            )

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        if agent_proc is not None:
            agent_proc.wait()
        else:
            while not stop_requested.is_set():
                time.sleep(0.5)
    finally:
        if agent_proc is not None and agent_proc.poll() is None:
            try:
                agent_proc.kill()
            except Exception:
                pass
        server.should_exit = True

    return agent_proc.returncode if agent_proc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
