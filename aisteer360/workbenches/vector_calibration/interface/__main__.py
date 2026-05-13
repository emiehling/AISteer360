"""Solo-dev convenience wrapper.

`python -m aisteer360.workbenches.vector_calibration.interface` starts the coordination
server, prints the dashboard URL with an owner token, and waits. The user configures a
run in the browser and hits "Run All"; the wrapper spots the new run, mints an agent
token, and spawns a local `aisteer360-agent` subprocess to drive it. Ctrl-C cancels the
run, terminates the agent, and stops the server.

The solo path exercises the same server + agent code as the production multi-user flow —
there is no second "simple" mode to maintain.
"""
from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time

import httpx
import uvicorn

from .app import create_app
from .db import mint_owner_token, resolve_data_root, sha256_hex

logger = logging.getLogger(__name__)


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


def _spawn_agent(command: str) -> subprocess.Popen:
    logger.info("Spawning agent: %s", command)
    return subprocess.Popen(
        command,
        shell=True,  # the command is constructed with shlex.quote on the server side
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _find_unclaimed_run(base_url: str, owner_token: str) -> str | None:
    """Return the id of the oldest unclaimed, non-terminal run, or None."""
    try:
        resp = httpx.get(
            base_url + "/api/runs",
            headers={"Authorization": f"Bearer {owner_token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("runs poll failed: %s", exc)
        return None
    runs = resp.json().get("runs", [])
    # the server returns newest first; walk in reverse so we pick up the earliest-created one
    for run in sorted(runs, key=lambda r: r.get("created_at") or 0.0):
        if run.get("status") == "created" and not run.get("claimed_at"):
            return run["id"]
    return None


def _regenerate_agent_command(base_url: str, owner_token: str, run_id: str) -> str | None:
    try:
        resp = httpx.post(
            f"{base_url}/api/runs/{run_id}/regenerate-token",
            headers={"Authorization": f"Bearer {owner_token}"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("regenerate-token failed for %s: %s", run_id, exc)
        return None
    return resp.json().get("agent_command", {}).get("command")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration dashboard (solo-dev wrapper)")
    parser.add_argument("--save-dir", default=None, help="Artefact directory (defaults to ./runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8360)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Do not auto-spawn a local agent when a run is created. Useful for remote agents.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = resolve_data_root(args.save_dir)
    app = create_app(data_root=data_root)
    server = _run_uvicorn(app, args.host, args.port, args.log_level)
    base_url = f"http://{args.host}:{args.port}"
    _wait_for_health(base_url)

    owner_token = mint_owner_token()
    logger.info("Owner token: %s (hash %s)", owner_token, sha256_hex(owner_token)[:12])

    browser_url = f"{base_url}/?owner_token={owner_token}"
    print(f"\n[dashboard] {browser_url}", flush=True)
    if args.no_agent:
        print("[agent]    auto-spawn disabled (pass without --no-agent to enable)\n", flush=True)
    else:
        print("[agent]    waiting for you to create a run in the browser…\n", flush=True)

    agent_proc: subprocess.Popen | None = None
    attached_run_id: str | None = None
    attached_command: str | None = None
    stop_requested = threading.Event()

    def _handle_sigint(_sig, _frame):
        if stop_requested.is_set():
            return
        stop_requested.set()
        logger.info("Interrupt received — cancelling run and shutting down.")
        if attached_run_id is not None:
            try:
                httpx.post(
                    f"{base_url}/api/runs/{attached_run_id}/cancel",
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
        if attached_command is not None and not args.no_agent:
            print(
                "\nTo reattach a new agent to this run after restart:\n"
                f"  {attached_command}\n",
                flush=True,
            )

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while not stop_requested.is_set():
            if attached_run_id is None and not args.no_agent:
                run_id = _find_unclaimed_run(base_url, owner_token)
                if run_id:
                    command = _regenerate_agent_command(base_url, owner_token, run_id)
                    if command:
                        print(f"\n[agent] spawning: {command}\n", flush=True)
                        agent_proc = _spawn_agent(command)
                        attached_run_id = run_id
                        attached_command = command

            if agent_proc is not None and agent_proc.poll() is not None:
                break

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
