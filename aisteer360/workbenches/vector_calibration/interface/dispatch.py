"""Agent dispatch: local subprocess or remote via system SSH.

Local dispatch spawns the agent as a subprocess of the server. Remote dispatch shells out to the
system `ssh` binary so the user inherits `~/.ssh/config`, agent forwarding, hardware keys, and
known_hosts handling for free.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def dispatch_local(argv: list[str]) -> None:
    """Spawn the agent as a local subprocess."""
    logger.info("Dispatching agent locally: %s", " ".join(shlex.quote(a) for a in argv))
    subprocess.Popen(argv)


def _extract_run_id(agent_argv: list[str]) -> str:
    for i, arg in enumerate(agent_argv):
        if arg == "--run-id" and i + 1 < len(agent_argv):
            return agent_argv[i + 1]
    return "unknown"


def dispatch_ssh(config: dict[str, Any], agent_argv: list[str]) -> None:
    """Start the agent on a remote machine via system SSH.

    Uses setsid for clean detachment and a per-run log file so concurrent runs don't stomp on
    each other.
    """
    host = config["host"]
    port = config.get("port", 22)
    username = config["username"]
    python = config.get("python_path") or "python3"

    if not host or not username:
        raise RuntimeError("SSH dispatch requires host and username")

    run_id = _extract_run_id(agent_argv)
    log_path = f"/tmp/aisteer360-agent-{run_id}.log"

    # Drop the leading argv[0] (the CLI script name); the remote runs the module directly.
    remote_args = shlex.join(agent_argv[1:])
    remote_cmd = (
        f"setsid {shlex.quote(python)} -m aisteer360.workbenches.vector_calibration.agent"
        f" {remote_args} > {shlex.quote(log_path)} 2>&1 &"
    )

    ssh_argv = [
        "ssh",
        "-o", "BatchMode=yes",
        "-p", str(port),
        f"{username}@{host}",
        remote_cmd,
    ]

    key_tmpfile = None
    if config.get("auth_method") == "key" and config.get("credential"):
        cred = config["credential"]
        if "\n" in cred or cred.startswith("-----"):
            key_tmpfile = tempfile.NamedTemporaryFile(
                mode="w", suffix=".pem", delete=False,
            )
            key_tmpfile.write(cred)
            key_tmpfile.close()
            os.chmod(key_tmpfile.name, 0o600)
            ssh_argv[1:1] = ["-i", key_tmpfile.name]
        else:
            ssh_argv[1:1] = ["-i", cred]

    try:
        result = subprocess.run(
            ssh_argv, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or f"ssh exited {result.returncode}"
            )
    finally:
        if key_tmpfile:
            try:
                os.unlink(key_tmpfile.name)
            except OSError:
                pass

    logger.info("Agent dispatched via SSH to %s (log: %s)", host, log_path)


def test_ssh(config: dict[str, Any], server_url: str) -> dict[str, Any]:
    """Probe the remote machine for Python, torch, device info, and server reachability.

    The probe script is piped via stdin to avoid shell quoting issues across remote shells.
    """
    host = config.get("host")
    port = config.get("port", 22)
    username = config.get("username")
    python = config.get("python_path") or "python3"

    if not host or not username:
        return {"ok": False, "error": "host and username are required"}

    server_url = server_url.rstrip("/")
    probe_script = (
        "import torch, urllib.request\n"
        "d = 'cuda' if torch.cuda.is_available() else "
        "('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() "
        "else 'cpu')\n"
        "n = torch.cuda.get_device_name(0) if d == 'cuda' else d\n"
        "c = torch.cuda.device_count() if d == 'cuda' else 1\n"
        "try:\n"
        f"    urllib.request.urlopen('{server_url}/api/server-info', timeout=5)\n"
        "    reach = 'ok'\n"
        "except Exception as e:\n"
        "    reach = str(e) or type(e).__name__\n"
        "print(f'{d}|{n}|{c}|{reach}')\n"
    )

    ssh_argv = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        f"{username}@{host}",
        f"{shlex.quote(python)} -",
    ]

    key_tmpfile = None
    if config.get("auth_method") == "key" and config.get("credential"):
        cred = config["credential"]
        if "\n" in cred or cred.startswith("-----"):
            key_tmpfile = tempfile.NamedTemporaryFile(
                mode="w", suffix=".pem", delete=False,
            )
            key_tmpfile.write(cred)
            key_tmpfile.close()
            os.chmod(key_tmpfile.name, 0o600)
            ssh_argv[1:1] = ["-i", key_tmpfile.name]
        else:
            ssh_argv[1:1] = ["-i", cred]

    try:
        result = subprocess.run(
            ssh_argv, input=probe_script,
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SSH connection timed out"}
    except FileNotFoundError:
        return {"ok": False, "error": "ssh binary not found on server host"}
    finally:
        if key_tmpfile:
            try:
                os.unlink(key_tmpfile.name)
            except OSError:
                pass

    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or "probe failed"}

    out = result.stdout.strip().splitlines()
    line = out[-1] if out else ""
    parts = line.split("|") if line else []
    if len(parts) < 4:
        return {
            "ok": False,
            "error": f"unexpected probe output: {result.stdout.strip()!r}",
        }

    reachable = parts[3] == "ok"
    resp: dict[str, Any] = {
        "ok": True,
        "device": parts[0],
        "device_name": parts[1],
        "device_count": int(parts[2]) if parts[2].isdigit() else 1,
        "server_reachable": reachable,
    }
    if not reachable:
        resp["reachability_error"] = parts[3]
    return resp


__all__ = ["dispatch_local", "dispatch_ssh", "test_ssh"]
