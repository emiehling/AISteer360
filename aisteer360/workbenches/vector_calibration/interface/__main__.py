"""Solo-dev convenience wrapper.

`python -m aisteer360.workbenches.vector_calibration.interface` starts the coordination server,
prints the dashboard URL with an owner token, and waits. When the user creates a run from the
browser, the server itself dispatches the agent — locally by default, or over SSH if the user
configured a remote machine in settings.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import uvicorn

from aisteer360.workbenches.common.interface.db import mint_owner_token, resolve_data_root, sha256_hex

from .app import create_app

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration dashboard")
    parser.add_argument("--save-dir", default=None, help="Artefact directory (defaults to ./runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8360)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Don't auto-dispatch locally when no compute config exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = resolve_data_root(args.save_dir)
    solo_mode = not args.no_agent
    public_server_url = f"http://127.0.0.1:{args.port}" if solo_mode else None
    app = create_app(
        data_root=data_root,
        solo_mode=solo_mode,
        public_server_url=public_server_url,
    )
    server = _run_uvicorn(app, args.host, args.port, args.log_level)

    owner_token = mint_owner_token()
    logger.info("Owner token: %s (hash %s)", owner_token, sha256_hex(owner_token)[:12])

    base_url = f"http://{args.host}:{args.port}"
    browser_url = f"{base_url}/?owner_token={owner_token}"
    print(f"\n[dashboard] {browser_url}\n", flush=True)

    stop = threading.Event()

    def _on_signal(_sig, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    stop.wait()
    server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
