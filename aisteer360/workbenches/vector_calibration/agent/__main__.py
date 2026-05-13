"""Command-line entry point for `aisteer360-agent`.

API keys are not accepted on the CLI or read from environment variables; they are delivered to
the agent by the server at claim time, having been set by the owner via the dashboard.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .client import AgentServerError, ServerClient
from .runner import AgentRunner

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aisteer360-agent",
        description="Run a single vector-calibration run against a coordination server.",
    )
    p.add_argument("--server", required=True, help="Base URL of the server (e.g. http://host:port)")
    p.add_argument("--run-id", required=True, help="Run id returned by the dashboard")
    p.add_argument("--agent-token", required=True, help="Per-run agent token (sk-run-...)")
    p.add_argument("--device", default=None, help="Device override (cuda, mps, cpu)")
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.device:
        os.environ.setdefault("AISTEER_AGENT_DEVICE", args.device)

    try:
        with ServerClient(args.server, args.run_id, args.agent_token) as client:
            runner = AgentRunner(client)
            runner.run()
    except AgentServerError as exc:
        logger.error("Server error: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Agent failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
