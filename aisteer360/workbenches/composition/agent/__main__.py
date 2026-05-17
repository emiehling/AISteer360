"""Command-line entry point for the composition workbench agent.

`python -m aisteer360.workbenches.composition.agent --server <url> --session-id <id>
                                                    --agent-token <sk-run-...>`
"""
from __future__ import annotations

import argparse
import logging
import sys

from aisteer360.workbenches.common.agent.client import AgentServerError, ServerClient

from .runner import SessionRunner

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aisteer360-compose-agent",
        description="Run a single composition workbench session against a coordination server.",
    )
    p.add_argument("--server", required=True, help="Base URL of the server (e.g. http://host:port)")
    p.add_argument("--session-id", required=True, help="Session id returned by the dashboard")
    p.add_argument("--agent-token", required=True, help="Per-session agent token (sk-run-...)")
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        with ServerClient(args.server, args.session_id, args.agent_token) as client:
            SessionRunner(client).run()
    except AgentServerError as exc:
        logger.error("Server error: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Session agent failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
