"""Generic agent entry point that dispatches to a workbench-specific runner.

`python -m aisteer360.workbenches.common.agent --workbench vector_calibration --run-id ...`
`python -m aisteer360.workbenches.common.agent --workbench composition       --session-id ...`

Workbenches keep their own console scripts (`aisteer360-agent`, `aisteer360-compose-agent`) which
stay as the documented entry points; those scripts are thin wrappers around their per-workbench
`__main__.main()`. This module is for cases that prefer a single dispatch surface (e.g. SSH
remote invocation when both workbenches are installed).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .client import AgentServerError, ServerClient

logger = logging.getLogger(__name__)


_WORKBENCHES = ("vector_calibration", "composition")
_MODES = ("run", "session")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aisteer360-agent (dispatcher)",
        description="Run an AISteer360 workbench agent.",
    )
    p.add_argument("--workbench", required=True, choices=_WORKBENCHES)
    p.add_argument(
        "--mode",
        default=None,
        choices=_MODES,
        help="Override the default mode for the workbench (rarely needed).",
    )
    p.add_argument("--server", required=True, help="Base URL of the server.")
    p.add_argument("--run-id", help="Run id (for --workbench vector_calibration).")
    p.add_argument("--session-id", help="Session id (for --workbench composition).")
    p.add_argument("--agent-token", required=True)
    p.add_argument("--device", default=None, help="Device override (cuda, mps, cpu).")
    p.add_argument("--log-level", default="INFO")
    return p


def _resolve_mode(workbench: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "run" if workbench == "vector_calibration" else "session"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode = _resolve_mode(args.workbench, args.mode)
    work_id = args.run_id if mode == "run" else args.session_id
    if not work_id:
        flag = "--run-id" if mode == "run" else "--session-id"
        logger.error("%s is required for --workbench %s", flag, args.workbench)
        return 2

    if args.device:
        os.environ.setdefault("AISTEER_AGENT_DEVICE", args.device)

    try:
        with ServerClient(args.server, work_id, args.agent_token) as client:
            if args.workbench == "vector_calibration":
                from aisteer360.workbenches.vector_calibration.agent.runner import (
                    AgentRunner,
                )
                AgentRunner(client).run()
            elif args.workbench == "composition":
                from aisteer360.workbenches.composition.agent.runner import SessionRunner
                SessionRunner(client).run()
            else:
                logger.error("Unsupported workbench: %s", args.workbench)
                return 2
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
