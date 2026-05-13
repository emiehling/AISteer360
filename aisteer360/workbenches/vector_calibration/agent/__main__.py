"""Command-line entry point for `aisteer360-agent`."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .client import AgentServerError, ServerClient
from .providers.base import ProviderKeys
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
    p.add_argument("--hf-token", default=None, help="HuggingFace token; overrides $HF_TOKEN")
    p.add_argument(
        "--anthropic-key",
        default=None,
        help="Anthropic API key; overrides $ANTHROPIC_API_KEY",
    )
    p.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key; overrides $OPENAI_API_KEY",
    )
    p.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL; overrides $OPENAI_BASE_URL",
    )
    p.add_argument("--device", default=None, help="Device override (cuda, mps, cpu)")
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    keys = ProviderKeys(
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        anthropic_key=args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY"),
        openai_key=args.openai_key or os.environ.get("OPENAI_API_KEY"),
        openai_base_url=args.openai_base_url or os.environ.get("OPENAI_BASE_URL"),
    )
    if args.device:
        os.environ.setdefault("AISTEER_AGENT_DEVICE", args.device)

    try:
        with ServerClient(args.server, args.run_id, args.agent_token) as client:
            runner = AgentRunner(client, keys)
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
