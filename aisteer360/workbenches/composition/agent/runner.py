"""SessionRunner: long-lived agent that holds a model and serves inference requests.

Lifecycle:

  1. Claim the session (fetches model name + provider keys + idle timeout).
  2. Enter the request loop: long-poll, build/cache the steered pipeline, generate, post result.
  3. Exit when:
     - the browser sends a close signal,
     - the idle timeout expires,
     - a fatal error occurs.

Caching:

  - The model is loaded lazily on the first request (so claim → ready handshake stays fast).
  - The agent caches the most recent `PipelineDefinition`. On a new request, if the definition's
    hash is unchanged, `steer()` is skipped.
  - If the request specifies a different `model_name_or_path`, the agent throws away the cached
    pipeline and rebuilds. This should be rare in normal use.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from aisteer360.workbenches.common.agent.client import AgentServerError, ServerClient
from aisteer360.workbenches.composition.interface.schemas import PipelineDefinition
from aisteer360.workbenches.composition.workbench import (
    CompositionWorkbench,
    hash_pipeline,
)

logger = logging.getLogger(__name__)


class SessionRunner:
    """Per-session agent loop."""

    def __init__(self, client: ServerClient, *, poll_timeout_s: float = 30.0):
        self.client = client
        self.poll_timeout_s = poll_timeout_s
        self.workbench = CompositionWorkbench()

        self._pipeline: Any | None = None
        self._pipeline_hash: str | None = None
        self._cached_definition: PipelineDefinition | None = None
        self._idle_timeout_s: float = 600.0
        self._provider_keys: dict[str, str | None] = {}
        self._last_request_at: float = time.monotonic()

    # ── public entry point ───────────────────────────────────────

    def run(self) -> None:
        try:
            claim = self.client.session_claim()
        except AgentServerError as exc:
            logger.error("Failed to claim session %s: %s", self.client.session_id, exc)
            raise

        self._idle_timeout_s = float(claim.get("idle_timeout_s") or 600.0)
        self._provider_keys = dict(claim.get("provider_keys") or {})
        logger.info(
            "Claimed session %s; model=%s idle_timeout=%.0fs",
            self.client.session_id,
            claim.get("model_name_or_path"),
            self._idle_timeout_s,
        )

        # ready handshake — model loads lazily on first request
        self.client.session_ready({"deferred_load": True})
        self._last_request_at = time.monotonic()

        try:
            self._loop()
        except Exception as exc:
            logger.exception("Session %s failed.", self.client.session_id)
            try:
                self.client.session_error(str(exc))
            except Exception:
                logger.warning("Failed to report session error to server.")
            raise
        finally:
            self._release_pipeline()

    # ── request loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            payload = self.client.session_poll(timeout_s=self.poll_timeout_s)
            now = time.monotonic()
            if payload is None:
                if (now - self._last_request_at) > self._idle_timeout_s:
                    logger.info(
                        "Session %s idle for %.0fs; closing.",
                        self.client.session_id,
                        now - self._last_request_at,
                    )
                    self._graceful_close()
                    return
                continue

            if payload.get("close"):
                logger.info("Session %s received close signal.", self.client.session_id)
                self._graceful_close()
                return

            request = payload.get("request") or {}
            request_id = request.get("request_id") or "unknown"
            self._last_request_at = now
            try:
                self._handle_request(request)
            except Exception as exc:
                logger.exception("Inference request %s failed.", request_id)
                try:
                    self.client.session_result(
                        request_id,
                        {
                            "generated_text": "",
                            "elapsed_ms": 0.0,
                            "pipeline_hash": self._pipeline_hash or "",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    logger.warning("Failed to report inference error.")

    # ── single-request handling ──────────────────────────────────

    def _handle_request(self, request: dict[str, Any]) -> None:
        request_id = request["request_id"]
        prompt = request.get("prompt") or ""
        gen_kwargs = dict(request.get("gen_kwargs") or {})
        definition = PipelineDefinition.model_validate(request["pipeline"])

        self._ensure_pipeline(definition)

        t0 = time.perf_counter()
        text = self._pipeline.generate_text(prompts=prompt, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(text, list):
            text = text[0] if text else ""

        self.client.session_result(
            request_id,
            {
                "generated_text": str(text),
                "elapsed_ms": elapsed_ms,
                "pipeline_hash": self._pipeline_hash or "",
            },
        )

    # ── pipeline caching ─────────────────────────────────────────

    def _ensure_pipeline(self, definition: PipelineDefinition) -> None:
        if (
            self._pipeline is not None
            and not self.workbench.model_changed(self._cached_definition, definition)
            and not self.workbench.definition_changed(self._cached_definition, definition)
        ):
            return

        if self._pipeline is not None and self.workbench.model_changed(
            self._cached_definition, definition
        ):
            logger.info(
                "Model changed (%s → %s); reloading pipeline.",
                getattr(self._cached_definition, "model_name_or_path", None),
                definition.model_name_or_path,
            )
            self._release_pipeline()

        new_hash = hash_pipeline(definition)
        logger.info(
            "Building pipeline (hash=%s, controls=%d).",
            new_hash, len(definition.nodes),
        )
        pipeline = self.workbench.build_pipeline(definition)
        pipeline.steer()

        self._pipeline = pipeline
        self._pipeline_hash = new_hash
        self._cached_definition = definition

    def _release_pipeline(self) -> None:
        if self._pipeline is None:
            return
        try:
            cleanup = getattr(self._pipeline, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception as exc:
            logger.debug("Pipeline cleanup raised: %s", exc)
        self._pipeline = None
        self._pipeline_hash = None
        self._cached_definition = None

    def _graceful_close(self) -> None:
        try:
            self.client.session_close()
        except Exception as exc:
            logger.debug("session_close failed: %s", exc)


__all__ = ["SessionRunner"]
