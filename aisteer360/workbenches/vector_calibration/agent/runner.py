"""Agent runner: orchestrates the three pipeline stages against a remote server.

The runner is the agent-side analog of the old `_pipeline_task` on the server. All compute
happens here; the server sees only progress POSTs and artefact uploads.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from aisteer360.workbenches.vector_calibration import VectorCalibrationWorkbench
from aisteer360.workbenches.vector_calibration.configs import CalibrationBuilderConfig

from .client import ServerClient
from .config_loader import from_server_config
from .providers.base import (
    GenerationProvider,
    JudgeProvider,
    ProviderKeys,
    build_generation_provider,
    build_judge_provider,
)

logger = logging.getLogger(__name__)


class _CancelPoller:
    """Background thread that polls the server's cancel flag."""

    def __init__(self, client: ServerClient, interval_s: float = 2.0):
        self._client = client
        self._interval = interval_s
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval + 1.0)

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._client.check_cancel():
                    self._event.set()
                    return
            except Exception as exc:
                logger.debug("cancel-check failed: %s", exc)
            self._stop.wait(self._interval)


class AgentRunner:
    """Execute the full three-stage pipeline for one run."""

    def __init__(self, client: ServerClient):
        self.client = client
        self._workbench: VectorCalibrationWorkbench | None = None
        self._gen_provider: GenerationProvider | None = None
        self._judge_provider: JudgeProvider | None = None
        self._poller: _CancelPoller | None = None

    def run(self) -> None:
        """Claim the run, execute the requested stages, report to the server."""
        claim = self.client.claim()
        raw_cfg = claim["config"]
        run_dir = Path(claim["run_dir"])
        stages = set(claim["stages"])
        logger.info("Claimed run %s at %s; stages=%s", self.client.run_id, run_dir, sorted(stages))

        server_keys = claim.get("provider_keys") or {}
        keys = ProviderKeys(
            hf_token=server_keys.get("hf_token"),
            anthropic_key=server_keys.get("anthropic_key"),
            openai_key=server_keys.get("openai_key"),
        )

        try:
            cfg = from_server_config(raw_cfg)

            if "generation" in stages:
                self._gen_provider = build_generation_provider(raw_cfg, keys)
            if "calibration" in stages:
                self._judge_provider = build_judge_provider(raw_cfg, keys)

            workbench = VectorCalibrationWorkbench(cfg)
            workbench._run_dir = run_dir  # force the active run dir
            self._workbench = workbench

            self._poller = _CancelPoller(self.client)
            self._poller.start()

            if "generation" in stages:
                self._run_stage_generation(workbench, run_dir)
                if self._poller.is_cancelled():
                    raise _Cancelled()
            if "extraction" in stages:
                self._run_stage_extraction(workbench, run_dir, cfg.generation.behavior)
                if self._poller.is_cancelled():
                    raise _Cancelled()
            if "calibration" in stages:
                self._run_stage_calibration(workbench, run_dir)
            self.client.complete()
            logger.info("Run %s complete.", self.client.run_id)
        except _Cancelled:
            logger.info("Run %s cancelled.", self.client.run_id)
            self.client.error("cancelled by user")
        except Exception as exc:
            logger.exception("Run %s failed.", self.client.run_id)
            try:
                self.client.error(str(exc))
            except Exception:
                logger.warning("Failed to report error to server.")
            raise
        finally:
            if self._poller is not None:
                self._poller.stop()
            self._release_providers()
            if self._workbench is not None:
                self._workbench.cleanup()

    # ── stages ───────────────────────────────────────────────────

    def _run_stage_generation(
        self,
        workbench: VectorCalibrationWorkbench,
        run_dir: Path,
    ) -> None:
        stage = "generation"
        self.client.stage_start(stage)

        total_hint = self._seed_total(workbench.config)
        self.client.post_progress(stage, completed=0, total=total_hint)

        def report(done: int, total: int) -> None:
            self.client.post_progress(stage, completed=done, total=total)

        result = workbench.run_generation(
            on_progress=report,
            run_dir=run_dir,
            cancel_check=self._cancel_check,
            generation_provider=self._gen_provider,
        )
        self._post_model_info_if_loaded(workbench)

        pairs_path = run_dir / "pairs.jsonl"
        if pairs_path.exists():
            self.client.upload_artifact("pairs", pairs_path)
        self.client.stage_complete(stage, notes=f"{len(result.pairs.positives)} pairs")
        if self._poller and self._poller.is_cancelled():
            raise _Cancelled()

    def _run_stage_extraction(
        self,
        workbench: VectorCalibrationWorkbench,
        run_dir: Path,
        behavior: str,
    ) -> None:
        stage = "extraction"
        self.client.stage_start(stage)
        self.client.post_progress(stage, completed=0, total=1)
        workbench.run_extraction(run_dir=run_dir)
        self.client.post_progress(stage, completed=1, total=1)
        self._post_model_info_if_loaded(workbench)

        svec_path = run_dir / f"{behavior}.svec"
        if svec_path.exists():
            self.client.upload_artifact("svec", svec_path)
        self.client.stage_complete(stage)

    def _run_stage_calibration(
        self,
        workbench: VectorCalibrationWorkbench,
        run_dir: Path,
    ) -> None:
        stage = "calibration"
        self.client.stage_start(stage)

        def report(data: dict[str, Any]) -> None:
            payload = {k: v for k, v in data.items() if k not in ("completed", "total")}
            self.client.post_progress(
                stage,
                completed=data.get("completed"),
                total=data.get("total"),
                payload=payload,
            )

        workbench.run_calibration(
            run_dir=run_dir,
            on_progress=report,
            judge_provider=self._judge_provider,
        )
        result_path = run_dir / "calibration_result.json"
        if result_path.exists():
            self.client.upload_artifact("calibration_result", result_path)
        self.client.stage_complete(stage)

    # ── helpers ──────────────────────────────────────────────────

    def _cancel_check(self) -> bool:
        return bool(self._poller and self._poller.is_cancelled())

    def _post_model_info_if_loaded(self, workbench: VectorCalibrationWorkbench) -> None:
        model = workbench._model
        if model is None:
            return
        cfg = model.config
        info = {
            "model_name": workbench.config.steered_model,
            "num_layers": getattr(cfg, "num_hidden_layers", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
            "dtype": str(model.dtype),
            "device": str(model.device),
            "model_type": getattr(cfg, "model_type", None),
        }
        try:
            self.client.post_model_info(info)
        except Exception as exc:
            logger.debug("post_model_info failed: %s", exc)

    @staticmethod
    def _seed_total(cfg: CalibrationBuilderConfig) -> int | None:
        seeds = cfg.generation.seed_prompts
        if isinstance(seeds, list):
            return len(seeds)
        return None

    def _release_providers(self) -> None:
        for provider in (self._gen_provider, self._judge_provider):
            if provider is None:
                continue
            try:
                provider.close()
            except Exception:
                pass


class _Cancelled(Exception):
    """Internal sentinel for graceful cancellation."""


__all__ = ["AgentRunner"]
