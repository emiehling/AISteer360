"""Sweeps over recipes under one protocol: expansion, support checks, and shared execution.

`expand_configurations` enumerates the points of a pipelines mapping (recipes and templates by
name), and `preflight` checks backend support for every point before any model or engine work.
`PipelineFactory` builds and steers one pipeline per recipe. It serves fits from the fit cache,
shares one preloaded base model on the Hugging Face backend, and keeps engine backends in a
`BackendPool` across recipes, so one engine serves every recipe whose fits are stored.
"""
from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import Artifact
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.fits import (
    FitCache,
    apply_control_hits,
    fit_cache_scope,
    record_control_fits,
    resolved_id,
    stores_a_fit,
    weights_context,
)
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.protocol import Protocol, ProtocolFacts
from steerability.algorithms.core.recipe import Point, Recipe, Template, expand
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.utils.tokenization import ensure_pad_token

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import Backend
    from steerability.algorithms.core.store import Store

logger = logging.getLogger(__name__)

Pipelines = Mapping[str, Recipe | Template | Sequence[Recipe | Template]]


def expand_configurations(pipelines: Pipelines) -> Iterator[tuple[str, Point]]:
    """Enumerate the points of a pipelines mapping, in order.

    Each value is a recipe, a template (expanded as a grid), or a sequence of them, flattened in
    order. The unsteered baseline is `BASELINE` (the empty recipe).

    Args:
        pipelines: Mapping from pipeline name to recipes and templates.

    Yields:
        `(pipeline name, point)` pairs.

    Raises:
        ValueError: If a pipeline has no recipe, or two of its points have the same recipe id.
        RecipeError: If a template does not expand as a grid.
    """
    for name, value in pipelines.items():
        sources = [value] if isinstance(value, (Recipe, Template)) else list(value)
        if not sources:
            raise ValueError(f"Pipeline {name!r} has no recipe; use BASELINE for the unsteered baseline.")
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, (Recipe, Template)):
                raise TypeError(f"Pipeline {name!r} contains a {type(source).__name__}; expected Recipe or Template.")
            for point in expand(source, strategy="grid"):
                if point.recipe.id in seen:
                    raise ValueError(f"Pipeline {name!r} has two points with recipe id {point.recipe.id}.")
                seen.add(point.recipe.id)
                yield name, point


def preflight(points: Iterable[tuple[str, Point]], protocol: Protocol) -> list[str]:
    """Check backend support for every point before any model or engine work.

    Each point's controls are checked by an unsteered pipeline on the protocol's backend
    (construction performs no I/O). The empty recipe is always supported.

    Args:
        points: `(pipeline name, point)` pairs.
        protocol: The protocol.

    Returns:
        One message per unsupported (point, control, phase), each naming the pipeline and the
        recipe id. Empty when every point is supported.
    """
    messages: list[str] = []
    for name, point in points:
        controls = point.recipe.instantiate()
        if not controls:
            continue
        probe = SteeringPipeline(
            model_name_or_path=protocol.model, controls=controls, backend=protocol.backend, fit=protocol.fit,
        )
        report = probe.check()
        for failure in report.failures:
            messages.append(f"{name} [{point.recipe.id}] {failure.control} ({failure.phase}): {failure.message}")
    return messages


@dataclass
class FactoryStats:
    """Counters of a `PipelineFactory`.

    Attributes:
        pipelines: Pipelines steered.
        fits_run: Fits that ran and were stored.
        fits_reused: Fits served from the cache.
        engine_boots: Engine backends the pool built.
        engine_releases_for_fits: Times the pool released its engines so a fit could run on a
            staged in-process model.
        setup_seconds: Total time spent building and steering pipelines.
    """

    pipelines: int = 0
    fits_run: int = 0
    fits_reused: int = 0
    engine_boots: int = 0
    engine_releases_for_fits: int = 0
    setup_seconds: float = 0.0


def _artifact_key(artifacts: Sequence[Artifact]) -> tuple:
    return tuple((type(artifact).__name__, str(artifact.path), getattr(artifact, "base_model", None))
                 for artifact in artifacts)


class BackendPool:
    """Engine backends shared across the pipelines of one factory.

    One backend is kept per spec, keyed by the spec and the structural artifacts it serves; a
    request for the same spec with other artifacts releases the kept backend first, so two
    engines of one spec are never up at once.

    Args:
        stats: The factory counters the pool updates (`engine_boots`,
            `engine_releases_for_fits`), or None.
    """

    def __init__(self, stats: FactoryStats | None = None) -> None:
        self.stats = stats
        self._backends: dict[BackendSpec, tuple[tuple, Backend]] = {}

    def get(self, spec: BackendSpec, artifacts: Sequence[Artifact], build: Callable[[], Backend]) -> Backend:
        """The pooled backend for `spec` serving `artifacts`, built with `build` on first use."""
        key = _artifact_key(artifacts)
        kept = self._backends.get(spec)
        if kept is not None and kept[0] == key:
            return kept[1]
        if kept is not None:
            self._release(spec)
        backend = build()
        self._backends[spec] = (key, backend)
        if self.stats is not None:
            self.stats.engine_boots += 1
        return backend

    def has_engine(self) -> bool:
        """Whether a pooled backend lacks `Capability.IN_PROCESS_TORCH` (an engine is up)."""
        return any(
            Capability.IN_PROCESS_TORCH not in backend.negotiated_capabilities().atoms
            for _, backend in self._backends.values()
        )

    def _release(self, spec: BackendSpec) -> None:
        _, backend = self._backends.pop(spec)
        try:
            backend.release()
        except Exception:
            logger.warning("Backend release failed", exc_info=True)

    def release_all(self, *, for_fits: bool = False) -> None:
        """Release every pooled backend.

        Args:
            for_fits: Whether the release makes room for a fit on a staged model, counted in
                `engine_releases_for_fits`.
        """
        if for_fits and self.stats is not None and self._backends:
            self.stats.engine_releases_for_fits += 1
        for spec in list(self._backends):
            self._release(spec)
        gc.collect()


@dataclass(frozen=True)
class Steered:
    """A steered pipeline with the identity of what it runs.

    Attributes:
        pipeline: The steered pipeline.
        recipe: The recipe it was built from.
        resolved_id: The recipe's id resolved with the fits it used, the key of its rollouts
            and evidence.
        protocol: The protocol it runs under.
    """

    pipeline: SteeringPipeline
    recipe: Recipe
    resolved_id: str
    protocol: Protocol


def _has_structural_control(controls: Sequence[Any]) -> bool:
    return any(isinstance(control, StructuralControl) and getattr(control, "enabled", True) for control in controls)


class PipelineFactory:
    """Build and steer one pipeline per recipe under one protocol, sharing fits, models, and engines.

    Each `steered` call instantiates the recipe, replaces controls whose fits are stored by their
    frozen forms, and steers inside the recipe's fit cache, so stored fits are not repeated. On the
    Hugging Face backend, recipes without an enabled structural control share one preloaded base
    model, checked by a fingerprint after each recipe. Engine backends are kept in a `BackendPool`
    until `release()`, so a recipe whose fits are stored costs control construction, `check()`,
    binding, and lowering. One steered pipeline generates at a time on the Hugging Face backend,
    while several may generate concurrently on a pooled engine.

    Args:
        protocol: The protocol.
        store: The store fits, recipes, resolutions, and observed facts are written to, or None
            for an in-memory cache per recipe.
        device_map: Device placement for in-process loads.
        trust_remote_code: Trust remote code when loading tokenizers.

    Attributes:
        stats: The factory's counters.
    """

    def __init__(
        self,
        protocol: Protocol,
        *,
        store: Store | None = None,
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.protocol = protocol
        self.store = store
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.stats = FactoryStats()
        self.pool = BackendPool(self.stats)
        self._base_model: PreTrainedModel | None = None
        self._base_tokenizer: PreTrainedTokenizerBase | None = None
        self._base_fingerprint: str | None = None
        self._facts_recorded = False

    @property
    def backend_kind(self) -> str:
        """The protocol's backend kind (`"huggingface"`, `"vllm"`, or `"vllm-serve"`)."""
        return self.protocol.backend.kind if self.protocol.backend is not None else "huggingface"

    @property
    def shared_base_fingerprint(self) -> str | None:
        """Fingerprint of the resident shared base model, or None when none is resident."""
        return self._base_fingerprint

    def _tokenizer_source(self) -> tuple[str, str | None]:
        if self.protocol.tokenizer is not None:
            return self.protocol.tokenizer, None
        return self.protocol.model, self.protocol.model_revision

    def _ensure_base_model(self) -> None:
        if self._base_model is not None and self._base_tokenizer is not None:
            return
        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.protocol.model, device_map=self.device_map, **self.protocol.load_kwargs,
        )
        source, revision = self._tokenizer_source()
        self._base_tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(
            source, revision=revision, trust_remote_code=self.trust_remote_code,
        ))
        try:
            self._base_fingerprint = model_fingerprint(self._base_model)
        except Exception:
            logger.debug("Model fingerprint unavailable; shared-model guard disabled.", exc_info=True)
            self._base_fingerprint = None
        self._record_facts(self._base_model, self._base_tokenizer)

    def _record_facts(self, model, tokenizer) -> None:
        if self.store is None or self._facts_recorded:
            return
        self.store.put_protocol(self.protocol, ProtocolFacts.observe(model, tokenizer))
        self._facts_recorded = True

    def _drop_shared_base(self) -> None:
        if self._base_model is None and self._base_tokenizer is None:
            return
        self._base_model = None
        self._base_tokenizer = None
        self._base_fingerprint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _verify_shared_base_model(self, recipe: Recipe) -> None:
        if self._base_model is None or self._base_fingerprint is None:
            return
        try:
            current = model_fingerprint(self._base_model)
        except Exception:
            return
        if current == self._base_fingerprint:
            return
        logger.warning(
            "Shared base model changed while steering recipe %s (fingerprint %s -> %s); its results reflect the "
            "mutated weights. Dropping the shared model so the next recipe reloads a clean base.",
            recipe.id, self._base_fingerprint, current,
        )
        self._drop_shared_base()

    def _pipeline(self, controls: list) -> tuple[SteeringPipeline, bool]:
        """A pipeline for `controls` and whether it uses the shared base model."""
        shared = self.backend_kind == "huggingface" and not _has_structural_control(controls)
        common: dict[str, Any] = {
            "controls": controls, "backend": self.protocol.backend, "fit": self.protocol.fit,
            "backend_pool": self.pool,
        }
        if shared:
            self._ensure_base_model()
            return SteeringPipeline(model=self._base_model, tokenizer=self._base_tokenizer, **common), True
        if self.backend_kind == "huggingface":
            self._drop_shared_base()
        return SteeringPipeline(
            model_name_or_path=self.protocol.model,
            tokenizer_name_or_path=self.protocol.tokenizer,
            device_map=self.device_map,
            hf_model_kwargs=self.protocol.load_kwargs,
            trust_remote_code=self.trust_remote_code,
            **common,
        ), False

    def _cache(self, recipe: Recipe) -> FitCache:
        return FitCache(self.protocol, weights_context(recipe), self.store)

    @contextmanager
    def steered(self, recipe: Recipe) -> Iterator[Steered]:
        """Build and steer the pipeline of one recipe, releasing it on exit.

        On exit (also on error) every control's `cleanup()` runs, the pipeline's own backends are
        released, and the shared-base fingerprint check runs. Pooled backends stay up until
        `release()`.

        Args:
            recipe: The recipe.

        Yields:
            The `Steered` pipeline.
        """
        started = time.monotonic()
        cache = self._cache(recipe)
        controls, pending = apply_control_hits(recipe.instantiate(), cache)
        pipeline: SteeringPipeline | None = None
        shared = False
        try:
            pipeline, shared = self._pipeline(controls)
            with fit_cache_scope(cache):
                report = pipeline.check()
                if report.plan.stages and self.pool.has_engine():
                    self.pool.release_all(for_fits=True)
                pipeline.steer()
            record_control_fits(pending, cache)
            identity = resolved_id(recipe, cache.used())
            if not shared and not _has_structural_control(controls):
                self._record_facts(None, pipeline.tokenizer)
            if self.store is not None:
                self.store.put_recipe(recipe)
                if not cache.has_local():
                    self.store.put_resolution(self.protocol.id, recipe.id, identity)
            self.stats.pipelines += 1
            self.stats.fits_run += cache.stored_count
            self.stats.fits_reused += cache.hit_count
            self.stats.setup_seconds += time.monotonic() - started
            yield Steered(pipeline, recipe, identity, self.protocol)
        finally:
            if pipeline is not None:
                for control in (*pipeline.structural_controls, *pipeline.input_controls,
                                *pipeline.state_controls, *pipeline.output_controls):
                    try:
                        control.cleanup()
                    except Exception:
                        logger.warning("Control cleanup failed", exc_info=True)
                pipeline.release_backends()
                del pipeline
            if shared:
                self._verify_shared_base_model(recipe)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare(self, recipes: Iterable[Recipe]) -> None:
        """Run the fits a batch of recipes needs, so that later `steered` calls only bind.

        Recipes whose steer plan has no fit under the cache are skipped, as are recipes whose fits
        are not stored (a recipe with a structural control, or a control without a fit identity),
        since their fits run again at every steer. When a plan has a step on a staged in-process
        model, the pool's engines are released once and every such recipe fits its staged steps
        (`SteeringPipeline.fit_staged`). Every recipe that still has a fit is then steered once
        through `steered`, with its staged fits now served from the cache, so an engine boots once
        and stays up. Call this once per batch, before generation.

        Args:
            recipes: The recipes.
        """
        planned: list[tuple[Recipe, Any]] = []
        for recipe in dict.fromkeys(recipes):
            cache = self._cache(recipe)
            controls, pending = apply_control_hits(recipe.instantiate(), cache)
            if not stores_a_fit(controls, pending, cache):
                continue
            probe = SteeringPipeline(
                model_name_or_path=self.protocol.model, controls=controls, backend=self.protocol.backend,
                fit=self.protocol.fit,
            )
            with fit_cache_scope(cache):
                plan = probe.check().plan
            if plan.fits:
                planned.append((recipe, plan))
        staged = [recipe for recipe, plan in planned if plan.stages]
        if staged:
            self.pool.release_all(for_fits=self.pool.has_engine())
            for recipe in staged:
                cache = self._cache(recipe)
                controls, pending = apply_control_hits(recipe.instantiate(), cache)
                pipeline, _ = self._pipeline(controls)
                try:
                    with fit_cache_scope(cache):
                        pipeline.fit_staged()
                    staged_pending = [control for control in pending if control in pipeline._enabled_controls()]
                    record_control_fits(staged_pending, cache)
                finally:
                    pipeline.release_backends()
        for recipe, _ in planned:
            with self.steered(recipe):
                pass

    def release(self) -> None:
        """Release the pooled backends and drop the shared base model; the factory stays usable."""
        self.pool.release_all()
        self._drop_shared_base()


__all__ = [
    "BackendPool",
    "FactoryStats",
    "PipelineFactory",
    "Pipelines",
    "Steered",
    "expand_configurations",
    "preflight",
]
