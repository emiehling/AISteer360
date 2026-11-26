"""Sweeps over recipes under one protocol: expansion, support checks, and shared execution.

`expand_configurations` enumerates the points of a pipelines mapping (recipes and templates by
name), and `preflight` checks backend support for every point before any model or engine work.
`PipelineFactory` builds and steers one pipeline per recipe. It serves fits from the fit cache,
fits the products of structural controls (trainings and merges) before anything else of a recipe,
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

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import Artifact
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.fits import (
    FitCache,
    apply_control_hits,
    control_fit_digests,
    fit_cache_scope,
    fit_key,
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
        structural_fits: Products of structural controls (trainings and merges) that were fitted
            and stored.
        model_loads: Models loaded in process on the Hugging Face backend, i.e., the shared base
            model and the model of each pipeline that does not use it.
        engine_boots: Engine backends the pool built.
        engine_releases_for_fits: Times the pool released its engines so a fit could run on a
            staged in-process model.
        setup_seconds: Total time spent building and steering pipelines.
    """

    pipelines: int = 0
    fits_run: int = 0
    fits_reused: int = 0
    structural_fits: int = 0
    model_loads: int = 0
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


def _installs_adapters(*, controls: Sequence[Any]) -> bool:
    """Whether every enabled structural control installs a LoRA adapter from disk (none trains or merges here).

    Read under the recipe's fit cache, where a control whose stored product is an adapter declares
    `ModelAccess.FACTS`.
    """
    return all(
        control.steer_access() == ModelAccess.FACTS and control.artifact_capability() == Capability.SERVE_LORA
        for control in controls if isinstance(control, StructuralControl) and getattr(control, "enabled", True)
    )


class PipelineFactory:
    """Build and steer one pipeline per recipe under one protocol, sharing fits, models, and engines.

    Each `steered` call instantiates the recipe, replaces controls whose fits are stored by their
    frozen forms, and steers inside the recipe's fit cache, so stored fits are not repeated. A
    structural control that trains or merges has one fit, its product, which is fitted once per
    fit key in a pipeline of its own and stored in the store's artifacts. No other fit and no
    generation runs on weights that are not a stored product, since every pipeline that `steered`
    yields installs its structural products from the store. On the Hugging Face backend, recipes
    without an enabled structural control share one preloaded base model, checked by a
    fingerprint after each recipe. Engine backends are kept in a `BackendPool` until `release()`,
    so a recipe whose fits are stored costs control construction, `check()`, binding, and
    lowering. One steered pipeline generates at a time on the Hugging Face backend, while several
    may generate concurrently on a pooled engine.

    On the Hugging Face backend, a recipe whose structural controls all install a LoRA adapter
    from disk (`steer_access()` is `FACTS` and `artifact_capability()` is `SERVE_LORA` under the
    recipe's fit cache) also uses the shared base model. When its pipeline is released, each
    structural control undoes its change, last first (`StructuralControl.restore`), and the
    fingerprint check confirms the base model. A change that cannot be undone (`restore` returns
    None) drops the shared base model, and the next recipe that uses it loads it again.

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
        self.stats.model_loads += 1
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

    def _pipeline(self, controls: list, *, own_model: bool = False) -> tuple[SteeringPipeline, bool]:
        """A pipeline for `controls` and whether it uses the shared base model.

        On the Hugging Face backend the shared base model is used unless `own_model` is set or a structural control
        does more than install an adapter from disk (`_installs_adapters`, read under the active fit cache).
        """
        shared = self.backend_kind == "huggingface" and not own_model and _installs_adapters(controls=controls)
        common: dict[str, Any] = {
            "controls": controls, "backend": self.protocol.backend, "fit": self.protocol.fit,
            "backend_pool": self.pool,
        }
        if shared:
            self._ensure_base_model()
            return SteeringPipeline(model=self._base_model, tokenizer=self._base_tokenizer, **common), True
        if self.backend_kind == "huggingface":
            self._drop_shared_base()
            self.stats.model_loads += 1
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

    def pending_structural_fits(self, *, recipes: Iterable[Recipe]) -> frozenset[str]:
        """The fit keys of the structural fits (trainings and merges) that recipes need and the store does not contain.

        Each enabled structural control that declares a fit contributes the key of each of its digests
        (`control_fit_digests`) under its recipe's weights context. Without a store, every such fit is pending.
        No model is loaded.

        Args:
            recipes: The recipes.

        Returns:
            The pending fit keys.
        """
        pending: set[str] = set()
        for recipe in dict.fromkeys(recipes):
            weights = weights_context(recipe=recipe)
            for control in recipe.instantiate():
                if not isinstance(control, StructuralControl):
                    continue
                for digest in control_fit_digests(control=control):
                    key = fit_key(protocol=self.protocol, weights=weights, digest=digest)
                    if self.store is None or self.store.get_fit(fit_key=key) is None:
                        pending.add(key)
        return frozenset(pending)

    def _fit_products(self, *, recipe: Recipe, cache: FitCache) -> None:
        """Fit the products of a recipe's structural controls that the cache does not contain, alone.

        The structural controls are fitted one at a time in entry order. Each is steered in a pipeline of the
        structural controls up to and including it, under the recipe's fit cache and on a model of its own (never
        the shared base model). The controls before it are cache hits because they were fitted first. A plan that
        stages runs `fit_staged` after the pool's engines are released, and any other plan runs `steer`. The
        pipeline is released afterwards.

        Args:
            recipe: The recipe.
            cache: The recipe's fit cache.

        Raises:
            RuntimeError: If a structural control declares a fit and its product is not stored afterwards.
        """
        controls = [control for control in recipe.instantiate() if isinstance(control, StructuralControl)]
        for index, control in enumerate(controls):
            digests = control_fit_digests(control=control)
            if all(cache.has(digest=digest) for digest in digests):
                continue
            prefix = [item for item in recipe.instantiate() if isinstance(item, StructuralControl)][:index + 1]
            fitted = cache.product_count
            pipeline, _ = self._pipeline(controls=prefix, own_model=True)
            try:
                with fit_cache_scope(cache=cache):
                    if pipeline.check().plan.stages:
                        self.pool.release_all(for_fits=self.pool.has_engine())
                        pipeline.fit_staged()
                    else:
                        pipeline.steer()
            finally:
                for item in prefix:
                    try:
                        item.cleanup()
                    except Exception:
                        logger.warning("Control cleanup failed", exc_info=True)
                pipeline.release_backends()
                del pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not all(cache.has(digest=digest) for digest in digests):
                raise RuntimeError(
                    f"{type(control).__name__} declares a fit, and its product was not stored; a structural "
                    "control that trains or merges resolves its product through resolve_product, and a merged "
                    "LoRA run writes a product only to merged_output_dir."
                )
            self.stats.structural_fits += cache.product_count - fitted

    @contextmanager
    def steered(self, recipe: Recipe) -> Iterator[Steered]:
        """Build and steer the pipeline of one recipe, releasing it on exit.

        The products of the recipe's structural controls that the store does not contain are fitted
        first, each in a pipeline of its own. The recipe's pipeline then installs every product
        from the store. The yielded pipeline trains nothing, and its other controls fit on the
        stored products. On exit (also on error) every control's `cleanup()` runs, the pipeline's
        own backends are released, and the shared-base fingerprint check runs. Pooled backends
        stay up until `release()`.

        Args:
            recipe: The recipe.

        Yields:
            The `Steered` pipeline.

        Raises:
            RuntimeError: If a structural control's product is not stored after its fit, or a
                product is fitted while the recipe's own pipeline steers.
        """
        started = time.monotonic()
        cache = self._cache(recipe)
        self._fit_products(recipe=recipe, cache=cache)
        controls, pending = apply_control_hits(recipe.instantiate(), cache)
        pipeline: SteeringPipeline | None = None
        shared = False
        restored = True
        try:
            with fit_cache_scope(cache=cache):
                pipeline, shared = self._pipeline(controls=controls)
            products = cache.product_count
            with fit_cache_scope(cache):
                report = pipeline.check()
                if report.plan.stages and self.pool.has_engine():
                    self.pool.release_all(for_fits=True)
                pipeline.steer()
            if cache.product_count != products:
                raise RuntimeError(
                    f"A structural product of recipe {recipe.id} was fitted while the recipe's own pipeline "
                    "steered; every product is fitted alone before the pipeline that uses it."
                )
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
                if shared and _has_structural_control(controls=pipeline.structural_controls):
                    restored = self._restore_shared_base(pipeline=pipeline)
                pipeline.release_backends()
                del pipeline
            if shared and restored:
                self._verify_shared_base_model(recipe)
            elif shared:
                self._drop_shared_base()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _restore_shared_base(self, *, pipeline: SteeringPipeline) -> bool:
        """Undo the structural controls' changes to the shared base model, last control first.

        Returns:
            Whether every change was undone and the restored model is the shared base model.
        """
        model = pipeline.model
        for control in reversed([item for item in pipeline.structural_controls if item.enabled]):
            try:
                model = control.restore(model=model)
            except Exception:
                logger.warning(
                    "Restoring the shared base model after %s failed.",
                    type(control).__name__,
                    exc_info=True,
                )
                return False
            if model is None:
                return False
        return model is self._base_model

    def prepare(self, recipes: Iterable[Recipe]) -> None:
        """Run the fits a batch of recipes needs, so that later `steered` calls only bind.

        The products of structural controls (trainings and merges) are fitted first, each once
        and alone (as in `steered`). Recipes whose steer plan has no fit left under the cache are
        then skipped, as are recipes whose fits are not stored (a control without a fit identity),
        since their fits run again at every steer. When a plan has a step on a staged in-process
        model, the pool's engines are released once and every such recipe fits its staged steps
        (`SteeringPipeline.fit_staged`). Every planned recipe that still has a fit when its turn
        comes is then steered once through `steered`, with its staged fits served from the cache,
        so an engine boots once and stays up and recipes that share a fit are steered once
        between them. Call this once per batch, before generation.

        Args:
            recipes: The recipes.

        Raises:
            RuntimeError: If a structural control's product is not stored after its fit.
        """
        recipes = list(dict.fromkeys(recipes))
        for recipe in recipes:
            self._fit_products(recipe=recipe, cache=self._cache(recipe=recipe))
        planned = [(recipe, plan) for recipe in recipes if (plan := self._fit_plan(recipe=recipe)) is not None]
        staged = [recipe for recipe, plan in planned if plan.stages]
        if staged:
            self.pool.release_all(for_fits=self.pool.has_engine())
            for recipe in staged:
                cache = self._cache(recipe)
                controls, pending = apply_control_hits(recipe.instantiate(), cache)
                pipeline, _ = self._pipeline(controls=controls)
                try:
                    with fit_cache_scope(cache):
                        pipeline.fit_staged()
                    staged_pending = [control for control in pending if control in pipeline._enabled_controls()]
                    record_control_fits(staged_pending, cache)
                finally:
                    pipeline.release_backends()
        for recipe, _ in planned:
            if self._fit_plan(recipe=recipe) is None:
                continue
            with self.steered(recipe):
                pass

    def _fit_plan(self, *, recipe: Recipe) -> Any:
        """The steer plan of a recipe that still has a fit to run and store under its cache, or None."""
        cache = self._cache(recipe=recipe)
        controls, pending = apply_control_hits(controls=recipe.instantiate(), cache=cache)
        if not stores_a_fit(controls=controls, pending=pending, cache=cache):
            return None
        probe = SteeringPipeline(
            model_name_or_path=self.protocol.model,
            controls=controls,
            backend=self.protocol.backend,
            fit=self.protocol.fit,
        )
        with fit_cache_scope(cache=cache):
            plan = probe.check().plan
        return plan if plan.fits else None

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
