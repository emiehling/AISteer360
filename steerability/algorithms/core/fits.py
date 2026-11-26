"""Fits: their keys, the fit cache, and the routes through which steering consults it.

A fit is the model-side work a control's `steer()` does to produce a reusable artifact. Its digest
(`fit_digest`) covers the inputs of that work, and its key (`fit_key`) adds the protocol's fit
context and the recipe's structural weights. The `FitCache` serves stored fits by key, so a recipe
whose fits are all stored steers without model work.

Three routes consult the cache. Intervention controls resolve their fit sources through
`resolve_source`, which returns a stored value on a hit and stores the result of a miss; while a
cache is active, `source_access` reports `ModelAccess.FACTS` for a stored source, so the steer
plan needs no model for it. A structural control that trains or merges has one fit, its product
(an adapter or a checkpoint in the store's artifacts), which it resolves through `resolve_product`
in the same way. While the cache contains the product (`product_is_stored`), the control
installs it and declares `ModelAccess.FACTS`. Other controls with a fit identity are replaced before steering by
their frozen form on a hit (`apply_control_hits`), and their exported state is stored after
steering (`record_control_fits`).

A recipe's resolved id (`resolved_id`) combines the recipe id with the stored values of every
fit it used, so rollouts and evidence keyed by it never mix two fits of one recipe.
"""
from __future__ import annotations

import contextvars
import dataclasses
import json
import logging
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.identity import content_id, json_id

if TYPE_CHECKING:
    from steerability.algorithms.core.base_control import BaseControl
    from steerability.algorithms.core.protocol import Protocol
    from steerability.algorithms.core.recipe import Recipe
    from steerability.algorithms.core.store import Store
    from steerability.spipe.store import ArtifactStore

logger = logging.getLogger(__name__)

_PROCESS_TOKEN = uuid.uuid4().hex[:12]

_ACTIVE: contextvars.ContextVar[FitCache | None] = contextvars.ContextVar("steerability_fit_cache", default=None)

_WARNED_KINDS: set[str] = set()


def _qualname(value_type: type) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def weights_context(recipe: Recipe) -> str:
    """The identity of the weights a recipe's fits run on.

    Returns:
        The content id of the canonical entries whose method is a structural control, in entry
        order, or `"base"` when the recipe has none.
    """
    structural = [item for item in recipe.canonical()["entries"] if item["method"].startswith("structural_control/")]
    return json_id(structural) if structural else "base"


def source_identity(source: Any) -> Any:
    """The fit identity of a fit source.

    Returns:
        `source.fit_ingredients()` when the source defines it, else the init fields of a dataclass
        source.

    Raises:
        TypeError: If the source defines neither.
    """
    ingredients = getattr(source, "fit_ingredients", None)
    if callable(ingredients):
        return ingredients()
    if dataclasses.is_dataclass(source) and not isinstance(source, type):
        return {field.name: getattr(source, field.name) for field in dataclasses.fields(source) if field.init}
    raise TypeError(
        f"{type(source).__name__} has no fit identity; define fit_ingredients() or make it a dataclass."
    )


def fit_digest(kind: str, identity: Any) -> str:
    """The digest of a fit, used by the cache and by bundles.

    Args:
        kind: The qualified class name of a fit source, or the method key of a control.
        identity: The fit identity.

    Returns:
        `content_id({"kind": kind, "identity": identity})`.

    Raises:
        SpipeSaveError: If the identity has no serialized form.
    """
    return content_id({"kind": kind, "identity": identity})


def source_digest(source: Any) -> str:
    """The digest of a fit source: `fit_digest` of its qualified class name and its identity.

    Raises:
        TypeError: If the source has no fit identity.
        SpipeSaveError: If its identity has no serialized form.
    """
    return fit_digest(_qualname(type(source)), source_identity(source))


def fit_key(protocol: Protocol, weights: str, digest: str) -> str:
    """The key of a fit: `content_id` of the protocol's fit context id, the weights context, and the digest."""
    return content_id({"context": protocol.fit_context_id, "weights": weights, "fit": digest})


def control_fit_digests(control: BaseControl) -> tuple[str, ...]:
    """The digests of the fits a control declares, in order, independent of any cache.

    An intervention control contributes one digest per fit source. Another control with a fit
    contributes the digest of its fit identity under its method key, or nothing when the
    identity is None. The fits a control declares are read with no cache active, since a
    control whose fit the active cache contains declares none.

    Args:
        control: A constructed control.

    Returns:
        The digests.
    """
    from steerability.algorithms.core.registry import method_key_for
    from steerability.algorithms.state_control.base import InterventionControl

    if not getattr(control, "enabled", True):
        return ()
    if isinstance(control, InterventionControl):
        return tuple(source_digest(source) for source in control.declared_fit_sources())
    with fit_cache_scope(cache=None):
        declared = control.steer_fits()
    if not declared:
        return ()
    identity = control.fit_identity()
    if identity is None:
        return ()
    return (fit_digest(method_key_for(type(control)), identity),)


def fit_digests(recipe: Recipe) -> tuple[str, ...]:
    """The digest of every fit the recipe's controls declare, in entry order. Needs no model.

    Two recipes with equal digests and equal weights contexts share every fit.

    Args:
        recipe: The recipe.

    Returns:
        The digests.
    """
    return tuple(digest for control in recipe.instantiate() for digest in control_fit_digests(control))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FitCache:
    """Stored fits for one pipeline, under one protocol and weights context.

    With a store, values are encoded in save mode into the store's artifacts and their JSON is kept
    in its `fits` table. Without one, the cache keeps them in memory with a temporary artifact
    store of its own. Two kinds of fit are not stored and are recorded as process-local: a value
    the encoder rejects (logged once per type at warning level), and the fit of a control whose
    `fit_identity()` is None (`mark_local`). A process-local fit makes the pipeline's resolved id
    unique to the process.

    Args:
        protocol: The protocol whose fit context keys the fits.
        weights: The recipe's weights context.
        store: The persistent store, or None.

    Attributes:
        hit_count: Values served by `get`.
        stored_count: Values written by `put` and `put_own`.
        product_count: Products of structural controls fitted and written by `resolve_product`.
    """

    def __init__(self, protocol: Protocol, weights: str, store: Store | None):
        self.protocol = protocol
        self.weights = weights
        self.store = store
        self._memory: dict[str, str] = {}
        self._used: dict[str, str] = {}
        self._local = False
        self.hit_count = 0
        self.stored_count = 0
        self.product_count = 0
        self._scratch: tempfile.TemporaryDirectory | None = None
        if store is None:
            from steerability.spipe.store import ArtifactStore

            self._scratch = tempfile.TemporaryDirectory(prefix="steerability-fits-")
            self._artifacts = ArtifactStore(Path(self._scratch.name))
        else:
            self._artifacts = store.artifacts

    @property
    def artifacts(self) -> ArtifactStore:
        """The artifact store that fitted values are encoded into."""
        return self._artifacts

    def key(self, digest: str) -> str:
        """The fit key of a digest under this cache's protocol and weights."""
        return fit_key(self.protocol, self.weights, digest)

    def _stored(self, key: str) -> str | None:
        if self.store is None:
            return self._memory.get(key)
        row = self.store.get_fit(key)
        return None if row is None else row.value

    def has(self, digest: str) -> bool:
        """Whether a value is stored for the digest. A stored None counts."""
        return self._stored(self.key(digest)) is not None

    def decode(self, value_json: str) -> Any:
        """A fresh object decoded from a stored value's JSON."""
        from steerability.spipe.codec import DecodeContext, decode

        return decode(json.loads(value_json), DecodeContext(store=self._artifacts, allow_code=True))

    def get(self, digest: str) -> Any | None:
        """The stored value for the digest, decoded freshly on every call, and recorded as used.

        None can be a stored value, so test with `has()`.
        """
        key = self.key(digest)
        value_json = self._stored(key)
        if value_json is None:
            return None
        self._used[key] = value_json
        self.hit_count += 1
        return self.decode(value_json)

    def encode(self, value: Any) -> str:
        """The JSON of a value encoded in save mode into this cache's artifact store.

        Raises:
            SpipeSaveError: If the value has no serialized form.
        """
        from steerability.spipe.codec import EncodeContext, encode

        return json.dumps(encode(value, EncodeContext(store=self._artifacts)), sort_keys=True, separators=(",", ":"))

    def _write(self, key: str, digest: str, value_json: str, seconds: float) -> str:
        if self.store is None:
            return self._memory.setdefault(key, value_json)
        from steerability.algorithms.core.store import FitRow

        row = self.store.put_fit(FitRow(
            fit_key=key, fit_context_id=self.protocol.fit_context_id, weights_context=self.weights,
            fit_digest=digest, value=value_json, seconds=float(seconds), created_at=_now(),
        ))
        return row.value

    def put(self, digest: str, value: Any, seconds: float) -> Any:
        """Store a fitted value and return the stored value, which the caller must use.

        When another writer stored the key first, its value is returned. A value the encoder
        rejects is not stored: it is recorded as a process-local fit and returned unchanged.

        Args:
            digest: The fit's digest.
            value: The fitted value.
            seconds: The duration of the fit.

        Returns:
            A freshly decoded copy of the stored value, or `value` for a process-local fit.
        """
        from steerability.spipe.errors import SpipeSaveError

        key = self.key(digest)
        try:
            value_json = self.encode(value)
        except SpipeSaveError as exc:
            kind = _qualname(type(value))
            if kind not in _WARNED_KINDS:
                _WARNED_KINDS.add(kind)
                logger.warning("A fitted %s has no serialized form (%s); it is not stored and refits in every "
                               "process.", kind, exc)
            self._used[f"{key}:local-{_PROCESS_TOKEN}"] = json.dumps(kind)
            self._local = True
            return value
        stored = self._write(key, digest, value_json, seconds)
        self._used[key] = stored
        self.stored_count += 1
        return self.decode(stored)

    def put_own(self, digest: str, value: Any, seconds: float) -> None:
        """Store a fitted value that the caller keeps using, recording the caller's own value as used.

        The hook route keeps each control's own steered state. When another writer stored the key
        first, that row is kept and the caller's value is recorded under a process-local key, so
        the resolved id reflects the value that was applied.

        Args:
            digest: The fit's digest.
            value: The fitted value the caller applies.
            seconds: The duration of the fit.
        """
        from steerability.spipe.errors import SpipeSaveError

        key = self.key(digest)
        try:
            value_json = self.encode(value)
        except SpipeSaveError:
            self.put(digest, value, seconds)
            return
        stored = self._write(key, digest, value_json, seconds)
        self.stored_count += 1
        if stored == value_json:
            self._used[key] = stored
        else:
            logger.info("Fit %s was stored by another writer first; this pipeline applies its own fit.", key)
            self._used[f"{key}:local-{_PROCESS_TOKEN}"] = value_json
            self._local = True

    def mark_local(self, kind: str) -> None:
        """Record a fit that is not stored, e.g. of a control whose fit identity is None."""
        self._used[f"{kind}:local-{_PROCESS_TOKEN}"] = json.dumps(kind)
        self._local = True

    def used(self) -> dict[str, str]:
        """Mapping from fit key to the stored value JSON of every hit and put of this pipeline."""
        return dict(self._used)

    def has_local(self) -> bool:
        """Whether a process-local fit was recorded."""
        return self._local


@contextmanager
def fit_cache_scope(cache: FitCache | None) -> Iterator[None]:
    """Make `cache` the active fit cache for the enclosed block (a context variable)."""
    token = _ACTIVE.set(cache)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_fit_cache() -> FitCache | None:
    """The fit cache of the enclosing `fit_cache_scope`, or None."""
    return _ACTIVE.get()


def _cached_digest(source: Any) -> str | None:
    """The source's digest when it identifies itself, else None."""
    from steerability.spipe.errors import SpipeSaveError

    try:
        return source_digest(source)
    except (TypeError, SpipeSaveError):
        return None


def source_access(source: Any) -> ModelAccess:
    """The model access a fit source needs under the active cache.

    Returns:
        `ModelAccess.FACTS` when the active cache has a value for the source, else the source's
        own `access` (`ModelAccess.MODULE` when it declares none).
    """
    declared = getattr(source, "access", ModelAccess.MODULE)
    cache = active_fit_cache()
    if cache is None or declared == ModelAccess.FACTS:
        return declared
    digest = _cached_digest(source)
    if digest is not None and cache.has(digest):
        return ModelAccess.FACTS
    return declared


def resolve_source(source: Any, model, tokenizer, *, session=None, **kwargs: Any) -> Any:
    """Resolve a fit source through the active cache.

    On a hit the stored value is returned. Otherwise the source's own method runs (`resolve`, or
    `resolve_gate` on a gate source, with `kwargs` such as `layout`), its result is stored, and
    the stored value is returned. Without an active cache, or for a source that declares
    `ModelAccess.FACTS` or has no fit identity, the source's method runs directly.

    Args:
        source: The fit source.
        model: The loaded model, or None.
        tokenizer: The tokenizer.
        session: The steering session, or None.
        **kwargs: Further arguments of the source's method.

    Returns:
        The resolved value.
    """
    method = getattr(source, "resolve_gate", None) or source.resolve
    cache = active_fit_cache()
    if cache is None or getattr(source, "access", ModelAccess.MODULE) == ModelAccess.FACTS:
        return method(model, tokenizer, session=session, **kwargs)
    digest = _cached_digest(source)
    if digest is None:
        return method(model, tokenizer, session=session, **kwargs)
    if cache.has(digest):
        return cache.get(digest)
    started = time.monotonic()
    value = method(model, tokenizer, session=session, **kwargs)
    return cache.put(digest, value, time.monotonic() - started)


def _product_digest(*, control: BaseControl) -> str | None:
    """The digest of a control's product under its method key, or None when its fit identity is None."""
    from steerability.algorithms.core.registry import method_key_for

    identity = control.fit_identity()
    return None if identity is None else fit_digest(kind=method_key_for(control_cls=type(control)), identity=identity)


def product_is_stored(*, control: BaseControl) -> bool:
    """Whether the active fit cache contains the product of a control that trains or merges.

    Args:
        control: The control.

    Returns:
        False without an active cache or for a control whose `fit_identity()` is None.
    """
    cache = active_fit_cache()
    if cache is None:
        return False
    digest = _product_digest(control=control)
    return digest is not None and cache.has(digest=digest)


def stored_product(*, control: BaseControl) -> Any | None:
    """The stored state of a control's product under the active fit cache, recorded as used.

    Args:
        control: The control.

    Returns:
        A freshly decoded copy of the stored state, whose artifacts are paths into the store's
        artifacts, or None when the active cache contains no product of the control (`product_is_stored`).
    """
    if not product_is_stored(control=control):
        return None
    return active_fit_cache().get(digest=_product_digest(control=control))


def resolve_product(*, control: BaseControl, fit: Callable[[], Any]) -> Any:
    """Resolve the product of a control that trains or merges through the active fit cache.

    With an active cache and a control whose `fit_identity()` is not None, a stored product is
    returned without running `fit` (`FitCache.get`). Otherwise `fit` runs, and the state it
    returns is stored with the measured duration (`FitCache.put`, which copies an adapter or
    checkpoint directory into the store's artifacts). The stored value is returned, and the
    control must use it. The cache counts the products it fitted (`FitCache.product_count`).
    Without an active cache, or for a control without a fit identity, `fit` runs and its state
    is returned.

    Args:
        control: The control.
        fit: A function that trains or merges and returns the control's state (`export_state()`).

    Returns:
        The stored state (its artifacts decoded to paths into the store's artifacts), or the
        state `fit` returned when nothing is stored.
    """
    cache = active_fit_cache()
    digest = _product_digest(control=control) if cache is not None else None
    if digest is None:
        return fit()
    if cache.has(digest=digest):
        return cache.get(digest=digest)
    started = time.monotonic()
    state = fit()
    stored = cache.put(digest=digest, value=state, seconds=time.monotonic() - started)
    cache.product_count += 1
    return stored


def stores_a_fit(controls: Sequence[BaseControl], pending: Sequence[BaseControl], cache: FitCache) -> bool:
    """Whether steering a recipe's controls stores a fit that a later steer of the recipe reuses.

    A pending control with a fit identity stores a fit. So does an enabled structural control
    whose product (a fit digest of `control_fit_digests`) the cache does not contain, and an
    intervention fit source with a digest and no stored value. An artifact loader declares no
    fit and stores none, and it does not keep the other controls' fits from counting.

    Args:
        controls: The controls after `apply_control_hits`.
        pending: The controls `apply_control_hits` left to fit.
        cache: The recipe's fit cache.

    Returns:
        Whether a fit would be stored.
    """
    from steerability.algorithms.state_control.base import InterventionControl
    from steerability.algorithms.structural_control.base import StructuralControl

    enabled = [control for control in controls if getattr(control, "enabled", True)]
    if any(control.fit_identity() is not None for control in pending):
        return True
    for control in enabled:
        if isinstance(control, StructuralControl):
            if any(not cache.has(digest=digest) for digest in control_fit_digests(control=control)):
                return True
            continue
        if not isinstance(control, InterventionControl):
            continue
        for source in control.declared_fit_sources():
            if getattr(source, "access", ModelAccess.MODULE) == ModelAccess.FACTS:
                continue
            digest = _cached_digest(source)
            if digest is not None and not cache.has(digest):
                return True
    return False


def _hook_route(control: BaseControl) -> bool:
    """Whether a control's fits go through the hook route (not intervention, not structural)."""
    from steerability.algorithms.state_control.base import InterventionControl
    from steerability.algorithms.structural_control.base import StructuralControl

    if isinstance(control, (InterventionControl, StructuralControl)) or not getattr(control, "enabled", True):
        return False
    return bool(control.steer_fits())


def apply_control_hits(controls: list[BaseControl], cache: FitCache) -> tuple[list[BaseControl], list[BaseControl]]:
    """Replace each control whose fit is stored by its frozen form.

    `control.frozen_form(state)` is called on the current control with the stored state, so the
    application arguments come from the current recipe and only the fitted state from the cache.
    The frozen arguments are encoded in save mode and decoded before the control is constructed,
    which turns path markers into store paths. A frozen form that is not a single entry of the
    control's own method counts as a miss.

    Args:
        controls: The recipe's controls.
        cache: The fit cache.

    Returns:
        The new control list and the controls still to fit.
    """
    from steerability.algorithms.core.base_control import NotFreezableError
    from steerability.algorithms.core.registry import method_key_for
    from steerability.spipe.codec import DecodeContext, EncodeContext, decode, encode

    result: list[BaseControl] = []
    pending: list[BaseControl] = []
    for control in controls:
        if not _hook_route(control):
            result.append(control)
            continue
        identity = control.fit_identity()
        if identity is None:
            result.append(control)
            pending.append(control)
            continue
        digest = fit_digest(method_key_for(type(control)), identity)
        if not cache.has(digest):
            result.append(control)
            pending.append(control)
            continue
        state = cache.get(digest)
        try:
            form = control.frozen_form(state or {})
        except (NotFreezableError, KeyError):
            form = None
        own_key = method_key_for(type(control))
        if not isinstance(form, tuple) or form[0] != own_key:
            result.append(control)
            pending.append(control)
            continue
        encoded = encode(dict(form[1]), EncodeContext(store=cache.artifacts))
        kwargs = decode(encoded, DecodeContext(store=cache.artifacts, allow_code=True))
        frozen = type(control)(**kwargs)
        frozen.enabled = control.enabled
        result.append(frozen)
    return result, pending


def record_control_fits(pending: list[BaseControl], cache: FitCache) -> None:
    """After steering, store the exported state of each control that fitted.

    A control whose fit identity is None is recorded with `cache.mark_local`.

    Args:
        pending: The controls `apply_control_hits` left to fit.
        cache: The fit cache.
    """
    from steerability.algorithms.core.registry import method_key_for

    for control in pending:
        identity = control.fit_identity()
        if identity is None:
            cache.mark_local(method_key_for(type(control)))
            continue
        cache.put_own(fit_digest(method_key_for(type(control)), identity), control.export_state(), 0.0)


def resolved_id(recipe: Recipe, used: Mapping[str, str]) -> str:
    """The id of a recipe resolved with the fits it used.

    Returns:
        `content_id({"schema": "resolved/1", "recipe": recipe.id, "fits": sorted(used.items())})`.
    """
    fits = [list(item) for item in sorted(used.items())]
    return content_id({"schema": "resolved/1", "recipe": recipe.id, "fits": fits})


__all__ = [
    "FitCache",
    "active_fit_cache",
    "apply_control_hits",
    "control_fit_digests",
    "fit_cache_scope",
    "fit_digest",
    "fit_digests",
    "fit_key",
    "product_is_stored",
    "record_control_fits",
    "resolve_product",
    "resolve_source",
    "resolved_id",
    "source_access",
    "source_digest",
    "source_identity",
    "stored_product",
    "stores_a_fit",
    "weights_context",
]
