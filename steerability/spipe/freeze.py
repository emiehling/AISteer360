"""Building an `SPipe` from a `SteeringPipeline` (`to_spipe` orchestration).

The manifest's entries are the recipe-entry JSON of the pipeline's enabled controls
(`Recipe.from_controls(...).to_json(store)`), and its `recipe_id` is that recipe's id. Freezing
additionally walks each enabled control's `export_state()` and `frozen_form()`, writes the exported
artifacts into the bundle's store with provenance and the control's fit digest, adds each entry's
`resolved` list, and records the protocol in the lock. The freeze walk runs before the recipe is
encoded, so a recipe value with the same content as an exported artifact reuses the artifact's
record.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from steerability.algorithms.core.base_control import NotFreezableError
from steerability.spipe.codec import EncodeContext, encode, encoded_size
from steerability.spipe.errors import SpipeSaveError
from steerability.spipe.format import ARTIFACTS_DIR, FORMAT_VERSION
from steerability.spipe.store import ArtifactStore

if TYPE_CHECKING:
    from steerability.algorithms.core.protocol import Protocol
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.spipe.spipe import SPipe

logger = logging.getLogger(__name__)

INLINE_LIMIT_BYTES = 1_000_000


def _toolkit_version() -> str:
    """Return `steerability.__version__`, which is `"unknown"` when the distribution is not installed."""
    import steerability

    return steerability.__version__


def _resolve_model_ref(pipeline, model_ref: str | None) -> str:
    if model_ref is not None:
        return str(model_ref)
    if pipeline.model_name_or_path is not None:
        return str(pipeline.model_name_or_path)
    model = pipeline.model
    if model is not None:
        ref = getattr(model, "name_or_path", None) or getattr(
            getattr(model, "config", None), "_name_or_path", None
        )
        if ref:
            return str(ref)
    raise SpipeSaveError(
        "The pipeline has no resolvable model reference; pass model_ref= to to_spipe()."
    )


def _artifact_class_of(value: Any) -> str:
    import torch

    from steerability.algorithms.core.internals.probes.probe import Probe
    from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    if isinstance(value, (Gate, Probe, ProbeSet)):
        return "calibrated"
    if isinstance(value, (SteeringVector, torch.Tensor)):
        return "direction"
    return "opaque"


def _is_artifact_value(value: Any) -> bool:
    """Whether an exported state value is a typed artifact, which the manifest records.

    Other state values (resolved layers, head maps, bound transforms) describe the structure
    `frozen_form` rebuilds the control from. Their encoded form may reference artifacts, which
    are recorded under the typed values that export them.
    """
    import torch

    from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
    from steerability.algorithms.core.internals.probes.probe import Probe
    from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    artifact_types = (
        torch.Tensor, SteeringVector, Probe, ProbeSet, TextMemory, PoolMemory, Gate, LoRAArtifact,
        CheckpointArtifact, Path,
    )
    return isinstance(value, artifact_types) or type(value).__name__ == "CPOMemory"


def _collect_artifact_ids(encoded: Any, found: list[str]) -> None:
    if isinstance(encoded, dict):
        for key, value in encoded.items():
            if key == "$artifact" and isinstance(value, str):
                found.append(value)
            else:
                _collect_artifact_ids(value, found)
    elif isinstance(encoded, list):
        for item in encoded:
            _collect_artifact_ids(item, found)


def _freeze_control(control, ctx: EncodeContext, entry_path: str) -> list[dict] | None:
    """Freeze one enabled control: encode its exported state and frozen form.

    Each exported artifact records the control's fit digest, the content id of the digests of
    the fits the control declares (`control_fit_digests`), in order.

    Returns:
        The `resolved` manifest value (a list of frozen items), or None for a control with
        nothing to pin.
    """
    from steerability.algorithms.core.fits import control_fit_digests
    from steerability.algorithms.core.identity import content_id
    from steerability.algorithms.core.registry import method_key_for

    state = control.export_state()
    fits = list(control.steer_fits())
    if not state and not fits:
        return None

    digests = control_fit_digests(control)
    fit_digest = content_id(list(digests)) if digests else None

    # encode each exported artifact with its metadata installed, giving artifact sidecars and
    # manifest records the fit provenance; content-equal values re-encoded later inside the
    # frozen args reuse these records (first write wins). The source names the declared fit of
    # the artifact's class, else the control's first fit
    artifacts: dict[str, dict] = {}
    remaining_fits = list(fits)
    for name, value in state.items():
        if not _is_artifact_value(value):
            continue
        artifact_class = _artifact_class_of(value)
        source = None
        if fit_digest is not None:
            matched = next((fit for fit in remaining_fits if fit[1] == artifact_class), None)
            if matched is not None:
                remaining_fits.remove(matched)
            source = (matched or (fits[0] if fits else (type(control).__name__,)))[0]
        ctx.artifact_fields = {
            "artifact_class": artifact_class,
            "source": source,
            "fit_digest": fit_digest,
        }
        encoded_value = encode(value, ctx, f"{entry_path}.state[{name!r}]")
        ctx.artifact_fields = {}
        ids: list[str] = []
        _collect_artifact_ids(encoded_value, ids)
        if ids:
            artifacts[name] = ctx.records[ids[0]].manifest_entry()

    forms = control.frozen_form(state)
    if isinstance(forms, tuple):
        forms = [forms]

    method_key = method_key_for(type(control))
    resolved_entries = []
    for form_index, (frozen_key, frozen_args) in enumerate(forms):
        encoded_args = {
            name: encode(value, ctx, f"{entry_path}.resolved[{form_index}].{name}")
            for name, value in frozen_args.items()
        }
        origin = None
        if frozen_key != method_key:
            origin = {"method": method_key, "args": None}  # args filled in by the caller
        resolved_entries.append({
            "method": frozen_key,
            "args": encoded_args,
            "artifacts": artifacts if form_index == 0 else {},
            "origin": origin,
        })

    return resolved_entries


def _recipe_frozen_blockers(controls: list) -> list[str]:
    """Reasons the enabled controls cannot be frozen from their recipe alone, in pipeline order.

    A control is recipe-frozen when its steer step declares `ModelAccess.FACTS`, declares no
    fits, and exports no state, which makes its recipe its frozen form. `export_state()` is
    called only for a control that passes the access and fit checks, since it is documented as
    post-`steer()`.

    Args:
        controls: The pipeline's controls in manifest order (structural, input, state, output).

    Returns:
        One `"controls[<i>] (<ClassName>): <reasons>"` string per blocking control.
    """
    from steerability.algorithms.core.execution.access import ModelAccess

    blockers = []
    for i, control in enumerate(controls):
        if not control.enabled:
            continue
        reasons = []
        access = control.steer_access()
        if access != ModelAccess.FACTS:
            reasons.append(f"declares steer access {access.name}")
        fits = control.steer_fits()
        if fits:
            reasons.append(f"declares fits {', '.join(fit[0] for fit in fits)}")
        if not reasons:
            try:
                state = control.export_state()
            except Exception as exc:
                reasons.append(f"export_state() raised {type(exc).__name__}: {exc}")
            else:
                if state:
                    reasons.append(f"exports state {', '.join(sorted(state))}")
        if reasons:
            blockers.append(f"controls[{i}] ({type(control).__name__}): {', '.join(reasons)}")
    return blockers


def _has_code_refs(encoded: Any) -> bool:
    if isinstance(encoded, dict):
        return "$ref" in encoded or any(_has_code_refs(value) for value in encoded.values())
    if isinstance(encoded, list):
        return any(_has_code_refs(item) for item in encoded)
    return False


def build_spipe(
    pipeline: SteeringPipeline,
    *,
    freeze: bool | None = None,
    model_ref: str | None = None,
    protocol: Protocol | None = None,
    certificate: Mapping | None = None,
) -> "SPipe":
    """Build an `SPipe` from a pipeline.

    An unsteered pipeline freezes when every enabled control is recipe-frozen (FACTS access, no
    fits, no exported state); otherwise `freeze=True` raises. Disabled controls are not written.

    Args:
        pipeline: The `SteeringPipeline` to serialize.
        freeze: Freeze the resolution (default: the pipeline's steered state). `False` forces
            a recipe-only spipe from a steered pipeline.
        model_ref: Explicit model reference, overriding the pipeline's.
        protocol: The protocol a frozen bundle records in its lock; defaults to
            `Protocol.from_pipeline(pipeline)`.
        certificate: A mapping with `"schema": "certificate/2"` recorded in a frozen bundle's
            lock, or None.

    Returns:
        The built `SPipe` (backed by a temporary store directory until saved).

    Raises:
        SpipeSaveError: If the model reference is unresolvable, a control is unregistered or
            cannot serialize, freezing is requested on an unsteered pipeline whose controls are
            not all recipe-frozen, a control cannot freeze, or a certificate is given without
            freezing or is malformed.
    """
    from steerability.algorithms.core.protocol import Protocol, ProtocolFacts
    from steerability.algorithms.core.recipe import Recipe, RecipeError
    from steerability.algorithms.core.registry import RegistryError
    from steerability.spipe.format import CERTIFICATE_SCHEMA
    from steerability.spipe.spipe import SPipe

    if freeze is None:
        freeze = bool(pipeline._is_steered)
    if certificate is not None:
        if not freeze:
            raise SpipeSaveError("A certificate is recorded only in a frozen bundle; pass freeze=True.")
        if not isinstance(certificate, Mapping) or certificate.get("schema") != CERTIFICATE_SCHEMA:
            raise SpipeSaveError(f"A certificate must be a mapping with 'schema': {CERTIFICATE_SCHEMA!r}.")

    ref = _resolve_model_ref(pipeline, model_ref)
    revision = pipeline.hf_model_kwargs.get("revision") if pipeline.hf_model_kwargs else None

    controls = [
        control for control in (
            *pipeline.structural_controls, *pipeline.input_controls,
            *pipeline.state_controls, *pipeline.output_controls,
        ) if control.enabled
    ]

    if freeze and not pipeline._is_steered:
        blockers = _recipe_frozen_blockers(controls)
        if blockers:
            raise SpipeSaveError(
                "freeze=True on an unsteered pipeline requires every enabled control to be "
                "recipe-frozen (steer access FACTS, no fits, no exported state); call steer() "
                f"first. Blocking: {'; '.join(blockers)}"
            )

    temp = tempfile.TemporaryDirectory(prefix="spipe-")
    base_dir = Path(temp.name)
    store = ArtifactStore(base_dir / ARTIFACTS_DIR)

    provenance: dict[str, Any] = {}
    spec = pipeline._resolve_backend_spec(pipeline.backend)
    provenance["backend_spec_hash"] = spec.spec_hash
    provenance["model_ref"] = ref
    model = pipeline.model
    default_model_type = None
    if model is not None:
        from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta

        meta = artifact_provenance_meta(model, pipeline.tokenizer)
        provenance["model_fingerprint"] = meta.get("model_fingerprint")
        provenance["tokenizer_fingerprint"] = meta.get("chat_template_fingerprint")
        default_model_type = getattr(model.config, "model_type", None)

    ctx = EncodeContext(store=store, provenance=provenance, default_model_type=default_model_type)

    # the freeze walk runs before the recipe is encoded: exported artifacts land in the store
    # with their fit provenance first, and a content-equal recipe value reuses the record
    resolved_by_index: dict[int, list[dict] | None] = {}
    for i, control in enumerate(controls):
        entry_path = f"controls[{i}]"
        if getattr(control, "gate_driven_externally", False):
            raise SpipeSaveError(
                f"{entry_path}: {type(control).__name__} follows an externally driven shared "
                "gate, an in-memory relationship that does not serialize; save the driving "
                "pipeline without the follower, or gate the control directly."
            )
        if freeze:
            try:
                resolved_by_index[i] = _freeze_control(control, ctx, entry_path)
            except NotFreezableError as exc:
                raise SpipeSaveError(
                    f"{entry_path}: {exc} The pipeline can still be saved with freeze=False."
                ) from exc

    try:
        recipe = Recipe.from_controls(controls)
        entries = recipe.to_json(store)["entries"]
        recipe_id = recipe.id
    except (RecipeError, RegistryError) as exc:
        raise SpipeSaveError(str(exc)) from exc

    for i, entry in enumerate(entries):
        entry_path = f"controls[{i}]"
        resolved = resolved_by_index.get(i)
        entry["resolved"] = resolved
        for item in resolved or []:
            if item.get("origin") is not None:
                item["origin"]["args"] = entry["args"]
        size = encoded_size(entry)
        if size > INLINE_LIMIT_BYTES:
            raise SpipeSaveError(
                f"{entry_path}: the entry's inline JSON is {size} bytes, over the "
                f"{INLINE_LIMIT_BYTES}-byte limit; convert large dataset fields to a $data "
                "reference (DataRef), or save the fitted pipeline frozen and share that."
            )

    lock = None
    if freeze:
        protocol = protocol if protocol is not None else Protocol.from_pipeline(pipeline, model=ref)
        lock = {
            "protocol_id": protocol.id,
            "protocol": protocol.identity(),
            "facts": ProtocolFacts.observe(model, pipeline.tokenizer).to_dict(),
            "certificate": dict(certificate) if certificate is not None else None,
        }

    manifest = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toolkit_version": _toolkit_version(),
        "code_dependent": bool(ctx.code_refs) or _has_code_refs(entries),
        "model": {"ref": ref, "revision": revision},
        "recipe_id": recipe_id,
        "controls": entries,
        "lock": lock,
    }

    return SPipe(manifest, store=store, base_dir=base_dir, allow_code=True, _temp=temp)
