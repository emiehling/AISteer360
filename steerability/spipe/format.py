"""The `spipe/2` manifest schema and archive packing.

The canonical form of a `.spipe` is a directory containing `spipe.json` and an optional
`artifacts/` store; a `.spipe` file is a zip of that directory. Zips are written
deterministically (sorted member names, stored uncompressed, fixed timestamps) under a size
cap, and unpacked with a zip-slip guard, symlink rejection, and the same cap on the
decompressed size.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping

from steerability.spipe.errors import SpipeFormatError
from steerability.spipe.store import is_artifact_id

logger = logging.getLogger(__name__)

FORMAT_VERSION = "spipe/2"
MANIFEST_NAME = "spipe.json"
ARTIFACTS_DIR = "artifacts"

SIZE_CAP_BYTES = 20 * 1024**3

_TOP_LEVEL_KEYS = {
    "format", "created_at", "toolkit_version", "code_dependent", "model", "recipe_id", "controls", "lock",
}
_MODEL_KEYS = {"ref", "revision"}
_ENTRY_KEYS = {"method", "args", "resolved"}
_RESOLVED_KEYS = {"method", "args", "artifacts", "origin"}
_ARTIFACT_RECORD_KEYS = {"id", "encoding", "type", "artifact_class", "source", "fit_digest", "provenance"}
_LOCK_KEYS = {"protocol_id", "protocol", "facts", "certificate"}
CERTIFICATE_SCHEMA = "certificate/2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpipeFormatError(message)


def _check_keys(mapping: Mapping, allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    _require(not unknown, f"{where}: unknown key(s) {unknown}; the {FORMAT_VERSION} schema rejects them.")


def _validate_resolved_entry(resolved: Mapping, where: str) -> None:
    _require(isinstance(resolved, Mapping), f"{where}: resolved entry must be an object.")
    _check_keys(resolved, _RESOLVED_KEYS, where)
    _require(isinstance(resolved.get("method"), str) and resolved["method"],
             f"{where}.method: required non-empty string.")
    _require(isinstance(resolved.get("args"), Mapping), f"{where}.args: required object.")
    artifacts = resolved.get("artifacts", {})
    _require(isinstance(artifacts, Mapping), f"{where}.artifacts: must be an object.")
    for name, record in artifacts.items():
        record_where = f"{where}.artifacts[{name!r}]"
        _require(isinstance(record, Mapping), f"{record_where}: must be an object.")
        _check_keys(record, _ARTIFACT_RECORD_KEYS, record_where)
        _require(is_artifact_id(record.get("id")),
                 f"{record_where}.id: required 'sha256:<hex>' string with 64 lowercase hex digits.")
        _require(record.get("encoding") in ("tensors", "tree"),
                 f"{record_where}.encoding: must be 'tensors' or 'tree'.")
        _require(record.get("artifact_class") in ("direction", "calibrated", "opaque"),
                 f"{record_where}.artifact_class: must be 'direction', 'calibrated', or 'opaque'.")
        _require(isinstance(record.get("type"), str) and record["type"],
                 f"{record_where}.type: required non-empty string.")
        for key in ("source", "fit_digest"):
            _require(record.get(key) is None or isinstance(record[key], str),
                     f"{record_where}.{key}: must be null or a string.")
        _require(record.get("provenance") is None or isinstance(record["provenance"], Mapping),
                 f"{record_where}.provenance: must be null or an object.")
    origin = resolved.get("origin")
    if origin is not None:
        _require(isinstance(origin, Mapping) and set(origin) <= {"method", "args"},
                 f"{where}.origin: must be null or an object with 'method' and 'args'.")


def validate_manifest(manifest: Any) -> None:
    """Validate a parsed `spipe.json` against the `spipe/2` schema.

    A manifest records the model reference, the recipe id, one entry per enabled control (its
    recipe-entry JSON, `method` and `args`, plus a `resolved` list of frozen items or null), and a
    lock. The lock is null for a bundle that is not frozen. A frozen bundle's lock records the
    protocol it was fitted or measured under (`protocol_id`, `protocol`), the facts observed on its
    model (`facts`), and an optional `certificate`, a mapping whose `schema` is `"certificate/2"`.

    Args:
        manifest: The parsed manifest.

    Raises:
        SpipeFormatError: On any schema violation; the message names the offending field.
            An unsupported `format` value is reported first, naming the found and supported
            versions.
    """
    _require(isinstance(manifest, Mapping), "spipe.json: manifest must be a JSON object.")
    found = manifest.get("format")
    _require(
        found == FORMAT_VERSION,
        f"Unsupported spipe format {found!r}; this toolkit supports {FORMAT_VERSION!r}.",
    )
    _check_keys(manifest, _TOP_LEVEL_KEYS, "spipe.json")
    for key in ("created_at", "toolkit_version"):
        _require(isinstance(manifest.get(key), str), f"spipe.json.{key}: required string.")
    _require(isinstance(manifest.get("code_dependent"), bool), "spipe.json.code_dependent: required boolean.")

    _require(isinstance(manifest.get("recipe_id"), str) and manifest["recipe_id"],
             "spipe.json.recipe_id: required non-empty string.")

    model = manifest.get("model")
    _require(isinstance(model, Mapping), "spipe.json.model: required object.")
    _check_keys(model, _MODEL_KEYS, "spipe.json.model")
    _require(isinstance(model.get("ref"), str) and model["ref"], "spipe.json.model.ref: required non-empty string.")
    _require(model.get("revision") is None or isinstance(model["revision"], str),
             "spipe.json.model.revision: must be a string or null.")

    controls = manifest.get("controls")
    _require(isinstance(controls, list), "spipe.json.controls: required array.")
    for i, entry in enumerate(controls):
        where = f"spipe.json.controls[{i}]"
        _require(isinstance(entry, Mapping), f"{where}: must be an object.")
        _check_keys(entry, _ENTRY_KEYS, where)
        _require(isinstance(entry.get("method"), str) and "/" in entry.get("method", ""),
                 f"{where}.method: required '<category>_control/<name>' string.")
        _require(isinstance(entry.get("args"), Mapping), f"{where}.args: required object.")
        resolved = entry.get("resolved")
        if resolved is None:
            continue
        _require(isinstance(resolved, list) and bool(resolved),
                 f"{where}.resolved: must be null or a non-empty array of frozen entries.")
        for j, item in enumerate(resolved):
            _validate_resolved_entry(item, f"{where}.resolved[{j}]")

    lock = manifest.get("lock")
    if lock is not None:
        _require(isinstance(lock, Mapping), "spipe.json.lock: must be an object or null.")
        _check_keys(lock, _LOCK_KEYS, "spipe.json.lock")
        _require(isinstance(lock.get("protocol_id"), str) and lock["protocol_id"],
                 "spipe.json.lock.protocol_id: required string.")
        protocol = lock.get("protocol")
        _require(isinstance(protocol, Mapping) and protocol.get("schema") == "protocol/1",
                 "spipe.json.lock.protocol: required 'protocol/1' identity object.")
        _require(isinstance(lock.get("facts"), Mapping), "spipe.json.lock.facts: required object.")
        certificate = lock.get("certificate")
        _require(certificate is None or (isinstance(certificate, Mapping)
                                         and certificate.get("schema") == CERTIFICATE_SCHEMA),
                 f"spipe.json.lock.certificate: must be null or an object with schema {CERTIFICATE_SCHEMA!r}.")


def write_manifest(manifest: Mapping, directory: str | Path) -> None:
    """Write `spipe.json` into `directory` (sorted keys, two-space indent, UTF-8)."""
    path = Path(directory) / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")


def read_manifest(directory: str | Path) -> dict:
    """Read and validate `spipe.json` from `directory`.

    Raises:
        SpipeFormatError: If the manifest is absent, is not valid JSON, or violates the
            schema.
    """
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise SpipeFormatError(f"{directory} contains no {MANIFEST_NAME}; not a spipe bundle.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpipeFormatError(f"{path} is not valid JSON: {exc}.") from exc
    validate_manifest(manifest)
    return manifest


def pack_zip(directory: str | Path, target: str | Path) -> None:
    """Write `directory` as a deterministic zip at `target`.

    Members are written in sorted arcname order, stored uncompressed, with a fixed
    `(1980, 1, 1, 0, 0, 0)` timestamp and no extra fields. Equal directories therefore
    produce byte-equal files. Each member is copied in chunks, so packing holds no whole file
    in memory. The size check runs before anything is written.

    Raises:
        SpipeFormatError: If `directory` contains a symlink, or its files total more than the
            20 GB cap that unpacking enforces.
    """
    directory = Path(directory)
    members = sorted(
        (path for path in directory.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in members:
        if path.is_symlink():
            raise SpipeFormatError(f"Refusing to pack symlink {path} into a spipe archive.")
    total = sum(path.stat().st_size for path in members)
    if total > SIZE_CAP_BYTES:
        raise SpipeFormatError(
            f"{directory} holds {total} bytes, over the {SIZE_CAP_BYTES}-byte cap of a spipe archive; "
            "save the bundle as a directory, or with artifacts='thin'."
        )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            arcname = path.relative_to(directory).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.file_size = path.stat().st_size
            with open(path, "rb") as source, archive.open(info, "w") as sink:
                shutil.copyfileobj(source, sink, 1 << 20)


def unpack_zip(archive_path: str | Path, dest: str | Path) -> None:
    """Extract a `.spipe` zip into `dest` with the format's safety guards.

    Every member name is normalized and rejected if absolute, containing `..`, or resolving
    outside `dest`; symlink members are rejected; the total decompressed size is capped at
    20 GB.

    Raises:
        SpipeFormatError: If the archive is not a zip, a member violates the guards, or the
            size cap is exceeded.
    """
    dest = Path(dest).resolve()
    if not zipfile.is_zipfile(archive_path):
        raise SpipeFormatError(f"{archive_path} is not a zip archive.")
    with zipfile.ZipFile(archive_path) as archive:
        total = sum(info.file_size for info in archive.infolist())
        if total > SIZE_CAP_BYTES:
            raise SpipeFormatError(
                f"Archive decompresses to {total} bytes, over the {SIZE_CAP_BYTES}-byte cap."
            )
        for info in archive.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or (len(name) > 1 and name[1] == ":"):
                raise SpipeFormatError(f"Archive member {name!r} escapes the extraction root.")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SpipeFormatError(f"Archive member {name!r} is a symlink; symlinks are rejected.")
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise SpipeFormatError(f"Archive member {name!r} escapes the extraction root.")
        for info in archive.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
