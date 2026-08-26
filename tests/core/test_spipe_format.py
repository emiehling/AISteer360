"""Manifest schema validation and archive packing for `spipe/1`."""
import json
import zipfile

import pytest

from aisteer360.spipe.errors import SpipeFormatError
from aisteer360.spipe.format import pack_zip, unpack_zip, validate_manifest


def minimal_manifest(**overrides):
    manifest = {
        "format": "spipe/1",
        "created_at": "2026-08-25T00:00:00Z",
        "toolkit_version": "0.5.0",
        "code_dependent": False,
        "model": {"ref": "org/model", "revision": None},
        "controls": [],
        "lock": None,
    }
    manifest.update(overrides)
    return manifest


def test_minimal_manifest_validates():
    validate_manifest(minimal_manifest())


def test_version_refusal_names_versions():
    with pytest.raises(SpipeFormatError, match=r"'spipe/2'.*'spipe/1'"):
        validate_manifest(minimal_manifest(format="spipe/2"))


def test_unknown_top_level_key_rejected():
    with pytest.raises(SpipeFormatError, match="unknown key"):
        validate_manifest(minimal_manifest(extra=1))


def test_unknown_entry_key_rejected():
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": None, "extra": 1}
    with pytest.raises(SpipeFormatError, match=r"controls\[0\].*unknown key"):
        validate_manifest(minimal_manifest(controls=[entry]))


def test_resolved_object_and_array_forms():
    resolved = {"method": "state_control/caa", "args": {}, "artifacts": {}, "origin": None}
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": resolved}
    validate_manifest(minimal_manifest(controls=[entry]))
    entry_list = dict(entry, resolved=[resolved, dict(resolved)])
    validate_manifest(minimal_manifest(controls=[entry_list]))
    with pytest.raises(SpipeFormatError, match="non-empty"):
        validate_manifest(minimal_manifest(controls=[dict(entry, resolved=[])]))


def test_artifact_record_validation():
    record = {"id": "notahash", "encoding": "tensors", "type": "SteeringVector",
              "artifact_class": "direction", "source": None, "fit_digest": None, "provenance": {}}
    resolved = {"method": "state_control/caa", "args": {}, "artifacts": {"v": record}, "origin": None}
    entry = {"method": "state_control/caa", "enabled": True, "args": {}, "resolved": resolved}
    with pytest.raises(SpipeFormatError, match="sha256"):
        validate_manifest(minimal_manifest(controls=[entry]))


def test_zip_determinism(tmp_path):
    src = tmp_path / "bundle"
    (src / "artifacts").mkdir(parents=True)
    (src / "spipe.json").write_text(json.dumps(minimal_manifest()))
    (src / "artifacts" / "blob").write_bytes(b"payload")
    pack_zip(src, tmp_path / "a.spipe")
    pack_zip(src, tmp_path / "b.spipe")
    assert (tmp_path / "a.spipe").read_bytes() == (tmp_path / "b.spipe").read_bytes()


def test_zip_slip_rejected(tmp_path):
    evil = tmp_path / "evil.spipe"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../outside.txt", "boom")
    with pytest.raises(SpipeFormatError, match="escapes"):
        unpack_zip(evil, tmp_path / "dest")


def test_zip_symlink_member_rejected(tmp_path):
    evil = tmp_path / "evil.spipe"
    with zipfile.ZipFile(evil, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(SpipeFormatError, match="symlink"):
        unpack_zip(evil, tmp_path / "dest")


def test_not_a_zip_rejected(tmp_path):
    bogus = tmp_path / "bogus.spipe"
    bogus.write_text("not a zip")
    with pytest.raises(SpipeFormatError, match="not a zip"):
        unpack_zip(bogus, tmp_path / "dest")
