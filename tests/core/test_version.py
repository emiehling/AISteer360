"""The package version and the toolkit version check in `SPipe.verify()`."""
import tomllib
from pathlib import Path

import pytest

import steerability
from steerability.spipe import SPipe

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def recipe_manifest(toolkit_version: str) -> dict:
    return {
        "format": "spipe/1",
        "created_at": "2026-08-25T00:00:00Z",
        "toolkit_version": toolkit_version,
        "code_dependent": False,
        "model": {"ref": "org/model", "revision": None},
        "controls": [],
        "lock": None,
    }


def version_warnings(report) -> list[str]:
    return [message for message in report.warnings if "written by steerability" in message]


def test_version_matches_pyproject():
    with open(PYPROJECT, "rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]
    assert steerability.__version__ == expected


def test_manifest_records_the_package_version():
    from steerability.spipe.freeze import _package_versions, _toolkit_version

    assert _toolkit_version() == steerability.__version__
    assert _package_versions()["steerability"] == steerability.__version__


def test_verify_skips_an_unknown_saved_version():
    spipe = SPipe(recipe_manifest("unknown"), store=None, base_dir=None, allow_code=False)
    report = spipe.verify()
    assert report.ok
    assert not version_warnings(report)


def test_verify_skips_an_unknown_current_version(monkeypatch):
    monkeypatch.setattr("steerability.spipe.freeze._toolkit_version", lambda: "unknown")
    report = SPipe(recipe_manifest("7.0.0"), store=None, base_dir=None, allow_code=False).verify()
    assert not version_warnings(report)


@pytest.mark.parametrize("saved, warned", [("0.5.0", False), ("7.0.0", True)])
def test_verify_compares_major_versions(monkeypatch, saved, warned):
    monkeypatch.setattr("steerability.spipe.freeze._toolkit_version", lambda: "0.5.3")
    report = SPipe(recipe_manifest(saved), store=None, base_dir=None, allow_code=False).verify()
    assert bool(version_warnings(report)) is warned
