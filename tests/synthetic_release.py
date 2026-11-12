"""A synthetic LF+MF release directory for tests: a synthetic backbone bundle (untrained networks), a synthetic MF
record bound to it, and release.json. Also a stand-in for the installed model package. No trained model, no real
number."""

import importlib
import json
import shutil
import sys
from pathlib import Path

from hcd_analysis.emulator import release_manifest as RM
from hcd_analysis.emulator import release_testing as RT


def make_release(root, *, version="synthetic-release-1", n_members=2):
    """Write the release under ``root`` (created). Returns {"root": Path, "sha": the sha256 of release.json,
    "bundle": the backbone's SyntheticBundle, "mf_sha": the MF record's sha256}."""
    from tests.test_predictor_mf import write_mf

    root = Path(root)
    root.mkdir(parents=True)
    b = RT.write_synthetic_bundle(str(root / "backbone"), n_members=n_members)
    s = write_mf(root / "mf", manifest=b.manifest_sha256, cache=b.lineage["cache"]["sha256"])
    _, rs = RM.write_release_manifest(
        str(root),
        release_version=version,
        backbone_dir="backbone",
        backbone_manifest_sha256=b.manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=s,
    )
    return {"root": root, "sha": rs, "bundle": b, "mf_sha": s}


def model_package(
    site, release, *, version="0+synthetic", synthetic=True, name="lfmf", anchor=None
):
    """Write a model package ``hcd_priya_models`` under ``site`` (a directory to put on the import path), holding
    ``release`` under ``name``: the code of the real package (copied from the one that is installed) with its own
    ``_version.py`` and ``data/``. Returns the package directory."""
    import hcd_priya_models as installed

    pkg = Path(site) / "hcd_priya_models"
    (pkg / "data").mkdir(parents=True)
    shutil.copyfile(installed.__file__, pkg / "__init__.py")
    (pkg / "_version.py").write_text(f'__version__ = "{version}"\n')
    shutil.copytree(release["root"], pkg / "data" / name)
    entry = {
        "name": name,
        "format": "hcd_priya.release/1",
        "path": name,
        "anchor_sha256": release["sha"] if anchor is None else anchor,
    }
    doc = {
        "format": installed.FORMAT,
        "version": version,
        "synthetic": synthetic,
        "models": [entry],
    }
    (pkg / "data" / "models.json").write_text(json.dumps(doc))
    return pkg


def use_model_package(monkeypatch, site):
    """Make ``import hcd_priya_models`` find the package under ``site`` for the rest of the test. Call it after
    ``model_package`` has written the package there."""
    assert (Path(site) / "hcd_priya_models" / "__init__.py").is_file(), "write the package first"
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "hcd_priya_models", raising=False)
    monkeypatch.delitem(sys.modules, "hcd_priya_models._version", raising=False)
    importlib.invalidate_caches()
