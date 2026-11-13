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

# A plainly synthetic incidence table: round numbers that no simulation gives.
INCIDENCE_TABLES = {
    "z": [2.0, 3.0, 5.0],
    "n_sims": [1, 2, 2],
    "dndx_ratio": [[2.0, 3.0, 0.5], [4.0, 3.0, 0.5], [4.0, 5.0, 0.25]],
    "log_nhi_centres": [17.5, 20.5],
    "cddf_ratio": [[2.0, 0.5], [4.0, 0.5], [4.0, 0.25]],
}


def write_incidence(path, *, cache, tables=None, **over):
    """Write a synthetic incidence MF record into the new directory ``path``, fitted (in name) on the LF cache
    ``cache``. ``over`` replaces arguments of ``incidence_mf.write_record``. Returns the record's sha256.
    """
    from hcd_analysis.emulator import incidence_mf as IM

    kw = dict(
        version="synthetic-incidence-1",
        lf_cache={"file": RT.SYNTHETIC_CACHE_NAME, "sha256": cache},
        hr_incidence={"file": "synthetic_hr_incidence.npz", "sha256": "ab" * 32},
        groups_not_in_product=["synthsim9 snapshot 3"],
        composition={
            "id": "synthetic-composition",
            "excluded_sims": ["synthsim3"],
            "ruling_ids": ["synthetic ruling"],
        },
        train_sims=["synthsim1", "synthsim2"],
        n_pairs=5,
        fit_code={"commit": "c" * 40, "dirty": False},
    )
    kw.update(over)
    return IM.write_record(str(path), INCIDENCE_TABLES if tables is None else tables, **kw)[1]


def make_release(root, *, version="synthetic-release-1", n_members=2, incidence=False):
    """Write the release under ``root`` (created). Returns {"root": Path, "sha": the sha256 of release.json,
    "bundle": the backbone's SyntheticBundle, "mf_sha": the MF record's sha256}. With ``incidence`` the release is of
    format 2 and also holds a synthetic incidence MF record ("incidence_sha": its sha256)."""
    from tests.test_predictor_mf import write_mf

    root = Path(root)
    root.mkdir(parents=True)
    b = RT.write_synthetic_bundle(str(root / "backbone"), n_members=n_members)
    cache = b.lineage["cache"]["sha256"]
    s = write_mf(root / "mf", manifest=b.manifest_sha256, cache=cache)
    out = {"root": root, "bundle": b, "mf_sha": s}
    extra = {}
    if incidence:
        out["incidence_sha"] = write_incidence(root / "incidence", cache=cache)
        extra = dict(incidence_dir="incidence", incidence_record_sha256=out["incidence_sha"])
    _, out["sha"] = RM.write_release_manifest(
        str(root),
        release_version=version,
        backbone_dir="backbone",
        backbone_manifest_sha256=b.manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=s,
        **extra,
    )
    return out


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
        "format": json.loads((Path(release["root"]) / "release.json").read_text())["format"],
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
