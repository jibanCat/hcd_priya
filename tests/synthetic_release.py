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
def _cddf_tables(n_sims=(1, 2, 2), n_bins=30):
    """The synthetic CDDF table: 30 bins like the model's CDDF head (centres 17.1, 17.3, ..., 22.9), and one of
    each case a real table has. Bin 0: no pair at any redshift (the catalogue threshold). Bin 28: no pair at the
    middle redshift only. Bin 29: fewer pairs than the redshift has. Every other bin: all the pairs.
    """
    ratio = [[2.0] * n_bins, [4.0] * n_bins, [4.0] * n_bins]
    pairs = [[n] * n_bins for n in n_sims]
    for i in range(3):
        ratio[i][0], pairs[i][0] = 1.0, 0
        ratio[i][29], pairs[i][29] = 0.25, 1
    ratio[0][28] = ratio[2][28] = 0.5
    ratio[1][28], pairs[1][28] = 1.0, 0
    return [round(17.1 + 0.2 * i, 1) for i in range(n_bins)], ratio, pairs


_LOG_NHI, _CDDF_RATIO, _CDDF_PAIRS = _cddf_tables()
INCIDENCE_TABLES = {
    "z": [2.0, 3.0, 5.0],
    "n_sims": [1, 2, 2],
    "dndx_ratio": [[2.0, 3.0, 0.5], [4.0, 3.0, 0.5], [4.0, 5.0, 0.25]],
    "log_nhi_centres": _LOG_NHI,
    "cddf_ratio": _CDDF_RATIO,
    "cddf_n_pairs": _CDDF_PAIRS,
}


def heldout_error(n_bins=30):
    """A synthetic held-out error block for the synthetic record: two of its three redshifts (none at 2.0), one
    class without an LF+MF estimate at z 5.0, and no estimate in the first CDDF bin. Every number is made up.
    """
    cddf = [[None] + [v] * (n_bins - 1) for v in (0.3, 0.31)]
    cddf_n = [[0] + [6] * (n_bins - 1) for _ in range(2)]
    cddf_mf = [[None] + [v] * (n_bins - 1) for v in (0.2, 0.21)]
    cddf_mf_n = [[0] + [5] * (n_bins - 1) for _ in range(2)]
    return {
        "kind": "rms_log_error_heldout",
        "definition": "A synthetic definition of the synthetic numbers.",
        "lf": "Synthetic: how the uncorrected value was held out.",
        "lfmf": "Synthetic: how the corrected value was held out.",
        "what_it_does_not_cover": "Synthetic: everything; the numbers are made up.",
        "n_hr_simulations": 2,
        "z": [3.0, 5.0],
        "dndx": {
            "lf": [[0.11, 0.12, 0.13], [0.21, 0.22, 0.23]],
            "lf_n": [[6, 6, 6], [6, 6, 6]],
            "lfmf": [[0.05, 0.06, 0.07], [0.15, None, 0.17]],
            "lfmf_n": [[5, 5, 5], [5, 0, 5]],
        },
        "cddf": {"lf": cddf, "lf_n": cddf_n, "lfmf": cddf_mf, "lfmf_n": cddf_mf_n},
        "source": {"file": "synthetic_heldout.npz", "sha256": "ef" * 32, "how": "made up"},
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


def make_release(
    root, *, version="synthetic-release-1", n_members=2, incidence=False, heldout=False
):
    """Write the release under ``root`` (created). Returns {"root": Path, "sha": the sha256 of release.json,
    "bundle": the backbone's SyntheticBundle, "mf_sha": the MF record's sha256}. With ``incidence`` the release
    also names a synthetic incidence MF record, the optional part ("incidence_sha": its sha256)."""
    from tests.test_predictor_mf import write_mf

    root = Path(root)
    root.mkdir(parents=True)
    b = RT.write_synthetic_bundle(str(root / "backbone"), n_members=n_members)
    cache = b.lineage["cache"]["sha256"]
    s = write_mf(root / "mf", manifest=b.manifest_sha256, cache=cache)
    out = {"root": root, "bundle": b, "mf_sha": s}
    extra = {}
    if incidence:
        more = {"heldout_error": heldout_error()} if heldout else {}  # the optional held-out error
        out["incidence_sha"] = write_incidence(root / "incidence", cache=cache, **more)
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
