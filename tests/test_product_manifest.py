"""Gate E amendment A1 rev 1 section 7: the product manifest pins every emulator-error and forward product by sha256,
names the training cache and the ensemble manifest, and is refused on any digest mismatch, unknown kind, missing
required product or schema drift."""
import json

import numpy as np
import pytest

from hcd_analysis.emulator import product_manifest as PM


def _files(tmp_path):
    out = {}
    for kind in ("mf", "dla_core", "cemu_t1", "cemu_t3"):
        p = tmp_path / f"{kind}.npz"
        np.savez(p, x=np.arange(3) + len(kind))
        out[kind] = str(p)
    return out


def test_build_then_verify_returns_the_paths(tmp_path):
    files = _files(tmp_path)
    m = PM.build(files, cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)
    path = tmp_path / "products.json"
    path.write_text(json.dumps(m))
    got = PM.verify(str(path), cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)
    assert got == {**files, "cemu_t2": None}


def test_digest_mismatch_and_wrong_cache_are_refused(tmp_path):
    files = _files(tmp_path)
    path = tmp_path / "products.json"
    path.write_text(json.dumps(PM.build(files, cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)))
    np.savez(files["cemu_t1"], x=np.arange(9))                          # the file changed after pinning
    with pytest.raises(PM.ProductManifestError, match="sha256"):
        PM.verify(str(path), cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)
    path.write_text(json.dumps(PM.build(files, cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)))
    with pytest.raises(PM.ProductManifestError, match="cache"):
        PM.verify(str(path), cache_sha256="d" * 64, ensemble_manifest_sha256="e" * 64)
    with pytest.raises(PM.ProductManifestError, match="ensemble"):
        PM.verify(str(path), cache_sha256="c" * 64, ensemble_manifest_sha256="f" * 64)


def test_required_products_and_kinds(tmp_path):
    files = _files(tmp_path)
    with pytest.raises(PM.ProductManifestError, match="required"):
        PM.build({k: v for k, v in files.items() if k != "dla_core"}, cache_sha256="c" * 64,
                 ensemble_manifest_sha256="e" * 64)
    with pytest.raises(PM.ProductManifestError, match="unknown"):
        PM.build({**files, "cemu_t9": files["mf"]}, cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)


def test_t2_is_recorded_as_held_until_registered(tmp_path):
    files = _files(tmp_path)
    m = PM.build(files, cache_sha256="c" * 64, ensemble_manifest_sha256="e" * 64)
    assert m["products"]["cemu_t2"] is None and "S14" in m["notes"]["cemu_t2"]
