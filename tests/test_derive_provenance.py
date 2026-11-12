"""The derivation script of the packaged dN/dX record writes the cache's repository-relative path (review m2): a
regeneration of hcd_analysis/emulator/hcd_lit_dndx_corrected.json must not bring back this machine's absolute path.
Synthetic inputs; the JSON is not regenerated."""

import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "derive_hcd_dndx_corrected.py")
_spec = importlib.util.spec_from_file_location("_derive_hcd_dndx_corrected", SCRIPT)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def test_the_cache_provenance_block_records_a_repository_relative_path():
    inputs = dict(
        path=os.path.join(D.REPO, "hcd_analysis", "_emulator_data", "observables_tau0_lf.h5"),
        shapes={"snap_f_nhi": (3, 30)},
        sha256_f_nhi="0" * 64,
    )
    assert D.cache_provenance(inputs, {"flag": "x"}) == dict(
        path="hcd_analysis/_emulator_data/observables_tau0_lf.h5",
        shapes={"snap_f_nhi": (3, 30)},
        sha256_f_nhi="0" * 64,
        same_file_consistency={"flag": "x"},
    )


def test_a_cache_outside_the_repository_is_refused():
    inputs = dict(path="/" + "scratch_area/observables.h5", shapes={}, sha256_f_nhi="0" * 64)
    with pytest.raises(ValueError, match="not inside the repository"):
        D.cache_provenance(inputs, {})


def test_main_writes_the_block_through_that_function():
    src = open(SCRIPT, encoding="utf-8").read()
    assert "cache_provenance=cache_provenance(inputs, sfc)" in src
    assert 'path=inputs["path"]' not in src
