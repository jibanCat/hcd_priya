"""The training-cache builder and the per-class P1D patch script refuse the catalogues the pipeline resume paths refuse.

Both read Phase-1 ``catalog.npz`` files. They now load them through ``catalog.load_validated_catalog`` (the guard
``pipeline.py`` uses on both resume paths), so a catalogue built by the pre-fix finder (a box-edge system listed twice,
or a whole-skewer system spanning more than the skewer) cannot enter the per-class P1D or the cache's dN/dX silently.
Synthetic catalogues only; no fake_spectra, no external data."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import catalog as C
from hcd_analysis.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

N = 100                     # synthetic skewer length (pixels)


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# (skewer, pix_start, pix_end, NHI): a system crossing the box edge is listed once by the fixed finder
POST_FIX = [(0, 95, 104, 1e18), (1, 10, 20, 1e20), (2, 40, 41, 1e21)]
PRE_FIX = {
    "box_edge_duplicate": POST_FIX + [(0, 0, 4, 1e17 * 3)],        # the head of the wrapping system, listed again
    "whole_skewer_twice": POST_FIX + [(3, 0, 2 * N - 1, 1e19)],    # an all-merged skewer spanning 2n pixels
}


def _write_catalogue(path, entries, z=3.0, dv=10.0):
    sk, ps, pe, nhi = (np.array(x) for x in zip(*entries))
    np.savez(path, sim_name="synthsim", snap=1, z=z, dv_kms=dv, skewer_idx=sk.astype(np.int32),
             pix_start=ps.astype(np.int32), pix_end=pe.astype(np.int32), NHI=nhi.astype(np.float64),
             b_kms=np.full(sk.size, np.nan), fit_success=np.zeros(sk.size, bool), fast_mode=np.ones(sk.size, bool))


def test_the_synthetic_catalogues_are_what_the_guard_says():
    """Sanity: the pre-fix fixtures are refused by the existing guard, and the post-fix one is accepted."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ok.npz"
        _write_catalogue(p, POST_FIX)
        assert len(C.load_validated_catalog(p, N).absorbers) == len(POST_FIX)
        for kind, entries in PRE_FIX.items():
            q = Path(d) / f"{kind}.npz"
            _write_catalogue(q, entries)
            with pytest.raises(C.CatalogIntegrityError):
                C.load_validated_catalog(q, N)


# --- scripts/build_emulator_cache_tau0.py: build_tau0_rows ------------------------------------------------------

class _Reached(Exception):
    """Raised by a stub placed just after the catalogue load: the build got past the catalogue guard."""


def _builder_snapdir(tmp_path, entries, monkeypatch):
    bt0 = _module("build_emulator_cache_tau0")
    hdr = types.SimpleNamespace(nbins=N, redshift=3.0, box=120000.0, hubble=0.7, Hz=300.0)
    sd = tmp_path / "snap_001"
    sd.mkdir()
    _write_catalogue(sd / "catalog.npz", entries)
    (sd / "meta.json").write_text(json.dumps({"nbins": N, "dv_kms": bt0.vmax_from_header(hdr) / N, "z": 3.0,
                                              "n_absorbers": {"LLS": 1, "subDLA": 1, "DLA": 1, "forest": 0}}))
    np.savez(sd / "cddf.npz", total_path=np.float64(1.0), f_nhi=np.zeros(3), n_absorbers=np.zeros(3, int),
             log_nhi_centres=np.zeros(3), log_nhi_edges=np.zeros(4))
    import hcd_analysis.io as io
    monkeypatch.setattr(io, "read_header", lambda path: hdr)
    monkeypatch.setattr(bt0, "_read_priya_params", lambda path: np.zeros(9))
    stub = types.ModuleType("hcd_analysis.priya_p1d")             # the real one needs fake_spectra; never reached
    for name in ("compute_tier_c_p1d", "bin_sightlines_by_nhi", "_apply_priya_filter"):
        setattr(stub, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("P1D computed")))
    stub.N_TIER_C_BINS = 15
    monkeypatch.setitem(sys.modules, "hcd_analysis.priya_p1d", stub)

    def reached(*a, **k):
        raise _Reached
    monkeypatch.setattr(bt0, "catalogue_z_label", reached)
    return bt0, sd


def test_builder_accepts_a_post_fix_catalogue(tmp_path, monkeypatch):
    bt0, sd = _builder_snapdir(tmp_path, POST_FIX, monkeypatch)
    with pytest.raises(_Reached):
        bt0.build_tau0_rows("synthsim", 1, sd, tmp_path / "raw.hdf5", np.array([1.0]), n_k=4)


@pytest.mark.parametrize("kind", sorted(PRE_FIX))
def test_builder_refuses_a_pre_fix_catalogue(tmp_path, monkeypatch, kind):
    bt0, sd = _builder_snapdir(tmp_path, PRE_FIX[kind], monkeypatch)
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
        bt0.build_tau0_rows("synthsim", 1, sd, tmp_path / "raw.hdf5", np.array([1.0]), n_k=4)


def test_builder_validates_with_the_paired_files_skewer_length(tmp_path, monkeypatch):
    """The guard needs the skewer length n; it is the paired raw file's (equal to meta nbins by the pairing check).
    A wrapping entry is legal only modulo that n: validating against a longer skewer would hide the duplicate."""
    bt0, sd = _builder_snapdir(tmp_path, PRE_FIX["box_edge_duplicate"], monkeypatch)
    seen = []
    real = C.load_validated_catalog

    def spy(path, n):
        seen.append(n)
        return real(path, n)
    monkeypatch.setattr(C, "load_validated_catalog", spy)
    with pytest.raises(C.CatalogIntegrityError):
        bt0.build_tau0_rows("synthsim", 1, sd, tmp_path / "raw.hdf5", np.array([1.0]), n_k=4)
    assert seen == [N]


# --- scripts/patch_per_class_p1d.py: patch_one -------------------------------------------------------------------

def _patch_snapdir(tmp_path, entries, monkeypatch):
    P = _module("patch_per_class_p1d")
    root = tmp_path / "outputs"
    sd = root / "synthsim" / "snap_001"
    sd.mkdir(parents=True)
    _write_catalogue(sd / "catalog.npz", entries)
    (sd / "meta.json").write_text(json.dumps({"sim_name": "synthsim", "snap": 1, "nbins": N, "dv_kms": 10.0,
                                              "z": 3.0}))
    (tmp_path / "lf" / "synthsim").mkdir(parents=True)
    monkeypatch.setattr(P, "OUTPUT_ROOT", root)
    monkeypatch.setattr(P, "LF_DATA_ROOT", tmp_path / "lf")
    monkeypatch.setattr(P, "spectra_files_for_sim", lambda info, prefer_grid=True: {1: tmp_path / "raw.hdf5"})
    calls = []

    def compute(**kw):
        calls.append(kw)
        return {f"n_sightlines_{c}": 0 for c in ("clean", "LLS", "subDLA", "DLA")}

    def save(path, res, **kw):
        Path(path).write_text("written")
    monkeypatch.setattr(P, "compute_p1d_per_class", compute)
    monkeypatch.setattr(P, "save_p1d_per_class_hdf5", save)
    return P, sd, calls


def test_patch_accepts_a_post_fix_catalogue(tmp_path, monkeypatch):
    P, sd, calls = _patch_snapdir(tmp_path, POST_FIX, monkeypatch)
    msg = P.patch_one(sd)
    assert msg.startswith("OK") and len(calls) == 1 and (sd / "p1d_per_class.h5").exists()
    assert len(calls[0]["catalog"].absorbers) == len(POST_FIX)


@pytest.mark.parametrize("kind", sorted(PRE_FIX))
def test_patch_refuses_a_pre_fix_catalogue_and_writes_nothing(tmp_path, monkeypatch, kind):
    P, sd, calls = _patch_snapdir(tmp_path, PRE_FIX[kind], monkeypatch)
    msg = P.patch_one(sd)
    assert msg.startswith("REFUSED") and "cached catalogue refused" in msg
    assert calls == [] and not (sd / "p1d_per_class.h5").exists()
