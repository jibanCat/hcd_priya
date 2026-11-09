"""The training-cache builder and the per-class P1D patch script refuse the catalogues the pipeline resume paths refuse.

Both read Phase-1 ``catalog.npz`` files through ``catalog.load_validated_catalog`` (the guard ``pipeline.py`` uses on
both resume paths), so a catalogue built by the pre-fix finder (a box-edge system listed twice, or a whole-skewer
system spanning more than the skewer) cannot enter the per-class P1D or the cache's dN/dX silently.

Ported to the merged interface (the integration of PR #18 with the PR-A layer of PR #19): the builder's release path
reads a snapshot through ``tau0_cache.load_snapshot_inputs`` (``snapshot_record.verify_snapshot_dir``), whose
``load_validated_catalog`` checks the lineage record first and the entries second. The fixtures are therefore a real
catalogue-only snapshot rebuilt by the pipeline from a synthetic raw file (``tests/tau0_world.py``), with its records.
Each pre-fix variant is that catalogue plus its box-edge heads (or an over-long entry), written with a record
re-digested to match, so the record check passes and only the entry check (``validate_catalog_entries``) can refuse
it. Synthetic optical depths only; the fake_spectra and PyYAML stand-ins of ``tests/tau0_world.py``; no external data.
The historical-reproduction path (``load_snapshot_inputs_historical``) reads a pre-fix catalogue on purpose, with a
reason, and nothing built from it is release-eligible; it is not tested here."""
import copy
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import catalog as C
from hcd_analysis import lineage as L
from hcd_analysis import snapshot_record as SR
from hcd_analysis import tau0_cache as T
from hcd_analysis.paths import REPO_ROOT
from tests import tau0_world as W

sys.path.insert(0, str(REPO_ROOT / "scripts"))

SIM, SNAP, Z = W.GROUPS[0]
NB = W.NBINS[(SIM, SNAP)]                     # synthetic skewer length (pixels) of the raw file
PRE_FIX = ("box_edge_duplicate", "whole_skewer_twice")


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One catalogue-only snapshot rebuilt by the pipeline (release finder settings, records), and its raw file."""
    base = tmp_path_factory.mktemp("guard_world")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(L, "code_identity", lambda repo_dir=None: dict(W.CLEAN))
        W.use_yaml_stand_in(mp)
        W.use_fake_spectra_stand_in(mp)
        raw = W.write_raw(base / "emu", SIM, SNAP, Z, seed=0)
        past = 1_700_000_000_000_000_000                       # the raw file is older than everything else
        os.utime(raw, ns=(past, past))
        snap_dir = W.rebuild_snapshot(base / "rebuilt", base / "emu", SIM, SNAP, Z)
    return {"emu": base / "emu", "raw": raw, "snap_dir": snap_dir}


@pytest.fixture(autouse=True)
def _stand_ins(monkeypatch):
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(W.CLEAN))
    W.use_yaml_stand_in(monkeypatch)
    W.use_fake_spectra_stand_in(monkeypatch)


def _copy(world, root):
    """A fresh copy of the rebuilt snapshot directory at root/<sim>/snap_<NNN> (the raw file is shared, unchanged)."""
    return Path(shutil.copytree(world["snap_dir"], Path(root) / SIM / f"snap_{SNAP:03d}"))


def _pre_fix(snap_dir, kind):
    """Rewrite snap_dir/catalog.npz as a pre-fix catalogue whose record is consistent with its entries."""
    cat = C.AbsorberCatalog.load_npz(snap_dir / "catalog.npz")
    rec = copy.deepcopy(cat.lineage)

    def absorber(sk, start, end, nhi):
        return C.Absorber(skewer_idx=sk, pix_start=start, pix_end=end, v_center=0.0, NHI=nhi, b_kms=np.nan,
                          log_NHI=float(np.log10(nhi)), absorber_class=C.classify_system(nhi), fit_success=True,
                          fast_mode=True)
    if kind == "box_edge_duplicate":            # the old finder listed the head of a wrapping system again
        extra = [absorber(a.skewer_idx, 0, a.pix_end - NB, 0.5 * a.NHI) for a in cat.absorbers if a.pix_end >= NB]
    else:                                       # an all-merged skewer spanning 2n pixels
        extra = [absorber(3, 0, 2 * NB - 1, 1e19)]
    assert extra
    cat.absorbers.extend(extra)
    rec["product"]["payload_sha256"] = L.arrays_sha256(cat._arrays())
    rec["product"]["n_absorbers"] = len(cat.absorbers)
    L.write_npz(snap_dir / "catalog.npz", cat._arrays(), rec)


def test_the_fixtures_are_what_the_guard_says(world, tmp_path):
    """Sanity: the rebuilt catalogue passes the release loader; each pre-fix variant passes the record check (its
    record is consistent) and is refused by the entry check alone."""
    ok = _copy(world, tmp_path / "ok")
    assert len(C.load_validated_catalog(ok / "catalog.npz", NB).absorbers) > 0
    for kind in PRE_FIX:
        q = _copy(world, tmp_path / kind)
        _pre_fix(q, kind)
        C.check_catalog_record(C.AbsorberCatalog.load_npz(q / "catalog.npz"), NB)      # the record check passes
        with pytest.raises(C.CatalogIntegrityError, match="box-edge fix") as e:
            C.load_validated_catalog(q / "catalog.npz", NB)
        assert e.type is C.CatalogIntegrityError                                       # not a provenance refusal


# --- scripts/build_emulator_cache_tau0.py ------------------------------------------------------------------------

class _Reached(Exception):
    """Raised by a stub placed just after the catalogue use: the build got past the catalogue guard."""


def test_builder_accepts_a_post_fix_catalogue(world, tmp_path, monkeypatch):
    bt0 = _module("build_emulator_cache_tau0")

    def reached(*a, **k):
        raise _Reached
    monkeypatch.setattr(bt0, "catalogue_z_label", reached)
    sd = _copy(world, tmp_path)
    inputs, _ = T.load_snapshot_inputs(sd, world["raw"], sim=SIM, snap=SNAP)
    with pytest.raises(_Reached):
        bt0.build_tau0_rows(SIM, SNAP, inputs, world["raw"], np.array([1.0]), n_k=4)


@pytest.mark.parametrize("kind", PRE_FIX)
def test_builder_refuses_a_pre_fix_catalogue(world, tmp_path, monkeypatch, kind):
    """The builder's main reads every snapshot through load_snapshot_inputs: a pre-fix catalogue is refused there, by
    the entry check, before any row (P1D or dN/dX) is built, and no cache is written."""
    bt0 = _module("build_emulator_cache_tau0")
    monkeypatch.setattr(bt0, "build_tau0_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rows built")))
    _pre_fix(_copy(world, tmp_path / "hcd"), kind)
    out = tmp_path / "cache.h5"
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused.*box-edge fix") as e:
        bt0.main(["--hcd-root", str(tmp_path / "hcd"), "--emu-root", str(world["emu"]), "--output", str(out)])
    assert e.type is C.CatalogIntegrityError and not out.exists()


def test_builder_validates_with_the_paired_files_skewer_length(world, tmp_path, monkeypatch):
    """The entry check needs the skewer length n. On the release path it is meta.json's nbins, which
    verify_snapshot_dir ties to the record and the raw file (and the builder's pairing check to the paired file's
    header). A wrapping entry is legal only modulo that n: validating against a longer skewer would hide the
    duplicate."""
    sd = _copy(world, tmp_path)
    _pre_fix(sd, "box_edge_duplicate")
    seen, real = [], SR.load_validated_catalog

    def spy(path, n, **kw):
        seen.append(n)
        return real(path, n, **kw)
    monkeypatch.setattr(SR, "load_validated_catalog", spy)
    with pytest.raises(C.CatalogIntegrityError, match="box-edge fix"):
        T.load_snapshot_inputs(sd, world["raw"], sim=SIM, snap=SNAP)
    assert seen == [NB]


# --- scripts/patch_per_class_p1d.py: patch_one -------------------------------------------------------------------

def _patch_snapdir(world, tmp_path, monkeypatch, kind=None):
    P = _module("patch_per_class_p1d")
    root = tmp_path / "outputs"
    sd = _copy(world, root)
    if kind is not None:
        _pre_fix(sd, kind)
    (tmp_path / "lf" / SIM).mkdir(parents=True)
    monkeypatch.setattr(P, "OUTPUT_ROOT", root)
    monkeypatch.setattr(P, "LF_DATA_ROOT", tmp_path / "lf")
    monkeypatch.setattr(P, "spectra_files_for_sim", lambda info, prefer_grid=True: {SNAP: world["raw"]})
    calls = []

    def compute(**kw):
        calls.append(kw)
        return {f"n_sightlines_{c}": 0 for c in ("clean", "LLS", "subDLA", "DLA")}

    def save(path, res, **kw):
        Path(path).write_text("written")
    monkeypatch.setattr(P, "compute_p1d_per_class", compute)
    monkeypatch.setattr(P, "save_p1d_per_class_hdf5", save)
    return P, sd, calls


def test_patch_accepts_a_post_fix_catalogue(world, tmp_path, monkeypatch):
    P, sd, calls = _patch_snapdir(world, tmp_path, monkeypatch)
    n = len(C.AbsorberCatalog.load_npz(sd / "catalog.npz").absorbers)
    msg = P.patch_one(sd)
    assert msg.startswith("OK") and len(calls) == 1 and (sd / "p1d_per_class.h5").exists()
    assert len(calls[0]["catalog"].absorbers) == n


@pytest.mark.parametrize("kind", PRE_FIX)
def test_patch_refuses_a_pre_fix_catalogue_and_writes_nothing(world, tmp_path, monkeypatch, kind):
    P, sd, calls = _patch_snapdir(world, tmp_path, monkeypatch, kind)
    msg = P.patch_one(sd)
    assert msg.startswith("REFUSED") and "cached catalogue refused" in msg and "box-edge fix" in msg
    assert calls == [] and not (sd / "p1d_per_class.h5").exists()
