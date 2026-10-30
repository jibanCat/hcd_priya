"""The absorber system finder and the catalogue integrity guards (docs/bugs_found.md, the box-edge duplicate).

The pre-fix find_systems_in_skewer listed every system that straddles the box edge twice: once whole (pix_end >= n) and
once as its own head [s', pix_end - n]. The guards make that class of error impossible to write silently:
  * an independent second finder (find_systems_in_skewer_reference) must agree with the production finder on every
    skewer (process_skewer_batch refuses otherwise);
  * no two catalogue entries of one skewer may share a pixel modulo n (validate_catalog_entries, run by build_catalog);
  * the stored class counts must reconcile three ways (labels, NHI values, the CDDF histogram) before meta.json is
    written (reconcile_counts, used by the pipeline);
  * every catalogue carries a lineage record stamped by build_catalog with FINDER_ID (only when the reference finder
    checked every searched skewer); save_npz, load_npz and the release loader load_validated_catalog refuse a
    catalogue without one, and load_catalog_historical is the only reader of an old one. A golden output pins
    FINDER_ID to what the finder and the N_HI estimator produce.
Synthetic optical depths only."""
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import catalog as C
from hcd_analysis import lineage as L

N, GAP, MINPIX, THR = 1342, 9, 2, 100.0
CLEAN = {"commit": "c" * 40, "dirty": False}


def _tau(on, n=N):
    t = np.zeros(n)
    t[np.asarray(list(on)) % n] = 1e3
    return t


def _both(tau, gap=GAP, minpix=MINPIX, thr=THR):
    a = C.find_systems_in_skewer(tau, thr, gap, minpix)
    b = C.find_systems_in_skewer_reference(tau, thr, gap, minpix)
    assert a == b, (a, b)
    return a


# --------------------------------------------------------- regression cases for bug 8 (docs/bugs_found.md)
def test_a_system_straddling_the_box_edge_is_listed_once():
    out = _both(_tau(range(N - 5, N + 10)))
    assert out == [(N - 5, N + 9)] and out[0][1] >= N


def test_a_straddling_system_and_a_middle_system_give_two_entries():
    assert _both(_tau(list(range(N - 5, N + 10)) + list(range(600, 620)))) == [(600, 619), (N - 5, N + 9)]


def test_a_head_merged_across_a_gap_at_the_edge_is_one_entry():
    assert _both(_tau(list(range(N - 3, N)) + list(range(4, 10)))) == [(N - 3, N + 9)]


def test_tau_above_threshold_everywhere_is_one_entry_covering_the_skewer_once():
    assert _both(np.full(N, 1e3)) == [(0, N - 1)]


def test_all_gaps_merged_is_one_entry_covering_the_skewer_once():
    on = [i for i in range(N) if i % 7]                         # gaps of 1 pixel everywhere, merged
    assert _both(_tau(on)) == [(0, N - 1)]


def test_no_wrap_and_one_pixel_cases_are_unchanged():
    assert _both(_tau(list(range(10, 20)) + list(range(400, 450)))) == [(10, 19), (400, 449)]
    assert _both(_tau([700])) == []                             # shorter than min_pixels
    assert _both(_tau([700]), minpix=1) == [(700, 700)]
    assert _both(np.zeros(N)) == []


# ----------------------------------------------------------------------------- the two finders agree everywhere
@pytest.mark.parametrize("n", [37, 100, 1342])
@pytest.mark.parametrize("gap,minpix", [(1, 1), (3, 2), (9, 2), (9, 5)])
def test_production_and_reference_finders_agree_on_random_skewers(n, gap, minpix):
    rng = np.random.default_rng(20261027 + n + 10 * gap + minpix)
    for density in (0.01, 0.05, 0.2, 0.5, 0.9, 0.99):
        for _ in range(60):
            tau = np.where(rng.random(n) < density, 1e3, 0.0)
            out = _both(tau, gap=gap, minpix=minpix)
            if out:
                C.validate_catalog_entries([0] * len(out), [s for s, _ in out], [e for _, e in out], n)


# ----------------------------------------------------------------------------- guards
def test_validator_and_audit_catch_the_pre_fix_duplicate():
    sk, ps, pe = [5, 5, 6], [N - 5, 0, 10], [N + 9, 9, 20]      # skewer 5: whole (wraps) and its head
    dup = C.find_wrap_duplicates(sk, ps, pe, N)
    assert dup.tolist() == [False, True, False]
    with pytest.raises(C.CatalogIntegrityError, match="box-edge duplicate"):
        C.validate_catalog_entries(sk, ps, pe, N)
    keep = ~dup
    C.validate_catalog_entries(np.asarray(sk)[keep], np.asarray(ps)[keep], np.asarray(pe)[keep], N)


@pytest.mark.parametrize("sk,ps,pe,match", [
    ([1, 1], [10, 15], [20, 30], "overlapping neighbours"),
    ([1], [N], [N + 3], "impossible pixel spans"),
    ([1], [5], [N + 5], "impossible pixel spans"),
    ([1], [9], [8], "impossible pixel spans"),
    ([5, 5], [N - 5, 9], [N + 9, 9], "box-edge duplicate"),          # the head ends exactly at the wrapped tail end
])
def test_validator_refuses_other_impossible_catalogues(sk, ps, pe, match):
    with pytest.raises(C.CatalogIntegrityError, match=match):
        C.validate_catalog_entries(sk, ps, pe, N)


def _batch(n_sk=24, n=N, seed=3):
    rng = np.random.default_rng(seed)
    tau = np.zeros((n_sk, n))
    for i in range(n_sk):
        for _ in range(rng.integers(0, 4)):
            c = int(rng.integers(0, n))
            w = int(rng.integers(3, 40))
            tau[i, np.arange(c - w, c + w) % n] = 10 ** rng.uniform(3, 7)
    tau[0, np.arange(N - 6, N + 12) % n] = 1e6                  # one certain straddler
    return tau


def _process(tau, **kw):
    return C.process_skewer_batch(tau, 0, 10.0, THR, 100.0, MINPIX, 30.0, (1.0, 300.0), 1e6, 50, True,
                                  min_log_nhi=17.0, **kw)


def test_batch_catalogue_has_no_duplicates_and_validates():
    ab = _process(_batch())
    sk, ps, pe = ([a.skewer_idx for a in ab], [a.pix_start for a in ab], [a.pix_end for a in ab])
    assert any(e >= N for e in pe)
    assert not C.find_wrap_duplicates(sk, ps, pe, N).any()
    C.validate_catalog_entries(sk, ps, pe, N)


def test_batch_refuses_when_the_finders_disagree(monkeypatch):
    monkeypatch.setattr(C, "find_systems_in_skewer_reference", lambda *a, **k: [(0, 1)])
    with pytest.raises(C.CatalogIntegrityError, match="finders disagree"):
        _process(_batch())
    assert _process(_batch(), verify_finder=False)               # the guard is what refused


def _cat(logn, labels=None):
    cat = C.AbsorberCatalog(sim_name="synthetic", snap=1, z=3.0, dv_kms=10.0)
    for i, l in enumerate(logn):
        cat.absorbers.append(C.Absorber(skewer_idx=i, pix_start=0, pix_end=3, v_center=1.0, NHI=10 ** l, b_kms=np.nan,
                                        log_NHI=l, absorber_class=(labels[i] if labels else C.classify_system(10 ** l)),
                                        fit_success=True, fast_mode=True))
    return cat


def _legacy_save(cat, path):
    """catalog.npz as it was written before catalogues carried a lineage record (the arrays only)."""
    np.savez(str(path), **cat._arrays())


def _stamp(cat, nbins=N, finder_ids=(C.FINDER_ID,), verified=True, code=CLEAN):
    """Attach a lineage record to a hand-made catalogue, as a hand edit could (the stated limit of the record check:
    the value checks are the second line). Not a code path of the package."""
    raw = {"sha256": "0" * 64, "size": 1, "mtime_ns": 1}
    product = {"sim_name": cat.sim_name, "snap": int(cat.snap), "z": float(cat.z), "dv_kms": float(cat.dv_kms),
               "nbins": int(nbins), "n_skewers_built": 4, "n_absorbers": len(cat.absorbers),
               "payload_sha256": L.arrays_sha256(cat._arrays()), "finder": {"verified": verified}}
    cat.lineage = L.build_record("catalog", C.CATALOG_KIND_VERSION,
                                 inputs=[L.input_entry("raw_spectra", "raw_spectra", raw)],
                                 finder_ids=list(finder_ids), settings={}, code=code,
                                 allowed_finder_ids=C.RELEASE_FINDER_IDS, product=product)
    return cat


def test_counts_reconcile_three_ways_or_refuse():
    logn = [17.5, 18.2, 19.4, 20.1, 20.5, 21.3]
    edges = np.linspace(17.0, 23.0, 31)
    counts = np.histogram(logn, bins=edges)[0]
    assert C.reconcile_counts(_cat(logn), counts, edges) == {"LLS": 2, "subDLA": 2, "DLA": 2, "forest": 0}
    with pytest.raises(C.CatalogIntegrityError, match="labels"):
        C.reconcile_counts(_cat(logn, ["LLS"] * 6), counts, edges)
    with pytest.raises(C.CatalogIntegrityError, match="CDDF holds"):
        C.reconcile_counts(_cat(logn), counts + np.eye(30, dtype=int)[3], edges)


def test_pipeline_writes_reconciled_counts_and_refuses_a_mismatch(tmp_path, monkeypatch):
    PL = _pipeline(monkeypatch)
    logn = [17.5, 19.4, 20.5]
    edges = np.linspace(17.0, 23.0, 31)
    cddf = {"log_nhi_edges": edges, "n_absorbers": np.histogram(logn, bins=edges)[0], "f_nhi": np.zeros(30)}
    hdr = types.SimpleNamespace(nbins=N, n_skewers=3, box=120000.0, hubble=0.7)
    args = dict(p1d_variants={}, p1d_ratios={}, perturbed_p1d=None, excl_result=None, ics_meta={}, timing={},
                header=hdr, sim_name="synthetic", snap=1, z=3.0, dv_kms=10.0)
    PL._save_snap_outputs(tmp_path, _cat(logn), cddf_result=cddf, **args)
    meta = json.load(open(tmp_path / "meta.json"))
    assert meta["n_absorbers"] == {"LLS": 1, "subDLA": 1, "DLA": 1, "forest": 0}
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(C.CatalogIntegrityError):
        PL._save_snap_outputs(bad, _cat(logn), cddf_result=dict(cddf, n_absorbers=cddf["n_absorbers"] * 2), **args)
    assert not any(bad.iterdir())                                    # nothing written past the failed check


# ----------------------------------------------------------------------------- review follow-ups
def test_a_whole_skewer_system_shorter_than_min_pixels_is_dropped_by_both_finders():
    short = 5
    assert _both(np.full(short, 1e3), gap=1, minpix=6) == []                       # above threshold everywhere
    assert _both(np.array([1e3, 1e3, 1e3, 1e3, 0.0]), gap=1, minpix=6) == []       # every gap merged
    assert _both(np.array([1e3, 1e3, 1e3, 1e3, 0.0]), gap=1, minpix=5) == [(0, 4)]
    assert _both(np.full(short, 1e3), gap=1, minpix=5) == [(0, 4)]


def test_audit_flags_over_span_entries_and_ignores_valid_ones():
    sk, ps, pe = [3, 4, 4, 5], [0, 10, N - 2, 0], [2 * N - 1, 20, N + 4, N - 1]   # skewer 3: the pre-fix all-above
    assert C.find_wrap_duplicates(sk, ps, pe, N).tolist() == [True, False, False, False]   # skewer 5: valid (0, n-1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_column_densities_are_refused_with_a_clear_message(bad):
    cat = _cat([17.5, 19.4])
    cat.absorbers[0].NHI = bad
    edges = np.linspace(17.0, 23.0, 31)
    with pytest.raises(C.CatalogIntegrityError, match="non-finite N_HI"):
        C.reconcile_counts(cat, np.histogram([19.4], bins=edges)[0], edges)
    with pytest.raises(C.CatalogIntegrityError, match="non-finite N_HI"):
        C.check_catalog_values(cat)


def test_label_value_mismatch_is_refused():
    with pytest.raises(C.CatalogIntegrityError, match="labels"):
        C.check_catalog_values(_cat([17.5, 19.4], ["DLA", "subDLA"]))


def test_a_cached_catalogue_with_valid_spans_and_a_non_finite_column_density_is_refused(tmp_path):
    cat = _cat([17.5, 19.4])
    cat.absorbers[1].skewer_idx, cat.absorbers[1].pix_start, cat.absorbers[1].pix_end = 1, 30, 40
    cat.absorbers[1].NHI = float("nan")
    _stamp(cat).save_npz(tmp_path / "nan.npz")                         # a record cannot vouch for the values
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused .*non-finite N_HI"):
        C.load_validated_catalog(tmp_path / "nan.npz", N)


def test_a_cached_pre_fix_catalogue_is_refused_on_reuse(tmp_path):
    good = _cat([17.5, 19.4])
    good.absorbers[1].skewer_idx, good.absorbers[1].pix_start, good.absorbers[1].pix_end = 0, N - 5, N + 9
    good.absorbers[0].skewer_idx, good.absorbers[0].pix_start, good.absorbers[0].pix_end = 0, 100, 120
    _stamp(good).save_npz(tmp_path / "good.npz")
    assert len(C.load_validated_catalog(tmp_path / "good.npz", N).absorbers) == 2
    bad = _cat([17.5, 19.4])
    bad.absorbers[1].skewer_idx, bad.absorbers[1].pix_start, bad.absorbers[1].pix_end = 0, N - 5, N + 9
    bad.absorbers[0].skewer_idx, bad.absorbers[0].pix_start, bad.absorbers[0].pix_end = 0, 0, 9   # its own head
    _legacy_save(bad, tmp_path / "bad.npz")                            # as the pre-fix finder wrote it: no record
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused .*no lineage record"):
        C.load_validated_catalog(tmp_path / "bad.npz", N)
    _stamp(bad).save_npz(tmp_path / "bad_forged.npz")                  # a hand-copied record: the value check refuses
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused .*overlap"):
        C.load_validated_catalog(tmp_path / "bad_forged.npz", N)


# ----------------------------------------------------------------------------- the pipeline's resume paths
def _pipeline(monkeypatch):
    """hcd_analysis.pipeline, importable without PyYAML: the config module imports it at load time, but the snapshot
    path exercised here never calls it, so a stub stands in when it is absent. The stub is removed from sys.modules at
    teardown, but hcd_analysis.config stays cached with it bound; any attribute other than a module dunder raises
    ImportError, as a missing PyYAML would."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        stub = types.ModuleType("yaml")

        def _absent(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)                         # so hasattr / inspect see a plain module
            raise ImportError(f"PyYAML is not installed (yaml.{name} requested; "
                              "stub from tests/test_catalog_systems.py)")
        stub.__getattr__ = _absent
        monkeypatch.setitem(sys.modules, "yaml", stub)
    from hcd_analysis import pipeline as PL
    return PL


def _synthetic_snapshot(root, nb=64, n_sk=4, z=3.0):
    """A tiny tau file: skewer 0 holds a system straddling the box edge, skewer 1 a system in the middle."""
    import h5py
    root.mkdir(parents=True, exist_ok=True)
    p = root / "s.hdf5"
    tau = np.random.default_rng(0).uniform(0.0, 0.3, size=(n_sk, nb)).astype(np.float32)
    tau[0, nb - 5:] = 1e5
    tau[0, :4] = 1e5
    tau[1, 20:26] = 1e5
    with h5py.File(p, "w") as f:
        g = f.create_group("Header")
        for k, v in dict(redshift=z, hubble=0.7, omegam=0.3, omegal=0.7, box=1200.0, nbins=nb, omegab=0.05).items():
            g.attrs[k] = v
        f.create_dataset("tau/H/1/1215", data=tau)
    return p


def _restamp(cat, like):
    """``cat`` with a copy of the lineage record of ``like`` (a catalogue build_catalog made for the same snapshot),
    its payload digest and count re-made to fit: what a hand edit of the record could produce."""
    rec = like.lineage
    product = dict(rec["product"], n_absorbers=len(cat.absorbers), payload_sha256=L.arrays_sha256(cat._arrays()))
    cat.lineage = L.build_record("catalog", rec["kind_version"], inputs=rec["inputs"], finder_ids=rec["finder_ids"],
                                 settings=rec["settings"], code=rec["code"], allowed_finder_ids=C.RELEASE_FINDER_IDS,
                                 run_settings_sha256=rec["run_settings_sha256"], product=product)
    return cat


def _pre_fix_catalogue(PL, hdr, nb, snap=1, z=3.0):
    """What the pre-fix finder wrote for that file: the straddling system on skewer 0 AND its own head."""
    cat = C.AbsorberCatalog(sim_name="synthsim", snap=snap, z=z, dv_kms=PL.pixel_dv_kms(hdr))
    for sk, ps, pe in ((0, nb - 5, nb + 3), (0, 0, 3), (1, 20, 25)):
        cat.absorbers.append(C.Absorber(skewer_idx=sk, pix_start=ps, pix_end=pe, v_center=0.0, NHI=10 ** 20.5,
                                        b_kms=np.nan, log_NHI=20.5, absorber_class="DLA", fit_success=True,
                                        fast_mode=True))
    return cat


def test_pipeline_resume_refuses_a_cached_pre_fix_catalogue_done_or_not(tmp_path, monkeypatch):
    PL = _pipeline(monkeypatch)
    from hcd_analysis.config import PipelineConfig
    from hcd_analysis.io import SimInfo, read_header
    from hcd_analysis.snapshot_map import SimSnapshot, SnapEntry
    h5 = _synthetic_snapshot(tmp_path)
    hdr = read_header(h5)
    nb = int(hdr.nbins)
    entry = SnapEntry(snap=1, a=0.25, z=3.0, path=h5)
    entry2 = SnapEntry(snap=2, a=1 / 4.1, z=3.1, path=_synthetic_snapshot(tmp_path / "second", z=3.1))
    sim = SimInfo(name="synthsim", path=tmp_path / "simroot", params={})
    ss, ss2 = SimSnapshot(sim=sim, entries=[entry]), SimSnapshot(sim=sim, entries=[entry, entry2])

    def cfg(name):
        c = PipelineConfig(output_root=str(tmp_path / name), resume=True, n_workers=1)
        c.p1d.nhi_excl_thresholds = []
        PL.check_run_settings(c)              # record the root's settings before the scenarios below place files in it
        return c

    def files(d):
        return sorted((p.name, p.stat().st_mtime_ns) for p in d.iterdir())

    # a fresh build lists the straddling system once and completes
    c0 = cfg("fresh")
    d0 = PL.snap_output_dir(c0, "synthsim", 1)
    res = PL.run_one_snap(ss, entry, c0)
    assert res is not None and sum(json.load(open(d0 / "meta.json"))["n_absorbers"].values()) == 2
    # done and valid: reused (validated with the snapshot's own nbins), nothing rewritten
    before = files(d0)
    again = PL.run_one_snap(ss, entry, c0)
    assert again is not None and len(again.catalog.absorbers) == 2 and files(d0) == before
    # not done, with a valid cached catalogue: reused as it is (validated with the file's own nbins), then completed
    cv = cfg("cached_valid")
    dv = PL.snap_output_dir(cv, "synthsim", 1)
    shutil.copy2(d0 / "catalog.npz", dv / "catalog.npz")
    stamp = (dv / "catalog.npz").stat().st_mtime_ns
    reused = PL.run_one_snap(ss, entry, cv)
    assert reused is not None and len(reused.catalog.absorbers) == 2 and PL.is_done(cv, "synthsim", 1)
    assert (dv / "catalog.npz").stat().st_mtime_ns == stamp
    # the refused catalogues: the pre-fix one as it was written (no lineage record), the pre-fix one with a record
    # copied onto it by hand (the value check is the second line), and a valid one damaged to hold a non-finite N_HI
    # (with its record re-made by hand, so the value check is what refuses it)
    built = C.AbsorberCatalog.load_npz(d0 / "catalog.npz")
    nan_cat = C.AbsorberCatalog.load_npz(d0 / "catalog.npz")
    nan_cat.absorbers[1].NHI = float("nan")
    _restamp(nan_cat, built)
    forged = _restamp(_pre_fix_catalogue(PL, hdr, nb), built)

    def legacy(d):
        _legacy_save(_pre_fix_catalogue(PL, hdr, nb), d / "catalog.npz")

    def saved(cat):
        return lambda d: cat.save_npz(d / "catalog.npz")
    # not done, with a cached refused catalogue: the snapshot fails before anything else is written (no per-class
    # P1D, no meta.json, not marked done)
    for name, put, why in (("cached", legacy, "no lineage record"), ("cached_forged", saved(forged), "overlap"),
                           ("cached_nan", saved(nan_cat), "non-finite N_HI")):
        c1 = cfg(name)
        d1 = PL.snap_output_dir(c1, "synthsim", 1)
        put(d1)
        assert PL.run_one_snap(ss, entry, c1) is None
        assert sorted(p.name for p in d1.iterdir()) == ["catalog.npz", "error.txt"]
        assert not PL.is_done(c1, "synthsim", 1)
        err = (d1 / "error.txt").read_text()
        assert "cached catalogue refused" in err and why in err
    # ... and at simulation level the rest of the simulation runs and is stacked without that snapshot
    cm = cfg("cached_mixed")
    legacy(PL.snap_output_dir(cm, "synthsim", 1))
    out = PL.run_sim_all_z(ss2, cm)
    assert out[0] is None and out[1] is not None
    with np.load(Path(cm.output_root) / "synthsim" / "cddf_stacked.npz") as st:
        zs = [st[k].tolist() for k in st.files if k.endswith("__z_snapshots")]
    assert zs and all(z == [3.1] for z in zs)
    # done, with a refused catalogue: refused loudly, never reused, nothing written, and the simulation's other
    # snapshot (which on its own is stacked: control below) is not stacked without it
    done_cases = []
    for name, put in (("done", legacy), ("done_forged", saved(forged)), ("done_nan", saved(nan_cat))):
        c = cfg(name)
        d = PL.snap_output_dir(c, "synthsim", 1)
        shutil.copytree(d0, d, dirs_exist_ok=True)                 # a completed snapshot ...
        put(d)                                                     # ... whose catalogue was then replaced
        done_cases.append((c, d))
    for c, d in done_cases:
        before = files(d)
        sim_dir = Path(c.output_root) / "synthsim"
        with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
            PL.run_one_snap(ss, entry, c)
        with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
            PL.run_sim_all_z(ss2, c)
        assert files(d) == before and not (sim_dir / "snap_002").exists()          # the run stopped at snapshot 1
        assert not (sim_dir / "cddf_stacked.npz").exists()
        with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
            PL.run_sim_all_z(SimSnapshot(sim=sim, entries=[entry2, entry]), c)      # the refused snapshot second
        assert files(d) == before and not (sim_dir / "cddf_stacked.npz").exists()
    c4 = cfg("control")
    PL.run_sim_all_z(SimSnapshot(sim=sim, entries=[entry2]), c4)
    assert (Path(c4.output_root) / "synthsim" / "cddf_stacked.npz").exists()


def test_a_fresh_build_refuses_a_non_finite_column_density_before_saving_the_catalogue(tmp_path, monkeypatch):
    PL = _pipeline(monkeypatch)
    from hcd_analysis.config import PipelineConfig
    from hcd_analysis.io import SimInfo
    from hcd_analysis.snapshot_map import SimSnapshot, SnapEntry
    entry = SnapEntry(snap=1, a=0.25, z=3.0, path=_synthetic_snapshot(tmp_path))
    ss = SimSnapshot(sim=SimInfo(name="synthsim", path=tmp_path / "simroot", params={}), entries=[entry])
    measure = C.measure_nhi_for_system

    def nan_for_the_middle_system(tau_row, pix_start, *args):
        nhi, b, ok = measure(tau_row, pix_start, *args)
        return (float("nan") if pix_start == 20 else nhi), b, ok
    monkeypatch.setattr(C, "measure_nhi_for_system", nan_for_the_middle_system)
    c = PipelineConfig(output_root=str(tmp_path / "out"), resume=True, n_workers=1)
    c.p1d.nhi_excl_thresholds = []
    d = PL.snap_output_dir(c, "synthsim", 1)
    assert PL.run_one_snap(ss, entry, c) is None
    assert sorted(p.name for p in d.iterdir()) == ["error.txt"]       # no catalog.npz: refused inside build_catalog
    assert "non-finite N_HI" in (d / "error.txt").read_text()


# ----------------------------------------------------------------------------- finder identity and lineage
def _golden_tau():
    """Twelve skewers of 400 pixels, every tau exactly representable in float32 (so the N_HI sums are exact)."""
    n = 400
    t = np.full((12, n), 0.5)
    lls, sub, dla = 13440.0, 673000.0, 6720000.0
    t[0, 392:] = dla
    t[0, :8] = dla                                          # a DLA straddling the box edge
    t[1, 100:110] = lls
    t[1, 115:125] = lls                                     # two blocks merged across a short gap ...
    t[1, 300:310] = sub                                     # ... and a separate subDLA
    t[2, 396:] = sub
    t[2, 3:9] = sub                                         # a head merged across a gap at the edge
    t[3, :] = 200.0
    t[3, 50:60] = lls                                       # above threshold everywhere: one whole-skewer system
    t[4, 200] = dla                                         # one pixel: shorter than min_pixels
    t[5, 10:20] = 300.0                                     # below the pre-filter floor: not searched
    t[6, 10:20] = 1000.0                                    # searched, below min_log_nhi: dropped
    t[7, :] = lls
    t[7, ::7] = 0.5                                         # every gap merged: one whole-skewer system
    t[8, :] = 0.0                                           # empty
    t[9, 20:30] = lls
    t[9, 200:230] = dla                                     # an LLS and a DLA
    t[10, 100:160] = lls                                    # a wide block (b_eff clipped at 300 km/s)
    t[11, 395:] = sub
    t[11, 15:25] = sub                                      # a block touching the edge and a separate one
    return t.astype(np.float32)


# The catalogue each FINDER_ID produces from _golden_tau (dv 10 km/s, tau_threshold 100, merge 100 km/s, min_pixels 2,
# fast mode, min_log_nhi 17.2): (skewer_idx, pix_start, pix_end, N_HI as float.hex, class). A change of the finder or
# of the N_HI estimator that changes any entry fails this test: bump FINDER_ID in hcd_analysis/catalog.py and add the
# new id's output here (the old id's entry stays, as the record of what it produced).
GOLDEN = {
    "hcd_priya.finder/2": {
        "n_skewers_searched": 10,
        "entries": [
            (0, 392, 407, "0x1.5b14b1bb08266p+69", "DLA"),
            (1, 100, 124, "0x1.bc4481a4f2540p+60", "LLS"),
            (1, 300, 309, "0x1.5b98ea604335ap+65", "subDLA"),
            (2, 396, 408, "0x1.5b98ef740cef0p+65", "subDLA"),
            (3, 0, 399, "0x1.5f0c2753d7f80p+60", "LLS"),
            (7, 0, 399, "0x1.2749e76fd6648p+62", "LLS"),
            (9, 20, 29, "0x1.bc4372daebb63p+59", "LLS"),
            (9, 200, 229, "0x1.4563669f57a3fp+70", "DLA"),
            (10, 100, 159, "0x1.2749e76fd6648p+62", "LLS"),
            (11, 15, 24, "0x1.5b98ea604335ap+65", "subDLA"),
            (11, 395, 399, "0x1.5b98ea604335ap+64", "subDLA"),
        ],
    },
}
GOLDEN_KW = dict(dv_kms=10.0, tau_threshold=THR, merge_dv_kms=100.0, min_pixels=2, fast_mode=True, min_log_nhi=17.2)


def _write_tau(path, tau, z=3.0):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("Header")
        for k, v in dict(redshift=z, hubble=0.7, omegam=0.3, omegal=0.7, box=1200.0, nbins=tau.shape[1],
                         omegab=0.05).items():
            g.attrs[k] = v
        f.create_dataset("tau/H/1/1215", data=tau)
    return path


def _golden_build(tmp_path, monkeypatch, code=CLEAN, **kw):
    pytest.importorskip("h5py")
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(code))
    h5 = _write_tau(tmp_path / "golden" / "tau.hdf5", _golden_tau())
    args = dict(GOLDEN_KW, batch_size=5)                   # several batches, one of them short
    args.update(kw)
    return h5, C.build_catalog(h5, "synthsim", 7, 3.0, **args)


def test_the_golden_catalogue_is_pinned_to_the_finder_id(tmp_path, monkeypatch):
    golden = GOLDEN[C.FINDER_ID]                                   # a new FINDER_ID needs its golden output
    assert C.FINDER_ID in C.RELEASE_FINDER_IDS
    h5, cat = _golden_build(tmp_path, monkeypatch)
    got = [(a.skewer_idx, a.pix_start, a.pix_end, float.hex(float(a.NHI)), a.absorber_class) for a in cat.absorbers]
    assert got == golden["entries"]
    assert all(a.fast_mode and a.fit_success and np.isnan(a.b_kms) for a in cat.absorbers)
    fin = cat.lineage["product"]["finder"]
    assert fin["n_skewers_searched"] == fin["n_skewers_verified"] == golden["n_skewers_searched"]
    assert cat.lineage["finder_ids"] == [C.FINDER_ID]


def test_build_catalog_stamps_the_record(tmp_path, monkeypatch):
    h5, cat = _golden_build(tmp_path, monkeypatch, run_settings_sha256="d" * 64)
    rec = cat.lineage
    L.check_record(rec, C.RELEASE_FINDER_IDS)
    assert rec["kind"] == "catalog" and rec["kind_version"] == C.CATALOG_KIND_VERSION
    assert rec["release_eligible"] and cat.release_eligible and rec["code"] == CLEAN
    assert rec["run_settings_sha256"] == "d" * 64
    raw = L.find_input(rec, "raw_spectra")
    assert {k: raw[k] for k in ("sha256", "size", "mtime_ns")} == L.file_identity(h5)
    assert raw["info"]["nbins"] == 400 and raw["info"]["n_skewers"] == 12
    assert rec["settings"] == C.finder_settings(tau_threshold=THR, merge_dv_kms=100.0, min_pixels=2, b_init=30.0,
                                                b_bounds=(1.0, 300.0), tau_fit_cap=1e6, voigt_max_iter=200,
                                                n_skewers=None, fast_mode=True, min_log_nhi=17.2)
    prod = rec["product"]
    assert (prod["sim_name"], prod["snap"], prod["nbins"], prod["n_skewers_built"], prod["n_absorbers"]) == \
        ("synthsim", 7, 400, 12, 11)
    assert prod["payload_sha256"] == L.arrays_sha256(cat._arrays())
    src = prod["finder"]["source_sha256"]
    assert set(src) == {"catalog.py", "voigt_utils.py"} and all(len(v) == 64 for v in src.values())
    assert src["catalog.py"] == L.sha256_file(C.__file__) and len(prod["finder"]["estimator_source_sha256"]) == 64
    _, few = _golden_build(tmp_path / "few", monkeypatch, n_skewers=7)
    assert few.lineage["product"]["n_skewers_built"] == 7 and few.lineage["settings"]["n_skewers"] == 7


@pytest.mark.parametrize("code, why", [({"commit": "c" * 40, "dirty": True}, "dirty"),
                                       ({"commit": None, "dirty": None}, "commit is unknown")])
def test_a_build_from_a_dirty_or_unknown_tree_is_stamped_but_not_release_eligible(tmp_path, monkeypatch, code, why):
    h5, cat = _golden_build(tmp_path, monkeypatch, code=code)
    assert cat.lineage["finder_ids"] == [C.FINDER_ID] and not cat.release_eligible
    assert any(why in r for r in cat.lineage["ineligible_reasons"])
    cat.save_npz(tmp_path / "c.npz")                                 # saved and reusable, never release-eligible
    C.load_validated_catalog(tmp_path / "c.npz", 400)
    with pytest.raises(C.CatalogProvenanceError, match="not release-eligible"):
        C.load_validated_catalog(tmp_path / "c.npz", 400, require_release_eligible=True)


def test_without_the_finder_cross_check_no_finder_id_is_stamped_and_nothing_can_be_saved(tmp_path, monkeypatch):
    h5, cat = _golden_build(tmp_path, monkeypatch, verify_finder=False)
    assert cat.lineage["finder_ids"] == [] and not cat.release_eligible
    assert cat.lineage["product"]["finder"]["verified"] is False
    with pytest.raises(C.CatalogProvenanceError, match="without a finder-stamped lineage record"):
        cat.save_npz(tmp_path / "c.npz")
    real = C._scan_skewer_batch

    def one_unchecked(*a, **k):                                    # a batch whose count shows a skewer unchecked
        ab, n_s, n_v = real(*a, **k)
        return ab, n_s, max(n_v - 1, 0)
    monkeypatch.setattr(C, "_scan_skewer_batch", one_unchecked)
    h5, cat = _golden_build(tmp_path / "partial", monkeypatch)
    assert cat.lineage["finder_ids"] == []
    with pytest.raises(C.CatalogProvenanceError, match="without a finder-stamped"):
        cat.save_npz(tmp_path / "p.npz")
    assert not (tmp_path / "c.npz").exists() and not (tmp_path / "p.npz").exists()


def test_a_timing_build_without_lineage_cannot_be_saved(tmp_path, monkeypatch):
    h5, cat = _golden_build(tmp_path, monkeypatch, record_lineage=False)
    assert cat.lineage is None and len(cat.absorbers) == 11
    with pytest.raises(C.CatalogProvenanceError):
        cat.save_npz(tmp_path / "c.npz")


def test_the_parallel_build_matches_the_serial_one(tmp_path, monkeypatch):
    pytest.importorskip("joblib")
    h5, serial = _golden_build(tmp_path, monkeypatch)
    _, par = _golden_build(tmp_path / "par", monkeypatch, n_workers=2)
    assert par._arrays().keys() == serial._arrays().keys()
    assert L.arrays_sha256(par._arrays()) == L.arrays_sha256(serial._arrays())
    assert par.lineage["product"]["finder"] == serial.lineage["product"]["finder"]
    assert par.lineage["finder_ids"] == [C.FINDER_ID]


def test_round_trip_and_the_release_loader_expectations(tmp_path, monkeypatch):
    h5, cat = _golden_build(tmp_path, monkeypatch)
    p = tmp_path / "catalog.npz"
    cat.save_npz(p)
    with np.load(p, allow_pickle=False) as d:                      # readable without pickle
        assert L.NPZ_KEY in d.files
    back = C.AbsorberCatalog.load_npz(p)
    assert back.lineage == cat.lineage and back._arrays().keys() == cat._arrays().keys()
    assert all(np.array_equal(back._arrays()[k], cat._arrays()[k], equal_nan=k == "b_kms") for k in cat._arrays())
    settings = cat.lineage["settings"]
    ok = C.load_validated_catalog(p, 400, settings=settings, raw_path=h5, n_skewers_built=12,
                                  require_release_eligible=True)
    assert ok.release_eligible and len(ok.absorbers) == 11
    with pytest.raises(C.CatalogProvenanceError, match="tau_threshold: recorded 100.0, expected 50.0"):
        C.load_validated_catalog(p, 400, settings=dict(settings, tau_threshold=50.0))
    with pytest.raises(C.CatalogProvenanceError, match="built from 12 skewers, expected 13"):
        C.load_validated_catalog(p, 400, n_skewers_built=13)
    with pytest.raises(C.CatalogProvenanceError, match="pixels, expected 401"):
        C.load_validated_catalog(p, 401)
    other = _write_tau(tmp_path / "other" / "tau.hdf5", _golden_tau()[::-1].copy())
    with pytest.raises(C.CatalogProvenanceError, match="not the raw_spectra input"):
        C.load_validated_catalog(p, 400, raw_path=other)
    import os
    os.utime(h5, ns=(1, 1))                                          # same bytes, another mtime: still accepted
    C.load_validated_catalog(p, 400, raw_path=h5)


def test_a_catalogue_changed_after_stamping_is_refused(tmp_path, monkeypatch):
    h5, cat = _golden_build(tmp_path, monkeypatch)
    cat.absorbers.pop()
    with pytest.raises(C.CatalogProvenanceError, match="changed after build_catalog stamped it"):
        cat.save_npz(tmp_path / "c.npz")
    h5, cat = _golden_build(tmp_path / "b", monkeypatch)
    p = tmp_path / "c.npz"
    cat.save_npz(p)
    with np.load(p, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    arrays["NHI"] = arrays["NHI"] * 1.001                            # the file edited, its record kept
    np.savez(str(p), **arrays)
    for load in (C.AbsorberCatalog.load_npz, lambda q: C.load_validated_catalog(q, 400)):
        with pytest.raises(C.CatalogProvenanceError, match="edited after it was made"):
            load(p)


@pytest.mark.parametrize("finder_ids, verified, why", [(["hcd_priya.finder/1"], True, "not in the release allow-list"),
                                                       ([C.FINDER_ID], False, "cross-check ran")])
def test_an_unknown_finder_id_or_an_unchecked_build_is_refused(tmp_path, finder_ids, verified, why):
    cat = _cat([17.5, 19.4])
    cat.absorbers[1].skewer_idx = 1
    _stamp(cat, finder_ids=finder_ids, verified=verified).save_npz(tmp_path / "c.npz")
    with pytest.raises(C.CatalogProvenanceError, match="cached catalogue refused .*" + why):
        C.load_validated_catalog(tmp_path / "c.npz", N)


def test_an_old_catalogue_is_read_only_through_the_historical_loader(tmp_path, caplog):
    old = _cat([17.5, 19.4])
    old.absorbers[1].skewer_idx, old.absorbers[1].pix_start, old.absorbers[1].pix_end = 0, N - 5, N + 9
    old.absorbers[0].skewer_idx, old.absorbers[0].pix_start, old.absorbers[0].pix_end = 0, 0, 9   # its own head
    p = tmp_path / "catalog.npz"
    _legacy_save(old, p)
    with pytest.raises(C.CatalogProvenanceError, match=r"no lineage record.*load_catalog_historical"):
        C.AbsorberCatalog.load_npz(p)
    for reason in ("", "  ", None):
        with pytest.raises(ValueError, match="non-empty reason"):
            C.load_catalog_historical(p, reason=reason)
    h = C.load_catalog_historical(p, reason="synthetic reproduction")
    assert "synthetic reproduction" in caplog.text
    assert len(h.absorbers) == 2 and h.lineage is None and not h.release_eligible    # read as it is: duplicates kept
    assert h.historical_reason == "synthetic reproduction"
    with pytest.raises(C.CatalogProvenanceError, match="read as historical"):
        h.save_npz(tmp_path / "again.npz")
    with pytest.raises(C.CatalogProvenanceError, match="read as historical"):
        C.check_catalog_record(h, N)
    assert not (tmp_path / "again.npz").exists()
    stamped = _stamp(_cat([17.5]))                                        # a stamped one read historically ...
    stamped.save_npz(tmp_path / "s.npz")
    hs = C.load_catalog_historical(tmp_path / "s.npz", reason="audit")
    assert hs.lineage == stamped.lineage and not hs.release_eligible      # ... keeps its record, stays historical
