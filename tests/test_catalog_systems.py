"""The absorber system finder and the catalogue integrity guards (docs/bugs_found.md, the box-edge duplicate).

The pre-fix find_systems_in_skewer listed every system that straddles the box edge twice: once whole (pix_end >= n) and
once as its own head [s', pix_end - n]. The guards make that class of error impossible to write silently:
  * an independent second finder (find_systems_in_skewer_reference) must agree with the production finder on every
    skewer (process_skewer_batch refuses otherwise);
  * no two catalogue entries of one skewer may share a pixel modulo n (validate_catalog_entries, run by build_catalog);
  * the stored class counts must reconcile three ways (labels, NHI values, the CDDF histogram) before meta.json is
    written (reconcile_counts, used by the pipeline).
Synthetic optical depths only."""
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import catalog as C

N, GAP, MINPIX, THR = 1342, 9, 2, 100.0


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
    cat.save_npz(tmp_path / "nan.npz")
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused .*non-finite N_HI"):
        C.load_validated_catalog(tmp_path / "nan.npz", N)


def test_a_cached_pre_fix_catalogue_is_refused_on_reuse(tmp_path):
    good = _cat([17.5, 19.4])
    good.absorbers[1].skewer_idx, good.absorbers[1].pix_start, good.absorbers[1].pix_end = 0, N - 5, N + 9
    good.absorbers[0].skewer_idx, good.absorbers[0].pix_start, good.absorbers[0].pix_end = 0, 100, 120
    good.save_npz(tmp_path / "good.npz")
    assert len(C.load_validated_catalog(tmp_path / "good.npz", N).absorbers) == 2
    bad = _cat([17.5, 19.4])
    bad.absorbers[1].skewer_idx, bad.absorbers[1].pix_start, bad.absorbers[1].pix_end = 0, N - 5, N + 9
    bad.absorbers[0].skewer_idx, bad.absorbers[0].pix_start, bad.absorbers[0].pix_end = 0, 0, 9   # its own head
    bad.save_npz(tmp_path / "bad.npz")
    with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
        C.load_validated_catalog(tmp_path / "bad.npz", N)


# ----------------------------------------------------------------------------- the pipeline's resume paths
def _pipeline(monkeypatch):
    """hcd_analysis.pipeline, importable without PyYAML: the config module imports it at load time, but the snapshot
    path exercised here never calls it, so a stub stands in when it is absent. The stub is removed from sys.modules at
    teardown, but hcd_analysis.config stays cached with it bound; any use of it raises ImportError, as a missing
    PyYAML would."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        stub = types.ModuleType("yaml")

        def _absent(name):
            raise ImportError(f"PyYAML is not installed (yaml.{name} requested; stub from tests/test_catalog_systems.py)")
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
    # the refused catalogues: the pre-fix one, and a valid one damaged to hold a non-finite N_HI
    nan_cat = C.AbsorberCatalog.load_npz(d0 / "catalog.npz")
    nan_cat.absorbers[1].NHI = float("nan")
    # not done, with a cached refused catalogue: the snapshot fails before anything else is written (no per-class
    # P1D, no meta.json, not marked done)
    for name, bad in (("cached", _pre_fix_catalogue(PL, hdr, nb)), ("cached_nan", nan_cat)):
        c1 = cfg(name)
        d1 = PL.snap_output_dir(c1, "synthsim", 1)
        bad.save_npz(d1 / "catalog.npz")
        assert PL.run_one_snap(ss, entry, c1) is None
        assert sorted(p.name for p in d1.iterdir()) == ["catalog.npz", "error.txt"]
        assert not PL.is_done(c1, "synthsim", 1)
        assert "cached catalogue refused" in (d1 / "error.txt").read_text()
    # done, with a refused catalogue: refused loudly, never reused, nothing written, and the simulation's other
    # snapshot (which on its own is stacked: control below) is not stacked without it
    c2 = cfg("done")
    d2 = PL.snap_output_dir(c2, "synthsim", 1)
    bad = _pre_fix_catalogue(PL, hdr, nb)
    bad.save_npz(d2 / "catalog.npz")
    PL._save_snap_outputs(d2, bad, {}, {}, PL.measure_cddf(bad, hdr), None, None, {}, {}, hdr, "synthsim", 1, 3.0,
                          PL.pixel_dv_kms(hdr))
    PL.mark_done(c2, "synthsim", 1)
    c3 = cfg("done_nan")
    d3 = PL.snap_output_dir(c3, "synthsim", 1)
    shutil.copytree(d0, d3, dirs_exist_ok=True)                    # a completed snapshot ...
    nan_cat.save_npz(d3 / "catalog.npz")                           # ... whose catalogue was then damaged
    for c, d in ((c2, d2), (c3, d3)):
        before = files(d)
        with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
            PL.run_one_snap(ss, entry, c)
        with pytest.raises(C.CatalogIntegrityError, match="cached catalogue refused"):
            PL.run_sim_all_z(ss2, c)
        assert files(d) == before
        assert not (Path(c.output_root) / "synthsim" / "cddf_stacked.npz").exists()
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
