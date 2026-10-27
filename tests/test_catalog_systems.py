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
import types

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


# ----------------------------------------------------------------------------- the notice's regression cases
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


def test_pipeline_writes_reconciled_counts_and_refuses_a_mismatch(tmp_path):
    pytest.importorskip("yaml")                                  # the pipeline's config module needs PyYAML
    from hcd_analysis import pipeline as PL
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
    assert not (bad / "meta.json").exists()
