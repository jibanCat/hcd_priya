"""Tests for the STAGE-V conditional-P1D truth-table tooling (pure functions only).

Covers: convention flags + stamp fail-loud paths, the mixture-arithmetic identities
(coarse merge, B_eff/unfiltered reconstruction, at-least-one bookkeeping), the deployed
P1D math (p1d_rows vs a brute-force DFT + a known tone), the segment (pixel-exclusion)
estimator including its agreement with the transplanted-window division estimator on
synthetic data with a known conditional signal, transplant determinism, KS-grid parsing
against the deployed loader, and the pool refusal paths (sha tamper, smoke member,
convention drift).

No fake_spectra required: only the numpy-pure functions of
scripts/build_xsel_truth_tables.py (module import is fake_spectra-free by design)."""
import json
import os

import numpy as np
import pytest

from scripts.build_xsel_truth_tables import (
    CLASS_RANGES, CURVE_KEYS, SCALAR_KEYS, SEG_KEYS, N_FINE,
    atleast_one_combine, bin_weighted, build_convention, coarse_merge,
    combine_unfiltered, dflux_rows, edges_from_centers_geom, interp_to_grid, ks_project,
    ks_leg_kgrid, mask_to_segments, native_kgrid, p1d_rows, powerspectrum_rows,
    segment_power_samples, sha256_file, snap_rng, stamp_sidecar, transplant_apply,
    trig_counts_by_fine_bin, verify_sidecar,
)
from scripts.analyze_xsel_truth_tables import null_fail_prob

RNG = np.random.default_rng(20260724)


# ---------------------------------------------------------------------------
# mixture arithmetic
# ---------------------------------------------------------------------------
def test_coarse_merge_equals_row_level_class_mean():
    """Bin-level count-weighted merge == direct mean over the class's rows."""
    n_k = 12
    counts = RNG.integers(0, 30, size=N_FINE)
    counts[0] = 50
    row_powers, bins = [], []
    for c, n in enumerate(counts):
        p = RNG.uniform(0.5, 2.0, size=(n, n_k))
        row_powers.append(p)
        bins.extend([c] * n)
    bins = np.array(bins)
    P_by_bin = np.stack([row_powers[c].mean(axis=0) if counts[c] else np.zeros(n_k)
                         for c in range(N_FINE)])
    merged = coarse_merge(P_by_bin, counts)
    allrows = np.concatenate([p for p in row_powers if len(p)], axis=0)
    for name, (lo, hi) in CLASS_RANGES.items():
        m = (bins >= lo) & (bins < hi)
        if m.sum():
            np.testing.assert_allclose(merged[name], allrows[m].mean(axis=0),
                                       rtol=1e-12)
        else:
            assert (merged[name] == 0).all()


def test_combine_unfiltered_identity():
    """P_unf reconstruction == direct unfiltered mean (the B_eff identity)."""
    n, n_trig, n_k = 40, 7, 9
    unf = RNG.uniform(0.5, 2.0, size=(n, n_k))
    filt = unf.copy()
    filt[:n_trig] = RNG.uniform(0.1, 1.0, size=(n_trig, n_k))  # filter alters trig rows
    P_filt = filt.mean(axis=0)
    rec = combine_unfiltered(P_filt, n, filt[:n_trig].mean(axis=0),
                             unf[:n_trig].mean(axis=0), n_trig)
    np.testing.assert_allclose(rec, unf.mean(axis=0), rtol=1e-12)
    # n_trig = 0 is a copy of P_filt
    np.testing.assert_array_equal(combine_unfiltered(P_filt, n, None, None, 0), P_filt)
    with pytest.raises(ValueError):
        combine_unfiltered(P_filt, n, P_filt, P_filt, n + 1)


def test_atleast_one_combine_bookkeeping():
    """Disjoint-part combination == direct mean over the at-least-one row union."""
    a = RNG.uniform(0.5, 2.0, size=(30, 8))     # partition-subDLA rows
    b = RNG.uniform(0.5, 2.0, size=(11, 8))     # DLA rows hosting a subDLA
    combo = atleast_one_combine([(30, a.mean(axis=0)), (11, b.mean(axis=0))])
    np.testing.assert_allclose(combo, np.concatenate([a, b]).mean(axis=0), rtol=1e-12)
    with pytest.raises(ValueError):
        atleast_one_combine([(0, a.mean(axis=0))])


def test_trig_counts_by_fine_bin():
    cls = np.array([0, 0, 3, 3, 9, 14, 14])
    trig = np.array([False, True, True, False, True, True, True])
    out = trig_counts_by_fine_bin(cls, trig)
    assert out[0] == 1 and out[3] == 1 and out[9] == 1 and out[14] == 2
    assert out.sum() == trig.sum()


# ---------------------------------------------------------------------------
# deployed P1D math
# ---------------------------------------------------------------------------
def test_p1d_rows_matches_bruteforce_and_tone():
    npix, dv = 256, 10.0
    vmax = npix * dv
    tau = RNG.uniform(0.1, 2.0, size=(5, npix))
    scale, tF = 0.9, 0.7
    got = p1d_rows(tau, vmax, scale, tF, chunk=2)
    # brute force: vmax * mean |rfft(dflux)|^2 / npix^2 (fake_spectra convention)
    d = np.exp(-scale * tau) / tF - 1.0
    ref = vmax * (np.abs(np.fft.rfft(d, axis=1)) ** 2 / npix ** 2).mean(axis=0)
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    # chunking invariance
    np.testing.assert_allclose(got, p1d_rows(tau, vmax, scale, tF, chunk=5000),
                               rtol=1e-12)
    # a pure tone in delta space lands its power in exactly one k bin:
    # delta = A cos(2 pi m j / npix) -> P[m] = vmax * A^2 / 4
    m, A = 7, 0.05
    j = np.arange(npix)
    delta = A * np.cos(2 * np.pi * m * j / npix)
    tau_tone = -np.log(tF * (delta + 1.0))     # exp(-tau)/tF - 1 == delta at scale=1
    P = p1d_rows(tau_tone[None, :], vmax, 1.0, tF)
    assert abs(P[m] - vmax * A ** 2 / 4) < 1e-10 * vmax
    off = np.delete(P[1:], m - 1)
    assert np.abs(off).max() < 1e-14 * vmax
    # empty input -> zeros
    assert (p1d_rows(np.empty((0, npix)), vmax, 1.0, tF) == 0).all()


def test_native_kgrid():
    k = native_kgrid(1196, 11960.0, 172)
    assert k.shape == (172,)
    np.testing.assert_allclose(k[0], 2 * np.pi / 11960.0, rtol=1e-15)
    with pytest.raises(ValueError):
        native_kgrid(100, 1000.0, 172)


# ---------------------------------------------------------------------------
# segments / pixel-exclusion estimator
# ---------------------------------------------------------------------------
def test_mask_to_segments_cases():
    n = 20
    assert mask_to_segments(np.zeros(n, bool)) == [(0, n)]
    assert mask_to_segments(np.ones(n, bool)) == []
    m = np.zeros(n, bool); m[5:9] = True
    segs = mask_to_segments(m)
    assert sorted(segs, key=lambda s: s[1]) == [(9, 16)]  # wraps 9..4
    m2 = np.zeros(n, bool); m2[0:3] = True; m2[10:12] = True
    segs2 = dict(mask_to_segments(m2))
    assert segs2 == {3: 7, 12: 8}
    # total unmasked pixels preserved
    assert sum(L for _, L in mask_to_segments(m2)) == int((~m2).sum())


def test_segment_power_tone_recovery():
    """A tone with integer periods inside the surviving segment is recovered at its k."""
    n, dv = 2048, 10.0
    mask = np.zeros(n, bool); mask[:1024] = True         # one 1024-pixel segment
    L = 1024
    m_seg, A = 16, 0.2                                    # 16 periods in the segment
    j = np.arange(n)
    delta = A * np.cos(2 * np.pi * m_seg * ((j - 1024) % n) / L)
    k, P, w = segment_power_samples(delta, mask, dv, lmin=128)
    assert k.size == L // 2 and (w == L).all()
    k_tone = 2 * np.pi * m_seg / (L * dv)
    i = int(np.argmin(np.abs(k - k_tone)))
    assert abs(k[i] - k_tone) < 1e-12
    np.testing.assert_allclose(P[i], (L * dv) * A ** 2 / 4, rtol=1e-10)
    # lmin filter drops short segments entirely
    k2, P2, w2 = segment_power_samples(delta, ~mask, dv, lmin=2048)
    assert k2.size == 0


def test_estimator_agreement_on_synthetic_conditional_signal():
    """Round-2 cosmo MUST-FIX machinery check: on synthetic rows whose 'selected'
    ensemble carries a known flat conditional power boost, the transplanted-window
    DIVISION estimator and the direct PIXEL-EXCLUSION estimator both recover the
    boost at low k, and agree with each other."""
    rng = np.random.default_rng(7)
    n_rows, npix, dv = 300, 1024, 10.0
    boost = 1.2                                           # amplitude x1.2 -> power x1.44
    # smooth GRF delta rows
    kmag = np.fft.rfftfreq(npix)
    amp = np.where(kmag > 0, (kmag + 0.02) ** -0.15, 0.0)

    def grf(n):
        ph = rng.normal(size=(n, npix // 2 + 1)) + 1j * rng.normal(size=(n, npix // 2 + 1))
        return 0.05 * np.fft.irfft(ph * amp, n=npix, axis=1) * np.sqrt(npix)

    sel = boost * grf(n_rows)
    ctrl = grf(n_rows)
    # windows: one ~150-pixel masked window per selected row
    masks = []
    for i in range(n_rows):
        m = np.zeros(npix, bool)
        s = int(rng.integers(0, npix)); L = int(rng.integers(120, 180))
        m[(s + np.arange(L)) % npix] = True
        masks.append(m)
    # division estimator: zero-fill windows, cyclic power, divide by the matched
    # transplanted control (same windows on ctrl rows)
    def cyc_power(rows, ms):
        r = rows.copy()
        for i, m in enumerate(ms):
            r[i][m] = 0.0
        return powerspectrum_rows(r).mean(axis=0)

    P_sel = cyc_power(sel, masks)
    P_ctrl = cyc_power(ctrl, masks)
    div_ratio = P_sel[1:40] / P_ctrl[1:40]
    # exclusion estimator on the same rows/masks
    edges = np.geomspace(2 * np.pi / (npix * dv) * 2, 2 * np.pi / (npix * dv) * 40, 10)

    def excl(rows, ms):
        ks, ps, ws = [], [], []
        for i in range(n_rows):
            k_s, p_s, w_s = segment_power_samples(rows[i], ms[i], dv, lmin=128)
            ks.append(k_s); ps.append(p_s); ws.append(w_s)
        return bin_weighted(np.concatenate(ks), np.concatenate(ps),
                            np.concatenate(ws), edges)

    excl_ratio = excl(sel, masks) / excl(ctrl, masks)
    target = boost ** 2
    assert abs(np.nanmean(div_ratio) / target - 1) < 0.07
    assert abs(np.nanmean(excl_ratio) / target - 1) < 0.07
    assert abs(np.nanmean(excl_ratio) / np.nanmean(div_ratio) - 1) < 0.07


def test_bin_weighted_and_edges():
    c = np.array([1.0, 2.0, 4.0, 8.0])
    e = edges_from_centers_geom(c)
    assert e.shape == (5,) and (np.diff(e) > 0).all()
    assert e[0] < c[0] < e[1] and e[-2] < c[-1] < e[-1]
    out = bin_weighted(np.array([1.0, 1.1, 4.0]), np.array([2.0, 4.0, 10.0]),
                       np.array([1.0, 3.0, 1.0]), e)
    np.testing.assert_allclose(out[0], (2.0 + 12.0) / 4.0)
    np.testing.assert_allclose(out[2], 10.0)
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert np.isnan(bin_weighted(np.array([]), np.array([]), np.array([]), e)).all()
    with pytest.raises(ValueError):
        edges_from_centers_geom(np.array([2.0, 1.0]))


# ---------------------------------------------------------------------------
# transplants / rng / interp
# ---------------------------------------------------------------------------
def test_transplant_apply_deterministic_and_filled():
    npix = 200
    tau = np.abs(RNG.uniform(0.1, 1.0, size=(50, npix)))
    donors = []
    for L in (10, 20, 30):
        m = np.zeros(npix, bool); m[:L] = True
        donors.append(m)
    r1 = snap_rng(1, "simA", 5)
    out1, ap1 = transplant_apply(tau, donors, 3.14, 0.6, r1)
    r2 = snap_rng(1, "simA", 5)
    out2, ap2 = transplant_apply(tau, donors, 3.14, 0.6, r2)
    np.testing.assert_array_equal(out1, out2)
    n_applied = sum(a is not None for a in ap1)
    assert 10 <= n_applied <= 50                     # p=0.6, n=50: loose bounds
    for i, a in enumerate(ap1):
        if a is None:
            np.testing.assert_array_equal(out1[i], tau[i])
        else:
            assert (out1[i][a] == 3.14).all()
            np.testing.assert_array_equal(out1[i][~a], tau[i][~a])
    # different (sim, snap) -> different stream
    out3, _ = transplant_apply(tau, donors, 0.6, 0.6, snap_rng(1, "simB", 5))
    assert not np.array_equal(out1, out3)
    # no donors -> untouched
    out4, ap4 = transplant_apply(tau, [], 3.14, 1.0, snap_rng(1, "simA", 5))
    np.testing.assert_array_equal(out4, tau)
    assert all(a is None for a in ap4)


def test_interp_bounds_fail_loud():
    k = np.linspace(0.001, 0.1, 50)
    P = np.ones(50)
    np.testing.assert_allclose(interp_to_grid(k, P, np.array([0.01, 0.05])), 1.0)
    with pytest.raises(ValueError):
        interp_to_grid(k, P, np.array([0.2]))


def test_cell_denylist_pinned():
    # the one verified bad deployed pairing (2026-07-25 label audit) stays pinned;
    # an accidental clear of the denylist would resurrect the tripwire failure
    from scripts.build_xsel_truth_tables import XSEL_CELL_DENYLIST
    assert ("ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144"
            "hireionz7.47bhfeedback0.0347", 17) in XSEL_CELL_DENYLIST
    assert all(isinstance(s, str) and isinstance(n, int)
               for s, n in XSEL_CELL_DENYLIST)


def test_ks_project_z_window_gate():
    # regression for job 54771172: at z=5.4 (outside the KS window) some sims'
    # native Nyquist falls below the last KS bin centre -- NaN, not a refusal
    k_ks = np.array([0.0055, 0.03, 0.0627126])
    k_short = np.linspace(0.00036, 0.0618, 40)   # under-covers the last KS bin
    k_full = np.linspace(0.0003, 0.08, 40)
    P = np.ones(40)
    # out-of-window + under-coverage -> all-NaN row, no raise
    out = ks_project(k_short, P, k_ks, z_grid=5.4)
    assert out.shape == k_ks.shape and np.isnan(out).all()
    # out-of-window below the window behaves the same
    assert np.isnan(ks_project(k_short, P, k_ks, z_grid=2.2)).all()
    # in-window + full coverage -> interp passthrough
    np.testing.assert_allclose(ks_project(k_full, P, k_ks, z_grid=3.0), 1.0)
    # in-window + under-coverage -> still fail-loud (would corrupt the tables)
    with pytest.raises(ValueError):
        ks_project(k_short, P, k_ks, z_grid=4.6)
    # in-window + non-finite curve -> NaN (pre-existing behavior preserved)
    Pbad = P.copy()
    Pbad[3] = np.nan
    assert np.isnan(ks_project(k_full, Pbad, k_ks, z_grid=3.0)).all()


def test_dflux_and_powerspectrum_conventions():
    tau = np.array([[0.0, 0.5, 1.0, 2.0]])
    d = dflux_rows(tau, 1.0, 0.7)
    np.testing.assert_allclose(d, np.exp(-tau) / 0.7 - 1.0)
    p = powerspectrum_rows(np.ones((2, 8)))
    np.testing.assert_allclose(p[:, 0], 1.0)         # DC of ones: |8|^2/64
    assert (p[:, 1:] < 1e-30).all()


# ---------------------------------------------------------------------------
# stamps / conventions / pool refusals
# ---------------------------------------------------------------------------
def test_stamp_roundtrip_and_tamper_detection(tmp_path):
    npz = tmp_path / "x.npz"
    np.savez(npz, a=np.arange(3))
    side = stamp_sidecar(npz, dict(kind="test", sim="s", full_suite=True))
    st = verify_sidecar(npz)
    assert st["kind"] == "test" and st["npz_sha256"] == sha256_file(npz)
    assert os.path.exists(side)
    np.savez(npz, a=np.arange(4))                    # tamper
    with pytest.raises(RuntimeError, match="sha mismatch"):
        verify_sidecar(npz)
    with pytest.raises(FileNotFoundError):
        verify_sidecar(tmp_path / "missing.npz")


def test_convention_flags_pinned():
    c = build_convention("lf", 10, 1.011182770662847, 20260724)
    assert c["frame"] == "deployed"
    assert c["n_k"] == 172
    assert "1e+06" in c["filter"] and "0.25" in c["filter"]
    assert c["wide_thresh2"] == 0.125
    assert "highest-class" in c["partition"]
    assert c["ks_cuts"] == {"z_lo": 2.4, "z_hi": 4.6, "k_max": 0.069}
    ch = build_convention("hr", 10, 1.011182770662847, 20260724)
    assert ch["n_k"] == 525 and ch != c


def _write_member(outdir, fidelity, sim, z, conv, full=True, n_k=6, n_ks=11):
    """Minimal synthetic per-(sim,z) member npz + stamp for pool tests."""
    from scripts.build_xsel_truth_tables import stamp_sidecar as _stamp
    d = {"k_native": np.linspace(0.001, 0.06, n_k), "k_ks": np.linspace(0.005, 0.06, n_ks),
         "inc_host_class": np.zeros((5, 3)), "trig_by_fine_bin": np.zeros(15, np.int64)}
    for key in CURVE_KEYS:
        d[key] = np.ones(n_k); d[key + "_ks"] = np.ones(n_ks)
    for key in SEG_KEYS:
        d[key] = np.ones(n_ks)
    for key in SCALAR_KEYS:
        d[key] = np.float64(1.0)
    d["z_grid"] = np.float64(z); d["snap"] = np.int64(9)
    p = outdir / fidelity / sim
    p.mkdir(parents=True, exist_ok=True)
    npz = p / f"xsel_z{z:.1f}.npz"
    np.savez(npz, **d)
    _stamp(npz, dict(kind="xsel_truth_snap", sim=sim, snap=9, z_grid=z,
                     fidelity=fidelity, full_suite=full, convention=conv))
    return npz


def test_pool_suite_and_refusals(tmp_path):
    from scripts.build_xsel_truth_tables import pool_suite
    conv = build_convention("lf", 10, 1.011, 1)
    _write_member(tmp_path, "lf", "simA", 2.6, conv)
    _write_member(tmp_path, "lf", "simA", 2.8, conv)
    _write_member(tmp_path, "lf", "simB", 2.6, conv)
    out = pool_suite(tmp_path, "lf")
    st = verify_sidecar(out)
    assert st["n_members"] == 3
    d = np.load(out)
    assert d["P_filt_dla"].shape == (3, 6) and d["z_grid"].shape == (3,)
    assert set(d["sim"]) == {"simA", "simB"}
    # refusal 1: smoke member
    _write_member(tmp_path, "lf", "simC", 2.6, conv, full=False)
    with pytest.raises(RuntimeError, match="smoke/subset"):
        pool_suite(tmp_path, "lf")
    os.remove(tmp_path / "lf/simC/xsel_z2.6.npz")
    os.remove(tmp_path / "lf/simC/xsel_z2.6.npz.stamp.json")
    # refusal 2: convention drift
    conv2 = dict(conv); conv2["wide_thresh2"] = 0.5
    _write_member(tmp_path, "lf", "simD", 2.6, conv2)
    with pytest.raises(RuntimeError, match="convention flags differ"):
        pool_suite(tmp_path, "lf")
    os.remove(tmp_path / "lf/simD/xsel_z2.6.npz")
    os.remove(tmp_path / "lf/simD/xsel_z2.6.npz.stamp.json")
    # refusal 3: sha tamper
    npz = tmp_path / "lf/simA/xsel_z2.6.npz"
    data = dict(np.load(npz)); data["P_filt_dla"] = data["P_filt_dla"] * 2
    np.savez(npz, **data)
    with pytest.raises(RuntimeError, match="sha mismatch"):
        pool_suite(tmp_path, "lf")


# ---------------------------------------------------------------------------
# KS grid + gate-power MC
# ---------------------------------------------------------------------------
KS_FILE = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/" \
          "final-conservative-p1d-karacayli_etal2021.txt"


@pytest.mark.skipif(not os.path.exists(KS_FILE), reason="KS data files not present")
def test_ks_kgrid_matches_deployed_loader():
    ku = ks_leg_kgrid()
    assert ku.shape == (11,)
    assert ku[0] == pytest.approx(0.0055) and ku[-1] <= 0.069
    from hcd_analysis.emulator import data_likelihood as DL
    leg = DL.load_ks_leg()
    np.testing.assert_array_equal(ku, np.unique(np.asarray(leg.k, float)))


def test_emit_contract_roundtrip_consumer_validated(tmp_path):
    """emit_contract writes the X-battery contract npz and it passes the CONSUMER'S
    validator (ks_xsel_arms.load_truth_tables) at the fresh sha: keys, shapes,
    positivity, convention equality, 8 gate-power entries."""
    from types import SimpleNamespace
    import json as _json
    from scripts.analyze_xsel_truth_tables import emit_contract
    import scripts.ks_xsel_arms as XA

    n_z, k_ks = 12, np.geomspace(0.0055, 0.0627, 11)
    zs = np.round(np.arange(2.4, 4.61, 0.2), 1)
    leg_k = np.tile(k_ks, n_z)
    leg_z = np.repeat(zs, k_ks.size)
    N = leg_k.size
    leg = SimpleNamespace(k=leg_k, z=zs, z_idx=np.repeat(np.arange(n_z), k_ks.size),
                          C_data=np.eye(N) * 0.04, P_data=np.ones(N))
    d = {"hr_z_grid": zs.astype(float)}
    curves = {"P_filt_clean_ks": 1.0, "X1_corrected_ks": 0.8, "P_filt_dla_ks": 0.55,
              "P_filt_sub_ks": 0.9, "P_filt_lls_ks": 1.1, "X1_corrected_wide_ks": 0.82}
    for key, val in curves.items():
        d[f"hr_{key}"] = np.full((n_z, k_ks.size), val)
    sigma_post = {"ns": dict(pooled_rms=0.14), "Ap": dict(pooled_rms=0.13)}
    out = tmp_path / "xsel_truth_tables.npz"
    path, sha = emit_contract(d, leg, k_ks, [float(z) for z in zs], sigma_post,
                              n_pairs=16, gate=0.30, r_floor=0.05, n_mc=5000,
                              out_path=out)
    tt = XA.load_truth_tables(path=path, expect_sha=sha)   # consumer validator
    assert tt["leg_k"].shape == (N,)
    np.testing.assert_allclose(tt["ratios"]["ratio_rows_X1_dla100"], 0.8)
    np.testing.assert_allclose(tt["ratios"]["ratio_rows_X1b_dla100_diluted"], 0.55)
    lo, hi = tt["band"]
    assert (lo <= hi).all() and np.allclose(lo, 0.8) and np.allclose(hi, 0.82)
    assert set(tt["gate_power"]) >= set(XA.GATE_POWER_ARMS)
    for aid in XA.GATE_POWER_ARMS:
        gp = tt["gate_power"][aid]
        assert gp["D"] > 0 and gp["sigma_pair_expected"] > 0
        assert 0.0 <= gp["p_part1_fail_null"] <= 1.0
    # K8 arms: in-manifold convention (D = 1 prior sigma, coherent floor)
    assert tt["gate_power"]["K8a_eps_hi"]["D"] == 1.0
    assert tt["gate_power"]["K8a_eps_hi"]["sigma_pair_expected"] == 0.05
    # X4 derived-curve entry differs from X3 (profile scaling)
    assert tt["gate_power"]["X4_prof"]["D"] < tt["gate_power"]["X3_lls100"]["D"]
    # convention: REQUIRED entries exact + provenance extras allowed
    for ck, cv in XA.REQUIRED_CONVENTION.items():
        assert tt["convention"][ck] == cv
    assert "d_convention" in tt["convention"]


def test_null_fail_prob_monotone_and_limits():
    p0 = null_fail_prob(1e-6, 16, 0.30, n_mc=20000)
    p1 = null_fail_prob(0.30, 16, 0.30, n_mc=20000)
    p2 = null_fail_prob(1.414, 16, 0.30, n_mc=20000)
    assert p0 == 0.0
    assert p0 < p1 < p2
    assert p2 > 0.5
    # near the analytic 5% threshold r* ~ 0.33 (|mean|+2SE vs 0.3 sigma_post at n=16)
    p_star = null_fail_prob(0.33, 16, 0.30, n_mc=100000)
    assert 0.01 < p_star < 0.15
