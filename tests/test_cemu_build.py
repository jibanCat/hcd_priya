"""Gate E amendment A1 building blocks (hcd_analysis/emulator/cemu_build.py): simulation-level folds, per-cell second
moments of the ensemble residuals, PSD-preserving smoothing along ln(mode) and z, the Gaussian predictive score, and
the modes that bracket the data bins over the sampling box. Synthetic inputs."""
import numpy as np
import pytest

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def test_folds_are_by_simulation_sorted_names_mod_10():
    sims = np.array([f"s{i:02d}" for i in range(25)] * 3)
    f = CB.cv_folds(sims, n_folds=10)
    names = sorted(set(sims))
    assert all(f[n] == i % 10 for i, n in enumerate(names))


def test_second_moment_per_cell_is_the_uncentered_mean_outer_product():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (50, 4, 6))
    cell = np.repeat([0, 1], 25)
    rho, n = CB.second_moment_cells(r, cell, 2)
    assert rho.shape == (2, 4, 4, 6) and list(n) == [25, 25]
    np.testing.assert_allclose(rho[1, :, :, 3], np.einsum("rc,rd->cd", r[25:, :, 3], r[25:, :, 3]) / 25, rtol=1e-13)


def test_mode_smoothing_keeps_psd_and_h0_is_identity():
    rng = np.random.default_rng(1)
    A = rng.normal(0, 0.01, (3, 4, 4, K))
    rho = np.einsum("zcek,zdek->zcdk", A, A)
    np.testing.assert_array_equal(CB.smooth_modes(rho, 0.0), rho)
    s = CB.smooth_modes(rho, 0.2)
    for z in range(3):
        for k in (0, 50, 171):
            assert np.linalg.eigvalsh(s[z, :, :, k]).min() >= -1e-18
    const = np.ones((1, 4, 4, K))
    np.testing.assert_allclose(CB.smooth_modes(const, 0.3), const, rtol=1e-13)      # a constant is unchanged


def test_z_smoothing_is_a_normalised_kernel():
    z = np.arange(2.2, 4.61, 0.2)
    rho = np.ones((z.size, 4, 4, 5)) * z[:, None, None, None]
    np.testing.assert_array_equal(CB.smooth_z(rho, z, 0.0), rho)
    s = CB.smooth_z(rho, z, 0.2)
    np.testing.assert_allclose(s[6], rho[6], rtol=1e-12)        # linear in z: interior unchanged by a symmetric kernel


def test_gaussian_score_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, (7, 4, 3))
    A = rng.normal(0, 0.01, (4, 4, 3)); cov = np.einsum("cek,dek->cdk", A, A) + 1e-6 * np.eye(4)[:, :, None]
    got = CB.gaussian_score(r, np.broadcast_to(cov, (7, 4, 4, 3)), jitter=0.0)   # scipy has no jitter
    expect = sum(multivariate_normal(np.zeros(4), cov[:, :, k], allow_singular=False).logpdf(r[i, :, k])
                 for i in range(7) for k in range(3))
    np.testing.assert_allclose(got, expect, rtol=1e-10)


def test_bracketing_modes_cover_every_bin_over_the_box():
    k_lo, k_hi = 1.2e-3, 0.06
    lo, hi = CB.bracket_modes(KCOM, 3.0, k_lo, k_hi, np.zeros(9), np.ones(9))
    from hcd_analysis.emulator.kcoord import kbounds_over_box, k_skm_from_kcom
    from hcd_analysis.emulator.data import PARAM_LIMITS
    lim = np.asarray(PARAM_LIMITS, float)
    for hub in np.linspace(lim[5, 0], lim[5, 1], 7):
        for om in np.linspace(lim[6, 0], lim[6, 1], 7):
            k = np.asarray(k_skm_from_kcom(KCOM, 3.0, hub, om))
            u_lo, u_hi = k_lo / k[0], k_hi / k[0]
            assert lo <= int(np.floor(u_lo)) and int(np.ceil(u_hi)) <= hi


EVAL = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateC_eval"


def test_loo_ensemble_residuals_are_member_means_on_the_held_out_rows():
    import os
    from tests.gate_helpers import real_cache_path
    from hcd_analysis.emulator.data import load_cache
    if not os.path.exists(f"{EVAL}/eval_loo60_s00.npz"):
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail("gate C evaluations absent")
        pytest.skip("gate C evaluations absent")
    d = load_cache(real_cache_path("lf"))
    R = CB.load_loo_ensemble_residuals(EVAL, d, sims=[0, 7])
    assert R["r"].shape == (720, 4, K) and np.all(np.isfinite(R["r"]))
    m = [np.load(f"{EVAL}/eval_loo60_s07.npz", allow_pickle=True)["res_cls"]]
    m += [np.load(f"{EVAL}/loo60_ensemble/eval_loo60_s07_seed{s}.npz", allow_pickle=True)["res_cls"] for s in range(1, 5)]
    np.testing.assert_allclose(R["r"][360:], np.mean(m, axis=0), rtol=1e-14)
    names = np.asarray(d["sim_name"]).astype(str)
    assert len(set(names[R["rows"][:360]])) == 1 and len(set(names[R["rows"][360:]])) == 1
    assert set(R["sim"][:360]) == {names[R["rows"][0]]}


# --------------------------------------------------------------------------------------------- #
#  T1 selection (amendment A1 rev 1 section 1): deployed tau0 interpolation, production class
#  coefficients, the production-combined score, paired simulation-bootstrap SE, the rules
# --------------------------------------------------------------------------------------------- #
def test_rho_interp_matches_the_deployed_rho_at_tau0():
    from hcd_analysis.emulator.likelihood import rho_at_tau0
    from hcd_analysis.emulator.data import KIM_AMP, KIM_SLOPE
    rng = np.random.default_rng(3)
    rho = rng.normal(0, 1e-4, (4, 4, 9, 4))
    centres = np.array([0.6, 0.9, 1.1, 1.5])
    z = 3.2
    for alpha in (0.4, 0.6, 0.75, 1.0, 1.49, 2.0):
        tau0 = alpha * KIM_AMP * (1 + z) ** KIM_SLOPE
        np.testing.assert_allclose(CB.rho_interp_alpha(rho, centres, np.array([alpha]))[0],
                                   np.asarray(rho_at_tau0(rho, centres, z, tau0)), rtol=1e-12, atol=1e-20)


def test_class_coefficients_are_production_with_and_without_the_dla_mask():
    w = np.array([[0.70, 0.10, 0.15, 0.05]])
    np.testing.assert_allclose(CB.class_coef(w, masked=False), [[0.70, 0.10, 0.15, 0.05]], rtol=1e-14)
    np.testing.assert_allclose(CB.class_coef(w, masked=True), [[0.75, 0.10, 0.15, 0.0]], rtol=1e-14)


def test_combined_variance_is_the_deployed_emu_var_algebra():
    from hcd_analysis.emulator import data_likelihood as DL
    rng = np.random.default_rng(4)
    Kc = 7
    A = rng.normal(0, 1e-2, (4, 4, Kc))
    rho = np.einsum("cek,dek->cdk", A, A)
    P = rng.uniform(0.5, 2.0, (4, Kc))
    a = np.array([0.1, 0.15, 0.05])
    coef = CB.class_coef(np.concatenate([[1 - a.sum()], a])[None], masked=False)[0]
    var_frac, P_obs = CB.combined_variance(P, coef, rho)
    ev = np.asarray(DL.emu_var_modes(P, 3.0, 1.0, a, dla_core=np.zeros(Kc), alpha_centres=np.array([1.0]),
                                     rho_zb=rho[..., None]))
    np.testing.assert_allclose(var_frac * P_obs ** 2, ev, rtol=1e-12)


def test_combined_score_is_the_1d_gaussian_of_the_combined_fractional_residual():
    from scipy.stats import norm
    rng = np.random.default_rng(5)
    Kc = 6
    A = rng.normal(0, 1e-2, (4, 4, Kc))
    rho = np.einsum("cek,dek->cdk", A, A)
    P = rng.uniform(0.5, 2.0, (4, Kc))
    r = rng.normal(0, 1e-2, (4, Kc))
    coef = np.array([0.7, 0.1, 0.15, 0.05])
    mask = np.array([True, True, False, True, True, False])
    got = CB.combined_logpdf(r, P, coef, rho, mask)
    P_obs = coef @ P
    e = (coef[:, None] * P * r).sum(0) / P_obs
    var = np.einsum("c,d,cdk,ck,dk->k", coef, coef, rho, P, P) / P_obs ** 2
    np.testing.assert_allclose(got, norm(0, np.sqrt(var[mask])).logpdf(e[mask]).sum(), rtol=1e-12)


def test_paired_bootstrap_se_resamples_simulations():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, 60)
    b = a + rng.normal(0.1, 0.05, 60)              # strongly paired: unpaired SE would be ~20x larger
    se = CB.paired_se(a, b, n_boot=2000, seed=0)
    assert abs(se / (np.std(a - b, ddof=1) / np.sqrt(60)) - 1) < 0.1
    assert se == CB.paired_se(a, b, n_boot=2000, seed=0)          # deterministic


def test_selection_is_argmax_with_ties_to_more_smoothing():
    means = np.array([-10.0, -9.0, -9.0 * (1 + 1e-8), -9.5])
    smooth_rank = np.array([0, 1, 2, 3])             # larger = more smoothing
    assert CB.select_argmax(means, smooth_rank) == 2
    assert CB.select_argmax(np.array([-3.0, -1.0, -2.0]), np.array([0, 1, 2])) == 1


def test_tau0_pooling_preferred_unless_banded_wins_by_two_paired_se():
    rng = np.random.default_rng(7)
    pooled = rng.normal(0, 1, 60)
    noise = rng.normal(0, 1, 60)
    assert CB.choose_tau0(pooled + noise - noise.mean() + 0.001, pooled, seed=0) == "pooled"   # tiny gain, SE ~ 0.13
    assert CB.choose_tau0(pooled + 1.0 + rng.normal(0, 0.01, 60), pooled, seed=0) == "banded"
    assert CB.choose_tau0(pooled - 1.0, pooled, seed=0) == "pooled"


def test_guard_acts_on_the_whole_z_cell_with_the_largest_h_within_one_se_of_raw():
    rng = np.random.default_rng(8)
    n_sim, n_h, n_z, n_band = 60, 4, 2, 3
    base = rng.normal(0, 1, (n_sim, 1, 1, 1))
    s = np.broadcast_to(base, (n_sim, n_h, n_z, n_band)).copy() + rng.normal(0, 1e-3, (n_sim, n_h, n_z, n_band))
    # z cell 1, band 2: raw (h index 0) beats everything else strongly; h index 1 within noise of raw
    s[:, 2:, 1, 2] -= 1.0
    s[:, 1, 1, 2] = s[:, 0, 1, 2] + 5e-4            # h index 1 slightly better than raw there: within 1 SE
    h_of_z = CB.guard(s, selected=3, n_boot=500, seed=0)
    assert list(h_of_z) == [3, 1]                    # z cell 0 untouched; z cell 1 takes h index 1 for ALL bands


# --------------------------------------------------------------------------------------------- #
#  T1 cross-validation procedure on synthetic residuals (behaviour, not numbers)
# --------------------------------------------------------------------------------------------- #
def _synthetic_t1(seed=0, n_sim=20, Ksyn=40, spike=None, tau0_dep=False):
    rng = np.random.default_rng(seed)
    z_cells = np.array([2.4, 3.0])
    alphas = np.array([0.7, 0.9, 1.1, 1.3])
    n = np.arange(1, Ksyn + 1)
    sig = 0.01 * (1 + 0.5 * np.log(n) / np.log(Ksyn))          # smooth in ln(mode)
    if spike is not None:
        sig = sig.copy()
        sig[spike] *= 5.0                                       # a real narrow feature, in every simulation
    L = np.linalg.cholesky(0.5 * np.eye(4) + 0.5 * np.ones((4, 4)))
    rows = []
    for s in range(n_sim):
        for iz in range(2):
            for a in alphas:
                amp = (a if tau0_dep else 1.0)
                eps = rng.normal(0, 1, (Ksyn, 4)) @ L.T
                rows.append((f"sim{s:02d}", iz, a, (eps * sig[:, None] * amp).T))
    sim = np.array([r[0] for r in rows])
    zc = np.array([r[1] for r in rows])
    alpha = np.array([r[2] for r in rows])
    r = np.stack([r[3] for r in rows])
    band = np.searchsorted(alphas, alpha)
    T = CB.T1Data(r=r, P=np.ones_like(r), coef=CB.class_coef(np.tile([0.7, 0.1, 0.15, 0.05], (len(rows), 1)), masked=True),
                  alpha=alpha, band=band, centres=alphas.copy(), zc=zc, sim=sim, z_cells=z_cells,
                  mask=np.ones((2, Ksyn), bool), kband=np.tile(np.repeat([0, 1, 2, 3], Ksyn // 4), (2, 1)))
    return T


CANDS_SYN = [(h, sz, pooled) for pooled in (True, False) for h in (0.0, 0.05, 0.2, 0.5) for sz in (0.0, 0.2)]


def test_t1_cv_prefers_smoothing_for_a_smooth_noisy_truth_and_pools_tau0():
    T = _synthetic_t1(seed=1)
    cv = CB.t1_cv(T, CANDS_SYN, n_folds=5)
    assert cv["main"].shape == (20, len(CANDS_SYN))
    sel = CB.t1_select(cv, CANDS_SYN, n_boot=500, seed=0)
    h, sz, pooled = CANDS_SYN[sel["chosen"]]
    assert h > 0 and pooled and sel["tau0"] == "pooled"


def test_t1_cv_flags_a_real_tau0_dependence():
    T = _synthetic_t1(seed=2, tau0_dep=True)
    sel = CB.t1_select(CB.t1_cv(T, CANDS_SYN, n_folds=5), CANDS_SYN, n_boot=500, seed=0)
    assert sel["tau0"] == "banded"                  # the caller STOPS for the PI on this (amendment A1 rev 1)


def test_t1_final_product_keeps_a_real_narrow_feature():
    T = _synthetic_t1(seed=3, spike=17)
    cv = CB.t1_cv(T, CANDS_SYN, n_folds=5)
    sel = CB.t1_select(cv, CANDS_SYN, n_boot=500, seed=0)
    rho = CB.t1_build(T, np.arange(len(T.sim)), CANDS_SYN, sel)            # (n_z, B, 4, 4, K)
    raw = CB.t1_build(T, np.arange(len(T.sim)), CANDS_SYN, dict(sel, chosen=CANDS_SYN.index((0.0, 0.0, True)),
                                                                h_of_z=[0.0, 0.0]))
    for iz in range(2):
        peak = rho[iz, 0, 0, 0, 17] / np.median(rho[iz, 0, 0, 0])
        assert peak > 0.5 * raw[iz, 0, 0, 0, 17] / np.median(raw[iz, 0, 0, 0])


def test_t1_cv_cell_split_sums_to_the_simulation_total():
    T = _synthetic_t1(seed=4, n_sim=10)
    cands = [(0.0, 0.0, True), (0.2, 0.0, False)]
    cv = CB.t1_cv(T, cands, n_folds=5)
    np.testing.assert_allclose(cv["cell"].sum(axis=(2, 3)), cv["main"], rtol=1e-12)
    assert cv["sims"] == sorted(set(T.sim))
