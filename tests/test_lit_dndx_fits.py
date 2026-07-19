"""TDD tests for the fit machinery of the corrected-law re-derivation (spec steps 3-4, 7).

Covers: the shared wls_powerlaw (refactored out of derive_hcd_lls_width.py) reproducing
ALL THREE deployed laws bit-for-bit from the tombstoned wrong-object arrays; the Poisson
GLM (log link, plain numpy) reproducing PW09 ~(0.0076, 1.592) and recovering synthetic
draws; the GLS with common-mode kernel covariance; the fit ordering + internal-consistency
and telescoping checks; and the sensitivity arms.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_lit_dndx_fits.py -q
"""
import subprocess
import sys

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator import lit_dndx as LD

PY = sys.executable
REPO = "/home/mfho/hcd_priya"


# --------------------------------------------------------------------------- #
#  Shared WLS: bit-for-bit regression vs the three DEPLOYED laws               #
# --------------------------------------------------------------------------- #
def test_wls_reproduces_deployed_lls_law_from_tombstone():
    t = LD.LLS_TAU2_OLD_DEFECTS
    fit = LD.wls_powerlaw(t["z"], t["lx"], t["err"])
    assert round(float(fit["A"]), 4) == 0.0201
    assert round(float(fit["gamma"]), 3) == 2.127


def test_wls_reproduces_deployed_subdla_law_from_tombstone():
    t = LD.ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL
    fit = LD.wls_powerlaw(t["z"], t["lx"], t["err"])
    assert round(float(fit["A"]), 4) == 0.0211
    assert round(float(fit["gamma"]), 3) == 0.937


def test_wls_reproduces_deployed_dla_law():
    a = LD.ELL_X_DLA_GE20P3_PW09_T1
    # deployed convention: symmetric errors = the UPPER errors (derive_hcd_lls_width.py:36-38)
    fit = LD.wls_powerlaw(a["z_bar"], a["lx"], a["err_hi"])
    assert round(float(fit["A"]), 4) == 0.0076
    assert round(float(fit["gamma"]), 3) == 1.592


def test_wls_matches_old_inline_implementation_bitwise():
    """The refactored function must equal the historical inline implementation to the bit
    on the same inputs (regression anchor: the machinery did not move)."""
    z = np.array([2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78])
    e = np.array([0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19])
    fit = LD.wls_powerlaw(z, v, e)
    # historical inline computation (verbatim port of derive_hcd_lls_width.py:44-78)
    x = np.log(1.0 + z); y = np.log(v); w = 1.0 / (e / v) ** 2
    X = np.vstack([np.ones_like(x), x]).T
    XtWX = X.T @ np.diag(w) @ X
    beta = np.linalg.solve(XtWX, X.T @ np.diag(w) @ y)
    assert float(fit["logA"]) == float(beta[0])
    assert float(fit["gamma"]) == float(beta[1])


def test_derive_width_script_output_unchanged_after_refactor():
    """derive_hcd_lls_width.py (now importing the shared wls_powerlaw) must print the
    identical RESULT numbers (byte-identical behavior; spec step 3)."""
    r = subprocess.run(
        [PY, f"{REPO}/scripts/derive_hcd_lls_width.py"], capture_output=True, text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": REPO, "JAX_PLATFORMS": "cpu",
             "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
        timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert "LLS      1x sigma/mu = 0.160   2x sigma/mu = 0.320   (WLS gamma=+2.127, A=0.0201)" in out
    assert "subDLA   1x sigma/mu = 0.102   2x sigma/mu = 0.205   (WLS gamma=+0.937, A=0.0211)" in out
    assert "DLA      1x sigma/mu = 0.094   2x sigma/mu = 0.187   (WLS gamma=+1.592, A=0.0076)" in out


# --------------------------------------------------------------------------- #
#  Poisson GLM (log link, plain numpy Newton/IRLS)                             #
# --------------------------------------------------------------------------- #
def test_poisson_glm_reproduces_pw09_dla_law():
    a = LD.ELL_X_DLA_GE20P3_PW09_T1
    g = LD.poisson_glm_powerlaw(a["z_bar"], a["m"], a["dX"])
    assert g["converged"]
    # expected ~(0.0076, 1.592) (spec 4a); GLM vs the WLS anchor differ only at rounding level
    assert abs(g["A"] - 0.0076) / 0.0076 < 0.05
    assert abs(g["gamma"] - 1.592) < 0.08
    assert g["deviance_dof"] < 3.0


def test_poisson_glm_exact_on_two_points():
    """2 points, 2 params -> saturated fit: mu == n exactly, deviance 0."""
    z = np.array([2.5, 3.5])
    n = np.array([20, 60])
    dX = np.array([100.0, 100.0])
    g = LD.poisson_glm_powerlaw(z, n, dX)
    np.testing.assert_allclose(g["mu_hat"], n, rtol=1e-8)
    assert abs(g["deviance"]) < 1e-8
    # closed form: gamma = ln(60/20)/ln(4.5/3.5)
    assert g["gamma"] == pytest.approx(np.log(3.0) / np.log(4.5 / 3.5), rel=1e-8)


def test_poisson_glm_synthetic_recovery():
    """Draws from a known law are recovered without bias (300-seed average within ~3 SE;
    measured per-fit sd(gamma)~0.24 at these counts -> SE_300(gamma)~0.014)."""
    rng = np.random.default_rng(1234)
    A_true, g_true = 0.006, 2.35
    z = np.array([1.8, 2.3, 2.8, 3.2, 3.7, 4.2])
    dX = np.array([500.0, 800.0, 900.0, 700.0, 500.0, 300.0])
    mu = dX * A_true * (1 + z) ** g_true
    gams, amps = [], []
    for _ in range(300):
        n = rng.poisson(mu)
        g = LD.poisson_glm_powerlaw(z, n, dX)
        assert g["converged"]
        gams.append(g["gamma"]); amps.append(g["A_pivot"])
    assert abs(np.mean(gams) - g_true) < 0.045          # ~3 SE
    assert abs(np.mean(amps) / (A_true * 4 ** g_true) - 1.0) < 0.01


def test_poisson_glm_handles_low_counts():
    """The Zafar-like n=4 row must not break the GLM (spec risk 6)."""
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    g = LD.poisson_glm_powerlaw(a["z_bar"], a["n"], a["dX"])
    assert g["converged"]
    assert 0.004 < g["A"] < 0.008          # spec expected region (0.005-0.007)
    assert 2.0 < g["gamma"] < 2.7          # spec expected region (2.2-2.5)
    assert g["deviance_dof"] < 2.0         # healthy where the old law showed chi2/dof=5.41


def test_poisson_vs_logwls_comparison():
    """The documented Poisson-vs-logWLS comparison (spec 4a): both computed on the same
    points; the log-WLS on sqrt(n)/dX errors is biased low in amplitude at low counts
    (E[ln n] < ln E[n]) — the comparison dict must expose both and their difference."""
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    cmp = LD.poisson_vs_logwls(a["z_bar"], a["n"], a["dX"])
    for k in ("glm", "wls", "delta_A_frac", "delta_gamma"):
        assert k in cmp
    assert cmp["glm"]["A"] != cmp["wls"]["A"]


# --------------------------------------------------------------------------- #
#  GLS with common-mode kernel covariance                                      #
# --------------------------------------------------------------------------- #
def test_gls_reduces_to_wls_with_zero_offdiag():
    z = np.array([2.21, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78])
    e = np.array([0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19])
    sig = e / v
    g = LD.gls_powerlaw_log(z, v, sig)
    w = LD.wls_powerlaw(z, v, e)
    assert g["gamma"] == pytest.approx(float(w["gamma"]), rel=1e-10)
    assert g["A"] == pytest.approx(float(w["A"]), rel=1e-10)


def test_gls_common_mode_inflates_amplitude_not_slope():
    """A fully-correlated common-mode block must inflate sigma_lnA at the pivot but leave
    sigma_gamma ~unchanged (the r(3) decomposition rationale, spec sec 3)."""
    z = np.array([2.21, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78])
    sig = np.full(8, 0.15)
    g0 = LD.gls_powerlaw_log(z, v, sig)
    g1 = LD.gls_powerlaw_log(z, v, sig, s_common=0.10)
    assert g1["sigma_lnAp"] > np.hypot(g0["sigma_lnAp"], 0.099)
    assert abs(g1["sigma_gamma"] - g0["sigma_gamma"]) < 1e-6 * (1 + g0["sigma_gamma"])
    # central values unchanged by a pure common mode? (GLS with J block shifts weights
    # uniformly -> the point estimate is unchanged)
    assert g1["gamma"] == pytest.approx(g0["gamma"], abs=1e-9)


def test_gls_eta_tilt_widens_slope():
    z = np.array([2.21, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78])
    sig = np.full(8, 0.15)
    g0 = LD.gls_powerlaw_log(z, v, sig)
    g1 = LD.gls_powerlaw_log(z, v, sig, sigma_eta=0.10)
    assert g1["sigma_gamma"] > g0["sigma_gamma"]


# --------------------------------------------------------------------------- #
#  Constrained GLS (gamma FIXED): the deployed-LLS estimator (PI 2026-07-18    #
#  decision 1c — keep the deployed z-slope 2.127, fit the amplitude only,      #
#  same covariance treatment; the free-gamma fit is the consistency evidence)  #
# --------------------------------------------------------------------------- #
def test_gls_constrained_gamma_exact_slope_and_analytic_amplitude():
    """gamma_fixed pins the slope EXACTLY (sigma_gamma=0) and the amplitude equals the
    closed-form GLS mean lnA_p = (1' Si (y - gamma*x)) / (1' Si 1) under the SAME Sigma
    (diag + common-mode + eta blocks) as the free fit."""
    z = np.array([2.21, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.24, 0.28, 0.33, 0.47, 0.36, 0.45, 0.63, 0.70])
    sig = np.array([0.17, 0.24, 0.31, 0.21, 0.17, 0.15, 0.20, 0.23])
    gfix = 2.127
    g = LD.gls_powerlaw_log(z, v, sig, s_common=0.10, sigma_eta=0.05, gamma_fixed=gfix)
    assert g["gamma"] == gfix                        # exact, not approx
    assert g["sigma_gamma"] == 0.0
    assert g.get("gamma_fixed") is True
    # closed form under the full Sigma
    x = np.log((1.0 + z) / 4.0)
    S = np.diag(sig ** 2) + 0.10 ** 2 * np.ones((8, 8)) + 0.05 ** 2 * np.outer(x, x)
    Si = np.linalg.inv(S)
    one = np.ones(8)
    lnAp = float(one @ Si @ (np.log(v) - gfix * x)) / float(one @ Si @ one)
    assert g["A_pivot"] == pytest.approx(np.exp(lnAp), rel=1e-12)
    assert g["sigma_lnAp"] == pytest.approx(np.sqrt(1.0 / float(one @ Si @ one)), rel=1e-12)
    assert g["A"] == pytest.approx(np.exp(lnAp) / 4.0 ** gfix, rel=1e-12)


def test_gls_constrained_at_free_gamma_matches_free_fit():
    """Constraining gamma AT the free-fit optimum reproduces the free amplitude (the
    constraint is consistent, not a different estimator) and never lowers chi2."""
    z = np.array([2.21, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23])
    v = np.array([0.24, 0.28, 0.33, 0.47, 0.36, 0.45, 0.63, 0.70])
    sig = np.full(8, 0.18)
    free = LD.gls_powerlaw_log(z, v, sig, s_common=0.08)
    con = LD.gls_powerlaw_log(z, v, sig, s_common=0.08, gamma_fixed=free["gamma"])
    assert con["A_pivot"] == pytest.approx(free["A_pivot"], rel=1e-10)
    assert con["chi2"] == pytest.approx(free["chi2"], rel=1e-8)
    # off-optimum constraint costs chi2
    con2 = LD.gls_powerlaw_log(z, v, sig, s_common=0.08, gamma_fixed=free["gamma"] - 0.3)
    assert con2["chi2"] > free["chi2"]


# --------------------------------------------------------------------------- #
#  Fit ordering + gates (spec sec 4; kernel-free parts)                        #
# --------------------------------------------------------------------------- #
def _toy_floor_factor(z):
    return np.full_like(np.asarray(z, float), 1.30)


def test_fit_ordering_runs_and_gates():
    # reference_laws = the DEPLOYED constants (mirrors inference.HCD_LIT_DNDX_LAW; the
    # module-vs-JSON pin lives in test_lit_dndx_corrected case 8b) — the consistency field
    # compares REFERENCE vs refit, never self-vs-self (review meta finding 6: the old
    # self-compare made the derive script's "[GATES] ALL PASS" vacuous).
    deployed = {"subDLA": (0.004832763139114936, 2.438007767664778),
                "DLA": (0.0076, 1.592)}
    res = LD.run_fit_ordering(floor_factor=_toy_floor_factor, reference_laws=deployed)
    for k in ("subDLA", "DLA", "cum_uncorrected", "K3"):
        assert k in res
    # internal-consistency gate: |weighted-mean fractional residual| < 5% per fitted class
    # (exact score identities make this ~0 for honest fits; a transplant with systematic
    # bias trips either this or the law-vs-refit gate)
    for cls in ("subDLA", "DLA", "cum_uncorrected"):
        assert abs(res[cls]["stats"]["wmean_frac_resid"]) < 0.05, cls
    # deployed-vs-refit: NON-vacuous for the deployed classes, None where no referent exists
    for cls in ("subDLA", "DLA"):
        assert res[cls]["consistency_vs_refit"]["max_frac_dev"] < 0.05, cls
    assert res["cum_uncorrected"]["consistency_vs_refit"] is None
    assert res["K3"]["law"]["consistency_vs_refit"] is None
    # the corrected K3 LLS law also passes vs its own (corrected) input points
    assert abs(res["K3"]["law"]["stats"]["wmean_frac_resid"]) < 0.05
    # DLA GLM-vs-WLS anchor agreement (PI flag otherwise)
    assert abs(res["DLA"]["glm_vs_wls"]["delta_gamma"]) < 0.1


def test_fit_ordering_gate_trips_on_wrong_object_reference():
    """The gate must actually FAIL when the reference is the tombstoned wrong-object law."""
    res = LD.run_fit_ordering(floor_factor=_toy_floor_factor,
                              reference_laws={"subDLA": (0.0211, 0.937)})
    assert res["subDLA"]["consistency_vs_refit"]["max_frac_dev"] > 0.30


def test_internal_consistency_gate_semantics():
    """Gate semantics: 'internal consistency' = a deployed law equals the refit of its own
    input points with the recorded estimator (max |law/refit - 1| < 5% over the fitted z).

    MEASURED FINDING (refutes the spec's ~15% premise): the sibling's transplant
    (0.0050, 2.426) is within 2.2% of our own Poisson-GLM refit of the verified Zafar
    counts (it IS the logWLS of those counts; deviance/dof = 0.73, healthy) — the
    '~15% internally inconsistent' claim was per-point Poisson scatter, not law-level
    inconsistency. The gate therefore trips on genuinely wrong-object laws (the old
    deployed (0.0211, 0.937), ~-46% at z=3), not on the transplant."""
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    refit = LD.poisson_glm_powerlaw(a["z_bar"], a["n"], a["dX"])
    # the transplant is (measured) CONSISTENT with the counts to ~2%
    dev_t = LD.law_consistency_vs_refit(0.0050, 2.426, refit, a["z_bar"])
    assert dev_t["max_frac_dev"] < 0.05
    # the old wrong-object law fails the gate massively (the real transplant-killer)
    dev_w = LD.law_consistency_vs_refit(0.0211, 0.937, refit, a["z_bar"])
    assert dev_w["max_frac_dev"] > 0.30
    # and the honest refit passes trivially
    dev_ok = LD.law_consistency_vs_refit(refit["A"], refit["gamma"], refit, a["z_bar"])
    assert dev_ok["max_frac_dev"] < 1e-9


def test_telescoping_consistency_k3():
    """K3 by construction: LLS_law/floor + subDLA_law + DLA_law ~ cumulative law within
    the compilation error at z in {2.5, 3.0, 3.5, 4.0} (spec sec 4)."""
    res = LD.run_fit_ordering(floor_factor=_toy_floor_factor)
    t = res["K3"]["telescoping"]
    assert np.all(np.abs(np.asarray(t["pull"])) < 1.0), t


def test_sensitivity_arms():
    res = LD.run_fit_ordering(floor_factor=_toy_floor_factor)
    arms = res["sensitivity"]
    for k in ("drop_z180", "larger_side", "journal_fix"):
        assert k in arms
    # z=1.80 drop: subDLA fit still healthy, gamma moves by a finite amount
    assert arms["drop_z180"]["subDLA"]["deviance_dof"] < 3.0
    assert arms["drop_z180"]["subDLA"]["gamma"] != res["subDLA"]["gamma"]
    # larger-side errors and journal-fix arms perturb the cumulative law only mildly
    assert abs(arms["larger_side"]["cum_uncorrected"]["gamma"] - res["cum_uncorrected"]["gamma"]) < 0.2
    assert abs(arms["journal_fix"]["cum_uncorrected"]["gamma"] - res["cum_uncorrected"]["gamma"]) < 0.2
