"""Gate E amendment A1 rev 1 section 6 (E7): the calibration statistics and the registered FAIL rule. Q and T are their
definitions (pseudo-inverse, rank, uncentered standardized projection); under a correctly sized covariance their null
centres are 1; the FAIL rule needs both materiality and a family-wise bootstrap interval excluding 1; the parametric
null and the power behave. Cache-free."""
import numpy as np
import pytest

from hcd_analysis.emulator import e7_calib as E7


def _psd(rng, N, rank=None):
    X = rng.normal(0, 1, (N, rank or N))
    return X @ X.T / (rank or N)


def test_Q_is_the_pseudo_inverse_quadratic_over_the_rank():
    rng = np.random.default_rng(0)
    C = _psd(rng, 8, rank=5)
    r = rng.normal(0, 1, 8)
    w, U = np.linalg.eigh(C)
    keep = w >= 1e-8 * w.max()
    expect = float((U[:, keep].T @ r) ** 2 @ (1 / w[keep])) / keep.sum()
    np.testing.assert_allclose(E7.Q_one(r, C), expect, rtol=1e-12)


def test_T_is_the_standardized_projected_quadratic():
    rng = np.random.default_rng(1)
    N, n = 12, 5
    J = rng.normal(0, 1, (N, n)); Cemu = _psd(rng, N); Ctot = Cemu + np.eye(N); P = np.eye(n) * 0.1
    r = rng.normal(0, 1, N)
    idx = [0, 1, 3]
    T, q = E7.T_one(r, J, Ctot, Cemu, P, idx)
    Ci = np.linalg.inv(Ctot); F = J.T @ Ci @ J + P; Fi = np.linalg.inv(F)
    d = (Fi @ J.T @ Ci @ r)[idx]; V = (Fi @ J.T @ Ci @ Cemu @ Ci @ J @ Fi)[np.ix_(idx, idx)]
    np.testing.assert_allclose(T, d @ np.linalg.solve(V, d) / 3, rtol=1e-10)
    np.testing.assert_allclose(q, d ** 2 / np.diag(V), rtol=1e-10)


def test_null_centres_are_one_for_a_correct_covariance():
    rng = np.random.default_rng(2)
    N, n, S = 10, 4, 400
    sims = []
    for _ in range(S):
        J = rng.normal(0, 1, (N, n)); Cemu = _psd(rng, N); Ctot = Cemu + 0.5 * np.eye(N)
        sims.append(dict(J=J, Cemu=Cemu, Ctot=Ctot))
    P = np.eye(n) * 0.01
    Qs, Ts = [], []
    for s in sims:
        r = np.linalg.cholesky(s["Cemu"] + 1e-12 * np.eye(N)) @ rng.normal(0, 1, N)
        Qs.append(E7.Q_one(r, s["Cemu"])); Ts.append(E7.T_one(r, s["J"], s["Ctot"], s["Cemu"], P, [0, 1, 2])[0])
    assert abs(np.mean(Qs) - 1) < 0.05 and abs(np.mean(Ts) - 1) < 0.15


def test_fail_needs_materiality_and_a_family_wise_interval_excluding_one():
    rng = np.random.default_rng(3)
    big = rng.normal(2.0, 0.1, 60)                  # clearly over 1.5 and significant
    small_shift = rng.normal(1.3, 0.1, 60)          # significant but inside [0.67, 1.5]
    z = rng.normal(0, 1, 60)
    noisy = 1.8 + 5.0 * (z - z.mean()) / z.std()    # mean exactly 1.8 (outside the band), not significant
    band = (0.67, 1.5)
    assert E7.fail(big, band, n_tests=8, n_boot=4000, seed=0)["fail"] == "high"
    assert E7.fail(small_shift, band, n_tests=8, n_boot=4000, seed=0)["fail"] is None
    assert E7.fail(noisy, band, n_tests=8, n_boot=4000, seed=0)["fail"] is None
    assert E7.fail(rng.normal(0.4, 0.05, 60), band, n_tests=8, n_boot=4000, seed=0)["fail"] == "low"


def test_amplitude_heterogeneity_is_mean_square_one():
    a = E7.amplitudes(np.random.default_rng(4), 400000, sigma_ln=0.35)
    np.testing.assert_allclose(np.mean(a ** 2), 1.0, rtol=0.01)
    small = E7.amplitudes(np.random.default_rng(5), 60, sigma_ln=0.35)
    assert abs(np.mean(small ** 2) - 1.0) > 1e-6          # the population normalization, not the sample's


@pytest.mark.parametrize("tails", [None, 5])
def test_unit_draws_have_unit_variance(tails):
    x = E7.unit_draws(np.random.default_rng(6), 400000, tails)
    np.testing.assert_allclose(np.var(x), 1.0, rtol=0.03)


def test_power_grows_away_from_the_correct_scale():
    rng = np.random.default_rng(5)
    N, n = 8, 4
    sims = []
    for _ in range(60):
        J = rng.normal(0, 1, (N, n)); Cemu = _psd(rng, N); Ctot = Cemu + 0.5 * np.eye(N)
        sims.append(dict(J=J, Cemu=Cemu, Ctot=Ctot, n_vec=2))
    out = E7.null_and_power(sims, P=np.eye(n) * 0.01, idx=[0, 1, 2], sigma_ln=0.3, scales=(1.0, 2.0), n_rep=150,
                            n_tests=8, seed=0, n_boot=500)
    assert out[1.0]["p_fail_any"] < 0.2 and out[2.0]["p_fail_any"] > 0.6
