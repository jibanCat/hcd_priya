"""Research round R1, T3 benchmark kit (hcd_analysis/emulator/r1_bench.py): the factorial likelihood, the mode-aligned T3
factor bound at the query theta, the bounded MAP with Laplace widths, and the benchmark metrics. Cache-free."""
import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hcd_analysis.emulator import cemu_t3 as T3
from hcd_analysis.emulator import r1_bench as RB
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def _psd(rng, n):
    X = rng.normal(0, 1, (n, n))
    return X @ X.T + n * np.eye(n)


def test_factorial_loglik_is_the_gaussian_when_both_matrices_agree():
    rng = np.random.default_rng(0)
    C = _psd(rng, 7); r = rng.normal(0, 1, 7)
    np.testing.assert_allclose(float(RB.factorial_loglik(jnp.asarray(r), jnp.asarray(C), jnp.asarray(C))),
                               multivariate_normal(np.zeros(7), C).logpdf(r), rtol=1e-12)


def test_factorial_loglik_splits_quadratic_and_log_determinant():
    rng = np.random.default_rng(1)
    Cq, Cd = _psd(rng, 5), _psd(rng, 5); r = rng.normal(0, 1, 5)
    expect = -0.5 * r @ np.linalg.solve(Cq, r) - 0.5 * np.linalg.slogdet(Cd)[1] - 2.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(RB.factorial_loglik(jnp.asarray(r), jnp.asarray(Cq), jnp.asarray(Cd))), expect,
                               rtol=1e-12)


def test_mode_aligned_factor_is_the_numpy_binding_times_the_mode_basis():
    rng = np.random.default_rng(2)
    zs = np.round(np.arange(2.0, 4.41, 0.2), 1)
    kb = np.array([0.012, 0.02, 0.031, 0.015, 0.045]); iz = np.array([3, 3, 3, 6, 6])
    th = rng.uniform(0.1, 0.9, 9)
    Mi = T3.mode_set(KCOM, zs, iz, kb, np.zeros(9), np.ones(9))
    U = rng.normal(0, 0.01, (Mi.size, 3))
    rows = np.array([1, 2, 4, 6, 7]); N = 9
    got = np.asarray(RB.mode_aligned_factor(jnp.asarray(th), KCOM, zs, iz, kb, Mi, jnp.asarray(U), rows, N))
    W = T3.weights(KCOM, th, zs, iz, kb)[:, Mi]
    expect = np.zeros((N, 3)); expect[rows] = W @ U
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-18)


def test_map_and_laplace_on_a_linear_gaussian_model_are_exact():
    rng = np.random.default_rng(3)
    n, N = 4, 30
    A = rng.normal(0, 1, (N, n)); C = _psd(rng, N) * 0.1
    p_true = np.array([0.3, -0.2, 0.5, 0.1])
    d = A @ p_true + rng.multivariate_normal(np.zeros(N), C)
    nlp = lambda p: 0.5 * (d - A @ p) @ jnp.linalg.solve(jnp.asarray(C), d - A @ p)
    out = RB.map_laplace(nlp, p0=np.zeros(n), bounds=[(-10, 10)] * n)
    F = A.T @ np.linalg.solve(C, A)
    np.testing.assert_allclose(out["map"], np.linalg.solve(F, A.T @ np.linalg.solve(C, d)), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(out["sigma"], np.sqrt(np.diag(np.linalg.inv(F))), rtol=1e-6)
    assert out["converged"] and not np.any(out["at_bound"])


def test_pull_coverage_and_bootstrap_by_truth():
    pulls = np.array([[0.2, 3.0], [-0.5, 0.1], [1.5, -0.2], [0.0, 2.5]])          # (mocks, params)
    groups = np.array([0, 0, 1, 1])                                                 # two noise draws per truth
    m = RB.pull_summary(pulls, groups, n_boot=200, seed=0)
    np.testing.assert_allclose(m["mean"], pulls.mean(0)); np.testing.assert_allclose(m["rms"], np.sqrt((pulls ** 2).mean(0)))
    np.testing.assert_allclose(m["cov68"], [0.75, 0.5]); np.testing.assert_allclose(m["cov95"], [1.0, 0.5])
    assert m["mean_ci"].shape == (2, 2)


def test_paired_difference_resamples_whole_truths():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 96); b = a + 0.3
    groups = np.repeat(np.arange(48), 2)
    lo, hi = RB.paired_ci(a, b, groups, n_boot=500, seed=0)
    np.testing.assert_allclose([lo, hi], [0.3, 0.3], atol=1e-12)            # paired b - a: a constant shift has no spread
