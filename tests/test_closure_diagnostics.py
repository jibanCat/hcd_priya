"""Phase-C T4 — the SBC statistical machinery (closure_diagnostics), ported + verified by
the Bayesian/PPL co-design agent (2026-06-04). NUTS-free, fast.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_closure_diagnostics.py -v
"""
import numpy as np

from hcd_analysis.emulator import closure_diagnostics as D


def test_sbc_power_N_matches_design():
    """N≈621 to detect a 0.2σ posterior-mean bias at 95% bands / 80% power."""
    out = D.sbc_power_N(bias_sigma=0.2)
    assert 550 <= out["N"] <= 750, out
    # weaker biases need fewer mocks
    assert D.sbc_power_N(bias_sigma=0.5)["N"] < out["N"]


def test_ecdf_pit_bands_null_passes_biased_fails():
    """Säilynoja+2022 ECDF bands: calibrated (uniform) ranks PASS; piled-low ranks FAIL."""
    rng = np.random.default_rng(1); L = 99; N = 400
    uniform = rng.integers(0, L + 1, N)
    res_u = D.ecdf_pit_bands(uniform, n_draws=L, rng=np.random.default_rng(2))
    assert res_u[3], "calibrated ranks must pass the ECDF bands"
    biased = rng.integers(0, (L + 1) // 3, N)          # ranks concentrated low
    res_b = D.ecdf_pit_bands(biased, n_draws=L, rng=np.random.default_rng(3))
    assert not res_b[3], "non-uniform ranks must fail the ECDF bands"


def test_loglik_rank_in_range():
    rng = np.random.default_rng(0)
    r = D.loglik_rank(0.5, rng.normal(size=99))
    assert 0 <= r <= 99


def test_whitening_test_correct_C_unit_variance():
    """Residuals drawn from N(0,C) whiten to ~unit variance, χ²/dof≈1."""
    rng = np.random.default_rng(0); K, M = 20, 300
    A = rng.normal(size=(K, K)); C = A @ A.T + K * np.eye(K)
    L = np.linalg.cholesky(C)
    resid = np.stack([L @ rng.normal(size=K) for _ in range(M)])
    out = D.whitening_test(resid, C)                    # shared C
    assert abs(out["var"] - 1.0) < 0.15, out
    assert abs(out["chi2_over_dof"] - 1.0) < 0.15
    assert out["ks_p"] > 0.01
    # a 2x under-estimated C inflates the whitened variance
    out2 = D.whitening_test(resid, C / 4.0)
    assert out2["var"] > 3.0 and out2["ks_p"] < 1e-6


def test_empirical_coverage_wilson_ci():
    """68% intervals over many calibrated draws give ~68% coverage."""
    rng = np.random.default_rng(0); M = 500
    truths = rng.normal(size=M)
    lo, hi = -1.0, 1.0                                  # ~68% of N(0,1)
    intervals = np.tile([lo, hi], (M, 1))
    out = D.empirical_coverage(truths, intervals)
    assert out["ci_low"] <= 0.68 <= out["ci_high"]
