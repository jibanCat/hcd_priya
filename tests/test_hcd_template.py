"""
Minimal unit tests for hcd_analysis.hcd_template.

Run as:
    python3 tests/test_hcd_template.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcd_analysis.hcd_template import (
    template_factor,
    template_factor_from_cyclic_k,
    template_contributions,
    fit_alpha,
    correct_p1d,
)


def _user_reference_DLA4corr(kf, z, alpha):
    """Verbatim copy of the user-supplied DLA4corr formula, for cross-check."""
    z_0 = 2
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
    a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
    b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1
    factor = np.ones(kf.size)
    for i in range(4):
        factor += (alpha[i] * ((1 + z) / (1 + z_0)) ** -3.55
                   * ((a_z[i] * np.exp(b_z[i] * kf) - 1) ** -2))
    return factor


def test_matches_user_reference_across_z_and_k():
    kf = np.array([0.002, 0.005, 0.01, 0.03, 0.1, 0.2])
    for z in [2.2, 3.0, 4.2, 5.0]:
        for alpha in [np.full(4, 0.1), np.array([0.2, 0.1, 0.05, 0.3])]:
            ours = template_factor(kf, z, alpha)
            ref = _user_reference_DLA4corr(kf, z, alpha)
            assert np.allclose(ours, ref, atol=1e-12), \
                f"z={z} alpha={alpha}: mismatch max={np.max(np.abs(ours-ref))}"
    print("  ✓ template_factor matches user DLA4corr exactly")


def test_cyclic_wrapper():
    k_cyc = np.array([0.001, 0.005, 0.02])
    alpha = np.full(4, 0.1)
    f_cyc = template_factor_from_cyclic_k(k_cyc, 3.0, alpha)
    f_ang = template_factor(2 * np.pi * k_cyc, 3.0, alpha)
    assert np.allclose(f_cyc, f_ang)
    print("  ✓ cyclic wrapper = 2π·angular call")


def test_zero_alpha_gives_unity():
    kf = np.linspace(0.001, 0.2, 40)
    f = template_factor(kf, 3.0, np.zeros(4))
    assert np.allclose(f, 1.0)
    print("  ✓ alpha=0 → factor=1 at every k")


def test_contributions_sum_to_total():
    kf = np.array([0.002, 0.01, 0.05])
    alpha = np.array([0.1, 0.2, 0.15, 0.3])
    c = template_contributions(kf, 3.0, alpha)
    tot = template_factor(kf, 3.0, alpha)
    recon = np.ones_like(kf)
    for name, curve in c.items():
        recon = recon + (curve - 1.0)
    assert np.allclose(recon, tot)
    print("  ✓ sum of per-class contributions = total factor")


def test_correct_p1d_inverts_template():
    kf = np.linspace(0.002, 0.05, 20)
    P_forest = 0.1 + np.exp(-kf * 30.)  # arbitrary smooth model
    alpha_true = np.array([0.08, 0.15, 0.12, 0.25])
    P_obs = P_forest * template_factor(kf, 3.0, alpha_true)
    P_back = correct_p1d(kf, P_obs, 3.0, alpha_true)
    assert np.allclose(P_back, P_forest, rtol=1e-12)
    print("  ✓ correct_p1d = P_observed / template_factor (exact)")


def test_fit_recovers_alpha_on_dense_lowk_grid():
    # Use enough low-k points for the 4-parameter fit to be non-degenerate.
    kf = np.concatenate([
        np.linspace(0.0005, 0.005, 30),
        np.linspace(0.006, 0.05, 40),
    ])
    P_forest = 0.5 + np.exp(-kf * 40.)
    alpha_true = np.array([0.12, 0.18, 0.10, 0.22])
    np.random.seed(42)
    P_obs = P_forest * template_factor(kf, 3.0, alpha_true) \
            * (1 + 0.005 * np.random.randn(kf.size))  # 0.5 % multiplicative noise
    res = fit_alpha(kf, P_obs, P_forest, z=3.0, alpha0=np.full(4, 0.05))
    err = np.abs(res["alpha"] - alpha_true)
    # Tolerance generous to accommodate the poor constraint on Large-DLA when
    # the noise is above that class's contribution level.
    assert np.all(err < 0.08), \
        f"recovered α = {res['alpha']}, true α = {alpha_true}, err = {err}"
    print(f"  ✓ fit recovered α (err {err.round(3)}) at 0.5% noise")


def main():
    print("Running hcd_template tests:")
    for fn in [
        test_matches_user_reference_across_z_and_k,
        test_cyclic_wrapper,
        test_zero_alpha_gives_unity,
        test_contributions_sum_to_total,
        test_correct_p1d_inverts_template,
        test_fit_recovers_alpha_on_dense_lowk_grid,
    ]:
        fn()
    print("All tests passed.")


if __name__ == "__main__":
    main()
