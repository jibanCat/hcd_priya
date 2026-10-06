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
