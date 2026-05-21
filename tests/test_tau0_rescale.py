"""Tests for hcd_analysis/tau0_rescale.py and the compute_p1d_per_class
tau_transform hook.

Run with: python3 tests/test_tau0_rescale.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.tau0_rescale import (
    freeze_core_rescale, make_alpha_grid, tau0_from_mean_flux,
    TAU_FREEZE_DEFAULT, N_ALPHA_DEFAULT,
)


def test_freeze_core_rescale_freezes_cores_scales_thin():
    tau = np.array([1.0e3, 5.0e6, 1.0e6, 2.0e6, 0.5])
    out = freeze_core_rescale(tau, alpha=1.3, tau_freeze=1.0e6)
    # tau > 1e6 frozen; tau <= 1e6 scaled by alpha
    expected = np.array([1.3e3, 5.0e6, 1.3e6, 2.0e6, 0.65])
    assert np.allclose(out, expected), f"got {out}"


def test_freeze_core_rescale_uniform_when_tau_freeze_inf():
    tau = np.array([1.0e3, 5.0e6, 2.0e6])
    out = freeze_core_rescale(tau, alpha=0.7, tau_freeze=np.inf)
    assert np.allclose(out, 0.7 * tau), f"got {out}"


def test_freeze_core_rescale_alpha_one_is_identity():
    tau = np.array([1.0e3, 5.0e6, 2.0e6, 0.1])
    out = freeze_core_rescale(tau, alpha=1.0, tau_freeze=TAU_FREEZE_DEFAULT)
    assert np.allclose(out, tau), f"got {out}"


def test_make_alpha_grid_spans_range_inclusive():
    grid = make_alpha_grid(n=20, lo=0.66, hi=1.36)
    assert grid.shape == (20,)
    assert np.isclose(grid[0], 0.66)
    assert np.isclose(grid[-1], 1.36)
    assert np.all(np.diff(grid) > 0), "grid must be strictly increasing"


def test_make_alpha_grid_default_count():
    assert make_alpha_grid().shape == (N_ALPHA_DEFAULT,)


def test_tau0_from_mean_flux_inverts_exp():
    assert np.isclose(tau0_from_mean_flux(np.exp(-2.0)), 2.0)
    assert np.isclose(tau0_from_mean_flux(1.0), 0.0)


def _make_synthetic_tau_hdf5(path, tau):
    """Write a (n_skewers, nbins) tau array as a SPECTRA-style HDF5 file."""
    with h5py.File(path, "w") as f:
        f.create_dataset("tau/H/1/1215", data=tau.astype(np.float32))


def test_compute_p1d_per_class_tau_transform_changes_mean_flux():
    from hcd_analysis.p1d import compute_p1d_per_class
    from hcd_analysis.catalog import AbsorberCatalog

    rng = np.random.default_rng(0)
    nbins = 128
    tau = rng.uniform(0.0, 2.0, size=(64, nbins))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.hdf5"
        _make_synthetic_tau_hdf5(path, tau)
        cat = AbsorberCatalog(sim_name="syn", snap=0, z=3.0, dv_kms=10.0)

        base = compute_p1d_per_class(path, nbins=nbins, dv_kms=10.0, catalog=cat)
        scaled = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=lambda t: 2.0 * t,
        )
        ident = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=lambda t: t,
        )

    # empty catalog => every sightline is "clean"
    assert base["n_sightlines_clean"] == 64
    assert np.isclose(base["mean_F_clean"], np.exp(-tau).mean())
    assert np.isclose(scaled["mean_F_clean"], np.exp(-2.0 * tau).mean())
    # identity transform reproduces the no-transform result exactly
    assert np.allclose(ident["P_clean"], base["P_clean"])
    assert np.isclose(ident["mean_F_clean"], base["mean_F_clean"])


def test_kim_slope_alpha_to_target_F_matches_obs_mean_tau():
    """The Tier P / Tier C convention is target_F = exp(-alpha * obs_mean_tau_Kim(z)).
    Verify the helper agrees with the Kim 2013 (0711.1862) formula."""
    from hcd_analysis.tau0_rescale import slope_alpha_to_target_F
    # at z=3, Kim 2013: obs_mean_tau = 2.3e-3 * 4^3.65
    target = slope_alpha_to_target_F(alpha_slope=1.0, z=3.0)
    expected = np.exp(-2.3e-3 * 4.0 ** 3.65)
    assert np.isclose(target, expected, rtol=1e-12), f"got {target}, expected {expected}"


if __name__ == "__main__":
    test_freeze_core_rescale_freezes_cores_scales_thin()
    test_freeze_core_rescale_uniform_when_tau_freeze_inf()
    test_freeze_core_rescale_alpha_one_is_identity()
    test_make_alpha_grid_spans_range_inclusive()
    test_make_alpha_grid_default_count()
    test_tau0_from_mean_flux_inverts_exp()
    test_compute_p1d_per_class_tau_transform_changes_mean_flux()
    test_kim_slope_alpha_to_target_F_matches_obs_mean_tau()
    print("OK")
