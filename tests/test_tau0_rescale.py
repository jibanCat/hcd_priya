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


if __name__ == "__main__":
    test_freeze_core_rescale_freezes_cores_scales_thin()
    test_freeze_core_rescale_uniform_when_tau_freeze_inf()
    test_freeze_core_rescale_alpha_one_is_identity()
    test_make_alpha_grid_spans_range_inclusive()
    test_make_alpha_grid_default_count()
    test_tau0_from_mean_flux_inverts_exp()
    print("OK")
