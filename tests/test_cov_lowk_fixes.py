"""Back-compat + behavior tests for the env-gated low-k covariance fixes (2026-06-18).

The two fixes (DESI SNR>3 measurement+cov, and the Fernandez σ_CV ~2% finite-box floor) are
ENV-GATED and REVERSIBLE: env UNSET ⇒ byte-identical to the deployed path. These tests pin:
  1. the pure taper helpers (_cv_floor_frac / _add_cv_floor) — cache-free;
  2. env-UNSET ⇒ the loaders return BYTE-IDENTICAL covariance (the back-compat guarantee);
  3. each flag, when SET, actually WIDENS the low-k covariance (no silent no-op).
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
_have_desi = os.path.exists(DESI_NPZ)
_have_snr3 = os.path.exists(DL.DESI_SNR3_NPZ)
_have_eboss = os.path.exists(EBOSS_NPZ)


# --------------------------------------------------------------------------- #
#  1. pure taper helpers (cache-free)
# --------------------------------------------------------------------------- #
def test_cv_floor_frac_taper():
    """σ_CV(k): full CV_FLOOR_FRAC at k≤K_FULL, linearly →0 by K_ZERO, 0 above."""
    f_lo = DL._cv_floor_frac(np.array([1e-4, DL.CV_FLOOR_K_FULL]))
    np.testing.assert_allclose(f_lo, DL.CV_FLOOR_FRAC, rtol=0, atol=1e-12)
    # at K_ZERO and above → 0
    f_hi = DL._cv_floor_frac(np.array([DL.CV_FLOOR_K_ZERO, 0.01, 0.06]))
    np.testing.assert_allclose(f_hi, 0.0, atol=1e-12)
    # midpoint is between (monotone non-increasing)
    kmid = 0.5 * (DL.CV_FLOOR_K_FULL + DL.CV_FLOOR_K_ZERO)
    fmid = float(DL._cv_floor_frac(np.array([kmid]))[0])
    assert 0.0 < fmid < DL.CV_FLOOR_FRAC


def test_add_cv_floor_diag_and_rank1():
    """_add_cv_floor adds (f·P)² on the diagonal (diag) or s sᵀ (rank1); off above K_ZERO it is 0."""
    k = np.array([1e-4, DL.CV_FLOOR_K_FULL, 0.02, 0.05])      # 2 low-k rows + 2 high-k rows
    P = np.array([0.2, 0.15, 0.1, 0.08])
    C0 = np.eye(4) * 1e-3
    Cd = DL._add_cv_floor(C0, k, P, rank1=False)
    s = DL._cv_floor_frac(k) * P
    np.testing.assert_allclose(np.diag(Cd) - np.diag(C0), s ** 2, rtol=0, atol=1e-14)
    # high-k rows (k≥K_ZERO) get ZERO floor
    assert Cd[2, 2] == C0[2, 2] and Cd[3, 3] == C0[3, 3]
    # rank1 adds the fully-correlated outer product (off-diagonals appear on the low-k block)
    Cr = DL._add_cv_floor(C0, k, P, rank1=True)
    np.testing.assert_allclose(Cr - C0, np.outer(s, s), rtol=0, atol=1e-14)
    assert Cr[0, 1] != 0.0 and Cd[0, 1] == C0[0, 1]          # rank1 correlates; diag does not
    # the input is not mutated
    assert np.array_equal(C0, np.eye(4) * 1e-3)


# --------------------------------------------------------------------------- #
#  2. env-UNSET ⇒ byte-identical (the back-compat guarantee)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_env_unset_byte_identical(monkeypatch):
    """With HCD_DESI_SNR3 / HCD_CV_FLOOR UNSET, load_desi_leg is BYTE-IDENTICAL to the explicit
    deployed path (use_snr3=False, add_cv_floor=False) — the reversibility guarantee."""
    for v in ("HCD_DESI_SNR3", "HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1"):
        monkeypatch.delenv(v, raising=False)
    leg_env = DL.load_desi_leg()                              # env unset → reads the flags (all off)
    leg_explicit = DL.load_desi_leg(use_snr3=False, add_cv_floor=False)
    np.testing.assert_array_equal(leg_env.P_data, leg_explicit.P_data)
    np.testing.assert_array_equal(leg_env.C_data, leg_explicit.C_data)
    np.testing.assert_array_equal(leg_env.k, leg_explicit.k)


@pytest.mark.skipif(not _have_eboss, reason="eBOSS npz not present")
def test_eboss_env_unset_byte_identical(monkeypatch):
    """eBOSS: HCD_CV_FLOOR unset ⇒ load_eboss_leg byte-identical to add_cv_floor=False."""
    for v in ("HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1"):
        monkeypatch.delenv(v, raising=False)
    leg_env = DL.load_eboss_leg()
    leg_explicit = DL.load_eboss_leg(add_cv_floor=False)
    np.testing.assert_array_equal(leg_env.C_data, leg_explicit.C_data)
    np.testing.assert_array_equal(leg_env.P_data, leg_explicit.P_data)


# --------------------------------------------------------------------------- #
#  3. each flag, when SET, WIDENS the low-k covariance (no silent no-op)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_cv_floor_widens_lowk_only():
    """add_cv_floor=True raises the diagonal of the low-k rows (k<K_ZERO) and leaves high-k rows
    byte-identical — the floor is a low-k-only inflation."""
    leg0 = DL.load_desi_leg(add_cv_floor=False)
    leg1 = DL.load_desi_leg(add_cv_floor=True)
    d0, d1 = np.diag(leg0.C_data), np.diag(leg1.C_data)
    assert np.all(d1 >= d0 - 1e-18)
    lowk = leg1.k < DL.CV_FLOOR_K_ZERO
    if lowk.any():
        assert np.any(d1[lowk] > d0[lowk]), "cv-floor must raise some low-k diagonal entries"
    highk = leg1.k >= DL.CV_FLOOR_K_ZERO
    if highk.any():
        np.testing.assert_array_equal(d1[highk], d0[highk])  # high-k untouched


@pytest.mark.skipif(not (_have_desi and _have_snr3), reason="DESI baseline or SNR3 npz not present")
def test_desi_snr3_swaps_cov_and_inflates_stat():
    """use_snr3=True loads the SNR>3 measurement+cov (DIFFERENT from the SNR>1 baseline) and the
    +5% stat inflation rides along (the cov is not byte-identical to the SNR>1 path)."""
    leg1 = DL.load_desi_leg(use_snr3=False)
    leg3 = DL.load_desi_leg(use_snr3=True)
    # same z/k grid, DIFFERENT covariance (different subsample + the stat inflation).
    assert leg1.C_data.shape == leg3.C_data.shape
    assert not np.allclose(leg1.C_data, leg3.C_data), "SNR>3 cov must differ from the SNR>1 baseline"
