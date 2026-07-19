"""Numerical cross-check that our DESI resolution surgery + forward factor match cup1d.

Two permanent guards documenting "our DESI implementation is numerically identical to cup1d":

  1. COVARIANCE. Our option-b ``cov_b`` is built SUBTRACTIVELY -- take the shipped total
     covariance (= cov_stat + cov_syst) and remove the per-z-block rank-1 resolution mode
     ``cov_b = C - sum_z outer(e_res|z)`` (data_likelihood._assemble_leg, resolution_mode
     "rank1"). cup1d (p1ds/data_DESIY1.compute_cov) instead builds cov ADDITIVELY from the
     systematics vectors: cov = cov_stat + sum_{corr} outer(e|z) + sum_{ucorr} diag(e^2).
     "Build-without-resolution" = the same additive sum with E_RESOLUTION dropped from the
     corr set. This test asserts our subtractive cov_b == cup1d's additive build to rtol
     1e-10 on the SAME post-cut grid (both SPD). It also pins that cup1d's 'fid' split
     reproduces the shipped cov_syst exactly (the split is correct).

  2. FORWARD FACTOR. Our resolution template is the EXP ``exp(2 b_res k^2 R_z^2)``
     (data_likelihood._resolution_factor); cup1d's is the first-order LINEAR
     ``1 + 2 R_coeff R_z^2 k^2`` (contaminants/resolution_class.Resolution.get_contamination).
     Same argument x = 2 b R_z^2 k^2, so the two differ only by exp(x) vs 1+x. This test
     asserts the in-band fractional gap is <= 1% at b_res=0.02 (after the half-Nyquist
     k < 0.5 pi/R_z cut), and that our R_z proxy equals cup1d's get_Rz_Naim exactly.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_resolution_cup1d_consistency.py -q
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64 before jax arrays
import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have_desi = os.path.exists(DESI_NPZ)

# cup1d compute_cov (QMLE) systematic labels + the 'fid' correlated/uncorrelated split, mapped
# to the npz keys the converter stored (scripts/convert_desi_dr1_p1d.py SYST_COLS).
_LAB2KEY = {
    "E_DLA_COMPLETENESS": "syst_e_dla_completeness",
    "E_BAL_COMPLETENESS": "syst_e_bal_completeness",
    "E_RESOLUTION":       "syst_e_resolution",
    "E_CONTINUUM":        "syst_e_continuum",
    "E_CONTINUUM_ADD":    "syst_e_continuum_add",
    "E_NOISE_SCALE":      "syst_e_noise_scale",
    "E_NOISE_ADD":        "syst_e_noise_add",
}
_SYS_LABELS = ["E_DLA_COMPLETENESS", "E_BAL_COMPLETENESS", "E_RESOLUTION",
               "E_CONTINUUM", "E_CONTINUUM_ADD", "E_NOISE_SCALE", "E_NOISE_ADD"]
_CORR_FID = ["E_DLA_COMPLETENESS", "E_BAL_COMPLETENESS", "E_RESOLUTION",
             "E_CONTINUUM", "E_NOISE_SCALE"]
_UCORR_FID = ["E_NOISE_ADD", "E_CONTINUUM_ADD"]


def _compute_cov_cup1d(z_col, syst_vecs, corr_labels, ucorr_labels):
    """Faithful replication of cup1d.p1ds.data_DESIY1.compute_cov (QMLE, per-z-block).

    corr labels get a per-z-block rank-1 outer(e|z) added; ucorr labels add diag(e^2).
    Iterates the SYS_LABELS in cup1d's order over contiguous z-major blocks (slice from
    the first to last index of each unique z)."""
    zz_unique = np.unique(z_col)
    n = len(z_col)
    C = np.zeros((n, n))
    diag = np.diag_indices_from(C)
    for lab in _SYS_LABELS:
        v = syst_vecs[lab]
        if lab in ucorr_labels:
            C[diag] += v ** 2
        elif lab in corr_labels:
            for z in zz_unique:
                ind = np.argwhere(z_col == z)[:, 0]
                sl = slice(ind[0], ind[-1] + 1)
                C[sl, sl] += np.outer(v[ind], v[ind])
    return C


def _get_Rz_Naim(z):
    """cup1d contaminants/resolution_class.get_Rz_Naim: c*0.8/((1+z)*1215.67) [s/km]."""
    c_kms = 2.99792458e5
    return c_kms * 0.8 / (1.0 + np.asarray(z)) / 1215.67


def test_desi_resolution_R_matches_cup1d_get_Rz_Naim():
    """Cache-free: our desi_resolution_R proxy == cup1d get_Rz_Naim to machine precision.
    This is the shared R_z that both the cov half-Nyquist cut and the forward factor use;
    if it drifts, everything downstream drifts, so pin it exactly."""
    z = np.linspace(2.0, 5.4, 35)
    np.testing.assert_allclose(np.asarray(DL.desi_resolution_R(z)), _get_Rz_Naim(z), rtol=1e-12)


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_npz_fid_split_reproduces_shipped_cov_syst():
    """cup1d's 'fid' correlated/uncorrelated split, rebuilt from the syst vectors, reproduces
    the shipped cov_syst (and cov == cov_stat + cov_syst). Pins that the split we subtract
    the resolution mode from is the SAME one cup1d ships."""
    d = np.load(DESI_NPZ, allow_pickle=True)
    z_all = np.asarray(d["z"], float)
    cov = np.asarray(d["cov"], float)
    cov_stat = np.asarray(d["cov_stat"], float)
    cov_syst_shipped = np.asarray(d["cov_syst"], float)
    syst_vecs = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}

    # cov == cov_stat + cov_syst (npz internal consistency)
    np.testing.assert_allclose(cov, cov_stat + cov_syst_shipped, rtol=0, atol=1e-18)
    # the 'fid' rebuild reproduces the shipped cov_syst exactly
    cov_syst_fid = _compute_cov_cup1d(z_all, syst_vecs, _CORR_FID, _UCORR_FID)
    np.testing.assert_allclose(cov_syst_fid, cov_syst_shipped, rtol=1e-12, atol=1e-18)


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_cov_b_equals_cup1d_additive_build_without_resolution():
    """PERMANENT GUARD: our subtractive DESI cov_b == cup1d's additive build-without-resolution
    to rtol 1e-10 on the same post-cut grid; both SPD. (add_cov_diag_inflation=False for an
    apples-to-apples compare -- the inflation is an orthogonal shared diagonal term cup1d omits.)"""
    d = np.load(DESI_NPZ, allow_pickle=True)
    z_all = np.asarray(d["z"], float)
    k_all = np.asarray(d["k"], float)
    cov_stat = np.asarray(d["cov_stat"], float)
    syst_vecs = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}

    # cup1d additive build-WITHOUT-resolution (full grid): drop E_RESOLUTION from the corr set.
    corr_nores = [c for c in _CORR_FID if c != "E_RESOLUTION"]
    cup1d_full = cov_stat + _compute_cov_cup1d(z_all, syst_vecs, corr_nores, _UCORR_FID)

    # our option-b cov_b (subtractive per-z rank-1 removal), no diag inflation for the compare.
    # dla_cov_reduce=False: THIS test pins the RESOLUTION-mode equality in isolation. Since
    # 2026-07-17 the DESI default ALSO removes syst_e_dla_completeness (DESI_DLA_COV_REDUCE, the
    # cup1d "red" convention); that full reduced build is pinned in tests/test_dla_cov_reduce.py
    # (test_reduced_equals_cup1d_red_build + the opt-out back-compat pin).
    leg_b = DL.load_desi_leg(resolution_float=True, add_cov_diag_inflation=False,
                             dla_cov_reduce=False)
    our_cov_b = np.asarray(leg_b.C_data, float)
    assert leg_b.resolution_on is True, "resolution_float must flip resolution_on True (float f_res)"

    # recompute the SAME keep mask load_desi_leg uses, to sub-select cup1d's full-grid build.
    R_row = DL.desi_resolution_R(z_all)
    k_hi_row = 0.5 * np.pi / R_row
    keep = ((z_all >= 2.2 - 1e-6) & (z_all <= 4.2 + 1e-6)
            & (k_all > DL.DESI_KMIN) & (k_all < k_hi_row))
    idx = np.where(keep)[0]
    cup1d_cov_b = cup1d_full[np.ix_(idx, idx)]

    assert our_cov_b.shape == cup1d_cov_b.shape, "post-cut grid size mismatch"
    # the load-bearing equality: subtractive == additive to rtol 1e-10.
    np.testing.assert_allclose(our_cov_b, cup1d_cov_b, rtol=1e-10, atol=1e-12)
    # and it is a TIGHT machine-precision agreement, not just within 1e-10.
    assert np.max(np.abs(our_cov_b - cup1d_cov_b)) < 1e-11
    # both SPD (Cholesky succeeds, min eigenvalue positive).
    np.linalg.cholesky(our_cov_b)
    np.linalg.cholesky(cup1d_cov_b)
    assert np.linalg.eigvalsh(our_cov_b).min() > 0.0


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_forward_factor_exp_vs_linear_inband_gap_leq_1pct():
    """PERMANENT GUARD: our exp resolution factor vs cup1d's linear one differ <= 1% in-band
    at b_res=0.02 (after the half-Nyquist cut). Both use the same R_z, and our
    _resolution_factor equals the raw exp form."""
    d = np.load(DESI_NPZ, allow_pickle=True)
    z_all = np.asarray(d["z"], float)
    k_all = np.asarray(d["k"], float)
    R_row = DL.desi_resolution_R(z_all)
    keep = ((z_all >= 2.2 - 1e-6) & (z_all <= 4.2 + 1e-6)
            & (k_all > DL.DESI_KMIN) & (k_all < 0.5 * np.pi / R_row))
    k = k_all[keep]
    Rz = DL.desi_resolution_R(z_all[keep])

    b = 0.02
    x = 2.0 * b * k ** 2 * Rz ** 2
    ours = np.exp(x)                      # our template
    cup1d = 1.0 + x                       # cup1d get_contamination (linear)
    frac = np.abs(ours - cup1d) / np.abs(cup1d)
    assert frac.max() <= 0.01, f"exp-vs-linear in-band gap {frac.max()*100:.3f}% exceeds 1% at b=0.02"

    # our _resolution_factor is exactly the exp form (differentiable knob) -> matches `ours`.
    ours_fn = np.asarray(DL._resolution_factor(k, Rz, b_res=b))
    np.testing.assert_allclose(ours_fn, ours, rtol=0, atol=1e-13)
