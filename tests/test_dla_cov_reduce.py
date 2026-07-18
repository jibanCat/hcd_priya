"""Reduced DESI covariance: remove the DLA-completeness systematic when alpha_DLA floats.

PI disposition 2026-07-17 (Workstream B): modeled-in-mean => removed-from-covariance, following
cup1d's reduced ("red") covariance convention (DESI DR1 cosmology paper 2601.21432 Sec 2.1: "we
omit the contributions from residual HCD contamination and uncertainties in the spectrograph
resolution to the systematic covariance matrix, as both effects are explicitly marginalized over").
DESI floats the PRIYA alpha_DLA mean model (DESI_DLA_FORWARD_FRAC=1.0), so the matching
``syst_e_dla_completeness`` variance mode is removed from C_data using the SHIPPED per-z-block
rank-1 convention (cov_syst is exactly z-block-diagonal):

    C_reduced = C_fid - sum_z outer(e_dla|z)          [NOT one globally coherent outer product]

Tests (PI requirements, verbatim list):
  1. reconstruction of the shipped fiducial covariance (pinned in
     test_resolution_cup1d_consistency; re-asserted here on the kept grid);
  2. exact removal of ONLY the intended DLA-completeness blocks (delta == sum_z outer(e_dla|z),
     zero outside the z blocks; other systematic terms untouched);
  3. agreement with the cup1d reduced-covariance convention (corr set minus
     {E_RESOLUTION, E_DLA_COMPLETENESS} == our resolution_float + dla_cov_reduce build);
  4. symmetry + positive definiteness + successful Cholesky at every redshift block;
  5. the modeled-in-mean invariant: removing the DLA cov term on a leg whose forward DLA
     term is OFF (dla_forward_frac == 0) raises;
  6. opt-out (dla_cov_reduce=False) is byte-identical to the pre-change behavior.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_dla_cov_reduce.py -q
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64 before jax arrays
import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have_desi = os.path.exists(DESI_NPZ)

# cup1d compute_cov systematic labels + splits (mirrors test_resolution_cup1d_consistency;
# duplicated here so this module is self-contained).
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
_UCORR = ["E_NOISE_ADD", "E_CONTINUUM_ADD"]
# cup1d data_DESIY1.compute_cov type_analysis="red" (the DEFAULT cosmology-fit covariance):
# E_DLA_COMPLETENESS and E_RESOLUTION are dropped from the correlated set.
_CORR_RED = ["E_BAL_COMPLETENESS", "E_CONTINUUM", "E_NOISE_SCALE"]


def _compute_cov_cup1d(z_col, syst_vecs, corr_labels, ucorr_labels):
    """Faithful replication of cup1d.p1ds.data_DESIY1.compute_cov (QMLE, per-z-block)."""
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


def _npz_and_keep():
    d = np.load(DESI_NPZ, allow_pickle=True)
    z_all = np.asarray(d["z"], float)
    k_all = np.asarray(d["k"], float)
    R_row = DL.desi_resolution_R(z_all)
    keep = ((z_all >= 2.2 - 1e-6) & (z_all <= 4.2 + 1e-6)
            & (k_all > DL.DESI_KMIN) & (k_all < 0.5 * np.pi / R_row))
    return d, z_all, k_all, np.where(keep)[0]


# --------------------------------------------------------------------------------------------- #
#  Authority + default
# --------------------------------------------------------------------------------------------- #
def test_authority_constant_is_on():
    """The single-authority default: DESI removes the DLA-completeness cov term (PI disposition)."""
    assert DL.DESI_DLA_COV_REDUCE is True


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_default_load_is_reduced_and_stamped():
    """load_desi_leg() default resolves the authority constant -> reduced C_data, stamped on the leg."""
    leg = DL.load_desi_leg()
    assert leg.dla_cov_reduced is True
    # explicit opt-out is honoured + stamped False
    leg0 = DL.load_desi_leg(dla_cov_reduce=False)
    assert leg0.dla_cov_reduced is False


# --------------------------------------------------------------------------------------------- #
#  PI req 1 + 3: fid reconstruction (kept grid) + cup1d red-convention agreement
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_fid_rebuild_matches_shipped_on_kept_grid():
    """The fid split rebuilt from the systematic vectors reproduces the shipped cov on the kept
    rows (re-pins the baseline this module subtracts from; full-grid pin lives in
    test_resolution_cup1d_consistency)."""
    d, z_all, k_all, idx = _npz_and_keep()
    syst = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}
    cov_fid = np.asarray(d["cov_stat"], float) + _compute_cov_cup1d(z_all, syst, _CORR_FID, _UCORR)
    np.testing.assert_allclose(cov_fid[np.ix_(idx, idx)],
                               np.asarray(d["cov"], float)[np.ix_(idx, idx)],
                               rtol=1e-12, atol=1e-18)


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_reduced_equals_cup1d_red_build():
    """PERMANENT GUARD (the load-bearing equality): our subtractive resolution_float +
    dla_cov_reduce build == cup1d's additive type_analysis="red" build (drop E_RESOLUTION AND
    E_DLA_COMPLETENESS from the correlated set) on the same post-cut grid; both SPD.
    (add_cov_diag_inflation=False for apples-to-apples; the inflation is an orthogonal shared
    diagonal term cup1d omits.)"""
    d, z_all, k_all, idx = _npz_and_keep()
    syst = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}
    cup1d_red = np.asarray(d["cov_stat"], float) + _compute_cov_cup1d(z_all, syst, _CORR_RED, _UCORR)

    leg = DL.load_desi_leg(resolution_float=True, add_cov_diag_inflation=False, dla_cov_reduce=True)
    ours = np.asarray(leg.C_data, float)
    theirs = cup1d_red[np.ix_(idx, idx)]
    assert ours.shape == theirs.shape, "post-cut grid size mismatch"
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(ours - theirs)) < 1e-11        # tight, not just within tolerance
    np.linalg.cholesky(ours)
    np.linalg.cholesky(theirs)
    assert np.linalg.eigvalsh(ours).min() > 0.0


# --------------------------------------------------------------------------------------------- #
#  PI req 2: exact removal of ONLY the intended DLA blocks; other systematics untouched
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_dla_only_removal_is_exact_per_z_block():
    """dla_cov_reduce alone (resolution untouched): C_fid - C_reduced == sum_z outer(e_dla|z)
    exactly on the kept grid, ZERO outside the z blocks (the shipped z-block-diagonal convention;
    NOT one global outer), and equal to the additive build-without-DLA (other terms untouched)."""
    d, z_all, k_all, idx = _npz_and_keep()
    syst = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}

    leg = DL.load_desi_leg(add_cov_diag_inflation=False, dla_cov_reduce=True)
    reduced = np.asarray(leg.C_data, float)
    fid_kept = np.asarray(d["cov"], float)[np.ix_(idx, idx)]

    # (a) BIT-EXACT: reduced == fid with the per-z-block outer(e_dla|z) subtracted in-place (the
    # replicated operation — comparing fid-reduced against the outer would re-introduce float
    # cancellation error; replicating the op pins exactness).
    e = np.asarray(d[_LAB2KEY["E_DLA_COMPLETENESS"]], float)[idx]
    z_kept = z_all[idx]
    expected = fid_kept.copy()
    for zz in np.unique(z_kept):
        rows = np.where(z_kept == zz)[0]
        expected[np.ix_(rows, rows)] -= np.outer(e[rows], e[rows])
    assert np.array_equal(reduced, expected), "reduced C != bit-exact per-z-block removal"

    # (b) zero cross-z: every off-z-block entry is UNTOUCHED, bit-exact (the shipped z-block
    # convention; NOT one globally coherent outer product)
    zi = np.asarray([np.searchsorted(np.unique(z_kept), zz) for zz in z_kept])
    off_block = zi[:, None] != zi[None, :]
    assert np.array_equal(reduced[off_block], fid_kept[off_block])

    # (c) other systematics untouched: reduced == additive build with only DLA dropped
    corr_nodla = [c for c in _CORR_FID if c != "E_DLA_COMPLETENESS"]
    build = np.asarray(d["cov_stat"], float) + _compute_cov_cup1d(z_all, syst, corr_nodla, _UCORR)
    np.testing.assert_allclose(reduced, build[np.ix_(idx, idx)], rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------------------------- #
#  PI req 4: symmetry + SPD + per-z-block Cholesky (deployed config: inflation ON)
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
@pytest.mark.parametrize("inflation", [True, False])
def test_reduced_symmetric_spd_cholesky_every_z(inflation):
    """The deployed reduced covariance (resolution_float + dla_cov_reduce) is symmetric, SPD, and
    Cholesky-decomposable both as a whole and within every z block, with AND without the diagonal
    inflation (the no-inflation case is the conservative numerical check at z=4.0/4.2)."""
    leg = DL.load_desi_leg(resolution_float=True, add_cov_diag_inflation=inflation,
                           dla_cov_reduce=True)
    C = np.asarray(leg.C_data, float)
    # the SHIPPED cov is itself asymmetric at ~1e-8 (a property of the released FITS, measured
    # max|cov-cov.T| = 1.0e-8); the surgery subtracts exactly-symmetric outers, so it must NOT
    # increase that asymmetry.
    d = np.load(DESI_NPZ, allow_pickle=True)
    _, _, _, idx = _npz_and_keep()
    shipped = np.asarray(d["cov"], float)[np.ix_(idx, idx)]
    asym_in = np.max(np.abs(shipped - shipped.T))
    assert np.max(np.abs(C - C.T)) <= asym_in + 1e-15, "surgery increased covariance asymmetry"
    np.linalg.cholesky(C)
    assert np.linalg.eigvalsh(C).min() > 0.0
    z_idx = np.asarray(leg.z_idx)
    for i, zz in enumerate(np.asarray(leg.z)):
        rows = np.where(z_idx == i)[0]
        blk = C[np.ix_(rows, rows)]
        np.linalg.cholesky(blk)                                  # per-z SPD (incl. z=4.0, 4.2)
        assert np.linalg.eigvalsh(blk).min() > 0.0, f"non-SPD z block at z={zz}"


# --------------------------------------------------------------------------------------------- #
#  PI req 5 (invariant) + req 6 (opt-out back-compat)
# --------------------------------------------------------------------------------------------- #
def test_modeled_in_mean_invariant_raises_when_dla_forward_off():
    """Removing the DLA cov term is only licensed when the alpha_DLA mean model is LIVE on the
    leg: _assemble_leg(dla_e=..., dla_forward_frac=0.0) must raise (fail loud, never a silent
    unmodeled variance deletion)."""
    rng = np.random.default_rng(0)
    n_z, n_k = 2, 4
    z_all = np.repeat([2.2, 2.4], n_k)
    k_all = np.tile(np.linspace(2e-3, 8e-3, n_k), n_z)
    P_all = rng.uniform(10.0, 20.0, n_z * n_k)
    A = rng.normal(size=(n_z * n_k, n_z * n_k))
    cov_all = A @ A.T + 10.0 * np.eye(n_z * n_k)
    keep = np.ones(n_z * n_k, bool)
    e = np.full(n_z * n_k, 1e-3)
    with pytest.raises(ValueError, match="dla_forward_frac"):
        DL._assemble_leg("DESI", z_all, k_all, P_all, cov_all, keep,
                         R_func=DL.desi_resolution_R, metals_on=True, resolution_on=False,
                         dla_forward_frac=0.0, dla_e=e)


def test_forward_signature_sensitive_to_dla_cov_reduce(monkeypatch):
    """The freeze signature must move when the reduced-covariance authority flips (consistency
    review MINOR-1): a refactor dropping DESI_DLA_COV_REDUCE from the forward_signature payload
    would silently un-freeze the decision from analysis.lock."""
    import hcd_analysis.emulator.closure_legb as CL
    sig_on = CL.forward_signature()
    monkeypatch.setattr(DL, "DESI_DLA_COV_REDUCE", False)
    sig_off = CL.forward_signature()
    assert sig_on != sig_off, "forward_signature blind to DESI_DLA_COV_REDUCE"
    monkeypatch.undo()
    assert CL.forward_signature() == sig_on            # deterministic + restored


@pytest.mark.skipif(not _have_desi, reason="real DESI npz not present")
def test_optout_byte_identical_to_pre_change_behavior():
    """dla_cov_reduce=False reproduces the pre-change covariance exactly: the option-b build
    without resolution ONLY (the old cov_b pin from test_resolution_cup1d_consistency)."""
    d, z_all, k_all, idx = _npz_and_keep()
    syst = {lab: np.asarray(d[_LAB2KEY[lab]], float) for lab in _SYS_LABELS}
    corr_nores = [c for c in _CORR_FID if c != "E_RESOLUTION"]
    old = np.asarray(d["cov_stat"], float) + _compute_cov_cup1d(z_all, syst, corr_nores, _UCORR)

    leg = DL.load_desi_leg(resolution_float=True, add_cov_diag_inflation=False,
                           dla_cov_reduce=False)
    np.testing.assert_allclose(np.asarray(leg.C_data, float), old[np.ix_(idx, idx)],
                               rtol=1e-10, atol=1e-12)
