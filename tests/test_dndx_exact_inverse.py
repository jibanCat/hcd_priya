"""Tests for dndx_wc.alpha_to_dndx_exact -- the EXACT fail-loud alpha -> dN/dX inverse
(readout defect B, 2026-07-22) -- plus the frozen-behavior regression pin on the OLD
approximate alpha_to_dndx (retained solely to reproduce pre-2026-07-22 artifacts).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_dndx_exact_inverse.py -q
"""
import jax
jax.config.update("jax_enable_x64", True)   # the readout is float64 end to end

import numpy as np
import jax.numpy as jnp
import pytest

from hcd_analysis.emulator.dndx_wc import (
    w_c_corrected, alpha_to_dndx, alpha_to_dndx_exact, _LOG_FLOOR,
)

# Every deployed z in [2.2, 4.6]: the union of the per-survey analysis grids
# (DESI np.arange(2.2, 4.21, 0.1); KS/DESI+KS np.arange(2.4, 4.61, 0.1);
# eBOSS np.arange(2.2, 4.61, 0.1)) -- i.e. the 0.1-spaced grid over [2.2, 4.6].
Z_DEPLOYED = np.round(np.arange(2.2, 4.601, 0.1), 10)
XBAR = 0.04 * (1.0 + Z_DEPLOYED) ** 2          # plausible Xbar(z) magnitudes (0.41..1.25)

# The deployed KS prior CENTRE alpha at z=4.6 under the current (pre-reparameterization)
# KS prior: outside the occupancy simplex (sum = 1.0473 > 1). The OLD map silently
# saturated its LLS slot at -log(_LOG_FLOOR)/Xbar; the EXACT map must refuse it.
KS_Z46_ALPHA = np.array([0.8802, 0.1573, 0.0098])
SATURATED_MU = -np.log(1e-12)                  # = 27.631021115928547


# --------------------------------------------------------------- (a) machine-precision round trip
def test_roundtrip_through_wc_corrected_machine_precision():
    """forward w_c_corrected -> alpha -> alpha_to_dndx_exact recovers dN/dX to machine
    precision (median rel err < 1e-12) on a grid of valid dndx covering every deployed z."""
    rng = np.random.default_rng(20260722)
    n_draws = 200
    dndx = rng.uniform(1e-4, 1.5, size=(n_draws, len(Z_DEPLOYED), 3))
    alpha = np.asarray(w_c_corrected(jnp.asarray(dndx), jnp.asarray(XBAR),
                                     jnp.asarray(Z_DEPLOYED)))[..., 1:]
    rec = alpha_to_dndx_exact(alpha, XBAR, Z_DEPLOYED)
    rel = np.abs(rec / dndx - 1.0)
    assert np.median(rel) < 1e-12, f"median rel err {np.median(rel):.3e}"
    # the approximate map fails this by ~10 orders of magnitude (median ~2e-3)
    assert np.max(rel) < 1e-8, f"max rel err {np.max(rel):.3e}"


def test_roundtrip_all_zero_alpha_is_zero_dndx():
    # alpha = 0 is ON the valid domain boundary in the harmless direction (mu = 0 exactly).
    out = alpha_to_dndx_exact(np.zeros(3), 0.642, 3.0)
    np.testing.assert_array_equal(out, np.zeros(3))


# ------------------------------------------------------------------- (b, c) fail-loud raise mode
def test_raise_on_sum_alpha_over_one():
    with pytest.raises(ValueError, match="outside the occupancy simplex") as ei:
        alpha_to_dndx_exact(KS_Z46_ALPHA, 0.9, 4.6)
    msg = str(ei.value)
    assert "1 input row(s)" in msg                     # violation count reported
    assert f"{float(KS_Z46_ALPHA.sum())!r}" in msg     # worst sum(alpha) reported
    assert "pre-2026-07-22" in msg and "mask" in msg   # historical-draw escape hatch named


def test_raise_on_negative_alpha():
    with pytest.raises(ValueError, match="negative alpha"):
        alpha_to_dndx_exact(np.array([-0.01, 0.05, 0.01]), 0.642, 3.0)


def test_raise_counts_both_violation_kinds_in_a_batch():
    bad = np.array([[0.1, 0.05, 0.01],      # valid
                    [0.9, 0.2, 0.01],       # sum >= 1
                    [-0.1, 0.05, 0.01]])    # negative
    with pytest.raises(ValueError) as ei:
        alpha_to_dndx_exact(bad, 0.642, 3.0)
    msg = str(ei.value)
    assert "2 input row(s)" in msg
    assert "negative alpha" in msg and "outside the occupancy simplex" in msg


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        alpha_to_dndx_exact(np.array([0.1, 0.05, 0.01]), 0.642, 3.0, mode="clip")


# ----------------------------------------------------------------------------- (d) mask mode
def test_mask_mode_nan_and_mask_consistent_with_raise_domain():
    rows = np.array([[0.10, 0.05, 0.01],     # valid
                     [0.90, 0.20, 0.01],     # sum >= 1  -> invalid
                     [-0.10, 0.05, 0.01],    # negative  -> invalid
                     [0.50, 0.25, 0.25],     # sum == 1.0 EXACTLY (dyadic) -> invalid (strict >)
                     [0.0, 0.0, 0.0]])       # boundary-valid (all zero)
    dd, valid = alpha_to_dndx_exact(rows, 0.642, 3.0, mode="mask")   # no exception
    np.testing.assert_array_equal(valid, [True, False, False, False, True])
    # NaN exactly on the invalid rows, finite on the valid ones
    np.testing.assert_array_equal(np.isnan(dd).all(axis=-1), ~valid)
    assert np.all(np.isfinite(dd[valid]))
    # mask agrees with raise-mode row by row
    for r, ok in zip(rows, valid):
        if ok:
            alpha_to_dndx_exact(r, 0.642, 3.0)           # must not raise
        else:
            with pytest.raises(ValueError):
                alpha_to_dndx_exact(r, 0.642, 3.0)
    # valid rows carry the same values as raise mode
    np.testing.assert_array_equal(dd[0], alpha_to_dndx_exact(rows[0], 0.642, 3.0))


# ------------------------------------- (e) frozen-behavior regression pin on the OLD function
def test_old_map_still_silently_saturates_ks_z46_centre_and_exact_refuses():
    """FROZEN-ARTIFACT-REPRODUCTION CONTRACT. Pre-2026-07-22 paper artifacts were built by
    the approximate alpha_to_dndx, which silently SATURATES the LLS slot of the deployed KS
    z=4.6 centre (sum(alpha) = 1.0473, outside the simplex) at mu = -log(1e-12)/Xbar =
    27.631021/Xbar. The old function must keep doing exactly that (byte-level artifact
    reproduction), and alpha_to_dndx_exact must REFUSE the same input."""
    xbar = 0.9
    old = np.asarray(alpha_to_dndx(jnp.asarray(KS_Z46_ALPHA), jnp.asarray(xbar),
                                   jnp.asarray(4.6)))
    assert abs(old[0] * xbar - SATURATED_MU) < 1e-9, (
        f"old alpha_to_dndx LLS slot = {old[0] * xbar!r}, expected the saturated "
        f"-log(1e-12) = {SATURATED_MU!r}: the frozen-artifact reproduction contract broke")
    assert _LOG_FLOOR == 1e-12                    # the clip floor the pin derives from
    assert np.all(np.isfinite(old))               # old behavior: finite everywhere, no raise
    with pytest.raises(ValueError, match="outside the occupancy simplex"):
        alpha_to_dndx_exact(KS_Z46_ALPHA, xbar, 4.6)


# ------------------------------------------------- (f) near-boundary conditioning documentation
def test_near_boundary_valid_input_is_finite_and_roundtrips():
    """1 - sum(alpha) = 1e-6: domain-valid, and per the documented conditioning law
    (round-trip error ~ 1/w0_clean) still numerically fine -- the docstring places the
    unreliability threshold at 1 - sum(alpha) < ~1e-12, six orders below this."""
    a = np.array([0.7, 0.2, 0.1 - 1e-6])
    assert 0 < (1.0 - a.sum()) < 2e-6
    dd = alpha_to_dndx_exact(a, 0.642, 3.0)
    assert np.all(np.isfinite(dd)) and np.all(dd > 0)
    # round trip back through the deployed forward: alpha -> dndx -> alpha
    a_rt = np.asarray(w_c_corrected(jnp.asarray(dd), jnp.asarray(0.642),
                                    jnp.asarray(3.0)))[1:]
    rel = np.max(np.abs(a_rt / a - 1.0))
    assert rel < 1e-8, f"near-boundary (1e-6) round trip rel err {rel:.3e}"


def test_docstring_states_domain_conditioning_and_supersession():
    doc = alpha_to_dndx_exact.__doc__
    assert "1e-12" in doc and "UNRELIABLE" in doc      # the conditioning threshold
    assert "1e-14" in doc                              # the quoted degradation point
    assert "1/w0_clean" in doc                         # the conditioning law
    assert "SUPERSEDES" in doc and "alpha_to_dndx" in doc
    # and the old function carries the deprecation banner + corrected error claim
    old_doc = alpha_to_dndx.__doc__
    assert "DEPRECATED" in old_doc and "alpha_to_dndx_exact" in old_doc
    assert "~2e-3" in old_doc and "27.631021" in old_doc
