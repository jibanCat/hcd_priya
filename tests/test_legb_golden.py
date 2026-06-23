"""Golden-regression guard for the LF Leg-B forward (the closing-step refactor gate).

The (B+slope) factorized-alpha change moves per-leg alpha(z) assembly into the binding and adds a
per-leg anchor — a real contract change (spec 2026-06-06 §7 / E-C). The LEGACY (3,)-alpha broadcast
path MUST stay byte-for-byte identical. This test pins (P_model, C_total) on both legs against the
committed reference tests/golden/legb_lf_golden.npz (regenerate ONLY with explicit reason via
scripts/make_legb_golden.py). rtol=1e-12 (bit-level intent; absolute tol 0).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_golden.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import build_legb_ctx
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "legb_lf_golden.npz")
MF_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "legb_mf_golden.npz")
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="golden reference not generated")
@pytest.mark.skipif(not os.path.exists(_DESI_NPZ), reason="DESI data not present")
def test_legb_legacy_forward_matches_golden():
    """The legacy (3,)-alpha broadcast forward reproduces the frozen (P_model, C_total) exactly."""
    g = np.load(GOLDEN, allow_pickle=True)
    ctx, d = build_legb_ctx()
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])

    for leg in ctx.legs:
        tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)
        P_model, C_total = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats,
            dla_core=core, cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        P = np.asarray(P_model); C = np.asarray(C_total)
        P_ref = g[f"{leg.name}_P"]; C_ref = g[f"{leg.name}_C"]
        # bit-level intent: relative tol 1e-12, no absolute slack (jax-traps: never rtol=0 on
        # large-magnitude reductions, but these are O(1-20) P1D values + O(1) covariances).
        assert P.shape == P_ref.shape, f"{leg.name} P shape {P.shape} != golden {P_ref.shape}"
        assert C.shape == C_ref.shape, f"{leg.name} C shape {C.shape} != golden {C_ref.shape}"
        np.testing.assert_allclose(P, P_ref, rtol=1e-12, atol=0.0,
                                   err_msg=f"{leg.name} P_model regressed vs golden")
        np.testing.assert_allclose(C, C_ref, rtol=1e-12, atol=0.0,
                                   err_msg=f"{leg.name} C_total regressed vs golden")


# --------------------------------------------------------------------------- #
#  MF-PATH golden (Task 1.2 — res_corr anchor + upcoming alpha-nuisance guard).
#
#  The LEGACY golden above calls predict_P_obs_on_leg with mf=None, so it does NOT
#  exercise the multi-fidelity / res_corr forward at all.  After Task 1.1 the
#  production MF forward (build_legb_ctx(with_mf=True) -> ctx.mf ->
#  _predict_P_obs_mf -> _mf_corr_on_cache -> mf.res_corr) picks up the ANCHORED
#  res_corr (res_corr -> 1 below 5x the L15 box fundamental, anchor_mult=5.0 by
#  DEFAULT) automatically.  This golden PINS that production MF forward at the
#  fiducial theta so:
#    * Task 1.3 (the alpha-res nuisance, threaded forward-only into _mf_corr_on_cache
#      as `g + alpha_z * log_rc`) MUST reproduce this byte-for-byte at alpha_z==1
#      (its no-op default: alpha_res_amp=1, alpha_res_slope=0); and
#    * any future res_corr / MF forward edit is caught at the production default.
#
#  CONTRACT (see scripts/make_legb_golden.py mf= arm — your CS partner generates the
#  npz):
#    * config  : build_legb_ctx(with_mf=True, mf_with_floor=True) — the production
#                MF correction (fold 0, all-HR head, anchored res_corr default) on
#                the real DESI+KS legs with the cross-class C_emu (use_xclass=True).
#    * forward : predict_P_obs_on_leg(ctx.model, theta9, tau0_vec, alpha3, mf=ctx.mf,
#                ...) — the SAME call shape as the LF golden PLUS mf=ctx.mf (no floor /
#                emucoh / shape cov: the golden pins the bare anchored-res_corr MF
#                forward, which is exactly the path the alpha nuisance scales).
#    * theta   : theta9 = 0.5 (unit-cube centre); tau0 = becker13(z_leg);
#                alpha3 = median structural w_c (the legacy (3,) broadcast) — IDENTICAL
#                fiducial to the LF golden, so the ONLY difference vs legb_lf_golden is
#                the MF correction (anchored res_corr * exp(g)).
#    * pinned  : per leg, P_model (the load-bearing array — the alpha nuisance & res_corr
#                enter P_model only) AND C_total, for BOTH legs (DESI is the primary
#                high-k, n_s-driving leg the anchor/alpha most affect).
#    * rtol    : 1e-10, atol 0 (the plan's MF-golden tolerance; the MF forward adds the
#                res_corr table interp + extra reductions vs the LF path, so 1e-10 not
#                the LF golden's 1e-12 — still a bit-level-intent regression guard).
@pytest.mark.skipif(not os.path.exists(MF_GOLDEN), reason="MF golden reference not generated")
@pytest.mark.skipif(not os.path.exists(_DESI_NPZ), reason="DESI data not present")
def test_legb_mf_golden():
    """The production MF forward (anchored res_corr, alpha-OFF) reproduces the frozen
    (P_model, C_total) on both legs — the regression guard for the res_corr anchor and
    the upcoming alpha-res nuisance (which must be a no-op at alpha_z==1)."""
    g = np.load(MF_GOLDEN, allow_pickle=True)
    # PRODUCTION MF ctx: with_mf=True attaches ctx.mf (the anchored-res_corr correction).
    ctx, d = build_legb_ctx(with_mf=True, mf_with_floor=True)
    assert ctx.mf is not None, "build_legb_ctx(with_mf=True) did not attach ctx.mf"
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])

    for leg in ctx.legs:
        tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)
        # mf=ctx.mf routes step (1) through _predict_P_obs_mf -> the ANCHORED res_corr.
        P_model, C_total = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats,
            dla_core=core, cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb,
            mf=ctx.mf)
        P = np.asarray(P_model); C = np.asarray(C_total)
        P_ref = g[f"{leg.name}_P"]; C_ref = g[f"{leg.name}_C"]
        assert P.shape == P_ref.shape, f"{leg.name} P shape {P.shape} != golden {P_ref.shape}"
        assert C.shape == C_ref.shape, f"{leg.name} C shape {C.shape} != golden {C_ref.shape}"
        # P_model is the load-bearing array (the alpha-res nuisance + res_corr enter here);
        # rtol 1e-10, atol 0 — the MF-golden tolerance (the res_corr table interp adds a few
        # extra reductions vs the LF path; jax-traps: avoid rtol=0 on O(1-20) P1D reductions).
        np.testing.assert_allclose(P, P_ref, rtol=1e-10, atol=0.0,
                                   err_msg=f"{leg.name} MF P_model regressed vs golden")
        np.testing.assert_allclose(C, C_ref, rtol=1e-10, atol=0.0,
                                   err_msg=f"{leg.name} MF C_total regressed vs golden")


if __name__ == "__main__":
    test_legb_legacy_forward_matches_golden()
    print("[golden] legacy forward matches reference (rtol 1e-12) on all legs.")
    if os.path.exists(MF_GOLDEN):
        test_legb_mf_golden()
        print("[golden] MF forward matches reference (rtol 1e-10) on all legs.")
    else:
        print("[golden] MF golden not generated yet — run scripts/make_legb_golden.py mf arm.")
