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


if __name__ == "__main__":
    test_legb_legacy_forward_matches_golden()
    print("[golden] legacy forward matches reference (rtol 1e-12) on all legs.")
