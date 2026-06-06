"""Freeze the LF Leg-B forward (P_model, C_total) on BOTH legs at a FIXED (theta9, tau0, alpha=(3,))
into tests/golden/legb_lf_golden.npz — the golden-regression guard for the closing-step refactor.

WHY: the (B+slope) factorized-alpha change (checkpoint 2026-06-06, spec §7 / E-C) is a ~40-60 line
contract change to the per-leg alpha assembly in data_likelihood.predict_P_obs_on_leg /
_data_loglik_legcore. The LEGACY path takes a (3,) alpha broadcast over z; the refactor must reproduce
it BYTE-FOR-BYTE on that legacy path. This script pins the reference; tests/test_legb_golden.py asserts
allclose(rtol=1e-12, atol=0) after the refactor. Hard gate: nothing in the refactor lands until green.

Uses the PRODUCTION ctx (build_legb_ctx: real DESI+KS legs, pf_stats, cross-class rho_zb C_emu,
alpha_centres, cemu_inflate) so the freeze is the exact config the closure runs. The fixed inputs are
deterministic (no RNG): theta9 = 0.5 (unit-cube centre), tau0 = becker13 on each leg's z, alpha = the
cache-median structural w_c. Regenerate ONLY with an explicit reason (the legacy forward is meant to be
immutable); the test compares against whatever is committed here.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/make_legb_golden.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import build_legb_ctx
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF

OUT = Path("/home/mfho/hcd_priya/tests/golden/legb_lf_golden.npz")

# Production ctx with the cross-class C_emu (the config the closure runs).
ctx, d = build_legb_ctx()

# FIXED, DETERMINISTIC inputs (no RNG): unit-cube centre theta, becker13 tau0 per leg,
# cache-median structural w_c as the (3,) alpha (the legacy broadcast path).
theta9 = jnp.full((9,), 0.5)
alpha3 = jnp.asarray(np.median(d["w_c_cache"][:, 1:], axis=0))   # (3,) LLS, subDLA, DLA

saved = {"theta9": np.asarray(theta9), "alpha3": np.asarray(alpha3)}
meta = dict(note="legacy (3,)-alpha broadcast forward; build_legb_ctx production config; "
                 "theta=0.5 unit, tau0=becker13(z_leg), alpha=median w_c_cache")
for leg in ctx.legs:
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
    # predict_P_obs_on_leg wants ONE (K,) core; _data_loglik_legcore passes the z-MEAN of the
    # per-leg-z stack (closure_legb.py) — mirror that exactly so the golden is the production path.
    core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)     # (n_z,K) -> (K,)
    P_model, C_total = DL.predict_P_obs_on_leg(
        ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats,
        dla_core=core, cache_k=ctx.cache_k, leg=leg,
        sigma_zb=szb, alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
    P = np.asarray(P_model); C = np.asarray(C_total)
    assert np.isfinite(P).all() and np.isfinite(C).all(), f"{leg.name}: non-finite golden"
    saved[f"{leg.name}_P"] = P
    saved[f"{leg.name}_C"] = C
    saved[f"{leg.name}_tau0"] = np.asarray(tau0_vec)
    saved[f"{leg.name}_k"] = np.asarray(leg.k)
    print(f"[golden] {leg.name}: P{P.shape} C{C.shape}  P[:3]={np.round(P[:3],6)}  "
          f"diagC[:3]={np.round(np.diag(C)[:3],6)}")

saved["_meta"] = np.array(repr(meta))
saved["leg_names"] = np.array([leg.name for leg in ctx.legs])
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **saved)
print(f"[golden] wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KiB, legs={list(saved['leg_names'])})")
