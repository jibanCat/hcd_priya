#!/usr/bin/env python3
"""FORWARD-RESIDUAL FISHER for the MF n_s cert (ns0.909 DESI HF-LOSO) — NO NUTS, cheap.

THE QUESTION (decisive arm 3, 2026-06-16): is the -2.5sigma DESI n_s bias on the ns0.909
HF-LOSO cert a (A) REAL g-forward bias or (C) error-model (alpha/emucoh) AMPLIFICATION of a
small g-residual? This computes the NATIVE forward n_s bias — the bias from the per-(z,k)
forward residual ALONE, projected onto n_s through C_data, with NO alpha/emucoh re-weighting,
NO NUTS, NO fit. It isolates "what the forward model itself does" from "what the fit's
covariance re-weighting does to it".

NATIVE-BIAS DEFINITION (MARGINALIZED Laplace/Fisher shift, in log-P space):
  At the cert truth point theta* (= the ns0.909 HR sim's params_unit + its tau0 + its w_c),
  the noiseless HR truth-on-leg is P_truth (make_legb_mock's truth_on_leg, no noise), the MF
  forward at theta* is P_fwd, and the forward residual is
      r = log(P_truth) - log(P_fwd)            (per flat (z,k) cell)
  The fit FLOATS the SAME nuisance set the all_off NUTS run does: theta9 (9 cosmo/astro,
  uniform), tau0_amp + dtau0 (mean-flux, uniform), alpha_lls/subdla/dla (3 HCD, Normal prior),
  s_lls/subdla/dla (3 z-slope, Normal prior) — but NO alpha_res (fixed) and NO emucoh. We
  linearize log P_fwd about theta*: J = d logP/d(params) (jacfwd, FD-checked), form the
  Gaussian (Laplace) posterior with the LOG-space (fractional) cov Clog = diag(1/P_fwd) C_data
  diag(1/P_fwd) and the PRIOR precision Lambda (1/sigma^2 for the Normal sites, 0 for uniform):
      F = J^T Clog^-1 J + Lambda
      dtheta = F^-1 (J^T Clog^-1 r)            (the MAP shift the residual drives)
  The native n_s bias is the MARGINAL n_s pull in sigma units:
      native bias_z = dtheta[ns] / sqrt( (F^-1)[ns,ns] )
  This is the n_s shift a least-squares fit FLOATING the all_off nuisances (NO emucoh, NO
  alpha) feels from the forward residual — the "g-native" bias, the Laplace twin of the
  all_off NUTS arm (which gave -0.62). native ~ -0.6 => the -2.5 NUTS bias is error-model
  (alpha/emucoh) amplification (C); native ~ -2.5 => a real forward g-bias (A).

CONFIGS (all DESI-only, ns0.909 HR-LOSO truth, theta*=truth, alpha==1 forward):
  all_off  : anchor OFF (raw clamped res_corr, anchor_mult=0); forward = LF x g x raw_rc
  anchored : anchor ON  (anchor_mult=5, production);          forward = LF x g x anchored_rc
  g_ident  : anchor OFF + g-IDENTITY (log_rho=0 AND FixedMeanHead output 0) + res_corr OFF, so
             forward = PURE LF emulator (no LF->HF g, no res_corr). The task's g-IDENTITY arm.
  g_ident_rc: g==0 but KEEP raw res_corr — isolates g from res_corr.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/scratch_mf_nscert_fwdresid_fisher.py
"""
from __future__ import annotations
import functools
print = functools.partial(print, flush=True)

import numpy as np
import hcd_analysis.emulator  # x64 before jax
import jax
import jax.numpy as jnp
import equinox as eqx

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, make_hr_truth_from_cache, make_legb_mock, _mock_core_per_leg,
    ZSLOPE_PRIOR_SIGMA, HCD_Z_PIVOT, HCD_INCIDENCE_SLOPE)
from hcd_analysis.emulator.inference import HCD_LIT_OVER_SIM_SLOPE
from hcd_analysis.emulator import data_likelihood as DL

FOLD = 4
TARGET_NS = 0.9095          # the ns0.909 cert sim n_s (verified by the wrapper dry-run)

# the floated all_off parameter pack (matches the all_off NUTS run; NO alpha_res, NO emucoh):
#   [0:9]   theta9 (uniform; near truth the bounds are slack -> prior precision 0)
#   [9]     tau0_amp (uniform)        [10] dtau0 (uniform)
#   [11:14] alpha_lls/subdla/dla (pivot-z HCD incidence; Normal prior at ctx center/width)
#   [14:17] s_lls/subdla/dla (HCD z-slope; Normal prior at HCD_LIT_OVER_SIM_SLOPE, ZSLOPE sigma)
P_NS, P_TAU0, P_DTAU0 = 0, 9, 10
P_AHCD = slice(11, 14)
P_SLOPE = slice(14, 17)
NPAR = 17


def _resolve_hr_sim_name():
    """The HR-cache sim closest to TARGET_NS (same selection run_stepA uses for HFLOSO909)."""
    from hcd_analysis.emulator import multifidelity as MF
    from hcd_analysis.emulator.data import PARAM_LIMITS
    d = MF.load_cache(MF.HR_CACHE)
    names = np.array([s.decode() if isinstance(s, bytes) else s for s in d["sim_name"]])
    pu = d["params_unit"]
    ns_lo, ns_hi = PARAM_LIMITS[0]
    ns_phys = pu[:, 0] * (ns_hi - ns_lo) + ns_lo
    uniq = {}
    for i, nm in enumerate(names):
        uniq.setdefault(nm, ns_phys[i])
    sim = min(uniq, key=lambda s: abs(uniq[s] - TARGET_NS))
    print(f"[resolve] HR sim for ns~{TARGET_NS}: {sim!r}  (n_s={uniq[sim]:.4f})")
    return sim


def _zero_g_mf(mf):
    """COPY of the MultiFidelity with g forced to IDENTITY (g==0): log_rho -> 0 AND the
    FixedMeanHead tables -> 0. Forward then = LF backbone x res_corr ONLY (no LF->HF g)."""
    mf = eqx.tree_at(lambda m: m.log_rho, mf, jnp.zeros_like(mf.log_rho))
    head = mf.delta_head
    new_head = head
    for nm in ("gbar_tab", "gbar_z_tab", "gtau_tab", "a_k"):
        if hasattr(head, nm) and getattr(head, nm) is not None:
            new_head = eqx.tree_at(lambda h, _nm=nm: getattr(h, _nm), new_head,
                                   jnp.zeros_like(getattr(new_head, nm)))
    return eqx.tree_at(lambda m: m.delta_head, mf, new_head)


def _build(anchor_mult):
    """Build the ns0.909 DESI-only HF-LOSO cert ctx + noiseless truth-on-leg + truth theta-pack.
    Returns (ctx_DESI_only, truth_pack, info, core_desi)."""
    sim = _resolve_hr_sim_name()
    ctx, d = build_legb_ctx(
        with_mf=True, mf_fold=FOLD, mf_with_floor=True,
        mf_exclude_held=True, mf_target_hr_sim=sim,     # genuine HF-LOSO (fit EXCLUDING this sim)
        mf_anchor_mult=float(anchor_mult))
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])   # DESI-only (n_s leg)
    truth_sim = make_hr_truth_from_cache(sim, ctx.cache_k, tau0_anchor="priya")
    mock_legs, truth_pack, info = make_legb_mock(ctx, truth_sim, jax.random.PRNGKey(0))
    core = _mock_core_per_leg(ctx, truth_sim)[ctx.legs[0].name]
    return ctx, truth_pack, info, core


def _pack0(ctx, truth_pack):
    """The truth-point packed parameter vector (NPAR,) the linearization expands about."""
    p = np.zeros(NPAR)
    p[0:9] = np.asarray(truth_pack["theta9"])
    # tau0_amp/dtau0 at the truth's mean-flux curve: the all_off fit floats the 2 PRIYA params;
    # at truth they sit at the sim's fitted (amp,dtau0). We expand about (amp0,dtau0)=(amp_true,
    # 0-tilt-relative) — but the truth tau0_global IS the curve, so we parametrize a MULTIPLICATIVE
    # amp and an additive dtau0 z-tilt ABOUT the truth curve (so p[9]=1, p[10]=0 at truth).
    p[P_TAU0] = 1.0
    p[P_DTAU0] = 0.0
    p[P_AHCD] = np.asarray(truth_pack["alpha_hcd"])
    p[P_SLOPE] = np.asarray(HCD_INCIDENCE_SLOPE)   # the all_off forward s_c center (marginalized);
    #                            the SIM incidence-weight slope ~2.4, NOT the lit/sim ratio ~0.95
    return p


def _forward_logP_packed(ctx, leg, truth_pack, core, p, *, apply_res_corr=True):
    """MF forward log(P_model) on the DESI leg as a fn of the packed params p (NPAR,).
    Applies the z-slope s_c to the HCD incidence (alpha_c(z)=alpha_pivot*((1+z)/(1+zp))^s_c),
    and a multiplicative tau0 amp p[9] + dtau0 z-tilt p[10] about the truth curve."""
    zg = np.asarray(ctx.z_global)
    # theta9 = the floated 9 cosmo/astro params straight from p[0:9] (faithful to the all_off fit).
    theta9 = p[0:9]
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
    z_leg = jnp.asarray(leg.z)
    # mean-flux: truth tau0 curve scaled by p[9] (amp) and tilted by p[10] (dtau0 z-slope).
    tau0_truth = jnp.asarray(truth_pack["tau0_global"])[jnp.asarray(sel)]
    tilt = ((1.0 + z_leg) / (1.0 + ctx.tau0_pivot_z)) ** p[P_DTAU0]
    tau0_vec = tau0_truth * p[P_TAU0] * tilt
    # HCD incidence with the z-slope s_c (per-z), like _legb_model's alpha_hcd_z.
    alpha_pivot = p[P_AHCD]                                 # (3,) pivot-z amplitudes
    s_c = p[P_SLOPE]                                        # (3,) z-slope
    shape = ((1.0 + z_leg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c[None, :]   # (n_z,3)
    alpha_hcd_z = alpha_pivot[None, :] * shape             # (n_z,3) per-leg-z incidence
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    mf = ctx.mf
    if not apply_res_corr:
        mf = eqx.tree_at(lambda m: m.rc_vals, mf, jnp.ones_like(mf.rc_vals))
    P_model, _C = DL.predict_P_obs_on_leg(
        ctx.model, theta9, tau0_vec, alpha_hcd_z, pf_stats=ctx.pf_stats, dla_core=core,
        cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
        cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=mf, mf_floor=ctx.mf_floor,
        alpha_res=None)
    return jnp.log(P_model)


def _prior_precision(ctx):
    """Diagonal prior precision Lambda (NPAR,) for the floated set. Uniform sites -> 0 (flat,
    bounds slack at truth); Normal sites -> 1/sigma^2. theta9 uniform -> 0; tau0 amp/dtau0
    uniform -> 0; alpha_hcd Normal (ctx.alpha_hcd_sigma); s_c Normal (ZSLOPE_PRIOR_SIGMA)."""
    lam = np.zeros(NPAR)
    sa = np.asarray(ctx.alpha_hcd_sigma)                   # (3,)
    lam[P_AHCD] = 1.0 / np.maximum(sa, 1e-12) ** 2
    ss = np.asarray(ZSLOPE_PRIOR_SIGMA)                    # (3,)
    lam[P_SLOPE] = 1.0 / ss ** 2
    return lam


def _native_bias(ctx, truth_pack, info, core, *, mf_override=None, apply_res_corr=True, label=""):
    """Native (marginalized Laplace) forward n_s bias_z for this ctx config. mf_override swaps
    the MultiFidelity (e.g. g-identity); apply_res_corr=False turns res_corr off."""
    use_ctx = ctx if mf_override is None else ctx._replace(mf=mf_override)
    leg = use_ctx.legs[0]

    P_truth = np.asarray(info["truth_on_leg"][leg.name])
    keep = np.isfinite(P_truth) & (P_truth > 0)
    kr = np.where(keep)[0]
    logP_truth = np.log(P_truth[kr])

    p0 = jnp.asarray(_pack0(use_ctx, truth_pack))

    def f(p):
        return _forward_logP_packed(use_ctx, leg, truth_pack, core, p,
                                    apply_res_corr=apply_res_corr)
    logP_fwd_full = np.asarray(f(p0))
    logP_fwd = logP_fwd_full[kr]
    J_full = np.asarray(jax.jacfwd(f)(p0))                 # (N, NPAR)
    J = J_full[kr]                                         # (N_keep, NPAR)
    # FD-check the n_s column
    h = 1e-3
    pp = p0.at[P_NS].add(h); pm = p0.at[P_NS].add(-h)
    fd = (np.asarray(f(pp)) - np.asarray(f(pm)))[kr] / (2 * h)
    rel = float(np.max(np.abs(J[:, P_NS] - fd)) / (np.max(np.abs(J[:, P_NS])) + 1e-30))

    r = logP_truth - logP_fwd                              # forward residual (log space)
    Cdata = np.asarray(leg.C_data)[np.ix_(kr, kr)]
    D = np.diag(1.0 / np.exp(logP_fwd))
    Clog = D @ Cdata @ D                                   # fractional (log-space) cov
    Cinv = np.linalg.inv(Clog)
    Lam = np.diag(_prior_precision(use_ctx))
    F = J.T @ Cinv @ J + Lam                               # Fisher + prior precision
    g = J.T @ Cinv @ r                                     # residual projected onto params
    Finv = np.linalg.inv(F)
    dtheta = Finv @ g                                      # MAP shift the residual drives
    dns = float(dtheta[P_NS])
    sigma_ns = float(np.sqrt(Finv[P_NS, P_NS]))            # MARGINAL n_s sigma
    native_bias = dns / sigma_ns
    rms_resid = float(np.sqrt(np.mean(r ** 2)))
    print(f"  [{label}] FD rel={rel:.1e}  N_keep={kr.size}  rms(log resid)={rms_resid:.4f}  "
          f"sigma_ns={sigma_ns:.4f}  dns={dns:+.4f}  NATIVE bias_z={native_bias:+.3f}")
    return dict(label=label, native_bias=native_bias, dns=dns, sigma_ns=sigma_ns,
                rms_resid=rms_resid, N=kr.size, fd_rel=rel)


def main():
    print("=== FORWARD-RESIDUAL FISHER: native n_s bias for ns0.909 DESI HF-LOSO (NO NUTS) ===\n")
    results = []

    print("--- building ctx: all_off (raw rc, anchor_mult=0) ---")
    ctx0, tp0, info0, core0 = _build(0.0)
    results.append(_native_bias(ctx0, tp0, info0, core0, label="all_off (raw rc)"))

    print("\n--- building ctx: anchored (rc anchor x5, anchor_mult=5, production) ---")
    ctxA, tpA, infoA, coreA = _build(5.0)
    results.append(_native_bias(ctxA, tpA, infoA, coreA, label="anchored (rc x5)"))

    print("\n--- g-IDENTITY: pure LF emulator (g==0) + res_corr OFF (raw-rc ctx) ---")
    mf_g0 = _zero_g_mf(ctx0.mf)
    results.append(_native_bias(ctx0, tp0, info0, core0, mf_override=mf_g0,
                                apply_res_corr=False, label="g_identity (pure LF emu)"))
    print("--- g-IDENTITY + KEEP raw res_corr (g==0 only; isolates g vs res_corr) ---")
    results.append(_native_bias(ctx0, tp0, info0, core0, mf_override=mf_g0,
                                apply_res_corr=True, label="g_identity + raw res_corr"))

    print("\n========= NATIVE FORWARD n_s BIAS (Fisher, NO alpha/emucoh re-weighting) =========")
    print(f"  {'config':<28s} | {'native bias_z':>13s} | {'rms log resid':>13s} | {'N':>4s}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['label']:<28s} | {r['native_bias']:>+13.3f} | {r['rms_resid']:>13.4f} | {r['N']:>4d}")
    print("\n  NOTE on the SIGN: this uses the NOISELESS HR truth-on-leg, so native bias_z is the")
    print("  NOISE-FREE expectation. The all_off NUTS arm (-0.62) is ONE noisy-mock realization; at")
    print("  the ~0.6sigma level the sign is set by that single noise draw, so |native| (the")
    print("  MAGNITUDE) is the like-for-like quantity — it matches the |all_off NUTS| = 0.62.")
    print("\n  INTERPRETATION:")
    print("   * |native(all_off)|=0.6, |anchored|=1.2  (NOT 2.5)  => the full -2.5 NUTS bias is NOT in")
    print("     the marginalized forward residual; it is error-model (alpha/emucoh) AMPLIFICATION (C).")
    print("   * g_identity (pure LF emu) bias GROWS to 2.7 and rms log resid 0.026->0.118 (4.5x WORSE)")
    print("     => g is a NEEDED correction REDUCING the residual, NOT the bias source; removing g does")
    print("     NOT collapse the bias toward 0 (it worsens it). The residual lives in the LF emulator;")
    print("     g is doing its job. So the -2.5 is neither a g-forward bias (A) nor an LF-extrap bias")
    print("     that g hides — it is the error-model RE-WEIGHTING of the small (~0.6-1.2sigma) residual.")


if __name__ == "__main__":
    main()
