#!/usr/bin/env python3
"""METAL SMOOTH-vs-OSCILLATORY COSMOLOGY-ALIASING FISHER (DESI leg) — NO NUTS, cheap.

THE QUESTION (load-bearing physics diagnostic): when the metal-contamination residual the
``metal_misspec`` Gate-B arm injects (the desi_full SiIII/SiII model, including the UNFITTABLE
additive Gaussian-damped SiII-SiII term) hits the cosmology, does the SMOOTH part of the residual
alias into (A_p, n_s) while the OSCILLATORY part (the cos(k·Δv) metal wiggles) stays
cosmology-orthogonal? Prediction: the cosmology response d logP/d{A_p,n_s} is SMOOTH in log k
(amplitude + tilt + curvature), so a low-order poly in log10(k) captures exactly the
cosmology-aliasing modes; the SiIII/SiII oscillations land in r_osc and project ~0 onto cosmology
(and are absorbed by the floated metal amplitudes a_SiIII/a_SiII).

This MIRRORS the validated forward-residual marginalized-Laplace/Fisher of
``scripts/scratch_mf_nscert_fwdresid_fisher.py`` (log-space cov Clog = diag(1/P_fwd) C_data
diag(1/P_fwd); F = J^T Clog^-1 J + Lambda; dtheta = F^-1 J^T Clog^-1 r; native bias =
dtheta[p]/sqrt(Finv[p,p]) in sigma units), built on the DEPLOYED ctx from
``scripts/diag_perleg_fisher_ns.py`` (N=5 ensemble, with_mf + floor + emucoh-offdiag, metals_on,
sample_metals, hierarchical_hcd=False).

FITTED PARAM SET (16) = theta9(9, unit-cube; n_s=theta9[0], A_p=theta9[1]) + [tau0_amp, dtau0]
  + [alpha_lls, alpha_subdla, alpha_dla] + [a_SiIII, a_SiII].
  -> metals ARE floated/marginalized; the bias is what SURVIVES after the fit absorbs what it can
     via the 2 metal amplitudes. (Deployment-consistency caveat: the DEPLOYED model samples ONLY
     a_SiIII; a_SiII is fixed=0 in production. The task floats BOTH on purpose — it is the
     "do the metal params absorb the oscillation?" test. _metal_factor/predict_P_obs_on_leg both
     accept a_SiII, so the forward is otherwise byte-consistent.)

RESIDUAL: the metal_misspec arm injects metal_inject(P_clean, k, <F>, form="desi_full") per z-block
  (defaults f_SiIII=0.009, f_SiII=0.004, f_SiII_SiII=0.002, r_doublet=0.5, k_damp=0.05,
  k_decorr=0.05); <F>(z) = _meanflux_on_leg (truth tau0 = -ln<F>). r = log(P_cont) - log(P_clean).

FIDUCIAL TRUTH: theta9 = unit-cube CENTRE (0.5); tau0_amp=1.0, dtau0=0.0 (Kim/PRIYA prior mean);
  alpha at the HCD-prior MEAN (ctx.alpha_hcd_mu). a_SiIII=a_SiII=0 for P_clean.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=4 /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_metal_smooth_bias.py

READ-ONLY on all modules (a NEW script). Outputs to the NOTES repo 05_likelihood/.
"""
from __future__ import annotations
import os, glob, json, functools
print = functools.partial(print, flush=True)

import numpy as np
import hcd_analysis.emulator  # x64 before jax
import jax
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock,
    _mock_core_per_leg, _meanflux_on_leg, metal_inject,
    HCD_Z_PIVOT)
import hcd_analysis.emulator.data_likelihood as DL

FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
REPO = "/home/mfho/hcd_priya"
os.makedirs(FIG, exist_ok=True)

# fitted param pack (16):
PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz",
               "bhfeedback", "tau0_amp", "dtau0", "alpha_lls", "alpha_subdla", "alpha_dla",
               "a_SiIII", "a_SiII"]
NPAR = 16
P_NS, P_AP, P_TAU0, P_DTAU0 = 0, 1, 9, 10
P_AHCD = slice(11, 14)
P_ASIIII, P_ASIII = 14, 15
REPORT = ["Ap", "ns", "tau0_amp", "dtau0"]
RIDX = {"Ap": P_AP, "ns": P_NS, "tau0_amp": P_TAU0, "dtau0": P_DTAU0}

POLY_DEG = 3   # PRIMARY smooth/osc split: poly in log10(k), per z-bin (degree 3 = amp+tilt+curv+).

# metal_inject defaults the metal_misspec arm uses (desi_full); kept here for the physical split.
MK = dict(form="desi_full", f_SiIII=0.009, f_SiII=0.004, f_SiII_SiII=0.002,
          r_doublet=0.5, k_damp=0.05, k_decorr=0.05)


# ---------------------------------------------------------------------------- #
#  BUILD the deployed DESI-leg ctx (mirror diag_perleg_fisher_ns.py exactly).
# ---------------------------------------------------------------------------- #
def build_ctx():
    members = sorted(p[:-4] for p in glob.glob(f"{REPO}/checkpoints/final_prod_seed*.eqx"))
    assert members, "no final_prod_seed*.eqx ensemble members found"
    print(f"ensemble members ({len(members)}):", [os.path.basename(m) for m in members])
    ctx, d = build_legb_ctx(
        ensemble_ckpts=members, use_xclass=True, with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True, with_eboss=True, metals_on=True,
        sample_metals=True, hierarchical_hcd=False)
    # DESI-only (the n_s/A_p leg with the full desi_full metal model + the additive SiII-SiII term).
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    leg = ctx.legs[0]
    assert leg.name == "DESI" and bool(leg.metals_on), "DESI leg must be metals_on for this probe"
    return ctx, d


def build_fiducial(ctx, d):
    """A fiducial truth_pack at theta9 CENTRE / tau0 prior mean / alpha at HCD-prior mean, plus the
    DLA core per leg. We build a real held-out-sim truth ONLY to obtain a valid dla_core structure
    + a truth_pack template (its tau0_global z-grid + alpha_hcd_z keys), then OVERRIDE the fiducial
    physics fields. _meanflux_on_leg reads truth_pack['tau0_global'] (= tau_eff on z_global)."""
    sims, _ = held_out_sims(d, fold=0)
    truth = make_truth_from_sim(d, sims[0], fold=0, mf=ctx.mf)
    core_per_leg = _mock_core_per_leg(ctx, truth)
    _mock_legs, tp, _info = make_legb_mock(ctx, truth, jax.random.PRNGKey(0))

    zg = np.asarray(ctx.z_global)
    # FIDUCIAL OVERRIDES:
    theta9_fid = np.full(9, 0.5)                              # unit-cube CENTRE
    alpha_pivot = np.asarray(ctx.alpha_hcd_mu, float)         # HCD-prior MEAN (3,) pivot-z incidence
    # tau0_global (= tau_eff(z) = -ln<F>) at the Kim/PRIYA prior mean (tau0_amp=1, dtau0=0):
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya, KIM_AMP, KIM_SLOPE
    z_jnp = jnp.asarray(zg)
    kim = KIM_AMP * (1.0 + z_jnp) ** KIM_SLOPE
    tau0_global_fid = np.asarray(tau0_alpha_priya(z_jnp, 1.0, 0.0, z_pivot=ctx.tau0_pivot_z) * kim)
    # z-resolved alpha for _meanflux/closure-style truth bookkeeping (pivot scaled by the SIM-truth
    # incidence slope is irrelevant here; _meanflux only reads tau0_global). Keep a z-resolved alpha
    # at the prior mean via the same power-law the forward uses (HCD_Z_PIVOT pivot, sim slope).
    tp_fid = dict(tp)
    tp_fid["theta9"] = theta9_fid
    tp_fid["tau0_global"] = tau0_global_fid
    tp_fid["alpha_hcd"] = alpha_pivot
    return tp_fid, core_per_leg, alpha_pivot


# ---------------------------------------------------------------------------- #
#  DEPLOYED forward log P on the DESI leg as a fn of the 16-param pack.
# ---------------------------------------------------------------------------- #
def make_forward(ctx, core_per_leg, alpha_pivot):
    leg = ctx.legs[0]
    zg = np.asarray(ctx.z_global)
    z_leg = jnp.asarray(np.asarray(leg.z))
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    mec = (ctx.mf_emucoh_per_leg.get(leg.name)
           if getattr(ctx, "mf_emucoh_per_leg", None) is not None else None)
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya, KIM_AMP, KIM_SLOPE

    def _kim(z):
        return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE

    # the per-class HCD incidence z-slope (sim-truth incidence slope, the deployed closure center)
    from hcd_analysis.emulator.data import SAMPLING_LIMITS  # noqa: F401 (kept for parity/imports)
    from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
    s_c = jnp.asarray(np.asarray(HCD_INCIDENCE_SLOPE, float))   # (3,)

    def forward_logP(p):
        """log P_model on the DESI leg from the 16-param pack p (deployed config)."""
        th9 = p[:9]
        tau0_amp, dtau0 = p[P_TAU0], p[P_DTAU0]
        alpha_pivot_p = p[P_AHCD]                              # (3,) pivot-z incidence
        a_siiii, a_siii = p[P_ASIIII], p[P_ASIII]
        # mean-flux (PRIYA Kim07) on the leg z:
        tau0_vec = tau0_alpha_priya(z_leg, tau0_amp, dtau0, z_pivot=ctx.tau0_pivot_z) * _kim(z_leg)
        # z-resolved HCD incidence alpha_hcd_z (n_z,3), power-law about HCD_Z_PIVOT (deployed shape):
        shape = ((1.0 + z_leg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c[None, :]   # (n_z,3)
        alpha_hcd_z = alpha_pivot_p[None, :] * shape                            # (n_z,3)
        P, _C = DL.predict_P_obs_on_leg(
            ctx.model, th9, tau0_vec, alpha_hcd_z, pf_stats=ctx.pf_stats,
            dla_core=core_per_leg[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb,
            a_SiIII=a_siiii, a_SiII=a_siii, mf=ctx.mf, mf_floor=ctx.mf_floor,
            mf_emucoh_cov=mec, mf_emucoh_infl=ctx.mf_emucoh_infl,
            mf_emucoh_offdiag_only=ctx.mf_emucoh_offdiag_only)
        return jnp.log(P)

    def forward_P(p):
        return jnp.exp(forward_logP(p))

    return forward_logP, forward_P, sel


def pack0(ctx, alpha_pivot):
    p = np.zeros(NPAR)
    p[:9] = 0.5                              # theta9 centre
    p[P_TAU0] = 1.0
    p[P_DTAU0] = 0.0
    p[P_AHCD] = np.asarray(alpha_pivot, float)
    p[P_ASIIII] = 0.0
    p[P_ASIII] = 0.0
    return p


def prior_precision(ctx):
    """Diagonal prior precision Lambda (NPAR,). Uniform sites -> 0 (theta9, tau0_amp, dtau0, and the
    two metal amplitudes a_SiIII/a_SiII which are Uniform on [0, a_siiii_max] -> flat -> 0). HCD
    alpha Normal -> 1/sigma^2 (ctx.alpha_hcd_sigma)."""
    lam = np.zeros(NPAR)
    sa = np.asarray(ctx.alpha_hcd_sigma, float)                # (3,)
    lam[P_AHCD] = 1.0 / np.maximum(sa, 1e-12) ** 2
    return lam


# ---------------------------------------------------------------------------- #
#  SMOOTH / OSCILLATORY decomposition of the residual r (per z-bin).
# ---------------------------------------------------------------------------- #
def split_poly(r, k, z_idx, n_z, deg=POLY_DEG):
    """PRIMARY split: per z-bin fit r(k) with a degree-`deg` poly in log10(k) -> r_smooth;
    r_osc = r - r_smooth."""
    r_smooth = np.zeros_like(r)
    for iz in range(n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        x = np.log10(k[rows])
        d = min(deg, rows.size - 1)
        if d < 0:
            continue
        coef = np.polyfit(x, r[rows], d)
        r_smooth[rows] = np.polyval(coef, x)
    return r_smooth, r - r_smooth


def split_physical(k, z_idx, n_z, Fbar):
    """SECONDARY (physical) split of the desi_full metal residual r_phys = log(1 + C) where C is the
    metal_inject contamination (the residual against the SAME P_clean). The SMOOTH-physical piece =
    the a^2 constants + the ADDITIVE Gaussian-damped SiII-SiII term (no cos cross-term oscillation);
    the OSC-physical piece = the cos(k·Δv) cross-terms (SiIII + SiII doublet + intra-doublet).
    Built on the SAME 1+C decomposition metal_inject uses, so r_full_phys == log(1+C) == the injected
    residual exactly (cross-check the poly split lands close)."""
    C_KMS = DL.C_KMS
    dvA = C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    dva = C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiII)
    from hcd_analysis.emulator.closure_legb import _LAMBDA_SiIIb
    dvb = C_KMS * np.log(DL.LAMBDA_LYA / _LAMBDA_SiIIb)
    dvd = C_KMS * np.log(_LAMBDA_SiIIb / DL.LAMBDA_SiII)
    f_SiIII, f_SiII, f_SiII_SiII = MK["f_SiIII"], MK["f_SiII"], MK["f_SiII_SiII"]
    rdb, kdamp, kdec = MK["r_doublet"], MK["k_damp"], MK["k_decorr"]

    C_smooth = np.zeros(k.shape[0])     # a^2 constants + additive SiII-SiII (no cos)
    C_osc = np.zeros(k.shape[0])        # the cos cross-terms
    for iz in range(n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        kk = k[rows]
        omF = max(1.0 - float(Fbar[iz]), 1e-3)
        A3 = f_SiIII / omF
        A2 = f_SiII / omF
        D = 2.0 - 2.0 / (1.0 + np.exp(-kk / kdec))
        # constants (a^2) -> smooth
        c_const = A3 ** 2 + A2 ** 2 * (1.0 + rdb ** 2)
        # additive same-ion SiII-SiII Gaussian-damped (NO cos cross-term in the SMOOTH part):
        #   the term is f_SiII_SiII*(1+r^2 + 2r cos(dvd k))*exp(-(k/kdamp)^2); its NON-oscillatory
        #   piece (1+r^2) is smooth, the 2r cos(dvd k) piece is oscillatory.
        gdamp = np.exp(-(kk / kdamp) ** 2)
        c_ss_smooth = f_SiII_SiII * (1.0 + rdb ** 2) * gdamp
        c_ss_osc = f_SiII_SiII * (2.0 * rdb * np.cos(dvd * kk)) * gdamp
        # cos cross-terms (SiIII + SiII doublet) -> oscillatory
        c_osc = (2.0 * A3 * np.cos(dvA * kk) * D
                 + 2.0 * A2 * (np.cos(dvb * kk) + rdb * np.cos(dva * kk)) * D
                 + c_ss_osc)
        C_smooth[rows] = c_const + c_ss_smooth
        C_osc[rows] = c_osc
    # the residual against P_clean is log(1 + C_smooth + C_osc). The physical split assigns the
    # smooth/osc *of the residual* as the corresponding additive-in-log decomposition. Because log
    # is nonlinear, define r_smooth_phys = log(1+C_smooth) and r_osc_phys = log(1+C) - log(1+C_smooth)
    # so they sum EXACTLY to r_full_phys = log(1+C) (preserves the linearity identity for the split).
    r_full = np.log1p(C_smooth + C_osc)
    r_smooth = np.log1p(C_smooth)
    r_osc = r_full - r_smooth
    return r_full, r_smooth, r_osc


# ---------------------------------------------------------------------------- #
def main():
    print("=== METAL SMOOTH-vs-OSC COSMOLOGY-ALIASING FISHER (DESI leg, NO NUTS) ===\n")
    ctx, d = build_ctx()
    leg = ctx.legs[0]
    tp_fid, core_per_leg, alpha_pivot = build_fiducial(ctx, d)
    print(f"DESI leg: N_rows={leg.k.shape[0]}  n_z={leg.n_z}  z={np.round(np.asarray(leg.z),3)}")
    print(f"alpha_hcd_mu (LLS,subDLA,DLA) = {np.round(alpha_pivot,4)}")
    print(f"alpha_hcd_sigma               = {np.round(np.asarray(ctx.alpha_hcd_sigma),4)}")

    forward_logP, forward_P, sel = make_forward(ctx, core_per_leg, alpha_pivot)
    p0 = jnp.asarray(pack0(ctx, alpha_pivot))

    # ---- P_clean (deployed forward at the fiducial, a_SiIII=a_SiII=0) ----
    logP_clean_full = np.asarray(forward_logP(p0))
    P_clean_full = np.exp(logP_clean_full)
    keep = np.isfinite(P_clean_full) & (P_clean_full > 0)
    kr = np.where(keep)[0]
    k_keep = np.asarray(leg.k)[kr]
    z_idx_keep = np.asarray(leg.z_idx)[kr]
    print(f"\nkept rows: {kr.size}/{leg.k.shape[0]}")

    # ---- RESIDUAL: inject the metal contamination per z-block (the metal_misspec arm) ----
    Fbar = _meanflux_on_leg(ctx, leg, tp_fid)                  # (n_z,) <F>(z) from truth tau0
    print(f"<F>(z) per leg-z = {np.round(np.asarray(Fbar),4)}")
    P_cont_full = np.array(P_clean_full, float)
    z_idx_all = np.asarray(leg.z_idx)
    k_all = np.asarray(leg.k)
    for iz in range(leg.n_z):
        rows = np.where(z_idx_all == iz)[0]
        if rows.size == 0:
            continue
        P_cont_full[rows] = metal_inject(P_clean_full[rows], k_all[rows], float(Fbar[iz]), **MK)
    r_full_all = np.log(P_cont_full) - logP_clean_full        # log-space residual (all rows)
    r_full = r_full_all[kr]

    # ---- DECOMPOSE (PRIMARY: poly in log10 k per z) ----
    r_smooth, r_osc = split_poly(r_full, k_keep, z_idx_keep, leg.n_z, deg=POLY_DEG)
    # poly-degree SENSITIVITY (1,2,3): the smooth/osc split — and hence its cosmology projection —
    # depends on how flexible the "smooth" basis is. Degree 2 = pure amplitude+tilt+curvature (the
    # literal cosmology d logP/d{A_p,n_s} shape); higher degrees give the poly room to track part of
    # the oscillation envelope. We sweep so the headline is not an artefact of one degree choice.
    poly_sweep = {}

    # ---- DECOMPOSE (SECONDARY: physical split) ----
    rfp_all, rsp_all, rop_all = split_physical(k_all, z_idx_all, leg.n_z, Fbar)
    # sanity: the physical r_full reconstruction must equal the injected residual (1+C path)
    phys_recon_err = float(np.max(np.abs(rfp_all[kr] - r_full)))
    print(f"physical r_full vs injected residual max|diff| = {phys_recon_err:.2e} "
          f"(should be ~0; both are log(1+C))")
    r_smooth_phys = rsp_all[kr]
    r_osc_phys = rop_all[kr]

    # ---- Fisher building blocks (config-INDEPENDENT: forward, residual, cov are shared) ----
    J_full = np.asarray(jax.jacfwd(forward_logP)(p0))         # (N_all, NPAR) = d logP/d params
    J_all = J_full[kr]                                        # (N_keep, NPAR) all 16 columns
    # FD-check the n_s column AND the a_SiIII column (the metal-absorption column).
    def fd_col(idx, h):
        pp = p0.at[idx].add(h); pm = p0.at[idx].add(-h)
        return (np.asarray(forward_logP(pp)) - np.asarray(forward_logP(pm)))[kr] / (2 * h)
    fd_ns = fd_col(P_NS, 1e-3)
    rel_ns = float(np.max(np.abs(J_all[:, P_NS] - fd_ns)) / (np.max(np.abs(J_all[:, P_NS])) + 1e-30))
    fd_a3 = fd_col(P_ASIIII, 1e-4)
    rel_a3 = float(np.max(np.abs(J_all[:, P_ASIIII] - fd_a3)) / (np.max(np.abs(J_all[:, P_ASIIII])) + 1e-30))
    print(f"FD-check: n_s col rel={rel_ns:.1e}   a_SiIII col rel={rel_a3:.1e}")

    Cdata = np.asarray(leg.C_data)[np.ix_(kr, kr)]
    P_fwd = np.exp(logP_clean_full[kr])
    Dinv = np.diag(1.0 / P_fwd)
    Clog = Dinv @ Cdata @ Dinv                               # fractional (log-space) cov
    Cinv = np.linalg.inv(Clog)
    Lam_all = prior_precision(ctx)                           # (NPAR,) diagonal prior precision

    # ================================================================== #
    #  FISHER over a COLUMN SUBSET (the fit's floated basis). Returns the
    #  full bias table + diagnostics for that config.  cols = the kept
    #  parameter indices (a SUBSET of range(NPAR)). The forward/residual
    #  are IDENTICAL across configs; only the absorbing subspace changes.
    # ================================================================== #
    def fisher_config(cols, label):
        cols = list(cols)
        cpos = {p: cols.index(p) for p in (P_NS, P_AP, P_TAU0, P_DTAU0)}  # report-param row in F-space
        has_a2 = (P_ASIII in cols)
        Jc = J_all[:, cols]                                  # (N_keep, ncol)
        Lam = np.diag(Lam_all[cols])
        F = Jc.T @ Cinv @ Jc + Lam
        Finv = np.linalg.inv(F)
        M = Finv @ Jc.T @ Cinv                              # (ncol, N_keep) bias map
        sig = np.sqrt(np.diag(Finv))                        # (ncol,) marginal sigma

        def bias_row(r):
            dth = M @ r
            return {nm: float(dth[cpos[RIDX[nm]]] / sig[cpos[RIDX[nm]]]) for nm in REPORT}

        b_full = bias_row(r_full)
        b_smooth = bias_row(r_smooth); b_osc = bias_row(r_osc)
        b_sphys = bias_row(r_smooth_phys); b_ophys = bias_row(r_osc_phys)
        b_fphys = bias_row(rfp_all[kr])

        # metal subspace this CONFIG can absorb (a_SiIII only, or a_SiIII+a_SiII)
        metal_cols_idx = [P_ASIIII] + ([P_ASIII] if has_a2 else [])
        G = J_all[:, metal_cols_idx]                        # (N, n_metal)
        n3 = float(np.linalg.norm(J_all[:, P_ASIIII]))
        n2 = float(np.linalg.norm(J_all[:, P_ASIII]))
        cos_a = float(np.dot(J_all[:, P_ASIIII], J_all[:, P_ASIII]) / (n3 * n2 + 1e-30))
        GtCinv = G.T @ Cinv
        coefo = np.linalg.solve(GtCinv @ G + 1e-12 * np.eye(G.shape[1]), GtCinv @ r_osc)
        r_osc_proj = G @ coefo
        frac_osc_absorbed = float(
            1.0 - (r_osc - r_osc_proj) @ Cinv @ (r_osc - r_osc_proj)
            / ((r_osc @ Cinv @ r_osc) + 1e-30))
        coeff = np.linalg.solve(GtCinv @ G + 1e-12 * np.eye(G.shape[1]), GtCinv @ r_full)
        r_after = r_full - G @ coeff                        # residual the cosmology still sees
        rs_am, ro_am = split_poly(r_after, k_keep, z_idx_keep, leg.n_z, deg=POLY_DEG)
        b_after = bias_row(r_after)
        b_after_s = bias_row(rs_am); b_after_o = bias_row(ro_am)
        sf_ns_am = float(abs(b_after_s["ns"]) / (abs(b_after_s["ns"]) + abs(b_after_o["ns"]) + 1e-30))
        sf_ap_am = float(abs(b_after_s["Ap"]) / (abs(b_after_s["Ap"]) + abs(b_after_o["Ap"]) + 1e-30))

        # poly-degree sweep (this config's M)
        psweep = {}
        for dgr in (1, 2, 3):
            rs_d, ro_d = split_poly(r_full, k_keep, z_idx_keep, leg.n_z, deg=dgr)
            bs_d = bias_row(rs_d); bo_d = bias_row(ro_d)
            psweep[dgr] = dict(smooth=bs_d, osc=bo_d,
                               sf_ns=float(abs(bs_d["ns"]) / (abs(bs_d["ns"]) + abs(bo_d["ns"]) + 1e-30)),
                               sf_Ap=float(abs(bs_d["Ap"]) / (abs(bs_d["Ap"]) + abs(bo_d["Ap"]) + 1e-30)))

        # linearity on the PHYSICAL split (the LEAD decomposition): bias(full)=bias(s_phys)+bias(o_phys)
        lin_resid = {}
        for nm in REPORT:
            lhs = b_full[nm]; rhs = b_sphys[nm] + b_ophys[nm]
            lin_resid[nm] = float(abs(lhs - rhs) / max(abs(lhs), 1e-12))
        lin_max = max(lin_resid.values()); lin_pass = lin_max < 0.01
        # poly-split linearity too (sanity; same M, linear ⇒ also exact)
        lin_resid_poly = {nm: float(abs(b_full[nm] - (b_smooth[nm] + b_osc[nm]))
                                    / max(abs(b_full[nm]), 1e-12)) for nm in REPORT}

        sf_ns = float(abs(b_sphys["ns"]) / (abs(b_sphys["ns"]) + abs(b_ophys["ns"]) + 1e-30))
        sf_ap = float(abs(b_sphys["Ap"]) / (abs(b_sphys["Ap"]) + abs(b_ophys["Ap"]) + 1e-30))
        sf_ns_poly = float(abs(b_smooth["ns"]) / (abs(b_smooth["ns"]) + abs(b_osc["ns"]) + 1e-30))
        sf_ap_poly = float(abs(b_smooth["Ap"]) / (abs(b_smooth["Ap"]) + abs(b_osc["Ap"]) + 1e-30))

        metals_absorb_osc = (n3 > 0) and (frac_osc_absorbed > 0.05)
        full_finite = all(np.isfinite(list(b_full.values())))
        full_modest = all(abs(v) < 50 for v in b_full.values())

        return dict(
            label=label, cols=cols, ncol=len(cols), has_a2=has_a2,
            bias=dict(full=b_full, smooth_poly=b_smooth, osc_poly=b_osc,
                      smooth_phys=b_sphys, osc_phys=b_ophys, full_phys=b_fphys),
            sigma_marginal={nm: float(sig[cpos[RIDX[nm]]]) for nm in REPORT},
            smooth_fraction_phys=dict(ns=sf_ns, Ap=sf_ap),
            smooth_fraction_poly=dict(ns=sf_ns_poly, Ap=sf_ap_poly),
            linearity_phys=dict(rel_resid=lin_resid, max=lin_max, pass_=lin_pass),
            linearity_poly=dict(rel_resid=lin_resid_poly, max=max(lin_resid_poly.values())),
            metal_cols=dict(norm_aSiIII=n3, norm_aSiII=n2, cos_a3_a2=cos_a,
                            frac_osc_absorbed=frac_osc_absorbed,
                            metals_absorb_osc=bool(metals_absorb_osc)),
            poly_degree_sweep={str(dg): psweep[dg] for dg in (1, 2, 3)},
            after_metal_marginalization=dict(
                bias=b_after, smooth=b_after_s, osc=b_after_o,
                smooth_fraction=dict(ns=sf_ns_am, Ap=sf_ap_am)),
            full_finite=full_finite, full_modest=full_modest)

    # DEPLOYED (PRIMARY): float a_SiIII ONLY — a_SiII FIXED=0 (matches _legb_model:1410-1411).
    cols_deployed = [c for c in range(NPAR) if c != P_ASIII]
    res_dep = fisher_config(cols_deployed, "DEPLOYED (a_SiIII only)")
    # BOTH-FLOATED (SECONDARY comparison): float a_SiIII AND a_SiII (NOT deployed — shows how much an
    # un-modeled a_SiII WOULD absorb).
    res_both = fisher_config(list(range(NPAR)), "both-floated (a_SiIII+a_SiII)")

    # PRIMARY = deployed: bind its quantities to the names the figures/JSON downstream use.
    P = res_dep
    b_full = P["bias"]["full"]; b_smooth = P["bias"]["smooth_poly"]; b_osc = P["bias"]["osc_poly"]
    b_sphys = P["bias"]["smooth_phys"]; b_ophys = P["bias"]["osc_phys"]; b_fphys = P["bias"]["full_phys"]
    sig_rep = P["sigma_marginal"]
    sf_ns = P["smooth_fraction_phys"]["ns"]; sf_ap = P["smooth_fraction_phys"]["Ap"]
    sf_ns_poly = P["smooth_fraction_poly"]["ns"]; sf_ap_poly = P["smooth_fraction_poly"]["Ap"]
    poly_sweep = {dg: P["poly_degree_sweep"][str(dg)] for dg in (1, 2, 3)}
    sf_ns_am = P["after_metal_marginalization"]["smooth_fraction"]["ns"]
    sf_ap_am = P["after_metal_marginalization"]["smooth_fraction"]["Ap"]
    b_after = P["after_metal_marginalization"]["bias"]
    lin_resid = P["linearity_phys"]["rel_resid"]; lin_max = P["linearity_phys"]["max"]
    lin_pass = P["linearity_phys"]["pass_"]
    n3 = P["metal_cols"]["norm_aSiIII"]; n2 = P["metal_cols"]["norm_aSiII"]
    cos_a = P["metal_cols"]["cos_a3_a2"]; frac_osc_absorbed = P["metal_cols"]["frac_osc_absorbed"]
    metals_absorb_osc = P["metal_cols"]["metals_absorb_osc"]

    print(f"\nLINEARITY (physical split, DEPLOYED): max rel residual = {lin_max:.2e}  "
          f"({'PASS' if lin_pass else 'FAIL'} threshold 1%)")
    for nm in REPORT:
        print(f"  {nm:<9s}: full={b_full[nm]:+.4f}  s_phys+o_phys={b_sphys[nm]+b_ophys[nm]:+.4f}  "
              f"rel.resid={lin_resid[nm]:.2e}")
    print(f"\nDEPLOYED metal absorption: |J[a_SiIII]|={n3:.3e}  frac_osc_absorbed={frac_osc_absorbed:.3f} "
          f"(a_SiII NOT floated)")

    full_finite = P["full_finite"]; full_modest = P["full_modest"]
    sanity_pass = bool(full_finite and full_modest and lin_pass and metals_absorb_osc
                       and phys_recon_err < 1e-8 and rel_ns < 1e-2 and rel_a3 < 1e-2)

    # ============================ TABLE / PRINT ============================
    # LEAD with the PHYSICAL split (basis-free, decisive). Poly split demoted to a caveat.
    print("\n===== PRIMARY = DEPLOYED (a_SiIII floated, a_SiII FIXED=0) — PHYSICAL SPLIT =====")
    print("      (native bias in sigma; physical split is basis-free and the LEAD decomposition)")
    hdr = f"  {'component':<14s} | " + " | ".join(f"{nm:>9s}" for nm in REPORT)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, b in [("full", b_full), ("smooth_phys", b_sphys), ("osc_phys", b_ophys)]:
        print(f"  {label:<14s} | " + " | ".join(f"{b[nm]:>+9.4f}" for nm in REPORT))
    print(f"\n  smooth_phys FRACTION of |bias|:  n_s={sf_ns:.3f}   A_p={sf_ap:.3f}")
    print(f"  DEPLOYED full bias:  n_s={b_full['ns']:+.4f}σ   A_p={b_full['Ap']:+.4f}σ   "
          f"tau0_amp={b_full['tau0_amp']:+.4f}σ   dtau0={b_full['dtau0']:+.4f}σ")

    print("\n  --- CAVEAT: poly split is degree-dependent / UNRELIABLE (it over-flexes the cosmology "
          "subspace) ---")
    print(f"  {'deg':>3s} | {'ns_smooth':>9s} {'ns_osc':>9s} {'sf_ns':>6s} | "
          f"{'Ap_smooth':>9s} {'Ap_osc':>9s} {'sf_Ap':>6s}")
    for dg in (1, 2, 3):
        ps = poly_sweep[dg]
        print(f"  {dg:>3d} | {ps['smooth']['ns']:>+9.4f} {ps['osc']['ns']:>+9.4f} "
              f"{ps['sf_ns']:>6.3f} | {ps['smooth']['Ap']:>+9.4f} {ps['osc']['Ap']:>+9.4f} "
              f"{ps['sf_Ap']:>6.3f}")

    print("\n===== DEPLOYED vs both-floated (full metal bias in sigma) =====")
    print(f"  {'config':<26s} | " + " | ".join(f"{nm:>9s}" for nm in REPORT))
    print("  " + "-" * 74)
    for res in (res_dep, res_both):
        bf = res["bias"]["full"]
        print(f"  {res['label']:<26s} | " + " | ".join(f"{bf[nm]:>+9.4f}" for nm in REPORT))
    d_ns = res_dep["bias"]["full"]["ns"] - res_both["bias"]["full"]["ns"]
    d_ap = res_dep["bias"]["full"]["Ap"] - res_both["bias"]["full"]["Ap"]
    print(f"  Δ(deployed − both) :  n_s={d_ns:+.4f}σ   A_p={d_ap:+.4f}σ   "
          f"(fixing a_SiII {'INCREASES' if abs(res_dep['bias']['full']['ns'])>abs(res_both['bias']['full']['ns']) else 'does not increase'} |n_s| leakage)")

    # ============================ JSON ============================
    def res_json(res):
        return dict(
            label=res["label"], ncol=res["ncol"], floats_a_SiII=res["has_a2"],
            bias=res["bias"], sigma_marginal=res["sigma_marginal"],
            smooth_fraction_phys=res["smooth_fraction_phys"],
            smooth_fraction_poly=res["smooth_fraction_poly"],
            linearity_phys=res["linearity_phys"], linearity_poly=res["linearity_poly"],
            metal_cols=res["metal_cols"], poly_degree_sweep=res["poly_degree_sweep"],
            after_metal_marginalization=res["after_metal_marginalization"])

    out = dict(
        leg="DESI", npar=NPAR, param_names=PARAM_NAMES, poly_deg=POLY_DEG,
        report_params=REPORT, n_keep=int(kr.size),
        primary_config="DEPLOYED (a_SiIII only, a_SiII fixed=0)",
        sanity_pass=sanity_pass,
        fd_rel_ns=rel_ns, fd_rel_aSiIII=rel_a3, phys_recon_err=phys_recon_err,
        # PRIMARY (deployed) at the top level (back-compat keys), LEADING with the physical split.
        bias=res_dep["bias"],
        sigma_marginal=res_dep["sigma_marginal"],
        smooth_fraction=res_dep["smooth_fraction_phys"],          # physical-split smooth fraction (LEAD)
        smooth_fraction_poly=res_dep["smooth_fraction_poly"],     # poly (caveat: degree-dependent)
        linearity_check=res_dep["linearity_phys"],                # physical-split linearity (LEAD)
        linearity_poly=res_dep["linearity_poly"],
        metal_cols=res_dep["metal_cols"],
        poly_degree_sweep=res_dep["poly_degree_sweep"],
        after_metal_marginalization=res_dep["after_metal_marginalization"],
        # both configs side by side
        configs=dict(deployed=res_json(res_dep), both_floated=res_json(res_both)),
        deployed_vs_both=dict(
            full_ns=dict(deployed=res_dep["bias"]["full"]["ns"], both=res_both["bias"]["full"]["ns"],
                         delta=d_ns),
            full_Ap=dict(deployed=res_dep["bias"]["full"]["Ap"], both=res_both["bias"]["full"]["Ap"],
                         delta=d_ap)),
        meanflux_Fbar=[float(x) for x in np.asarray(Fbar)],
        alpha_hcd_mu=[float(x) for x in np.asarray(alpha_pivot)],
        alpha_hcd_sigma=[float(x) for x in np.asarray(ctx.alpha_hcd_sigma)],
        rms_log_resid=dict(full=float(np.sqrt(np.mean(r_full**2))),
                           smooth_phys=float(np.sqrt(np.mean(r_smooth_phys**2))),
                           osc_phys=float(np.sqrt(np.mean(r_osc_phys**2)))),
        note=("PRIMARY = DEPLOYED config: the leg-A model _legb_model samples ONLY a_SiIII "
              "(Uniform[0,0.15]); a_SiII is FIXED=0 in production AND in the live metal_misspec "
              "NUTS arm. The both-floated config (configs.both_floated) is a SECONDARY comparison "
              "showing how much an un-modeled a_SiII would absorb. LEAD with the PHYSICAL split "
              "(basis-free); the poly split is degree-dependent / unreliable (it over-flexes the "
              "cosmology subspace — see poly_degree_sweep). Bias in native marginal sigma; theta9 "
              "unit-cube (ns=theta9[0], Ap=theta9[1])."))
    jpath = f"{FIG}/metal_smooth_bias.json"
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {jpath}")

    # ============================ FIGURE 1: residual decomposition ============================
    zvals = np.asarray(leg.z)
    # pick ~3 representative z (low, mid, high) that have rows
    have = [iz for iz in range(leg.n_z) if np.any(z_idx_keep == iz)]
    pick = [have[0], have[len(have) // 2], have[-1]] if len(have) >= 3 else have
    fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 4.2), squeeze=False)
    for ax, iz in zip(axes[0], pick):
        rows = np.where(z_idx_keep == iz)[0]
        order = np.argsort(k_keep[rows])
        kk = k_keep[rows][order]
        ax.plot(kk, r_full[rows][order], "k-", lw=1.6, label="r_full")
        ax.plot(kk, r_smooth[rows][order], "C0--", lw=2.0, label="r_smooth (poly%d)" % POLY_DEG)
        ax.plot(kk, r_osc[rows][order], "C3-", lw=1.0, alpha=0.8, label="r_osc")
        ax.plot(kk, r_smooth_phys[rows][order], "C2:", lw=1.6, label="r_smooth_phys")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [s/km]")
        ax.set_title(f"z = {zvals[iz]:.2f}")
        if ax is axes[0, 0]:
            ax.set_ylabel(r"$\log P_{\rm cont} - \log P_{\rm clean}$")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Metal-contamination residual: full vs smooth (poly in log k) vs oscillatory — DESI",
                 fontsize=11)
    fig.tight_layout()
    p1 = f"{FIG}/metal_smooth_resid_decomp.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"wrote {p1}")

    # ============================ FIGURE 2: bias bars ============================
    # TOP row (LEAD) = DEPLOYED config, PHYSICAL split (basis-free): full / smooth_phys / osc_phys.
    # BOTTOM row = DEPLOYED vs both-floated full-bias comparison (the a_SiII deployment-consistency
    # check). The poly split is intentionally NOT shown as a bar (degree-dependent / unreliable —
    # it lives in the json poly_degree_sweep as a caveat).
    bf_dep = res_dep["bias"]["full"]; bf_both = res_both["bias"]["full"]
    fig, axes = plt.subplots(2, len(REPORT), figsize=(3.4 * len(REPORT), 7.2), squeeze=False)
    # top: physical split (deployed)
    comps = [("full", b_full, "#444444"), ("smooth_phys", b_sphys, "#1b9e77"),
             ("osc_phys", b_ophys, "#d95f02")]
    for ax, nm in zip(axes[0], REPORT):
        vals = [b[nm] for _, b, _ in comps]; cols = [c for _, _, c in comps]
        bars = ax.bar([lab for lab, _, _ in comps], vals, color=cols)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + (0.01 if v >= 0 else -0.01) * (1 + abs(v)),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{nm}  [DEPLOYED, physical split]", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        if ax is axes[0, 0]:
            ax.set_ylabel("native bias [σ]")
        lim = max(0.05, 1.25 * max(abs(min(vals)), abs(max(vals))))
        ax.set_ylim(-lim, lim)
    # bottom: deployed vs both-floated full bias
    for ax, nm in zip(axes[1], REPORT):
        vals = [bf_dep[nm], bf_both[nm]]
        bars = ax.bar(["deployed\n(a_SiIII only)", "both-floated\n(a_SiIII+a_SiII)"], vals,
                      color=["#7570b3", "#999999"])
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + (0.01 if v >= 0 else -0.01) * (1 + abs(v)),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{nm}  [full bias: deployed vs both]", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        if ax is axes[1, 0]:
            ax.set_ylabel("native full bias [σ]")
        lim = max(0.05, 1.25 * max(abs(min(vals)), abs(max(vals))))
        ax.set_ylim(-lim, lim)
    sfstr = (f"DEPLOYED physical-split smooth frac n_s={sf_ns:.2f}/A_p={sf_ap:.2f}  |  "
             f"deployed full n_s={bf_dep['ns']:+.3f}σ A_p={bf_dep['Ap']:+.3f}σ tau0={bf_dep['tau0_amp']:+.3f}σ")
    fig.suptitle("Metal-residual cosmology bias (DESI)\n"
                 "TOP = DEPLOYED (a_SiIII only), PHYSICAL split — the LEAD result\n"
                 "BOTTOM = deployed vs both-floated full bias (a_SiII deployment-consistency)\n" + sfstr,
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = f"{FIG}/metal_smooth_bias_bars.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"wrote {p2}")

    # ============================ verdict ============================
    print("\n========= SANITY =========")
    print(f"  linearity (bias linear in r): {'PASS' if lin_pass else 'FAIL'} (max rel {lin_max:.2e})")
    print(f"  metals absorb oscillation:    {'PASS' if metals_absorb_osc else 'FAIL'} "
          f"(frac_osc_absorbed={frac_osc_absorbed:.3f})")
    print(f"  full bias finite+modest:      {'PASS' if (full_finite and full_modest) else 'FAIL'}")
    print(f"  physical-split reconstruction:{'PASS' if phys_recon_err < 1e-8 else 'FAIL'} "
          f"({phys_recon_err:.1e})")
    print(f"  FD jacobian (ns, a_SiIII):    {'PASS' if (rel_ns<1e-2 and rel_a3<1e-2) else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if sanity_pass else 'FAIL (see above) — interpret with care'}")
    if not sanity_pass:
        print("  *** A SANITY CHECK FAILED — per the spec, the headline below is NOT trustworthy. ***")
    print("done.")


if __name__ == "__main__":
    main()
