"""GATING rerun (forward-only) of the HCD slope-prior bias — the checkpoint-corrected measurement.

Supersedes diag_legb_slope_prior_tradeoff.py, which the 4-lens checkpoint (2026-06-06, spec §7)
found measured the WRONG configuration (E-A: it used g_fixed = the SIM's own w_c(z), the closure-only
framing; E-B: it reported only the A_p bias, omitting the n_s bias which was −0.22σ, OVER gate).

This rerun fixes all of it:
  (1) PRODUCTION g_fixed: g_fixed_c(z) = w_c_from_mu(dN/dX_lit(z)·X̄(z))[HCD], normalized to z=z_p.
      dN/dX_lit = power-law fits to the literature dN/dX (LLS O'Meara/Fumagalli/Prochaska τ≥2;
      subDLA Zafar13; DLA PW09 — the same LIT data as plot_dndx_vs_literature.py). X̄(z) from the sim
      cache geometry (snap_total_path_dX / N_SIGHTLINES, θ-independent). This is EXACTLY what the real
      fit runs — NEVER from dN/dX alone (the incidence-only shape is the original bug; Lyα A2).
  (2) DLA-MASKED truth: the held-out sim's P_obs rebuilt with α_DLA→0 (drop the DLA class + its core
      add-back), matching the data's DLA-masked state (plan §0c; KS ~0%, DESI ≥90% masked).
  (3) EXACT (non-linearized) bias, for BOTH A_p AND n_s:
        ΔP = P_truth_masked − P_0,   P_0 = production forward at truth (g_fixed lit-shape, s=0,
             α_pivot = the lit-derived pivot incidence, DLA-masked).
        bias(param) = [(F + P_prior)^{-1} Jᵀ C^{-1} ΔP]_param      (exact in ΔP; only J is linearized)
      vs the old J_s·gap, which was 13–15% off because (1+z)^s is convex over the large gap.
  (4) MULTIPLE held-out sims (the gap is sim-dependent; certify over the fold, Lyα A5): all 8 fold-0
      val sims (honestly held out from final_fold0). Plus INTERIOR-n_s spot-checks from other folds,
      explicitly flagged "emulator in-sample" — valid because the SHAPE-GAP bias uses the cache-MEASURED
      truth + lit g_fixed, independent of the emulator train/val split (the emulator enters only P_0/J).

Parameter vector for F: [theta9(9), tau0(nZg), a_pivot(3), s(3)]. Slope prior block = 1/σ_s²; the
slope CENTER is 0 under the production g_fixed (the lit shape is in g_fixed; Q2). Variance side reuses
the locked finding (×1.04, framing-robust). The width-vs-bias scan + the σ(A_p),σ(n_s) curve are emitted
the same way, now with the n_s bias and the exact ΔP.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_legb_slope_prior_rerun.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
    make_splits,
)
from hcd_analysis.emulator.closure_legb_figs import truth_tau0_on_leg
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT
from hcd_analysis.emulator.dndx_wc import w_c_from_mu

FIGDIR = "/home/mfho/hcd_priya/figures/analysis/05_likelihood"
RESULTS = f"{FIGDIR}/legb_slope_prior_rerun.txt"
HOLD0 = "/home/mfho/hcd_priya/checkpoints/error_vector_xclass_holdout0.npz"
N_SIGHTLINES = 691200   # verified constant (diag_head_a.py:62, calibrate_delta_c.py); Xbar = path/N

# Literature dN/dX (same data as plot_dndx_vs_literature.py LIT dict): z, value, err per class.
LIT = {
    "LLS":    ([2.4,2.8,3.35,3.47,3.58,3.74,3.97,4.23], [0.29,0.33,0.35,0.57,0.41,0.52,0.72,0.78],
               [0.05,0.08,0.14,0.12,0.07,0.08,0.15,0.19]),
    "subDLA": ([2.27,2.73,3.25,3.77,4.20], [0.07,0.06,0.08,0.10,0.10], [0.01,0.01,0.02,0.02,0.03]),
    "DLA":    ([2.31,2.57,2.86,3.22,3.70,4.39], [0.048,0.055,0.067,0.084,0.075,0.106],
               [0.006,0.005,0.006,0.006,0.009,0.018]),
}
CLASSES = ["LLS", "subDLA", "DLA"]
SIGMA_S_LIT = np.array([0.52, 0.53, 0.33])    # WLS lit slope 1σ (Lyα consult)
EDGE_BETA = 1.0
# Per-leg amplitude anchor δ_{c,leg} (log-offset), the PRIMARY amplitude lever (checkpoint Q4).
# KS may be biased-high LLS/subDLA (absorber-targeted selection); DESI large-area ~ lit.
SIGMA_ANCHOR = {"DESI": 0.12, "KS": 0.27}     # per-leg σ on δ (LLS,subDLA); DLA pinned (masked)
PIN = 1e8     # huge prior precision to pin the DLA nuisance components at 0 (§0c masked)

_rf = open(RESULTS, "w")
def emit(s):
    print(s, flush=True); _rf.write(s + "\n"); _rf.flush()

NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])

# ---- WLS power-law fit of each literature dN/dX: dN/dX_lit(z) = A·(1+z)^γ (weighted by 1/err²)
def fit_lit_powerlaw():
    A, gam = np.zeros(3), np.zeros(3)
    for j, c in enumerate(CLASSES):
        z, v, e = map(np.array, LIT[c])
        w = 1.0 / e**2
        x = np.log(1.0 + z); y = np.log(v)
        # weighted linear fit y = b0 + b1 x  (b1=γ, b0=lnA)
        W = w.sum(); Wx = (w*x).sum(); Wy = (w*y).sum(); Wxx = (w*x*x).sum(); Wxy = (w*x*y).sum()
        b1 = (W*Wxy - Wx*Wy) / (W*Wxx - Wx**2); b0 = (Wy - b1*Wx) / W
        A[j] = np.exp(b0); gam[j] = b1
    return A, gam

A_lit, gam_lit = fit_lit_powerlaw()
emit(f"# slope-prior RERUN (production g_fixed, DLA-masked, exact bias)  PARAM_NAMES={list(PARAM_NAMES)}")
emit(f"# lit dN/dX power-law: A={np.round(A_lit,4)}  gamma={np.round(gam_lit,3)} (LLS,subDLA,DLA)")
emit(f"# sigma_s_lit={SIGMA_S_LIT}  slope center=0 (production framing)  z_pivot={HCD_Z_PIVOT}")

ctx, d = build_legb_ctx(xclass_error_vector=HOLD0)

# ---- X̄(z): sim path-length geometry, θ-independent. Build a z->Xbar interpolant from the cache.
zrow = np.asarray(d["z_grid"]); grp = np.asarray(d["snap_group_idx"])
Xbar_row = np.asarray(d["snap_total_path_dX"])[grp] / N_SIGHTLINES        # (R,)
zu = np.unique(np.round(zrow, 4))
Xbar_of_z = np.array([np.nanmean(Xbar_row[np.isclose(zrow, zz, atol=1e-3)]) for zz in zu])
def xbar_at(z):
    return np.interp(np.asarray(z), zu, Xbar_of_z)

# ---- production g_fixed_c(z) per leg: w_c_from_mu(dN/dX_lit(z)·X̄(z))[HCD], normalized to z_p.
def production_g_fixed(z_leg):
    z_leg = np.asarray(z_leg)
    dndx = A_lit[None, :] * (1.0 + z_leg[:, None]) ** gam_lit[None, :]    # (n_z,3) lit incidence
    mu = dndx * xbar_at(z_leg)[:, None]                                  # (n_z,3) mean absorbers
    w = np.asarray(w_c_from_mu(jnp.asarray(mu)))[:, 1:]                  # (n_z,3) HCD w_c via M0
    # pivot row
    mu_p = (A_lit * (1.0 + HCD_Z_PIVOT) ** gam_lit) * xbar_at(HCD_Z_PIVOT)
    w_p = np.asarray(w_c_from_mu(jnp.asarray(mu_p[None, :])))[0, 1:]     # (3,)
    return w / w_p[None, :], w_p          # g_fixed normalized to z_p, and the pivot incidence

# ---- DLA-masked truth P_obs on the cache grid for one sim (α_DLA→0, drop DLA class + core)
def masked_truth_P(sim, fold):
    t = make_truth_from_sim(d, sim, fold)
    rows = np.asarray(t["rows"]); z = np.asarray(t["z"])
    P_filt = np.asarray(d["P_filt"]); w_c = np.asarray(d["w_c_cache"])
    K = P_filt.shape[-1]; P = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        a = w_c[r, 1:].copy(); a[2] = 0.0                  # MASK DLA: alpha_DLA -> 0
        coef = np.concatenate([[1.0 - a.sum()], a])        # (4,) no DLA contribution
        P_cls = np.stack([P_filt[r, 0], P_filt[r, 1], P_filt[r, 2], P_filt[r, 3]])  # no core add-back
        P[i] = np.einsum("c,ck->k", coef, P_cls)
    return dict(z=z, P=P, params_unit=t["params_unit"], rows=rows, t=t)

cache_k = np.asarray(ctx.cache_k)
zg = np.asarray(ctx.z_global); nZg = zg.size
LEGS = [leg.name for leg in ctx.legs]; nL = len(LEGS)
# param layout: theta9(9) | tau0(nZg) | a_pivot(3) | s(3) | δ_leg(3) per leg
nP = 9 + nZg + 6 + 3 * nL
AMP = np.arange(9 + nZg, 9 + nZg + 3)              # a_pivot (LLS,subDLA,DLA)
SL = np.arange(9 + nZg + 3, 9 + nZg + 6)           # slope s
DEL = {LEGS[li]: np.arange(9 + nZg + 6 + 3 * li, 9 + nZg + 6 + 3 * li + 3) for li in range(nL)}
DLA_J = 2                                          # DLA class index within each 3-block (masked → pinned)

# per-leg precompute: production g_fixed(z), lnz, sel, edge factor
leg_static = {}
for leg in ctx.legs:
    z_leg = np.asarray(leg.z)
    gfix, _ = production_g_fixed(z_leg)
    lnz = np.log((1.0 + z_leg) / (1.0 + HCD_Z_PIVOT))
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in z_leg])
    edge = 1.0 + EDGE_BETA * (np.clip(2.5 - z_leg, 0, None) + np.clip(z_leg - 3.5, 0, None))
    leg_static[leg.name] = dict(gfix=jnp.asarray(gfix), lnz=jnp.asarray(lnz),
                                sel=jnp.asarray(sel), z_leg=z_leg, edge_mean=float(np.mean(edge)))
edge_mean = float(np.mean([leg_static[l.name]["edge_mean"] for l in ctx.legs]))
sigma_s_edge = SIGMA_S_LIT * edge_mean

MASK_DLA = jnp.array([1.0, 1.0, 0.0])    # §0c: DLA incidence → 0 in BOTH truth and forward

def alpha_of_s(gfix, lnz, apivot, s, dleg):
    # α_c(z) = a_pivot,c·exp(δ_leg,c)·g_fixed,c(z)·exp(lnz·s_c), DLA masked to ~0 (§0c)
    amp = apivot * jnp.exp(dleg) * MASK_DLA
    return amp[None, :] * gfix * jnp.exp(lnz[:, None] * s[None, :])

# ---- per-sim: build F, ΔP (exact), then bias(A_p), bias(n_s) at a grid of slope widths
def run_one_sim(sim, fold, label):
    mt = masked_truth_P(sim, fold)
    th_truth = jnp.asarray(mt["params_unit"])
    # lit-derived pivot amplitude (production a_pivot): w_c of the lit incidence at z_p (per leg same)
    _, w_p = production_g_fixed([HCD_Z_PIVOT])
    apivot = jnp.asarray(w_p)            # (3,) lit pivot incidence (LLS,subDLA,DLA); DLA will be ~masked downstream
    s0 = jnp.zeros(3); d0 = jnp.zeros(3)
    # tau0 at truth on the global grid: nearest cache tau0 (use the sim's own, via make_legb_mock)
    mock_legs, tp, info = make_legb_mock(ctx, mt["t"], jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"])
    core = _mock_core_per_leg(ctx, mt["t"])

    # per-z SIM w_c (masked) — for the pipeline-validation baseline (forward at the sim's OWN shape;
    # should reproduce the known +0.03σ EXACT case → isolates irreducible emulator error).
    wc_perz_sim = np.asarray(d["w_c_cache"])[np.asarray(mt["rows"]), 1:].copy()
    wc_perz_sim[:, 2] = 0.0      # mask DLA to match the truth

    F = np.zeros((nP, nP)); gvec = np.zeros(nP); gvec_emu = np.zeros(nP)
    z_sim = mt["z"]; P_sim = mt["P"]
    for leg in ctx.legs:
        st = leg_static[leg.name]; sel = st["sel"]; gfix = st["gfix"]; lnz = st["lnz"]
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]

        # Jacobian wrt (theta9, tau0, a_pivot, s, δ_leg) — δ is THIS leg's anchor.
        def fwd(th, t0g, apv, s, dleg, _leg=leg, _sel=sel, _keep=jnp.asarray(keep), _g=gfix, _l=lnz):
            al = alpha_of_s(_g, _l, apv, s, dleg)
            P, _ = DL.predict_P_obs_on_leg(
                ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats, dla_core=core[_leg.name],
                cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
            return P[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2, 3, 4))(th_truth, t0_truth, apivot, s0, d0)
        # place blocks into the global (Nkeep, nP) Jacobian (δ only in THIS leg's columns)
        Nk = J[0].shape[0]; Jl = np.zeros((Nk, nP))
        Jl[:, 0:9] = np.asarray(J[0]); Jl[:, 9:9 + nZg] = np.asarray(J[1])
        Jl[:, AMP] = np.asarray(J[2]); Jl[:, SL] = np.asarray(J[3])
        Jl[:, DEL[leg.name]] = np.asarray(J[4])

        # P_0 = production forward at truth (s=0, δ=0, DLA masked); ΔP = truth_masked(on leg) − P_0
        P0 = np.asarray(fwd(th_truth, t0_truth, apivot, s0, d0))
        z_leg = st["z_leg"]; k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        Pfull = np.zeros(k.shape[0])
        for iz in range(leg.n_z):
            r = np.where(z_idx == iz)[0]
            j = int(np.argmin(np.abs(z_sim - float(z_leg[iz]))))
            Pfull[r] = np.asarray(jnp.interp(jnp.asarray(k[r]), jnp.asarray(cache_k),
                                             jnp.asarray(P_sim[j])))
        dP = Pfull[keep] - P0                                             # (Nkeep,) EXACT misspec residual

        # pipeline-validation baseline: forward at the SIM's OWN per-z w_c (masked), s=0, δ=0.
        # ΔP_emu = truth − P_exact should isolate emulator error only (→ known +0.03σ baseline).
        z_leg = st["z_leg"]
        sel_sim = np.array([int(np.argmin(np.abs(z_sim - zz))) for zz in z_leg])
        al_exact = jnp.asarray(wc_perz_sim[sel_sim])                       # (n_z,3) sim per-z w_c (DLA masked)
        P_exact, _ = DL.predict_P_obs_on_leg(
            ctx.model, th_truth, t0_truth[sel], al_exact, pf_stats=ctx.pf_stats,
            dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
        dP_emu = Pfull[keep] - np.asarray(P_exact)[keep]

        _, Ctot = DL.predict_P_obs_on_leg(
            ctx.model, th_truth, t0_truth[sel], alpha_of_s(gfix, lnz, apivot, s0, d0),
            pf_stats=ctx.pf_stats, dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg,
            sigma_zb=szb, alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl
        gvec += Jl.T @ Cinv @ dP                                          # full misspec force (shape+amp+emu)
        gvec_emu += Jl.T @ Cinv @ dP_emu                                  # emulator-error-only force (baseline)

    # prior precision base: tau0 + a_pivot amplitude + per-leg δ anchor; DLA components pinned (masked).
    Pp_base = np.zeros(nP)
    Pp_base[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp_base[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Pp_base[AMP[DLA_J]] = PIN; Pp_base[SL[DLA_J]] = PIN   # DLA amp + slope pinned (masked, moot)
    for leg in ctx.legs:
        sig_anch = SIGMA_ANCHOR[leg.name]
        Pp_base[DEL[leg.name]] = 1.0 / sig_anch ** 2
        Pp_base[DEL[leg.name][DLA_J]] = PIN              # DLA δ pinned (no float; masked)

    def metrics(sigma_s_vec):
        Pp = Pp_base.copy()
        Pp[SL] = 1.0 / np.asarray(sigma_s_vec) ** 2
        Pp[SL[DLA_J]] = PIN                              # keep DLA slope pinned regardless of width
        M = F + np.diag(Pp)
        Cpost = np.linalg.inv(M + 1e-15 * np.mean(np.diag(M)) * np.eye(nP))
        sAp = np.sqrt(Cpost[AP_I, AP_I]); sNs = np.sqrt(Cpost[NS_I, NS_I])
        # EXACT MAP-shift bias: x̂ − x_true = (F+P)^{-1} g   (g = Jᵀ Cinv ΔP; ΔP exact, no shape linearization)
        dx = Cpost @ gvec            # full (shape + amplitude + emulator) misspecification bias
        dxe = Cpost @ gvec_emu       # emulator-error-only baseline (forward at the sim's exact shape)
        return sAp, sNs, float(dx[AP_I]), float(dx[NS_I]), float(dxe[AP_I]), float(dxe[NS_I])

    return metrics, mt["params_unit"]

# ---- run all 8 fold-0 held-out sims + interior-n_s spot checks
# Bias columns: FULL = (lit-shape + lit-amp + emulator) misspec; EMU = emulator-only baseline
# (forward at the sim's EXACT per-z w_c → the known +0.03σ case). FULL−EMU = the shape+amplitude
# misspec the slope nuisance + per-leg anchor are meant to absorb. Slope width is varied; if FULL
# is width-insensitive, the slope is not the driver (the anchor/amplitude is).
sims0, _ = held_out_sims(d, 0)
emit(f"\n# ===== fold-0 HELD-OUT sims (emulator never saw them) — n={len(sims0)} =====")
emit(f"# z-edge inflation factor (mean over legs, beta={EDGE_BETA}) = {edge_mean:.2f}  "
     f"sigma_anchor={SIGMA_ANCHOR}")
emit("# sim_label                      ns_u  Ap_u | width sig(Ap) sig(ns) | bias/σ  A_p(FULL/EMU)  n_s(FULL/EMU)")
agg = {"lit": [], "edge": [], "2x": []}
for sim in sims0:
    metrics, pu = run_one_sim(sim, 0, sim[:26])
    for tag, ss in (("lit", SIGMA_S_LIT), ("edge", sigma_s_edge), ("2x", 2 * SIGMA_S_LIT)):
        sAp, sNs, bAp, bNs, beAp, beNs = metrics(ss)
        agg[tag].append((bAp / sAp, bNs / sNs, sAp, sNs, beAp / sAp, beNs / sNs))
    sAp, sNs, bAp, bNs, beAp, beNs = metrics(SIGMA_S_LIT)
    emit(f"  {sim[:30]:30s} {pu[NS_I]:.3f} {pu[AP_I]:.3f} | lit  {sAp:.4f}  {sNs:.4f} | "
         f"A_p {bAp/sAp:+.3f}/{beAp/sAp:+.3f}  n_s {bNs/sNs:+.3f}/{beNs/sNs:+.3f}")

def summ(tag):
    a = np.array(agg[tag])
    return (f"  [{tag:4s}] FULL bias/σ  A_p: mean {a[:,0].mean():+.3f} max|{np.abs(a[:,0]).max():.3f}|  "
            f"n_s: mean {a[:,1].mean():+.3f} max|{np.abs(a[:,1]).max():.3f}|  ||  "
            f"EMU-only  A_p: mean {a[:,4].mean():+.3f}  n_s: mean {a[:,5].mean():+.3f}  ||  "
            f"σ(A_p) {a[:,2].mean():.4f} σ(n_s) {a[:,3].mean():.4f}")
emit("\n# ===== fold-0 SUMMARY (gate: |bias|<0.2σ on BOTH A_p and n_s; EMU = pipeline baseline ~+0.03σ) =====")
for tag in ("lit", "edge", "2x"):
    emit(summ(tag))

# interior-n_s spot checks (emulator IN-SAMPLE — shape-gap bias only; flagged)
emit("\n# ===== INTERIOR-n_s spot checks (emulator IN-SAMPLE; shape-gap bias valid, emu error understated) =====")
emit("# sim_label                      ns_u  Ap_u | width sig(Ap) sig(ns) | bias/σ  A_p(FULL/EMU)  n_s(FULL/EMU)")
for fold in (3, 4, 5):
    _tr, va, _ho = make_splits(d, fold)
    names = np.asarray(d["sim_name"]); pu_all = np.asarray(d["params_unit"])
    # pick the sim closest to ns_unit 0.5 in this fold's val set
    vsims = sorted(set(names[va]))
    best = min(vsims, key=lambda s: abs(pu_all[np.where(names == s)[0][0], NS_I] - 0.5))
    metrics, pu = run_one_sim(best, fold, best[:26])
    sAp, sNs, bAp, bNs, beAp, beNs = metrics(SIGMA_S_LIT)
    emit(f"  {best[:30]:30s} {pu[NS_I]:.3f} {pu[AP_I]:.3f} | lit  {sAp:.4f}  {sNs:.4f} | "
         f"A_p {bAp/sAp:+.3f}/{beAp/sAp:+.3f}  n_s {bNs/sNs:+.3f}/{beNs/sNs:+.3f}  (fold {fold})")

# ---- figure: per-sim bias(A_p) and bias(n_s) at lit + edge widths, with the 0.2σ gate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a_lit = np.array(agg["lit"])     # cols: 0 fullAp/σ, 1 fullNs/σ, 2 σAp, 3 σNs, 4 emuAp/σ, 5 emuNs/σ
x = np.arange(len(sims0))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
for col, (nm, gfull, gemu) in enumerate((("A_p", 0, 4), ("n_s", 1, 5))):
    ax[col].axhline(0.2, color="r", ls="--", alpha=0.6, label="±0.2σ gate")
    ax[col].axhline(-0.2, color="r", ls="--", alpha=0.6)
    ax[col].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax[col].plot(x, a_lit[:, gfull], "o-", color="C3", label="FULL misspec (shape+amp+emu)")
    ax[col].plot(x, a_lit[:, gemu], "s--", color="C0", label="EMU-only baseline (sim exact shape)")
    ax[col].set_xticks(x); ax[col].set_xticklabels([s[:10] for s in sims0], rotation=40, ha="right", fontsize=7)
    ax[col].set_ylabel(f"bias({nm}) / σ({nm})  [in σ]"); ax[col].grid(alpha=0.3)
    ax[col].set_title(f"{nm}: exact misspec bias, 8 fold-0 held-out sims"); ax[col].legend(fontsize=8)
fig.suptitle("HCD slope-prior RERUN — production g_fixed, DLA-masked (both sides), per-leg anchor, "
             "EXACT bias; lit width; gate ±0.2σ. EMU baseline = pipeline check (~+0.03σ target)")
p = Path(FIGDIR) / "legb_slope_prior_rerun.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}")
emit("[done]")
