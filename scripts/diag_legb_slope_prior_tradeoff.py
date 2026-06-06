"""Slope-prior tradeoff sweep (forward-only, no NUTS) for the HCD-incidence z-slope nuisance.

PI decision (2026-06-06): the closure forward factorizes the HCD incidence z-shape as
    alpha_c(z) = alpha_pivot_c * g_fixed_c(z) * ((1+z)/(1+z_p))^{s_c}
where g_fixed_c(z) = w_c,sim(z)/w_c,sim(z_p) is the KNOWN steep abundance shape threaded from
the truth (sim w_c(z), abs slope ~2.6; the +0.03 sigma EXACT case), and s_c is the small
LITERATURE-correction ratio-slope, SAMPLED as a nuisance. The prior on s_c is anchored to the
literature dN/dX-slope uncertainty. This script measures the bias<->variance tradeoff on A_p as a
function of the slope-prior WIDTH, so the PI can pick the width on a real curve.

Two quantities vs the slope-prior width sigma_s (per class, scanned by a multiplier m on the
literature base widths):
  (1) VARIANCE   sigma(A_p), sigma(n_s) = sqrt diag of (F + P)^{-1}, slope MARGINALIZED.
  (2) BIAS       bias(A_p) when the TRUE incidence slope is offset from the forward's prior
                 center by the sim-vs-lit gap Delta_s. Linearized:
                   x_hat - x_true = (F+P)^{-1} P (x0 - x_true),  P nonzero only on the slope block
                   => bias(theta) = sum_c [(F+P)^{-1}]_{theta, s_c} * (1/sigma_{s,c}^2) * (s_center,c - s_true,c)
  As sigma_s grows: bias shrinks (the slope nuisance absorbs the offset) but sigma(A_p) inflates.
  The knee is the sweet spot; the literature width (m=1) and the z-edge-inflated width are marked.

Cross-check: the linear J_s * Delta_s is compared to the exact nonlinear Delta P = P(s=Delta_s) - P(s=0)
so the linearization is trusted (not assumed).

Reuses: scripts/diag_legb_perf_and_binding.py (the Fisher: jacrev -> J^T Cinv J + prior precision
-> invert) and scripts/diag_legb_zresolved_alpha_check.py (the forward construction path).
Forward-only; ~minutes; no NUTS. Writes a results table + figure to figures/analysis/05_likelihood/.
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
)
from hcd_analysis.emulator.closure_legb_figs import truth_tau0_on_leg
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT

FIGDIR = "/home/mfho/hcd_priya/figures/analysis/05_likelihood"
RESULTS = f"{FIGDIR}/legb_slope_prior_tradeoff.txt"
HOLD0 = "/home/mfho/hcd_priya/checkpoints/error_vector_xclass_holdout0.npz"  # de-circularized rho

# --- Literature slope priors (Lyα consult 2026-06-06; WLS fit of dN/dX vs PRIYA, errors propagated)
#     class order (LLS, subDLA, DLA). The code's HCD_LIT_OVER_SIM_SLOPE has NO uncertainty;
#     these widths come from a weighted re-fit with the quoted literature dN/dX error bars.
S_CENTER  = np.array([0.76, 0.05, 1.16])   # ratio-slope center s_c (WLS); code uses (0.95,0.15,0.40)
SIGMA_S_LIT = np.array([0.52, 0.53, 0.33]) # literature 1sigma on s_c (full-window WLS)
SIM_LIT_GAP = np.array([0.76, 0.05, 1.16]) # sim-vs-lit gap on the slope axis = s_c (the bias offset to test)
# z=2.5-3.5 trust window: outside it the quoted errors are overconfident (purity below /
# completeness above) -> inflate. edge(z)=1+beta*(clip(2.5-z,0)+clip(z-3.5,0)); the per-leg scalar
# is the z-mean of edge over the leg's bins. beta=1 reproduces the existing DLA dla_inflate slope.
EDGE_BETA = 1.0
CLASSES = ["LLS", "subDLA", "DLA"]

_rf = open(RESULTS, "w")
def emit(s):
    print(s, flush=True); _rf.write(s + "\n"); _rf.flush()

NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])
emit(f"# slope-prior tradeoff  PARAM_NAMES={list(PARAM_NAMES)}  n_s idx {NS_I}  A_p idx {AP_I}")
emit(f"# s_center={S_CENTER}  sigma_s_lit={SIGMA_S_LIT}  sim_lit_gap={SIM_LIT_GAP}  z_pivot={HCD_Z_PIVOT}")

# ----------------------------------------------------------------- build ctx + truth fiducial
ctx, d = build_legb_ctx(xclass_error_vector=HOLD0)
sims, _ = held_out_sims(d, 0)
truth = make_truth_from_sim(d, sims[0], 0)
mock_legs, tp, info = make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
core = _mock_core_per_leg(ctx, truth)
cache_k = np.asarray(ctx.cache_k)
th_truth = jnp.asarray(tp["theta9"]); t0_truth = jnp.asarray(tp["tau0_global"])
zg = np.asarray(ctx.z_global); nZg = zg.size

# per-z sim w_c and the pivot amplitude (the EXACT-case operating point, +0.03 sigma)
z_sim = np.asarray(truth["z"])
wc_perz = np.asarray(d["w_c_cache"])[np.asarray(truth["rows"]), 1:]   # (n_zsim, 3) per-z w_c
ip = int(np.argmin(np.abs(z_sim - HCD_Z_PIVOT))); wc_pivot = wc_perz[ip]   # (3,)
apivot_truth = jnp.asarray(wc_pivot)
s0 = jnp.zeros(3)
emit(f"# fiducial sim={sims[0][:26]}  z_pivot_row z={z_sim[ip]:.2f}  wc_pivot={np.round(wc_pivot,4)}")
emit(f"# A_p_unit={float(th_truth[AP_I]):.3f}  n_s_unit={float(th_truth[NS_I]):.3f}  nZ_global={nZg}")

# per-leg precompute: g_fixed(z)=wc(z)/wc_pivot and lnz=ln((1+z)/(1+z_p)) on the leg z grid
leg_static = {}
for leg in ctx.legs:
    z_leg = np.asarray(leg.z)
    sel_sim = np.array([int(np.argmin(np.abs(z_sim - zz))) for zz in z_leg])
    g_fixed = wc_perz[sel_sim] / wc_pivot[None, :]                 # (n_z,3)
    lnz = np.log((1.0 + z_leg) / (1.0 + HCD_Z_PIVOT))             # (n_z,)
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in z_leg])   # leg z -> global tau0 idx
    edge = 1.0 + EDGE_BETA * (np.clip(2.5 - z_leg, 0, None) + np.clip(z_leg - 3.5, 0, None))
    leg_static[leg.name] = dict(g_fixed=jnp.asarray(g_fixed), lnz=jnp.asarray(lnz),
                                sel=jnp.asarray(sel), z_leg=z_leg, edge_mean=float(np.mean(edge)))

def alpha_of_s(g_fixed, lnz, apivot, s):
    """alpha_c(z) = apivot_c * g_fixed_c(z) * exp(lnz(z) * s_c)  -> (n_z, 3)."""
    return apivot[None, :] * g_fixed * jnp.exp(lnz[:, None] * s[None, :])

# ----------------------------------------------------------------- Fisher over [theta9, tau0, apivot, slope]
nP = 9 + nZg + 6
F = np.zeros((nP, nP))
keep_leg = {leg.name: np.where(np.isfinite(np.asarray(ml.P_data)))[0]
            for leg, ml in zip(ctx.legs, mock_legs)}
lin_err = []  # nonlinear cross-check of J_s . Delta_s
for leg, ml in zip(ctx.legs, mock_legs):
    st = leg_static[leg.name]
    keep = jnp.asarray(keep_leg[leg.name]); sel = st["sel"]
    g_fixed = st["g_fixed"]; lnz = st["lnz"]
    szb = ctx.sigma_zb_per_leg.get(leg.name); rzb = ctx.rho_zb_per_leg.get(leg.name)

    def fwd(th, t0g, apivot, s, _leg=leg, _sel=sel, _keep=keep, _g=g_fixed, _l=lnz):
        al = alpha_of_s(_g, _l, apivot, s)                        # (n_z,3) per-z incidence
        P, _ = DL.predict_P_obs_on_leg(
            ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats, dla_core=core[_leg.name],
            cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
        return P[_keep]

    J = jax.jacrev(fwd, argnums=(0, 1, 2, 3))(th_truth, t0_truth, apivot_truth, s0)
    Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2], J[3]], axis=1))   # (Nkeep, nP)
    # C_total at the truth (de-biased s=0 operating point): full cosmic + cross-class C_emu
    al0 = alpha_of_s(g_fixed, lnz, apivot_truth, s0)
    _, Ctot = DL.predict_P_obs_on_leg(
        ctx.model, th_truth, t0_truth[sel], al0, pf_stats=ctx.pf_stats, dla_core=core[leg.name],
        cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
        cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
    keep_np = np.asarray(keep_leg[leg.name])
    Cii = np.asarray(Ctot)[np.ix_(keep_np, keep_np)]
    Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
    F += Jl.T @ Cinv @ Jl

    # nonlinear cross-check: J_s . gap  vs  P(s=gap) - P(s=0), on this leg
    Js = Jl[:, -3:]
    dP_lin = Js @ SIM_LIT_GAP
    P_s = np.asarray(fwd(th_truth, t0_truth, apivot_truth, jnp.asarray(SIM_LIT_GAP)))
    P_0 = np.asarray(fwd(th_truth, t0_truth, apivot_truth, s0))
    dP_nl = P_s - P_0
    denom = np.linalg.norm(dP_nl) + 1e-300
    lin_err.append((leg.name, float(np.linalg.norm(dP_lin - dP_nl) / denom)))

emit("\n# nonlinear cross-check  ||J_s.gap - (P(gap)-P(0))|| / ||P(gap)-P(0)||  (small => linear OK):")
for nm, e in lin_err:
    emit(f"#   {nm}: {e:.3e}")

# ----------------------------------------------------------------- prior precision + scan
Pp_base = np.zeros(nP)
Pp_base[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2          # tau0 Gaussian
Pp_base[9 + nZg:9 + nZg + 3] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2  # alpha_pivot amplitude
SL = np.arange(nP - 3, nP)        # slope param indices
AMP = np.arange(9 + nZg, 9 + nZg + 3)

def post_cov(sigma_s_vec):
    """(F + P)^{-1} with the slope block prior = 1/sigma_s^2."""
    Pp = Pp_base.copy()
    Pp[SL] = 1.0 / np.asarray(sigma_s_vec) ** 2
    M = F + np.diag(Pp)
    return np.linalg.inv(M + 1e-15 * np.mean(np.diag(M)) * np.eye(nP)), Pp

def metrics(sigma_s_vec, gap=SIM_LIT_GAP):
    Cpost, Pp = post_cov(sigma_s_vec)
    sAp = np.sqrt(Cpost[AP_I, AP_I]); sNs = np.sqrt(Cpost[NS_I, NS_I])
    # bias from a slope offset gap (truth = center - (-gap); center 0, truth = gap):
    # bias(theta) = sum_c [Cpost]_{theta, s_c} * Pp[s_c] * (0 - gap_c)
    bias_ap = float(np.sum(Cpost[AP_I, SL] * Pp[SL] * (-gap)))
    bias_ns = float(np.sum(Cpost[NS_I, SL] * Pp[SL] * (-gap)))
    # per-class A_p bias decomposition
    perclass = {CLASSES[i]: float(Cpost[AP_I, SL[i]] * Pp[SL[i]] * (-gap[i])) for i in range(3)}
    return sAp, sNs, bias_ap, bias_ns, perclass

# reference endpoints
sig_pinned = 1e-3 * np.ones(3)     # slope ~fixed (prior >> data) => min variance, max bias
sig_free = 1e3 * np.ones(3)        # slope free => max variance, ~0 bias
sAp_pin, sNs_pin, *_ = metrics(sig_pinned)
sAp_free, sNs_free, *_ = metrics(sig_free)
emit(f"\n# endpoints: slope-PINNED sigma(A_p)={sAp_pin:.4f}  slope-FREE sigma(A_p)={sAp_free:.4f} "
     f"(inflation x{sAp_free/sAp_pin:.2f})")

# edge-inflated literature width (z-mean over the union of leg z)
edge_mean = float(np.mean([leg_static[l.name]["edge_mean"] for l in ctx.legs]))
sigma_s_edge = SIGMA_S_LIT * edge_mean
emit(f"# z-edge inflation factor (mean over legs, beta={EDGE_BETA}) = {edge_mean:.2f} "
     f"-> edge-inflated sigma_s={np.round(sigma_s_edge,3)}")

mults = np.logspace(-1.0, 0.7, 22)          # 0.1 .. ~5 x the literature width
rows = []
emit("\n# m   sigma_s(LLS)  sigma(A_p)  sigma(n_s)  bias(A_p)/sig  bias(n_s)/sig  | A_p-bias by class (LLS/sub/DLA, in sigma)")
for m in mults:
    ss = m * SIGMA_S_LIT
    sAp, sNs, bAp, bNs, pc = metrics(ss)
    rows.append((m, ss[0], sAp, sNs, bAp, bNs, pc["LLS"], pc["subDLA"], pc["DLA"]))
    emit(f"  {m:5.2f}  {ss[0]:8.3f}   {sAp:.4f}    {sNs:.4f}    {bAp/sAp:+7.3f}     {bNs/sNs:+7.3f}    "
         f"{pc['LLS']/sAp:+.3f}/{pc['subDLA']/sAp:+.3f}/{pc['DLA']/sAp:+.3f}")

# headline at the literature width (m=1) and the edge-inflated width
sAp1, sNs1, bAp1, bNs1, pc1 = metrics(SIGMA_S_LIT)
sApe, sNse, bApe, bNse, pce = metrics(sigma_s_edge)
emit(f"\n# LITERATURE width (m=1):      sigma(A_p)={sAp1:.4f}  bias(A_p)={bAp1/sAp1:+.3f} sigma  "
     f"(LLS {pc1['LLS']/sAp1:+.3f}, subDLA {pc1['subDLA']/sAp1:+.3f})")
emit(f"# EDGE-INFLATED width:         sigma(A_p)={sApe:.4f}  bias(A_p)={bApe/sApe:+.3f} sigma  "
     f"(LLS {pce['LLS']/sApe:+.3f}, subDLA {pce['subDLA']/sApe:+.3f})")

# ----------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = np.array([r[0] for r in rows]); sAp_a = np.array([r[2] for r in rows])
sNs_a = np.array([r[3] for r in rows]); bAp_a = np.array([r[4]/r[2] for r in rows])
bLLS = np.array([r[6]/r[2] for r in rows]); bSub = np.array([r[7]/r[2] for r in rows])
xs = M * SIGMA_S_LIT[0]    # x-axis in absolute sigma_s(LLS) units

fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
ax[0].plot(xs, sAp_a, "o-", color="C0", label=r"$\sigma(A_p)$ [unit-cube]")
ax[0].plot(xs, sNs_a, "s--", color="C2", label=r"$\sigma(n_s)$ [unit-cube]")
ax[0].axhline(sAp_pin, color="C0", ls=":", alpha=0.6, label=r"slope-pinned $\sigma(A_p)$")
ax[0].axvline(SIGMA_S_LIT[0], color="k", ls="-", alpha=0.5, label=r"lit width $\sigma_s^{LLS}=0.52$")
ax[0].axvline(sigma_s_edge[0], color="grey", ls="--", alpha=0.6, label=f"edge-inflated ({edge_mean:.2f}x)")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"slope-prior width $\sigma_s$(LLS)")
ax[0].set_ylabel("forecast $\\sigma$ [unit-cube]"); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8); ax[0].set_title("VARIANCE: marginalizing the slope inflates $\\sigma(A_p)$")

ax[1].plot(xs, np.abs(bAp_a), "o-", color="C3", label=r"|bias($A_p$)| total")
ax[1].plot(xs, np.abs(bLLS), "^--", color="C1", label="|bias| from LLS slope")
ax[1].plot(xs, np.abs(bSub), "v:", color="C4", label="|bias| from subDLA slope")
ax[1].axhline(0.2, color="r", ls="--", alpha=0.6, label="0.2$\\sigma$ gate")
ax[1].axvline(SIGMA_S_LIT[0], color="k", ls="-", alpha=0.5)
ax[1].axvline(sigma_s_edge[0], color="grey", ls="--", alpha=0.6)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel(r"slope-prior width $\sigma_s$(LLS)")
ax[1].set_ylabel(r"|bias($A_p$)| / $\sigma(A_p)$  [in $\sigma$]"); ax[1].grid(alpha=0.3, which="both")
ax[1].legend(fontsize=8)
ax[1].set_title("BIAS: truth slope offset by the sim-vs-lit gap (0.76 LLS)")
fig.suptitle(f"HCD slope-prior tradeoff (forward-only Fisher; fiducial {sims[0][:18]}; "
             f"slope marginalized) — wider prior: less bias, more variance")
p = Path(FIGDIR) / "legb_slope_prior_tradeoff.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}")
emit("[done]")
