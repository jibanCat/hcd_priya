"""Per-leg likelihood-only Fisher sigma(n_s) for the three Lyα-P1D survey legs (KS, DESI, eBOSS).

LOAD-BEARING DIAGNOSTIC (read-only on the modules; a NEW script): demonstrate that the KS data
covariance leaves n_s PRIOR-DOMINATED, justifying the KS-first unblinding order.

Modeled EXACTLY on the validated scripts/_cov_fisher.py:
  * build the deployed production Leg-B ctx (same flags),
  * the same differentiable forward_leg(params, leg, ...),
  * the same per-leg covariance assembly,
  * add the prior-precision Lambda (HCD Gaussian alpha sites ONLY; the 9 cosmo params + tau0_amp +
    dtau0 are Uniform -> 0 precision).

Param vector (14) = theta9 (UNIT-CUBE [0,1]) + [tau0_amp, dtau0] + [alpha_lls, alpha_subdla, alpha_dla].
n_s is theta9[0] in unit-cube units => Uniform(0,1) prior sd = 1/sqrt(12) = 0.28868.

Steps:
 1. JOINT sanity anchor: reproduce sigma(n_s) ~ 0.0953, sigma(Ap) ~ 0.3584 (no Lambda, matching
    _cov_fisher_table.json key sigma.full). STOP if not within a few %.
 2. PER-LEG Fisher: F_leg = J_leg^T C_leg^-1 J_leg + Lambda, sigma = sqrt(diag(F_leg^-1)).
 3. ratio r = sigma_like(n_s|leg) / 0.28868 ; r >= 1 => prior-dominated.
"""
import os, glob, json
import numpy as np
import hcd_analysis.emulator  # x64
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg)
import hcd_analysis.emulator.data_likelihood as DL
from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya, KIM_AMP, KIM_SLOPE, TAU0_PIVOT_Z

FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
REPO = "/home/mfho/hcd_priya"
PARAM_NAMES = ["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback",
               "tau0_amp","dtau0","alpha_lls","alpha_subdla","alpha_dla"]
NPAR = 14
SD_PRIOR_NS = 1.0/np.sqrt(12.0)   # Uniform(0,1) sd on the unit cube = 0.288675

# joint sanity targets (from _cov_fisher_table.json, key sigma.full)
TARGET_NS, TARGET_AP = 0.0952766253544456, 0.3584244082946373

members = sorted(p[:-4] for p in glob.glob(f"{REPO}/checkpoints/final_prod_seed*.eqx"))
assert members, "no final_prod_seed*.eqx ensemble members found"
print(f"ensemble members ({len(members)}):", [os.path.basename(m) for m in members])

ctx, d = build_legb_ctx(
    ensemble_ckpts=members, use_xclass=True, with_mf=True, mf_with_floor=True,
    mf_emucoh=True, mf_emucoh_offdiag_only=True, with_eboss=True, metals_on=True,
    sample_metals=True, hierarchical_hcd=False)
legs = ctx.legs
print("legs:", [l.name for l in legs])
sims, _ = held_out_sims(d, fold=0)
truth = make_truth_from_sim(d, sims[0], fold=0, mf=ctx.mf)
mock_legs, tp, info = make_legb_mock(ctx, truth, jax.random.fold_in(jax.random.PRNGKey(0), 0))
theta9_t = jnp.asarray(tp["theta9"]); alpha_t = jnp.asarray(tp["alpha_hcd"])
zg = np.asarray(ctx.z_global)
core_per_leg = _mock_core_per_leg(ctx, truth)
amp_t, dt_t = float(truth["tau0_amp"]), float(truth["dtau0"])
print("truth tau0_amp,dtau0 =", round(amp_t,4), round(dt_t,4))

# prior precision Lambda: HCD Gaussian alpha sites ONLY (indices 11,12,13).
# alpha_hcd_sigma is the (3,) HCD-incidence prior width (LLS, subDLA, DLA), in physical alpha units.
alpha_sd = np.asarray(ctx.alpha_hcd_sigma, dtype=float)   # (3,)
print("alpha_hcd_sigma (LLS,subDLA,DLA):", alpha_sd)
Lambda = np.zeros((NPAR, NPAR))
for j in range(3):
    Lambda[11+j, 11+j] = 1.0/(alpha_sd[j]**2)

# precompute the per-leg emucoh covariance (the deployed off-diagonal C_emu term)
emucoh_cov = {l.name: (ctx.mf_emucoh_per_leg.get(l.name) if ctx.mf_emucoh_per_leg else None)
              for l in legs}

def _kim(z): return KIM_AMP*(1.0+jnp.asarray(z))**KIM_SLOPE

def forward_leg(params, leg):
    """Differentiable (P_model, C_total) for a leg from the flat 14-param vector, deployed config."""
    th9 = params[:9]
    tau0_amp, dtau0 = params[9], params[10]
    alpha = params[11:14]
    z_leg = jnp.asarray(np.asarray(leg.z))
    tau0_vec = tau0_alpha_priya(z_leg, tau0_amp, dtau0) * _kim(z_leg)
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    mec = emucoh_cov[leg.name]
    P, C = DL.predict_P_obs_on_leg(
        ctx.model, th9, tau0_vec, alpha, pf_stats=ctx.pf_stats, dla_core=core_per_leg[leg.name],
        cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
        cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor,
        mf_shape_cov=None, mf_shape_infl=ctx.mf_shape_infl,
        mf_emucoh_cov=mec, mf_emucoh_infl=ctx.mf_emucoh_infl,
        mf_emucoh_offdiag_only=ctx.mf_emucoh_offdiag_only)
    return P, C

params_t = jnp.asarray(np.concatenate([np.asarray(theta9_t), [amp_t, dt_t], np.asarray(alpha_t)]))

def Pmodel_leg(params, leg):
    return forward_leg(params, leg)[0]

# per-leg Jacobian J = dP/dparam (14 cols), and per-leg covariance C
J_leg, C_leg = {}, {}
for l in legs:
    J = np.asarray(jax.jacobian(lambda p: Pmodel_leg(p, l))(params_t))
    _, C = forward_leg(params_t, l)
    C = np.asarray(C)
    C = C + 1e-12*np.median(np.diag(C))*np.eye(C.shape[0])
    J_leg[l.name] = J; C_leg[l.name] = C
    print(f"J[{l.name}] shape={J.shape} finite={np.isfinite(J).all()}  "
          f"C[{l.name}] shape={C.shape} finite={np.isfinite(C).all()}")

def F_from_legs(leg_names, add_lambda):
    F = np.zeros((NPAR, NPAR))
    for nm in leg_names:
        J = J_leg[nm]; C = C_leg[nm]
        Cinv = np.linalg.inv(C)
        F += J.T @ Cinv @ J
    if add_lambda:
        F = F + Lambda
    return F

def sigmas(F):
    Finv = np.linalg.pinv(F)
    return np.sqrt(np.clip(np.diag(Finv), 0, None))

# ---------- STEP 1: JOINT sanity anchor (NO Lambda, matching _cov_fisher.py "full") ----------
all_names = [l.name for l in legs]
F_joint_noL = F_from_legs(all_names, add_lambda=False)
s_joint_noL = sigmas(F_joint_noL)
s_ns_joint = float(s_joint_noL[0]); s_ap_joint = float(s_joint_noL[1])
rel_ns = abs(s_ns_joint - TARGET_NS)/TARGET_NS
rel_ap = abs(s_ap_joint - TARGET_AP)/TARGET_AP
print("\n=== JOINT SANITY (no Lambda) ===")
print(f"sigma(ns)_joint = {s_ns_joint:.6f}  target {TARGET_NS:.6f}  rel.diff {rel_ns*100:.2f}%")
print(f"sigma(Ap)_joint = {s_ap_joint:.6f}  target {TARGET_AP:.6f}  rel.diff {rel_ap*100:.2f}%")
print(f"contraction sigma(ns)/sd_prior = {s_ns_joint/SD_PRIOR_NS:.4f} (expect ~0.33)")
SANITY_PASS = (rel_ns < 0.03) and (rel_ap < 0.03)
if not SANITY_PASS:
    print(f"\n*** SANITY FAIL: joint sigma(ns)={s_ns_joint:.6f} not within 3% of {TARGET_NS:.6f}. STOP. ***")
    with open(f"{FIG}/perleg_fisher_ns.json", "w") as f:
        json.dump(dict(sanity_pass=False, sigma_ns_joint=s_ns_joint, sigma_ap_joint=s_ap_joint,
                       target_ns=TARGET_NS, target_ap=TARGET_AP,
                       rel_diff_ns=rel_ns, rel_diff_ap=rel_ap, sd_prior=SD_PRIOR_NS), f, indent=1)
    raise SystemExit("joint sanity check failed -- aborting before per-leg, per the spec.")
print("SANITY PASS -> proceeding to per-leg.")

# joint WITH Lambda (the deployed prior on alpha sites) -- the apples-to-apples joint vs per-leg.
F_joint = F_from_legs(all_names, add_lambda=True)
s_joint = sigmas(F_joint)

# ---------- STEP 2/3: PER-LEG Fisher (WITH Lambda) ----------
per_leg = {}
for nm in all_names:
    s = sigmas(F_from_legs([nm], add_lambda=True))
    per_leg[nm] = s

def row(s):
    return {p: float(s[i]) for i, p in enumerate(PARAM_NAMES)}

report_params = ["ns","Ap","tau0_amp","dtau0","alpha_lls","alpha_subdla","alpha_dla"]
print("\n=== PER-LEG sigma_like(param) [WITH Lambda on alpha sites] ===")
hdr = "param".ljust(13) + "".join(nm.ljust(13) for nm in all_names) + "joint".ljust(13)
print(hdr)
for i, p in enumerate(PARAM_NAMES):
    line = p.ljust(13) + "".join(f"{per_leg[nm][i]:.4g}".ljust(13) for nm in all_names)
    line += f"{s_joint[i]:.4g}".ljust(13)
    print(line)

print("\n=== n_s prior-domination (ratio r = sigma_like(ns|leg)/sd_prior, sd_prior=%.5f) ===" % SD_PRIOR_NS)
print("(r >= 1 => likelihood WIDER than prior => n_s PRIOR-DOMINATED for that leg)")
ratios = {}
for nm in all_names:
    r = per_leg[nm][0]/SD_PRIOR_NS
    ratios[nm] = float(r)
    print(f"  {nm.ljust(8)} sigma_like(ns)={per_leg[nm][0]:.4f}  r={r:.4f}  "
          f"{'PRIOR-DOMINATED' if r>=1 else 'likelihood-informed'}")
r_joint = float(s_joint[0]/SD_PRIOR_NS)
print(f"  {'joint'.ljust(8)} sigma_like(ns)={s_joint[0]:.4f}  r={r_joint:.4f}")

# ---------- OUTPUTS ----------
out = dict(
    sanity_pass=True,
    sd_prior=float(SD_PRIOR_NS),
    target_ns=TARGET_NS, target_ap=TARGET_AP,
    joint_no_lambda=dict(sigma_ns=s_ns_joint, sigma_ap=s_ap_joint,
                         rel_diff_ns=float(rel_ns), rel_diff_ap=float(rel_ap),
                         contraction_ns=float(s_ns_joint/SD_PRIOR_NS)),
    joint_with_lambda=row(s_joint),
    joint_ratio_ns=r_joint,
    alpha_hcd_sigma=[float(x) for x in alpha_sd],
    leg_names=all_names,
    per_leg={nm: row(per_leg[nm]) for nm in all_names},
    ratio_ns={nm: ratios[nm] for nm in all_names},
    param_names=PARAM_NAMES,
)
with open(f"{FIG}/perleg_fisher_ns.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote {FIG}/perleg_fisher_ns.json")

# ---------- FIGURE ----------
# order bars KS, DESI, eBOSS, joint where present
desired = ["KS","DESI","eBOSS"]
bar_legs = [nm for nm in desired if nm in all_names] + [nm for nm in all_names if nm not in desired]
bar_labels = bar_legs + ["joint"]
ns_vals = [per_leg[nm][0] for nm in bar_legs] + [float(s_joint[0])]
ap_vals = [per_leg[nm][1] for nm in bar_legs] + [float(s_joint[1])]
ns_ratios = [v/SD_PRIOR_NS for v in ns_vals]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
colors = ["#d95f02","#1b9e77","#7570b3","#666666"][:len(bar_labels)]
bars = ax.bar(bar_labels, ns_vals, color=colors)
ax.axhline(SD_PRIOR_NS, color="k", ls="--", lw=1.5,
           label=f"prior sd = 1/√12 = {SD_PRIOR_NS:.3f}")
for b, v, r in zip(bars, ns_vals, ns_ratios):
    ax.text(b.get_x()+b.get_width()/2, v + 0.005, f"r={r:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel(r"$\sigma_{\rm like}(n_s)$  [unit-cube]")
ax.set_title("Per-leg likelihood-only Fisher $\\sigma(n_s)$\n(r = σ_like/σ_prior; r≥1 ⇒ prior-dominated)")
ax.set_ylim(0, max(max(ns_vals), SD_PRIOR_NS)*1.25)
ax.legend(loc="upper right")

ax = axes[1]
bars = ax.bar(bar_labels, ap_vals, color=colors)
for b, v in zip(bars, ap_vals):
    ax.text(b.get_x()+b.get_width()/2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel(r"$\sigma_{\rm like}(A_p)$  [unit-cube]")
ax.set_title("Per-leg likelihood-only Fisher $\\sigma(A_p)$")
ax.set_ylim(0, max(ap_vals)*1.2)

plt.tight_layout()
plt.savefig(f"{FIG}/perleg_fisher_ns.png", dpi=130)
plt.close()
print(f"wrote {FIG}/perleg_fisher_ns.png")
print("done.")
