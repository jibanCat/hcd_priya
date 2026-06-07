"""All-folds, honestly-held-out re-measurement of the low-k A_p (and n_s) EMU bias + bootstrap CI.

Checkpoint-2 (2026-06-06) flagged the +0.62σ EMU A_p bias was an n=8 estimate (t≈1.58, 4pos/4neg)
on the fold-0 val sims, which are ALL low-n_s (box edge); the interior-n_s spot-checks used
final_fold0 on sims it TRAINED on (emulator in-sample). This script fixes both: it loops ALL 8 LOSO
folds, each with its OWN held-out emulator (checkpoints/final_fold{f}), measuring the EMU-only bias
(forward at the sim's EXACT per-z w_c, DLA-masked → pure emulator error) on each fold's honestly
held-out val sims. KS is capped at k<0.06 (the locked cap the earlier rerun violated). Reports the
per-fold and pooled A_p/n_s bias with a bootstrap CI over sims, and the bias vs n_s_unit (to test the
"low-n_s box-edge" confound).

EMU bias per sim, per leg: bias(param) = [(F+P)^{-1} Jᵀ C^{-1} ΔP_emu]_param / σ(param), with
ΔP_emu = P_truth_masked − P_forward(sim exact shape, s=0, δ=0); J wrt (theta9, tau0, a_pivot(3)).
Pure Fisher MAP-shift, forward-only, no NUTS. Same machinery as diag_legb_slope_prior_rerun.py /
diag_emu_lowk_investigation.py PART B, looped over folds.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_bias_allfolds.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/emu_bias_allfolds.txt"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KS_KMAX = 0.06   # the LOCKED KS cap (earlier rerun leaked to 0.0627)

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])

def emu_bias_one_sim(ctx, d, sim, fold):
    """EMU-only A_p/n_s bias (in σ) for one held-out sim, summed over legs. Forward at the sim's
    EXACT per-z w_c (DLA-masked) → ΔP is pure emulator error."""
    t = make_truth_from_sim(d, sim, fold)
    th_truth = jnp.asarray(t["params_unit"])
    wc = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy(); wc[:, 2] = 0.0  # mask DLA
    z_sim = np.asarray(t["z"]); P_sim = np.asarray(t["P_obs_true"])
    # rebuild DLA-masked truth P_obs (drop DLA class + core add-back), matching the forward mask
    P_filt = np.asarray(d["P_filt"]); rows = np.asarray(t["rows"]); K = P_filt.shape[-1]
    P_masked = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        a = wc[i]; coef = np.concatenate([[1.0 - a.sum()], a])
        P_masked[i] = np.einsum("c,ck->k", coef, np.stack([P_filt[r, 0], P_filt[r, 1], P_filt[r, 2], P_filt[r, 3]]))
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    zg = np.asarray(ctx.z_global); nZg = zg.size; nP = 9 + nZg + 3; AMP = np.arange(9 + nZg, 9 + nZg + 3)
    cache_k = np.asarray(ctx.cache_k)
    F = np.zeros((nP, nP)); g = np.zeros(nP)
    for leg in ctx.legs:
        z_leg = np.asarray(leg.z); k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        sel = np.array([int(np.argmin(np.abs(zg - z))) for z in z_leg])
        ss = np.array([int(np.argmin(np.abs(z_sim - z))) for z in z_leg])
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        gfix = wc[ss] / np.clip(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))], 1e-12, None)
        apv0 = jnp.asarray(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))])
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        def fwd(th, t0g, apv, _leg=leg, _sel=jnp.asarray(sel), _keep=jnp.asarray(keep), _g=jnp.asarray(gfix)):
            al = apv[None, :] * _g
            P, _ = DL.predict_P_obs_on_leg(ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats,
                dla_core=core[_leg.name], cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
            return P[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2))(th_truth, t0_truth, apv0)
        Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2]], axis=1))
        P0 = np.asarray(fwd(th_truth, t0_truth, apv0)); Pfull = np.zeros(k.shape[0])
        for iz in range(leg.n_z):
            rr = np.where(z_idx == iz)[0]; j = int(np.argmin(np.abs(z_sim - float(z_leg[iz]))))
            Pfull[rr] = np.asarray(jnp.interp(jnp.asarray(k[rr]), jnp.asarray(cache_k),
                jnp.asarray(np.interp(cache_k, cache_k, P_masked[j]))))  # truth on leg-k from masked P_obs
        dP = Pfull[keep] - P0
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel], apv0[None, :] * jnp.asarray(gfix),
            pf_stats=ctx.pf_stats, dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl; g += Jl.T @ Cinv @ dP
    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    dx = Cpost @ g
    return (dx[AP_I] / np.sqrt(Cpost[AP_I, AP_I]), dx[NS_I] / np.sqrt(Cpost[NS_I, NS_I]),
            float(t["params_unit"][NS_I]), float(t["params_unit"][AP_I]))

emit("# EMU bias, ALL 8 folds, each fold's OWN held-out emulator, KS capped k<0.06. forward-only Fisher.")
emit("# fold  sim                            ns_u  Ap_u   bias(A_p)/σ  bias(n_s)/σ")
allAp, allNs, allns_u = [], [], []
for fold in range(8):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0, ks_kwargs={"k_max": KS_KMAX})
    sims, _ = held_out_sims(d, fold)
    for sim in sims:
        bAp, bNs, ns_u, ap_u = emu_bias_one_sim(ctx, d, sim, fold)
        allAp.append(bAp); allNs.append(bNs); allns_u.append(ns_u)
        emit(f"  {fold}    {sim[:30]:30s} {ns_u:.3f} {ap_u:.3f}   {bAp:+7.3f}     {bNs:+7.3f}")
allAp = np.array(allAp); allNs = np.array(allNs); allns_u = np.array(allns_u)

def boot(x, n=10000):
    # deterministic bootstrap: fixed integer rng seeded by index sums (no Math.random in env policy);
    # use numpy default_rng with a fixed seed (this is a script, allowed).
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

emit(f"\n# ===== POOLED over ALL {len(allAp)} honestly-held-out sims =====")
for nm, arr in (("A_p", allAp), ("n_s", allNs)):
    m, lo, hi = boot(arr)
    se = arr.std(ddof=1) / np.sqrt(len(arr)); t = arr.mean() / se
    npos = int((arr > 0).sum()); nneg = int((arr < 0).sum())
    emit(f"  {nm}: mean {arr.mean():+.3f}σ  bootstrap95% [{lo:+.3f}, {hi:+.3f}]  t={t:+.2f}  "
         f"signs {npos}+/{nneg}-  max|{np.abs(arr).max():.2f}|")
# drop-largest-|outlier| robustness
for nm, arr in (("A_p", allAp), ("n_s", allNs)):
    j = int(np.argmax(np.abs(arr))); keep = np.ones(len(arr), bool); keep[j] = False
    emit(f"  {nm}: drop largest |outlier| ({arr[j]:+.2f}) -> mean {arr[keep].mean():+.3f}σ")
# n_s-edge confound: correlation of |bias| with distance from box centre
edged = np.abs(allns_u - 0.5)
emit(f"  corr(|A_p bias|, n_s box-edge proximity) = {np.corrcoef(np.abs(allAp), edged)[0,1]:+.3f}  "
     f"(near 0 => NOT a box-edge artifact)")
emit(f"  n_s_unit range covered: {allns_u.min():.3f}..{allns_u.max():.3f} (was 0.013..0.117 for fold-0 only)")

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
for col, (nm, arr) in enumerate((("A_p", allAp), ("n_s", allNs))):
    ax[col].axhline(0.2, color="r", ls="--", alpha=0.6, label="±0.2σ gate"); ax[col].axhline(-0.2, color="r", ls="--", alpha=0.6)
    ax[col].axhline(0.0, color="k", lw=0.6)
    ax[col].scatter(allns_u, arr, c="C0")
    m, lo, hi = boot(arr)
    ax[col].axhline(arr.mean(), color="C3", ls="-", alpha=0.7, label=f"mean {arr.mean():+.2f} [{lo:+.2f},{hi:+.2f}]")
    ax[col].set_xlabel("n_s (unit-cube)"); ax[col].set_ylabel(f"{nm} EMU bias [σ]")
    ax[col].set_title(f"{nm} EMU bias vs n_s — {len(arr)} honestly-held-out sims (all folds)")
    ax[col].legend(fontsize=8); ax[col].grid(alpha=0.3)
fig.suptitle("EMU bias re-measured over ALL folds (each fold's own held-out emulator), KS<0.06, bootstrap CI")
p = Path(FIGDIR) / "emu_bias_allfolds.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}"); emit("[done]")
