"""z-bin attribution walkthrough for the emulator's coherent n_s EMU bias (-0.65sigma).

FORWARD-ONLY (Fisher MAP-shift, no NUTS, no retraining). Built directly on top of
scripts/diag_emu_bias_allfolds.py: the physics in `emu_bias_one_sim` is COPIED verbatim
and only INSTRUMENTED with a per-row -> per-z grouping of the gradient, plus a C_emu toggle.

Attribution math (linear-in-rows decomposition of the Fisher MAP shift):
  total gradient   g  = sum_legs sum_rows Jl[row]^T * (Cinv @ dP)[row]
  whitened resid   w  = Cinv @ dP                          (length n_keep, per leg)
  per-row grad     grow[row,:] = Jl[row,:] * w[row]
  group rows by z  -> g_z[zbin,:]   (accumulated over BOTH legs on ctx.z_global)
                   -> also a per-(z,leg) split
  Cpost = (F + diag(P))^{-1}   (FULL info, all z, exactly as the headline script)
  bias_z = (Cpost @ g_z)[ns] / sqrt(Cpost[ns,ns])
  GATE: sum_z bias_z == total bias per sim (1e-6 rel); sum_z <bias_z> ~ -0.65 sigma.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_nsbias_z_attribution.py
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
RESULTS = f"{FIGDIR}/nsbias_z_attribution.txt"
NPZ = f"{FIGDIR}/nsbias_z_attribution.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KS_KMAX = 0.06   # the LOCKED KS cap

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])


def _per_sim_pieces(ctx, d, sim, fold):
    """Replicate emu_bias_one_sim EXACTLY, but RETURN the per-leg pieces (Jl, Cinv, dP, keep,
    z_leg, z_idx, k, leg.name) and the global F, Pp so the caller can both reproduce the total
    bias AND decompose g by z-bin. cemu_mode in {'prod','data_only','inflate2'} switches the
    C_total construction only (sigma_zb/rho_zb None for data_only, cemu_inflate=2 for inflate2).
    """
    t = make_truth_from_sim(d, sim, fold)
    th_truth = jnp.asarray(t["params_unit"])
    wc = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy(); wc[:, 2] = 0.0  # mask DLA
    z_sim = np.asarray(t["z"]); P_sim = np.asarray(t["P_obs_true"])
    P_filt = np.asarray(d["P_filt"]); rows = np.asarray(t["rows"]); K = P_filt.shape[-1]
    P_masked = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        a = wc[i]; coef = np.concatenate([[1.0 - a.sum()], a])
        P_masked[i] = np.einsum("c,ck->k", coef, np.stack([P_filt[r, 0], P_filt[r, 1], P_filt[r, 2], P_filt[r, 3]]))
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    zg = np.asarray(ctx.z_global); nZg = zg.size; nP = 9 + nZg + 3; AMP = np.arange(9 + nZg, 9 + nZg + 3)
    cache_k = np.asarray(ctx.cache_k)

    legs_out = []   # one dict per leg with the instrumented pieces
    return dict(th_truth=th_truth, t0_truth=t0_truth, core=core, wc=wc, z_sim=z_sim,
                P_masked=P_masked, zg=zg, nZg=nZg, nP=nP, AMP=AMP, cache_k=cache_k,
                params_unit=t["params_unit"])


def emu_bias_one_sim_attrib(ctx, d, sim, fold, cemu_mode="prod"):
    """COPY of emu_bias_one_sim, instrumented. Returns (bAp, bNs, ns_u, ap_u, attr) where attr
    holds the per-z (DESI, KS, total) n_s bias contributions on the global z grid, plus the
    coherent residual material (dP/P, w per row with z and leg tag)."""
    P = _per_sim_pieces(ctx, d, sim, fold)
    th_truth, t0_truth, core = P["th_truth"], P["t0_truth"], P["core"]
    wc, z_sim, P_masked = P["wc"], P["z_sim"], P["P_masked"]
    zg, nZg, nP, AMP, cache_k = P["zg"], P["nZg"], P["nP"], P["AMP"], P["cache_k"]

    F = np.zeros((nP, nP)); g = np.zeros(nP)
    g_z = np.zeros((nZg, nP))                       # accumulated per-z gradient, BOTH legs
    g_zleg = {"DESI": np.zeros((nZg, nP)), "KS": np.zeros((nZg, nP))}
    resid = {}                                       # per leg: dict(z, k, dPoverP, w, P0)

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
            Pm, _ = DL.predict_P_obs_on_leg(ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats,
                dla_core=core[_leg.name], cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
            return Pm[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2))(th_truth, t0_truth, apv0)
        Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2]], axis=1))
        P0 = np.asarray(fwd(th_truth, t0_truth, apv0)); Pfull = np.zeros(k.shape[0])
        for iz in range(leg.n_z):
            rr = np.where(z_idx == iz)[0]; j = int(np.argmin(np.abs(z_sim - float(z_leg[iz]))))
            Pfull[rr] = np.asarray(jnp.interp(jnp.asarray(k[rr]), jnp.asarray(cache_k),
                jnp.asarray(np.interp(cache_k, cache_k, P_masked[j]))))
        dP = Pfull[keep] - P0

        # --- C_total construction with the toggle (physics identical for cemu_mode='prod') ---
        if cemu_mode == "prod":
            szb_use, rzb_use, infl = szb, rzb, ctx.cemu_inflate
        elif cemu_mode == "data_only":
            szb_use, rzb_use, infl = None, None, ctx.cemu_inflate   # both None -> C_emu = 0
        elif cemu_mode == "inflate2":
            szb_use, rzb_use, infl = szb, rzb, 2.0 * ctx.cemu_inflate
        else:
            raise ValueError(cemu_mode)
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel], apv0[None, :] * jnp.asarray(gfix),
            pf_stats=ctx.pf_stats, dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb_use,
            alpha_centres=(None if (szb_use is None and rzb_use is None) else ctx.alpha_centres),
            cemu_inflate=infl, rho_zb=rzb_use)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl; g += Jl.T @ Cinv @ dP

        # --- INSTRUMENTATION: per-row whitened residual -> per-z gradient grouping ---
        w = Cinv @ dP                                # (n_keep,) whitened residual scalar/row
        grow = Jl * w[:, None]                        # (n_keep, nP) per-row gradient contribution
        # each kept row's z (via z_idx) -> nearest global z bin
        z_keep = z_leg[z_idx[keep]]
        zb_keep = np.array([int(np.argmin(np.abs(zg - zz))) for zz in z_keep])
        for ib in range(nZg):
            m = zb_keep == ib
            if m.any():
                contrib = grow[m].sum(0)
                g_z[ib] += contrib
                g_zleg[leg.name][ib] += contrib

        # coherent-residual material (kept rows): dP/P0 and w, with z and k
        resid[leg.name] = dict(z=z_keep, k=k[keep], dPoverP=dP / np.clip(P0, 1e-30, None),
                               w=w, P0=P0, dP=dP)

    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    dx = Cpost @ g
    bAp = dx[AP_I] / np.sqrt(Cpost[AP_I, AP_I])
    bNs = dx[NS_I] / np.sqrt(Cpost[NS_I, NS_I])

    # per-z n_s bias contribution (linear in g_z because dx = Cpost @ (sum_z g_z))
    sig_ns = np.sqrt(Cpost[NS_I, NS_I])
    bias_z = (g_z @ Cpost[:, NS_I]) / sig_ns            # (nZg,) ; (Cpost @ g_z)_ns = g_z . Cpost[:,ns]
    bias_z_leg = {nm: (g_zleg[nm] @ Cpost[:, NS_I]) / sig_ns for nm in g_zleg}

    # GATE: sum_z bias_z == total bias
    gate_err = abs(bias_z.sum() - bNs) / max(abs(bNs), 1e-12)
    assert gate_err < 1e-6, f"z-attribution GATE FAILED: sum_z bias_z={bias_z.sum():.6f} vs bNs={bNs:.6f} (rel {gate_err:.2e})"

    attr = dict(bias_z=bias_z, bias_z_leg=bias_z_leg, resid=resid)
    return (bAp, bNs, float(P["params_unit"][NS_I]), float(P["params_unit"][AP_I]), attr)


# ============================================================================ #
#  MAIN: all-folds loop, instrumented
# ============================================================================ #
emit("# n_s EMU-bias z-bin ATTRIBUTION. all 8 folds, each fold's own held-out emulator, KS<0.06.")
emit("# forward-only Fisher MAP-shift, instrumented from diag_emu_bias_allfolds.py. cemu_mode=prod.")

# build one ctx to get the global z grid + leg k grids (used for the residual heatmap)
ctx0, d0 = build_legb_ctx(ckpt=f"{REPO}/checkpoints/final_fold0", xclass_error_vector=HOLD0,
                          ks_kwargs={"k_max": KS_KMAX})
zg = np.asarray(ctx0.z_global); nZg = zg.size
desi_leg = [l for l in ctx0.legs if l.name == "DESI"][0]
ks_leg = [l for l in ctx0.legs if l.name == "KS"][0]
desi_k = np.unique(np.asarray(desi_leg.k))
emit(f"# z_global ({nZg}): " + " ".join(f"{z:.3f}" for z in zg))
emit(f"# legs: DESI(n_z={desi_leg.n_z}) KS(n_z={ks_leg.n_z}, k<{KS_KMAX})")

allNs, allAp, allns_u, allfold = [], [], [], []
bias_z_acc = np.zeros(nZg); bias_z_desi_acc = np.zeros(nZg); bias_z_ks_acc = np.zeros(nZg)
nsum = 0
# coherent residual accumulators on a common z grid per leg (sum + count)
res_acc = {"DESI": {}, "KS": {}}     # res_acc[leg][zb] = dict(sum_dPoP, n, k_ref) keyed by zbin
res_w_acc = {"DESI": {}, "KS": {}}

for fold in range(8):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    if fold == 0:
        ctx, d = ctx0, d0
    else:
        ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0, ks_kwargs={"k_max": KS_KMAX})
    sims, _ = held_out_sims(d, fold)
    for sim in sims:
        bAp, bNs, ns_u, ap_u, attr = emu_bias_one_sim_attrib(ctx, d, sim, fold, cemu_mode="prod")
        allNs.append(bNs); allAp.append(bAp); allns_u.append(ns_u); allfold.append(fold)
        bias_z_acc += attr["bias_z"]
        bias_z_desi_acc += attr["bias_z_leg"]["DESI"]
        bias_z_ks_acc += attr["bias_z_leg"]["KS"]
        nsum += 1
        # coherent residuals: bin each leg's rows to global z, accumulate mean dP/P and w on the leg k grid
        for nm in ("DESI", "KS"):
            r = attr["resid"][nm]
            zr = r["z"]; kr = r["k"]; dpop = r["dPoverP"]; wr = r["w"]
            zb = np.array([int(np.argmin(np.abs(zg - zz))) for zz in zr])
            for ib in np.unique(zb):
                m = zb == ib
                ks = kr[m]; order = np.argsort(ks)
                key = int(ib)
                if key not in res_acc[nm]:
                    res_acc[nm][key] = dict(sum=np.zeros(m.sum()), n=0, k=ks[order])
                    res_w_acc[nm][key] = np.zeros(m.sum())
                # only accumulate if the k-vector length matches the stored reference (same leg z-bin -> same k)
                if res_acc[nm][key]["sum"].shape[0] == m.sum():
                    res_acc[nm][key]["sum"] += dpop[m][order]
                    res_acc[nm][key]["n"] += 1
                    res_w_acc[nm][key] += wr[m][order]

allNs = np.array(allNs); allAp = np.array(allAp); allns_u = np.array(allns_u); allfold = np.array(allfold)
bias_z_mean = bias_z_acc / nsum
bias_z_desi_mean = bias_z_desi_acc / nsum
bias_z_ks_mean = bias_z_ks_acc / nsum

emit(f"\n# ===== ANCHOR: pooled n_s bias over {nsum} held-out sims =====")
emit(f"  n_s mean bias = {allNs.mean():+.4f} sigma   (headline emu_bias_allfolds.txt = -0.646)")
emit(f"  A_p mean bias = {allAp.mean():+.4f} sigma   (headline = +0.031)")
emit(f"  signs n_s: {(allNs>0).sum()}+/{(allNs<0).sum()}-")

# GATE: sum_z <bias_z> == <total>
gate = abs(bias_z_mean.sum() - allNs.mean())
emit(f"\n# GATE sum_z <bias_z> = {bias_z_mean.sum():+.4f}  vs  <total n_s bias> = {allNs.mean():+.4f}  (|diff|={gate:.2e})")
assert gate < 1e-6, "POOLED z-attribution gate failed"

emit(f"\n# ===== per-z <bias_z>(n_s) attribution (sigma), {nsum} sims =====")
emit(f"#   z      DESI       KS        total      %share(|.|)")
tot_abs = np.abs(bias_z_mean).sum()
for i in range(nZg):
    share = 100.0 * abs(bias_z_mean[i]) / max(tot_abs, 1e-30)
    emit(f"  {zg[i]:.3f}  {bias_z_desi_mean[i]:+.4f}  {bias_z_ks_mean[i]:+.4f}  {bias_z_mean[i]:+.4f}   {share:5.1f}%")
emit(f"  ----   --------   --------   --------")
emit(f"  TOTAL  {bias_z_desi_mean.sum():+.4f}  {bias_z_ks_mean.sum():+.4f}  {bias_z_mean.sum():+.4f}")
emit(f"  DESI total = {bias_z_desi_mean.sum():+.4f} sigma  ({100*bias_z_desi_mean.sum()/bias_z_mean.sum():.0f}% of total)")
emit(f"  KS   total = {bias_z_ks_mean.sum():+.4f} sigma  ({100*bias_z_ks_mean.sum()/bias_z_mean.sum():.0f}% of total)")

# top contributing z-bins
order = np.argsort(np.abs(bias_z_mean))[::-1]
emit(f"\n# top contributing z-bins (by |contribution|):")
for i in order[:6]:
    emit(f"  z={zg[i]:.3f}: {bias_z_mean[i]:+.4f} sigma  ({100*abs(bias_z_mean[i])/tot_abs:.1f}% share)")

# prior-claim check: ~53% at z 2.0-2.3, ~25% at z 3.5-3.7
m_low = (zg >= 1.95) & (zg <= 2.35)
m_hi = (zg >= 3.45) & (zg <= 3.75)
sh_low = 100 * bias_z_mean[m_low].sum() / bias_z_mean.sum()
sh_hi = 100 * bias_z_mean[m_hi].sum() / bias_z_mean.sum()
emit(f"\n# PRIOR-CLAIM CHECK (signed share of total -0.65sigma):")
emit(f"  z in [2.0,2.3]: contribution {bias_z_mean[m_low].sum():+.4f} sigma = {sh_low:.0f}% of total  (prior claim ~53%)")
emit(f"  z in [3.5,3.7]: contribution {bias_z_mean[m_hi].sum():+.4f} sigma = {sh_hi:.0f}% of total  (prior claim ~25%)")

# cumulative
cum = np.cumsum(bias_z_mean)
emit(f"\n# cumulative sum_z'<=z <bias_z'>(n_s):")
for i in range(nZg):
    emit(f"  z<={zg[i]:.3f}: {cum[i]:+.4f} sigma")

# ============================================================================ #
#  C_emu TOGGLE on pooled n_s bias (prod / data-only / inflate x2)
# ============================================================================ #
emit(f"\n# ===== C_emu TOGGLE on pooled n_s bias =====")
toggle = {"prod": allNs.mean()}     # prod already computed
for mode in ("data_only", "inflate2"):
    vals = []
    for fold in range(8):
        ckpt = f"{REPO}/checkpoints/final_fold{fold}"
        ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0, ks_kwargs={"k_max": KS_KMAX})
        sims, _ = held_out_sims(d, fold)
        for sim in sims:
            _, bNs, _, _, _ = emu_bias_one_sim_attrib(ctx, d, sim, fold, cemu_mode=mode)
            vals.append(bNs)
    toggle[mode] = float(np.mean(vals))
    emit(f"  {mode:11s}: pooled n_s bias = {toggle[mode]:+.4f} sigma")
emit(f"  prod       : pooled n_s bias = {toggle['prod']:+.4f} sigma  (C_total = C_data + C_emu, prod)")
emit(f"  data_only  : C_emu=0 (cosmic-only)  -> {toggle['data_only']:+.4f} sigma")
emit(f"  inflate2   : C_emu x2               -> {toggle['inflate2']:+.4f} sigma")

# ============================================================================ #
#  FOLD STRUCTURE: per-fold mean bias vs (fold_mean_ns - global_mean_ns)
# ============================================================================ #
emit(f"\n# ===== FOLD STRUCTURE: LOSO folds are contiguous sorted-ns blocks =====")
global_ns = allns_u.mean()
fold_means_bias, fold_means_ns = [], []
emit(f"#  fold  n  mean_ns_u  (mean_ns-global)  mean_bias(n_s)")
for f in range(8):
    m = allfold == f
    fmb = allNs[m].mean(); fmn = allns_u[m].mean()
    fold_means_bias.append(fmb); fold_means_ns.append(fmn)
    emit(f"   {f}    {m.sum():2d}   {fmn:.3f}      {fmn-global_ns:+.3f}         {fmb:+.4f}")
fold_means_bias = np.array(fold_means_bias); fold_means_ns = np.array(fold_means_ns)
r_fold = np.corrcoef(fold_means_ns - global_ns, fold_means_bias)[0, 1]
emit(f"  global mean ns_u = {global_ns:.3f}")
emit(f"  corr( fold_mean_ns - global,  fold_mean_bias ) = {r_fold:+.3f}")
# per-sim linear fit bias vs ns_unit
A = np.vstack([allns_u, np.ones_like(allns_u)]).T
slope, icpt = np.linalg.lstsq(A, allNs, rcond=None)[0]
r_sim = np.corrcoef(allns_u, allNs)[0, 1]
emit(f"  per-sim fit: bias = {slope:+.3f}*ns_u + {icpt:+.3f}   corr(ns_u,bias)={r_sim:+.3f}")

# save npz
np.savez(NPZ, z_global=zg, bias_z_mean=bias_z_mean, bias_z_desi_mean=bias_z_desi_mean,
         bias_z_ks_mean=bias_z_ks_mean, cum=cum, allNs=allNs, allAp=allAp, allns_u=allns_u,
         allfold=allfold, fold_means_bias=fold_means_bias, fold_means_ns=fold_means_ns,
         global_ns=global_ns, toggle_prod=toggle["prod"], toggle_data_only=toggle["data_only"],
         toggle_inflate2=toggle["inflate2"], nsum=nsum)

# coherent residual arrays -> build (z,k) heatmap material on the DESI leg
def stack_resid(acc, wacc):
    """returns (zbins sorted, list of (k, mean_dPoP, mean_w)) per present z bin."""
    out = []
    for ib in sorted(acc.keys()):
        n = acc[ib]["n"]
        if n == 0: continue
        out.append((zg[ib], acc[ib]["k"], acc[ib]["sum"] / n, wacc[ib] / n))
    return out
desi_resid = stack_resid(res_acc["DESI"], res_w_acc["DESI"])
ks_resid = stack_resid(res_acc["KS"], res_w_acc["KS"])

# ============================================================================ #
#  FIGURES
# ============================================================================ #
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# FIG 1: z attribution bar chart (DESI vs KS grouped) + |share| twin
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(nZg); wbar = 0.38
ax.bar(x - wbar/2, bias_z_desi_mean, wbar, label="DESI", color="C0")
ax.bar(x + wbar/2, bias_z_ks_mean, wbar, label="KS (k<0.06)", color="C1")
ax.plot(x, bias_z_mean, "k.-", lw=1.4, ms=8, label="total / z-bin")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{z:.2f}" for z in zg], rotation=45, fontsize=8)
ax.set_xlabel("redshift z"); ax.set_ylabel(r"$\langle$bias$_z\rangle$(n$_s$) [$\sigma$]")
ax.set_title(f"n_s EMU-bias z-attribution ({nsum} held-out sims, all folds)\n"
             f"total = {bias_z_mean.sum():+.3f}sigma  (DESI {bias_z_desi_mean.sum():+.2f}, KS {bias_z_ks_mean.sum():+.2f})")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
ax2 = ax.twinx()
share = 100.0 * np.abs(bias_z_mean) / max(tot_abs, 1e-30)
ax2.plot(x, share, "C3o:", alpha=0.5, label="|contribution| share %")
ax2.set_ylabel("|contribution| share [%]", color="C3"); ax2.tick_params(axis='y', labelcolor="C3")
ax.annotate(f"total {bias_z_mean.sum():+.3f}sigma", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
p1 = Path(FIGDIR) / "nsbias_z_attribution.png"; fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

# FIG 2: cumulative
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(zg, cum, "C0o-", lw=2, label=r"$\sum_{z'\leq z}\langle$bias$_{z'}\rangle$")
ax.axhline(bias_z_mean.sum(), color="C3", ls="-", alpha=0.8, label=f"total {bias_z_mean.sum():+.3f}sigma")
ax.axhline(-0.2, color="r", ls="--", alpha=0.6, label="+/-0.2sigma gate"); ax.axhline(0.2, color="r", ls="--", alpha=0.6)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("redshift z"); ax.set_ylabel(r"cumulative $\langle$bias$\rangle$(n$_s$) [$\sigma$]")
ax.set_title(f"Cumulative build-up of the n_s EMU bias across z ({nsum} sims)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
p2 = Path(FIGDIR) / "nsbias_cumulative_z.png"; fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)

# FIG 3: coherent residual (z,k) on DESI leg -- heatmap + line cuts
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
# heatmap: interp each z's mean dP/P onto a common DESI k grid
kgrid = desi_k
H = np.full((len(desi_resid), len(kgrid)), np.nan)
zvals = []
for i, (z, k, dpop, w) in enumerate(desi_resid):
    zvals.append(z)
    o = np.argsort(k)
    H[i] = np.interp(kgrid, k[o], dpop[o], left=np.nan, right=np.nan)
im = axs[0].pcolormesh(kgrid, np.array(zvals), H, shading="nearest", cmap="RdBu_r",
                       vmin=-np.nanmax(np.abs(H)), vmax=np.nanmax(np.abs(H)))
axs[0].set_xlabel("k [s/km] (DESI)"); axs[0].set_ylabel("z")
axs[0].set_title("coherent mean residual <dP/P_forward>  (DESI, held-out)")
fig.colorbar(im, ax=axs[0], label="<dP/P>")
# line cuts: a few z
for (z, k, dpop, w) in desi_resid:
    o = np.argsort(k)
    axs[1].plot(k[o], dpop[o], ".-", label=f"z={z:.2f}", alpha=0.8)
axs[1].axhline(0, color="k", lw=0.6)
axs[1].set_xlabel("k [s/km] (DESI)"); axs[1].set_ylabel("<dP/P_forward>")
axs[1].set_title("per-z line cuts (low-k DESI = large C_emu/C_data)")
axs[1].legend(fontsize=7, ncol=2); axs[1].grid(alpha=0.3)
p3 = Path(FIGDIR) / "nsbias_coherent_residual_zk.png"; fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)

# FIG 4: fold structure -- per-sim bias vs ns_unit colored by fold
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(allns_u, allNs, c=allfold, cmap="viridis", s=45, edgecolor="k", lw=0.3)
for f in range(8):
    m = allfold == f
    ax.plot(allns_u[m].mean(), allNs[m].mean(), "rs", ms=12, mfc="none", mew=2)
    ax.annotate(f"f{f}", (allns_u[m].mean(), allNs[m].mean()), fontsize=8, color="r")
xs = np.linspace(allns_u.min(), allns_u.max(), 50)
ax.plot(xs, slope * xs + icpt, "C3--", lw=1.6, label=f"fit bias={slope:+.2f}*ns_u{icpt:+.2f} (r={r_sim:+.2f})")
ax.axhline(allNs.mean(), color="k", ls="-", alpha=0.7, label=f"global mean {allNs.mean():+.3f}sigma")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("n_s (unit-cube)"); ax.set_ylabel("per-sim total n_s EMU bias [sigma]")
ax.set_title(f"Fold structure: LOSO folds are contiguous sorted-ns blocks\n"
             f"corr(fold_mean_ns-global, fold_mean_bias) = {r_fold:+.2f}")
fig.colorbar(sc, ax=ax, label="fold"); ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
p4 = Path(FIGDIR) / "nsbias_fold_structure.png"; fig.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(fig)

# FIG 5: C_emu toggle
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["prod\n(C_data+C_emu)", "data-only\n(C_emu=0)", "inflate\n(C_emu x2)"]
vals = [toggle["prod"], toggle["data_only"], toggle["inflate2"]]
bars = ax.bar(labels, vals, color=["C0", "C2", "C1"])
ax.axhline(0, color="k", lw=0.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width()/2, v), ha="center",
                va="bottom" if v > 0 else "top", fontsize=10)
ax.set_ylabel("pooled n_s EMU bias [sigma]")
ax.set_title(f"C_emu toggle on the pooled n_s bias ({nsum} sims)")
ax.grid(alpha=0.3, axis="y")
p5 = Path(FIGDIR) / "nsbias_cemu_toggle.png"; fig.savefig(p5, dpi=150, bbox_inches="tight"); plt.close(fig)

emit(f"\n[fig] {p1}")
emit(f"[fig] {p2}")
emit(f"[fig] {p3}")
emit(f"[fig] {p4}")
emit(f"[fig] {p5}")
emit(f"[npz] {NPZ}")
emit("[done]")
