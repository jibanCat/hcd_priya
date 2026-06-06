"""Is the +0.5σ low-k A_p EMU bias an emulator-FIDELITY problem — and specifically high-k
under-resolution leaking to low-k through the GLOBAL SVD basis? (PI Q1) And is it correlated with
MF-not-implemented? (PI Q2). Plus: reconcile the +0.5σ (real-leg ruler) vs the Phase-2 0.067σ.

Forward-only, held-out fold-0 sims. Four parts → figures/analysis/04_emulator/emu_lowk_investigation.png
+ legb... text. Reuses build_legb_ctx (production model + real legs), predict_P_filt (emulator),
d["P_filt"] (cache-measured truth), and model.head_b.p_filt_basis (the (n_basis,n_k) global SVD basis).

KEY MECHANISM under test: the output basis is ONE shared (n_basis, n_k) matrix over the FULL cache
k-grid (model.py:135). 121 of 172 k-bins are high-k (k>0.02); only 12 are low-k (k<5e-3, the A_p
regime). A basis fit to minimize FULL-range error is high-k-dominated → low-k may be under-served, and
a high-k mode with low-k support can leak. Every prior fidelity metric (sub-%, 0.067σ, grad 7e-8) was
FULL-k POOLED — this script resolves by band.

PART A: per-k-band per-class reconstruction error |P_emu/P_true−1| (coherent mean + scatter), 8 sims.
PART B: A_p/n_s bias ATTRIBUTION by k-band — recompute (F+P)^{-1} Jᵀ C^{-1} ΔP_emu with ΔP_emu masked
        to each leg-k band. Where is the +0.5σ SOURCED?
PART C: SVD rank-n_basis TRUNCATION FLOOR vs emulator error, per band — the decisive test:
        emu_lowk ≈ floor → rank/architecture-limited (Q1 leakage/under-served low-k);
        emu_lowk ≫ floor → training-limited.
PART D: leakage probe — does the per-sim low-k residual CORRELATE with the high-k residual across sims
        (shared-basis coupling)? + the structural MF (Q2) argument by residual k-location.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_lowk_investigation.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg, make_splits,
)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.predict import predict_P_filt
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT

FIGDIR = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/emu_lowk_investigation.txt"
HOLD0 = "/home/mfho/hcd_priya/checkpoints/error_vector_xclass_holdout0.npz"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]
BANDS = [("low  k<5e-3", 0.0, 5e-3), ("mid  5e-3-0.02", 5e-3, 0.02), ("high 0.02-0.069", 0.02, 1.0)]

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()

NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])

ctx, d = build_legb_ctx(xclass_error_vector=HOLD0)
sims0, _ = held_out_sims(d, 0)
cache_k = np.asarray(ctx.cache_k)                       # (K,)
pf = ctx.pf_stats
mu_marg = np.asarray(pf["mu_marg"]); sig_marg = np.asarray(pf["sig_marg"])   # (4,K)
basis = np.asarray(ctx.model.head_b.p_filt_basis)       # (n_basis, K) shared global SVD basis
nb = basis.shape[0]
# the basis is a TRAINED parameter (not static) → rows are NOT exactly orthonormal after training,
# so the span-projector is the proper oblique form Bᵀ(BBᵀ)⁻¹B, not BᵀB.
_BBt_inv = np.linalg.inv(basis @ basis.T + 1e-12 * np.eye(nb))
Vproj = basis.T @ _BBt_inv @ basis                       # (K,K) projector onto the basis span
emit(f"# basis orthonormality check: ||BBᵀ−I||_max = {np.abs(basis @ basis.T - np.eye(nb)).max():.3e}")
band_idx = [np.where((cache_k >= lo) & (cache_k < hi))[0] for _, lo, hi in BANDS]
emit(f"# EMU low-k investigation  n_basis={nb}  K={len(cache_k)}  bins/band="
     f"{[len(b) for b in band_idx]} (low/mid/high)  classes={CLASSES}")
emit(f"# basis is GLOBAL over all k (model.py:135); 121/172 bins are high-k -> low-k under-served test")

# irreducible rank-nb floor: a FRESH SVD of the standardized-log TRUTH ensemble (all in-range cache
# rows), per class. The trained basis can do no better than this rank-nb truncation; comparing the
# emulator error to it separates RANK-limited (emu≈floor, irreducible → carry as C_emu / train HR)
# from TRAINING-limited (emu≫floor → retrainable). std-log space zf=(logP−mu_marg)/sig_marg.
P_filt = np.asarray(d["P_filt"])                         # (R,4,K) cache-measured truth
inrange = (np.asarray(d["z_grid"]) >= 2.2) & (np.asarray(d["z_grid"]) <= 4.6)
Vtrunc = {}
for ci in range(4):
    Z = (np.log(np.clip(P_filt[inrange, ci], 1e-30, None)) - mu_marg[ci]) / sig_marg[ci]  # (Ninr,K)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    Vk = Vt[:nb]                                          # (nb,K) top-nb right singular vecs of TRUTH
    Vtrunc[ci] = Vk.T @ Vk                                # orthonormal rank-nb projector

emit("\n# ===== PART A/C: per-k-band emulator error vs the IRREDUCIBLE rank-%d TRUTH-SVD floor =====" % nb)
emit("# std-log units; coherent = signed mean over sims (the bias-relevant part); emu/floor: ≈1 rank-limited, ≫1 trainable")
recon_err = {c: {b: [] for b in range(3)} for c in CLASSES}
floor = {c: {b: [] for b in range(3)} for c in CLASSES}
coh_err = {c: {b: [] for b in range(3)} for c in CLASSES}
for sim in sims0:
    t = make_truth_from_sim(d, sim, 0)
    rows = np.asarray(t["rows"]); z = np.asarray(t["z"])
    ipz = int(np.argmin(np.abs(z - HCD_Z_PIVOT))); r = int(rows[ipz])
    th = jnp.asarray(t["params_unit"]); z_unit = float(d["x"][r, 9]); tau0 = float(d["tau0"][r])
    P_emu = np.asarray(predict_P_filt(ctx.model, th, z_unit, tau0, pf))     # (4,K) linear
    P_tru = P_filt[r]
    zf_tru = (np.log(np.clip(P_tru, 1e-30, None)) - mu_marg) / sig_marg
    zf_emu = (np.log(np.clip(P_emu, 1e-30, None)) - mu_marg) / sig_marg
    for ci, c in enumerate(CLASSES):
        e = zf_emu[ci] - zf_tru[ci]                                # emulator error (std-log)
        fl = zf_tru[ci] - zf_tru[ci] @ Vtrunc[ci]                  # irreducible rank-nb truth floor
        for b in range(3):
            bi = band_idx[b]
            recon_err[c][b].append(np.sqrt(np.mean(e[bi] ** 2)))
            floor[c][b].append(np.sqrt(np.mean(fl[bi] ** 2)))
            coh_err[c][b].append(np.mean(e[bi]))
emit("# class    band            emu_err(RMS)   truth-floor(RMS)   emu/floor   coherent_mean±scatter")
for c in CLASSES:
    for b in range(3):
        er = np.array(recon_err[c][b]); fr = np.array(floor[c][b]); co = np.array(coh_err[c][b])
        ratio = er.mean() / max(fr.mean(), 1e-30)
        emit(f"  {c:7s} {BANDS[b][0]:15s}  {er.mean():.4f}        {fr.mean():.4f}          "
             f"{ratio:6.2f}x     {co.mean():+.4f} ± {co.std():.4f}")

# ---------------------------------------------------------------- PART B: A_p/n_s bias band-attribution
emit("\n# ===== PART B: where in k is the A_p/n_s EMU bias SOURCED? (mask ΔP_emu by leg-k band) =====")
zg = np.asarray(ctx.z_global); nZg = zg.size
nP = 9 + nZg + 3
AMP = np.arange(9 + nZg, 9 + nZg + 3)
def lit_pivot_apivot():   # use the sim w_c pivot as the amplitude operating point (forward-only)
    return None
band_bias = {b: {"Ap": [], "Ns": []} for b in range(3)}
full_bias = {"Ap": [], "Ns": []}
for sim in sims0:
    t = make_truth_from_sim(d, sim, 0)
    th_truth = jnp.asarray(t["params_unit"])
    wc_perz = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy()   # (n_zsim,3)
    z_sim = np.asarray(t["z"]); P_sim = np.asarray(t["P_obs_true"])
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    F = np.zeros((nP, nP)); gband = {b: np.zeros(nP) for b in range(3)}; gfull = np.zeros(nP)
    for leg in ctx.legs:
        z_leg = np.asarray(leg.z); k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in z_leg])
        sel_sim = np.array([int(np.argmin(np.abs(z_sim - zz))) for zz in z_leg])
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        gfix_leg = jnp.asarray(wc_perz[sel_sim] / np.clip(wc_perz[sel_sim][
            np.argmin(np.abs(z_sim[sel_sim] - HCD_Z_PIVOT))], 1e-12, None))   # (n_z,3) shape, pivot-normed
        apv0 = jnp.asarray(wc_perz[sel_sim][np.argmin(np.abs(z_sim[sel_sim] - HCD_Z_PIVOT))])  # (3,) pivot
        # α(z) = a_pivot[None,:] * g_fixed(z); differentiate the 3-vector a_pivot (clean α block).
        def fwd(th, t0g, apv, _leg=leg, _sel=jnp.asarray(sel), _keep=jnp.asarray(keep), _g=gfix_leg):
            al = apv[None, :] * _g                                # (n_z,3) sim exact shape, amp=a_pivot
            P, _ = DL.predict_P_obs_on_leg(ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats,
                dla_core=core[_leg.name], cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
            return P[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2))(th_truth, t0_truth, apv0)
        Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2]], axis=1))   # (Nkeep, 9+nZg+3) = nP exactly
        P0 = np.asarray(fwd(th_truth, t0_truth, apv0))
        Pfull = np.zeros(k.shape[0])
        for iz in range(leg.n_z):
            rr = np.where(z_idx == iz)[0]; j = int(np.argmin(np.abs(z_sim - float(z_leg[iz]))))
            Pfull[rr] = np.asarray(jnp.interp(jnp.asarray(k[rr]), jnp.asarray(cache_k), jnp.asarray(P_sim[j])))
        dP = Pfull[keep] - P0
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel], apv0[None, :] * gfix_leg,
            pf_stats=ctx.pf_stats, dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl; gfull += Jl.T @ Cinv @ dP
        kk = k[keep]
        for b, (_, lo, hi) in enumerate(BANDS):
            m = ((kk >= lo) & (kk < hi)).astype(float)
            gband[b] += Jl.T @ Cinv @ (dP * m)
    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    sAp = np.sqrt(Cpost[AP_I, AP_I]); sNs = np.sqrt(Cpost[NS_I, NS_I])
    full_bias["Ap"].append((Cpost @ gfull)[AP_I] / sAp); full_bias["Ns"].append((Cpost @ gfull)[NS_I] / sNs)
    for b in range(3):
        band_bias[b]["Ap"].append((Cpost @ gband[b])[AP_I] / sAp)
        band_bias[b]["Ns"].append((Cpost @ gband[b])[NS_I] / sNs)
emit(f"# (bias in σ; EMU baseline = forward at sim's EXACT shape, so this is PURE emulator error)")
emit(f"#   FULL (all k):  A_p mean {np.mean(full_bias['Ap']):+.3f}  n_s mean {np.mean(full_bias['Ns']):+.3f}")
for b in range(3):
    emit(f"#   {BANDS[b][0]:15s}: A_p mean {np.mean(band_bias[b]['Ap']):+.3f} "
         f"(scatter {np.std(band_bias[b]['Ap']):.3f})   n_s mean {np.mean(band_bias[b]['Ns']):+.3f}")

# ---------------------------------------------------------------- PART D: leakage correlation + MF
emit("\n# ===== PART D: leakage probe (low-k vs high-k residual correlation across sims) + MF (Q2) =====")
lowk_co = np.array([np.mean([coh_err[c][0][i] for c in ("clean", "LLS")]) for i in range(len(sims0))])
highk_co = np.array([np.mean([coh_err[c][2][i] for c in ("clean", "LLS")]) for i in range(len(sims0))])
if lowk_co.std() > 0 and highk_co.std() > 0:
    rr = np.corrcoef(lowk_co, highk_co)[0, 1]
else:
    rr = np.nan
emit(f"#   corr(low-k coherent err, high-k coherent err) across {len(sims0)} sims = {rr:+.3f}")
emit(f"#   (strong correlation => the global basis COUPLES the bands = the leakage channel; ~0 => independent)")
emit(f"#   Q2/MF: the closure is LF-cache-truth vs LF-emulated; MF (LF→HF correction) is in NEITHER side")
emit(f"#   => MF cannot CAUSE this residual (structural). But MF acts at high-k (k>0.02); if PART B shows")
emit(f"#   the A_p bias is sourced from high-k leaking to low-k, MF wiring WILL perturb the same channel")
emit(f"#   => re-measure after MF. If A_p bias is sourced from low-k bins directly, MF is irrelevant to it.")

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
# panel 1: emu err vs floor per band (LLS = the binding class)
c = "LLS"; x = np.arange(3)
er = [np.mean(recon_err[c][b]) for b in range(3)]; fr = [np.mean(floor[c][b]) for b in range(3)]
ax[0].bar(x - 0.2, er, 0.4, label="emulator error (RMS, std-log)", color="C3")
ax[0].bar(x + 0.2, fr, 0.4, label=f"rank-{nb} truncation floor", color="C0")
ax[0].set_xticks(x); ax[0].set_xticklabels([b[0] for b in BANDS], fontsize=8)
ax[0].set_ylabel("std-log reconstruction error"); ax[0].set_title(f"PART C: {c} — error vs SVD rank floor\n(emu≈floor → rank-limited; emu≫floor → training)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
# panel 2: A_p bias band-attribution
xb = np.arange(3)
mAp = [np.mean(band_bias[b]["Ap"]) for b in range(3)]; sApb = [np.std(band_bias[b]["Ap"]) for b in range(3)]
ax[1].bar(xb, mAp, yerr=sApb, capsize=4, color=["C0", "C1", "C2"])
ax[1].axhline(np.mean(full_bias["Ap"]), color="C3", ls="--", label=f"FULL {np.mean(full_bias['Ap']):+.2f}σ")
ax[1].set_xticks(xb); ax[1].set_xticklabels([b[0] for b in BANDS], fontsize=8)
ax[1].set_ylabel("A_p bias contribution [σ]"); ax[1].set_title("PART B: where the A_p EMU bias is SOURCED")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
# panel 3: leakage scatter
ax[2].scatter(lowk_co, highk_co, c="C4")
ax[2].axhline(0, color="k", lw=0.5); ax[2].axvline(0, color="k", lw=0.5)
ax[2].set_xlabel("low-k coherent err (clean+LLS)"); ax[2].set_ylabel("high-k coherent err")
ax[2].set_title(f"PART D: band leakage probe  r={rr:+.2f}\n(corr → global basis couples bands)")
ax[2].grid(alpha=0.3)
fig.suptitle(f"EMU low-k investigation (Q1: high-k→low-k leakage via global SVD basis; Q2: MF) — "
             f"8 fold-0 held-out sims, n_basis={nb}")
p = Path(FIGDIR) / "emu_lowk_investigation.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}")
emit("[done]")
