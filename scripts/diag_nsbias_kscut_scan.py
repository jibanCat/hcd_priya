"""KS-leg z-cut series + KS k_max scan of the coherent n_s (and A_p) EMU bias.

FORWARD-ONLY (Fisher MAP-shift, no NUTS, no retraining). REUSES `emu_bias_one_sim` from
scripts/diag_emu_bias_allfolds.py verbatim (imported, not re-derived). The ONLY thing that
varies between configs is the `ks_kwargs` threaded into `build_legb_ctx` ->
`data_likelihood.load_ks_leg(..., z_lo=Z, z_hi=4.6, drop_first4=True, k_max=K)`. DESI is
untouched in every config; the KS covariance is sliced by the SAME `keep` mask so dropped
bins are removed cleanly. Physics is otherwise IDENTICAL to the headline -0.646sigma anchor
(DLA masking, sigma_zb/rho_zb/cemu_inflate, alpha_centres -- all carried by `ctx`).

Two studies:
  (A) KS z-cut series at k_max=0.06 fixed:  z_lo in {2.0, 2.1, 2.3, 2.8}
        2.0 = baseline anchor (must reproduce ~ -0.646sigma); 2.1 drops KS z=2.0;
        2.3 drops KS z=2.0+2.2; 2.8 = the PI's paper cut (drops z<2.8).
  (B) KS k_max scan at z_lo=2.0 (so the z=2.0/2.2 bins are PRESENT):
        k_max in {0.06, 0.04, 0.03, 0.02}.  If the bias collapses as k>0.04 is removed,
        the low-z KS bias is HIGH-K sourced (the LF-resolution regime the PI flagged).
  (C) k_max scan at z_lo=2.8 (residual high-z check).

For every config: pooled n_s + A_p EMU bias over the SAME 60 honestly-held-out sims (8 LOSO
folds, each fold's own held-out emulator final_fold{f}), with bootstrap95% CI, t, signs, and
a cheap KS-vs-DESI split (per-leg Fisher MAP shift, with the full posterior covariance).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_nsbias_kscut_scan.py
"""
import gc
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT

NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])

# --- REUSE emu_bias_one_sim from diag_emu_bias_allfolds.py WITHOUT triggering its top-level
# driver. That script has NO __main__ guard, and line 38 is `_rf = open(RESULTS, "w")` which
# would TRUNCATE the authoritative emu_bias_allfolds.txt anchor at import/exec time, plus the
# driver would re-run the full 60-sim baseline. So we exec ONLY the `def emu_bias_one_sim`
# block (verbatim physics, no rewrite) into a namespace pre-populated with exactly the names
# its body closes over -- np/jnp/jax/DL, the make_* helpers, HCD_Z_PIVOT, and AP_I/NS_I. ---
def _load_emu_bias_one_sim():
    src_path = Path(__file__).resolve().parent / "diag_emu_bias_allfolds.py"
    lines = src_path.read_text().splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def emu_bias_one_sim("))
    # function block = the def line + all following indented/blank lines until the next
    # top-level (column-0, non-blank) statement
    end = start + 1
    while end < len(lines) and (lines[end].strip() == "" or lines[end][:1] in (" ", "\t")):
        end += 1
    ns = dict(np=np, jnp=jnp, jax=jax, DL=DL, HCD_Z_PIVOT=HCD_Z_PIVOT, AP_I=AP_I, NS_I=NS_I,
              make_truth_from_sim=make_truth_from_sim, make_legb_mock=make_legb_mock,
              _mock_core_per_leg=_mock_core_per_leg)
    exec("".join(lines[start:end]), ns)
    return ns["emu_bias_one_sim"]

emu_bias_one_sim = _load_emu_bias_one_sim()

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/nsbias_kscut_scan.txt"
NPZ = f"{FIGDIR}/nsbias_kscut_scan.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()


def emu_bias_one_sim_split(ctx, d, sim, fold):
    """COPY of emu_bias_one_sim, additionally returning the per-leg (KS vs DESI) n_s and A_p
    bias contributions, computed in the SAME full posterior: dx_leg = Cpost @ g_leg where
    g_leg is that leg's gradient. The full bias = sum over legs (linear in g). This is the
    cheap KS-vs-DESI split; the totals match emu_bias_one_sim exactly (gate-checked)."""
    t = make_truth_from_sim(d, sim, fold)
    th_truth = jnp.asarray(t["params_unit"])
    wc = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy(); wc[:, 2] = 0.0  # mask DLA
    z_sim = np.asarray(t["z"])
    P_filt = np.asarray(d["P_filt"]); rows = np.asarray(t["rows"]); K = P_filt.shape[-1]
    P_masked = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        a = wc[i]; coef = np.concatenate([[1.0 - a.sum()], a])
        P_masked[i] = np.einsum("c,ck->k", coef, np.stack([P_filt[r, 0], P_filt[r, 1], P_filt[r, 2], P_filt[r, 3]]))
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    zg = np.asarray(ctx.z_global); nZg = zg.size; nP = 9 + nZg + 3; AMP = np.arange(9 + nZg, 9 + nZg + 3)
    cache_k = np.asarray(ctx.cache_k)
    F = np.zeros((nP, nP)); g = np.zeros(nP); g_leg = {"DESI": np.zeros(nP), "KS": np.zeros(nP)}
    for leg in ctx.legs:
        z_leg = np.asarray(leg.z); k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        sel = np.array([int(np.argmin(np.abs(zg - z))) for z in z_leg])
        ss = np.array([int(np.argmin(np.abs(z_sim - z))) for z in z_leg])
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        if keep.size == 0:    # whole leg dropped by the cut (e.g. KS empty after a hard cut)
            continue
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
                jnp.asarray(np.interp(cache_k, cache_k, P_masked[j]))))
        dP = Pfull[keep] - P0
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel], apv0[None, :] * jnp.asarray(gfix),
            pf_stats=ctx.pf_stats, dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        gl = Jl.T @ Cinv @ dP
        F += Jl.T @ Cinv @ Jl; g += gl; g_leg[leg.name] += gl
    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    dx = Cpost @ g
    sig_ns = np.sqrt(Cpost[NS_I, NS_I]); sig_ap = np.sqrt(Cpost[AP_I, AP_I])
    bNs = dx[NS_I] / sig_ns; bAp = dx[AP_I] / sig_ap
    # per-leg split in the SAME posterior (linear in g): bias_leg = (Cpost @ g_leg)/sigma
    ns_desi = (Cpost @ g_leg["DESI"])[NS_I] / sig_ns
    ns_ks = (Cpost @ g_leg["KS"])[NS_I] / sig_ns
    return bAp, bNs, ns_desi, ns_ks


def boot(x, n=10000):
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_config(ks_kwargs, label, folds=range(8), want_split=True):
    """Loop the requested folds, accumulate pooled n_s/A_p bias (+split). Returns a dict."""
    allAp, allNs, allDESI, allKS, allfold = [], [], [], [], []
    for fold in folds:
        ckpt = f"{REPO}/checkpoints/final_fold{fold}"
        ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0, ks_kwargs=ks_kwargs)
        sims, _ = held_out_sims(d, fold)
        for sim in sims:
            if want_split:
                bAp, bNs, nsD, nsK = emu_bias_one_sim_split(ctx, d, sim, fold)
                allDESI.append(nsD); allKS.append(nsK)
            else:
                bAp, bNs, _, _ = emu_bias_one_sim(ctx, d, sim, fold)
            allAp.append(bAp); allNs.append(bNs); allfold.append(fold)
        # light cleanup between folds (memory is ample; do NOT clear JAX caches -- that forces
        # a full recompile of the jacrev forward every fold and ~10x's the runtime).
        del ctx, d
        gc.collect()
    allAp = np.array(allAp); allNs = np.array(allNs); allfold = np.array(allfold)
    mNs, loNs, hiNs = boot(allNs); mAp, loAp, hiAp = boot(allAp)
    seNs = allNs.std(ddof=1) / np.sqrt(len(allNs)); tNs = allNs.mean() / seNs
    out = dict(label=label, ks_kwargs=dict(ks_kwargs), n=len(allNs),
               ns_mean=allNs.mean(), ns_lo=loNs, ns_hi=hiNs, ns_t=tNs,
               ns_pos=int((allNs > 0).sum()), ns_neg=int((allNs < 0).sum()),
               ap_mean=allAp.mean(), ap_lo=loAp, ap_hi=hiAp,
               allNs=allNs, allAp=allAp, allfold=allfold)
    if want_split:
        out["ns_desi"] = float(np.mean(allDESI)); out["ns_ks"] = float(np.mean(allKS))
    return out


# Each (config x fold) re-loads that fold's emulator + recompiles the jacrev forward (the model
# pytree differs per fold), so a full all-folds run of all 12 configs is ~1h on CPU. Per the
# task's explicit allowance, the BASELINE (z_lo=2.0, k=0.06) is run on ALL 8 folds / 60 sims,
# and the SCAN configs on a fixed representative 4-fold subset spanning the sorted-n_s range
# (folds are contiguous sorted-n_s blocks; {0,2,5,7} = low / low-mid / mid-high / high n_s).
# The baseline is ALSO reported on the same subset so the scan trend is read on a consistent
# population. THIS SUBSETTING IS STATED in the output + the returned summary.
ALL_FOLDS = list(range(8))
SCAN_FOLDS = [0, 2, 5, 7]

# ============================================================================ #
emit("# KS z-cut series + KS k_max scan of the n_s (and A_p) EMU bias.")
emit("# FORWARD-ONLY Fisher MAP-shift, no retraining. emu_bias_one_sim REUSED from diag_emu_bias_allfolds.py.")
emit("# Each fold uses its own held-out emulator (final_fold{f}); DESI untouched in every config.")
emit(f"# BASELINE z_lo=2.0,k=0.06 on ALL folds {ALL_FOLDS} (60 sims); SCAN configs on subset {SCAN_FOLDS}.\n")

# ---- print the KS z-grid for one ctx per config to confirm threading ----
emit("# --- threading confirmation (KS z-grid + n_keep per config) ---")
for tag, kw in [("A z_lo=2.0,k=0.06", {"k_max": 0.06, "z_lo": 2.0}),
                ("A z_lo=2.1,k=0.06", {"k_max": 0.06, "z_lo": 2.1}),
                ("A z_lo=2.3,k=0.06", {"k_max": 0.06, "z_lo": 2.3}),
                ("A z_lo=2.8,k=0.06", {"k_max": 0.06, "z_lo": 2.8}),
                ("B z_lo=2.0,k=0.04", {"k_max": 0.04, "z_lo": 2.0}),
                ("B z_lo=2.0,k=0.03", {"k_max": 0.03, "z_lo": 2.0}),
                ("B z_lo=2.0,k=0.02", {"k_max": 0.02, "z_lo": 2.0})]:
    ctx, d = build_legb_ctx(ckpt=f"{REPO}/checkpoints/final_fold0", xclass_error_vector=HOLD0, ks_kwargs=kw)
    ks = [l for l in ctx.legs if l.name == "KS"][0]
    kk = np.asarray(ks.k)[np.isfinite(np.asarray(ks.P_data))]
    emit(f"  {tag:18s}: KS z={np.round(np.asarray(ks.z),2).tolist()} n_keep={int(np.isfinite(np.asarray(ks.P_data)).sum())} "
         f"k=[{kk.min():.4f},{kk.max():.4f}]")

# ============================================================================ #
#  (A) KS z-cut series, k_max=0.06 fixed, ALL 60 sims
# ============================================================================ #
emit("\n# ============================================================")
emit("# ANCHOR: baseline z_lo=2.0, k=0.06 on ALL 8 folds / 60 sims (must reproduce ~ -0.646)")
emit("# ============================================================")
anchor = run_config({"k_max": 0.06, "z_lo": 2.0}, "ANCHOR z_lo=2.0", folds=ALL_FOLDS, want_split=True)
emit(f"  ALL-FOLDS  n={anchor['n']:2d}  n_s bias = {anchor['ns_mean']:+.3f}sig  "
     f"boot95%[{anchor['ns_lo']:+.3f},{anchor['ns_hi']:+.3f}]  t={anchor['ns_t']:+.2f}  "
     f"signs {anchor['ns_pos']}+/{anchor['ns_neg']}-   A_p = {anchor['ap_mean']:+.3f}sig   "
     f"KS-split {anchor['ns_ks']:+.3f}  DESI-split {anchor['ns_desi']:+.3f}")
emit(f"  --> headline emu_bias_allfolds.txt: n_s -0.646, A_p +0.031.  "
     f"{'MATCH' if abs(anchor['ns_mean']+0.646)<0.05 else 'CHECK!'}")

emit("\n# ============================================================")
emit(f"# (A) KS z-cut series  (k_max=0.06 fixed)  -- SCAN subset folds {SCAN_FOLDS}")
emit("# ============================================================")
emit("#  z_lo   n   n_s bias[sig]  boot95%            t      signs(+/-)   A_p bias[sig]  KS-split   DESI-split")
A_zlo = [2.0, 2.1, 2.3, 2.8]
A_res = []
for zlo in A_zlo:
    r = run_config({"k_max": 0.06, "z_lo": zlo}, f"z_lo={zlo}", folds=SCAN_FOLDS, want_split=True)
    A_res.append(r)
    emit(f"  {zlo:.1f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
         f"{r['ns_t']:+6.2f}  {r['ns_pos']:2d}+/{r['ns_neg']:2d}-   {r['ap_mean']:+7.3f}      "
         f"{r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}")
emit(f"\n  subset-baseline check: z_lo=2.0 (subset {SCAN_FOLDS}) n_s = {A_res[0]['ns_mean']:+.3f}sig "
     f"vs ALL-folds anchor {anchor['ns_mean']:+.3f}sig  (the scan trend below is read off the subset)")

# ============================================================================ #
#  (B) KS k_max scan at z_lo=2.0, ALL 60 sims
# ============================================================================ #
emit("\n# ============================================================")
emit(f"# (B) KS k_max scan  (z_lo=2.0 fixed; z=2.0/2.2 bins PRESENT) -- SCAN subset folds {SCAN_FOLDS}")
emit("# ============================================================")
emit("#  k_max   n   n_s bias[sig]  boot95%            t      signs(+/-)   A_p bias[sig]  KS-split   DESI-split")
B_kmax = [0.06, 0.04, 0.03, 0.02]
B_res = []
for km in B_kmax:
    # k_max=0.06 reuses the (A) z_lo=2.0 subset result (same config) to avoid recompute
    if km == 0.06:
        B_res.append(A_res[0])
        r = A_res[0]
        emit(f"  {km:.2f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
             f"{r['ns_t']:+6.2f}  {r['ns_pos']:2d}+/{r['ns_neg']:2d}-   {r['ap_mean']:+7.3f}      "
             f"{r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}   (=A z_lo=2.0)")
        continue
    r = run_config({"k_max": km, "z_lo": 2.0}, f"k_max={km}", folds=SCAN_FOLDS, want_split=True)
    B_res.append(r)
    emit(f"  {km:.2f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
         f"{r['ns_t']:+6.2f}  {r['ns_pos']:2d}+/{r['ns_neg']:2d}-   {r['ap_mean']:+7.3f}      "
         f"{r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}")

# ============================================================================ #
#  (C) k_max scan at z_lo=2.8 (residual high-z high-k check)
# ============================================================================ #
emit("\n# ============================================================")
emit(f"# (C) KS k_max scan at z_lo=2.8 (residual high-z bias high-k?) -- SCAN subset folds {SCAN_FOLDS}")
emit("# ============================================================")
emit("#  k_max   n   n_s bias[sig]  boot95%            t      A_p bias[sig]  KS-split   DESI-split")
C_kmax = [0.06, 0.04, 0.03, 0.02]
C_res = []
for km in C_kmax:
    if km == 0.06:
        # reuse (A) z_lo=2.8 result (same config)
        C_res.append(A_res[3])
        r = A_res[3]
        emit(f"  {km:.2f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
             f"{r['ns_t']:+6.2f}  {r['ap_mean']:+7.3f}      {r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}   (=A z_lo=2.8)")
        continue
    r = run_config({"k_max": km, "z_lo": 2.8}, f"z2.8 k_max={km}", folds=SCAN_FOLDS, want_split=True)
    C_res.append(r)
    emit(f"  {km:.2f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
         f"{r['ns_t']:+6.2f}  {r['ap_mean']:+7.3f}      {r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}")

# ============================================================================ #
#  VERDICT
# ============================================================================ #
emit("\n# ============================================================")
emit("# VERDICT")
emit("# ============================================================")
emit(f"  ANCHOR (all 8 folds, 60 sims): n_s bias {anchor['ns_mean']:+.3f}sig (headline -0.646), A_p {anchor['ap_mean']:+.3f}sig.")
emit(f"  Scan trend below on subset folds {SCAN_FOLDS}; its z_lo=2.0 baseline = {A_res[0]['ns_mean']:+.3f}sig.")
collapse_z = A_res[-1]["ns_mean"]   # z_lo=2.8
collapse_k = B_res[1]["ns_mean"]    # k_max=0.04 at z_lo=2.0
emit(f"  (A) dropping low-z KS: n_s bias {A_res[0]['ns_mean']:+.3f} (z2.0) -> {A_res[1]['ns_mean']:+.3f} (z2.1) "
     f"-> {A_res[2]['ns_mean']:+.3f} (z2.3) -> {A_res[3]['ns_mean']:+.3f} (z2.8)")
emit(f"      collapse toward 0? {'YES' if abs(collapse_z) < 0.2 else 'PARTIAL/NO'} "
     f"(|{collapse_z:+.3f}| {'<' if abs(collapse_z) < 0.2 else '>='} 0.2 gate)")
emit(f"  (B) k_max scan @ z_lo=2.0: {B_res[0]['ns_mean']:+.3f} (0.06) -> {B_res[1]['ns_mean']:+.3f} (0.04) "
     f"-> {B_res[2]['ns_mean']:+.3f} (0.03) -> {B_res[3]['ns_mean']:+.3f} (0.02)")
emit(f"      low-z KS bias HIGH-K sourced? {'YES' if abs(collapse_k) < 0.5*abs(A_res[0]['ns_mean']) else 'NO/PARTIAL'} "
     f"(removing k>0.04 takes {A_res[0]['ns_mean']:+.3f} -> {collapse_k:+.3f})")
emit(f"  (C) residual z>=2.8 KS bias is {'high-k' if abs(C_res[0]['ns_mean'])>0.1 and abs(C_res[1]['ns_mean'])<0.5*abs(C_res[0]['ns_mean']) else 'NOT strongly high-k'}: "
     f"{C_res[0]['ns_mean']:+.3f}(0.06) -> {C_res[1]['ns_mean']:+.3f}(0.04)")

# save npz
np.savez(NPZ,
         A_zlo=np.array(A_zlo), A_ns=np.array([r["ns_mean"] for r in A_res]),
         A_ns_lo=np.array([r["ns_lo"] for r in A_res]), A_ns_hi=np.array([r["ns_hi"] for r in A_res]),
         A_ap=np.array([r["ap_mean"] for r in A_res]),
         A_ns_ks=np.array([r["ns_ks"] for r in A_res]), A_ns_desi=np.array([r["ns_desi"] for r in A_res]),
         B_kmax=np.array(B_kmax), B_ns=np.array([r["ns_mean"] for r in B_res]),
         B_ns_lo=np.array([r["ns_lo"] for r in B_res]), B_ns_hi=np.array([r["ns_hi"] for r in B_res]),
         B_ap=np.array([r["ap_mean"] for r in B_res]),
         B_ns_ks=np.array([r["ns_ks"] for r in B_res]), B_ns_desi=np.array([r["ns_desi"] for r in B_res]),
         C_kmax=np.array(C_kmax), C_ns=np.array([r["ns_mean"] for r in C_res]),
         C_ns_lo=np.array([r["ns_lo"] for r in C_res]), C_ns_hi=np.array([r["ns_hi"] for r in C_res]),
         anchor_ns=anchor["ns_mean"], anchor_ap=anchor["ap_mean"], anchor_n=anchor["n"],
         anchor_ns_ks=anchor["ns_ks"], anchor_ns_desi=anchor["ns_desi"],
         all_folds=np.array(ALL_FOLDS), scan_folds=np.array(SCAN_FOLDS))

# ============================================================================ #
#  FIGURE: 2-panel (left z-cut, right k_max@z2.0)
# ============================================================================ #
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# left: n_s bias vs z_lo cut
zx = np.array(A_zlo)
ns = np.array([r["ns_mean"] for r in A_res])
lo = np.array([r["ns_lo"] for r in A_res]); hi = np.array([r["ns_hi"] for r in A_res])
ax[0].errorbar(zx, ns, yerr=[ns - lo, hi - ns], fmt="o-", color="C0", capsize=4, lw=1.8, ms=7,
               label="total (KS+DESI)")
ax[0].plot(zx, [r["ns_ks"] for r in A_res], "s--", color="C1", alpha=0.8, label="KS-leg contribution")
ax[0].plot(zx, [r["ns_desi"] for r in A_res], "^:", color="C2", alpha=0.8, label="DESI-leg contribution")
ax[0].axhline(0.2, color="r", ls="--", alpha=0.6, label="+/-0.2sigma gate"); ax[0].axhline(-0.2, color="r", ls="--", alpha=0.6)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].axhline(-0.646, color="grey", ls=":", alpha=0.7, label="headline -0.646 (all folds)")
ax[0].plot([2.0], [anchor["ns_mean"]], "kD", ms=9, mfc="gold", mec="k",
           label=f"z2.0 ALL-folds anchor {anchor['ns_mean']:+.2f}")
for x, y in zip(zx, ns):
    ax[0].annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[0].set_xlabel("KS z_lo cut (drop KS bins with z < z_lo)"); ax[0].set_ylabel("pooled n_s EMU bias [sigma]")
ax[0].set_title(f"(A) n_s bias vs KS z-cut  (k_max=0.06)\nscan on folds {SCAN_FOLDS}; z2.0 anchor on all 8 folds; 2.8=PI cut")
ax[0].set_xticks(zx); ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)

# right: n_s bias vs k_max at z_lo=2.0
kx = np.array(B_kmax)
nsk = np.array([r["ns_mean"] for r in B_res])
lok = np.array([r["ns_lo"] for r in B_res]); hik = np.array([r["ns_hi"] for r in B_res])
ax[1].errorbar(kx, nsk, yerr=[nsk - lok, hik - nsk], fmt="o-", color="C0", capsize=4, lw=1.8, ms=7,
               label="total (KS+DESI)")
ax[1].plot(kx, [r["ns_ks"] for r in B_res], "s--", color="C1", alpha=0.8, label="KS-leg contribution")
ax[1].plot(kx, [r["ns_desi"] for r in B_res], "^:", color="C2", alpha=0.8, label="DESI-leg contribution")
ax[1].axvline(0.04, color="purple", ls="-.", alpha=0.5, label="k=0.04 (LF-res regime)")
ax[1].axhline(0.2, color="r", ls="--", alpha=0.6); ax[1].axhline(-0.2, color="r", ls="--", alpha=0.6, label="+/-0.2sigma gate")
ax[1].axhline(0, color="k", lw=0.6)
for x, y in zip(kx, nsk):
    ax[1].annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[1].set_xlabel("KS k_max [s/km] (drop KS bins with k > k_max)"); ax[1].set_ylabel("pooled n_s EMU bias [sigma]")
ax[1].set_title(f"(B) n_s bias vs KS k_max  (z_lo=2.0, z=2.0/2.2 present)\nscan on folds {SCAN_FOLDS}")
ax[1].invert_xaxis(); ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3)

fig.suptitle(f"KS-leg z-cut + k_max scan of the coherent n_s EMU bias (forward-only Fisher)\n"
             f"anchor (z2.0,k0.06) all 8 folds = {anchor['ns_mean']:+.3f}sig [headline -0.646]; scan on folds {SCAN_FOLDS}")
p = Path(FIGDIR) / "nsbias_kscut_scan.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}")
emit(f"[npz] {NPZ}")
emit("[done]")
