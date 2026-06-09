"""EXTENSION of scripts/diag_nsbias_kscut_scan.py: add KS z_lo = 2.4 and 2.6 to the
n_s (and A_p) EMU-bias z-cut series, plus an all-folds (60-sim) number for z_lo=2.6.

FORWARD-ONLY (Fisher MAP-shift, no NUTS, no retraining). REUSES the EXACT machinery of
scripts/diag_nsbias_kscut_scan.py -- `_load_emu_bias_one_sim` (which in turn reuses
`emu_bias_one_sim` verbatim from diag_emu_bias_allfolds.py), `emu_bias_one_sim_split`,
`boot`, and `run_config` -- by exec'ing ONLY those function-definition blocks out of the
prior script (avoiding its top-level driver, which truncates the .txt and re-runs all 12
configs). The ONLY thing that varies between configs is `ks_kwargs={"k_max":0.06,"z_lo":Z}`
threaded into build_legb_ctx -> data_likelihood.load_ks_leg. DESI untouched; the KS cov is
sliced by the SAME `keep` mask. Subset folds {0,2,5,7} match the prior scan exactly so the
new points are directly comparable.

What this run produces:
  - threading confirmation (KS z-grid + n_keep) for z_lo in {2.3, 2.4, 2.6}
  - z_lo=2.3 reproduced as a consistency anchor (prior: +0.041sig)
  - z_lo=2.4 and z_lo=2.6 pooled n_s + A_p bias, boot95%, t, signs, KS/DESI split (subset)
  - z_lo=2.6 ALL-folds (60-sim) n_s + A_p bias + split (if budget allows; gated by env)
  - combined readout appended to nsbias_kscut_scan.txt
  - regenerated nsbias_kscut_scan.png / .npz with the full z series {2.0,2.1,2.3,2.4,2.6,2.8}

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_nsbias_kscut_scan_2426.py
"""
import gc
import os
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

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/nsbias_kscut_scan.txt"
NPZ = f"{FIGDIR}/nsbias_kscut_scan.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"

# ---- reuse the prior script's functions verbatim, WITHOUT running its driver ----
PRIOR = Path(REPO) / "scripts" / "diag_nsbias_kscut_scan.py"
_lines = PRIOR.read_text().splitlines(keepends=True)


def _slice_func(name):
    start = next(i for i, ln in enumerate(_lines) if ln.startswith(f"def {name}("))
    end = start + 1
    while end < len(_lines) and (_lines[end].strip() == "" or _lines[end][:1] in (" ", "\t")):
        end += 1
    return "".join(_lines[start:end])


# Namespace pre-populated with exactly the names the prior functions close over.
_ns = dict(
    np=np, jnp=jnp, jax=jax, gc=gc, DL=DL, Path=Path,
    HCD_Z_PIVOT=HCD_Z_PIVOT, AP_I=AP_I, NS_I=NS_I,
    build_legb_ctx=build_legb_ctx, held_out_sims=held_out_sims,
    make_truth_from_sim=make_truth_from_sim, make_legb_mock=make_legb_mock,
    _mock_core_per_leg=_mock_core_per_leg,
    REPO=REPO, HOLD0=HOLD0, __file__=str(PRIOR),
)
# `_load_emu_bias_one_sim` (used inside run_config via emu_bias_one_sim) reads __file__'s
# sibling diag_emu_bias_allfolds.py; PRIOR's parent is the same scripts/ dir, so OK.
exec(_slice_func("_load_emu_bias_one_sim"), _ns)
_ns["emu_bias_one_sim"] = _ns["_load_emu_bias_one_sim"]()
exec(_slice_func("emu_bias_one_sim_split"), _ns)
exec(_slice_func("boot"), _ns)
exec(_slice_func("run_config"), _ns)
run_config = _ns["run_config"]

# ---- append (do NOT truncate) to the prior readout ----
_rf = open(RESULTS, "a")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()

SCAN_FOLDS = [0, 2, 5, 7]
ALL_FOLDS = list(range(8))
RUN_ALLFOLDS_26 = os.environ.get("RUN_ALLFOLDS_26", "1") == "1"

emit("\n\n# ############################################################")
emit("# EXTENSION (diag_nsbias_kscut_scan_2426.py): add KS z_lo=2.4 and z_lo=2.6")
emit("# Same machinery, same subset folds [0,2,5,7], k_max=0.06 fixed. z_lo=2.3 re-run as anchor.")
emit("# ############################################################")

# ---- threading confirmation for the new + anchor cuts ----
emit("\n# --- threading confirmation (KS z-grid + n_keep) for z_lo in {2.3, 2.4, 2.6} ---")
for tag, kw in [("z_lo=2.3,k=0.06", {"k_max": 0.06, "z_lo": 2.3}),
                ("z_lo=2.4,k=0.06", {"k_max": 0.06, "z_lo": 2.4}),
                ("z_lo=2.6,k=0.06", {"k_max": 0.06, "z_lo": 2.6})]:
    ctx, d = build_legb_ctx(ckpt=f"{REPO}/checkpoints/final_fold0", xclass_error_vector=HOLD0, ks_kwargs=kw)
    ks = [l for l in ctx.legs if l.name == "KS"][0]
    finite = np.isfinite(np.asarray(ks.P_data))
    kk = np.asarray(ks.k)[finite]
    emit(f"  {tag:16s}: KS z={np.round(np.asarray(ks.z),2).tolist()} n_keep={int(finite.sum())} "
         f"k=[{kk.min():.4f},{kk.max():.4f}]")
    del ctx, d; gc.collect()

# ---- subset scan: 2.3 (anchor), 2.4, 2.6 ----
emit("\n# ============================================================")
emit(f"# z-cut subset scan (k_max=0.06; folds {SCAN_FOLDS})")
emit("# ============================================================")
emit("#  z_lo   n   n_s bias[sig]  boot95%            t      signs(+/-)   A_p bias[sig]  KS-split   DESI-split")
new_zlo = [2.3, 2.4, 2.6]
new_res = {}
for zlo in new_zlo:
    r = run_config({"k_max": 0.06, "z_lo": zlo}, f"z_lo={zlo}", folds=SCAN_FOLDS, want_split=True)
    new_res[zlo] = r
    gate = "IN-GATE" if abs(r["ns_mean"]) < 0.2 else "OUT-OF-GATE"
    emit(f"  {zlo:.1f}   {r['n']:2d}   {r['ns_mean']:+7.3f}     [{r['ns_lo']:+.3f},{r['ns_hi']:+.3f}]  "
         f"{r['ns_t']:+6.2f}  {r['ns_pos']:2d}+/{r['ns_neg']:2d}-   {r['ap_mean']:+7.3f}      "
         f"{r['ns_ks']:+7.3f}    {r['ns_desi']:+7.3f}   [{gate}]")

emit(f"\n  consistency anchor: z_lo=2.3 here = {new_res[2.3]['ns_mean']:+.3f}sig  (prior scan: +0.041sig)")

# ---- all-folds (60-sim) for z_lo=2.6 ----
allfold_26 = None
if RUN_ALLFOLDS_26:
    emit("\n# ============================================================")
    emit(f"# z_lo=2.6 ALL FOLDS {ALL_FOLDS} (60 sims) -- recommended-cut all-folds number")
    emit("# ============================================================")
    allfold_26 = run_config({"k_max": 0.06, "z_lo": 2.6}, "z_lo=2.6 ALL", folds=ALL_FOLDS, want_split=True)
    gate = "IN-GATE" if abs(allfold_26["ns_mean"]) < 0.2 else "OUT-OF-GATE"
    emit(f"  ALL-FOLDS  n={allfold_26['n']:2d}  n_s bias = {allfold_26['ns_mean']:+.3f}sig  "
         f"boot95%[{allfold_26['ns_lo']:+.3f},{allfold_26['ns_hi']:+.3f}]  t={allfold_26['ns_t']:+.2f}  "
         f"signs {allfold_26['ns_pos']}+/{allfold_26['ns_neg']}-   A_p = {allfold_26['ap_mean']:+.3f}sig   "
         f"KS-split {allfold_26['ns_ks']:+.3f}  DESI-split {allfold_26['ns_desi']:+.3f}   [{gate}]")
    emit(f"  (cf. z_lo=2.6 subset {SCAN_FOLDS} = {new_res[2.6]['ns_mean']:+.3f}sig; "
         f"z_lo=2.0 all-folds anchor was -0.646, subset -0.775)")

# ---- combined readout merging prior {2.0,2.1,2.3,2.8} (subset) with new {2.4,2.6} ----
prior = np.load(NPZ, allow_pickle=True)
pz = list(np.round(prior["A_zlo"], 2))
def pidx(z): return pz.index(round(z, 2))
combined = []
for z in [2.0, 2.1, 2.3, 2.4, 2.6, 2.8]:
    if z in (2.4, 2.6):
        r = new_res[z]
        combined.append(dict(z=z, ns=r["ns_mean"], lo=r["ns_lo"], hi=r["ns_hi"], t=r["ns_t"],
                             ap=r["ap_mean"], ks=r["ns_ks"], desi=r["ns_desi"],
                             pos=r["ns_pos"], neg=r["ns_neg"], src="new(subset)"))
    else:
        i = pidx(z)
        combined.append(dict(z=z, ns=float(prior["A_ns"][i]), lo=float(prior["A_ns_lo"][i]),
                             hi=float(prior["A_ns_hi"][i]), t=float("nan"),
                             ap=float(prior["A_ap"][i]), ks=float(prior["A_ns_ks"][i]),
                             desi=float(prior["A_ns_desi"][i]), pos=-1, neg=-1, src="prior(subset)"))

emit("\n# ============================================================")
emit("# COMBINED z-cut series (subset folds [0,2,5,7], k_max=0.06): z_lo in {2.0,2.1,2.3,2.4,2.6,2.8}")
emit("# ============================================================")
emit("#  z_lo    n_s bias[sig]   boot95%             A_p bias[sig]   KS-split   DESI-split   in-gate?  source")
for c in combined:
    gate = "YES" if abs(c["ns"]) < 0.2 else "NO "
    emit(f"  {c['z']:.1f}    {c['ns']:+7.3f}      [{c['lo']:+.3f},{c['hi']:+.3f}]   {c['ap']:+7.3f}       "
         f"{c['ks']:+7.3f}    {c['desi']:+7.3f}      {gate}     {c['src']}")
if allfold_26 is not None:
    a = allfold_26
    gate = "YES" if abs(a["ns_mean"]) < 0.2 else "NO "
    emit(f"  2.6*   {a['ns_mean']:+7.3f}      [{a['ns_lo']:+.3f},{a['ns_hi']:+.3f}]   {a['ap_mean']:+7.3f}       "
         f"{a['ns_ks']:+7.3f}    {a['ns_desi']:+7.3f}      {gate}     ALL-FOLDS(60)")
emit(f"  anchor z2.0 ALL-folds (prior) = {float(prior['anchor_ns']):+.3f}sig "
     f"(KS-split {float(prior['anchor_ns_ks']):+.3f}, DESI-split {float(prior['anchor_ns_desi']):+.3f})")

# ---- verdict ----
emit("\n# ============================================================")
emit("# VERDICT (do 2.4/2.6 close as well as 2.8?)")
emit("# ============================================================")
r24, r26 = new_res[2.4], new_res[2.6]
for z, r in [(2.4, r24), (2.6, r26)]:
    g = "INSIDE" if abs(r["ns_mean"]) < 0.2 else "OUTSIDE"
    emit(f"  z_lo={z}: n_s {r['ns_mean']:+.3f}sig (t={r['ns_t']:+.2f}, {r['ns_pos']}+/{r['ns_neg']}-), "
         f"A_p {r['ap_mean']:+.3f}sig, KS-split {r['ns_ks']:+.3f}, DESI-split {r['ns_desi']:+.3f} -> {g} +/-0.2 gate")
emit(f"  z_lo=2.3 anchor reproduced: {new_res[2.3]['ns_mean']:+.3f}sig (prior +0.041) -- "
     f"{'consistent' if abs(new_res[2.3]['ns_mean'] - 0.041) < 0.05 else 'CHECK'}.")
both_in = abs(r24["ns_mean"]) < 0.2 and abs(r26["ns_mean"]) < 0.2
emit(f"  ONE-LINE: z_lo=2.4 and 2.6 {'BOTH close as cleanly as 2.8 (all inside +/-0.2sig)' if both_in else 'do NOT both close'}; "
     f"closure already achieved the instant KS z=2.0+2.2 are dropped (z_lo>=2.4).")

# ---- save extended npz (preserve prior keys, add the combined series) ----
out = {k: prior[k] for k in prior.files}
comb_z = np.array([c["z"] for c in combined])
out["comb_zlo"] = comb_z
out["comb_ns"] = np.array([c["ns"] for c in combined])
out["comb_ns_lo"] = np.array([c["lo"] for c in combined])
out["comb_ns_hi"] = np.array([c["hi"] for c in combined])
out["comb_ap"] = np.array([c["ap"] for c in combined])
out["comb_ns_ks"] = np.array([c["ks"] for c in combined])
out["comb_ns_desi"] = np.array([c["desi"] for c in combined])
out["new_zlo"] = np.array([2.4, 2.6])
out["new_ns"] = np.array([r24["ns_mean"], r26["ns_mean"]])
out["new_ap"] = np.array([r24["ap_mean"], r26["ap_mean"]])
out["new_ns_ks"] = np.array([r24["ns_ks"], r26["ns_ks"]])
out["new_ns_desi"] = np.array([r24["ns_desi"], r26["ns_desi"]])
out["anchor23_recheck"] = float(new_res[2.3]["ns_mean"])
if allfold_26 is not None:
    out["allfold26_ns"] = float(allfold_26["ns_mean"])
    out["allfold26_ap"] = float(allfold_26["ap_mean"])
    out["allfold26_ns_ks"] = float(allfold_26["ns_ks"])
    out["allfold26_ns_desi"] = float(allfold_26["ns_desi"])
    out["allfold26_n"] = int(allfold_26["n"])
np.savez(NPZ, **out)

# ---- regenerate figure: left panel now shows the full z series {2.0,2.1,2.3,2.4,2.6,2.8} ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

zx = comb_z
ns = out["comb_ns"]; lo = out["comb_ns_lo"]; hi = out["comb_ns_hi"]
ax[0].errorbar(zx, ns, yerr=[ns - lo, hi - ns], fmt="o-", color="C0", capsize=4, lw=1.8, ms=7,
               label="total (KS+DESI)")
ax[0].plot(zx, out["comb_ns_ks"], "s--", color="C1", alpha=0.8, label="KS-leg contribution")
ax[0].plot(zx, out["comb_ns_desi"], "^:", color="C2", alpha=0.8, label="DESI-leg contribution")
# highlight the two new points
new_mask = np.isin(np.round(zx, 2), [2.4, 2.6])
ax[0].plot(zx[new_mask], ns[new_mask], "o", ms=13, mfc="none", mec="green", mew=2.0,
           label="NEW (z_lo=2.4, 2.6)")
if allfold_26 is not None:
    ax[0].plot([2.6], [allfold_26["ns_mean"]], "P", ms=11, mfc="magenta", mec="k",
               label=f"z2.6 ALL-folds {allfold_26['ns_mean']:+.2f}")
ax[0].axhline(0.2, color="r", ls="--", alpha=0.6, label="+/-0.2sigma gate"); ax[0].axhline(-0.2, color="r", ls="--", alpha=0.6)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].axhline(-0.646, color="grey", ls=":", alpha=0.7, label="headline -0.646 (all folds)")
ax[0].plot([2.0], [float(prior["anchor_ns"])], "kD", ms=9, mfc="gold", mec="k",
           label=f"z2.0 ALL-folds anchor {float(prior['anchor_ns']):+.2f}")
for x, y in zip(zx, ns):
    ax[0].annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[0].set_xlabel("KS z_lo cut (drop KS bins with z < z_lo)"); ax[0].set_ylabel("pooled n_s EMU bias [sigma]")
ax[0].set_title(f"(A) n_s bias vs KS z-cut  (k_max=0.06)\nscan on folds {SCAN_FOLDS}; z2.0 anchor on all 8 folds; 2.4/2.6 NEW")
ax[0].set_xticks(zx); ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].grid(alpha=0.3)

# right panel: keep the prior k_max@z2.0 scan unchanged
kx = prior["B_kmax"]; nsk = prior["B_ns"]; lok = prior["B_ns_lo"]; hik = prior["B_ns_hi"]
ax[1].errorbar(kx, nsk, yerr=[nsk - lok, hik - nsk], fmt="o-", color="C0", capsize=4, lw=1.8, ms=7,
               label="total (KS+DESI)")
ax[1].plot(kx, prior["B_ns_ks"], "s--", color="C1", alpha=0.8, label="KS-leg contribution")
ax[1].plot(kx, prior["B_ns_desi"], "^:", color="C2", alpha=0.8, label="DESI-leg contribution")
ax[1].axvline(0.04, color="purple", ls="-.", alpha=0.5, label="k=0.04 (LF-res regime)")
ax[1].axhline(0.2, color="r", ls="--", alpha=0.6); ax[1].axhline(-0.2, color="r", ls="--", alpha=0.6, label="+/-0.2sigma gate")
ax[1].axhline(0, color="k", lw=0.6)
for x, y in zip(kx, nsk):
    ax[1].annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[1].set_xlabel("KS k_max [s/km] (drop KS bins with k > k_max)"); ax[1].set_ylabel("pooled n_s EMU bias [sigma]")
ax[1].set_title(f"(B) n_s bias vs KS k_max  (z_lo=2.0, z=2.0/2.2 present)\nscan on folds {SCAN_FOLDS}")
ax[1].invert_xaxis(); ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3)

fig.suptitle(f"KS-leg z-cut + k_max scan of the coherent n_s EMU bias (forward-only Fisher)\n"
             f"z-cut series now z_lo in {{2.0,2.1,2.3,2.4,2.6,2.8}}; anchor (z2.0) all 8 folds = {float(prior['anchor_ns']):+.3f}sig")
p = Path(FIGDIR) / "nsbias_kscut_scan.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
emit(f"\n[fig] {p}")
emit(f"[npz] {NPZ}")
emit("[done-2426]")
