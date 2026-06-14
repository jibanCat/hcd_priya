"""Cosmology-safety arm (#9 referee-panel must-do): does a MIS-SPECIFIED subDLA prior CENTER drag
n_s/A_p? For each joint-DESI+KS mock (XS_f*), the subDLA prior center was run at 0 / +1σ / −1σ
(subdla_center_shift). The panel's pass condition: the cosmology MEANS (n_s, A_p) must stay within
0.3σ of the unshifted (s0) run under a ±1σ subDLA prior-center mis-specification — i.e. cosmology is
robust to subDLA prior mis-centering (the residual risk behind the closure corr(subDLA,n_s)=+0.82).

Reads checkpoints/stepA/XS_<fid>_{s0,sp,sm}_c{0..3}.npz, converts ns/Ap unit→physical, and reports
Δ(mean)/σ for n_s and A_p (sp vs s0, sm vs s0). Also the subDLA-mean shift (sanity: the prior-center
shift must actually move the subDLA posterior, else the knob did nothing).

ENV: PYTHONNOUSERSITE=1 /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_xs_cosmo_safety.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "/home/mfho/hcd_priya/checkpoints/stepA"
OUTDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
os.makedirs(OUTDIR, exist_ok=True)
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9)}
FIDS = ["XS_f6", "XS_f4"]
SHIFTS = [("s0", "0σ"), ("sp", "+1σ"), ("sm", "−1σ")]
PASS = 0.30   # |Δmean|/σ threshold


def to_phys(x, nm):
    lo, hi = LIMS[nm]
    v = lo + x * (hi - lo)
    return v * 1e9 if nm == "Ap" else v


def load(base):
    ps = sorted(p for p in glob.glob(f"{CKPT}/{base}_c*.npz")
                if os.path.basename(p).split(base)[1].startswith("_c"))
    if not ps:
        return None
    chains, names, truth, ndiv = [], None, None, 0
    for p in ps:
        z = np.load(p, allow_pickle=True)
        if names is None:
            names = [str(x) for x in z["names"]]; truth = np.asarray(z["truth_vec"])
        chains.append(np.asarray(z["packed"]))
        ndiv += int(z["divergences"]) if "divergences" in z else 0
    P = np.concatenate(chains, 0)
    return dict(P=P, names=names, truth=truth, nchain=len(ps), ndiv=ndiv)


def col(arm, nm):
    i = arm["names"].index(nm)
    x = arm["P"][:, i]
    return (to_phys(x, nm) if nm in LIMS else x)


rows = []
print(f"{'fiducial':10s} {'param':6s} {'s0 mean±σ':>16s} {'+1σ Δ/σ':>10s} {'−1σ Δ/σ':>10s}  verdict")
for fid in FIDS:
    arms = {tag: load(f"{fid}_{tag}") for tag, _ in SHIFTS}
    if arms["s0"] is None:
        print(f"{fid}: no s0 chains yet"); continue
    rec = {"fid": fid}
    for nm in ("ns", "Ap", "alpha_subdla"):
        s0 = col(arms["s0"], nm); m0, sd0 = s0.mean(), s0.std()
        dsp = (col(arms["sp"], nm).mean() - m0) / sd0 if arms.get("sp") else np.nan
        dsm = (col(arms["sm"], nm).mean() - m0) / sd0 if arms.get("sm") else np.nan
        rec[nm] = (m0, sd0, dsp, dsm)
        if nm in ("ns", "Ap"):
            ok = (abs(dsp) < PASS and abs(dsm) < PASS)
            v = "PASS" if ok else "FAIL"
            mstr = f"{m0:.4f}±{sd0:.4f}" if nm == "ns" else f"{m0:.3f}±{sd0:.3f}"
            print(f"{fid:10s} {nm:6s} {mstr:>16s} {dsp:>+10.2f} {dsm:>+10.2f}  {v}")
    # subDLA sanity: the shift must move the subDLA mean
    sm0, ssd0, sdsp, sdsm = rec["alpha_subdla"]
    print(f"{'':10s} subDLA mean {sm0:.4f}±{ssd0:.4f}  +1σ→{sdsp:+.2f}σ  −1σ→{sdsm:+.2f}σ  "
          f"(knob {'ACTIVE' if max(abs(sdsp), abs(sdsm)) > 0.3 else 'INERT?'}); div={arms['s0']['ndiv']}")
    rows.append(rec)

# verdict — GUARDED: a missing/failed arm must NOT silently pass (max(0.0, nan)=0.0 would print a
# spurious PASS on incomplete data). Require every (fid × {ns,Ap} × {+1σ,−1σ}) Δ to be finite.
deltas, complete = [], bool(rows)
for r in rows:
    for nm in ("ns", "Ap"):
        for dd in (r[nm][2], r[nm][3]):
            if np.isfinite(dd):
                deltas.append(abs(dd))
            else:
                complete = False
n_expect = len(FIDS) * 2 * 2
complete = complete and (len(deltas) == n_expect)
print(f"\n=== COSMOLOGY-SAFETY VERDICT ===")
if not complete:
    print(f"  ⏳ INCOMPLETE — {len(deltas)}/{n_expect} finite Δ over {len(rows)}/{len(FIDS)} mocks; "
          "rerun after all 24 chains land (health_xs.json all 'done'). NO verdict emitted.")
else:
    worst = max(deltas)
    print(f"  worst |Δ(cosmology mean)|/σ under ±1σ subDLA prior-center shift = {worst:.2f}σ (threshold {PASS}σ)")
    print("  ✅ PASS — cosmology robust to subDLA prior mis-centering" if worst < PASS
          else "  ⚠️  EXCEEDS threshold — a mis-centered subDLA prior moves cosmology; investigate")

# figure
if rows:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, nm, lab in zip(axes, ("ns", "Ap"), (r"$n_s$", r"$A_p\,[10^{-9}]$")):
        x = np.arange(len(rows))
        for j, (tag, tlab) in enumerate(SHIFTS):
            means, sds = [], []
            for r in rows:
                arm = load(f"{r['fid']}_{tag}")
                c = col(arm, nm) if arm else None
                means.append(c.mean() if c is not None else np.nan)
                sds.append(c.std() if c is not None else np.nan)
            ax.errorbar(x + (j - 1) * 0.12, means, yerr=sds, fmt="osD"[j], ms=6, capsize=3,
                        label=f"subDLA center {tlab}")
        # truth markers
        tv = [to_phys(load(f"{r['fid']}_s0")["truth"][load(f"{r['fid']}_s0")["names"].index(nm)], nm) for r in rows]
        ax.scatter(x, tv, marker="_", s=400, color="k", zorder=6, label="truth")
        ax.set_xticks(x); ax.set_xticklabels([r["fid"] for r in rows])
        ax.set_ylabel(lab); ax.set_title(f"{lab} vs subDLA prior-center shift")
        ax.grid(alpha=0.25)
        if nm == "ns":
            ax.legend(fontsize=8)
    fig.suptitle("Cosmology-safety: n_s/A_p under ±1σ subDLA prior-CENTER mis-specification (joint DESI+KS)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{OUTDIR}/xs_cosmo_safety.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")
