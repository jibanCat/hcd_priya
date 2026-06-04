"""Aggregate per-(sim, snap) .npz files from the multipoint sbatch array into:
  - a summary table (CSV-style printed to stdout + written to docs/figs/),
  - a single-figure overview: ratio bands per (sim, z) overlaid vs k.

Reads:  docs/superpowers/figs/multipoint/sim{idx}_snap{N}.npz  (12 files)
Writes:
  docs/superpowers/figs/2026-05-20-priya-p1d-multipoint-summary.csv
  docs/superpowers/figs/2026-05-20-priya-p1d-multipoint-ratios.png
"""
from pathlib import Path
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
files = sorted(glob.glob(str(_FIGS / "multipoint" / "*.npz")))
if not files:
    raise SystemExit("No multipoint .npz files found.")

# Collect summary rows
summary = []
for f in files:
    d = np.load(f, allow_pickle=True)
    for a_idx, alpha in enumerate(d["alphas"]):
        summary.append(dict(
            file=os.path.basename(f),
            sim_idx=int(d["sim_idx"]),
            snap=int(d["snap"]),
            z=float(d["z"]),
            z_idx=int(d["z_idx"]),
            alpha=float(alpha),
            target_F=float(d["target_F"][a_idx]),
            median=float(d["medians"][a_idx]),
            std=float(d["stds"][a_idx]),
            max_abs_dev=float(d["maxabs"][a_idx]),
        ))

# Sort by (sim_idx, z, alpha)
summary.sort(key=lambda r: (r["sim_idx"], r["z"], r["alpha"]))

# Print + write CSV
print(f"{'sim':>4} {'snap':>5} {'z':>5} {'alpha':>8} {'target_F':>9} {'median':>12} {'std':>11} {'max|r-1|':>11}")
print("-" * 80)
csv_path = _FIGS / "2026-05-20-priya-p1d-multipoint-summary.csv"
with open(csv_path, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for r in summary:
        w.writerow(r)
        print(f"{r['sim_idx']:>4} {r['snap']:>5} {r['z']:>5.2f} {r['alpha']:>8.4f} "
              f"{r['target_F']:>9.5f} {r['median']:>12.8f} {r['std']:>11.3e} {r['max_abs_dev']:>11.3e}")

# Overall stats
all_max = np.array([r["max_abs_dev"] for r in summary])
all_std = np.array([r["std"] for r in summary])
all_med = np.array([r["median"] for r in summary])
print(f"\nOverall (n={len(summary)} test points):")
print(f"  worst max|r-1|:         {all_max.max():.3e}")
print(f"  worst std:              {all_std.max():.3e}")
print(f"  median of medians:      {np.median(all_med):.8f}")
print(f"  all max|r-1| < 1e-4 ?   {bool((all_max < 1e-4).all())}")
print(f"  all max|r-1| < 1e-3 ?   {bool((all_max < 1e-3).all())}")
print(f"  user 1% target (< 1e-2)?{bool((all_max < 1e-2).all())}")

# ---- Plot: ratio vs k for each (sim, z) combo
# Layout: each sim is a column, each z is a row. In each panel, overlay the 10
# α curves.
sim_ids = sorted({r["sim_idx"] for r in summary})
z_vals = sorted({r["z"] for r in summary})
ncols, nrows = len(sim_ids), len(z_vals)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 2.3*nrows),
                         sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.05, "wspace": 0.05})
axes = np.atleast_2d(axes)
if nrows == 1: axes = axes.reshape(1, -1)
if ncols == 1: axes = axes.reshape(-1, 1)

# colormap for alpha
import matplotlib.cm as cm
norm = plt.matplotlib.colors.Normalize(vmin=0.65, vmax=1.30)
cmap = cm.viridis

for sci, sid in enumerate(sim_ids):
    for ri, zv in enumerate(z_vals):
        ax = axes[ri, sci]
        matched = [f for f in files
                   if abs(float(np.load(f, allow_pickle=True)["sim_idx"]) - sid) < 0.1
                   and abs(float(np.load(f, allow_pickle=True)["z"]) - zv) < 0.05]
        ax.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylim(0.9999, 1.0001)
        if matched:
            d = np.load(matched[0], allow_pickle=True)
            kp = d["k_priya"]
            for a_idx, alpha in enumerate(d["alphas"]):
                r = d["ratio_grid"][a_idx]
                col = cmap(norm(float(alpha)))
                ax.semilogx(kp, r, "-", lw=0.7, color=col)
        if ri == 0:
            ax.set_title(f"sim_idx = {sid}", fontsize=10)
        if sci == 0:
            ax.set_ylabel(f"z = {zv}\n" + r"$P_\mathrm{mine}/P_\mathrm{PRIYA}$",
                          fontsize=9)
        if ri == nrows - 1:
            ax.set_xlabel(r"$k$ [rad s km$^{-1}$]", fontsize=9)
        ax.grid(True, which="both", alpha=0.15)
        ax.tick_params(labelsize=8)

# colorbar
fig.subplots_adjust(right=0.91)
cax = fig.add_axes([0.93, 0.15, 0.013, 0.7])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_label(r"$\alpha$ (Kim 2013 slope)", fontsize=9)

fig.suptitle(
    f"P_mine / P_PRIYA across {ncols} sims × {nrows} redshifts × 10 α "
    f"({len(summary)} test points)\n"
    f"worst max|r-1| = {all_max.max():.2e}  |  worst std = {all_std.max():.2e}  "
    f"|  median of medians = {np.median(all_med):.7f}",
    fontsize=11, y=1.005)

out_png = _FIGS / "2026-05-20-priya-p1d-multipoint-ratios.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nWrote {out_png}")
print(f"Wrote {csv_path}")
