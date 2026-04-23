"""
Parameter sensitivity of Ω_HI and dN/dX (per class) at z=3, across
the 60 LF sims.  Reads figures/analysis/hcd_summary_lf.h5.

For each HCD quantity y ∈ {Ω_HI^LLS, Ω_HI^subDLA, Ω_HI^DLA,
dN/dX^LLS, dN/dX^subDLA, dN/dX^DLA}, and each PRIYA input
parameter θ ∈ {ns, A_p, herei, heref, alphaq, hub, omegamh2,
hireionz, bhfeedback}, we scatter y vs θ and annotate the Spearman
rank correlation ρ with p-value.

Outputs:
  figures/analysis/param_sens_omega_hi.png   (3 rows × 9 cols)
  figures/analysis/param_sens_dndx.png       (3 rows × 9 cols)
  figures/analysis/param_sens_summary.csv    machine-readable ρ table

Run:
    python3 scripts/plot_hcd_param_sensitivity.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir
DATA = data_dir()
SUMMARY = DATA / "hcd_summary_lf.h5"
OUT = ROOT / "figures" / "analysis" / "02_param_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
PARAM_KEYS = ["ns", "Ap", "herei", "heref", "alphaq",
              "hub", "omegamh2", "hireionz", "bhfeedback"]
CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}


def load_z3():
    with h5py.File(SUMMARY, "r") as f:
        z = f["z"][:]
        sel = np.abs(z - 3.0) < 0.05
        out = {}
        for cls in CLASSES:
            out[f"dndx_{cls}"] = f[f"dndx/{cls}"][:][sel]
            out[f"omega_{cls}"] = f[f"Omega_HI/{cls}"][:][sel]
        for pk in PARAM_KEYS:
            out[pk] = f[f"params/{pk}"][:][sel]
        out["sim"] = np.array([s.decode() for s in f["sim"][:]])[sel]
        out["z"] = z[sel]
    return out


def _panel(ax, xs, ys, color, pk, ylabel):
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    ax.scatter(xs, ys, s=10, alpha=0.75, color=color)
    if len(xs) > 3:
        rho, p = spearmanr(xs, ys)
        ax.text(0.02, 0.97, f"ρ={rho:+.2f}\np={p:.1g}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="gray", alpha=0.75))
    ax.set_xlabel(pk, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)


def plot_grid(d, quantity, outfile):
    """
    Build a 3-row × 9-col grid.  Rows = LLS / subDLA / DLA.
    Cols = 9 input params.  Each panel is y_{class} vs θ.
    """
    fig, axes = plt.subplots(3, 9, figsize=(22, 9), sharey=False)
    rows_data = []
    for i, cls in enumerate(CLASSES):
        ys = d[f"{quantity}_{cls}"]
        for j, pk in enumerate(PARAM_KEYS):
            xs = d[pk]
            ylabel = f"{quantity.replace('dndx','dN/dX').replace('omega','Ω_HI')}({cls})"
            _panel(axes[i, j], xs, ys, CLASS_COLORS[cls], pk, ylabel if j == 0 else "")
            if i == 0:
                axes[i, j].set_title(pk, fontsize=10)
            rho, p = spearmanr(xs[np.isfinite(xs) & np.isfinite(ys)],
                               ys[np.isfinite(xs) & np.isfinite(ys)])
            rows_data.append((quantity, cls, pk, rho, p))
    fig.suptitle(
        f"{quantity.replace('dndx','dN/dX').replace('omega','Ω_HI')} "
        f"vs 9 PRIYA parameters at z=3  (60 LF sims)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    return rows_data


def main():
    d = load_z3()
    print(f"Loaded z≈3 data for {len(d['sim'])} sims.")

    results = []
    print("Plotting Ω_HI sensitivity grid…")
    results += plot_grid(d, "omega", OUT / "param_sens_omega_hi.png")
    print(f"  wrote {OUT/'param_sens_omega_hi.png'}")

    print("Plotting dN/dX sensitivity grid…")
    results += plot_grid(d, "dndx", OUT / "param_sens_dndx.png")
    print(f"  wrote {OUT/'param_sens_dndx.png'}")

    # Also save a compact CSV so downstream scripts can pick up the ρ table.
    csv_path = DATA / "param_sens_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "class", "param", "spearman_rho", "pvalue"])
        for row in results:
            w.writerow(row)
    print(f"  wrote {csv_path}")

    # Summary: top-3 most-correlated param per (quantity, class)
    print("\nTop-3 strongest |ρ| per (quantity, class):")
    from collections import defaultdict
    groups = defaultdict(list)
    for q, c, pk, rho, p in results:
        groups[(q, c)].append((pk, rho, p))
    for (q, c), rows in groups.items():
        top = sorted(rows, key=lambda r: -abs(r[1]))[:3]
        parts = ", ".join(f"{pk} ρ={rho:+.2f}(p={p:.1g})"
                          for pk, rho, p in top)
        print(f"  {q:<6s} {c:<7s}: {parts}")


if __name__ == "__main__":
    main()
