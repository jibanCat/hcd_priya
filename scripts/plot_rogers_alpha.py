"""
Plot the fitted Rogers+2018 α per (sim, z) summary produced by
scripts/fit_rogers_alpha.py.

Two figures:
  - rogers_alpha_vs_z.png    — α_i(z) per class, LF suite mean + 1σ band,
                                HR suite mean overlaid.
  - rogers_alpha_chi2.png    — reduced-χ² distribution per z (sanity check).

Run:
    python3 scripts/plot_rogers_alpha.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir

SUMMARY = data_dir() / "rogers_alpha_summary.h5"
OUT = ROOT / "figures" / "analysis" / "03_templates_and_p1d"
OUT.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["α_LLS", "α_Sub", "α_Small", "α_Large"]
COMPONENT_COLORS = ["C2", "C1", "C0", "C3"]


def load():
    with h5py.File(SUMMARY, "r") as f:
        d = {
            "suite": np.array([s.decode() for s in f["suite"][:]]),
            "sim":   np.array([s.decode() for s in f["sim"][:]]),
            "snap":  np.array([s.decode() for s in f["snap"][:]]),
            "z":     f["z"][:],
            "alpha": f["alpha"][:],
            "alpha_err": f["alpha_err"][:],
            "chi2":  f["chi2"][:],
            "dof":   f["dof"][:],
        }
    return d


def _bin_by_z(z, y, z_bins, suite_mask):
    zc, m, s = [], [], []
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        sel = suite_mask & (z >= lo) & (z < hi) & np.isfinite(y)
        if not sel.any():
            continue
        zc.append(0.5 * (lo + hi))
        m.append(float(np.mean(y[sel])))
        s.append(float(np.std(y[sel])))
    return np.array(zc), np.array(m), np.array(s)


def plot_alpha_vs_z(d: dict, outpath: Path):
    z_bins = np.arange(1.9, 5.7, 0.2)
    is_lf = d["suite"] == "lf"
    is_hr = d["suite"] == "hr"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
    for i, (name, col) in enumerate(zip(COMPONENTS, COMPONENT_COLORS)):
        ax = axes[i]
        y = d["alpha"][:, i]
        # LF mean + 1σ across sims per z bin
        zc_lf, m_lf, s_lf = _bin_by_z(d["z"], y, z_bins, is_lf)
        ax.plot(zc_lf, m_lf, "o-", color=col, lw=1.8, ms=6,
                label="LF mean (60 sims)")
        ax.fill_between(zc_lf, m_lf - s_lf, m_lf + s_lf,
                        color=col, alpha=0.2, label="LF 1σ")
        # HR mean overlay
        zc_hr, m_hr, s_hr = _bin_by_z(d["z"], y, z_bins, is_hr)
        ax.errorbar(zc_hr, m_hr, yerr=s_hr, fmt="s--", color=col, mec="k",
                    ms=7, capsize=3, label="HR mean (4 sims)")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if i == 0:
            ax.set_ylabel(r"$\alpha$")

    fig.suptitle(
        "Rogers+2018 fitted α per class vs z  —  LF mean ± 1σ across sims, HR overlaid"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_reduced_chi2(d: dict, outpath: Path):
    red = d["chi2"] / np.maximum(d["dof"], 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(red[d["suite"] == "lf"], bins=40, alpha=0.6,
            label=f"LF (n={sum(d['suite']=='lf')})", color="C0",
            histtype="stepfilled")
    ax.hist(red[d["suite"] == "hr"], bins=40, alpha=0.6,
            label=f"HR (n={sum(d['suite']=='hr')})", color="C3",
            histtype="stepfilled")
    ax.axvline(1.0, color="k", lw=1.0, ls="--", label="χ²/DOF = 1")
    ax.set_xscale("log")
    ax.set_xlabel(r"reduced $\chi^2$")
    ax.set_ylabel("count")
    ax.set_title("Reduced χ² distribution for Rogers α fits across (sim, z)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    d = load()
    print(f"Loaded {len(d['z'])} rows (LF + HR)")
    plot_alpha_vs_z(d, OUT / "rogers_alpha_vs_z.png")
    print(f"  wrote {OUT / 'rogers_alpha_vs_z.png'}")
    plot_reduced_chi2(d, OUT / "rogers_alpha_chi2.png")
    print(f"  wrote {OUT / 'rogers_alpha_chi2.png'}")


if __name__ == "__main__":
    main()
