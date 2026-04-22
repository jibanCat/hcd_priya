"""
Per-class P_dirty/P_clean template figures for docs/analysis.md.

Reads p1d_per_class.h5 files on /scratch/.../hcd_outputs/ and produces:

  per_class_ratio_vs_z.png     one sim (ns0.803), every z-snap, curves
                               P_LLS/P_clean, P_subDLA/P_clean, P_DLA/P_clean

  per_class_ratio_vs_sim.png   one z (snap_017 z=3), every LF sim for which
                               p1d_per_class.h5 exists, same three ratios
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
OUTPUT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT = ROOT / "figures" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"


def load_per_class(h5_path):
    with h5py.File(h5_path, "r") as f:
        d = {k: f[k][...] for k in f.keys()}
        d["attrs"] = dict(f.attrs)
    return d


def ratio(d, cls):
    """P_<cls>_only / P_clean."""
    num = d[f"P_{cls}_only"]; den = d["P_clean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


# ---- figure 1: ratio vs z for ns0.803 ----
sim_dir = OUTPUT / TARGET_SIM
h5s = sorted(sim_dir.glob("snap_*/p1d_per_class.h5"))
print(f"Found {len(h5s)} snaps for {TARGET_SIM[:30]}...")

if h5s:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    zs = []
    for h5 in h5s:
        d = load_per_class(h5)
        z = float(d["attrs"]["z"])
        zs.append(z)
    zs_sorted = sorted(zs)
    cmap = plt.cm.plasma(np.linspace(0, 1, len(h5s)))
    z_to_color = {z: cmap[i] for i, z in enumerate(zs_sorted)}

    for ax, cls, color_base in [(axes[0], "LLS", "C2"),
                                  (axes[1], "subDLA", "C1"),
                                  (axes[2], "DLA", "C3")]:
        for h5 in h5s:
            d = load_per_class(h5)
            z = float(d["attrs"]["z"])
            k = d["k"]
            r = ratio(d, cls)
            ax.plot(k[1:], r[1:], lw=1.2, color=z_to_color[z], label=f"z={z:.1f}")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km, cyclic]")
        ax.set_ylabel(f"P_{cls}_only / P_clean")
        ax.set_title(f"{cls}-only sightlines  vs  clean")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.5, 3.0)
        ax.set_xlim(1e-3, 5e-2)
        if ax is axes[-1]:
            # shrink legend
            handles, labels = ax.get_legend_handles_labels()
            # dedupe
            uniq = list(dict(zip(labels, handles)).items())
            ax.legend([h for _, h in uniq], [l for l, _ in uniq],
                       fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(f"Per-class P_dirty/P_clean across z for sim: {TARGET_SIM[:40]}...")
    fig.tight_layout()
    outp = OUT / "per_class_ratio_vs_z.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")

# ---- figure 2: ratio vs sim at z=3 (snap_017) ----
sim_files = list(OUTPUT.glob("ns*/snap_017/p1d_per_class.h5"))
print(f"Found {len(sim_files)} sims with snap_017 per-class data")

if sim_files:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # Collect per-sim Ap for colour-coding
    from hcd_analysis.io import parse_sim_params
    colors = []; sim_entries = []
    for h5 in sim_files:
        sim_name = h5.parent.parent.name
        params = parse_sim_params(sim_name) or {}
        sim_entries.append((sim_name, params, h5))
    # sort by Ap
    sim_entries.sort(key=lambda x: x[1].get("Ap", 0))
    Aps = np.array([s[1].get("Ap", 1e-9) for s in sim_entries])
    norm = plt.Normalize(vmin=Aps.min(), vmax=Aps.max())
    cmap = plt.cm.viridis

    for ax, cls in zip(axes, ["LLS", "subDLA", "DLA"]):
        for sim_name, params, h5 in sim_entries:
            d = load_per_class(h5)
            k = d["k"]; r = ratio(d, cls)
            ax.plot(k[1:], r[1:], lw=0.8, alpha=0.6,
                     color=cmap(norm(params.get("Ap", 1e-9))))
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km, cyclic]")
        ax.set_ylabel(f"P_{cls}_only / P_clean")
        ax.set_title(f"{cls}-only, all LF sims at z≈3")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.5, 3.0)
        ax.set_xlim(1e-3, 5e-2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes.tolist(), label=r"$A_p$ (initial power amplitude)", shrink=0.8)
    fig.suptitle("Per-class P_dirty/P_clean across LF parameter space at z≈3")
    outp = OUT / "per_class_ratio_vs_sim.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")
