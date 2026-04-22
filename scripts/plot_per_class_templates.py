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
sys.path.insert(0, str(ROOT))
from hcd_analysis.hcd_template import template_contributions, fit_alpha
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


# k-axis range in PRIYA angular convention (rad·s/km).
# k_cyc = k_ang / (2π).  emulator-relevant range 0.009 → 0.2 rad·s/km.
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20

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

    # Map from sim class → (axis, Rogers-template class key)
    cls_map = [
        (axes[0], "LLS",    "LLS"),
        (axes[1], "subDLA", "Sub-DLA"),
        (axes[2], "DLA",    "DLA_sum"),  # sum of Small-DLA + Large-DLA
    ]

    # Pre-fit α_full = (α_LLS, α_Sub, α_Small, α_Large) per z snapshot
    # by least-squares against the measured ratios across all three
    # per-class curves simultaneously — for the overlay we then show
    # the fitted Rogers curve as a black dashed line at each z.
    # (Cheap per-sim fit; see docs/fast_mode_physics.md and hcd_template.py.)
    alpha_by_z = {}
    for h5 in h5s:
        d = load_per_class(h5)
        z = float(d["attrs"]["z"])
        k_cyc = d["k"]; k_ang = 2.0 * np.pi * k_cyc
        # Combine three class templates into one ratio to fit α.
        # Use clean subset as reference; stack (LLS, subDLA, DLA) in series.
        # Since P_i = (1 + α_i · f_z · g_i) · P_clean, fit α_LLS to LLS curve alone,
        # α_sub to subDLA curve, and (α_Small + α_Large) collectively on DLA curve.
        # Simplest acceptable fit: run four separate α via class→index map.
        #   P_LLS / P_clean       → α_LLS
        #   P_subDLA / P_clean    → α_Sub-DLA
        #   P_DLA_only / P_clean  → α_Small + α_Large (under-constrained, but
        #                            we just want a visual overlay, so fit
        #                            both freely and report the total).
        sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX) & np.isfinite(ratio(d, "LLS"))
        if not sel.any():
            continue
        k_sel = k_ang[sel]
        alpha_best = np.zeros(4)
        # LLS
        res = fit_alpha(k_sel, ratio(d, "LLS")[sel],
                        np.ones_like(k_sel), z=z,
                        alpha0=np.array([0.1, 0, 0, 0]),
                        bounds=(np.array([0,0,0,0]), np.array([5, 0.001, 0.001, 0.001])))
        alpha_best[0] = res["alpha"][0]
        # Sub-DLA
        res = fit_alpha(k_sel, ratio(d, "subDLA")[sel],
                        np.ones_like(k_sel), z=z,
                        alpha0=np.array([0, 0.1, 0, 0]),
                        bounds=(np.array([0,0,0,0]), np.array([0.001, 5, 0.001, 0.001])))
        alpha_best[1] = res["alpha"][1]
        # DLA — split Small and Large freely
        res = fit_alpha(k_sel, ratio(d, "DLA")[sel],
                        np.ones_like(k_sel), z=z,
                        alpha0=np.array([0, 0, 0.1, 0.1]),
                        bounds=(np.array([0,0,0,0]), np.array([0.001, 0.001, 5, 5])))
        alpha_best[2] = res["alpha"][2]; alpha_best[3] = res["alpha"][3]
        alpha_by_z[z] = alpha_best

    for ax, cls, rogers_key in cls_map:
        for h5 in h5s:
            d = load_per_class(h5)
            z = float(d["attrs"]["z"])
            k_cyc = d["k"]
            k_ang = 2.0 * np.pi * k_cyc  # PRIYA convention
            r = ratio(d, cls)
            sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX)
            ax.plot(k_ang[sel], r[sel], lw=1.0, color=z_to_color[z], label=f"z={z:.1f}", alpha=0.85)

        # Rogers template overlay at the median z (z=3) with best-fit α.
        median_z = sorted(alpha_by_z.keys())[len(alpha_by_z)//2] if alpha_by_z else None
        if median_z is not None:
            k_ang_grid = np.logspace(np.log10(K_ANG_MIN), np.log10(K_ANG_MAX), 200)
            alpha = alpha_by_z[median_z]
            contribs = template_contributions(k_ang_grid, median_z, alpha)
            if rogers_key == "DLA_sum":
                curve = 1 + (contribs["Small-DLA"] - 1) + (contribs["Large-DLA"] - 1)
                lbl = (f"Rogers α fit, z={median_z:.1f}\n"
                       f"α_Small={alpha[2]:.3f}  α_Large={alpha[3]:.3f}")
            else:
                curve = contribs[rogers_key]
                lbl = f"Rogers α fit, z={median_z:.1f}  α={alpha[{'LLS':0,'Sub-DLA':1}[rogers_key]]:.3f}"
            ax.plot(k_ang_grid, curve, "k--", lw=1.8, label=lbl)

        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k [rad·s/km]  (PRIYA angular convention)")
        ax.set_ylabel(f"P_{cls}_only / P_clean")
        ax.set_title(f"{cls}-only sightlines  vs  clean")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.5, 8.0)
        ax.set_yscale("log")
        ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
        # annotate 2π/b
        ax.axvline(2 * np.pi * (1.0 / 30.0), color="gray", ls=":", alpha=0.6)
        if ax is axes[0]:
            ax.text(2 * np.pi / 30.0 * 0.95, 6.0,
                     "k = 2π/b\n(b=30 km/s)", fontsize=7, ha="right", color="gray")
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = list(dict(zip(labels, handles)).items())
        # Keep Rogers entry visible; z legend only on rightmost panel
        if ax is axes[-1]:
            ax.legend([h for _, h in uniq], [l for l, _ in uniq],
                       fontsize=6, loc="upper right", ncol=2)
        else:
            # just show Rogers label
            rogers_h = [h for l, h in uniq if "Rogers" in l]
            rogers_l = [l for l, _ in uniq if "Rogers" in l]
            if rogers_h:
                ax.legend(rogers_h, rogers_l, fontsize=7, loc="upper right")
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
            k_cyc = d["k"]
            k_ang = 2.0 * np.pi * k_cyc
            r = ratio(d, cls)
            sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX)
            ax.plot(k_ang[sel], r[sel], lw=0.8, alpha=0.6,
                     color=cmap(norm(params.get("Ap", 1e-9))))
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.axvline(2 * np.pi * (1.0 / 30.0), color="gray", ls=":", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("k [rad·s/km]  (PRIYA angular convention)")
        ax.set_ylabel(f"P_{cls}_only / P_clean")
        ax.set_title(f"{cls}-only, all LF sims at z≈3")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.5, 3.0)
        ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes.tolist(), label=r"$A_p$ (initial power amplitude)", shrink=0.8)
    fig.suptitle("Per-class P_dirty/P_clean across LF parameter space at z≈3")
    outp = OUT / "per_class_ratio_vs_sim.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")
