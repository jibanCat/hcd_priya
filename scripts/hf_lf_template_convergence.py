"""
HR/LF convergence of the Rogers+2018 per-class templates
T_class(k, z) ≡ P_class_only(k, z) / P_clean(k, z).

For each of the 3 common sims (HR, LF) × z matched within |Δz|≤0.05,
we load `p1d_per_class.h5` on both sides and compute

    R_class(k, z, sim) = T_class_HR(k, z) / T_class_LF(k, z)

The key MF question: is R_class(k, z) approximately the same for all
3 sims, or does it vary with input parameters?  If the former, a
fixed HR/LF correction curve at each z suffices.  If the latter,
an MF emulator (analytical or GP) over the template is warranted.

Outputs
-------
  figures/analysis/hf_lf_template_vs_k_and_z.png  — ratio vs k, colour by z
  figures/analysis/hf_lf_template_sim_spread.png  — per-sim spread vs k at z≈3
  figures/analysis/mf_necessity_templates.csv     — per-class MF verdicts

Run:
    python3 scripts/hf_lf_template_convergence.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "analysis" / "04_hcd_mf"
DATA = ROOT / "figures" / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")

CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}

# PRIYA angular k range
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20


def _load_percclass(p: Path):
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return {
            "z": float(f.attrs["z"]),
            "k": f["k"][:],
            "P_clean": f["P_clean"][:],
            "P_LLS_only": f["P_LLS_only"][:],
            "P_subDLA_only": f["P_subDLA_only"][:],
            "P_DLA_only": f["P_DLA_only"][:],
            "n_clean": int(f["n_sightlines_clean"][()]),
            "n_LLS": int(f["n_sightlines_LLS"][()]),
            "n_subDLA": int(f["n_sightlines_subDLA"][()]),
            "n_DLA": int(f["n_sightlines_DLA"][()]),
        }


def _enumerate_snaps(sim_dir: Path):
    """Return {snap_name: (path, z)} for snaps with per-class h5 present."""
    out = {}
    for p in sorted(sim_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("snap_"):
            continue
        pc = p / "p1d_per_class.h5"
        meta = p / "meta.json"
        if not (pc.exists() and meta.exists() and (p / "done").exists()):
            continue
        try:
            z = float(json.load(open(meta))["z"])
        except Exception:
            continue
        out[p.name] = (p, z)
    return out


def build_pairs(z_tol=0.05):
    """For each common sim, z-match HR snap → nearest LF snap."""
    lf_root = SCRATCH
    hr_root = SCRATCH / "hires"
    lf_sims = {p.name for p in lf_root.iterdir()
               if p.is_dir() and p.name.startswith("ns")}
    hr_sims = {p.name for p in hr_root.iterdir()
               if p.is_dir() and p.name.startswith("ns")}
    common = sorted(lf_sims & hr_sims)

    pairs = []   # (sim, z_hr, lf_snapdir, hr_snapdir)
    for sim in common:
        lf_snaps = _enumerate_snaps(lf_root / sim)
        hr_snaps = _enumerate_snaps(hr_root / sim)
        for hr_name, (hr_dir, z_hr) in hr_snaps.items():
            best_lf = None
            best_dz = 1e9
            for lf_name, (lf_dir, z_lf) in lf_snaps.items():
                dz = abs(z_lf - z_hr)
                if dz < best_dz:
                    best_dz = dz
                    best_lf = (lf_dir, z_lf)
            if best_lf is None or best_dz > z_tol:
                continue
            pairs.append((sim, z_hr, best_lf[0], hr_dir))
    return pairs


def compute_template_ratios(pairs):
    """
    Returns nested dict:
       R[sim][cls] = sorted list of (z, k, R_k)  — common-k truncated.
    """
    R = defaultdict(lambda: defaultdict(list))
    for sim, z_hr, lf_dir, hr_dir in pairs:
        lf = _load_percclass(lf_dir / "p1d_per_class.h5")
        hr = _load_percclass(hr_dir / "p1d_per_class.h5")
        if lf is None or hr is None:
            continue

        # Truncate to common k (match indices by proximity; both grids start at 0
        # with same dv_kms so they are the same for the overlapping range).
        nk = min(len(lf["k"]), len(hr["k"]))
        k = lf["k"][:nk]
        # Keep only finite, positive k in the PRIYA target range (angular):
        k_ang = 2 * np.pi * k
        sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX) & (k > 0)
        if not sel.any():
            continue

        for cls in CLASSES:
            Pc_lf = lf["P_clean"][:nk][sel]
            Pc_hr = hr["P_clean"][:nk][sel]
            Pcl_lf = lf[f"P_{cls}_only"][:nk][sel]
            Pcl_hr = hr[f"P_{cls}_only"][:nk][sel]
            # Only where both are positive
            mask = (Pc_lf > 0) & (Pc_hr > 0) & (Pcl_lf > 0) & (Pcl_hr > 0)
            if not mask.any():
                continue
            T_lf = Pcl_lf[mask] / Pc_lf[mask]
            T_hr = Pcl_hr[mask] / Pc_hr[mask]
            R_k = T_hr / T_lf
            R[sim][cls].append((z_hr, k[sel][mask], R_k))
    return R


def plot_R_vs_k_colored_by_z(R, outpath):
    """One row per class, one col per sim.  Lines = z, colored plasma."""
    sims = sorted(R.keys())
    n_sim = len(sims)
    fig, axes = plt.subplots(3, n_sim, figsize=(4 * n_sim + 2, 10), sharex=True,
                             sharey="row")
    for j, sim in enumerate(sims):
        # gather all z's for this sim
        zs = sorted({z for cls in CLASSES for z, _, _ in R[sim][cls]})
        norm = plt.Normalize(vmin=min(zs), vmax=max(zs))
        cmap = plt.cm.plasma
        for i, cls in enumerate(CLASSES):
            ax = axes[i, j] if n_sim > 1 else axes[i]
            for z, k, r in sorted(R[sim][cls], key=lambda t: t[0]):
                ax.plot(2 * np.pi * k, r, lw=1.0, alpha=0.75,
                        color=cmap(norm(z)))
            ax.axhline(1.0, color="k", lw=0.5, ls="--")
            ax.set_xscale("log")
            ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"{sim[:18]}…", fontsize=8)
            if i == 2:
                ax.set_xlabel("k_ang  [rad·s/km]")
            if j == 0:
                ax.set_ylabel(f"R_{cls}(k,z) = T_{cls}^HR / T_{cls}^LF")
        # single colorbar per sim — add at the bottom of each column
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = plt.colorbar(sm, ax=[axes[ii, j] for ii in range(3)]
                            if n_sim > 1 else axes,
                            fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label("z", fontsize=8)
    fig.suptitle(
        "HR/LF ratio of per-class templates   T_class ≡ P_class_only / P_clean",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sim_spread_at_z3(R, outpath):
    """At z≈3, overlay the 3 sims' R(k) per class to eye the parameter dependence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    sims = sorted(R.keys())
    sim_colors = {s: f"C{i}" for i, s in enumerate(sims)}
    for ax, cls in zip(axes, CLASSES):
        for sim in sims:
            for z, k, r in R[sim][cls]:
                if abs(z - 3.0) < 0.1:
                    ax.plot(2 * np.pi * k, r, lw=1.2, color=sim_colors[sim],
                            label=(sim[:18] + "…") if ax is axes[0] else None)
                    break
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
        ax.set_title(f"{cls}: HR/LF template ratio at z≈3")
        ax.set_xlabel("k_ang [rad·s/km]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("R(k) = T_HR / T_LF")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Per-sim spread in HR/LF template ratio at z ≈ 3")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120); plt.close(fig)


def mf_necessity_templates(R):
    """
    For each class, at each z, compute σ_sim(k) across the 3 sims and
    compare to the in-z running variability of a single sim as an
    approximate noise floor.  If σ_sim / σ_k_within is > threshold,
    recommend MF.

    Also report the peak relative spread across sims: max_k |R_max - R_min| / <R>.
    """
    verdicts = []
    for cls in CLASSES:
        # collect R(k, z, sim) onto a common grid per z
        by_z = defaultdict(list)   # z → list of (sim, k, r)
        for sim, classes in R.items():
            for z, k, r in classes[cls]:
                by_z[round(z, 3)].append((sim, k, r))
        worst_spread = 0.0
        zs_used = []
        spread_per_z = []
        for zv, triples in by_z.items():
            if len({s for s, _, _ in triples}) < 2:
                continue
            # Interpolate each onto the first sim's k (all share common grids)
            k_ref = triples[0][1]
            ys = []
            for sim, k, r in triples:
                ys.append(np.interp(k_ref, k, r))
            ys = np.array(ys)
            # spread per k:
            R_range = np.max(ys, axis=0) - np.min(ys, axis=0)
            R_mean = np.mean(ys, axis=0)
            frac = R_range / np.where(R_mean > 0, R_mean, np.nan)
            peak = float(np.nanmax(frac))
            spread_per_z.append(peak)
            zs_used.append(zv)
            worst_spread = max(worst_spread, peak)

        # Summary: what's the median fractional spread across z?
        typ = float(np.median(spread_per_z)) if spread_per_z else float("nan")
        if worst_spread > 0.03:
            verdict = "MF recommended"
        elif worst_spread > 0.01:
            verdict = "borderline"
        else:
            verdict = "flat template-correction suffices"
        verdicts.append({
            "class": cls,
            "n_z": len(zs_used),
            "median_frac_spread_across_sims": typ,
            "peak_frac_spread_across_sims": worst_spread,
            "verdict": verdict,
        })
    return verdicts


def main():
    print("Building HR/LF pairs for per-class templates…")
    pairs = build_pairs()
    print(f"  {len(pairs)} pairs across {len({p[0] for p in pairs})} common sims")

    R = compute_template_ratios(pairs)
    n_z_per_sim = {s: len(R[s]["DLA"]) for s in R}
    print(f"  sims with DLA template data: {n_z_per_sim}")

    plot_R_vs_k_colored_by_z(R, OUT / "hf_lf_template_vs_k_and_z.png")
    print(f"  wrote {OUT/'hf_lf_template_vs_k_and_z.png'}")
    plot_sim_spread_at_z3(R, OUT / "hf_lf_template_sim_spread.png")
    print(f"  wrote {OUT/'hf_lf_template_sim_spread.png'}")

    verdicts = mf_necessity_templates(R)
    print("\nMF necessity per per-class template:")
    print(f"  {'class':<8s} {'nz':>3s} {'<spread%>':>10s} {'peak%':>8s}  verdict")
    for v in verdicts:
        print(f"  {v['class']:<8s} {v['n_z']:>3d} "
              f"{v['median_frac_spread_across_sims']*100:>9.1f}%"
              f"{v['peak_frac_spread_across_sims']*100:>8.1f}% "
              f" {v['verdict']}")

    csv_path = DATA / "mf_necessity_templates.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n_z_bins", "median_frac_spread", "peak_frac_spread", "verdict"])
        for v in verdicts:
            w.writerow([v["class"], v["n_z"],
                        v["median_frac_spread_across_sims"],
                        v["peak_frac_spread_across_sims"],
                        v["verdict"]])
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
