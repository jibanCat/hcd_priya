"""
Matched-pair HR vs LF comparison for dN/dX and Ω_HI per class.

Shows each of the 3 common (sim, z) pairs side-by-side rather than
suite-averaged — this isolates the pure resolution effect from any
parameter drift between the HR and LF suites (the HR suite has only
4 sims, so its ensemble average weights different parameters than
the 60-LF average).

Outputs
-------
  figures/analysis/matched_pair_dndx_hr_vs_lf.png   (3 classes × z evolution)
  figures/analysis/matched_pair_omega_hi_hr_vs_lf.png
  figures/analysis/matched_pair_hr_vs_lf_table.csv

Run:
    python3 scripts/matched_pair_hr_vs_lf.py
"""
from __future__ import annotations

import csv
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
LF = DATA / "hcd_summary_lf.h5"
HR = DATA / "hcd_summary_hr.h5"

Z_TOL = 0.05
CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}


def _load(path):
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        for i in range(len(sims)):
            r = {"z": float(zs[i]),
                 "Ap": float(f["params/Ap"][i]),
                 "ns": float(f["params/ns"][i])}
            for cls in CLASSES:
                r[f"dndx_{cls}"] = float(f[f"dndx/{cls}"][i])
                r[f"omega_{cls}"] = float(f[f"Omega_HI/{cls}"][i])
                r[f"counts_{cls}"] = int(f[f"counts/{cls}"][i])
            out[(sims[i], round(float(zs[i]), 3))] = r
    return out


def _pair(hr, lf):
    lf_by_sim = defaultdict(list)
    for (sim, _), r in lf.items():
        lf_by_sim[sim].append(r)
    pairs = []   # list of (sim, z_hr, lf_rec, hr_rec)
    for (sim, _), h in hr.items():
        if sim not in lf_by_sim:
            continue
        best = min(lf_by_sim[sim], key=lambda r: abs(r["z"] - h["z"]))
        if abs(best["z"] - h["z"]) <= Z_TOL:
            pairs.append((sim, h["z"], best, h))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


# Obs for DLA-panel annotation
PW09_DLA = np.array([
    [2.2, 2.4, 0.048, 0.006], [2.4, 2.7, 0.055, 0.005],
    [2.7, 3.0, 0.067, 0.006], [3.0, 3.5, 0.084, 0.006],
    [3.5, 4.0, 0.075, 0.009], [4.0, 5.5, 0.106, 0.018],
])
HO21_DNDX_Z = np.array([
    2.083, 2.250, 2.417, 2.583, 2.750, 2.917, 3.083, 3.250, 3.417, 3.583,
    3.750, 3.917, 4.083, 4.250, 4.417, 4.583, 4.750, 4.917,
])
HO21_DNDX_MEDIAN = np.array([
    0.0337, 0.0430, 0.0462, 0.0494, 0.0622, 0.0664, 0.0706, 0.0748, 0.0763,
    0.0777, 0.0630, 0.0646, 0.0577, 0.0725, 0.1015, 0.0821, 0.1033, 0.0674,
])


def plot_matched(pairs, prefix, ylabel_fmt, outpath, log=True):
    """One row × 3 columns.  Each column = one class.  Lines = one per sim."""
    sims = sorted({p[0] for p in pairs})
    sim_colors = {s: f"C{i}" for i, s in enumerate(sims)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, cls in zip(axes, CLASSES):
        for sim in sims:
            sim_pairs = [p for p in pairs if p[0] == sim]
            sim_pairs.sort(key=lambda p: p[1])
            z = np.array([p[1] for p in sim_pairs])
            y_lf = np.array([p[2][f"{prefix}_{cls}"] for p in sim_pairs])
            y_hr = np.array([p[3][f"{prefix}_{cls}"] for p in sim_pairs])
            Ap_val = sim_pairs[0][2]["Ap"]
            ns_val = sim_pairs[0][2]["ns"]
            col = sim_colors[sim]
            label = f"A_p={Ap_val:.2e}, n_s={ns_val:.2f}"
            ax.plot(z, y_lf, "o-", ms=4, color=col, alpha=0.5,
                    label=f"LF • {label}")
            ax.plot(z, y_hr, "s--", ms=7, color=col, mec="k",
                    label=f"HR • {label}")
        # DLA-panel obs overlay
        if cls == "DLA":
            z_mid = 0.5 * (PW09_DLA[:, 0] + PW09_DLA[:, 1])
            z_err = 0.5 * (PW09_DLA[:, 1] - PW09_DLA[:, 0])
            ax.errorbar(z_mid, PW09_DLA[:, 2], xerr=z_err, yerr=PW09_DLA[:, 3],
                        fmt="D", color="black", ms=5, capsize=2,
                        label="PW09", alpha=0.8)
            ax.errorbar(HO21_DNDX_Z, HO21_DNDX_MEDIAN, fmt="d",
                        color="dimgray", ms=4, alpha=0.8, label="Ho+21")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_ylabel(ylabel_fmt.format(cls=cls))
        ax.set_title(f"{cls}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Matched-pair {prefix.replace('dndx','dN/dX').replace('omega','Ω_HI')} "
        f"per class:  3 common sims, LF (solid-circle) vs HR (dashed-square)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=120); plt.close(fig)


def main():
    hr = _load(HR)
    lf = _load(LF)
    pairs = _pair(hr, lf)
    n_sims = len({p[0] for p in pairs})
    print(f"Matched pairs: {len(pairs)} pairs across {n_sims} common sims")

    plot_matched(
        pairs, "dndx", "dN/dX ({cls})",
        OUT / "matched_pair_dndx_hr_vs_lf.png", log=True,
    )
    print(f"  wrote {OUT/'matched_pair_dndx_hr_vs_lf.png'}")

    plot_matched(
        pairs, "omega", "Ω_HI ({cls})",
        OUT / "matched_pair_omega_hi_hr_vs_lf.png", log=True,
    )
    print(f"  wrote {OUT/'matched_pair_omega_hi_hr_vs_lf.png'}")

    # CSV table of R = Q_HR / Q_LF per (sim, z, quantity, class)
    csv_p = DATA / "matched_pair_hr_vs_lf_table.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sim", "z", "Ap", "ns",
                    "dndx_LLS_LF", "dndx_LLS_HR", "R_dndx_LLS",
                    "dndx_subDLA_LF", "dndx_subDLA_HR", "R_dndx_subDLA",
                    "dndx_DLA_LF", "dndx_DLA_HR", "R_dndx_DLA",
                    "omega_LLS_LF", "omega_LLS_HR", "R_omega_LLS",
                    "omega_subDLA_LF", "omega_subDLA_HR", "R_omega_subDLA",
                    "omega_DLA_LF", "omega_DLA_HR", "R_omega_DLA"])
        for sim, z_hr, lr, hr_ in pairs:
            row = [sim, round(z_hr, 3), lr["Ap"], lr["ns"]]
            for prefix in ["dndx", "omega"]:
                for cls in CLASSES:
                    lval = lr[f"{prefix}_{cls}"]
                    hval = hr_[f"{prefix}_{cls}"]
                    R = hval / lval if (lval > 0 and np.isfinite(lval)) else np.nan
                    row += [lval, hval, R]
            w.writerow(row)
    print(f"  wrote {csv_p}")


if __name__ == "__main__":
    main()
