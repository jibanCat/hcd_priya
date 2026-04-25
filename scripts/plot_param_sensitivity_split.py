"""
Median-split parameter sensitivity grid, for dN/dX and Ω_HI per class.

For each of the 9 PRIYA input parameters θ, split the 60 LF sims at
the median of θ into two halves (high-θ ≥ median, low-θ < median)
and plot the average quantity vs z for each half.  Implicitly
marginalises over the other 8 dimensions.

Outputs 4 figures, grouped to keep each panel's dynamic range
reasonable (LLS / subDLA / DLA vary by ~10x within a single panel
otherwise):

  - param_sens_split_cosmo_dndx.png       4 rows (ns, Ap, h, Ωm h²) × 3 cols (LLS, subDLA, DLA)
  - param_sens_split_reion_astro_dndx.png 5 rows (herei, heref, hireionz, alphaq, bhfeedback) × 3 cols
  - param_sens_split_cosmo_omega_hi.png    same rows, for Ω_HI
  - param_sens_split_reion_astro_omega_hi.png  same

Each panel: high (solid) vs low (dashed) median-split; shaded = 1σ
across sims in the half.

Run:
    python3 scripts/plot_param_sensitivity_split.py
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

DATA = data_dir()
OUT = ROOT / "figures" / "analysis" / "02_param_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

LF = DATA / "hcd_summary_lf.h5"

# Grouped parameter keys — keeps each figure's dynamic range manageable.
PARAM_GROUPS = {
    "cosmo":       ["ns", "Ap", "hub", "omegamh2"],
    "reion_astro": ["herei", "heref", "hireionz", "alphaq", "bhfeedback"],
}
PARAM_LABELS = {
    "ns": r"$n_s$",
    "Ap": r"$A_p$",
    "hub": r"$h$",
    "omegamh2": r"$\Omega_m h^2$",
    "herei": r"HeII $z_i$",
    "heref": r"HeII $z_f$",
    "hireionz": r"HI reion $z$",
    "alphaq": r"$\alpha_q$ (QSO spectral index)",
    "bhfeedback": r"BH feedback",
}

CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}


def load_lf() -> dict:
    with h5py.File(LF, "r") as f:
        out = {
            "sim": np.array([s.decode() for s in f["sim"][:]]),
            "z":   f["z"][:],
        }
        for cls in CLASSES:
            out[f"dndx_{cls}"]  = f[f"dndx/{cls}"][:]
            out[f"omega_{cls}"] = f[f"Omega_HI/{cls}"][:]
        for g in PARAM_GROUPS.values():
            for pk in g:
                out[pk] = f[f"params/{pk}"][:]
    return out


def _half_masks(param_values: np.ndarray):
    uniq = np.unique(param_values[~np.isnan(param_values)])
    median = np.median(uniq)
    return param_values >= median, param_values < median, float(median)


def _avg_per_z(z, y, sel, z_bins):
    zc, m, s = [], [], []
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        mask = sel & (z >= lo) & (z < hi) & np.isfinite(y)
        if not mask.any():
            continue
        zc.append(0.5 * (lo + hi))
        m.append(float(np.mean(y[mask])))
        s.append(float(np.std(y[mask])))
    return np.array(zc), np.array(m), np.array(s)


def make_grid(d: dict, quantity: str, quantity_label: str,
              group: str, params: list[str], outpath: Path):
    """One figure: rows = params in this group, cols = 3 classes (LLS, subDLA, DLA)."""
    z_bins = np.arange(1.9, 5.7, 0.2)
    n_rows = len(params)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 2.9 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i, pk in enumerate(params):
        high, low, med = _half_masks(d[pk])
        n_high = len(np.unique(d["sim"][high]))
        n_low  = len(np.unique(d["sim"][low]))

        for j, cls in enumerate(CLASSES):
            ax = axes[i, j]
            c = CLASS_COLORS[cls]
            y = d[f"{quantity}_{cls}"]
            zc_hi, m_hi, s_hi = _avg_per_z(d["z"], y, high, z_bins)
            zc_lo, m_lo, s_lo = _avg_per_z(d["z"], y, low,  z_bins)

            ax.plot(zc_hi, m_hi, "-",  color=c, lw=1.8,
                    label=f"high (≥{med:.3g}, n={n_high})")
            ax.fill_between(zc_hi, m_hi - s_hi, m_hi + s_hi,
                            color=c, alpha=0.2)
            ax.plot(zc_lo, m_lo, "--", color=c, lw=1.8,
                    label=f"low  (<{med:.3g}, n={n_low})")
            ax.fill_between(zc_lo, m_lo - s_lo, m_lo + s_lo,
                            color=c, alpha=0.1, hatch="//", edgecolor=c, linewidth=0)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(cls, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{quantity_label}\n{PARAM_LABELS[pk]}  median={med:.3g}",
                              fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("z")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"{quantity_label}  —  median-split over {group.replace('_', ' + ')} "
        f"parameters (cosmology-only vs reion+astro in separate figures)\n"
        f"solid = high-θ half, dashed = low-θ half;  shaded = 1σ across sims in that half",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    print("Loading LF summary…")
    d = load_lf()
    print(f"  {len(np.unique(d['sim']))} unique sims, {len(d['z'])} (sim, z) records")

    # Delete old combined-class figures (classes with mismatched dynamic ranges)
    for legacy in ["param_sens_split_dndx.png", "param_sens_split_omega_hi.png"]:
        p = OUT / legacy
        if p.exists():
            p.unlink()
            print(f"  removed legacy figure {legacy}")

    for quantity, qlabel in [("dndx", "dN/dX"), ("omega", "Ω_HI")]:
        for group, params in PARAM_GROUPS.items():
            outpath = OUT / f"param_sens_split_{group}_{quantity}.png"
            print(f"Plotting {quantity} median-split for {group}…")
            make_grid(d, quantity, qlabel, group, params, outpath)
            print(f"  wrote {outpath}")


if __name__ == "__main__":
    main()
