"""
Alternative dN/dX figure that avoids mean-regression — plot all 60 LF
sims' dN/dX(z) as thin spaghetti lines and highlight the two extreme
sims (lowest and highest median dN/dX over z ∈ [2, 4]) so the shape of
individual simulations is visible.

Rationale: showing the mean ± 1σ across 60 sims washes out curvature
that is present in any single simulation (since the peak/plateau of
dN/dX(z) sits at different z for different parameter choices).  The
two extreme sims bracket the suite envelope and keep their own
redshift-dependent shape.

Output:
    figures/analysis/01_catalog_obs/dndx_extreme_sims_per_class.png

Run:
    python3 scripts/plot_dndx_extreme_sims.py
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
LF_H5 = DATA / "hcd_summary_lf.h5"
OUT = ROOT / "figures" / "analysis" / "01_catalog_obs"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COL = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}

# Fiducial cosmology for dN/dz ↔ dN/dX conversion of LLS obs
OMEGA_M = 0.315
OMEGA_L = 0.685

# Range of z used to rank sims (pick the range where every sim has snaps
# and where observational constraints are strongest)
Z_RANK_LO, Z_RANK_HI = 2.0, 4.0


# -----------------------------------------------------------------------------
# Observations (same tables as plot_hcd_vs_obs_with_hr.py)
# -----------------------------------------------------------------------------
PW09_DLA = np.array([
    [2.2, 2.4, 0.048, 0.006], [2.4, 2.7, 0.055, 0.005],
    [2.7, 3.0, 0.067, 0.006], [3.0, 3.5, 0.084, 0.006],
    [3.5, 4.0, 0.075, 0.009], [4.0, 5.5, 0.106, 0.018],
])
N12_Z     = np.array([2.15, 2.45, 2.75, 3.05, 3.35])
N12_DNDZ  = np.array([0.20, 0.20, 0.25, 0.29, 0.36])
N12_DZDX  = np.array([3690/11625., 4509/14841., 2867/9900., 1620/5834., 789/2883.])
N12_DLA   = np.column_stack([N12_Z, N12_DNDZ * N12_DZDX])
HO21_Z = np.array([
    2.083, 2.250, 2.417, 2.583, 2.750, 2.917, 3.083, 3.250, 3.417, 3.583,
    3.750, 3.917, 4.083, 4.250, 4.417, 4.583, 4.750, 4.917,
])
HO21_MED = np.array([
    0.0337, 0.0430, 0.0462, 0.0494, 0.0622, 0.0664, 0.0706, 0.0748, 0.0763,
    0.0777, 0.0630, 0.0646, 0.0577, 0.0725, 0.1015, 0.0821, 0.1033, 0.0674,
])
HO21_LO = np.array([
    0.0330, 0.0421, 0.0452, 0.0482, 0.0607, 0.0647, 0.0685, 0.0722, 0.0729,
    0.0736, 0.0584, 0.0583, 0.0503, 0.0637, 0.0888, 0.0684, 0.0812, 0.0506,
])
HO21_HI = np.array([
    0.0345, 0.0438, 0.0472, 0.0506, 0.0637, 0.0682, 0.0729, 0.0777, 0.0800,
    0.0822, 0.0687, 0.0717, 0.0666, 0.0857, 0.1205, 0.1049, 0.1402, 0.1180,
])


def dXdz(z):
    return (1 + z) ** 2 / np.sqrt(OMEGA_M * (1 + z) ** 3 + OMEGA_L)


def load_lofz_literature():
    """Parse ~/data/lofz_literature.txt → list (z_lo, z_hi, lz, up, down, ref)."""
    path = Path.home() / "data" / "lofz_literature.txt"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                z_lo, z_hi, lz, up, dw = map(float, parts[:5])
                ref = int(parts[5])
                rows.append((z_lo, z_hi, lz, up, dw, ref))
            except ValueError:
                continue
    return rows


def main():
    print("Loading hcd_summary_lf.h5…")
    with h5py.File(LF_H5, "r") as f:
        sim = np.array([s.decode() for s in f["sim"][:]])
        z = f["z"][:]
        dndx = {cls: f[f"dndx/{cls}"][:] for cls in ["LLS", "subDLA", "DLA"]}
    sims_unique = sorted(np.unique(sim))
    print(f"  {len(sims_unique)} unique LF sims, {len(z)} records")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    lofz = load_lofz_literature()

    for ax, cls in zip(axes, ["LLS", "subDLA", "DLA"]):
        col = CLASS_COL[cls]

        # Per-sim (z, dndx) sorted by z
        sim_curves = {}
        for s in sims_unique:
            m = sim == s
            zz = z[m]
            dd = dndx[cls][m]
            order = np.argsort(zz)
            sim_curves[s] = (zz[order], dd[order])

        # Rank sims by median dN/dX in the ranking z-range
        rank_scores = {}
        for s, (zz, dd) in sim_curves.items():
            sel = (zz >= Z_RANK_LO) & (zz <= Z_RANK_HI) & np.isfinite(dd)
            if sel.any():
                rank_scores[s] = float(np.median(dd[sel]))
        if not rank_scores:
            ax.set_title(f"{cls}: no data")
            continue
        s_min = min(rank_scores, key=rank_scores.get)
        s_max = max(rank_scores, key=rank_scores.get)

        # Spaghetti underlay: every sim in thin light colour
        for s, (zz, dd) in sim_curves.items():
            ax.plot(zz, dd, "-", color=col, lw=0.6, alpha=0.18)

        # Highlighted extremes
        zz, dd = sim_curves[s_max]
        ax.plot(zz, dd, "-", color=col, lw=2.2,
                label=f"max-median sim  (score={rank_scores[s_max]:.3g})")
        zz, dd = sim_curves[s_min]
        ax.plot(zz, dd, "--", color=col, lw=2.2,
                label=f"min-median sim  (score={rank_scores[s_min]:.3g})")

        # Observational overlays
        if cls == "LLS" and lofz:
            # Convert dN/dz → dN/dX with fiducial cosmology
            colors = plt.cm.tab10(np.linspace(0, 1, 5))
            refs_plotted = set()
            for z_lo, z_hi, lz, up, dw, ref in lofz:
                z_mid = 0.5 * (z_lo + z_hi)
                dx = dXdz(z_mid)
                y = lz / dx
                ax.errorbar(z_mid, y, xerr=0.5 * (z_hi - z_lo),
                            yerr=[[dw / dx], [up / dx]],
                            fmt="D", color=colors[ref % len(colors)],
                            ms=5, capsize=3, alpha=0.85,
                            label=f"LLS obs ref {ref}" if ref not in refs_plotted else None)
                refs_plotted.add(ref)
        elif cls == "DLA":
            z_mid = 0.5 * (PW09_DLA[:, 0] + PW09_DLA[:, 1])
            z_err = 0.5 * (PW09_DLA[:, 1] - PW09_DLA[:, 0])
            ax.errorbar(z_mid, PW09_DLA[:, 2], xerr=z_err, yerr=PW09_DLA[:, 3],
                        fmt="s", color="black", ms=6, capsize=3,
                        label="PW09 (SDSS DR5)")
            ax.errorbar(N12_DLA[:, 0], N12_DLA[:, 1], xerr=0.15,
                        fmt="^", color="dimgray", ms=6, capsize=3,
                        label="N12 (BOSS DR9)")
            ax.errorbar(HO21_Z, HO21_MED, yerr=[HO21_MED - HO21_LO, HO21_HI - HO21_MED],
                        fmt="D", color="darkslategray", ms=5, capsize=3,
                        alpha=0.85, label="Ho+21 (SDSS DR16)")

        ax.set_yscale("log")
        ax.set_xlabel("z"); ax.set_ylabel(f"dN/dX ({cls})")
        ax.grid(alpha=0.3)
        ax.set_title(f"{cls}  —  60 sims (spaghetti) + extremes + obs", fontsize=11)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "dN/dX per class — individual-sim view\n"
        "Thin lines = all 60 LF sims; bold solid/dashed = max/min median-dN/dX sims "
        f"(ranked on z ∈ [{Z_RANK_LO:.1f}, {Z_RANK_HI:.1f}]); markers = observations",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outp = OUT / "dndx_extreme_sims_per_class.png"
    fig.savefig(outp, dpi=130); plt.close(fig)
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
