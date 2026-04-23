"""
Comprehensive PRIYA-vs-observation comparison, with the HR suite shown
alongside the LF suite to expose resolution effects.

Obs sources (all pulled from local tabulations, no fabrication):
  * DLA dN/dX  : PW09, N12, Ho21  (sbird/dla_data verbatim)
  * LLS dN/dz  : Prochaska+2010, Fumagalli+2013, Crighton+2018,
                  O'Meara+2013, Ribaudo+2011, BOSS (Crighton+2018)
                  — compilation at ~/data/lofz_literature.txt
  * Ω_subDLA(z) + Ω_DLA(z) : Berg+2019 (XQ-100) table B4,
                  ~/data/omega_hi_tableB4_full_sample.csv
                  (medians + 16/84 percentiles used as 1σ)

Conversion convention: LLS dN/dz is converted to dN/dX using
    dX/dz = (1+z)² · H₀ / H(z)      with H²(z)/H₀² = Ω_M(1+z)³ + Ω_Λ
using fiducial Planck cosmology (Ω_M = 0.315, Ω_Λ = 0.685) so that
PRIYA dN/dX (also in absorption-distance units) can be plotted on the
same axis as the LLS literature.

Outputs:
  figures/analysis/dndx_hr_vs_lf_vs_obs_per_class.png
  figures/analysis/omega_hi_hr_vs_lf_vs_obs.png

Run:
    python3 scripts/plot_hcd_vs_obs_with_hr.py
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "analysis" / "01_catalog_obs"
DATA = ROOT / "figures" / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
LF_H5 = DATA / "hcd_summary_lf.h5"
HR_H5 = DATA / "hcd_summary_hr.h5"

OMEGA_HI_CSV = Path.home() / "data" / "omega_hi_tableB4_full_sample.csv"
LOFZ_FILE = Path.home() / "data" / "lofz_literature.txt"
DLA_DATA_ROOT = Path("/home/mfho/DLA_data")

# PW09 Ω_HI(DLA) conversion constants (from sbird/dla_data omegahi_pro):
#   conv: 10⁸ M_sun/Mpc³ → g/cm³       [6.7699e-33]
#   ρ_crit(z=0): 9.3126e-30 g/cm³       [h=1 convention]
# PW09 rho_HI is pure HI mass (no helium correction).
PW09_CONV = 6.7699111782945424e-33
RHO_CRIT_0 = 9.3125685124148235e-30

# Noterdaeme+2012 Ω_HI^pure = published Ω_gas × 0.76 (remove helium,
# per sbird/dla_data omegahi_not docstring).
N12_HE_CORR = 0.76

# Crighton+2015 at z=4, 4.9:
#   raw Ω_DLA × 10³ = 1.18, 0.98  → Ω_HI^pure × 10³ = 0.76 × raw
C15_Z = np.array([4.0, 4.9])
C15_OMEGA_RAW = np.array([1.18, 0.98])
C15_XERR = np.array([[4.0 - 3.56, 4.45 - 4.0],
                     [4.9 - 4.45, 5.31 - 4.9]]).T  # shape (2, 2) for xerr
C15_YERR = np.array([[1.18 - 0.92, 1.44 - 1.18],
                     [0.98 - 0.80, 1.18 - 0.98]]).T

# Fiducial cosmology used to convert dN/dz ↔ dN/dX for the LLS compilation.
OMEGA_M = 0.315
OMEGA_L = 0.685

# -----------------------------------------------------------------------------
# Observational DLA dN/dX — verbatim from sbird/dla_data (matches prior figure)
# -----------------------------------------------------------------------------
PW09_DLA = np.array([
    [2.2, 2.4, 0.048, 0.006],
    [2.4, 2.7, 0.055, 0.005],
    [2.7, 3.0, 0.067, 0.006],
    [3.0, 3.5, 0.084, 0.006],
    [3.5, 4.0, 0.075, 0.009],
    [4.0, 5.5, 0.106, 0.018],
])
N12_Z  = np.array([2.15, 2.45, 2.75, 3.05, 3.35])
N12_DNDZ = np.array([0.20, 0.20, 0.25, 0.29, 0.36])
N12_DZDX = np.array([3690/11625., 4509/14841., 2867/9900., 1620/5834., 789/2883.])
N12_DLA = np.column_stack([N12_Z, N12_DNDZ * N12_DZDX])
HO21_DNDX_Z = np.array([
    2.083, 2.250, 2.417, 2.583, 2.750, 2.917, 3.083, 3.250, 3.417, 3.583,
    3.750, 3.917, 4.083, 4.250, 4.417, 4.583, 4.750, 4.917,
])
HO21_DNDX_MEDIAN = np.array([
    0.0337, 0.0430, 0.0462, 0.0494, 0.0622, 0.0664, 0.0706, 0.0748, 0.0763,
    0.0777, 0.0630, 0.0646, 0.0577, 0.0725, 0.1015, 0.0821, 0.1033, 0.0674,
])
HO21_DNDX_68_LO = np.array([
    0.0330, 0.0421, 0.0452, 0.0482, 0.0607, 0.0647, 0.0685, 0.0722, 0.0729,
    0.0736, 0.0584, 0.0583, 0.0503, 0.0637, 0.0888, 0.0684, 0.0812, 0.0506,
])
HO21_DNDX_68_HI = np.array([
    0.0345, 0.0438, 0.0472, 0.0506, 0.0637, 0.0682, 0.0729, 0.0777, 0.0800,
    0.0822, 0.0687, 0.0717, 0.0666, 0.0857, 0.1205, 0.1049, 0.1402, 0.1180,
])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def dXdz(z, Om=OMEGA_M, OL=OMEGA_L):
    """dX/dz = (1+z)^2 H_0 / H(z);  H^2/H_0^2 = Ω_M (1+z)^3 + Ω_Λ."""
    z = np.asarray(z, dtype=float)
    E = np.sqrt(Om * (1 + z) ** 3 + OL)
    return (1 + z) ** 2 / E


def load_lofz_literature():
    """Parse ~/data/lofz_literature.txt → list of (z_lo, z_hi, lz, yerr_hi, yerr_lo, ref)."""
    refs = {}
    rows = []
    with open(LOFZ_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Parse the header with ref numbers (not strictly needed)
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            z_lo, z_hi, lz, up, dw, ref = parts[:6]
            rows.append((float(z_lo), float(z_hi), float(lz),
                         float(up), float(dw), int(ref)))
    return rows


# Reference names, verbatim from the file header
LOFZ_REFS = {
    0: "Prochaska+2010",
    1: "Fumagalli+2013",
    2: "Crighton+2018",
    3: "O'Meara+2013 / Ribaudo+2011",
    4: "BOSS",
}


def load_berg19_omega_hi():
    """Load Berg+2019 (XQ-100) Ω_subDLA and Ω_DLA tables."""
    zs = []
    ohi_sub = {"p17": [], "p50": [], "p83": []}
    ohi_dla = {"p17": [], "p50": [], "p83": []}
    with open(OMEGA_HI_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            zs.append(float(row["z"]))
            ohi_sub["p17"].append(float(row["OmegaHI_sub_p17"]))
            ohi_sub["p50"].append(float(row["OmegaHI_sub_med"]))
            ohi_sub["p83"].append(float(row["OmegaHI_sub_p83"]))
            ohi_dla["p17"].append(float(row["OmegaHI_dla_p17"]))
            ohi_dla["p50"].append(float(row["OmegaHI_dla_med"]))
            ohi_dla["p83"].append(float(row["OmegaHI_dla_p83"]))
    zs = np.array(zs)
    for d in (ohi_sub, ohi_dla):
        for k in d:
            d[k] = np.array(d[k])
    return zs, ohi_sub, ohi_dla


def load_pw09_omega_hi():
    """PW09 Ω_HI(DLA) from DLA_data/dndx.txt cols rho_HI, err_rho_HI.
    Returns (z_lo, z_hi, Ω_HI, σ_Ω) in dimensionless Ω units, μ=1 (pure HI).
    Skips the full 2.2-5.5 overview row (row 0).
    """
    data = np.loadtxt(DLA_DATA_ROOT / "dndx.txt")
    rho_hi = data[1:, 4]     # 10^8 M_sun/Mpc³
    err_rho = data[1:, 5]
    z_lo = data[1:, 0]; z_hi = data[1:, 1]
    omega = rho_hi * PW09_CONV / RHO_CRIT_0
    sigma = err_rho * PW09_CONV / RHO_CRIT_0
    return z_lo, z_hi, omega, sigma


def noterdaeme12_omega_hi():
    """N12 Ω_HI(DLA) with 0.76 He correction (μ=1) — sbird/dla_data omegahi_not."""
    z = np.array([2.15, 2.45, 2.75, 3.05, 3.35])
    raw_x1e3 = np.array([0.99, 0.87, 1.04, 1.10, 1.27])
    err_x1e3 = np.array([0.05, 0.04, 0.05, 0.08, 0.13])
    return z, raw_x1e3 * N12_HE_CORR * 1e-3, err_x1e3 * N12_HE_CORR * 1e-3


def crighton15_omega_hi():
    """Crighton+2015 Ω_HI(DLA) with 0.76 He correction — sbird/dla_data crighton_omega."""
    omega = N12_HE_CORR * C15_OMEGA_RAW * 1e-3
    # Y err in raw × 10⁻³ times 0.76:
    yerr = C15_YERR * N12_HE_CORR * 1e-3
    return C15_Z, omega, yerr, C15_XERR


def bin_average_by_z(z_arr, y_arr, z_bins):
    """Return (z_centres, mean, sigma) per bin; sigma = std across sims in bin."""
    z_c, mean, sig = [], [], []
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        sel = (z_arr >= lo) & (z_arr < hi)
        if not sel.any():
            continue
        z_c.append(0.5 * (lo + hi))
        mean.append(np.nanmean(y_arr[sel]))
        sig.append(np.nanstd(y_arr[sel]))
    return np.array(z_c), np.array(mean), np.array(sig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Load PRIYA summary
    print("Loading HCD summary HDF5s…")
    with h5py.File(LF_H5, "r") as f:
        z_lf = f["z"][:]
        dndx_lf = {cls: f[f"dndx/{cls}"][:] for cls in ["LLS", "subDLA", "DLA"]}
        ohi_lf = {cls: f[f"Omega_HI/{cls}"][:] for cls in ["LLS", "subDLA", "DLA"]}
    with h5py.File(HR_H5, "r") as f:
        z_hr = f["z"][:]
        dndx_hr = {cls: f[f"dndx/{cls}"][:] for cls in ["LLS", "subDLA", "DLA"]}
        ohi_hr = {cls: f[f"Omega_HI/{cls}"][:] for cls in ["LLS", "subDLA", "DLA"]}
    print(f"  LF: {len(z_lf)} records,  HR: {len(z_hr)} records")

    # Load obs
    lofz = load_lofz_literature()
    z_berg, ohi_sub_berg, ohi_dla_berg = load_berg19_omega_hi()

    # Unified z-binning for PRIYA curves
    z_bins = np.arange(1.9, 5.7, 0.2)

    # ========= Figure 1: dN/dX per class, HR vs LF vs obs =========
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, cls, col in zip(axes, ["LLS", "subDLA", "DLA"], ["C2", "C1", "C3"]):
        # LF suite: mean ± std across sims at each z
        z_c_lf, m_lf, s_lf = bin_average_by_z(z_lf, dndx_lf[cls], z_bins)
        ax.errorbar(z_c_lf, m_lf, yerr=s_lf, fmt="o-", color=col, lw=1.5, ms=5,
                    capsize=3, label=f"PRIYA LF (60 sims, 1σ suite spread)")
        # HR suite
        z_c_hr, m_hr, s_hr = bin_average_by_z(z_hr, dndx_hr[cls], z_bins)
        ax.errorbar(z_c_hr, m_hr, yerr=s_hr, fmt="s--", color=col,
                    alpha=0.8, mec="k", lw=1.5, ms=7,
                    capsize=3, label=f"PRIYA HR (4 sims, 1σ suite spread)")

        if cls == "LLS":
            # Overlay LLS literature, converted dN/dz → dN/dX at each z centre
            colors = plt.cm.tab10(np.linspace(0, 1, 5))
            plotted = set()
            for z_lo, z_hi, lz, up, dw, ref in lofz:
                z_mid = 0.5 * (z_lo + z_hi)
                dxdz_val = dXdz(z_mid)
                y = lz / dxdz_val
                y_up = up / dxdz_val
                y_dw = dw / dxdz_val
                label = LOFZ_REFS[ref] if ref not in plotted else None
                plotted.add(ref)
                ax.errorbar(z_mid, y,
                            xerr=0.5 * (z_hi - z_lo),
                            yerr=[[y_dw], [y_up]],
                            fmt="D", color=colors[ref], ms=6, capsize=3,
                            label=label)
            ax.set_title(f"dN/dX (LLS)  vs obs (ℓ(z) converted via Planck cosmology)")
        elif cls == "DLA":
            # PW09
            z_mid = 0.5 * (PW09_DLA[:, 0] + PW09_DLA[:, 1])
            z_err = 0.5 * (PW09_DLA[:, 1] - PW09_DLA[:, 0])
            ax.errorbar(z_mid, PW09_DLA[:, 2], xerr=z_err, yerr=PW09_DLA[:, 3],
                        fmt="s", color="black", ms=6, capsize=3,
                        label="PW09 (SDSS DR5)")
            ax.errorbar(N12_DLA[:, 0], N12_DLA[:, 1], xerr=0.15,
                        fmt="^", color="dimgray", ms=6, capsize=3,
                        label="N12 (BOSS DR9)")
            ho21_yerr_lo = HO21_DNDX_MEDIAN - HO21_DNDX_68_LO
            ho21_yerr_hi = HO21_DNDX_68_HI - HO21_DNDX_MEDIAN
            ax.errorbar(HO21_DNDX_Z, HO21_DNDX_MEDIAN,
                        yerr=[ho21_yerr_lo, ho21_yerr_hi],
                        fmt="D", color="darkslategray", ms=5, capsize=3,
                        alpha=0.85, label="Ho+2021 (SDSS DR16)")
            ax.set_title("dN/dX (DLA)  vs obs")
        else:
            ax.set_title("dN/dX (subDLA)  — no direct obs")
        ax.set_yscale("log")
        ax.set_xlabel("z"); ax.set_ylabel(f"dN/dX ({cls})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "dN/dX per class:  HR (solid-square, dashed) vs LF (circle-line) vs observations"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outp = OUT / "dndx_hr_vs_lf_vs_obs_per_class.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")

    # ========= Figure 2: Ω_HI vs obs (Berg+2019) per class =========
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, cls, col in zip(axes, ["LLS", "subDLA", "DLA"], ["C2", "C1", "C3"]):
        z_c_lf, m_lf, s_lf = bin_average_by_z(z_lf, ohi_lf[cls], z_bins)
        ax.errorbar(z_c_lf, m_lf, yerr=s_lf, fmt="o-", color=col, lw=1.5, ms=5,
                    capsize=3, label="PRIYA LF (60 sims, 1σ spread)")
        z_c_hr, m_hr, s_hr = bin_average_by_z(z_hr, ohi_hr[cls], z_bins)
        ax.errorbar(z_c_hr, m_hr, yerr=s_hr, fmt="s--", color=col,
                    alpha=0.85, mec="k", lw=1.5, ms=7, capsize=3,
                    label="PRIYA HR (4 sims, 1σ spread)")

        # Berg+2019 overlays for DLA and subDLA
        if cls == "DLA":
            ax.plot(z_berg, ohi_dla_berg["p50"], "k-", lw=2,
                    label="Berg+2019 XQ-100 (median)")
            ax.fill_between(z_berg, ohi_dla_berg["p17"], ohi_dla_berg["p83"],
                            color="k", alpha=0.2, label="Berg+2019 68 % CI")
            # PW09: bin-edge format with errorbars
            z_lo, z_hi, o_pw, s_pw = load_pw09_omega_hi()
            z_mid = 0.5 * (z_lo + z_hi)
            z_err = 0.5 * (z_hi - z_lo)
            ax.errorbar(z_mid, o_pw, xerr=z_err, yerr=s_pw,
                        fmt="D", color="magenta", ms=7, capsize=3,
                        label="PW09 (SDSS DR5; pure HI)")
            # N12: z + sigma at z-mid
            z_n, o_n, s_n = noterdaeme12_omega_hi()
            ax.errorbar(z_n, o_n, xerr=0.15, yerr=s_n,
                        fmt="s", color="dimgray", ms=6, capsize=3,
                        label="N12 (BOSS DR9; × 0.76)")
            # Crighton+2015
            z_c, o_c, y_c, x_c = crighton15_omega_hi()
            ax.errorbar(z_c, o_c, xerr=x_c.T, yerr=y_c.T,
                        fmt="^", color="brown", ms=7, capsize=3,
                        label="Crighton+15 (× 0.76)")
        elif cls == "subDLA":
            ax.plot(z_berg, ohi_sub_berg["p50"], "k-", lw=2,
                    label="Berg+2019 XQ-100 (median)")
            ax.fill_between(z_berg, ohi_sub_berg["p17"], ohi_sub_berg["p83"],
                            color="k", alpha=0.2, label="Berg+2019 68 % CI")
        else:
            ax.text(0.05, 0.95,
                    "No direct Ω_HI(LLS) obs tabulation in ~/data/",
                    transform=ax.transAxes, fontsize=8, va="top")
        ax.set_yscale("log")
        ax.set_xlabel("z"); ax.set_ylabel(f"Ω_HI ({cls})")
        ax.set_xlim(1.9, 5.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"Ω_HI ({cls})")
    fig.suptitle("Ω_HI per class:  HR vs LF vs Berg+2019 (XQ-100) for DLA and subDLA")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outp = OUT / "omega_hi_hr_vs_lf_vs_obs.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
