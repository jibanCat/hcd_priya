"""
Public CDDF comparison:  PRIYA LF CDDF vs Ho+2021 (arXiv:2103.10964) in
four z-bins.

Ho21 CDDF tables live at /home/mfho/DLA_data/ho21/cddf_z{225,253,34,45}.txt
(kept as a git-independent local mirror of the sbird/dla_data repo).

Ho21 file format (from sbird/dla_data.ho21_cddf):
    row 0: logN bin centres  (0.1-dex grid, 20.05–22.95)
    row 1: central f(N)
    row 2: lower 68 % CI    (f(N) − σ_lo)
    row 3: upper 68 % CI    (f(N) + σ_hi)
    row 4: lower 95 % CI
    row 5: upper 95 % CI
    values are f(N, X) in cm².  Rows with 0.0 values are non-detections
    in that logN bin and are dropped from the plot.

Output:
    figures/analysis/01_catalog_obs/cddf_priya_vs_ho21.png  (2×2 panels)

Run:
    python3 scripts/plot_cddf_vs_ho21.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hcd_analysis.cddf import absorption_path_per_sightline

SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
HO21_DIR = Path("/home/mfho/DLA_data/ho21")
OUT = ROOT / "figures" / "analysis" / "01_catalog_obs"
OUT.mkdir(parents=True, exist_ok=True)

# (z_lo, z_hi, Ho21 filename).  Ho21 provides z-binned CDDF tables from
# XQ-100 + BOSS+eBOSS + TNG; we match their binning.
Z_BINS = [
    (2.0, 2.5, "cddf_z225.txt"),
    (2.5, 3.0, "cddf_z253.txt"),
    (3.0, 4.0, "cddf_z34.txt"),
    (4.0, 5.0, "cddf_z45.txt"),
]

# 0.2-dex bins covering full PRIYA range
LOGNHI_BINS = np.linspace(17.0, 23.0, 31)
LOGNHI_CENTRES = 0.5 * (LOGNHI_BINS[:-1] + LOGNHI_BINS[1:])
DN_LINEAR = 10.0 ** LOGNHI_BINS[1:] - 10.0 ** LOGNHI_BINS[:-1]

CLASS_LINES = [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]


def load_ho21(fname: str) -> dict:
    """Parse Ho21 CDDF table.  Returns dict with logN centres, central
    f(N), and 68 % (1σ) lo/hi bounds, keeping only bins where the
    central value is strictly positive."""
    d = np.loadtxt(HO21_DIR / fname)
    if d.ndim != 2 or d.shape[0] < 4:
        raise ValueError(f"unexpected shape for {fname}: {d.shape}")
    logN = d[0]
    f    = d[1]
    lo68 = d[2]
    hi68 = d[3]
    ok = np.isfinite(f) & (f > 0)
    return {
        "logN": logN[ok], "f": f[ok],
        "lo68": lo68[ok], "hi68": hi68[ok],
    }


def _enum_records():
    for sim_dir in sorted(SCRATCH.iterdir()):
        if not sim_dir.is_dir() or not sim_dir.name.startswith("ns"):
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            if not snap_dir.name.startswith("snap_"):
                continue
            cat = snap_dir / "catalog.npz"
            meta_p = snap_dir / "meta.json"
            if not (cat.exists() and meta_p.exists() and (snap_dir / "done").exists()):
                continue
            try:
                meta = json.load(open(meta_p))
                z = float(meta["z"])
            except Exception:
                continue
            yield z, cat, meta


def priya_cddf_for_zbin(z_lo: float, z_hi: float) -> tuple:
    counts = np.zeros(len(LOGNHI_CENTRES), dtype=np.float64)
    total_path = 0.0
    n_rec = 0
    n_abs = 0
    for z, cat_path, meta in _enum_records():
        if not (z_lo <= z < z_hi):
            continue
        try:
            d = np.load(str(cat_path), allow_pickle=True)
            NHI = d["NHI"].astype(np.float64)
        except Exception:
            continue
        log_NHI = np.log10(np.maximum(NHI, 1e1))
        n_skewers = int(meta.get("n_skewers", 691200))
        hubble = float(meta.get("hub", meta.get("hubble", 0.71)))
        box_kpc_h = float(meta.get("box_kpc_h", 120000.0))
        omegam = float(meta.get("omegam", 0.3))
        omegal = 1.0 - omegam
        dX = absorption_path_per_sightline(box_kpc_h, hubble, omegam, omegal, z)
        total_path += n_skewers * dX
        c, _ = np.histogram(log_NHI, bins=LOGNHI_BINS)
        counts = counts + c
        n_rec += 1
        n_abs += int(c.sum())
    if total_path <= 0:
        return LOGNHI_CENTRES, np.full_like(LOGNHI_CENTRES, np.nan), 0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_nhi = np.where(DN_LINEAR * total_path > 0,
                         counts / (DN_LINEAR * total_path), np.nan)
    return LOGNHI_CENTRES, f_nhi, n_rec, n_abs


def plot_grid(outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (z_lo, z_hi, fname) in enumerate(Z_BINS):
        ax = axes[i]
        logN_p, f_p, n_rec, n_abs = priya_cddf_for_zbin(z_lo, z_hi)
        mask = np.isfinite(f_p) & (f_p > 0)
        if mask.any():
            ax.semilogy(
                logN_p[mask], f_p[mask], "-o", ms=3, lw=1.5, color="#2980b9",
                label=f"PRIYA LF  ({n_rec} snaps, {n_abs:,} absorbers)",
            )

        ho21 = load_ho21(fname)
        if len(ho21["logN"]):
            yerr_lo = np.clip(ho21["f"] - ho21["lo68"], 0, ho21["f"] * 0.999)
            yerr_hi = ho21["hi68"] - ho21["f"]
            ax.errorbar(
                ho21["logN"], ho21["f"],
                yerr=[yerr_lo, yerr_hi],
                fmt="o", color="black", ms=4, capsize=2, alpha=0.9,
                label="Ho+21 (68 % CI)",
            )

        for logN, cls in CLASS_LINES:
            ax.axvline(logN, ls="--", color="gray", alpha=0.4, lw=0.8)
            ax.text(logN + 0.05, 3e-17, cls, fontsize=7, color="gray", va="top")

        ax.set_title(f"{z_lo:.1f} ≤ z < {z_hi:.1f}", fontsize=11)
        ax.set_xlim(17, 23)
        ax.set_ylim(1e-29, 1e-16)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")
        if i % 2 == 0:
            ax.set_ylabel(r"$f(N_\mathrm{HI}, X)$  [cm$^2$]")
        if i >= 2:
            ax.set_xlabel(r"$\log_{10}(N_\mathrm{HI} / \mathrm{cm}^{-2})$")

    fig.suptitle(
        "CDDF: PRIYA LF suite vs Ho+21 (arXiv:2103.10964) in 4 redshift bins\n"
        "Ho21 uncertainties are 68 % CI; Ho21 tables cover log N_HI ≥ 20.05 only",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    print("Building PRIYA CDDF per z-bin…")
    plot_grid(OUT / "cddf_priya_vs_ho21.png")


if __name__ == "__main__":
    main()
