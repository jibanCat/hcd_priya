"""PRIYA sim CDDF f(N_HI,X) vs the OBSERVED literature CDDF (Head-A overlay companion to A6).

Head A predicts the column-density distribution f(N,X) = d²n/(dN dX) over 30 log-N_HI bins.
This overlays PRIYA's sim CDDF (mean±std over sims at z≈2.5–3) on the observed CDDF, so we
can see whether the emulator reproduces the canonical features (LLS knee ~10^17.5, the
sub-DLA→DLA transition at 10^20.3, the high-N break ~10^21.5).

Literature (2026-06-04 Lyα agent): DLA = Noterdaeme+2012 Table 1 (⟨z⟩=2.5, tabulated);
sub-DLA+DLA Γ-fit = Zafar+2013 Table 5 (z 1.51–3.10: log k_g=−22.30, log N_g=21.08,
α_g=−0.95); LLS = O'Meara+2013 power law (f=k·N^β, log f(10^19)=−20.2, β=−0.9, z~2.4).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_cddf_vs_literature.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUT = ROOT / "figures/analysis/06_performance_walkthrough/A7_cddf_vs_literature.png"

# Noterdaeme+2012 Table 1 (DLA, systematics-corrected log f(N,X)); ⟨z⟩=2.5
NOT12_LOGN = np.array([20.05, 20.25, 20.45, 20.65, 20.85, 21.05, 21.25, 21.45, 21.65,
                       21.85, 22.10, 22.30])
NOT12_LOGF = np.array([-21.44, -21.59, -21.82, -22.14, -22.51, -22.91, -23.28, -23.81,
                       -24.20, -24.85, -26.05, -26.25])


def zafar_gamma(logN, logk=-22.30, logNg=21.08, ag=-0.95):
    """Zafar+2013 Γ-function CDDF (z 1.51–3.10): f = k_g (N/N_g)^α_g exp(−N/N_g)."""
    N = 10.0 ** logN; Ng = 10.0 ** logNg
    return 10.0 ** logk * (N / Ng) ** ag * np.exp(-N / Ng)


def omeara_lls(logN, logf19=-20.2, beta=-0.9):
    """O'Meara+2013 LLS power law f = k N^β, anchored log f(10^19)=−20.2 (z~2.4)."""
    return 10.0 ** logf19 * (10.0 ** logN / 1e19) ** beta


def main():
    with h5py.File(CACHE, "r") as f:
        logn = np.asarray(f["log_nhi_centres"])       # (30,)
        fnhi = np.asarray(f["snap_f_nhi"])            # (Ngroup, 30) f(N,X)
        gid = np.asarray(f["snap_group_idx"]); zrow = np.asarray(f["z_grid"])
    ng = fnhi.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(ng)])
    sel = (zg >= 2.4) & (zg <= 3.2)                   # z≈2.5–3 (where the lit applies)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(fnhi > 0, fnhi, np.nan)[sel], 0)
        std = np.nanstd(np.where(fnhi > 0, fnhi, np.nan)[sel], 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ok = np.isfinite(mean) & (mean > 0)
    ax.fill_between(logn[ok], np.maximum(mean[ok] - std[ok], 1e-30), mean[ok] + std[ok],
                    color="tab:blue", alpha=0.25)
    ax.plot(logn[ok], mean[ok], "-o", color="tab:blue", ms=4, lw=1.8,
            label="PRIYA sim CDDF (z≈2.5–3, mean±std)")
    lgrid = np.linspace(19.0, 22.5, 100)
    ax.plot(lgrid, zafar_gamma(lgrid), "--", color="tab:green", lw=2,
            label="Zafar+2013 Γ-fit (subDLA+DLA)")
    ax.plot(NOT12_LOGN, 10.0 ** NOT12_LOGF, "s", color="tab:red", ms=6,
            label="Noterdaeme+2012 (DLA)")
    llsg = np.linspace(17.2, 19.0, 50)
    ax.plot(llsg, omeara_lls(llsg), ":", color="tab:purple", lw=2,
            label="O'Meara+2013 (LLS)")
    for xb, lbl in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax.axvline(xb, color="grey", ls="-", lw=0.7, alpha=0.6)
        ax.text(xb + 0.05, 1e-19, lbl, fontsize=8, color="grey", rotation=90, va="top")
    ax.set_yscale("log"); ax.set_xlim(17.0, 22.5); ax.set_ylim(1e-28, 1e-17)
    ax.set_xlabel("log10 N_HI [cm^-2]"); ax.set_ylabel("f(N, X)  [cm^2]")
    ax.set_title("Head-A CDDF: PRIYA sim vs observed literature (z≈2.5–3)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)
    # quick PRIYA/lit check at a few N in the DLA range (vs Noterdaeme)
    for ln in (20.3, 20.9, 21.5):
        i = int(np.argmin(np.abs(logn - ln)))
        lit = 10.0 ** np.interp(ln, NOT12_LOGN, NOT12_LOGF)
        if np.isfinite(mean[i]) and mean[i] > 0:
            print(f"  logN={ln}: PRIYA={mean[i]:.2e}  Noterdaeme={lit:.2e}  ratio={mean[i]/lit:.2f}")


if __name__ == "__main__":
    main()
