"""
Per-class single-sightline breakdown.  Picks one representative sightline for
each of:
  * clean (no HCD system in catalog)
  * LLS  (log N ∈ [17.2, 19))
  * subDLA (log N ∈ [19, 20.3))
  * small DLA (log N ∈ [20.3, 21))
  * large DLA (log N ≥ 21)

For each, plots (left→right):
  1. log10 τ(v)                 — absorption structure, width, damping wings
  2. F(v) = exp(-τ)             — how absorption appears in flux
  3. δF(v) = F/<F>_all - 1      — what the P1D FFT sees
  4. single-sightline |FFT|²    — Fourier signature, with cyclic *and* angular
                                   k axes so we can compare against PRIYA k

Overlays: unmasked, τ-space mask (my Phase B), PRIYA mask (exact recipe).

Purpose: let a reader identify which real-space scale maps to which k, and
confirm the mask artefact k-location matches the transition-width feature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import AbsorberCatalog
from hcd_analysis.io import read_header, pixel_dv_kms, iter_tau_batches
from hcd_analysis.masking import (
    apply_tauspace_mask_to_batch,
    priya_dla_mask_row,
)

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT_PATH = ROOT / "tests" / "out" / "snap_017_fastfix_catalog.npz"
DIAG = ROOT / "figures" / "diagnostics"


def pick_sightlines(catalog, n_max=691200):
    """Pick one representative sightline per class.  Returns dict
    class_label -> (sightline_idx, absorber_object_or_None)."""
    # Flatten catalog to arrays for speed
    si = np.array([a.skewer_idx for a in catalog.absorbers])
    lN = np.array([a.log_NHI for a in catalog.absorbers])
    absorbers = list(catalog.absorbers)

    # Build "highest log N per sightline" map
    order = np.lexsort((-lN, si))  # per sightline, highest lN first
    si_sorted = si[order]; lN_sorted = lN[order]
    first_idx = np.concatenate([[True], np.diff(si_sorted) != 0])
    top_si = si_sorted[first_idx]; top_lN = lN_sorted[first_idx]

    # Filter to requested class bands, pick the FIRST one that has a cleanly
    # isolated single absorber (easier to read in the plot).
    def pick(band_lo, band_hi, label):
        cand = (top_lN >= band_lo) & (top_lN < band_hi) & (top_si < n_max)
        for s, l in zip(top_si[cand], top_lN[cand]):
            # Prefer sightlines with exactly ONE absorber (for a clean picture)
            n_on = int((si == s).sum())
            if n_on == 1:
                for ab in absorbers:
                    if ab.skewer_idx == s:
                        return int(s), ab, label
        # Fall back to first match
        for s, l in zip(top_si[cand], top_lN[cand]):
            for ab in absorbers:
                if ab.skewer_idx == s and band_lo <= ab.log_NHI < band_hi:
                    return int(s), ab, label
        return None, None, label

    results = {}
    # Clean: sightline not in the catalog at all
    in_cat = set(np.unique(si[si < n_max]).tolist())
    for cand in range(n_max):
        if cand not in in_cat:
            results["clean"] = (cand, None, "clean (no HCD)")
            break
    results["LLS"]   = tuple(pick(17.2, 19.0, "LLS (17.2 ≤ log N < 19)")[:2]) + ("LLS (17.2 ≤ log N < 19)",)
    results["subDLA"]= tuple(pick(19.0, 20.3, "subDLA (19 ≤ log N < 20.3)")[:2])+ ("subDLA (19 ≤ log N < 20.3)",)
    results["smallDLA"]=tuple(pick(20.3, 21.0, "small DLA (20.3 ≤ log N < 21)")[:2])+ ("small DLA (20.3 ≤ log N < 21)",)
    results["largeDLA"]=tuple(pick(21.0, 23.0, "large DLA (log N ≥ 21)")[:2])+ ("large DLA (log N ≥ 21)",)
    return results


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    DIAG.mkdir(parents=True, exist_ok=True)
    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    cat = AbsorberCatalog.load_npz(CAT_PATH)
    print(f"snap_017 z={header.redshift:.2f}  dv={dv_kms:.2f}  "
          f"catalog: {cat.summary()}")

    # <F>_all for tau_eff
    F_sum = 0.0; F_n = 0
    for _, _, tau in iter_tau_batches(HDF5, batch_size=4096, n_skewers=None):
        F = np.exp(-tau.astype(np.float64))
        F_sum += F.sum(); F_n += F.size
    mean_F_all = F_sum / F_n
    tau_eff = -np.log(max(mean_F_all, 1e-30))
    print(f"  <F>_all = {mean_F_all:.4f}   τ_eff = {tau_eff:.4f}")

    picks = pick_sightlines(cat)
    for k, (si, ab, label) in picks.items():
        info = f"log N={ab.log_NHI:.2f}" if ab else "no absorber"
        print(f"  {k:>9s}:  sightline {si}   {info}   [{label}]")

    # Read all picked sightlines
    si_list = [picks[k][0] for k in ["clean","LLS","subDLA","smallDLA","largeDLA"]]
    with h5py.File(HDF5, "r") as f:
        rows = np.stack([f["tau/H/1/1215"][s, :].astype(np.float64) for s in si_list])

    v = np.arange(header.nbins) * dv_kms

    # Apply masks to each picked sightline.  apply_tauspace_mask_to_batch
    # expects batch + row_start; we pass a single-row batch and offset it so
    # the catalog lookup matches.
    tau_ts_all = np.zeros_like(rows)
    tau_priya_all = np.zeros_like(rows)
    for i, (s, ab, _lab) in enumerate(picks.values()):
        row = rows[i:i+1]  # shape (1, nbins)
        # τ-space (my Phase-B) mask
        if ab is not None:
            tau_ts_all[i] = apply_tauspace_mask_to_batch(
                row, s, cat,
                mask_classes=["LLS","subDLA","DLA"], tau_eff=tau_eff,
            )[0]
        else:
            tau_ts_all[i] = rows[i]
        # PRIYA mask
        mask_priya = priya_dla_mask_row(rows[i], tau_eff, tau_dla_detect=1e6, tau_mask_scale=0.25)
        if mask_priya is None:
            tau_priya_all[i] = rows[i]
        else:
            tau_priya_all[i] = rows[i].copy()
            tau_priya_all[i][mask_priya] = tau_eff

    # ---- Build figure: 5 rows (class) × 4 columns (τ, F, δF, P1D) ---------
    labels = ["clean","LLS","subDLA","smallDLA","largeDLA"]
    nrows = len(labels); ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 14))

    k_cyc = np.fft.rfftfreq(header.nbins, d=dv_kms)      # cyclic k [s/km]
    # angular PRIYA-convention k = 2π × cyclic
    k_ang = 2.0 * np.pi * k_cyc

    for i, lab in enumerate(labels):
        si, ab, desc = picks[lab]
        row = rows[i]; row_ts = tau_ts_all[i]; row_px = tau_priya_all[i]

        # (1) log tau
        ax = axes[i, 0]
        ax.plot(v, np.log10(np.maximum(row, 1e-3)), lw=1.2, color="k", label="unmasked")
        if ab is not None:
            ax.plot(v, np.log10(np.maximum(row_ts, 1e-3)), lw=1.0, color="C0", alpha=0.7, label="τ-space mask")
            ax.plot(v, np.log10(np.maximum(row_px, 1e-3)), lw=1.0, color="C3", alpha=0.7, label="PRIYA mask")
        ax.axhline(np.log10(tau_eff), color="gray", ls=":", alpha=0.6, lw=0.8)
        ax.set_ylabel(f"log10 τ\n[{desc}]", fontsize=9)
        if i == 0: ax.set_title("τ(v)")
        if i == nrows-1: ax.set_xlabel("v [km/s]")
        if i == 0: ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (2) F
        ax = axes[i, 1]
        ax.plot(v, np.exp(-row), lw=1.2, color="k")
        if ab is not None:
            ax.plot(v, np.exp(-row_ts), lw=1.0, color="C0", alpha=0.7)
            ax.plot(v, np.exp(-row_px), lw=1.0, color="C3", alpha=0.7)
        ax.axhline(mean_F_all, color="gray", ls=":", alpha=0.6, lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        if i == 0: ax.set_title("F(v) = exp(-τ)")
        if i == nrows-1: ax.set_xlabel("v [km/s]")
        ax.grid(alpha=0.3)

        # (3) δF
        ax = axes[i, 2]
        dF = np.exp(-row)/mean_F_all - 1
        ax.plot(v, dF, lw=1.2, color="k")
        if ab is not None:
            ax.plot(v, np.exp(-row_ts)/mean_F_all - 1, lw=1.0, color="C0", alpha=0.7)
            ax.plot(v, np.exp(-row_px)/mean_F_all - 1, lw=1.0, color="C3", alpha=0.7)
        ax.axhline(0, color="gray", ls=":", alpha=0.6, lw=0.8)
        ax.set_ylim(-1.2, 1.2)
        if i == 0: ax.set_title("δF(v) = F/⟨F⟩ - 1")
        if i == nrows-1: ax.set_xlabel("v [km/s]")
        ax.grid(alpha=0.3)

        # (4) single-sightline P1D
        ax = axes[i, 3]
        def _p1d(delta):
            ft = np.fft.rfft(delta * dv_kms)
            return np.abs(ft)**2 / (header.nbins * dv_kms)
        p_unm = _p1d(np.exp(-row)/mean_F_all - 1)
        ax.loglog(k_cyc[1:], p_unm[1:], lw=1.2, color="k", label="unmasked")
        if ab is not None:
            p_ts = _p1d(np.exp(-row_ts)/mean_F_all - 1)
            p_px = _p1d(np.exp(-row_px)/mean_F_all - 1)
            ax.loglog(k_cyc[1:], p_ts[1:], lw=1.0, color="C0", alpha=0.7, label="τ-space")
            ax.loglog(k_cyc[1:], p_px[1:], lw=1.0, color="C3", alpha=0.7, label="PRIYA")
        # Annotate feature scales.  1/b ~ 30 km/s → k_cyc ≈ 0.03, in PRIYA ≈ 0.19.
        ax.axvline(1/100., color="gray", ls=":", alpha=0.5)  # k=0.01 → 100 km/s
        ax.axvline(1/30., color="gray", ls=":", alpha=0.5)   # k=0.033 → 30 km/s (b)
        ax.text(1/100.*1.02, 1e-5, "1/(100 km/s)", fontsize=7, rotation=90, color="gray")
        ax.text(1/30.*1.02,  1e-5, "1/(30 km/s) ≈ b", fontsize=7, rotation=90, color="gray")
        if i == 0:
            ax.set_title("Single-sightline P1D")
            # Add an upper axis for PRIYA angular k
            secax = ax.secondary_xaxis('top', functions=(lambda x: 2*np.pi*x,
                                                           lambda x: x/(2*np.pi)))
            secax.set_xlabel("k (PRIYA angular, 2π·k) [rad·s/km]", fontsize=9)
        if i == 0: ax.legend(fontsize=8)
        if i == nrows-1: ax.set_xlabel("k (cyclic) [s/km]")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(5e-4, 5e-2)
        ax.set_ylim(1e-6, 1e2)

    fig.suptitle(
        f"Per-class real-space ↔ Fourier-space breakdown  (snap_017, z=3, sim ns0.803)\n"
        f"τ(v) → F(v) → δF(v) → |FFT|²    [black: unmasked, blue: τ-space mask (my Phase B), red: PRIYA mask]"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outpath = DIAG / "per_class_realspace_fourier.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")


if __name__ == "__main__":
    main()
