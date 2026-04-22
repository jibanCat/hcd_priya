"""
Diagnose the high-k rise in the HCD template and in the masked P1D ratio.

Questions:
  1. Does it appear per-sightline or only in aggregate?  If per-sightline,
     it is a feature of the absorber profile.
  2. Is it class-specific?  Split by DLA-only / subDLA-only / LLS-only
     sightlines and repeat the template.
  3. What does the δF profile look like around a saturated DLA, and what is
     the contribution to |FFT|^2 of that individual box?
  4. How does it respond to each mask variant?

Produces a multi-panel figure and a class-decomposed template.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    apply_mask_to_batch,
)
from hcd_analysis.p1d import P1DAccumulator

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT = ROOT / "tests" / "out" / "snap_017_fastfix_catalog.npz"
DIAG = ROOT / "figures" / "diagnostics"


def class_of_sightline(skewer_idx_arr, classes_arr, si):
    """Return the highest-class present on this sightline."""
    mask = skewer_idx_arr == si
    if not mask.any():
        return "clean"
    cls = classes_arr[mask]
    if "DLA" in cls: return "DLA"
    if "subDLA" in cls: return "subDLA"
    if "LLS" in cls: return "LLS"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skewers", type=int, default=100000)
    args = ap.parse_args()

    DIAG.mkdir(parents=True, exist_ok=True)
    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    catalog = AbsorberCatalog.load_npz(CAT)
    print(f"snap_017 z={header.redshift:.2f}  catalog: {catalog.summary()}")

    n_total = min(header.n_skewers, args.n_skewers)

    # Classify each sightline by its highest-class absorber
    si_arr = np.array([a.skewer_idx for a in catalog.absorbers])
    cl_arr = np.array([a.absorber_class for a in catalog.absorbers])
    # For each sightline, keep the highest class present
    sightline_class = np.full(header.n_skewers, "clean", dtype=object)
    for cls in ("LLS", "subDLA", "DLA"):  # order matters so DLA wins
        mask = cl_arr == cls
        sightline_class[si_arr[mask]] = cls
    sightline_class = sightline_class[:n_total]
    for cls in ("clean", "LLS", "subDLA", "DLA"):
        n = int((sightline_class == cls).sum())
        print(f"  {cls:>8s} sightlines: {n:>6d}  ({100*n/n_total:.1f}%)")

    # ---- Per-sightline <F> for each class, then τ_eff for mask fills ------
    F_sum = {c: 0.0 for c in ("all", "clean", "LLS", "subDLA", "DLA")}
    F_n   = {c: 0   for c in F_sum}
    for row_start, row_end, tau_batch in iter_tau_batches(
        HDF5, batch_size=4096, n_skewers=n_total
    ):
        tau_batch = tau_batch.astype(np.float64)
        F = np.exp(-tau_batch)
        cls_batch = sightline_class[row_start:row_end]
        F_sum["all"] += F.sum(); F_n["all"] += F.size
        for c in ("clean", "LLS", "subDLA", "DLA"):
            m = cls_batch == c
            if m.any():
                F_sum[c] += F[m].sum(); F_n[c] += F[m].size
    mean_F = {c: F_sum[c] / F_n[c] if F_n[c] > 0 else 1.0 for c in F_sum}
    tau_eff = -np.log(max(mean_F["all"], 1e-30))
    print(f"  <F>: all={mean_F['all']:.4f}  clean={mean_F['clean']:.4f}  "
          f"LLS={mean_F['LLS']:.4f}  subDLA={mean_F['subDLA']:.4f}  DLA={mean_F['DLA']:.4f}"
          f"  τ_eff={tau_eff:.4f}")

    # ---- P1D per class, plus each class with τ-space and pixrange masks ----
    # (We keep only the "highest-class" sightlines per class for a clean split.)
    accs_unmasked = {c: P1DAccumulator(header.nbins, dv_kms) for c in ("clean","LLS","subDLA","DLA")}
    accs_maskA    = {c: P1DAccumulator(header.nbins, dv_kms) for c in ("LLS","subDLA","DLA")}
    accs_maskB    = {c: P1DAccumulator(header.nbins, dv_kms) for c in ("LLS","subDLA","DLA")}

    # Recompute each masked subset's <F> for normalisation
    F_sum_mA = {c: 0.0 for c in accs_maskA}; F_n_mA = {c: 0 for c in accs_maskA}
    F_sum_mB = {c: 0.0 for c in accs_maskB}; F_n_mB = {c: 0 for c in accs_maskB}
    for row_start, row_end, tau_batch in iter_tau_batches(
        HDF5, batch_size=4096, n_skewers=n_total
    ):
        tau_batch = tau_batch.astype(np.float64)
        cls_batch = sightline_class[row_start:row_end]
        tau_A = apply_tauspace_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS","subDLA","DLA"], tau_eff=tau_eff,
        )
        tau_B = apply_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS","subDLA","DLA"], strategy="mean_flux",
        )
        for c in ("LLS","subDLA","DLA"):
            m = cls_batch == c
            if m.any():
                F_sum_mA[c] += np.exp(-tau_A[m]).sum(); F_n_mA[c] += tau_A[m].size
                F_sum_mB[c] += np.exp(-tau_B[m]).sum(); F_n_mB[c] += tau_B[m].size
    mean_F_mA = {c: F_sum_mA[c]/F_n_mA[c] if F_n_mA[c]>0 else 1.0 for c in F_sum_mA}
    mean_F_mB = {c: F_sum_mB[c]/F_n_mB[c] if F_n_mB[c]>0 else 1.0 for c in F_sum_mB}

    for row_start, row_end, tau_batch in iter_tau_batches(
        HDF5, batch_size=4096, n_skewers=n_total
    ):
        tau_batch = tau_batch.astype(np.float64)
        cls_batch = sightline_class[row_start:row_end]
        tau_A = apply_tauspace_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS","subDLA","DLA"], tau_eff=tau_eff,
        )
        tau_B = apply_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS","subDLA","DLA"], strategy="mean_flux",
        )
        for c in ("clean","LLS","subDLA","DLA"):
            m = cls_batch == c
            if m.any():
                accs_unmasked[c].add_batch(tau_batch[m], mean_F_global=mean_F[c])
                if c in accs_maskA:
                    accs_maskA[c].add_batch(tau_A[m], mean_F_global=mean_F_mA[c])
                    accs_maskB[c].add_batch(tau_B[m], mean_F_global=mean_F_mB[c])

    # Extract P1Ds (native k grid for maximum resolution)
    results = {}
    for c, acc in accs_unmasked.items():
        results[f"unm_{c}"] = acc.raw_power()
    for c, acc in accs_maskA.items():
        results[f"mA_{c}"] = acc.raw_power()
    for c, acc in accs_maskB.items():
        results[f"mB_{c}"] = acc.raw_power()

    k = results["unm_clean"][0]  # native k grid (s/km)

    # ---- Plot 1: class-decomposed template ------------------------------
    p_clean = results["unm_clean"][1]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    ax[0,0].set_title("Class-decomposed HCD template:\nP_unmasked(class) / P_clean")
    for c, color in [("LLS","C2"), ("subDLA","C1"), ("DLA","C3")]:
        r = results[f"unm_{c}"][1] / np.where(p_clean > 0, p_clean, np.nan)
        ax[0,0].plot(k, r, lw=1.8, label=f"{c}-only (n={int((sightline_class==c).sum())})", color=color)
    ax[0,0].axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax[0,0].axvline(0.02, color="gray", ls=":", alpha=0.5, label="k=0.02 s/km")
    ax[0,0].set_xscale("log"); ax[0,0].set_yscale("log")
    ax[0,0].set_xlabel("k [s/km]"); ax[0,0].set_ylabel("P_class / P_clean")
    ax[0,0].legend(); ax[0,0].grid(alpha=0.3, which="both")
    ax[0,0].set_xlim(1e-3, 5e-2); ax[0,0].set_ylim(0.8, 2.0)

    # Plot 2: absolute P1D (one panel per class) comparing unm / mA / mB
    for (c, color, axi) in [("LLS","C2",(0,1)),("subDLA","C1",(1,0)),("DLA","C3",(1,1))]:
        ax[axi].plot(k, results[f"unm_{c}"][1], lw=1.8, color=color, label=f"unmasked {c}-only")
        ax[axi].plot(k, results[f"mA_{c}"][1], lw=1.4, color="C0", ls="-", label="τ-space mask")
        ax[axi].plot(k, results[f"mB_{c}"][1], lw=1.4, color="C1", ls="-", label="pixrange mask")
        ax[axi].plot(k, p_clean, lw=1.2, color="k", ls="--", alpha=0.7, label="P_clean reference")
        ax[axi].axvline(0.02, color="gray", ls=":", alpha=0.5)
        ax[axi].set_xscale("log"); ax[axi].set_yscale("log")
        ax[axi].set_xlabel("k [s/km]"); ax[axi].set_ylabel("P1D (km/s)")
        ax[axi].set_title(f"{c}-only sightlines")
        ax[axi].legend(fontsize=8); ax[axi].grid(alpha=0.3, which="both")
        ax[axi].set_xlim(1e-3, 5e-2)

    fig.suptitle(f"Per-class HCD template + masking at snap_017  z=3  (n_skewers={n_total})")
    fig.tight_layout()
    out1 = DIAG / "template_per_class.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {out1}")

    # ---- Plot 2: pick one DLA sightline and visualise tau/F/δF and |FFT|^2 ----
    # Find a clean DLA sightline with no other strong absorbers (single DLA ≥ 20.3)
    target_si = None
    for ab in catalog.absorbers:
        if ab.absorber_class != "DLA": continue
        if ab.skewer_idx >= n_total: continue
        # check this is the only absorber on the sightline
        n_on = int((si_arr == ab.skewer_idx).sum())
        if n_on == 1:
            target_si = ab.skewer_idx
            break
    if target_si is None:
        print("(no clean single-DLA sightline found; picking first DLA-containing)")
        target_si = int(si_arr[cl_arr == "DLA"][0])
    print(f"  example DLA sightline: {target_si}")

    import h5py
    with h5py.File(HDF5, "r") as f:
        tau_row = f["tau/H/1/1215"][target_si, :].astype(np.float64)

    tau_A = apply_tauspace_mask_to_batch(
        tau_row[None, :], target_si, catalog,
        mask_classes=["LLS","subDLA","DLA"], tau_eff=tau_eff,
    )[0]
    tau_B = apply_mask_to_batch(
        tau_row[None, :], target_si, catalog,
        mask_classes=["LLS","subDLA","DLA"], strategy="mean_flux",
    )[0]

    v = np.arange(header.nbins) * dv_kms
    F = np.exp(-tau_row)
    FA = np.exp(-tau_A)
    FB = np.exp(-tau_B)
    dF = F/mean_F["all"] - 1
    dFA = FA/mean_F["all"] - 1
    dFB = FB/mean_F["all"] - 1

    def p1d_single(delta):
        """P1D of one sightline using the same convention as P1DAccumulator."""
        ft = np.fft.rfft(delta * dv_kms)
        return np.abs(ft)**2 / (header.nbins * dv_kms)
    k_native = np.fft.rfftfreq(header.nbins, d=dv_kms)
    p = p1d_single(dF); pA = p1d_single(dFA); pB = p1d_single(dFB)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0,0].plot(v, np.log10(np.maximum(tau_row,1e-3)), lw=1.4, label="unmasked", color="k")
    ax[0,0].plot(v, np.log10(np.maximum(tau_A,1e-3)), lw=1.2, label="τ-space fill", color="C0", alpha=0.7)
    ax[0,0].plot(v, np.log10(np.maximum(tau_B,1e-3)), lw=1.2, label="pixrange fill", color="C1", alpha=0.7)
    ax[0,0].axhline(np.log10(tau_eff), color="gray", ls=":", alpha=0.6, label=f"τ_eff")
    ax[0,0].set_xlabel("v [km/s]"); ax[0,0].set_ylabel("log10 τ(v)")
    ax[0,0].set_title(f"Sightline {target_si}: optical depth"); ax[0,0].legend(fontsize=8)
    ax[0,0].grid(alpha=0.3)

    ax[0,1].plot(v, F, lw=1.4, label="unmasked", color="k")
    ax[0,1].plot(v, FA, lw=1.2, label="τ-space mask", color="C0", alpha=0.7)
    ax[0,1].plot(v, FB, lw=1.2, label="pixrange mask", color="C1", alpha=0.7)
    ax[0,1].axhline(mean_F["all"], color="gray", ls=":", alpha=0.6, label="<F>_all")
    ax[0,1].set_xlabel("v [km/s]"); ax[0,1].set_ylabel("F(v)")
    ax[0,1].set_title("Flux (linear)"); ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=0.3)

    ax[1,0].plot(v, dF, lw=1.4, label="unmasked", color="k")
    ax[1,0].plot(v, dFA, lw=1.2, label="τ-space mask", color="C0", alpha=0.7)
    ax[1,0].plot(v, dFB, lw=1.2, label="pixrange mask", color="C1", alpha=0.7)
    ax[1,0].axhline(0, color="gray", ls=":", alpha=0.6)
    ax[1,0].set_xlabel("v [km/s]"); ax[1,0].set_ylabel("δF(v)")
    ax[1,0].set_title("Fractional flux fluctuation"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=0.3)

    ax[1,1].plot(k_native[1:], p[1:],  lw=1.4, label="unmasked", color="k")
    ax[1,1].plot(k_native[1:], pA[1:], lw=1.2, label="τ-space mask", color="C0", alpha=0.8)
    ax[1,1].plot(k_native[1:], pB[1:], lw=1.2, label="pixrange mask", color="C1", alpha=0.8)
    ax[1,1].axvline(0.02, color="gray", ls=":", alpha=0.4)
    ax[1,1].axvline(0.05, color="gray", ls=":", alpha=0.4)
    ax[1,1].text(0.021, 1.5e-2, "0.02", rotation=90, fontsize=8)
    ax[1,1].text(0.051, 1.5e-2, "Nyq", rotation=90, fontsize=8)
    ax[1,1].set_xscale("log"); ax[1,1].set_yscale("log")
    ax[1,1].set_xlabel("k [s/km]"); ax[1,1].set_ylabel("P1D(k) (km/s)")
    ax[1,1].set_title("Single-sightline P1D"); ax[1,1].legend(fontsize=8)
    ax[1,1].grid(alpha=0.3, which="both"); ax[1,1].set_xlim(5e-4, 5e-2)

    fig.suptitle(f"Single DLA sightline {target_si}  (τ_peak={tau_row.max():.2e})")
    fig.tight_layout()
    out2 = DIAG / "single_dla_breakdown.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Figure: {out2}")


if __name__ == "__main__":
    main()
