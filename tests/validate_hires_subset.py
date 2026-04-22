"""
Rogers-style subset test on the HiRes counterpart of ns0.909Ap1.98e-09...,
which also has the matching LF sim at snap_017.  Purpose: check whether the
high-k rise is a LF convergence artefact.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import build_catalog, AbsorberCatalog
from hcd_analysis.io import read_header, pixel_dv_kms, iter_tau_batches
from hcd_analysis.masking import apply_tauspace_mask_to_batch, apply_mask_to_batch
from hcd_analysis.p1d import P1DAccumulator

SIM = "ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449"
LF_HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
HR_HDF5 = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
OUT = ROOT / "tests" / "out"
DIAG = ROOT / "figures" / "diagnostics"


def build_or_load(hdf5_path, tag):
    """Build (or load) a fast-mode catalog for this file."""
    cat_path = OUT / f"cat_{tag}.npz"
    if cat_path.exists():
        return AbsorberCatalog.load_npz(cat_path)
    header = read_header(hdf5_path)
    dv_kms = pixel_dv_kms(header)
    print(f"  building catalog for {tag}...")
    t0 = time.time()
    cat = build_catalog(
        hdf5_path=hdf5_path, sim_name=SIM, snap=17, z=header.redshift, dv_kms=dv_kms,
        tau_threshold=100.0, merge_dv_kms=100.0, min_pixels=2,
        b_init=30.0, b_bounds=(1.0, 300.0),
        tau_fit_cap=1.0e6, voigt_max_iter=200,
        batch_size=4096, n_skewers=None,
        fast_mode=True, n_workers=8, min_log_nhi=17.2,
    )
    cat.save_npz(cat_path)
    print(f"    built in {time.time()-t0:.1f}s  -> {cat.summary()}")
    return cat


def run_pipeline(hdf5, tag, n_sub):
    """Return dict with k + P_clean + P_dirty + P_maskA + P_maskB."""
    header = read_header(hdf5)
    dv_kms = pixel_dv_kms(header)
    cat = build_or_load(hdf5, tag)
    print(f"  {tag}: z={header.redshift:.2f}  nbins={header.nbins}  dv={dv_kms:.2f}  "
          f"n_skewers={header.n_skewers}  catalog={cat.summary()}")

    n_total = min(header.n_skewers, n_sub)
    dirty = np.zeros(header.n_skewers, dtype=bool)
    for ab in cat.absorbers:
        if ab.skewer_idx < n_total:
            dirty[ab.skewer_idx] = True
    n_dirty = int(dirty[:n_total].sum())
    print(f"    dirty={n_dirty}/{n_total}  ({100*n_dirty/n_total:.1f}%)")

    # Pass 1a: unmasked <F>
    F_all = F_cl = F_di = 0.0
    N_all = N_cl = N_di = 0
    for s, e, tau in iter_tau_batches(hdf5, batch_size=4096, n_skewers=n_total):
        tau = tau.astype(np.float64); F = np.exp(-tau)
        is_d = dirty[s:e]
        F_all += F.sum(); N_all += F.size
        if (~is_d).any(): F_cl += F[~is_d].sum(); N_cl += F[~is_d].size
        if is_d.any():    F_di += F[is_d].sum(); N_di += F[is_d].size
    mean_F_all = F_all/N_all
    mean_F_cl = F_cl/N_cl
    mean_F_di = F_di/N_di
    tau_eff = -np.log(max(mean_F_all, 1e-30))

    # Pass 1b: masked <F>
    F_A = F_B = 0.0; N_A = N_B = 0
    for s, e, tau in iter_tau_batches(hdf5, batch_size=4096, n_skewers=n_total):
        tau = tau.astype(np.float64)
        is_d = dirty[s:e]
        if not is_d.any(): continue
        tA = apply_tauspace_mask_to_batch(tau, s, cat, ["LLS","subDLA","DLA"], tau_eff=tau_eff)
        tB = apply_mask_to_batch(tau, s, cat, ["LLS","subDLA","DLA"], strategy="mean_flux")
        FA = np.exp(-tA[is_d]); FB = np.exp(-tB[is_d])
        F_A += FA.sum(); N_A += FA.size
        F_B += FB.sum(); N_B += FB.size
    mean_F_A = F_A/N_A; mean_F_B = F_B/N_B

    # Pass 2: P1D accumulators
    acc_cl = P1DAccumulator(header.nbins, dv_kms)
    acc_di = P1DAccumulator(header.nbins, dv_kms)
    acc_A = P1DAccumulator(header.nbins, dv_kms)
    acc_B = P1DAccumulator(header.nbins, dv_kms)
    for s, e, tau in iter_tau_batches(hdf5, batch_size=4096, n_skewers=n_total):
        tau = tau.astype(np.float64)
        is_d = dirty[s:e]
        if (~is_d).any():
            acc_cl.add_batch(tau[~is_d], mean_F_global=mean_F_cl)
        if is_d.any():
            acc_di.add_batch(tau[is_d], mean_F_global=mean_F_di)
            tA = apply_tauspace_mask_to_batch(tau, s, cat, ["LLS","subDLA","DLA"], tau_eff=tau_eff)
            tB = apply_mask_to_batch(tau, s, cat, ["LLS","subDLA","DLA"], strategy="mean_flux")
            acc_A.add_batch(tA[is_d], mean_F_global=mean_F_A)
            acc_B.add_batch(tB[is_d], mean_F_global=mean_F_B)

    k_native, P_cl = acc_cl.raw_power()
    _, P_di = acc_di.raw_power()
    _, P_A = acc_A.raw_power()
    _, P_B = acc_B.raw_power()
    return {
        "k": k_native, "header": header, "dv_kms": dv_kms,
        "P_clean": P_cl, "P_dirty": P_di, "P_maskA": P_A, "P_maskB": P_B,
        "mean_F": {"all": mean_F_all, "clean": mean_F_cl, "dirty": mean_F_di,
                   "mA": mean_F_A, "mB": mean_F_B},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skewers", type=int, default=100000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); DIAG.mkdir(parents=True, exist_ok=True)

    print(f"SIM: {SIM[:40]}")
    print(f"\n[LF]  {LF_HDF5}")
    lf = run_pipeline(LF_HDF5, "lf_ns0909_s017", args.n_skewers)
    print(f"\n[HR]  {HR_HDF5}")
    hr = run_pipeline(HR_HDF5, "hr_ns0909_s017", args.n_skewers)

    # Table
    print(f"\n{'k [s/km]':>10}  {'LF template':>12}  {'HR template':>12}  "
          f"{'LF leftA':>10}  {'HR leftA':>10}  {'LF leftB':>10}  {'HR leftB':>10}")
    for k_tgt in [0.001, 0.002, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050]:
        j_lf = int(np.argmin(np.abs(lf["k"] - k_tgt)))
        j_hr = int(np.argmin(np.abs(hr["k"] - k_tgt)))
        tmpl_lf = lf["P_dirty"][j_lf] / lf["P_clean"][j_lf]
        tmpl_hr = hr["P_dirty"][j_hr] / hr["P_clean"][j_hr]
        lefA_lf = lf["P_maskA"][j_lf] / lf["P_clean"][j_lf]
        lefA_hr = hr["P_maskA"][j_hr] / hr["P_clean"][j_hr]
        lefB_lf = lf["P_maskB"][j_lf] / lf["P_clean"][j_lf]
        lefB_hr = hr["P_maskB"][j_hr] / hr["P_clean"][j_hr]
        print(f"  {k_tgt:>8.4f}  {tmpl_lf:>12.3f}  {tmpl_hr:>12.3f}  "
              f"{lefA_lf:>10.3f}  {lefA_hr:>10.3f}  {lefB_lf:>10.3f}  {lefB_hr:>10.3f}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(lf["k"], lf["P_dirty"]/lf["P_clean"], lw=1.8, color="C0",
                label=f"LF  (z={lf['header'].redshift:.2f})")
    ax[0].plot(hr["k"], hr["P_dirty"]/hr["P_clean"], lw=1.8, color="C3",
                label=f"HR  (z={hr['header'].redshift:.2f})")
    ax[0].axhline(1.0, color="gray", ls="--", alpha=0.5)
    for kv in (0.02, 0.05, 0.10):
        ax[0].axvline(kv, color="gray", ls=":", alpha=0.4)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("P_dirty / P_clean")
    ax[0].set_title("HCD template: LF vs HR convergence"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_xlim(1e-3, 0.1)

    ax[1].plot(lf["k"], lf["P_maskA"]/lf["P_clean"], lw=1.8, color="C0",
                ls="-", label="LF τ-space")
    ax[1].plot(hr["k"], hr["P_maskA"]/hr["P_clean"], lw=1.8, color="C3",
                ls="-", label="HR τ-space")
    ax[1].plot(lf["k"], lf["P_maskB"]/lf["P_clean"], lw=1.5, color="C0",
                ls="--", label="LF pixrange")
    ax[1].plot(hr["k"], hr["P_maskB"]/hr["P_clean"], lw=1.5, color="C3",
                ls="--", label="HR pixrange")
    ax[1].axhline(1.0, color="gray", ls="--", alpha=0.5)
    for kv in (0.02, 0.05, 0.10):
        ax[1].axvline(kv, color="gray", ls=":", alpha=0.4)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("P_masked / P_clean")
    ax[1].set_title("Mask quality: LF vs HR"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_xlim(1e-3, 0.1); ax[1].set_ylim(0.5, 3.0)

    fig.suptitle(f"{SIM[:40]}...  snap_017  (n_skewers={args.n_skewers})")
    fig.tight_layout()
    out = DIAG / "hires_vs_lf_subset.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()
