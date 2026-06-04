"""Diagnostic: decompose the forest+LLS+subDLA vs PRIYA (tau_thresh=1e6) P1D gap.

Question (user 2026-05-22): is the ~2% gap between the non-DLA per-class sum and
PRIYA's filtered P1D a NORMALIZATION offset (flat in k) or a k-dependent TILT
(from the DLA trough-fill removing power preferentially at some scales)?

Method: on the SAME filtered tau as Tier P (so the partition is exact),
  - P_nonDLA   = count-weighted sum of clean+LLS+subDLA classes (drop DLA)
  - P_dlacontr = count-weighted filtered-DLA contribution (classes 13,14)
  - P_nonDLA + P_dlacontr == P_p (=PRIYA), exact.
We plot ratio_nonDLA = P_nonDLA / PRIYA vs k. Flat -> normalization; sloped ->
tilt. We also show how forest-like the *healed* DLA sightlines are per scale
(P_DLA^filt / P_clean), which is the physical origin of any tilt.

Run (emu-3.9 + GSL env):
  export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
  /home/mfho/.conda/envs/emu-3.9/bin/python3 scripts/diag_tierc_priya_gap.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hcd_analysis.priya_p1d import compute_tier_p_p1d, compute_tier_c_p1d  # noqa: E402
from hcd_analysis.catalog import AbsorberCatalog  # noqa: E402

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP, NK, ZIDX, ROW = 17, 172, 8, 344
TAU = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
CAT = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
META = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
PRIYA = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
OUT = REPO / "figures" / "analysis" / "03_templates_and_p1d" / "tierc_priya_gap_z3.png"


def main():
    m = json.load(open(META))
    vmax = int(m["nbins"]) * float(m["dv_kms"])
    z = round(float(m["z"]) / 0.2) * 0.2
    with h5py.File(TAU, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(CAT)
    with h5py.File(PRIYA, "r") as f:
        alpha = float(f["params"][ROW, 0])
        P_priya_ref = f["flux_vectors"][ROW, ZIDX * NK:(ZIDX + 1) * NK].astype(np.float64)
        kk = f["kfkms"][ROW, ZIDX, :].astype(np.float64)      # s/km, native, NK bins

    # Tier P filters tau in place; reuse the SAME filtered tau for Tier C.
    tau_f = tau.copy()
    _, P_p, tF, scale = compute_tier_p_p1d(tau_f, vmax, alpha_slope=alpha, z=z)
    _, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau_f, vmax, alpha, z, catalog=catalog, external_scale=scale, external_target_F=tF)
    assert np.max(np.abs(P_p[:NK] / P_priya_ref - 1)) < 1e-4, "Tier P != PRIYA"

    N = int(n_by_bin.sum())
    n_dla = int(n_by_bin[13:].sum())
    P_priya = P_p[:NK]
    # count-weighted contributions (weights n/N) truncated to PRIYA's NK grid
    P_nonDLA = (n_by_bin[:13, None] / N * P_by_bin[:13]).sum(0)[:NK]
    P_dlacontr = (n_by_bin[13:, None] / N * P_by_bin[13:]).sum(0)[:NK]
    P_recon = P_nonDLA + P_dlacontr                            # == P_priya (exact)
    # per-sightline-average class P1Ds (how forest-like a healed DLA sightline is)
    P_clean_avg = P_by_bin[0][:NK]
    P_dla_avg = (n_by_bin[13:, None] * P_by_bin[13:]).sum(0)[:NK] / max(n_dla, 1)

    ratio_nonDLA = P_nonDLA / P_priya          # < 1 by the dropped-DLA contribution
    deficit = 1.0 - ratio_nonDLA               # = P_dlacontr / P_priya, the "gap" vs k

    # normalization-vs-tilt quantifiers
    lo = deficit[:NK // 4].mean()              # low-k quarter
    hi = deficit[-NK // 4:].mean()             # high-k quarter
    med = np.median(deficit)
    mx = deficit.max()
    print(f"z={z} alpha={alpha:.4f}  n_DLA={n_dla}/{N}={n_dla/N:.2%}")
    print(f"deficit (=1-nonDLA/PRIYA): median={med:.3%} max={mx:.3%}")
    print(f"  low-k quarter mean={lo:.3%}  high-k quarter mean={hi:.3%}  "
          f"tilt(hi-lo)={hi-lo:+.3%}")
    print(f"  -> {'TILT-dominated' if abs(hi-lo) > 0.5*med else 'NORMALIZATION-dominated'}")

    # ---- figure ----
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.07))
    # (a) P1D curves
    ax[0].loglog(kk, P_priya, "k-", lw=2.2, label="PRIYA (Tier P, τ-filter 1e6)")
    ax[0].loglog(kk, P_nonDLA, "C0--", lw=1.8, label="forest+LLS+subDLA (drop DLA)")
    ax[0].loglog(kk, P_recon, "C3:", lw=1.8, label="+ filtered-DLA  (= PRIYA, exact)")
    ax[0].set_ylabel(r"$P_{1D}(k)$  [km/s]")
    ax[0].legend(fontsize=8.5, loc="lower left")
    ax[0].set_title(f"Tier-C decomposition vs PRIYA  —  LF sim, z={z:.1f}, "
                    f"$n_{{DLA}}$={n_dla/N:.1%} of sightlines", fontsize=10)
    # (b) ratio to PRIYA — the normalization-vs-tilt panel
    ax[1].semilogx(kk, ratio_nonDLA, "C0-", lw=2,
                   label="forest+LLS+subDLA / PRIYA")
    ax[1].semilogx(kk, P_recon / P_priya, "C3:", lw=1.8, label="+filtered-DLA / PRIYA")
    ax[1].axhline(1.0, color="k", lw=0.7)
    ax[1].axhline(1.0 - med, color="C0", lw=0.7, ls=":",
                  label=f"median deficit {med:.2%}")
    ax[1].fill_between(kk, 0.99, 1.01, color="0.85", zorder=0, label="±1%")
    ax[1].set_ylabel("ratio to PRIYA")
    ax[1].set_ylim(1 - max(mx * 1.6, 0.03), 1 + 0.012)
    ax[1].legend(fontsize=8, loc="lower left", ncol=1)
    ax[1].text(0.97, 0.06,
               f"low-k {lo:.2%} → high-k {hi:.2%}  (tilt {hi-lo:+.2%})",
               transform=ax[1].transAxes, ha="right", fontsize=8.5,
               bbox=dict(boxstyle="round", fc="w", ec="0.6"))
    # (c) how forest-like are healed DLA sightlines, per scale
    ax[2].semilogx(kk, P_dla_avg / P_clean_avg, "C2-", lw=2,
                   label=r"$P_{DLA}^{\rm filt}/P_{\rm clean}$ (per-sightline avg)")
    ax[2].axhline(1.0, color="k", lw=0.7, label="= clean forest")
    ax[2].set_ylabel("healed-DLA / clean")
    ax[2].set_xlabel(r"$k$  [s/km]")
    ax[2].legend(fontsize=8.5, loc="best")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
