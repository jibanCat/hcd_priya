"""Diagnose the HCD excess templates in LINEAR units vs the Rogers+2018 FIXED template.

Rogers+2018 (MNRAS 476, 3716; arXiv:1706.08532) Eq. 6 — the per-class ratio
P1D_i / P1D_forest, a FIXED prediction (Table-2 coefficients, no free amplitude):

    P_i/P_forest(k,z) = ((1+z)/(1+z0))^(-3.55) * (a_i(z) e^{b_i(z) k} - 1)^-2 + c_i(z)

with z0=2, a_i=a0((1+z)/3)^a1, etc., and k the ANGULAR wavenumber (k=2*pi/lambda_v, s/km).
Rogers §3.2 (lines 540-554): "we use the convention of absorbing the 2*pi into the conjugate
variable" (FFT kernel e^{-ikx}); his fundamental k~9e-4 = 2*pi/7111 (L=7111 km/s @ z=2). The
WHOLE Lya community (Croft/McDonald/Palanque-Delabrouille/Rogers/PRIYA/DESI) uses this single
angular convention, and PRIYA/fake_spectra kfkms = 2*pi*rfftfreq/dv is the SAME angular grid
(fake_spectra _flux_power_bins: kf *= 2*pi*npix/vmax). So Rogers' b0 apply to kfkms DIRECTLY
(NO /2*pi) -- confirmed by two independent agents 2026-06-04.

Two questions this answers:
  Q1 "why not smooth?"  -> grey = raw per-sim cache ratios: the per-class P1D is measured
     from a finite # of sightlines (esp. rare DLAs) so the targets carry sample noise; the
     emulator (per-k flexible) tracks them rather than imposing a smooth shape. Rogers'
     2-3-coeff/class analytic form is smooth by construction.
  Q2 "are the normalization offsets real or a <P1D> sightline effect?" -> CONFIRMED REAL.
     BOTH our cache (priya_p1d.py:146, dflux=exp(-tau)/target_F-1, ALL classes share one
     global target_F) and Rogers (§3.2: "<F> is the average flux over ALL spectra at each
     redshift", Croft+1998) use a SINGLE GLOBAL <F>, NOT per-class. So sightline COUNT only
     adds noise (variance), never a mean offset. The offset is the real physical fact that
     contaminated sightlines are darker than the global mean -> their troughs carry genuine
     extra low-k power. Rogers ALSO says this normalization "must be allowed to float" (alpha0
     degenerate with mean-flux rescaling) and carries the high-k plateau in c(z) (c0=0.33 for
     large DLA -> a real sub-1 high-k suppression; Rogers is NOT pure-boost once c is included).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_hcd_template_vs_rogers.py
"""
from __future__ import annotations
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401  x64
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache, DATA_RANGE
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.predict import predict_P_filt

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
CKPT = str(ROOT / "checkpoints/final_fold0")
OUT = ROOT / "figures/analysis/05_likelihood/hcd_template_vs_rogers.png"
CLS = ("LLS", "subDLA", "DLA")
Z_FID = 3.0

# Rogers+2018 Table 2 (verbatim): [LLS, Sub-DLA, Small-DLA, Large-DLA]
RA0 = np.array([2.2001, 1.5083, 1.1415, 0.8633]); RA1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
RB0 = np.array([36.449, 81.388, 162.95, 429.58]); RB1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
RC0 = np.array([0.9849, 0.8667, 0.6572, 0.3339]); RC1 = np.array([-0.0631, 0.0196, 0.1169, 0.4653])
RZ0 = 2.0
# which Rogers class(es) overlay on each of our classes
ROGERS_FOR = {"LLS": [(0, "LLS")], "subDLA": [(1, "Sub-DLA")],
              "DLA": [(2, "Small-DLA"), (3, "Large-DLA")]}


def rogers_ratio(k_ang, z, i):
    """Rogers Eq.6 per-class ratio P_i/P_forest at ANGULAR k (s/km) for class index i."""
    zf = (1.0 + z) / (1.0 + RZ0)
    a = RA0[i] * zf ** RA1[i]; b = RB0[i] * zf ** RB1[i]; c = RC0[i] * zf ** RC1[i]
    zw = zf ** (-3.55)
    return zw * (a * np.exp(b * k_ang) - 1.0) ** -2 + c


def used_classes(P_filt4, dla_core):
    """Forward-model object: FILTERED for LLS/subDLA, UNFILTERED (filt+core) for DLA."""
    return P_filt4[0], np.stack([P_filt4[1], P_filt4[2], P_filt4[3] + dla_core])


def main():
    d = load_cache(CACHE)
    model, meta, norm = T.load_checkpoint(CKPT)
    pf = norm["P_filt"]
    z = d["z_grid"]
    rows = np.where(np.isclose(z, Z_FID, atol=0.03))[0]
    r0 = rows[np.argmin(np.abs(d["tau0"][rows] - np.median(d["tau0"][rows])))]
    kf = np.asarray(d["kfkms"][r0])                # ANGULAR (s/km) -- Rogers' own convention
    g = np.isfinite(kf) & (kf > 0)
    inr = g & (kf >= DATA_RANGE["k_min"])

    # emulated 4-class filtered power; build the as-used per-class power
    P_filt = np.asarray(predict_P_filt(model, jnp.asarray(d["x"][r0, :9]),
                                       float(d["x"][r0, 9]), float(d["tau0"][r0]), pf))
    P_clean, used = used_classes(P_filt, np.asarray(d["delta"][r0, 2]))
    ratio_emu = used / np.where(P_clean > 0, P_clean, np.nan)   # (3,K)
    excess_emu = used - P_clean

    # filtered DLA (cores masked) for the DLA-panel context line
    ratio_dla_filt = P_filt[3] / np.where(P_clean > 0, P_clean, np.nan)

    # raw per-sim cache ratios at z=3 (the sample-noise band)
    raw_rows = rows[np.linspace(0, len(rows) - 1, min(10, len(rows))).astype(int)]
    raw_ratios = []
    for r in raw_rows:
        Pc, us = used_classes(np.asarray(d["P_filt"][r]), np.asarray(d["delta"][r, 2]))
        raw_ratios.append(us / np.where(Pc > 0, Pc, np.nan))

    for ci, nm in enumerate(CLS):
        rg = "  ".join(f"{lab}:{rogers_ratio(kf[inr][0], Z_FID, i):.3f}" for i, lab in ROGERS_FOR[nm])
        print(f"{nm:7s} emu ratio@low-k={np.nanmedian(ratio_emu[ci][inr][:5]):.3f}  Rogers@low-k[{rg}]")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for ci, nm in enumerate(CLS):
        # --- row 0: RATIO P_c/P_clean (linear) — Rogers' native object ---
        a0 = ax[0, ci]
        for rr in raw_ratios:
            a0.plot(kf[inr], rr[ci][inr], color="0.75", lw=0.6, alpha=0.5, zorder=1)
        a0.plot(kf[inr], ratio_emu[ci][inr], color=f"C{ci}", lw=2.4, zorder=4,
                label="emulated  P_c/P_clean")
        if nm == "DLA":
            a0.plot(kf[inr], ratio_dla_filt[inr], color=f"C{ci}", lw=1.1, ls=":", zorder=3,
                    label="(filtered DLA, cores masked)")
        for i, lab in ROGERS_FOR[nm]:
            a0.plot(kf[inr], rogers_ratio(kf[inr], Z_FID, i), "--", lw=1.8, zorder=5,
                    label=f"Rogers+2018 {lab} (fixed)")
        a0.axhline(1.0, color="k", lw=0.5)
        ylim0 = {"LLS": (0.9, 1.5), "subDLA": (0.8, 2.4), "DLA": (0.0, 3.0)}[nm]
        a0.set_ylim(*ylim0)
        if nm == "DLA":
            a0.text(0.04, 2.6, "Rogers Large-DLA has a low-k POLE (a<1);\nclipped here",
                    fontsize=7, color="tab:red")
        a0.set_title(f"{nm}  —  ratio P_c/P_clean (linear)")
        a0.set_xlabel("k  [rad·s/km, angular]"); a0.grid(alpha=0.3)
        if ci == 0:
            a0.set_ylabel("P_c / P_clean")
            a0.plot([], [], color="0.75", lw=1, label="raw per-sim cache (sample noise)")
        a0.legend(fontsize=7, loc="upper right")

        # --- row 1: EXCESS R_c (linear), y-zoom to reveal high-k behaviour ---
        a1 = ax[1, ci]
        a1.plot(kf[inr], excess_emu[ci][inr], color=f"C{ci}", lw=2.4,
                label="emulated  R_c = P_c − P_clean")
        for i, lab in ROGERS_FOR[nm]:
            rog_ex = P_clean * (rogers_ratio(kf, Z_FID, i) - 1.0)
            a1.plot(kf[inr], rog_ex[inr], "--", lw=1.8, label=f"Rogers {lab} excess")
        a1.axhline(0.0, color="k", lw=0.5)
        hi = excess_emu[ci][inr]
        lim = np.nanpercentile(np.abs(hi[5:]), 97) if np.isfinite(hi[5:]).any() else 1.0
        a1.set_ylim(-1.5 * lim, 1.5 * lim)
        a1.set_title(f"{nm}  —  excess R_c (linear, y-zoom)")
        a1.set_xlabel("k  [rad·s/km, angular]"); a1.grid(alpha=0.3)
        if ci == 0:
            a1.set_ylabel("R_c  [P1D units]")
        a1.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "HCD excess templates (LINEAR) vs Rogers+2018 FIXED template (Eq.6 + c(z), cyclic k) — fold0, z=3\n"
        "Both use a single GLOBAL ⟨F⟩ (cache target_F=0.69 / Rogers Croft-⟨F⟩): the low-k offset is REAL "
        "(contaminated sightlines darker than the global mean), NOT a sightline-count effect. Rogers' c(z) "
        "plateau (c0_LargeDLA=0.33) gives a genuine sub-1 high-k suppression. Grey = raw per-sim cache scatter.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
