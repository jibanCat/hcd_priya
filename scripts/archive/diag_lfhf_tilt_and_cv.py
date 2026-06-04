"""Two read-only measurements that feed the multi-fidelity (MF) emulator design
and its error budget. Nothing here touches the production model/data/train/
likelihood -- it only reads the two tau0 caches and the paired CV run.

The MF emulator is P_MF(theta,k) = rho(k[,theta])*f_LF(theta,k) + delta(theta,k):
the LF suite (60 sims) carries the parameter dependence and the few HR sims
supply a parameter-dependent RESOLUTION CORRECTION delta at high k. This script
characterizes (1) the PHYSICAL LF<->HR resolution tilt that delta must learn,
measured directly on matched (sim,z,tau0) raw cache spectra, and (2) the
cosmic-variance (CV) floor from the fixed+paired HR run.

MEASUREMENT 1 -- physical LF<->HR resolution tilt
  The 6 HR cosmologies are EXACT subsets of the 60 LF design points (unit-cube
  distance 0.0; target_F matches to 0.0 so the tau0/mean-flux is identical and
  the ratio is a pure resolution effect, NOT a normalization artifact). For each
  matched (sim,z,alpha) and class we form P_HR/P_LF in the LF-finite k-range,
  interpolating HR onto the LF k-grid in log-log space (the grids differ ~6% in
  spacing: same dv~10 km/s, different box -> nbins_native). We quantify the
  amplitude, the k-slope (%/dex), and the spread across the 6 shared cosmologies
  (the thing that makes delta parameter-dependent rather than a fixed kernel).

MEASUREMENT 2 -- cosmic variance from the pair-fixed run
  cv_correction_flux_vectors_tau1000000.hdf5 holds one cosmology (ns0.979),
  HR resolution, as 2 realizations: base + phase-inverted "paired", each
  (13 z, 525 k). The fixed-paired MEAN 0.5(base+paired) is CV-suppressed; the
  per-(z,k) CV a single non-paired realization carries is ~0.5*|base-paired|
  (the pairing cancels the in-phase mode; the half-difference estimates the
  per-realization residual). We report the fractional per-(z,k) sigma in the
  low-k, KODIAQ (0.07-0.2), and small-scale bands, and compare it to the
  emulator residual-head generalization floor.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_lfhf_tilt_and_cv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache, normalize_params

OUT = Path("figures/analysis/04_emulator")
LF_PATH = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_PATH = "hcd_analysis/_emulator_data/observables_tau0_hr.h5"
CV_PATH = "/nfs/turbo/umor-yueyingn/mfho/emu_full_pf/cv_correction_flux_vectors_tau1000000.hdf5"
CLS = ("clean", "LLS", "subDLA", "DLA")

# KODIAQ survey band (the high-k that the MF result rests on).
KODIAQ_BAND = (0.07, 0.20)
# Lower edge of the clean overlap fit: skip the very-low-k modes (few of them,
# CV-dominated) and the LF-Nyquist roll-off near k_max.
TILT_FIT_LO = 0.01

# Emulator residual-head generalization floor and sigma_cosmo, from
# figures/analysis/04_emulator/feasibility_subpercent.json (exp3 nb24_e250
# a_val_sc, and sc_over_sm_perclass). Used only for the CV-vs-floor comparison.
SIGMA_COSMO_FRAC = {"clean": 0.0772, "LLS": 0.0791, "subDLA": 0.0784, "DLA": 0.0852}
RESID_FLOOR_SC = {"clean": 0.163, "LLS": 0.167, "subDLA": 0.214, "DLA": 0.568}


# ===========================================================================
# Row matching
# ===========================================================================
def _rowkey(d):
    """(params_unit rounded, z rounded, alpha_idx) key for exact row matching.

    Matching is on the unit-cube params (not sim_name strings): one HR sim
    differs from its LF twin only by name rounding (bhfeedback0.04), but all 6
    HR cosmologies hit LF design points at unit-cube distance 0.0."""
    pu = np.round(normalize_params(d["params"]), 6)
    z = np.round(d["z_grid"], 4)
    a = d["alpha_idx"].astype(int)
    return [(tuple(pu[i]), z[i], a[i]) for i in range(len(z))]


def match_hr_to_lf(lf, hr):
    idx = {k: i for i, k in enumerate(_rowkey(lf))}
    pairs = [(h, idx[k]) for h, k in enumerate(_rowkey(hr)) if k in idx]
    return pairs


def hr_over_lf_ratio(P_hr, k_hr, P_lf, k_lf):
    """P_HR/P_LF on the LF-finite k-grid, HR log-log interpolated onto LF k.

    Returns (k_overlap, ratio). Only LF bins with finite positive P are kept;
    HR is interpolated there from its finite positive bins (HR k-max >> LF
    k-max so no extrapolation in the overlap)."""
    ml = np.isfinite(P_lf) & (P_lf > 0)
    mh = np.isfinite(P_hr) & (P_hr > 0)
    ks = k_lf[ml]
    hr_i = np.exp(np.interp(np.log(ks), np.log(k_hr[mh]), np.log(P_hr[mh])))
    return ks, hr_i / P_lf[ml]


# ===========================================================================
# MEASUREMENT 1
# ===========================================================================
def measure_tilt(lf, hr, pairs):
    Plf, Phr = lf["P_filt"], hr["P_filt"]
    klf, khr = lf["kfkms"], hr["kfkms"]

    # Confirm the matched rows share the same tau0/mean flux (pure-resolution).
    dF = max(abs(hr["target_F"][h] - lf["target_F"][l]) for h, l in pairs)

    hr_ns = hr["params"][:, 0]
    cosmos = sorted(np.unique(hr_ns))
    # alpha closest to fiducial (tau0~mid). alpha_idx grid is 0..19; pick 10.
    alpha_fid = 10

    summary = {"n_matched": len(pairs), "max_dtargetF": float(dF),
               "n_shared_cosmologies": len(cosmos), "shared_ns": [float(x) for x in cosmos],
               "tilt_fit_lo": TILT_FIT_LO, "per_class": {}}

    # ---- per-class slope/amplitude at z=3, alpha_fid, averaged over cosmos ----
    for ci, c in enumerate(CLS):
        slopes, lowk, hik, meanr = [], [], [], []
        for h, l in pairs:
            if abs(hr["z_grid"][h] - 3.0) > 1e-3 or hr["alpha_idx"][h] != alpha_fid:
                continue
            ks, r = hr_over_lf_ratio(Phr[h, ci], khr[h], Plf[l, ci], klf[l])
            band = ks > TILT_FIT_LO
            if band.sum() < 3:
                continue
            sl = np.polyfit(np.log10(ks[band]), r[band], 1)[0]
            slopes.append(sl * 100.0)
            lowk.append(r[band][0])
            hik.append(r[band][-1])
            meanr.append(float(np.mean(r[band])))
        summary["per_class"][c] = {
            "z": 3.0, "alpha_idx": alpha_fid,
            "slope_pct_per_dex_mean": float(np.mean(slopes)),
            "slope_pct_per_dex_std": float(np.std(slopes)),
            "slope_pct_per_dex_range": [float(np.min(slopes)), float(np.max(slopes))],
            "lowk_ratio_mean": float(np.mean(lowk)),
            "lowk_ratio_std": float(np.std(lowk)),
            "hik_ratio_mean": float(np.mean(hik)),
            "mean_ratio_over_cosmos": float(np.mean(meanr)),
            "n_cosmos": len(slopes),
        }

    # ---- z-dependence of clean slope (one cosmo, alpha_fid) ----
    ns0 = cosmos[0]
    zdep = {}
    for h, l in pairs:
        if abs(hr["params"][h, 0] - ns0) > 1e-6 or hr["alpha_idx"][h] != alpha_fid:
            continue
        ks, r = hr_over_lf_ratio(Phr[h, 0], khr[h], Plf[l, 0], klf[l])
        band = (ks > TILT_FIT_LO) & (ks < 0.06)
        if band.sum() < 3:
            continue
        zv = float(round(hr["z_grid"][h], 1))
        zdep[zv] = {"slope_pct_per_dex": float(np.polyfit(np.log10(ks[band]), r[band], 1)[0] * 100),
                    "lowk_ratio": float(r[ks > TILT_FIT_LO][0])}
    summary["clean_z_dependence_ns%.3f" % ns0] = zdep

    return summary


def plot_tilt(lf, hr, pairs):
    Plf, Phr = lf["P_filt"], hr["P_filt"]
    klf, khr = lf["kfkms"], hr["kfkms"]
    cosmos = sorted(np.unique(hr["params"][:, 0]))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cosmos)))
    z_panels = (2.4, 3.0, 3.6, 4.4)
    alpha_fid = 10

    fig, axes = plt.subplots(len(CLS), len(z_panels),
                             figsize=(4 * len(z_panels), 3 * len(CLS)),
                             sharex=True)
    for ci, c in enumerate(CLS):
        for zi, zv in enumerate(z_panels):
            ax = axes[ci, zi]
            for cj, ns in enumerate(cosmos):
                for h, l in pairs:
                    if (abs(hr["params"][h, 0] - ns) > 1e-6
                            or abs(hr["z_grid"][h] - zv) > 1e-3
                            or hr["alpha_idx"][h] != alpha_fid):
                        continue
                    ks, r = hr_over_lf_ratio(Phr[h, ci], khr[h], Plf[l, ci], klf[l])
                    ax.semilogx(ks, r, color=cmap[cj], lw=1.2,
                                label=f"ns={ns:.3f}" if (ci == 0 and zi == 0) else None)
                    break
            ax.axhline(1.0, color="k", lw=0.7, ls=":")
            ax.axvspan(*KODIAQ_BAND, color="orange", alpha=0.10)
            ax.set_ylim(0.90, 1.06)
            if ci == 0:
                ax.set_title(f"z = {zv:.1f}")
            if zi == 0:
                ax.set_ylabel(f"{c}\n$P_{{HR}}/P_{{LF}}$")
            if ci == len(CLS) - 1:
                ax.set_xlabel("k [s/km]")
    axes[0, 0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Physical LF$\\to$HR resolution tilt $P_{HR}/P_{LF}$ "
                 "(matched sim, $\\tau_0$; the $\\delta$ correction MF must learn)\n"
                 "shaded = KODIAQ band 0.07$-$0.2 s/km; ratio rises with k then "
                 "rolls over at LF Nyquist", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "lfhf_tilt_ratio_panels.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ===========================================================================
# MEASUREMENT 2
# ===========================================================================
def measure_cv():
    with h5py.File(CV_PATH, "r") as h:
        fv = h["flux_vectors"][:].reshape(2, 13, 525)
        kf = h["kfkms"][:]
        z = h["zout"][:]
        params = [x.decode() if isinstance(x, bytes) else x for x in h["params"][:]]
    assert np.max(np.abs(kf[0] - kf[1])) == 0.0, "base/paired k-grids differ"
    k = kf[0]                       # (13, 525); per-z k-grid (all equal here)
    base, paired = fv[0], fv[1]
    mean = 0.5 * (base + paired)    # CV-suppressed fixed-paired mean
    # per-(z,k) CV a single non-paired realization carries:
    cv_frac = 0.5 * np.abs(base - paired) / np.where(mean > 0, mean, np.nan)

    kref = k[0]
    bins = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.29])
    kband = {}
    for i in range(len(bins) - 1):
        sel = (kref >= bins[i]) & (kref < bins[i + 1])
        if sel.sum() == 0:
            continue
        vals = cv_frac[:, sel]
        kband[f"{bins[i]:.3f}-{bins[i+1]:.3f}"] = {
            "median_pct": float(np.median(vals) * 100),
            "rms_pct": float(np.sqrt(np.mean(vals ** 2)) * 100),
            "n_modes": int(sel.sum()),
        }

    lowk = kref < 0.01
    kod = (kref >= KODIAQ_BAND[0]) & (kref <= KODIAQ_BAND[1])
    hik = kref > 0.20
    summary = {
        "cosmology": params[0].split("/")[-2],
        "n_realizations": 2, "n_z": 13, "n_k": 525,
        "z_range": [float(z.min()), float(z.max())],
        "cv_frac_per_kbin": kband,
        "cv_frac_lowk_median_pct": float(np.nanmedian(cv_frac[:, lowk]) * 100),
        "cv_frac_kodiaq_median_pct": float(np.nanmedian(cv_frac[:, kod]) * 100),
        "cv_frac_smallscale_median_pct": float(np.nanmedian(cv_frac[:, hik]) * 100),
    }

    # ---- compare to the residual-head generalization floor (clean channel) ----
    floor_clean = RESID_FLOOR_SC["clean"] * SIGMA_COSMO_FRAC["clean"]   # frac of P
    cv_kod = summary["cv_frac_kodiaq_median_pct"] / 100
    cv_low = summary["cv_frac_lowk_median_pct"] / 100
    summary["floor_comparison"] = {
        "clean_resid_floor_fracP": float(floor_clean),
        "clean_resid_floor_pct": float(floor_clean * 100),
        "cv_over_floor_amp_kodiaq": float(cv_kod / floor_clean),
        "cv_over_floor_amp_lowk": float(cv_low / floor_clean),
        "cv_over_floor_var_kodiaq": float((cv_kod / floor_clean) ** 2),
        "cv_over_floor_var_lowk": float((cv_low / floor_clean) ** 2),
        "per_class_floor_pct": {c: float(RESID_FLOOR_SC[c] * SIGMA_COSMO_FRAC[c] * 100)
                                for c in CLS},
    }
    return summary, z, kref, cv_frac, mean


def plot_cv(z, kref, cv_frac):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: CV fractional sigma vs k, one line per z
    ax = axes[0]
    cmap = plt.cm.plasma(np.linspace(0, 0.9, len(z)))
    for zi, zv in enumerate(z):
        ax.loglog(kref, cv_frac[zi] * 100, color=cmap[zi], lw=1.0,
                  label=f"z={zv:.1f}" if zi % 2 == 0 else None)
    ax.axvspan(*KODIAQ_BAND, color="orange", alpha=0.12, label="KODIAQ")
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel(r"per-realization CV  $0.5|P_{base}-P_{paired}|/\bar P$  [%]")
    ax.set_title("Residual cosmic variance vs k, per z\n(ns0.979, HR, fixed+paired)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # right: z-averaged CV vs k with the floor reference lines
    ax = axes[1]
    med = np.nanmedian(cv_frac, axis=0) * 100
    rms = np.sqrt(np.nanmean(cv_frac ** 2, axis=0)) * 100
    ax.loglog(kref, med, "C0", lw=1.8, label="median over z")
    ax.loglog(kref, rms, "C0", lw=1.0, ls="--", label="rms over z")
    floor_clean = RESID_FLOOR_SC["clean"] * SIGMA_COSMO_FRAC["clean"] * 100
    ax.axhline(floor_clean, color="C3", lw=1.3, ls="-",
               label=f"emu resid floor clean ({floor_clean:.2f}% P)")
    sc = SIGMA_COSMO_FRAC["clean"] * 100
    ax.axhline(sc, color="gray", lw=1.0, ls=":",
               label=f"$\\sigma_{{cosmo}}$ clean ({sc:.1f}% P)")
    ax.axvspan(*KODIAQ_BAND, color="orange", alpha=0.12)
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("fractional CV [%]")
    ax.set_title("CV floor vs emulator residual-head floor\n"
                 "CV is k-dependent (U-shape) -- NOT a flat 3%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = OUT / "cv_floor_vs_k.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ===========================================================================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lf = load_cache(LF_PATH)
    hr = load_cache(HR_PATH)
    pairs = match_hr_to_lf(lf, hr)

    tilt = measure_tilt(lf, hr, pairs)
    fig_tilt = plot_tilt(lf, hr, pairs)

    cv, z, kref, cv_frac, mean = measure_cv()
    fig_cv = plot_cv(z, kref, cv_frac)

    out = {"measurement_1_lfhf_tilt": tilt, "measurement_2_cosmic_variance": cv,
           "figures": [str(fig_tilt), str(fig_cv)]}
    jpath = OUT / "diag_lfhf_tilt_and_cv.json"
    jpath.write_text(json.dumps(out, indent=2))

    print("MEASUREMENT 1 -- LF->HR resolution tilt")
    print(f"  shared cosmologies: {tilt['n_shared_cosmologies']} "
          f"(matched rows {tilt['n_matched']}, max|dtarget_F|={tilt['max_dtargetF']:.1e})")
    for c in CLS:
        s = tilt["per_class"][c]
        print(f"  {c:7s} z3: slope={s['slope_pct_per_dex_mean']:+.2f}"
              f"+/-{s['slope_pct_per_dex_std']:.2f} %/dex "
              f"(range {s['slope_pct_per_dex_range'][0]:+.1f}..{s['slope_pct_per_dex_range'][1]:+.1f}), "
              f"lowk={s['lowk_ratio_mean']:.3f}+/-{s['lowk_ratio_std']:.3f}, "
              f"hik={s['hik_ratio_mean']:.3f}")
    print("\nMEASUREMENT 2 -- cosmic variance")
    print(f"  lowk(<0.01)={cv['cv_frac_lowk_median_pct']:.2f}%  "
          f"KODIAQ(0.07-0.2)={cv['cv_frac_kodiaq_median_pct']:.2f}%  "
          f"small-scale(>0.2)={cv['cv_frac_smallscale_median_pct']:.2f}%")
    fc = cv["floor_comparison"]
    print(f"  clean resid-head floor = {fc['clean_resid_floor_pct']:.2f}% P")
    print(f"  CV/floor (amp): KODIAQ={fc['cv_over_floor_amp_kodiaq']:.2f}  "
          f"lowk={fc['cv_over_floor_amp_lowk']:.2f}")
    print(f"  CV/floor (var): KODIAQ={fc['cv_over_floor_var_kodiaq']:.3f}  "
          f"lowk={fc['cv_over_floor_var_lowk']:.3f}")
    print(f"\nwrote {jpath}\n      {fig_tilt}\n      {fig_cv}")


if __name__ == "__main__":
    main()
