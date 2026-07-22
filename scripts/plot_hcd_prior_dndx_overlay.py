"""THE PI "show-me" overlay (2026-06-17 HCD prior re-determination): the PRIOR dN/dX(z) BAND under
the NEW litWLS center, with BOTH the 1× and 2× σ_LLS envelopes (shaded), overlaid with the
literature MEASUREMENTS (points + error bars) AND the sim/truth dN/dX(z) — per class (LLS/subDLA/DLA)
per survey (DESI / KS / DESI+KS / eBOSS). The PI can SEE whether the prior band brackets the
measurements at 1× vs 2×.

THE PRIOR BAND construction (matches the production forward exactly):
  center α_c(z) = α_pivot_c · ((1+z)/(1+z_p))^s_c,  α_pivot_c = the per-survey hcd_incidence_prior μ,
  s_c = the REAL-FIT forward z-slope (litWLS γ_LLS=2.127 for LLS; sim incidence slope for subDLA/DLA),
  → dN/dX_c(z) = alpha_to_dndx_exact(α_c(z), Xbar(z), z)  (the dndx_wc EXACT inverse, renorm included;
    fail-loud outside the occupancy simplex — 2026-07-22 readout defect B).
  The 1×/2× σ_LLS envelopes scale α_pivot_LLS by (1 ± σ/μ) at the deployed per-survey widths
(HCD_LLS_SURVEY_FRAC_SIGMA / _HEDGE2X; 0.287/0.574 DESI-family, 0.40 KS — corrected-law widths
2026-07-18) (the PI WIDTH RULE
  1× lit measurement error, 2× cosmic-variance hedge). subDLA/DLA use their fixed σ/μ (0.40, 0.50).
  DLA is the 10%-residual center (HCD_DLA_RESIDUAL_FRAC), so the DLA literature points are SCALED by
  0.10 to overlay on the same residual axis.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_hcd_prior_dndx_overlay.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # x64 before jax
import jax.numpy as jnp
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
from hcd_analysis.emulator.inference import (
    hcd_incidence_prior, HCD_Z_PIVOT, HCD_DLA_RESIDUAL_FRAC, HCD_PRIOR_FRAC_SIGMA,
    HCD_LLS_SURVEY_FRAC_SIGMA, HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X, HCD_LLS_REALFIT_ZSLOPE)
from hcd_analysis.emulator.closure_legb import CACHE_PATH, HCD_INCIDENCE_SLOPE

OUTDIR = Path("/home/mfho/hcd_priya_notes/figures/analysis/03_templates_and_p1d")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "hcd_prior_dndx_overlay_measurements.png"

CLS = ["LLS", "subDLA", "DLA"]
ZP = float(HCD_Z_PIVOT)

# Literature dN/dX (z, value, ±err, source) — the CORRECTED estimands (re-derivation
# 2026-07-18), single source hcd_analysis/emulator/lit_dndx.lit_points_for_display
# (old wrong-object arrays tombstoned there).
from hcd_analysis.emulator.lit_dndx import lit_points_for_display
LIT = lit_points_for_display()

# Per-survey z grids (match scratch_hcd_dndx_loso_vs_lit). Boost applied per survey (KS×2.5).
SURVEYS = ["DESI", "KS", "DESI+KS", "eBOSS"]
SURVEY_Z = {"DESI": np.arange(2.2, 4.21, 0.1), "KS": np.arange(2.4, 4.61, 0.1),
            "DESI+KS": np.arange(2.4, 4.61, 0.1), "eBOSS": np.arange(2.2, 4.61, 0.1)}


def build_xbar_and_sim_dndx():
    """Xbar(z) = X_tot/N_sl (mean absorption path per sightline) from the LF cache, smoothed in z;
    plus the sim-mean dN/dX(z) per class (the truth band) binned from the cache snap_dNdX."""
    d = load_cache(CACHE_PATH)
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    Xtot = np.asarray(d["snap_total_path_dX"]); wc = np.asarray(d["w_c_cache"])
    dndx = np.asarray(d["snap_dNdX"])                         # (Ngroup, 3) LLS, subDLA, DLA
    Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(axis=1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    xbar_fn = lambda z: np.polyval(cf, np.asarray(z))
    # sim-mean dN/dX(z) per class (the truth)
    w_c_med = np.median(np.asarray(d["w_c_cache"])[:, 1:], axis=0)   # the prior-center structural weights
    return xbar_fn, zg, dndx, w_c_med


def dndx_band_for_survey(survey, zgrid, w_c_med, xbar_fn):
    """The PRIOR dN/dX(z) center + 1×/2× σ_LLS envelopes for one survey, per class (nz,3).

    center α_pivot_c = hcd_incidence_prior(survey).μ ; s_c = (litWLS LLS, sim subDLA, sim DLA);
    1×/2× envelopes scale α_pivot_LLS by (1 ± σ/μ_LLS) and α_pivot_{sub,DLA} by (1 ± σ/μ_{sub,DLA})."""
    # the per-survey α-prior μ (center) and σ at the 1× (primary) and 2× (hedge) LLS widths.
    mu1, sd1 = map(np.asarray, hcd_incidence_prior(jnp.asarray(w_c_med), z=ZP, survey=survey))
    mu2, sd2 = map(np.asarray, hcd_incidence_prior(jnp.asarray(w_c_med), z=ZP, survey=survey,
                                                   use_lls_width_hedge2x=True))
    # forward z-slope: litWLS for LLS (the real-fit slope), sim incidence slope for subDLA/DLA.
    s_c = np.array([HCD_LLS_REALFIT_ZSLOPE, float(HCD_INCIDENCE_SLOPE[1]), float(HCD_INCIDENCE_SLOPE[2])])
    nz = len(zgrid)
    shape = ((1.0 + zgrid)[:, None] / (1.0 + ZP)) ** s_c[None, :]    # (nz,3) z-evolution
    Xb = np.asarray(xbar_fn(zgrid))

    def to_dndx(alpha_pivot):
        # EXACT inverse, mode="raise" (prior centres/edges): out-of-simplex refuses loudly
        # (expected for the current KS prior at z >= ~4.45) instead of silently saturating.
        az = alpha_pivot[None, :] * shape                            # (nz,3)
        out = np.empty((nz, 3))
        for j in range(nz):
            out[j] = alpha_to_dndx_exact(az[j], float(Xb[j]), float(zgrid[j]))
        return out

    center = to_dndx(mu1)
    # 1× / 2× envelopes: μ ± σ on each class pivot (the prior 1σ band), propagated through the map.
    lo1 = to_dndx(np.clip(mu1 - sd1, 1e-8, None)); hi1 = to_dndx(mu1 + sd1)
    lo2 = to_dndx(np.clip(mu2 - sd2, 1e-8, None)); hi2 = to_dndx(mu2 + sd2)
    return dict(center=center, lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2, sd1=sd1, mu1=mu1)


def sim_truth_dndx(zgrid, survey, zg, dndx_cache, w_c_med, xbar_fn):
    """The sim/truth dN/dX(z): the cache sim-mean dN/dX per class, with the survey LLS boost (KS×2.5)
    and the DLA 10%-residual scaling — so it sits on the SAME residual axis as the prior band/points."""
    boost = 2.5 if survey == "KS" else 1.0
    out = np.full((len(zgrid), 3), np.nan)
    for j in range(3):
        inr = (zg >= 2.1) & (zg <= 4.7) & (dndx_cache[:, j] > 0)
        # sim-mean dN/dX(z) binned in z, interpolated onto zgrid
        zb, vb = [], []
        for zz in np.unique(np.round(zg[inr], 1)):
            sel = np.isclose(zg, zz, atol=0.05) & inr
            if sel.sum() >= 1:
                zb.append(zz); vb.append(np.nanmean(dndx_cache[sel, j]))
        if len(zb) >= 2:
            sm = np.interp(zgrid, zb, vb, left=np.nan, right=np.nan)
            if j == 0:
                sm = sm * boost                                  # LLS survey boost
            elif j == 2:
                sm = sm * HCD_DLA_RESIDUAL_FRAC                  # DLA 10% residual
            out[:, j] = sm
    return out


def main():
    xbar_fn, zg, dndx_cache, w_c_med = build_xbar_and_sim_dndx()

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(3, 4, figsize=(26, 17), sharex="col")

    brackets = {}   # (survey,cls) -> dict(in1, in2, npts) bracketing diagnostics
    for jc, cls in enumerate(CLS):
        zl, vl, el, src = LIT[cls]
        zl = np.array(zl); vl = np.array(vl); el = np.array(el)
        dla_scale = HCD_DLA_RESIDUAL_FRAC if cls == "DLA" else 1.0
        for js, survey in enumerate(SURVEYS):
            ax = axes[jc, js]
            zgrid = SURVEY_Z[survey]
            band = dndx_band_for_survey(survey, zgrid, w_c_med, xbar_fn)
            c = band["center"][:, jc]; lo1 = band["lo1"][:, jc]; hi1 = band["hi1"][:, jc]
            lo2 = band["lo2"][:, jc]; hi2 = band["hi2"][:, jc]
            # 2× envelope (wider, lighter) UNDER the 1× envelope (tighter, darker)
            ax.fill_between(zgrid, lo2, hi2, color="tab:orange", alpha=0.18,
                            label=(r"prior $2\times\sigma_{LLS}$ "
                                   f"({HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X['DESI']}, hedge)")
                            if (jc == 0 and js == 0) else None)
            ax.fill_between(zgrid, lo1, hi1, color="tab:blue", alpha=0.30,
                            label=(r"prior $1\times\sigma_{LLS}$ "
                                   f"({HCD_LLS_SURVEY_FRAC_SIGMA['DESI']}, primary)")
                            if (jc == 0 and js == 0) else None)
            ax.plot(zgrid, c, "-", color="navy", lw=2.4,
                    label="prior center (litWLS)" if (jc == 0 and js == 0) else None)
            # sim/truth dN/dX(z)
            st = sim_truth_dndx(zgrid, survey, zg, dndx_cache, w_c_med, xbar_fn)[:, jc]
            ax.plot(zgrid, st, "--", color="tab:green", lw=2.0,
                    label="sim/truth dN/dX" if (jc == 0 and js == 0) else None)
            # literature measurements (DLA scaled to the 10% residual)
            ax.errorbar(zl, vl * dla_scale, yerr=el * dla_scale, fmt="D", color="k", ms=7,
                        capsize=3, zorder=10,
                        label="literature" if (jc == 0 and js == 0) else None)

            # bracketing diagnostic: fraction of lit points (±err) inside the 1× / 2× band envelope
            def frac_in(lo_env, hi_env):
                loi = np.interp(zl, zgrid, lo_env, left=np.nan, right=np.nan)
                hii = np.interp(zl, zgrid, hi_env, left=np.nan, right=np.nan)
                ok = np.isfinite(loi) & np.isfinite(hii)
                # the lit point's ±1σ interval OVERLAPS the band envelope
                inside = (vl * dla_scale + el * dla_scale >= loi) & (vl * dla_scale - el * dla_scale <= hii)
                return int(np.sum(inside & ok)), int(np.sum(ok))
            n1, npts = frac_in(lo1, hi1); n2, _ = frac_in(lo2, hi2)
            brackets[(survey, cls)] = dict(in1=n1, in2=n2, npts=npts)

            ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            if jc == 0:
                ax.set_title(survey, fontweight="bold")
            if js == 0:
                ax.set_ylabel(f"{cls}\n" + (r"dN/dX (10% resid.)" if cls == "DLA" else "dN/dX"))
            if jc == 2:
                ax.set_xlabel("z")
            # per-class y-limits (consistent across surveys; keeps the KS×2.5-boosted band readable
            # without a clip-floor spike from a low 2× envelope edge).
            YLIM = {"LLS": (0.08, 4.0), "subDLA": (3e-2, 0.35), "DLA": (2e-4, 0.3)}
            ax.set_ylim(*YLIM[cls])
            # annotate the bracketing tally (KS LLS=0/N is EXPECTED — the KS prior is the ×2.5
            # selection-boosted level, ABOVE the cosmic-average lit points, by design).
            tag = f"lit in band: 1×={n1}/{npts}  2×={n2}/{npts}"
            if survey == "KS" and cls == "LLS":
                tag += "\n(KS prior ×2.5 boosted: ABOVE\ncosmic-avg lit BY DESIGN)"
            ax.annotate(tag, xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9.5,
                        va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    # ONE figure legend (top), from the (0,0) handles
    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="upper center", ncol=5, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "HCD prior dN/dX(z) band under the NEW litWLS center vs literature measurements + sim/truth\n"
        f"LLS forward z-slope γ_LLS={HCD_LLS_REALFIT_ZSLOPE} (corrected lit law, constrained); σ_LLS "
        f"1×={HCD_LLS_SURVEY_FRAC_SIGMA['DESI']} (meas+kernel common-mode) / "
        f"2×={HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X['DESI']} (hedge); subDLA σ/μ={HCD_PRIOR_FRAC_SIGMA[1]}, "
        f"DLA σ/μ={HCD_PRIOR_FRAC_SIGMA[2]}; DLA on the 10%-unmasked residual axis (lit ×0.10)",
        fontsize=16, y=1.055)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

    # bracketing read-out (printed for the PI)
    print("\n=== bracketing: lit points (±1σ) overlapping the prior band envelope ===")
    for cls in CLS:
        tot1 = sum(brackets[(s, cls)]["in1"] for s in SURVEYS)
        tot2 = sum(brackets[(s, cls)]["in2"] for s in SURVEYS)
        totn = sum(brackets[(s, cls)]["npts"] for s in SURVEYS)
        print(f"  {cls:7s}  1× band brackets {tot1}/{totn} lit pts (all surveys);  2× brackets {tot2}/{totn}")
    for s in SURVEYS:
        b = brackets[(s, "LLS")]
        print(f"    LLS {s:8s}: 1×={b['in1']}/{b['npts']}  2×={b['in2']}/{b['npts']}")


if __name__ == "__main__":
    main()
