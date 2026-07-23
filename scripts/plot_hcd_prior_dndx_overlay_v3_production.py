"""v3 PRODUCTION overlay (PI 2026-06-17, the HCD-pivot CENTER-construction fix wired into production):
the PRIOR dN/dX(z) band built from the PRODUCTION center construction — NOT the scratch — overlaid
with the literature MEASUREMENTS (points + err) and the sim/truth dN/dX(z), per class per survey.
Confirms the low-z LLS points are bracketed at 1×σ.

WHAT IS "PRODUCTION" HERE (the fix): the α-pivot CENTER is built EXACTLY as build_legb_ctx /
run_stepA now build it after the CENTER-construction fix:
  - the z=3 STRUCTURAL w_c from the cache (closure_legb.hcd_pivot_wc_and_xbar) — NOT the all-z median
    nanmedian(w_c_cache[:,1:]) (= z≈3.6; the dN/dX low-z overshoot bug that put α_LLS 1.45× too high);
  - the REAL-FIT LLS pivot from the lit dN/dX law DIRECTLY (inference.hcd_lls_realfit_alpha_center,
    alt-(b) ≈0.172×boost (corrected law of record 2026-07-18), round-trips the lit dN/dX <0.2%);
  - subDLA/DLA pivots from hcd_incidence_prior at the z=3 w_c (lit/sim 1.00/1.34, DLA 10% residual).
The forward z-evolution + the σ_LLS 1×/2× envelopes are the production knobs (litWLS γ_LLS=2.127 for
LLS, sim incidence slope for subDLA/DLA; σ_LLS 0.287/0.574 DESI, 0.40 KS — corrected-law widths
2026-07-18). This is the v2 fix promoted
to the actual production functions (a future revert to the all-z median trips the pivot guard).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_hcd_prior_dndx_overlay_v3_production.py
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
    hcd_incidence_prior, hcd_lls_realfit_alpha_center, assert_hcd_pivot_z3,
    HCD_Z_PIVOT, HCD_DLA_RESIDUAL_FRAC, HCD_PRIOR_FRAC_SIGMA,
    HCD_LLS_SURVEY_BOOST, HCD_LLS_REALFIT_ZSLOPE)
from hcd_analysis.emulator.closure_legb import (
    CACHE_PATH, HCD_INCIDENCE_SLOPE, hcd_pivot_wc_and_xbar)

OUTDIR = Path("/home/mfho/hcd_priya_notes/figures/analysis/03_templates_and_p1d")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "hcd_prior_dndx_overlay_measurements_v3_production.png"

CLS = ["LLS", "subDLA", "DLA"]
ZP = float(HCD_Z_PIVOT)

# Literature dN/dX — the CORRECTED estimands (re-derivation 2026-07-18), single source
# lit_dndx.lit_points_for_display (old wrong-object arrays tombstoned there).
from hcd_analysis.emulator.lit_dndx import lit_points_for_display
LIT = lit_points_for_display()
SURVEYS = ["DESI", "KS", "DESI+KS", "eBOSS"]
SURVEY_Z = {"DESI": np.arange(2.2, 4.21, 0.1), "KS": np.arange(2.4, 4.61, 0.1),
            "DESI+KS": np.arange(2.4, 4.61, 0.1), "eBOSS": np.arange(2.2, 4.61, 0.1)}
LOWZ_CUT = 2.7        # the PI low-z definition (the leak region the prior MUST bracket)
# 1x/2x width dicts (deployed; LLS DESI-family 0.287/0.574, KS 0.40; subDLA/DLA same as 1x).
from hcd_analysis.emulator.inference import HCD_LLS_SURVEY_FRAC_SIGMA as SIG1_LLS
from hcd_analysis.emulator.inference import HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X as SIG2_LLS


def production_pivot(survey, Xbar_z3, w_c_z3):
    """The PRODUCTION α-pivot μ (LLS via the lit-law center, subDLA/DLA via hcd_incidence_prior at the
    z=3 w_c) + the σ/μ widths at 1× and 2×. This MIRRORS build_legb_ctx(survey=…) exactly."""
    mu, sd = map(np.asarray, hcd_incidence_prior(jnp.asarray(w_c_z3), z=ZP, survey=survey))
    boost = HCD_LLS_SURVEY_BOOST.get(survey, 1.0)
    # PRODUCTION real-fit LLS center = the lit dN/dX law directly (alt-(b)):
    a_lls = hcd_lls_realfit_alpha_center(Xbar_z3, z=ZP, boost=boost)
    mu = mu.copy(); mu[0] = a_lls
    assert_hcd_pivot_z3(float(mu[0]), z=ZP, where=f"v3 overlay survey={survey}", boost=boost)
    # widths: σ/μ at 1× (already in sd via hcd_incidence_prior; rescale LLS to the new center) and 2×.
    fl1 = float(sd[0] / np.asarray(hcd_incidence_prior(jnp.asarray(w_c_z3), z=ZP, survey=survey)[0])[0])
    sd = sd.copy(); sd[0] = fl1 * mu[0]
    sd2 = sd.copy(); sd2[0] = SIG2_LLS.get(survey, fl1) * mu[0]
    return mu, sd, sd2


def to_dndx_curve(alpha_pivot, zgrid, Xb, s_c):
    # EXACT inverse, mode="raise" (prior centres/edges): out-of-simplex refuses loudly
    # (expected for the current KS prior at z >= ~4.45) instead of silently saturating.
    az = alpha_pivot[None, :] * (((1.0 + zgrid)[:, None] / (1.0 + ZP)) ** s_c[None, :])  # (nz,3)
    return np.asarray(alpha_to_dndx_exact(az, np.asarray(Xb, float), np.asarray(zgrid, float)))


def build_xbar_fn_and_sim(d):
    """Xbar(z) deg-2 fit + the cache sim-mean dN/dX(z) per class (the truth), binned in z."""
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    Xtot = np.asarray(d["snap_total_path_dX"]); wc = np.asarray(d["w_c_cache"])
    dndx = np.asarray(d["snap_dNdX"]); Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    xbar_fn = lambda z: np.polyval(cf, np.asarray(z))
    return xbar_fn, zg, dndx


def main():
    d = load_cache(CACHE_PATH)
    w_c_z3, Xbar_z3 = hcd_pivot_wc_and_xbar(d)
    xbar_fn, zg, dndx_cache = build_xbar_fn_and_sim(d)
    s_c = np.array([HCD_LLS_REALFIT_ZSLOPE, float(HCD_INCIDENCE_SLOPE[1]), float(HCD_INCIDENCE_SLOPE[2])])

    print(f"PRODUCTION pivot inputs: z=3 w_c={w_c_z3}, Xbar(z=3)={Xbar_z3:.4f}")
    mu_desi, _, _ = production_pivot("DESI", Xbar_z3, w_c_z3)
    print(f"PRODUCTION α-pivot (DESI): LLS={mu_desi[0]:.4f} subDLA={mu_desi[1]:.4f} DLA={mu_desi[2]:.5f}")

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(3, 4, figsize=(26, 17), sharex="col")
    brackets = {}

    for jc, cls in enumerate(CLS):
        zl, vl, el, src = (np.array(LIT[cls][0]), np.array(LIT[cls][1]),
                           np.array(LIT[cls][2]), LIT[cls][3])
        dla_scale = HCD_DLA_RESIDUAL_FRAC if cls == "DLA" else 1.0
        for js, survey in enumerate(SURVEYS):
            ax = axes[jc, js]
            zgrid = SURVEY_Z[survey]; Xb = np.asarray(xbar_fn(zgrid))
            mu, sd1, sd2 = production_pivot(survey, Xbar_z3, w_c_z3)
            c = to_dndx_curve(mu, zgrid, Xb, s_c)[:, jc]
            lo1 = to_dndx_curve(np.clip(mu - sd1, 1e-8, None), zgrid, Xb, s_c)[:, jc]
            hi1 = to_dndx_curve(mu + sd1, zgrid, Xb, s_c)[:, jc]
            lo2 = to_dndx_curve(np.clip(mu - sd2, 1e-8, None), zgrid, Xb, s_c)[:, jc]
            hi2 = to_dndx_curve(mu + sd2, zgrid, Xb, s_c)[:, jc]
            ax.fill_between(zgrid, lo2, hi2, color="tab:orange", alpha=0.18,
                            label=r"prior $2\times\sigma_{LLS}$" if (jc == 0 and js == 0) else None)
            ax.fill_between(zgrid, lo1, hi1, color="tab:blue", alpha=0.30,
                            label=r"prior $1\times\sigma_{LLS}$" if (jc == 0 and js == 0) else None)
            ax.plot(zgrid, c, "-", color="navy", lw=2.4,
                    label="prior center (PRODUCTION)" if (jc == 0 and js == 0) else None)
            # sim/truth dN/dX(z)
            boost = 2.5 if (cls == "LLS" and survey == "KS") else 1.0
            zb, vb = [], []
            for zz in np.unique(np.round(zg, 1)):
                sel = np.isclose(zg, zz, atol=0.05) & (dndx_cache[:, jc] > 0)
                if sel.sum() >= 1:
                    zb.append(zz); vb.append(np.nanmean(dndx_cache[sel, jc]))
            st = np.interp(zgrid, zb, vb, left=np.nan, right=np.nan) * boost
            if cls == "DLA":
                st = st * HCD_DLA_RESIDUAL_FRAC
            ax.plot(zgrid, st, "--", color="tab:green", lw=2.0,
                    label="sim/truth dN/dX" if (jc == 0 and js == 0) else None)
            ax.errorbar(zl, vl * dla_scale, yerr=el * dla_scale, fmt="D", color="k", ms=7,
                        capsize=3, zorder=10, label="literature" if (jc == 0 and js == 0) else None)

            def frac_in(lo_env, hi_env, mask=None):
                loi = np.interp(zl, zgrid, lo_env, left=np.nan, right=np.nan)
                hii = np.interp(zl, zgrid, hi_env, left=np.nan, right=np.nan)
                ok = np.isfinite(loi) & np.isfinite(hii)
                if mask is not None:
                    ok = ok & mask
                inside = (vl * dla_scale + el * dla_scale >= loi) & (vl * dla_scale - el * dla_scale <= hii)
                return int(np.sum(inside & ok)), int(np.sum(ok))
            n1, npts = frac_in(lo1, hi1); n2, _ = frac_in(lo2, hi2)
            lzm = zl <= LOWZ_CUT
            n1_lz, nlz = frac_in(lo1, hi1, mask=lzm)
            brackets[(survey, cls)] = dict(in1=n1, in2=n2, npts=npts, in1_lz=n1_lz, n_lz=nlz)

            ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            if jc == 0:
                ax.set_title(survey, fontweight="bold")
            if js == 0:
                ax.set_ylabel(f"{cls}\n" + (r"dN/dX (10% resid.)" if cls == "DLA" else "dN/dX"))
            if jc == 2:
                ax.set_xlabel("z")
            YLIM = {"LLS": (0.08, 4.0), "subDLA": (3e-2, 0.35), "DLA": (2e-4, 0.3)}
            ax.set_ylim(*YLIM[cls])
            tag = f"lit in band: 1x={n1}/{npts}  2x={n2}/{npts}"
            if cls == "LLS":
                tag += f"\nlow-z(z<={LOWZ_CUT}) 1x: {n1_lz}/{nlz}"
            if survey == "KS" and cls == "LLS":
                tag += "\n(KS x2.5 boost: ABOVE cosmic-avg BY DESIGN)"
            ax.annotate(tag, xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9.5,
                        va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="upper center", ncol=5, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "v3 PRODUCTION HCD prior dN/dX(z) band (CENTER built by the PRODUCTION path after the pivot fix) "
        "vs literature + sim/truth\n"
        "LLS center = corrected lit dN/dX law directly (alt-(b), hcd_lls_realfit_alpha_center ≈0.172×boost), z=3 "
        "STRUCTURAL w_c (NOT the all-z median = z≈3.6 bug); subDLA/DLA via hcd_incidence_prior(z=3 w_c).\n"
        f"σ_LLS 1×={SIG1_LLS['DESI']}/2×={SIG2_LLS['DESI']} (DESI/eBOSS/DK), KS {SIG1_LLS['KS']}; "
        f"subDLA σ/μ={HCD_PRIOR_FRAC_SIGMA[1]}, "
        f"DLA σ/μ={HCD_PRIOR_FRAC_SIGMA[2]}; DLA on the 10% residual axis (lit ×0.10)",
        fontsize=15, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

    print("\n=== v3 PRODUCTION bracketing: lit points (±1σ) overlapping the prior band ===")
    for cls in CLS:
        t1 = sum(brackets[(s, cls)]["in1"] for s in SURVEYS)
        tn = sum(brackets[(s, cls)]["npts"] for s in SURVEYS)
        print(f"  {cls:7s} 1x {t1}/{tn}")
    print("  --- LLS per-survey (the leak class), incl. LOW-z(z<=2.7) ---")
    for s in SURVEYS:
        b = brackets[(s, "LLS")]
        print(f"    LLS {s:8s}: ALL 1x={b['in1']}/{b['npts']}  LOW-z 1x={b['in1_lz']}/{b['n_lz']}")
    # the load-bearing read: DESI low-z LLS bracketed at 1x?
    bd = brackets[("DESI", "LLS")]
    print(f"\n  READ: DESI low-z LLS bracketed at 1x = {bd['in1_lz']}/{bd['n_lz']} "
          f"({'BRACKETED' if bd['in1_lz'] == bd['n_lz'] else 'NOT fully bracketed'})")


if __name__ == "__main__":
    main()
