"""v2 overlay (PI 2026-06-17 re-validation): the RE-VALIDATED HCD prior dN/dX(z) center, built via
the ROUND-TRIP-EXACT lit-anchored forward, overlaid with the literature MEASUREMENTS (points+err)
and the sim/truth dN/dX(z) -- per class per survey. CONFIRMS the LOW-z LLS points are bracketed.

WHY v2 (the diagnosis): the v1 overlay (and the production prior) builds the LLS center as
  alpha_pivot_LLS = lit_over_sim[LLS] * w_c_med[LLS],  then alpha_c(z)=alpha_pivot*((1+z)/4)^gamma,
  then dN/dX = alpha_to_dndx(alpha_c(z), Xbar(z), z).
TWO bugs compound there:
  (1) w_c_med is the MEDIAN of the cache structural weight over ALL z-groups (z=2.0..5.4). Because
      w_c[LLS] rises monotonically with z (0.075 @z2 -> 0.49 @z5.4), the all-z median (0.274) equals
      the z~3.6 value, NOT the z=3 pivot (0.189) -> the pivot amplitude is ~1.48x too HIGH.
  (2) the dN/dX power-law slope gamma=2.127 is applied to the STRUCTURAL WEIGHT alpha, then mapped
      back through the NONLINEAR telescoping w_c map. alpha and dN/dX are NOT related by the same
      power law, so this distorts the z-shape (under/over at the edges).
Net: the v1/production LLS center OVERSHOOTS the lit dN/dX by 1.6-2.0x, WORST at low z (2.0x @z2.4).

THE FIX (this v2): build the LLS center from the lit dN/dX power-law DIRECTLY:
  dN/dX_LLS(z) = A_LLS * (1+z)^gamma_LLS  (now the CORRECTED law of record, A=0.0184,
  gamma=2.127 constrained; PI 2026-07-18 -- historically A=0.0201, the cumulative-estimand fit),
  *survey LLS boost,
  then alpha_LLS(z) = alpha_from_dndx_law(...)  ONLY to display on the same axis (round-trips <0.2%).
subDLA/DLA centers ALSO built from the DEPLOYED HCD_LIT_DNDX_LAW (corrected 2026-07-18: subDLA is
the Poisson-GLM law on the verified Zafar counts, NOT the tombstoned DLA-column WLS (0.0211, 0.937))
for a like-for-like, round-trip-exact display; DLA on the 10%-residual axis (x0.10). The sigma_LLS
1x/2x envelopes scale the dN/dX center by (1 +- sigma/mu) at the DEPLOYED per-survey widths
(0.287/0.574 DESI-family, 0.40/0.40 KS; corrected-law widths 2026-07-18). Laws + points + widths
all come from the deployed module constants / lit_dndx (no hard-codes).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_hcd_prior_dndx_overlay_v2.py
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
from hcd_analysis.emulator.closure_legb import CACHE_PATH

OUTDIR = Path("/home/mfho/hcd_priya_notes/figures/analysis/03_templates_and_p1d")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "hcd_prior_dndx_overlay_measurements_v2.png"

CLS = ["LLS", "subDLA", "DLA"]
ZP = 3.0
DLA_RESID = 0.10               # HCD_DLA_RESIDUAL_FRAC

# lit dN/dX power-laws + per-survey knobs -- from the DEPLOYED module constants (corrected
# laws of record 2026-07-18; the old (0.0201, 2.127)/(0.0211, 0.937) hard-codes were the
# wrong-object laws, now tombstoned in lit_dndx.py)
from hcd_analysis.emulator.inference import (
    HCD_LIT_DNDX_LAW, HCD_LLS_SURVEY_BOOST, HCD_LLS_SURVEY_FRAC_SIGMA,
    HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X, HCD_PRIOR_FRAC_SIGMA as _FRAC)
from hcd_analysis.emulator.lit_dndx import lit_points_for_display
LIT_LAW = HCD_LIT_DNDX_LAW
LLS_BOOST = HCD_LLS_SURVEY_BOOST
LLS_SIG1 = HCD_LLS_SURVEY_FRAC_SIGMA
LLS_SIG2 = HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X
SUB_SIG, DLA_SIG = _FRAC[1], _FRAC[2]  # subDLA/DLA fractional widths

LIT = lit_points_for_display()
SURVEYS = ["DESI", "KS", "DESI+KS", "eBOSS"]
SURVEY_Z = {"DESI": np.arange(2.2, 4.21, 0.1), "KS": np.arange(2.4, 4.61, 0.1),
            "DESI+KS": np.arange(2.4, 4.61, 0.1), "eBOSS": np.arange(2.2, 4.61, 0.1)}


def sim_truth_dndx_binned():
    """The cache sim-mean dN/dX(z) per class (the truth), binned in z."""
    d = load_cache(CACHE_PATH)
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    dndx = np.asarray(d["snap_dNdX"]); Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    zb, vb = [], []
    for zz in np.unique(np.round(zg, 1)):
        sel = np.isclose(zg, zz, atol=0.05)
        if sel.sum() >= 1:
            zb.append(zz); vb.append(np.nanmean(dndx[sel], axis=0))   # (3,)
    return np.array(zb), np.array(vb)


def lls_center_dndx(zgrid, survey):
    """LLS center dN/dX(z) = boost * A_LLS*(1+z)^gamma_LLS (the lit WLS law, real-fit anchor)."""
    A, g = LIT_LAW["LLS"]
    return LLS_BOOST[survey] * A * (1.0 + zgrid) ** g


def sub_center_dndx(zgrid):
    A, g = LIT_LAW["subDLA"]; return A * (1.0 + zgrid) ** g


def dla_center_dndx(zgrid):
    A, g = LIT_LAW["DLA"]; return DLA_RESID * A * (1.0 + zgrid) ** g   # 10% residual axis


def main():
    zb_sim, vb_sim = sim_truth_dndx_binned()

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(3, 4, figsize=(26, 17), sharex="col")
    brackets = {}

    for jc, cls in enumerate(CLS):
        zl, vl, el, src = LIT[cls]
        zl = np.array(zl); vl = np.array(vl); el = np.array(el)
        dla_scale = DLA_RESID if cls == "DLA" else 1.0
        for js, survey in enumerate(SURVEYS):
            ax = axes[jc, js]
            zgrid = SURVEY_Z[survey]
            if cls == "LLS":
                c = lls_center_dndx(zgrid, survey)
                s1, s2 = LLS_SIG1[survey], LLS_SIG2[survey]
            elif cls == "subDLA":
                c = sub_center_dndx(zgrid); s1 = s2 = SUB_SIG
            else:
                c = dla_center_dndx(zgrid); s1 = s2 = DLA_SIG
            lo1, hi1 = c * (1 - s1), c * (1 + s1)
            lo2, hi2 = c * (1 - s2), c * (1 + s2)
            # 2x (wider/lighter) UNDER the 1x (tighter/darker)
            ax.fill_between(zgrid, lo2, hi2, color="tab:orange", alpha=0.18,
                            label=r"prior $2\times\sigma$" if (jc == 0 and js == 0) else None)
            ax.fill_between(zgrid, lo1, hi1, color="tab:blue", alpha=0.30,
                            label=r"prior $1\times\sigma$" if (jc == 0 and js == 0) else None)
            ax.plot(zgrid, c, "-", color="navy", lw=2.4,
                    label="prior center (lit dN/dX law)" if (jc == 0 and js == 0) else None)
            # sim/truth dN/dX(z) (cache), with the KS LLS boost + DLA 10% residual scaling
            st = np.interp(zgrid, zb_sim, vb_sim[:, jc], left=np.nan, right=np.nan)
            if cls == "LLS" and survey == "KS":
                st = st * 2.5
            elif cls == "DLA":
                st = st * DLA_RESID
            ax.plot(zgrid, st, "--", color="tab:green", lw=2.0,
                    label="sim/truth dN/dX" if (jc == 0 and js == 0) else None)
            # literature measurements
            ax.errorbar(zl, vl * dla_scale, yerr=el * dla_scale, fmt="D", color="k", ms=7,
                        capsize=3, zorder=10, label="literature" if (jc == 0 and js == 0) else None)

            def frac_in(lo_env, hi_env):
                loi = np.interp(zl, zgrid, lo_env, left=np.nan, right=np.nan)
                hii = np.interp(zl, zgrid, hi_env, left=np.nan, right=np.nan)
                ok = np.isfinite(loi) & np.isfinite(hii)
                inside = (vl * dla_scale + el * dla_scale >= loi) & (vl * dla_scale - el * dla_scale <= hii)
                return int(np.sum(inside & ok)), int(np.sum(ok))
            n1, npts = frac_in(lo1, hi1); n2, _ = frac_in(lo2, hi2)
            # low-z-only (z<=2.7) bracketing — the load-bearing requirement
            lzm = zl <= 2.7
            loi1 = np.interp(zl, zgrid, lo1, left=np.nan, right=np.nan)
            hii1 = np.interp(zl, zgrid, hi1, left=np.nan, right=np.nan)
            in1_lz = int(np.sum(((vl * dla_scale + el * dla_scale >= loi1) &
                                 (vl * dla_scale - el * dla_scale <= hii1) & np.isfinite(loi1))[lzm]))
            brackets[(survey, cls)] = dict(in1=n1, in2=n2, npts=npts, in1_lz=in1_lz, n_lz=int(lzm.sum()))

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
                tag += f"\nlow-z(z<=2.7) 1x: {brackets[(survey,cls)]['in1_lz']}/{brackets[(survey,cls)]['n_lz']}"
            if survey == "KS" and cls == "LLS":
                tag += "\n(KS x2.5 boost: ABOVE cosmic-avg BY DESIGN)"
            ax.annotate(tag, xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9.5,
                        va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="upper center", ncol=5, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "v2 RE-VALIDATED HCD prior dN/dX(z) center (round-trip-exact lit dN/dX law) vs literature + sim/truth\n"
        "LLS center = boost x the DEPLOYED corrected lit law (A,g)=HCD_LIT_DNDX_LAW[LLS]; built from the lit dN/dX law DIRECTLY, "
        "NOT lit/sim x all-z-median-w_c with the slope on alpha (the v1 bug: 1.6-2.0x overshoot, worst low z).\n"
        f"sigma_LLS 1x={LLS_SIG1['DESI']} (DESI/eBOSS/DK) / 2x={LLS_SIG2['DESI']}; KS {LLS_SIG1['KS']}; "
        f"subDLA {SUB_SIG}, DLA {DLA_SIG}; DLA on the 10% residual axis (lit x0.10)",
        fontsize=15, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

    print("\n=== v2 bracketing: lit points (+-1sigma) overlapping the prior band ===")
    for cls in CLS:
        t1 = sum(brackets[(s, cls)]["in1"] for s in SURVEYS)
        t2 = sum(brackets[(s, cls)]["in2"] for s in SURVEYS)
        tn = sum(brackets[(s, cls)]["npts"] for s in SURVEYS)
        print(f"  {cls:7s} 1x {t1}/{tn};  2x {t2}/{tn}")
    print("  --- LLS per-survey (the leak class) ---")
    for s in SURVEYS:
        b = brackets[(s, "LLS")]
        print(f"    LLS {s:8s}: ALL 1x={b['in1']}/{b['npts']} 2x={b['in2']}/{b['npts']}  "
              f"LOW-z(z<=2.7) 1x={b['in1_lz']}/{b['n_lz']}")


if __name__ == "__main__":
    main()
