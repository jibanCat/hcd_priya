"""SCRATCH (read-only on code): per-HCD-class dN/dX(z) predictions from the 8-fold LOSO
posterior closures, per survey, overplotted with the literature dN/dX + the prior centre.

Map (the exact one the forward uses, verified by round-trip to the cache snap_dNdX to ~1%):
  posterior alpha_pivot_c  (LLS, subDLA, DLA)  [per-class incidence WEIGHT at z_pivot=3]
    -> alpha_c(z) = alpha_pivot_c * ((1+z)/(1+3))**s_c            (forward z-evolution)
    -> dN/dX_c(z) = alpha_to_dndx(alpha_c(z), Xbar(z), z)         (dndx_wc telescoping inverse)
s_c = HCD_INCIDENCE_SLOPE = (2.465,2.758,2.366) for the fixed-slope closures (the SIM incidence-weight
slope the mock truth carries; NOT the lit/sim RATIO slope HCD_LIT_OVER_SIM_SLOPE=(0.95,…)).
Xbar(z) = X_tot/N_sl from the LF cache (mean absorption path per sightline).

Literature dN/dX (z,val,err,src) and the lit/sim ratio centres are the repo's own constants
(scripts/plot_dndx_vs_literature.py LIT; inference.HCD_LIT_OVER_SIM(_SLOPE)).

Writes PNGs to the notes repo figures dir + a small JSON of the per-fold readouts.
Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/scratch_hcd_dndx_loso_vs_lit.py
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # x64 before jax
import jax.numpy as jnp
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
from hcd_analysis.emulator.inference import (HCD_LIT_OVER_SIM, HCD_LIT_OVER_SIM_SLOPE,
                                             HCD_Z_PIVOT, HCD_DLA_RESIDUAL_FRAC, lit_over_sim_at_z)
from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE

REPO = Path("/home/mfho/hcd_priya")
CACHE = str(REPO / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
STEPA = REPO / "checkpoints/stepA"
OUTDIR = Path("/home/mfho/hcd_priya_notes/figures/analysis/03_templates_and_p1d")
OUTDIR.mkdir(parents=True, exist_ok=True)

CLS = ["LLS", "subDLA", "DLA"]
ZP = float(HCD_Z_PIVOT)

# Literature dN/dX (z, value, ±err, source) — verbatim from scripts/plot_dndx_vs_literature.py
LIT = {
    "LLS":    ([2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23],
               [0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78],
               [0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19],
               "O'Meara13 / Fumagalli13 / Prochaska10 (tau>=2)"),
    "subDLA": ([2.27, 2.73, 3.25, 3.77, 4.20],
               [0.07, 0.06, 0.08, 0.10, 0.10],
               [0.01, 0.01, 0.02, 0.02, 0.03],
               "Zafar+2013 (Table 3)"),
    "DLA":    ([2.31, 2.57, 2.86, 3.22, 3.70, 4.39],
               [0.048, 0.055, 0.067, 0.084, 0.075, 0.106],
               [0.006, 0.005, 0.006, 0.006, 0.009, 0.018],
               "Prochaska & Wolfe 2009 (Table 1)"),
}

# Canonical per-survey LOSO mock sets (base names; EC1 = emucoh-ON infl=1 production config when present).
SURVEY_MOCKS = {
    "DESI": ["D_f3_EC1", "D_f4_EC1", "D_f6_EC1", "D_f7_EC1",            # A-family sim_mean prior
             "D_lmed3_15_EC1", "D_lmed5_15_EC1", "D_lmed7_15_EC1", "D_llsmed_EC1"],  # B lit prior
    "KS":   ["HFLOSO_KS859", "HFLOSO_KS885", "HFLOSO_KS909", "HFLOSO_KS972", "HFLOSO_KS979"],
    "DESI+KS": ["HFLOSO_DK859", "HFLOSO_DK885", "HFLOSO_DK909", "HFLOSO_DK972", "HFLOSO_DK979"],
    "eBOSS": ["E_f5", "E_f6", "E_f7"],
}
SURVEY_COL = {"DESI": "tab:blue", "KS": "tab:green", "DESI+KS": "tab:orange", "eBOSS": "tab:purple"}
SURVEY_Z = {"DESI": np.arange(2.2, 4.21, 0.2), "KS": np.arange(2.4, 4.61, 0.2),
            "DESI+KS": np.arange(2.4, 4.61, 0.2), "eBOSS": np.arange(2.2, 4.61, 0.2)}


def chains_for(base):
    cs = sorted(glob.glob(str(STEPA / f"{base}_c*.npz")))
    if not cs:
        cs = sorted(glob.glob(str(STEPA / f"{base}_EC*_c*.npz")))
    return cs


def build_xbar():
    """Xbar(z) = X_tot / N_sl from the LF cache (mean absorption path per sightline).
    N_sl inferred per snap-group from the clean-fraction Poisson identity, then averaged."""
    d = load_cache(CACHE)
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    Xtot = np.asarray(d["snap_total_path_dX"]); wc = np.asarray(d["w_c_cache"])
    dndx = np.asarray(d["snap_dNdX"])
    Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(axis=1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    # smooth Xbar(z): a quadratic fit in z over the in-range groups (robust to per-group scatter)
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    return lambda z: np.polyval(cf, np.asarray(z)), d, zg, dndx


def dndx_from_alpha_pivot(alpha_pivot, z, slope, xbar_fn):
    """(...,3) alpha_pivot -> dN/dX_c(z) array of shape (nz, 3). alpha_pivot can be (3,) or (N,3)."""
    a = np.atleast_2d(np.asarray(alpha_pivot))                       # (N,3)
    z = np.atleast_1d(np.asarray(z, float))                          # (nz,)
    s = np.asarray(slope)                                            # (3,)
    shape = ((1.0 + z)[None, :, None] / (1.0 + ZP)) ** s[None, None, :]   # (1,nz,3)
    alpha_z = a[:, None, :] * shape                                  # (N,nz,3)
    Xb = np.asarray(xbar_fn(z))                                      # (nz,)
    out = np.empty_like(alpha_z)
    for j, zz in enumerate(z):
        out[:, j, :] = np.asarray(alpha_to_dndx(jnp.asarray(alpha_z[:, j, :]),
                                                jnp.asarray(float(Xb[j])), jnp.asarray(float(zz))))
    return out                                                      # (N,nz,3)


def pool_posterior(base):
    """Return dict: alpha_draws (Ndraw,3) pooled over chains, truth_alpha (3,), meta."""
    cs = chains_for(base)
    if not cs:
        return None
    draws, meta = [], None
    for c in cs:
        z = np.load(c, allow_pickle=True)
        nm = list(z["names"]); pk = z["packed"]
        idx = [nm.index(x) for x in ("alpha_lls", "alpha_subdla", "alpha_dla")]
        draws.append(pk[:, idx])
        if meta is None:
            tv = z["truth_vec"]
            meta = dict(fold=int(z["fold"]), ns=float(z["n_s"]), survey=str(z["survey"]),
                        prior=str(z["prior_center"]), sim=str(z["sim"]),
                        truth_alpha=np.array([tv[i] for i in idx]),
                        zslope=bool(z["z_slope_marginalized"]))
    return dict(alpha=np.concatenate(draws, 0), **meta)


def truth_dndx_for_sim(sim, d, zg, dndx_cache):
    """The sim's OWN per-class dN/dX(z) directly from the cache snap_dNdX (the ground truth)."""
    names = np.asarray(d["sim_name"]); gid = np.asarray(d["snap_group_idx"])
    # map each snap-group to its sim via the first cache row of that group
    grp_sim = np.array([str(names[gid == g][0]) for g in range(dndx_cache.shape[0])])
    sel = grp_sim == sim
    if sel.sum() == 0:
        return None, None
    order = np.argsort(zg[sel])
    return zg[sel][order], dndx_cache[sel][order]


def main():
    xbar_fn, d, zg_cache, dndx_cache = build_xbar()
    # FORWARD z-slope = the SIM incidence-WEIGHT slope HCD_INCIDENCE_SLOPE (~2.4, the slope the
    # mock truth's w_c(z) carries → dN/dX(z) RISES with z), NOT the lit/sim RATIO slope
    # HCD_LIT_OVER_SIM_SLOPE (~0.95, the wrong-object slope that made the bands FALL with z).
    SLOPE = np.asarray(HCD_INCIDENCE_SLOPE)

    # --- gather per-survey, per-fold posterior dN/dX(z) ---
    records = {}   # survey -> list of dict(meta + zgrid + dndx_med/lo/hi (nz,3) + truth)
    for survey, bases in SURVEY_MOCKS.items():
        zgrid = SURVEY_Z[survey]
        recs = []
        for b in bases:
            p = pool_posterior(b)
            if p is None:
                print(f"  [skip] no chains for {b}"); continue
            slope = SLOPE  # fixed-slope closures (zslope=False) → the SIM incidence-weight slope
            dd = dndx_from_alpha_pivot(p["alpha"], zgrid, slope, xbar_fn)   # (N,nz,3)
            med = np.median(dd, 0); lo = np.percentile(dd, 16, 0); hi = np.percentile(dd, 84, 0)
            # truth alpha -> truth dN/dX(z) via the same map (the posterior-space truth)
            td = dndx_from_alpha_pivot(p["truth_alpha"][None, :], zgrid, slope, xbar_fn)[0]
            # the sim's native cache dN/dX (independent ground truth, full incidence)
            zt, ndt = truth_dndx_for_sim(p["sim"], d, zg_cache, dndx_cache)
            recs.append(dict(base=b, fold=p["fold"], ns=p["ns"], prior=p["prior"],
                             zgrid=zgrid, med=med, lo=lo, hi=hi, truth_alpha_dndx=td,
                             sim_z=zt, sim_dndx=ndt))
            print(f"  {survey:8s} {b:16s} fold={p['fold']} ns={p['ns']:.3f} "
                  f"dNdX@z3 LLS={med[np.argmin(abs(zgrid-3)),0]:.3f} "
                  f"sub={med[np.argmin(abs(zgrid-3)),1]:.3f} DLA={med[np.argmin(abs(zgrid-3)),2]:.4f}")
        records[survey] = recs

    # prior-centre dN/dX(z): (lit/sim)(z) * sim w_c -> but we plot it as the lit/sim * sim-native dN/dX
    # mean over sims, the same construction the prior uses. Use cache mean sim dN/dX * (lit/sim)(z) ratio.
    def prior_center_dndx(zgrid):
        out = np.zeros((len(zgrid), 3))
        # (nz,3) lit/sim ratio: lit_over_sim_at_z expects z broadcast against the (3,) class slope
        rr = np.asarray(lit_over_sim_at_z(jnp.asarray(zgrid)[:, None]))   # (nz,3)
        for j in range(3):
            inr = (zg_cache >= 2.2) & (zg_cache <= 4.6) & (dndx_cache[:, j] > 0)
            # sim-mean dN/dX(z): bin the cache by z
            sim_md = np.array([np.nanmean(dndx_cache[np.isclose(zg_cache, zz, atol=0.12) & inr, j])
                               for zz in zgrid])
            cen = sim_md * rr[:, j]
            if j == 2:  # DLA prior centre is the 10% unmasked residual
                cen = cen * HCD_DLA_RESIDUAL_FRAC
            out[:, j] = cen
        return out

    # =========================== FIGURE 1: dN/dX(z) per class ===========================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for j, c in enumerate(CLS):
        ax = axes[j]
        for survey, recs in records.items():
            col = SURVEY_COL[survey]
            for k, r in enumerate(recs):
                lbl = survey if k == 0 else None
                ax.plot(r["zgrid"], r["med"][:, j], "-", color=col, lw=1.3, alpha=0.85, label=lbl)
                ax.fill_between(r["zgrid"], r["lo"][:, j], r["hi"][:, j], color=col, alpha=0.06)
        # literature
        zl, vl, el, src = LIT[c]
        ax.errorbar(zl, vl, yerr=el, fmt="D", color="k", ms=6, capsize=3, zorder=10,
                    label=f"literature\n({src})")
        # prior centre (DESI z-grid)
        pc = prior_center_dndx(SURVEY_Z["DESI"])
        ax.plot(SURVEY_Z["DESI"], pc[:, j], "k:", lw=2, label="prior centre")
        ax.set_title(c, fontsize=13, fontweight="bold")
        ax.set_xlabel("z"); ax.set_ylabel("dN/dX")
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        if c == "DLA":
            ax.set_ylim(5e-4, 0.3)
            ax.annotate("posterior dN/dX_DLA is the\n10% UNMASKED-DLA residual\n(NOT the full DLA incidence)",
                        xy=(0.04, 0.04), xycoords="axes fraction", fontsize=8,
                        bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))
        ax.legend(fontsize=7, loc="upper left", ncol=1)
    fig.suptitle("Per-HCD-class dN/dX(z): 8-fold LOSO posterior predictions per survey vs literature\n"
                 "(lines = per-fold posterior median; shaded = 68%; black diamonds = observed; "
                 "dotted = prior centre)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = OUTDIR / "hcd_dndx_loso_vs_literature.png"
    fig.savefig(f1, dpi=140); plt.close(fig)
    print("wrote", f1)

    # ============== FIGURE 2: per-survey small multiples (DESI + KS + DESI+KS + eBOSS) =========
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    surveys = ["DESI", "KS", "DESI+KS", "eBOSS"]
    for jc, c in enumerate(CLS):
        zl, vl, el, src = LIT[c]
        for js, survey in enumerate(surveys):
            ax = axes[jc, js]
            recs = records[survey]; col = SURVEY_COL[survey]
            for r in recs:
                ax.plot(r["zgrid"], r["med"][:, jc], "-", color=col, lw=1.3,
                        label=f"f{r['fold']} ns{r['ns']:.3f}")
                ax.fill_between(r["zgrid"], r["lo"][:, jc], r["hi"][:, jc], color=col, alpha=0.08)
                if r["sim_z"] is not None:
                    ax.plot(r["sim_z"], r["sim_dndx"][:, jc], ":", color=col, lw=0.9, alpha=0.6)
            ax.errorbar(zl, vl, yerr=el, fmt="D", color="k", ms=5, capsize=2, zorder=10)
            ax.set_yscale("log"); ax.grid(alpha=0.3)
            if jc == 0:
                ax.set_title(survey, fontsize=12, fontweight="bold")
            if js == 0:
                ax.set_ylabel(f"{c}\ndN/dX", fontsize=11)
            if jc == 2:
                ax.set_xlabel("z")
            if c == "DLA":
                ax.set_ylim(2e-4, 0.3)
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("dN/dX(z) per HCD class x survey: per-fold posterior (solid+68% band), "
                 "sim native dN/dX (dotted, same colour), literature (black diamonds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f2 = OUTDIR / "hcd_dndx_loso_per_survey.png"
    fig.savefig(f2, dpi=130); plt.close(fig)
    print("wrote", f2)

    # =========================== JSON summary of the readouts ===========================
    summ = {}
    for survey, recs in records.items():
        summ[survey] = []
        for r in recs:
            iz3 = int(np.argmin(abs(r["zgrid"] - 3.0)))
            summ[survey].append(dict(base=r["base"], fold=r["fold"], ns=round(r["ns"], 3),
                                     prior=r["prior"],
                                     dndx_z3=[round(float(r["med"][iz3, j]), 5) for j in range(3)],
                                     truth_alpha_dndx_z3=[round(float(r["truth_alpha_dndx"][iz3, j]), 5)
                                                          for j in range(3)]))
    (OUTDIR / "hcd_dndx_loso_summary.json").write_text(json.dumps(summ, indent=2))
    print("wrote", OUTDIR / "hcd_dndx_loso_summary.json")
    return records, prior_center_dndx, d


def p1d_templates(records, d):
    """Per-class P1D EXCESS template alpha_c*(P_c - P_clean) at the production model, posterior-median
    alpha. Uses final_fold0 + the production pf_stats via predict_P_filt at a fiducial (theta, z, tau0)."""
    from hcd_analysis.emulator import train as T
    from hcd_analysis.emulator.predict import predict_P_filt
    import jax.numpy as jnp
    import numpy as np
    model, meta, norm = T.load_checkpoint(str(REPO / "checkpoints/final_fold0"))
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    kf = np.asarray(meta["kfkms"])
    from hcd_analysis.emulator.data import Z_LIMITS, KIM_AMP, KIM_SLOPE
    # fiducial theta = box centre; z=3.0; tau0 = Kim(3)
    theta = jnp.full(9, 0.5); z = 3.0
    z_unit = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    tau0 = float(KIM_AMP * (1 + z) ** KIM_SLOPE)
    dla_core = jnp.asarray(np.median(np.asarray(d["delta"])[:, 2], axis=0))   # representative DLA core
    Pf = np.asarray(predict_P_filt(model, theta, jnp.asarray(z_unit), jnp.asarray(tau0), pf))  # (4,K)
    P_clean = Pf[0]
    P_cls = np.stack([P_clean, Pf[1], Pf[2], Pf[3] + np.asarray(dla_core)])    # clean,LLS,sub,DLA(unf)
    excess = P_cls[1:] - P_clean[None, :]                                      # (3,K) per-class (P_c-P_clean)

    # posterior-median alpha at z3 per survey (mean over folds)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for j, c in enumerate(CLS):
        ax = axes[j]
        for survey, recs in records.items():
            col = SURVEY_COL[survey]
            # posterior-median alpha_pivot (the WEIGHT) per fold -> mean over folds for the template amp
            amps = []
            for r in recs:
                p = pool_posterior(r["base"])
                amps.append(np.median(p["alpha"][:, j]))
            amp = float(np.mean(amps))
            ax.plot(kf, amp * excess[j] / P_clean, "-", color=col, lw=1.6, label=f"{survey} (alpha={amp:.3f})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log"); ax.set_xlabel("k [s/km]")
        ax.set_ylabel(r"$\alpha_c\,(P_c-P_{clean})/P_{clean}$")
        ax.set_title(f"{c} P1D excess template", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("HCD-induced P1D EXCESS template per class (fractional, at z=3, box-centre theta)\n"
                 "amplitude = posterior-median alpha_c (incidence weight), survey-averaged over folds",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f3 = OUTDIR / "hcd_p1d_excess_templates.png"
    fig.savefig(f3, dpi=140); plt.close(fig)
    print("wrote", f3)


if __name__ == "__main__":
    recs, _pc, d = main()
    p1d_templates(recs, d)
