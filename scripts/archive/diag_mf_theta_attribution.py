#!/usr/bin/env python3
"""ATTRIBUTE the ~5% theta-departure of the LF->HF resolution correction to the 9
cosmology/IGM params, and test a TARGETED delta that adds flexibility over ONLY the
driving subset -- WITHOUT the n_s / A_p Fisher-bias tail of the full MLP delta-head.

CONTEXT (commits e092df1 MF module / d93fadc complexity sweep).  The LF->HF log-ratio
g(theta,z,k) = log P_HF - log P_hat_LF (HR cache = 6 sims = exact LF design points) is
~94-95% a FIXED resolution tilt + only ~5% theta-departure dg(theta,z,k).  The learned
MLP delta-head OVER-FITS that 5% -> n_s Fisher bias ~4 sigma; rho(k,z)-only drops it
(worst |n_s|~2, |A_p|~1.4) but at a higher KODIAQ P1D RMS.

PHYSICAL HYPOTHESIS: the resolution correction's theta-dependence should come from the
THERMAL / PRESSURE params (herei/heref/alphaq -> T0/gamma; hireionz -> filtering scale
k_F) that set the SMALL-SCALE cutoff, NOT n_s/A_p (the amplitude/tilt).  If true, a
delta that depends on ONLY the thermal/reion subset -- with n_s/A_p HELD OUT of its
inputs -- can capture the 5% WITHOUT being able to alias into n_s/A_p.

This script (READ-ONLY on production; reuses the e092df1 MF module + build_mf_delta
HF-LOSO/Fisher harness):

  PART 1 -- ATTRIBUTE dg to the 9 params (6-HF regression, with the noise floor).
    Per (sim,z) average g over the 20 alpha -> the per-sim mean correction; subtract
    the per-z fixed mean gbar(z,k) -> the theta-departure dg(sim,z,k).  Reduce to a
    per-sim KODIAQ-band scalar departure D_s (RMS-or-mean over the KODIAQ band & z).
    Regress D_s on the 6 sims' unit-cube params (each param standardized).  Because 6
    sims in 9-D is UNDER-DETERMINED, we DO NOT fit a joint 9-param model; we report,
    per param, the UNIVARIATE Pearson correlation r(param, D) and an OLS single-param
    slope, with significance judged against the 6-sim NOISE FLOOR (alpha scatter
    propagated into D_s).  HONEST: 6 points -> wide error bars; we flag which params
    clear the noise vs which are consistent with zero.  We report thermal/reion subset
    (herei/heref/alphaq/hireionz) vs n_s/A_p explicitly.

  PART 2 -- 60-sim LF cross-check (10x the statistics; a PROXY).
    The LF<->HF correction can't be measured from LF alone, but a proxy for WHICH
    params the resolution correction depends on = which params the LF P1D's SMALL-SCALE
    structure (the highest-k LF bins, near the LF Nyquist where resolution bites)
    depends on most.  Within each (z,alpha) CELL the 60 cosmologies vary only in theta;
    we take the high-k LF log-P1D (top LF bins), remove the within-cell mean (the
    theta-direction signal), and regress it on the 9 params over all cells.  We report
    each param's standardized sensitivity of the SMALL-SCALE power, and compare the
    ranking to the 6-HF dg attribution.  (Also a LOW-k control: the same regression on
    low-k LF power should be n_s/A_p-dominated, validating the method.)

  PART 3 -- TARGETED delta over the driving subset (n_s/A_p FIXED) -- HF-LOSO.
    Build a delta whose theta-inputs are ONLY a chosen subset (default the thermal/
    reion params herei/heref/alphaq/hireionz; the other params incl. n_s/A_p are
    REPLACED by 0.5 = the cube centre before the head sees them), so the correction is
    n_s/A_p-INDEPENDENT and CANNOT alias into them.  HF-LOSO over the 6 sims; compare
    on KODIAQ RMS + per-sim n_s/A_p Fisher bias:
      (a) rho-only           -- FixedMeanHead (no theta term; the validated default).
      (b) targeted-delta      -- DeltaHead seeing ONLY the subset (n_s/A_p frozen).
      (c) MLP                 -- the e092df1 DeltaHead seeing all 9 (the over-fit one).
    DOES (b) capture the 5% (lower KODIAQ RMS than (a)) WITHOUT the Fisher tail (keep
    |n_s|,|A_p| low like (a), unlike (c))?

Figures -> figures/analysis/05_multifidelity/:
  * mf_theta_attribution.png   -- per-param sensitivity of dg (6-HF) + the 60-sim LF
                                   high-k proxy, with the noise floor / significance.
  * mf_targeted_delta.png       -- rho-only vs targeted-delta vs MLP: KODIAQ RMS +
                                   |n_s|/|A_p| Fisher bias (mean + worst sim).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_theta_attribution.py [args]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import hcd_analysis.emulator  # enables x64
from hcd_analysis.emulator.data import (
    load_cache, cell_id, safe_log, Z_LIMITS, PARAM_LIMITS,
)
from hcd_analysis.emulator import multifidelity as MF

# reuse the build_mf_delta HF-LOSO / Fisher harness verbatim (same metrics, same Fisher
# bias, same HF-standalone target measurement) so the targeted-delta is scored
# IDENTICALLY to the rho-only / MLP baselines of d93fadc.
_spec = importlib.util.spec_from_file_location(
    "build_mf_delta", str(Path(__file__).with_name("build_mf_delta.py")))
BMF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BMF)

CLS = MF.COARSE_NAMES
KODIAQ_BAND = MF.KODIAQ_BAND               # (0.07, 0.2)
LF_NYQUIST = BMF.LF_NYQUIST                # 0.069
N_CLASSES = MF.N_CLASSES
PARAMS = BMF.PARAMS                        # 9 names, cache order
# the physical "thermal / reionization" subset hypothesized to drive the 5%
# (herei/heref/alphaq -> T0/gamma; hireionz -> filtering scale k_F). n_s/A_p excluded.
THERMAL_REION = ("herei", "heref", "alphaq", "hireionz")
AMPLITUDE_TILT = ("ns", "Ap")
PIDX = {p: i for i, p in enumerate(PARAMS)}


# ============================================================================ #
# PART 1 -- attribute the theta-departure dg to the 9 params (6-HF regression)
# ============================================================================ #
def per_sim_departure(tg, sim_of_row, k_eval, band=KODIAQ_BAND):
    """Per-sim KODIAQ-band theta-departure scalar D_s + its alpha-noise floor.

    At each (sim,z) the 20 rows differ ONLY in alpha (same cosmology).  We:
      * per-(sim,z) MEAN over alpha of g (the per-sim mean correction);
      * per-z fixed mean gbar(z,k) = mean of the per-sim means over the 6 sims;
      * dg(sim,z,k) = per-sim-mean - gbar(z,k)  -- the theta-departure;
      * collapse to a per-sim KODIAQ-band scalar D_s = RMS over (z, k in band) of dg
        (a single number per sim = the size of that sim's departure from the mean).

    NOISE FLOOR per sim: the alpha scatter propagated into D_s -- the within-(sim,z)
    variance of g over the 20 alpha, averaged over the band & z, sqrt -> sigma_alpha,
    divided by sqrt(n_alpha) (the per-(sim,z) MEAN's standard error) and RMS-combined
    over z; this is how much D_s would move from alpha noise alone if the true
    departure were zero -> the bar a real theta-signal must clear.

    Returns dict: D (6,), D_signed_classmean dict, noise (6,), per (sim,z,class) tables.
    """
    g = tg["g"]                                       # (M,4,K)
    z = np.round(tg["x"][:, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
    sims = sorted(set(sim_of_row))
    zvals = sorted(set(z))
    sel_k = (k_eval >= band[0]) & (k_eval <= band[1])

    # per-(sim,z): mean over alpha (4,K), within-cell var over alpha (4,K), n_alpha.
    cell_mean, cell_var, cell_n = {}, {}, {}
    for s in sims:
        for zz in zvals:
            m = (sim_of_row == s) & (z == zz)
            if m.sum() == 0:
                continue
            gc = g[m]                                  # (n,4,K)
            with np.errstate(invalid="ignore"):
                cell_mean[(s, zz)] = np.nanmean(gc, axis=0)
                cell_var[(s, zz)] = np.nanvar(gc, axis=0, ddof=1)
            cell_n[(s, zz)] = np.isfinite(gc).sum(axis=0)

    # per-z fixed mean over the 6 sims.
    gbar_z = {}
    for zz in zvals:
        stk = [cell_mean[(s, zz)] for s in sims if (s, zz) in cell_mean]
        if stk:
            with np.errstate(invalid="ignore"):
                gbar_z[zz] = np.nanmean(np.stack(stk), axis=0)

    # collapse dg & noise to per-sim band scalars (clean class = ci 0; also class-mean).
    D = np.zeros(len(sims))                            # per-sim |departure| (clean)
    D_clsmean = np.zeros(len(sims))                    # per-sim |departure| over 4 classes
    noise = np.zeros(len(sims))                        # per-sim alpha noise floor on D
    # also keep a SIGNED, class-mean departure spectrum per sim for plotting.
    dg_band_signed = np.zeros(len(sims))               # signed mean over band&z&class
    for si, s in enumerate(sims):
        ss_clean, ss_cls, nn = [], [], []
        sw_clean = []     # within-cell var of the MEAN (var/n) for the noise floor
        sg = []
        for zz in zvals:
            if (s, zz) not in cell_mean or zz not in gbar_z:
                continue
            dg = cell_mean[(s, zz)] - gbar_z[zz]       # (4,K)
            cv = cell_var[(s, zz)]; cn = np.maximum(cell_n[(s, zz)], 1)
            sem2 = cv / cn                              # variance of the per-cell MEAN
            with np.errstate(invalid="ignore"):
                ss_clean.append(np.nanmean(dg[0, sel_k] ** 2))
                ss_cls.append(np.nanmean(dg[:, sel_k] ** 2))
                sw_clean.append(np.nanmean(sem2[0, sel_k]))
                sg.append(np.nanmean(dg[:, sel_k]))
        D[si] = np.sqrt(np.nanmean(ss_clean))
        D_clsmean[si] = np.sqrt(np.nanmean(ss_cls))
        noise[si] = np.sqrt(np.nanmean(sw_clean))      # RMS sem over band&z = floor on D
        dg_band_signed[si] = np.nanmean(sg)
    return dict(sims=sims, D=D, D_clsmean=D_clsmean, noise=noise,
                dg_signed=dg_band_signed)


def attribute_params(dep, params_unit_per_sim):
    """Univariate attribution of the per-sim departure D_s to each of the 9 params.

    6 sims in 9-D is UNDER-DETERMINED, so we do NOT fit a joint model.  Per param p:
      * standardize the param across the 6 sims (z-score) and D across the 6 sims;
      * Pearson r(param_p, D) + its two-sided p-value (t with df=4);
      * single-param OLS slope of D on the standardized param (= r * std_D), i.e. how
        much D moves per 1-sigma of that param.
    Significance is judged BOTH by the regression p-value AND, physically, by whether
    the per-sim D itself clears the alpha noise floor (reported separately).  Returns a
    per-param dict + the ranking.
    """
    from scipy import stats as _st
    D = dep["D"]                                       # (6,)
    Dc = dep["D_clsmean"]
    nsim = len(D)
    out = {}
    for p in PARAMS:
        xp = params_unit_per_sim[:, PIDX[p]]           # (6,)
        # Pearson r against the (clean-class) departure magnitude.
        r, pval = _st.pearsonr(xp, D)
        rc, pvalc = _st.pearsonr(xp, Dc)
        # standardized single-param slope: D = a + b*z(xp); b = r * std(D).
        b = r * np.std(D, ddof=1)
        out[p] = dict(pearson_r=float(r), p_value=float(pval),
                      pearson_r_clsmean=float(rc), p_value_clsmean=float(pvalc),
                      slope_per_sigma=float(b))
    # ranking by |r| (clean class).
    rank = sorted(PARAMS, key=lambda p: -abs(out[p]["pearson_r"]))
    return out, rank


# ============================================================================ #
# PART 2 -- 60-sim LF high-k proxy: which params drive the small-scale power
# ============================================================================ #
def lf_smallscale_attribution(lf, k_lo_frac=0.75, n_top=12):
    """Within-cell (theta-direction) regression of high-k LF log-P1D on the 9 params.

    Within each (z,alpha) CELL the (up to) 60 cosmologies differ only in theta.  For
    the clean class we form y = mean over the top ``n_top`` finite LF k-bins of
    log P_filt (the SMALL-SCALE power near the LF Nyquist, where resolution bites),
    subtract the within-cell mean (removing the z/alpha-fixed part -> the pure theta
    response), and regress the de-meaned y on the de-meaned, globally-standardized 9
    params pooled over all cells.  We report:
      * a JOINT OLS standardized-coefficient vector (9,) -- 60 sims/cell >> 9 params, so
        this is well-determined (unlike the 6-HF case) -- + its bootstrap error;
      * each param's UNIVARIATE within-cell partial correlation (for ranking parity
        with Part 1).
    A LOW-k CONTROL (bottom n_top bins) is computed identically -- it should be
    n_s/A_p-dominated (validating the method: amplitude/tilt drive large-scale power).

    Returns dict with high-k & low-k standardized coefficient vectors + correlations.
    """
    P = lf["P_filt"][:, 0, :]                          # clean class (R,K)
    logP = safe_log(P)
    cells = cell_id(lf)
    pu = lf["params_unit"]                             # (R,9)
    K = logP.shape[1]

    # per-row finite top/bottom band means (top = near LF Nyquist; bottom = large scale)
    def band_mean(which):
        out = np.full(logP.shape[0], np.nan)
        for r in range(logP.shape[0]):
            fin = np.where(np.isfinite(logP[r]))[0]
            if fin.size < n_top:
                continue
            idx = fin[-n_top:] if which == "high" else fin[:n_top]
            out[r] = np.mean(logP[r, idx])
        return out

    y_hi = band_mean("high")
    y_lo = band_mean("low")

    def within_cell_regress(y):
        # de-mean y and params within each cell -> the pure theta response.
        Xd, yd = [], []
        for c in np.unique(cells):
            m = (cells == c) & np.isfinite(y)
            if m.sum() < 12:                           # need enough cosmologies in-cell
                continue
            yc = y[m] - y[m].mean()
            Xc = pu[m] - pu[m].mean(axis=0, keepdims=True)
            yd.append(yc); Xd.append(Xc)
        yd = np.concatenate(yd); Xd = np.concatenate(Xd)   # (N,), (N,9)
        # globally standardize each (de-meaned) param column so coeffs are per-sigma.
        sd = Xd.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xs = Xd / sd
        # joint OLS standardized coefficients (well-determined: N >> 9).
        beta, *_ = np.linalg.lstsq(Xs, yd, rcond=None)     # (9,) per-sigma slope of logP
        # bootstrap the coefficient error over cells-resampled rows.
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(200):
            ii = rng.integers(0, len(yd), len(yd))
            bb, *_ = np.linalg.lstsq(Xs[ii], yd[ii], rcond=None)
            boots.append(bb)
        boots = np.stack(boots)
        beta_err = boots.std(axis=0)
        # univariate within-cell correlation per param (for ranking parity).
        corr = np.array([np.corrcoef(Xs[:, j], yd)[0, 1] for j in range(9)])
        return dict(beta=beta, beta_err=beta_err, corr=corr, n=len(yd))

    hi = within_cell_regress(y_hi)
    lo = within_cell_regress(y_lo)
    rank_hi = sorted(range(9), key=lambda j: -abs(hi["beta"][j]))
    return dict(
        high_k={"beta": hi["beta"].tolist(), "beta_err": hi["beta_err"].tolist(),
                "corr": hi["corr"].tolist(), "n": hi["n"]},
        low_k={"beta": lo["beta"].tolist(), "beta_err": lo["beta_err"].tolist(),
               "corr": lo["corr"].tolist(), "n": lo["n"]},
        rank_high_k=[PARAMS[j] for j in rank_hi],
        n_top=n_top,
    )


# ============================================================================ #
# PART 3 -- TARGETED delta over a subset (n_s/A_p frozen) + HF-LOSO scoring
# ============================================================================ #
class SubsetDeltaHead(eqx.Module):
    """DeltaHead that sees ONLY a chosen subset of theta -- the rest frozen to 0.5.

    Wraps a standard ``MF.DeltaHead`` but MASKS the conditioning vector before the MLP:
    every cosmo-param index NOT in ``keep_idx`` is overwritten with 0.5 (the unit-cube
    centre), so the head is a function of ONLY the kept params (+ z_unit + tau0).  With
    n_s/A_p NOT in keep_idx the correction is mathematically INDEPENDENT of n_s/A_p ->
    d g / d n_s == d g / d A_p == 0, so it cannot alias the resolution correction into
    the n_s/A_p Fisher directions.  Fully differentiable; coeffs()/__call__ delegate to
    the inner head so it scores identically through MultiFidelity.g and the Fisher
    harness.  ``keep_mask`` (in_dim,) is 1 on kept inputs (subset params + z + tau0),
    0 on the frozen cosmo params; frozen entries are replaced by 0.5.
    """
    inner: MF.DeltaHead
    keep_mask: jax.Array       # (in_dim,) 1=keep input, 0=freeze to 0.5
    n_basis: int = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)

    def __init__(self, inner, keep_mask):
        self.inner = inner
        self.keep_mask = jnp.asarray(keep_mask)
        self.n_basis = inner.n_basis
        self.n_classes = inner.n_classes

    def _mask_cond(self, cond):
        # frozen inputs -> 0.5 (cube centre); kept inputs pass through unchanged.
        return jnp.where(self.keep_mask > 0, cond, 0.5)

    def coeffs(self, cond):
        return self.inner.coeffs(self._mask_cond(cond))

    def __call__(self, cond, basis):
        return self.inner(self._mask_cond(cond), basis)


def _keep_mask(subset, in_dim=11):
    """in_dim-vector: 1 on kept cosmo-param indices + ALWAYS z_unit(9) & tau0(10)."""
    m = np.zeros(in_dim)
    for p in subset:
        m[PIDX[p]] = 1.0
    m[9] = 1.0    # z_unit always kept (the correction is z-resolved)
    if in_dim > 10:
        m[10] = 1.0   # tau0 always kept
    return m


def train_subset_head(tg, basis, log_rho, eval_logk, *, train_rows, subset,
                      n_basis, width, n_layers, lr, epochs, coeff_l2, mean_prior_w,
                      seed=0):
    """Train a DeltaHead whose conditioning is masked to ``subset`` (n_s/A_p frozen).

    We TRAIN with the masked conditioning so the optimizer only ever sees the subset
    inputs (the frozen entries are constant 0.5 in every row -> zero gradient to them).
    Implemented by masking the cond ARRAY fed to ``MF.train_delta_head`` (which builds
    cond = [x(10), tau0]); then wrap the trained inner head in SubsetDeltaHead so eval
    re-applies the identical mask.  Returns the SubsetDeltaHead.
    """
    keep = _keep_mask(subset, in_dim=11)
    # Build a SHALLOW copy of tg with the frozen cosmo-param columns of x set to 0.5
    # (z_unit col 9 and tau0 untouched). train_delta_head reads tg["x"], tg["tau0"].
    x_masked = tg["x"].copy()
    for j in range(9):
        if keep[j] == 0:
            x_masked[:, j] = 0.5
    tg_masked = dict(tg); tg_masked["x"] = x_masked
    inner, hist = MF.train_delta_head(
        tg_masked, basis, log_rho, eval_logk, train_mask_rows=train_rows,
        n_basis=n_basis, width=width, n_layers=n_layers, lr=lr, epochs=epochs,
        coeff_l2=coeff_l2, mean_prior_w=mean_prior_w, seed=seed)
    return SubsetDeltaHead(inner, keep), hist


def score_variant(mf, tg, hf_abs, eval_rows, k_eval):
    """Held-out KODIAQ RMS frac error (4,) + A_p/n_s Fisher bias for one fold."""
    xs = jnp.asarray(tg["x"][eval_rows]); tts = jnp.asarray(tg["tau0"][eval_rows])
    logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, tts))           # (n,4,K)
    logP_true = hf_abs["logP"][eval_rows]                         # (n,4,K)
    err_mf = BMF.frac_err_vs_k(logP_mf, logP_true)                # (4,K)
    kod = BMF.band_rms(err_mf, k_eval, *KODIAQ_BAND)              # (4,)
    FB = BMF.mf_fisher_bias(mf, hf_abs, eval_rows, k_eval, z_fid=3.0)
    return err_mf, kod, FB


# ============================================================================ #
# Figures
# ============================================================================ #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_attribution(attr, rank, dep, lf_proxy, figdir):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    colors = ["C3" if p in THERMAL_REION else ("C0" if p in AMPLITUDE_TILT else "0.5")
              for p in PARAMS]
    # panel 1: 6-HF univariate Pearson r of dg vs each param.
    r = np.array([attr[p]["pearson_r"] for p in PARAMS])
    pv = np.array([attr[p]["p_value"] for p in PARAMS])
    x = np.arange(9)
    axes[0].bar(x, r, color=colors)
    for i in range(9):
        if pv[i] < 0.05:
            axes[0].annotate("*", (x[i], r[i] + 0.03 * np.sign(r[i] + 1e-9)),
                             ha="center", fontsize=14)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(PARAMS, rotation=45, ha="right")
    axes[0].set_ylabel("Pearson r( param , |dg| ) over 6 HF sims")
    axes[0].set_title("PART 1: 6-HF theta-departure attribution\n"
                      "(red=thermal/reion, blue=ns/Ap, grey=other; * p<0.05)")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].set_ylim(-1.05, 1.05)
    # panel 2: per-sim departure D vs the alpha noise floor (significance bar).
    sims = dep["sims"]; D = dep["D"]; noise = dep["noise"]
    xs = np.arange(len(sims))
    axes[1].bar(xs, D, color="C3", alpha=0.8, label="|theta-departure| D_s (clean,KODIAQ)")
    axes[1].errorbar(xs, D, yerr=noise, fmt="none", ecolor="k", capsize=4,
                     label="alpha noise floor (+/-1sigma)")
    axes[1].plot(xs, noise, "k_", ms=14)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"ns{float(s.split('Ap')[0][2:]):.3f}" for s in sims],
                            rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("KODIAQ-band |dg|")
    axes[1].set_title("Per-sim departure vs the 6-sim noise floor\n"
                      "(bar >> error => departure is REAL, not alpha noise)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    # panel 3: 60-sim LF high-k proxy standardized coeffs (+ low-k control).
    bh = np.array(lf_proxy["high_k"]["beta"]); eh = np.array(lf_proxy["high_k"]["beta_err"])
    bl = np.array(lf_proxy["low_k"]["beta"])
    w = 0.38
    axes[2].bar(x - w / 2, np.abs(bh), w, yerr=eh, color=colors, label="high-k (small scale)")
    axes[2].bar(x + w / 2, np.abs(bl), w, color="none", edgecolor="k", lw=1.2,
                label="low-k control (large scale)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(PARAMS, rotation=45, ha="right")
    axes[2].set_ylabel("|standardized OLS coeff| of LF logP")
    axes[2].set_title("PART 2: 60-sim LF small-scale power attribution\n"
                      "(within-cell theta-direction; 10x statistics)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle("Attributing the ~5% LF->HF theta-departure to the 9 params: "
                 "is it the THERMAL/REION subset (red) and NOT ns/Ap (blue)?")
    p = Path(figdir) / "mf_theta_attribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


def fig_targeted(sweep, subset, figdir):
    plt = _plt()
    order = ["a_rho_only", "b_targeted", "c_mlp"]
    labels = ["(a) rho-only\n(no theta)",
              "(b) targeted delta\n[" + "+".join(subset) + "]\n(ns/Ap frozen)",
              "(c) MLP\n(all 9; e092df1)"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    clean = [sweep[v]["kodiaq_rms_mean"][0] for v in order]
    allc = [float(np.sqrt(np.mean(np.array(sweep[v]["kodiaq_rms_mean"]) ** 2)))
            for v in order]
    axes[0].plot(x, clean, "o-", color="C0", lw=2, ms=10, label="clean class")
    axes[0].plot(x, allc, "s--", color="C1", lw=2, ms=9, label="RMS over 4 classes")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("held-out KODIAQ RMS frac P1D error")
    axes[0].set_title("Does targeted-delta CAPTURE the 5%?\n(lower than rho-only?)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for ax, key, col, name in [(axes[1], "ns", "C3", "n_s"), (axes[2], "ap", "C2", "A_p")]:
        mean = [sweep[v][f"{key}_bias_mean_abs"] for v in order]
        worst = [sweep[v][f"{key}_bias_worst_abs"] for v in order]
        ax.plot(x, mean, "o-", color=col, lw=2, ms=10, label="mean |bias|")
        ax.plot(x, worst, "s--", color=col, lw=2, ms=9, alpha=0.6, label="worst sim |bias|")
        ax.axhline(2.0, color="grey", lw=0.9, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"|{name} Fisher bias| [sigma]")
        ax.set_title(f"{name} bias (does targeted-delta AVOID the tail?)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("TARGETED delta (subset-only, ns/Ap frozen) vs rho-only vs MLP "
                 "(HF-LOSO): capture the 5% WITHOUT the ns/Ap Fisher tail?")
    p = Path(figdir) / "mf_targeted_delta.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


# ============================================================================ #
# main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lf-fold", type=int, default=0)
    ap.add_argument("--n-k", type=int, default=48)
    ap.add_argument("--k-max", type=float, default=MF.KODIAQ_KMAX)
    ap.add_argument("--n-basis", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--coeff-l2", type=float, default=1e-1)
    ap.add_argument("--mean-prior-w", type=float, default=1e-1)
    ap.add_argument("--subset", default=",".join(THERMAL_REION),
                    help="comma list of param names the targeted delta may depend on "
                         "(default thermal/reion: herei,heref,alphaq,hireionz). ns/Ap "
                         "MUST NOT be included for the Fisher-independence guarantee.")
    ap.add_argument("--n-top", type=int, default=12,
                    help="# top/bottom LF k-bins for the 60-sim small-scale proxy")
    ap.add_argument("--figdir", default="figures/analysis/05_multifidelity")
    ap.add_argument("--out",
                    default="figures/analysis/05_multifidelity/mf_theta_attribution_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    subset = tuple(s.strip() for s in args.subset.split(",") if s.strip())
    for p in subset:
        if p not in PIDX:
            raise SystemExit(f"unknown subset param {p!r}; choose from {PARAMS}")
    leak = [p for p in subset if p in AMPLITUDE_TILT]
    if leak:
        print(f"WARNING: subset includes {leak} (ns/Ap) -> the Fisher-independence "
              f"guarantee is VOIDED; the targeted delta CAN alias into them.")

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    t0 = time.time()

    lf = load_cache(MF.LF_CACHE)
    hr = load_cache(MF.HR_CACHE)
    lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=args.lf_fold)
    pairs = MF.match_hr_to_lf(lf, hr)
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=args.k_max, n_k=args.n_k)
    print(f"matched {len(pairs)} HR<->LF rows; eval {k_eval[0]:.4f}..{k_eval[-1]:.4f} "
          f"s/km ({args.n_k} bins); LF fold {args.lf_fold} ({time.time()-t0:.1f}s)")

    tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk, pairs)
    hf_abs = BMF.measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=args.n_basis))

    # per-sim unit-cube params (one row per sim; params are constant within a sim).
    params_unit_per_sim = np.zeros((len(sims), 9))
    for si, s in enumerate(sims):
        ri = np.where(sim_of_row == s)[0][0]
        params_unit_per_sim[si] = tg["x"][ri, :9]

    # -------- PART 1: 6-HF theta-departure attribution ---------------------- #
    print("\n========== PART 1: 6-HF THETA-DEPARTURE ATTRIBUTION ==========")
    dep = per_sim_departure(tg, sim_of_row, k_eval)
    attr, rank = attribute_params(dep, params_unit_per_sim)
    print(f"per-sim KODIAQ |dg| (clean): {np.array2string(dep['D'], precision=4)}")
    print(f"alpha noise floor on D     : {np.array2string(dep['noise'], precision=4)}")
    snr = dep["D"] / np.maximum(dep["noise"], 1e-30)
    print(f"D / noise (per sim)        : {np.array2string(snr, precision=2)}  "
          f"(>~2-3 => departure is real signal, not alpha noise)")
    print(f"\n{'param':12} {'r(clean)':>9} {'p':>8} {'r(clsmean)':>11} "
          f"{'slope/sig':>10} {'subset':>8}")
    for p in rank:
        a = attr[p]
        tag = ("THERMAL" if p in THERMAL_REION else ("ns/Ap" if p in AMPLITUDE_TILT else "other"))
        print(f"{p:12} {a['pearson_r']:+9.3f} {a['p_value']:8.3f} "
              f"{a['pearson_r_clsmean']:+11.3f} {a['slope_per_sigma']:+10.4f} {tag:>8}")

    # -------- PART 2: 60-sim LF small-scale proxy --------------------------- #
    print("\n========== PART 2: 60-sim LF SMALL-SCALE PROXY (within-cell theta) ==========")
    lf_proxy = lf_smallscale_attribution(lf, n_top=args.n_top)
    bh = np.array(lf_proxy["high_k"]["beta"]); eh = np.array(lf_proxy["high_k"]["beta_err"])
    bl = np.array(lf_proxy["low_k"]["beta"])
    print(f"within-cell N (high-k regress): {lf_proxy['high_k']['n']}  "
          f"(top {args.n_top} LF bins, clean class)")
    print(f"{'param':12} {'|beta_hi|':>10} {'+/-err':>8} {'z=b/err':>8} "
          f"{'|beta_lo|':>10} {'subset':>8}")
    order_hi = sorted(range(9), key=lambda j: -abs(bh[j]))
    for j in order_hi:
        p = PARAMS[j]
        tag = ("THERMAL" if p in THERMAL_REION else ("ns/Ap" if p in AMPLITUDE_TILT else "other"))
        zsig = abs(bh[j]) / max(eh[j], 1e-30)
        print(f"{p:12} {abs(bh[j]):10.4f} {eh[j]:8.4f} {zsig:8.1f} "
              f"{abs(bl[j]):10.4f} {tag:>8}")

    # -------- PART 3: targeted-delta vs rho-only vs MLP (HF-LOSO) ------------ #
    if args.smoke:
        sims = sims[:2]; args.epochs = min(args.epochs, 200)
    print(f"\n========== PART 3: TARGETED-DELTA HF-LOSO (subset={subset}) ==========")
    variants = ["a_rho_only", "b_targeted", "c_mlp"]
    acc = {v: {"kod": [], "ns": [], "ap": [], "err": []} for v in variants}

    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]
        eval_rows = np.where(held)[0]
        ts = time.time()
        log_rho = MF.mean_log_ratio_rho({"g": tg["g"][train_rows]}, eval_logk)
        log_rho = np.where(np.isfinite(log_rho), log_rho, 0.0)

        heads = {}
        # (a) rho-only: the validated FixedMeanHead (no theta term).
        heads["a_rho_only"] = MF.build_default_head(
            tg, log_rho, train_mask_rows=train_rows, n_basis=args.n_basis)
        # (b) targeted delta: DeltaHead seeing ONLY the subset (ns/Ap frozen to 0.5).
        h_b, _ = train_subset_head(
            tg, basis, log_rho, eval_logk, train_rows=train_rows, subset=subset,
            n_basis=args.n_basis, width=args.width, n_layers=args.n_layers,
            lr=args.lr, epochs=args.epochs, coeff_l2=args.coeff_l2,
            mean_prior_w=args.mean_prior_w, seed=0)
        heads["b_targeted"] = h_b
        # (c) MLP: the e092df1 DeltaHead seeing all 9 params (the over-fit one).
        h_c, _ = MF.train_delta_head(
            tg, basis, log_rho, eval_logk, train_mask_rows=train_rows,
            n_basis=args.n_basis, width=args.width, n_layers=args.n_layers, lr=args.lr,
            epochs=args.epochs, coeff_l2=args.coeff_l2,
            mean_prior_w=args.mean_prior_w, seed=0)
        heads["c_mlp"] = h_c

        line = f"[{s[:14]}..] {time.time()-ts:.1f}s "
        for v in variants:
            mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, heads[v],
                                        eval_logk=eval_logk, log_rho=log_rho,
                                        n_basis=args.n_basis,
                                        delta_mode=("mlp" if v != "a_rho_only" else "none"))
            err_mf, kod, FB = score_variant(mf, tg, hf_abs, eval_rows, k_eval)
            acc[v]["kod"].append(kod); acc[v]["err"].append(err_mf)
            acc[v]["ns"].append(FB["bias_in_sigma"]["ns"])
            acc[v]["ap"].append(FB["bias_in_sigma"]["Ap"])
            line += (f"| {v.split('_')[-1]}: kod={kod[0]:.3f} "
                     f"ns={FB['bias_in_sigma']['ns']:+.2f} ap={FB['bias_in_sigma']['Ap']:+.2f} ")
        print(line)

    sweep = {}
    for v in variants:
        kod = np.stack(acc[v]["kod"])                  # (nsim,4)
        kod_mean = np.sqrt(np.mean(kod ** 2, axis=0))
        ns = np.array(acc[v]["ns"]); apv = np.array(acc[v]["ap"])
        sweep[v] = dict(
            kodiaq_rms_mean=kod_mean.tolist(), kodiaq_rms_per_sim=kod.tolist(),
            ns_bias=ns.tolist(), ap_bias=apv.tolist(),
            ns_bias_mean_abs=float(np.mean(np.abs(ns))),
            ns_bias_worst_abs=float(np.max(np.abs(ns))),
            ap_bias_mean_abs=float(np.mean(np.abs(apv))),
            ap_bias_worst_abs=float(np.max(np.abs(apv))),
            ns_bias_rms=float(np.sqrt(np.mean(ns ** 2))),
            ap_bias_rms=float(np.sqrt(np.mean(apv ** 2))),
        )

    # -------- figures + report ---------------------------------------------- #
    fp_attr = fig_attribution(attr, rank, dep, lf_proxy, args.figdir)
    fp_tgt = fig_targeted(sweep, subset, args.figdir)

    print("\n========== TARGETED-DELTA SWEEP SUMMARY (HF-LOSO) ==========")
    nice = {"a_rho_only": "(a) rho-only", "b_targeted": f"(b) targeted[{'+'.join(subset)}]",
            "c_mlp": "(c) MLP (all 9)"}
    print(f"{'variant':28} {'kod(clean)':>11} {'kod(4cls)':>10} "
          f"{'|ns|mean':>9} {'|ns|worst':>10} {'|Ap|mean':>9} {'|Ap|worst':>10}")
    for v in variants:
        sw = sweep[v]
        kod4 = float(np.sqrt(np.mean(np.array(sw["kodiaq_rms_mean"]) ** 2)))
        print(f"{nice[v]:28} {sw['kodiaq_rms_mean'][0]:11.3f} {kod4:10.3f} "
              f"{sw['ns_bias_mean_abs']:9.2f} {sw['ns_bias_worst_abs']:10.2f} "
              f"{sw['ap_bias_mean_abs']:9.2f} {sw['ap_bias_worst_abs']:10.2f}")
    print("\nPer-sim n_s / A_p bias (sigma):")
    for i, s in enumerate(sims):
        row = "  ".join(f"{v.split('_')[-1]}: ns={sweep[v]['ns_bias'][i]:+.2f} "
                        f"ap={sweep[v]['ap_bias'][i]:+.2f}" for v in variants)
        print(f"  {s[:14]}..  {row}")

    out = {
        "config": vars(args), "subset": list(subset),
        "k_eval": k_eval.tolist(), "kodiaq_band": list(KODIAQ_BAND),
        "params": list(PARAMS), "thermal_reion": list(THERMAL_REION),
        "part1_attribution": {
            "per_param": attr, "rank": rank,
            "per_sim_D": dep["D"].tolist(), "per_sim_noise": dep["noise"].tolist(),
            "per_sim_D_over_noise": (dep["D"] / np.maximum(dep["noise"], 1e-30)).tolist(),
            "sims": dep["sims"],
            "params_unit_per_sim": params_unit_per_sim.tolist(),
        },
        "part2_lf_proxy": lf_proxy,
        "part3_sweep": sweep,
        "figures": {"attribution": fp_attr, "targeted": fp_tgt},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out}\nfigures:\n  {fp_attr}\n  {fp_tgt}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
