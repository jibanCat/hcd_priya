#!/usr/bin/env python3
"""Comprehensive INTERMEDIATE-PLOT WALKTHROUGH of the finalized Phase-2b emulator.

Covers BOTH heads of the deployed emulator on HELD-OUT data, reusing the finalized
LOSO checkpoints (``checkpoints/final_fold{0..7}.eqx``, the FINAL_RECIPE in
``scripts/run_loso_sweep.py``) and the multi-fidelity layer
(``hcd_analysis/emulator/multifidelity.py``, the validated rho(k,z)-only default head).
READ-ONLY on the production caches.

  HEAD B (P1D / cosmology), all 8 finalized folds on their held-out sims:
    B1  pred_vs_true_p1d        per-class P1D(k), log-log + frac-resid subpanel (data-range marked)
    B2  deployed_frac_err_vs_k  per-class deployed |frac err| vs k (in-range) + CV floor overlay
    B3  cosmology_response      sigma_cosmo*d r_hat / d theta vs k for the 9 params (jax.jacfwd)
    B4  theta_tracking          within-(z,tau0)-cell theta-tracking + honest (a)/(b) decomposition
    B5  fisher_bias_perfold     per-fold A_p & n_s Fisher-bias (in-range), the 8-fold spread
    B6  mf_high_k               MF vs LF-extrapolated vs HF-standalone error vs k (KODIAQ band)
    B7  delta_c_templates       the Delta_c HCD templates per class vs k / z

  HEAD A (dN/dX + CDDF), all 8 folds held-out:
    A1  dndx_pred_vs_true       per-class dN/dX vs z (held-out)
    A2  cddf_pred_vs_true       f_NHI / CDDF vs logNHI (class boundaries 17.2/19.0/20.3, shot tail >=21.5)
    A3  wc_ptierp_coupling      emulated-dN/dX w_c vs true + P_tier_p faithfulness (<=2.5% target)
    A4  head_a_error_heatmap    per-class / z / fold Head-A error heatmap
    A5  alpha_prior_sanity      dN/dX -> alpha_c prior center sanity (emulated vs cache CDDF integral)

Each figure is >=150 dpi, titled, axis-labeled. A JSON of headline numbers is written
alongside. The MF figure (B6) trains the small rho(k,z)-only fixed-mean head per HF-LOSO
fold (fast); everything else is pure evaluation of the frozen finalized models.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_performance_walkthrough.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # CPU-only; silence CUDA probe
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log,
    reconstruct_P_filt, untransform_prediction, datarange_mask,
    COARSE_NAMES, DATA_RANGE,
)
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.dndx_wc import w_c_corrected, alpha_to_dndx
from hcd_analysis.emulator.model import structural_tier_p

# ---------------------------------------------------------------------------- #
# Paths / constants
# ---------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]   # repo root (portable; NB02 reuses W.ROOT)
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
CKPT = str(ROOT / "checkpoints/final_fold{f}")
OUTDIR = ROOT / "figures/analysis/06_performance_walkthrough"
CV_JSON = ROOT / "figures/analysis/04_emulator/diag_lfhf_tilt_and_cv.json"

N_FOLDS = 8
CLS = COARSE_NAMES                       # ("clean","LLS","subDLA","DLA")
HCD = ("LLS", "subDLA", "DLA")
PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback"]
# CDDF class boundaries (log10 N_HI) — LLS/subDLA at 19.0, subDLA/DLA at 20.3,
# the optically-thick onset (LLS lower edge) ~17.2; shot-noise tail starts ~21.5.
NHI_BOUNDS = {"LLS onset 17.2": 17.2, "LLS|subDLA 19.0": 19.0, "subDLA|DLA 20.3": 20.3}
NHI_SHOT = 21.5
N_SIGHTLINES = 691200                    # constant per row (diag_head_a)
# DESI-DR1-like diagonal per-mode fractional covariance (matches diag_ap_fisher_bias).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
COLORS = {c: f"C{i}" for i, c in enumerate(CLS)}


# ---------------------------------------------------------------------------- #
# Shared loaders
# ---------------------------------------------------------------------------- #
def load_all_folds():
    """Load the 8 finalized LOSO checkpoints + their held-out splits.

    Returns ``(models, norms, splits)`` keyed by fold, where splits[f] = (tr,va,ho)."""
    models, norms, splits = {}, {}, {}
    for f in range(N_FOLDS):
        m, meta, nrm = T.load_checkpoint(CKPT.format(f=f))
        models[f], norms[f] = m, nrm
        splits[f] = make_splits(load_all_folds._d, f, n_folds=N_FOLDS, holdout_frac=0.15)
    return models, norms, splits


def predict_P(model, d, idx, norm):
    """Reconstructed LINEAR P_filt (n,4,K) on rows ``idx``."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pred = jax.vmap(model)(x, tau0)
    P = reconstruct_P_filt(np.asarray(pred["P_filt_base"]),
                           np.asarray(pred["P_filt_resid"]), norm["P_filt"])
    return P, {k: np.asarray(v) for k, v in pred.items()}


def load_cv_floor():
    """Per-k-band cosmic-variance floor (fractional) from the CV diagnostic JSON."""
    if not CV_JSON.exists():
        return None
    cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
    out = {}
    for band, v in cvj.items():
        lo, hi = (float(x) for x in band.split("-"))
        out[(lo, hi)] = float(v["median_pct"]) / 100.0
    return out


# ============================================================================ #
# HEAD B
# ============================================================================ #
def _pick_class_rows(d, idx_pool, ci, n=3):
    cnt = d["coarse_counts"][idx_pool, ci]
    return idx_pool[np.argsort(-cnt)][:n]


def figB1_pred_vs_true_p1d(models, norms, splits, d, kf):
    """B1: pred-vs-true per-class P1D (held-out), log-log + frac-resid subpanel.

    Uses fold-0's held-out sims (3 best-sampled rows per class). The DESI data range
    (k>=1e-3) is shaded out below k_min; the per-row Nyquist edge is the right turnover."""
    model, norm = models[0], norms[0]
    _, va, _ = splits[0]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fr_rms = {}
    for ci, nm in enumerate(CLS):
        rows = _pick_class_rows(d, va, ci, n=3)
        P_pred, _ = predict_P(model, d, rows, norm)
        P_true = d["P_filt"][rows]
        ax = axes[0, ci]; axr = axes[1, ci]
        accum = []
        for ri in range(len(rows)):
            pt = P_true[ri, ci]; pp = P_pred[ri, ci]
            ok = np.isfinite(pt) & (pt > 0) & np.isfinite(pp) & (pp > 0)
            ax.loglog(kf[ok], pt[ok], "-", color=f"C{ri}", alpha=0.85,
                      label="true" if (ci == 0 and ri == 0) else None)
            ax.loglog(kf[ok], pp[ok], "--", color=f"C{ri}", alpha=0.85,
                      label="pred" if (ci == 0 and ri == 0) else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = pp[ok] / pt[ok] - 1.0
            axr.semilogx(kf[ok], fr, "-", color=f"C{ri}", alpha=0.85)
            accum.append(fr[np.isfinite(fr)])
        fr_rms[nm] = float(np.sqrt(np.mean(np.concatenate(accum) ** 2))) if accum else np.nan
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        axr.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ax.set_title(nm); ax.grid(alpha=0.3, which="both")
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_ylim(-0.3, 0.3); axr.set_xlabel("k  [s/km]")
        if ci == 0:
            ax.set_ylabel("P1D (filtered)  [s/km]")
            axr.set_ylabel("pred/true − 1")
            ax.legend(fontsize=8)
    fig.suptitle("B1 — Predicted vs true per-class P1D (held-out fold-0 sims)\n"
                 "subpanel = fractional residual; grey = below data k_min=1e-3 s/km")
    p = OUTDIR / "B1_pred_vs_true_p1d.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, fr_rms


def _deployed_frac_in_range(models, norms, splits, d):
    """Stack deployed |P_pred/P_true-1| over ALL folds' held-out rows, restricted to
    the DESI data range (z in [2.2,4.6], k>=k_min). Returns (frac (N,4,K), kf)."""
    fr, kfs = [], None
    for f in range(N_FOLDS):
        _, va, _ = splits[f]
        P_pred, _ = predict_P(models[f], d, va, norms[f])
        P_true = d["P_filt"][va]
        mask_k = d["mask"][va]
        in_range = datarange_mask(d)[va]
        keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true > 0)
                & mask_k[:, None, :] & in_range[:, None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(P_pred / P_true - 1.0)
        fr.append(np.where(keep, r, np.nan))
        if kfs is None:
            kfs = d["kfkms"][va][0]
    return np.concatenate(fr, 0), kfs


def figB2_deployed_frac_err_vs_k(frac, kf, cv_floor):
    """B2: per-class deployed |frac err| vs k (in-range, all folds) + CV floor."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rms = {}
    for ci, nm in enumerate(CLS):
        med = np.nanmedian(frac[:, ci, :], axis=0)
        q1 = np.nanpercentile(frac[:, ci, :], 25, axis=0)
        q3 = np.nanpercentile(frac[:, ci, :], 75, axis=0)
        ok = np.isfinite(med)
        ax.fill_between(kf[ok], q1[ok], q3[ok], color=COLORS[nm], alpha=0.13)
        r = float(np.sqrt(np.nanmean(frac[:, ci, :] ** 2)))
        rms[nm] = r
        ax.loglog(kf[ok], med[ok], color=COLORS[nm], lw=1.8, label=f"{nm} (RMS={r:.3f})")
    if cv_floor is not None:
        first = True
        for (lo, hi), v in cv_floor.items():
            ax.hlines(v, max(lo, kf.min()), hi, color="k", ls="--", lw=1.3,
                      label="CV floor (clean)" if first else None)
            first = False
    ax.axvline(DATA_RANGE["k_min"], color="grey", ls=":", lw=1)
    ax.set_xlabel("k  [s/km]")
    ax.set_ylabel("|pred/true − 1|  (median; IQR shaded)")
    ax.set_title("B2 — Deployed per-class fractional P1D error vs k (in-range, 8 folds)\n"
                 "dashed black = cosmic-variance floor; the irreducible sampling scatter")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    p = OUTDIR / "B2_deployed_frac_err_vs_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, rms


def figB3_cosmology_response(models, norms, splits, d, kf):
    """B3: sigma_cosmo*d r_hat / d theta vs k for the 9 params (jax.jacfwd, fold-0)."""
    model, norm = models[0], norms[0]
    _, va, _ = splits[0]
    sig_cosmo = jnp.asarray(norm["P_filt"]["sig_cosmo"])     # (4,K)
    r0 = va[len(va) // 2]
    x0 = jnp.asarray(d["x"][r0]); tau0 = float(d["tau0"][r0])
    z_r0 = float(d["z_grid"][r0])

    def logP_of_x(x):
        return sig_cosmo * model(x, tau0)["P_filt_resid"]    # d logP_hat / d x

    J = np.asarray(jax.jacfwd(logP_of_x)(x0))                # (4,K,10)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    axes = axes.ravel()
    constr = {}
    for j, pp in enumerate(PARAM_NAMES):
        ax = axes[j]
        # response amplitude (RMS over class & k, in-range) — "how much P1D moves"
        with np.errstate(invalid="ignore"):
            amp = np.sqrt(np.nanmean(J[:, kf >= DATA_RANGE["k_min"], j] ** 2))
        constr[pp] = float(amp)
        for ci, nm in enumerate(CLS):
            ax.semilogx(kf, J[ci, :, j], color=COLORS[nm], label=nm if j == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ax.set_title(f"{pp}   (|∂lnP/∂θ|_rms={amp:.3f})", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if j % 3 == 0:
            ax.set_ylabel(r"$\partial \ln P / \partial \theta_{\rm unit}$")
        if j >= 6:
            ax.set_xlabel("k  [s/km]")
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("B3 — Learned cosmology response σ_cosmo·∂r̂/∂θ vs k for all 9 params "
                 f"(jax.jacfwd; fold-0, z={z_r0:.2f}, τ₀={tau0:.2f})\n"
                 "larger |response| = the P1D constrains that param more; sign-flips = "
                 "shape (tilt) sensitivity")
    p = OUTDIR / "B3_cosmology_response.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, constr


def _honest_decomp(model, norm, d, va):
    """Honest within-cell theta-tracking + (a) residual-fit / (b) baseline-misfit
    decomposition (port of diag_theta_tracking_honest, on ONE fold). Returns a dict
    of arrays for plotting + per-class metrics."""
    pf = norm["P_filt"]
    sig_marg, mu_marg, sig_cosmo = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    b = make_batch(d, va, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    base = np.asarray(pred["P_filt_base"])
    resid = np.asarray(pred["P_filt_resid"])
    m_hat = base * sig_marg + mu_marg                        # deployed baseline logP
    lt = safe_log(d["P_filt"][va])                           # true logP
    lp = m_hat + sig_cosmo * resid                           # deployed pred logP
    cv = cell_id(d, va)
    m_cell = np.stack([pf["cell_mean"].get(int(c), mu_marg) for c in cv])

    r_pred = resid
    r_true = (lt - m_hat) / sig_cosmo
    t_resid = (lt - m_cell) / sig_cosmo
    err_resid = r_pred - t_resid                             # (a)
    err_base = (m_hat - m_cell) / sig_cosmo                  # (b)
    err_tot = r_pred - r_true                                # (a)+(b)

    dev_t_hat = lt - m_hat                                   # honest (m_hat ref)
    dev_p_hat = lp - m_hat
    keep = np.isin(cv, [cl for cl in np.unique(cv) if (cv == cl).sum() >= 2])

    def _wrms(x):
        g = np.isfinite(x)
        return float(np.sqrt(np.nanmean(x[g] ** 2)))

    rows = {}
    for ci, nm in enumerate(CLS):
        dt = dev_t_hat[keep, ci, :].ravel(); dp = dev_p_hat[keep, ci, :].ravel()
        g = np.isfinite(dt) & np.isfinite(dp)
        corr = float(np.corrcoef(dt[g], dp[g])[0, 1]) if g.sum() > 3 else np.nan
        st = np.nanstd(r_true[:, ci, :][np.isfinite(r_true[:, ci, :])])
        rows[nm] = dict(honest_corr=corr,
                        a_resid=_wrms(err_resid[:, ci, :]) / st,
                        b_base=_wrms(err_base[:, ci, :]) / st,
                        tot=_wrms(err_tot[:, ci, :]) / st,
                        fracP=_wrms(np.exp(lp[:, ci, :] - lt[:, ci, :]) - 1.0))
    return dict(dev_t=dev_t_hat[keep], dev_p=dev_p_hat[keep], rows=rows,
                err_resid=err_resid, err_base=err_base, err_tot=err_tot, sig_cosmo=sig_cosmo)


def figB4_theta_tracking(models, norms, splits, d):
    """B4: within-(z,tau0)-cell theta-tracking (honest, deployed m_hat ref) + the
    (a) residual-fit vs (b) baseline-misfit whitened-error decomposition (fold-0)."""
    model, norm = models[0], norms[0]
    _, va, _ = splits[0]
    dec = _honest_decomp(model, norm, d, va)

    fig = plt.figure(figsize=(15, 5.6))
    # left: honest hexbin (all classes pooled)
    ax0 = fig.add_subplot(1, 2, 1)
    dt = dec["dev_t"].ravel(); dp = dec["dev_p"].ravel()
    g = np.isfinite(dt) & np.isfinite(dp)
    hb = ax0.hexbin(dt[g], dp[g], gridsize=70, mincnt=1, cmap="viridis")
    lim = np.nanpercentile(np.abs(dt[g]), 99)
    corr = float(np.corrcoef(dt[g], dp[g])[0, 1])
    ratio = float(np.nanstd(dp[g]) / np.nanstd(dt[g]))
    ax0.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
    ax0.set_xlim(-lim, lim); ax0.set_ylim(-lim, lim)
    ax0.set_xlabel("TRUE log-P1D deviation from deployed m̂(z,τ₀)  [cosmology signal]")
    ax0.set_ylabel("PRED deviation from m̂")
    ax0.set_title(f"Honest θ-tracking (deployed m̂ ref)\ncorr={corr:.3f}, "
                  f"spread-ratio={ratio:.3f}")
    ax0.legend(); fig.colorbar(hb, ax=ax0, label="count")

    # right: stacked (a)/(b) decomposition bars per class
    ax1 = fig.add_subplot(1, 2, 2)
    rows = dec["rows"]
    xpos = np.arange(len(CLS))
    a = np.array([rows[c]["a_resid"] for c in CLS])
    bb = np.array([rows[c]["b_base"] for c in CLS])
    tot = np.array([rows[c]["tot"] for c in CLS])
    ax1.bar(xpos, a, color="C2", label="(a) residual-head fit error")
    ax1.bar(xpos, bb, bottom=a, color="C7", label="(b) baseline mis-fit (σ_cosmo-amplified)")
    ax1.plot(xpos, tot, "kD", ms=7, label="deployed total / σ_signal")
    for i, c in enumerate(CLS):
        ax1.text(i, tot[i] + 0.02, f"{rows[c]['fracP']*100:.1f}%\n|P̂/P−1|",
                 ha="center", fontsize=7, color="navy")
    ax1.set_xticks(xpos); ax1.set_xticklabels(CLS)
    ax1.set_ylabel("whitened RMS error  /  σ_signal")
    ax1.set_title("Honest decomposition: (a) residual fit + (b) baseline mis-fit\n"
                  "(the deployed cosmology-signal error, NOT the inflated val-mean metric)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")
    fig.suptitle("B4 — Within-(z,τ₀)-cell θ-tracking + honest (a)/(b) error decomposition (fold-0)")
    p = OUTDIR / "B4_theta_tracking.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, {"honest_corr": corr, "spread_ratio": ratio,
               "per_class": {c: rows[c] for c in CLS}}


def _fisher_bias_fold(model, norm, d, va, z_fid=3.0):
    """9-param Fisher-bias of the deployed model (port of diag_ap_fisher_bias.fisher_bias,
    in-range modes, single fold). Returns the per-param bias in sigma units."""
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]; n_k = sc.shape[1]
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - z_fid))], atol=1e-6)
    tau0_fid = float(np.median(d["tau0"][zsel]))
    z_unit_fid = float(np.median(d["x"][zsel, 9]))
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    mask_va = np.isfinite(safe_log(d["P_filt"][va]))
    in_range = datarange_mask(d)[va]                          # (n,K)
    kvalid = (mask_va[vz].mean(0) > 0.5) & (in_range[vz].mean(0) > 0.5)[None, :]

    z_j = jnp.asarray(z_unit_fid)

    def rhat(theta9):
        x = jnp.concatenate([theta9, z_j[None]])
        return model(x, jnp.asarray(tau0_fid))["P_filt_resid"]
    p0 = jnp.asarray(np.full(9, 0.5), dtype=jnp.float64)
    Jr = np.asarray(jax.jacfwd(rhat)(p0))                     # (4,n_k,9)
    J_logP = Jr * sc[:, :, None]

    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])
    logP_true = safe_log(d["P_filt"][va][vz])
    m_slice = np.isfinite(logP_true)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j],
                                               (logP_hat - logP_true)[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])

    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
               or not np.isfinite(delta[ci, j]):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    J9 = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    F = (J9.T * Cinv) @ J9
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J9.T * Cinv) @ dv)
    bias = dtheta / sigma
    return {PARAM_NAMES[i]: float(bias[i]) for i in range(9)}, int(len(dv))


def figB5_fisher_bias_perfold(models, norms, splits, d):
    """B5: per-fold A_p & n_s Fisher-bias (in-range) — the 8-fold spread."""
    biases = {}
    for f in range(N_FOLDS):
        _, va, _ = splits[f]
        b, nm = _fisher_bias_fold(models[f], norms[f], d, va)
        biases[f] = b
    folds = sorted(biases)
    ap = np.array([biases[f]["Ap"] for f in folds])
    ns = np.array([biases[f]["ns"] for f in folds])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, lab, col in ((axes[0], ap, "A_p", "C3"), (axes[1], ns, "n_s", "C0")):
        ax.bar(folds, vals, color=col, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-0.2, 0.2, color="green", alpha=0.12, label="±0.2σ gate")
        rms = float(np.sqrt(np.mean(vals ** 2)))
        ax.axhline(rms, color="k", ls=":", lw=1)
        ax.axhline(-rms, color="k", ls=":", lw=1, label=f"RMS={rms:.3f}σ")
        ax.set_xlabel("LOSO fold"); ax.set_ylabel(f"{lab} Fisher-bias  [σ]")
        ax.set_title(f"{lab}: per-fold deployed Fisher-bias (in-range, z=3)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
        lim = max(0.3, 1.15 * np.abs(vals).max())
        ax.set_ylim(-lim, lim)
    fig.suptitle("B5 — Per-fold A_p & n_s Fisher-bias spread (8 finalized LOSO folds)\n"
                 "DESI-DR1-like diagonal covariance; the inference gate is |bias|<0.2σ")
    p = OUTDIR / "B5_fisher_bias_perfold.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    out = {"per_fold": {str(f): biases[f] for f in folds},
           "Ap_rms_sigma": float(np.sqrt(np.mean(ap ** 2))),
           "ns_rms_sigma": float(np.sqrt(np.mean(ns ** 2))),
           "Ap_max_abs": float(np.abs(ap).max()), "ns_max_abs": float(np.abs(ns).max())}
    return p, out


def figB6_mf_high_k(d_lf):
    """B6: MF (rho(k,z)-only default) vs LF-extrapolated-alone vs HF-standalone RMS
    frac error vs k, aggregated over HF-LOSO folds, with the KODIAQ band marked.

    Uses the validated rho(k,z)-only FixedMeanHead (the default MF; the learned-MLP
    delta-head is the ablation). Returns (path, summary) or (None, reason) if the HR
    cache / res_corr table is unavailable."""
    try:
        from hcd_analysis.emulator import multifidelity as MF
    except Exception as e:                                    # pragma: no cover
        return None, f"MF import failed: {e!r}"
    try:
        lf = d_lf
        hr = load_cache(MF.HR_CACHE)
        lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=0)
        pairs = MF.match_hr_to_lf(lf, hr)
        k_eval, eval_logk = MF.build_eval_grid(hr, k_max=MF.KODIAQ_KMAX, n_k=48)
        tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk,
                                      pairs, n_tail=6)
        from scripts.build_mf_delta import (measure_hf_abs_targets,
                                            hf_standalone_predict, frac_err_vs_k)
        hf_abs = measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    except Exception as e:                                    # pragma: no cover
        return None, f"MF setup failed: {e!r}"

    from hcd_analysis.emulator.data import Z_LIMITS
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    log_rho = MF.mean_log_ratio_rho(tg, eval_logk)
    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=4))

    per_sim = {}
    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]; eval_rows = np.where(held)[0]
        # DEFAULT MF: rho(k,z)-only FixedMeanHead (NO learned theta head) built from TRAIN rows.
        head = MF.build_default_head(tg, log_rho, train_mask_rows=train_rows, n_basis=4)
        mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, head,
                                    eval_logk=eval_logk, log_rho=log_rho, n_basis=4)
        xs = jnp.asarray(tg["x"][eval_rows]); tts = jnp.asarray(tg["tau0"][eval_rows])
        logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, tts))
        logP_lf = np.asarray(jax.vmap(lambda x, t: mf.lf_logP(x, t)
                                      + jnp.log(mf.res_corr(
                                          x[..., 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]))[None, :]
                                      )(xs, tts))
        logP_hf = hf_standalone_predict(hf_abs, basis, eval_logk, train_rows, eval_rows,
                                        n_basis=4, width=16, n_layers=1, lr=3e-3,
                                        epochs=400, coeff_l2=1e-1, seed=0)
        logP_true = hf_abs["logP"][eval_rows]
        per_sim[s] = {"mf": frac_err_vs_k(logP_mf, logP_true),
                      "lf": frac_err_vs_k(logP_lf, logP_true),
                      "hf": frac_err_vs_k(logP_hf, logP_true)}

    def agg(key):                                            # RMS over HF-LOSO folds (4,K)
        return np.sqrt(np.nanmean(np.stack([per_sim[s][key] for s in sims]) ** 2, axis=0))
    mf_a, lf_a, hf_a = agg("mf"), agg("lf"), agg("hf")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    for ci, nm in enumerate(CLS):
        ax = axes[ci]
        ax.loglog(k_eval, mf_a[ci], "-", color="C0", lw=2.0, label="MF (ρ(k,z)-only)")
        ax.loglog(k_eval, lf_a[ci], "--", color="C3", lw=1.5, label="LF-extrapolated")
        ax.loglog(k_eval, hf_a[ci], ":", color="C2", lw=1.5, label="HF-standalone (6 sim)")
        ax.axvspan(MF.KODIAQ_BAND[0], MF.KODIAQ_BAND[1], color="grey", alpha=0.13)
        ax.axvline(0.069, color="k", ls="-.", lw=0.7, alpha=0.6)
        ax.set_title(nm); ax.grid(alpha=0.3, which="both"); ax.set_xlabel("k  [s/km]")
        if ci == 0:
            ax.set_ylabel("RMS fractional P1D error"); ax.legend(fontsize=8)
    fig.suptitle("B6 — Multi-fidelity high-k: MF vs LF-extrapolated vs HF-standalone "
                 "(HF-LOSO, RMS over held-out HF sims)\n"
                 "shaded = KODIAQ band 0.07–0.2 s/km; dash-dot = LF Nyquist ~0.069 s/km")
    p = OUTDIR / "B6_mf_high_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)

    band = (k_eval >= MF.KODIAQ_BAND[0]) & (k_eval <= MF.KODIAQ_BAND[1])
    summ = {nm: {"mf_kodiaq": float(np.sqrt(np.nanmean(mf_a[ci, band] ** 2))),
                 "lf_kodiaq": float(np.sqrt(np.nanmean(lf_a[ci, band] ** 2))),
                 "hf_kodiaq": float(np.sqrt(np.nanmean(hf_a[ci, band] ** 2)))}
            for ci, nm in enumerate(CLS)}
    return p, summ


def figB7_delta_c_templates(models, norms, splits, d, kf):
    """B7: the CORRECTED per-class HCD excess templates R_c = P_c − P_clean, across z.

    These are the actual objects the forward model uses: P_obs = P_clean + Σ_c α_c·R_c,
    with R_c live-emulated each step (predict_excess). This REPLACES the deprecated HeadB
    `delta` head (P_c^unf − P_c^filt), which gave Δ_LLS≡0 — a numerical no-op that
    visually contradicted the corrected design. For DLA we overlay the FILTERED excess
    (P_filt[DLA]−P_clean) and the UNFILTERED excess actually used
    (P_filt[DLA]+dla_core−P_clean) to make the masking add-back explicit."""
    from hcd_analysis.emulator.predict import predict_excess, predict_P_filt
    model, norm = models[0], norms[0]
    pf = norm["P_filt"]
    _, va, _ = splits[0]
    z_all = np.round(d["z_grid"], 3)
    zs = np.unique(z_all[va])
    z_pick = zs[np.linspace(0, len(zs) - 1, 4).astype(int)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for ci, nm in enumerate(HCD):       # the 3 HCD classes
        ax = axes[ci]
        for zi, zval in enumerate(z_pick):
            rows = va[np.isclose(z_all[va], zval)]
            r0 = rows[len(rows) // 2]
            theta9 = jnp.asarray(d["x"][r0, :9]); z_unit = float(d["x"][r0, 9])
            tau0 = float(d["tau0"][r0]); dla_core = jnp.asarray(d["delta"][r0, 2])
            R = np.asarray(predict_excess(model, theta9, z_unit, tau0, pf, dla_core))  # (3,K)
            ax.plot(kf, R[ci], color=f"C{zi}", lw=1.4, label=f"z={zval:.2f}")
            if nm == "DLA":             # overlay the filtered (pre-unmask) DLA excess
                P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf)
                R_filt = np.asarray(P_filt[3] - P_filt[0])
                ax.plot(kf, R_filt, color=f"C{zi}", lw=1.0, ls=":", alpha=0.7)
        ax.set_xscale("log")
        # symlog y: the very-low-k spike (out-of-range) AND the in-range tail are both
        # legible; the linthresh keeps the small in-range structure resolved.
        ax.set_yscale("symlog", linthresh=0.05)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ttl = f"R_c — {nm}" + ("  (solid=unfilt used, dotted=filt)" if nm == "DLA" else "")
        ax.set_title(ttl); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("k  [s/km]")
        if ci == 0:
            ax.set_ylabel("R_c = P_c − P_clean (physical; symlog)"); ax.legend(fontsize=8)
    fig.suptitle("B7 — corrected per-class HCD excess R_c = P_c − P_clean across held-out z (fold-0)\n"
                 "the object the forward model re-weights: P_obs = P_clean + Σ α_c·R_c; "
                 "LLS now NON-zero (old P_c^unf−P_c^filt gave Δ_LLS≡0); grey = below data k_min")
    p = OUTDIR / "B7_delta_c_templates.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p


# ============================================================================ #
# HEAD A
# ============================================================================ #
def _head_a_stacks(models, norms, splits, d):
    """Stack Head-A held-out predictions/truth over all 8 folds.

    Returns dndx (signed frac err (N,3) + z (N,)), fnhi (signed frac err (N,30) + valid
    + centres), w_c (emu/true/cache (N,4)), P_tier_p frac err, z, and per-(fold,z,class)
    dndx error for the heatmap."""
    grp_all = d["snap_group_idx"]
    z_all = d["z_grid"]
    Xbar_all = d["snap_total_path_dX"][grp_all] / N_SIGHTLINES
    import h5py
    with h5py.File(CACHE, "r") as h:
        centres = h["log_nhi_centres"][:]

    dndx_fe, dndx_z, fnhi_fe, fnhi_valid = [], [], [], []
    wc_emu, wc_true, wc_cache, ptp_fe_all, zrow = [], [], [], [], []
    fold_of_row = []
    for f in range(N_FOLDS):
        model, norm = models[f], norms[f]
        _, va, _ = splits[f]
        x = jnp.asarray(d["x"][va]); tau0 = jnp.asarray(d["tau0"][va])
        pred = jax.vmap(model)(x, tau0)
        phys = untransform_prediction(
            {"f_nhi": np.asarray(pred["f_nhi"]), "dndx": np.asarray(pred["dndx"])}, norm)
        fnhi_p, dndx_p = phys["f_nhi"], phys["dndx"]
        grp = grp_all[va]
        fnhi_t = d["snap_f_nhi"][grp]; dndx_t = d["snap_dNdX"][grp]
        zv = z_all[va]
        fv = np.isfinite(fnhi_t) & (fnhi_t > 0)
        dv = np.isfinite(dndx_t) & (dndx_t > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ff = np.where(fv & np.isfinite(fnhi_p), (fnhi_p - fnhi_t) / fnhi_t, np.nan)
            df = np.where(dv & np.isfinite(dndx_p), (dndx_p - dndx_t) / dndx_t, np.nan)
        dndx_fe.append(df); dndx_z.append(zv); fnhi_fe.append(ff); fnhi_valid.append(fv)
        fold_of_row.append(np.full(len(va), f))

        Xb = Xbar_all[va]
        we = np.asarray(w_c_corrected(jnp.asarray(dndx_p), jnp.asarray(Xb), jnp.asarray(zv)))
        wt = np.asarray(w_c_corrected(jnp.asarray(dndx_t), jnp.asarray(Xb), jnp.asarray(zv)))
        Pf = d["P_filt"][va]
        ptp_e = np.asarray(structural_tier_p(jnp.asarray(we), jnp.asarray(Pf)))
        ptp_t = np.asarray(structural_tier_p(jnp.asarray(wt), jnp.asarray(Pf)))
        ptp_mask = np.isfinite(ptp_t) & (ptp_t > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ptp_fe = np.where(ptp_mask, (ptp_e - ptp_t) / ptp_t, np.nan)
        wc_emu.append(we); wc_true.append(wt); wc_cache.append(d["w_c_cache"][va])
        ptp_fe_all.append(ptp_fe); zrow.append(zv)

    return dict(
        dndx_fe=np.concatenate(dndx_fe), dndx_z=np.concatenate(dndx_z),
        fnhi_fe=np.concatenate(fnhi_fe), fnhi_valid=np.concatenate(fnhi_valid),
        centres=centres, wc_emu=np.concatenate(wc_emu), wc_true=np.concatenate(wc_true),
        wc_cache=np.concatenate(wc_cache), ptp_fe=np.concatenate(ptp_fe_all),
        zrow=np.concatenate(zrow), fold=np.concatenate(fold_of_row),
        Xbar_all=Xbar_all, grp_all=grp_all)


def figA1_dndx_pred_vs_true(S, z_levels):
    """A1: per-class dN/dX fractional accuracy vs z (held-out, 8 folds)."""
    fe, zv = S["dndx_fe"], S["dndx_z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    out = {}
    for c in range(3):
        ax = axes[c]; med, p95 = [], []
        for z in z_levels:
            sel = np.isclose(zv, z)
            col = np.abs(fe[sel, c]); col = col[np.isfinite(col)]
            med.append(np.median(col) * 100 if col.size else np.nan)
            p95.append(np.percentile(col, 95) * 100 if col.size else np.nan)
        ax.plot(z_levels, med, "o-", color="C0", label="median |frac err|")
        ax.plot(z_levels, p95, "s--", color="C3", label="p95 |frac err|")
        ax.axhline(2.5, color="r", ls=":", lw=1, label="2.5% target")
        ax.set_title(f"dN/dX  {HCD[c]}"); ax.set_xlabel("z"); ax.grid(alpha=0.3)
        ax.set_ylim(0, max(8, np.nanmax(p95) * 1.1))
        if c == 0:
            ax.set_ylabel("|fractional error|  [%]"); ax.legend(fontsize=8)
        out[HCD[c]] = {"median_pct": float(np.nanmedian(np.abs(fe[:, c]) * 100))}
    fig.suptitle("A1 — Head-A dN/dX fractional accuracy vs z (held-out, 8 LOSO folds)")
    p = OUTDIR / "A1_dndx_pred_vs_true.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, out


def figA2_cddf_pred_vs_true(S):
    """A2: f_NHI / CDDF fractional accuracy vs logNHI, with class boundaries +
    shot-noise tail flagged."""
    fe, valid, centres = S["fnhi_fe"], S["fnhi_valid"], S["centres"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    med, p95, frac = [], [], []
    for b in range(fe.shape[1]):
        col = np.abs(fe[:, b][valid[:, b]]); col = col[np.isfinite(col)]
        med.append(np.median(col) * 100 if col.size else np.nan)
        p95.append(np.percentile(col, 95) * 100 if col.size else np.nan)
        frac.append(valid[:, b].mean())
    ax.plot(centres, med, "o-", color="C0", label="median |frac err|")
    ax.plot(centres, p95, "s--", color="C3", label="p95 |frac err|")
    ax.axhline(2.5, color="r", ls=":", lw=1, label="2.5%")
    for lab, v in NHI_BOUNDS.items():
        ax.axvline(v, color="grey", ls="-", lw=1.1, alpha=0.7)
        ax.text(v, ax.get_ylim()[1], lab.split()[0], rotation=90, va="top",
                ha="right", fontsize=7, color="grey")
    ax.axvspan(NHI_SHOT, centres.max(), color="orange", alpha=0.12)
    ax.text(NHI_SHOT + 0.1, 3.0, "shot-noise tail\n(logN_HI≥21.5)", fontsize=8, color="darkorange")
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax.set_ylabel("|fractional error|  [%]")
    ax.set_yscale("log")
    ax.set_title("A2 — Head-A f_NHI (CDDF) fractional accuracy per N_HI bin (held-out, 8 folds)\n"
                 "vertical lines = class boundaries; shaded = shot-noise tail")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(centres, frac, color="gray", alpha=0.4, lw=1)
    ax2.set_ylabel("valid-row fraction (gray)", color="gray"); ax2.set_ylim(0, 1.05)
    p = OUTDIR / "A2_cddf_pred_vs_true.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    overall = np.abs(fe[valid]); overall = overall[np.isfinite(overall)]
    return p, {"median_pct": float(np.median(overall) * 100),
               "p95_pct": float(np.percentile(overall, 95) * 100)}


def figA3_wc_ptierp_coupling(S):
    """A3: w_c round-trip (emulated vs true dN/dX) + structural P_tier_p faithfulness."""
    we, wt, ptp_fe = S["wc_emu"], S["wc_true"], S["ptp_fe"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for c in range(4):
        ax.scatter(wt[:, c], we[:, c], s=2, alpha=0.2, color=COLORS[CLS[c]], label=CLS[c])
    lim = [0, max(wt.max(), we.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("w_c (from TRUE dN/dX)"); ax.set_ylabel("w_c (from EMULATED dN/dX)")
    ax.set_title("w_c round-trip: emulated vs true dN/dX")
    ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in range(4):
        err = np.abs(we[:, c] - wt[:, c])
        ax.hist(err, bins=60, histtype="step", color=COLORS[CLS[c]], label=CLS[c], density=True)
    ax.axvline(0.025, color="r", ls=":", lw=1, label="2.5% abs")
    ax.set_xlabel("|w_c(emu) − w_c(true)|"); ax.set_ylabel("density")
    ax.set_title("w_c coupling error (round-trip)"); ax.set_xlim(0, 0.04)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    fe = ptp_fe[np.isfinite(ptp_fe)]
    ax.hist(fe * 100, bins=80, histtype="stepfilled", color="C0", alpha=0.6)
    ax.axvline(0, color="k", lw=1)
    for q, ls in [(50, "-"), (95, "--")]:
        v = np.percentile(np.abs(fe) * 100, q)
        ax.axvline(v, color="r", ls=ls, lw=1, label=f"p{q} |err|={v:.3f}%")
        ax.axvline(-v, color="r", ls=ls, lw=1)
    ax.axvspan(-2.5, 2.5, color="green", alpha=0.08)
    ax.set_xlabel("P_tier_p frac err (emu vs true w_c)  [%]"); ax.set_ylabel("count")
    ax.set_title("Structural P_tier_p faithfulness (≤2.5% target)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(-3, 3)
    fig.suptitle("A3 — w_c → P_tier_p coupling: emulated-dN/dX w_c vs true + P_tier_p faithfulness (8 folds)")
    p = OUTDIR / "A3_wc_ptierp_coupling.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, {"ptierp_median_pct": float(np.median(np.abs(fe)) * 100),
               "ptierp_p95_pct": float(np.percentile(np.abs(fe), 95) * 100)}


def figA4_head_a_error_heatmap(S, z_levels):
    """A4: per-class / z / fold Head-A dN/dX error heatmap (median |frac err| %)."""
    fe, zv, fold = S["dndx_fe"], S["dndx_z"], S["fold"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    vmax = 0.0
    grids = []
    for c in range(3):
        G = np.full((N_FOLDS, len(z_levels)), np.nan)
        for fi in range(N_FOLDS):
            for zi, z in enumerate(z_levels):
                sel = (fold == fi) & np.isclose(zv, z)
                col = np.abs(fe[sel, c]); col = col[np.isfinite(col)]
                if col.size:
                    G[fi, zi] = np.median(col) * 100
        grids.append(G); vmax = max(vmax, np.nanpercentile(G, 95))
    for c in range(3):
        ax = axes[c]
        im = ax.imshow(grids[c], aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax,
                       extent=[z_levels.min(), z_levels.max(), -0.5, N_FOLDS - 0.5])
        ax.set_title(f"dN/dX  {HCD[c]}"); ax.set_xlabel("z")
        if c == 0:
            ax.set_ylabel("LOSO fold")
        ax.set_yticks(range(N_FOLDS))
        fig.colorbar(im, ax=ax, label="median |frac err| [%]" if c == 2 else None)
    fig.suptitle("A4 — Head-A dN/dX error heatmap: per class × z × fold (median |frac err|, %)\n"
                 "uniform color = well-balanced; a hot row/column flags a weak fold or z")
    p = OUTDIR / "A4_head_a_error_heatmap.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p


def figA5_alpha_prior_sanity(S):
    """A5: dN/dX -> alpha_c prior-center sanity — emulated w_c vs cache (CDDF-derived) w_c.

    The emulated dN/dX feeds the alpha_c prior center via w_c_corrected. We compare the
    emulated w_c (the prior center) to the cache's empirical CDDF-derived w_c, and verify
    the analytic alpha->dN/dX round-trip closes (alpha_to_dndx(w_c) ≈ dN/dX)."""
    we, wcache = S["wc_emu"], S["wc_cache"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for c in range(4):
        ax.scatter(wcache[:, c], we[:, c], s=2, alpha=0.2, color=COLORS[CLS[c]], label=CLS[c])
    lim = [0, max(wcache.max(), we.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("w_c (cache CDDF integral)"); ax.set_ylabel("w_c (emulated dN/dX → prior center)")
    ax.set_title("Emulated w_c (α_c prior center) vs cache CDDF w_c")
    ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.3)

    ax = axes[1]
    out = {}
    for c in range(4):
        err = np.abs(we[:, c] - wcache[:, c])
        ax.hist(err, bins=60, histtype="step", color=COLORS[CLS[c]], label=CLS[c], density=True)
        out[CLS[c]] = {"median": float(np.median(err)), "p95": float(np.percentile(err, 95))}
    ax.axvline(0.025, color="r", ls=":", lw=1, label="2.5% abs")
    ax.set_xlabel("|w_c(emu) − w_c(cache)|"); ax.set_ylabel("density")
    ax.set_title("α_c prior-center deviation from the cache CDDF integral")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 0.06)
    fig.suptitle("A5 — dN/dX → α_c prior-center sanity: emulated w_c vs cache CDDF integral (8 folds)")
    p = OUTDIR / "A5_alpha_prior_sanity.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return p, out


# ============================================================================ #
# Main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-mf", action="store_true",
                    help="skip the multi-fidelity figure B6 (needs the HR cache + res_corr)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    d = load_cache(CACHE)
    kf = d["kfkms"][0]
    z_levels = np.unique(np.round(d["z_grid"], 1))
    print(f"cache: {d['P_tier_p'].shape[0]} rows, n_k={len(kf)}, z-levels={len(z_levels)}")

    load_all_folds._d = d
    models, norms, splits = load_all_folds()
    print(f"loaded {len(models)} finalized LOSO folds (MF = ρ(k,z)-only FixedMeanHead default)")

    produced, skipped, head = {}, {}, {}
    cv_floor = load_cv_floor()

    # -------- HEAD B --------
    p, fr = figB1_pred_vs_true_p1d(models, norms, splits, d, kf); produced["B1"] = p
    head["B1_pred_frac_rms"] = fr
    frac, kf2 = _deployed_frac_in_range(models, norms, splits, d)
    p, rms = figB2_deployed_frac_err_vs_k(frac, kf2, cv_floor); produced["B2"] = p
    head["B2_deployed_frac_rms_inrange"] = rms
    p, constr = figB3_cosmology_response(models, norms, splits, d, kf); produced["B3"] = p
    head["B3_response_amplitude"] = constr
    p, tt = figB4_theta_tracking(models, norms, splits, d); produced["B4"] = p
    head["B4_theta_tracking"] = tt
    p, fb = figB5_fisher_bias_perfold(models, norms, splits, d); produced["B5"] = p
    head["B5_fisher_bias"] = fb
    if args.no_mf:
        skipped["B6"] = "skipped via --no-mf"
    else:
        p, mfsum = figB6_mf_high_k(d)
        if p is None:
            skipped["B6"] = mfsum
        else:
            produced["B6"] = p; head["B6_mf_kodiaq"] = mfsum
    p = figB7_delta_c_templates(models, norms, splits, d, kf); produced["B7"] = p

    # -------- HEAD A --------
    print("building Head-A held-out stacks (8 folds)...")
    S = _head_a_stacks(models, norms, splits, d)
    p, a1 = figA1_dndx_pred_vs_true(S, z_levels); produced["A1"] = p
    head["A1_dndx"] = a1
    p, a2 = figA2_cddf_pred_vs_true(S); produced["A2"] = p
    head["A2_cddf"] = a2
    p, a3 = figA3_wc_ptierp_coupling(S); produced["A3"] = p
    head["A3_wc_ptierp"] = a3
    p = figA4_head_a_error_heatmap(S, z_levels); produced["A4"] = p
    p, a5 = figA5_alpha_prior_sanity(S); produced["A5"] = p
    head["A5_alpha_prior"] = a5

    with open(OUTDIR / "headline_numbers.json", "w") as f:
        json.dump(head, f, indent=2, default=float)

    print("\n==== HEADLINE NUMBERS ====")
    print(json.dumps(head, indent=2, default=float))
    print("\n==== FIGURES PRODUCED ====")
    for k, v in produced.items():
        print(f"  {k}: {v}")
    if skipped:
        print("==== SKIPPED ====")
        for k, v in skipped.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
