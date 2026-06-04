#!/usr/bin/env python3
"""Phase-2b emulator walkthrough — train a fold-0 model and emit a comprehensive
set of intermediate/diagnostic figures (POST normalization REDESIGN).

Trains ONE fold-0 LOSO model (staged=False; the staged path is buggy) and from it
generates the figures described in figures/analysis/04_emulator/walkthrough/README.md:

  01 variance_decomposition   cosmology(θ) vs (z,τ₀) fraction of per-k log-var
  02 whitening_scales         σ_marg vs σ_cosmo per-k (+ ratio ~0.077)
  03 loss_curves              total + per-term loss vs epoch
  04 baseline_head_fit        m̂ vs (z,τ₀) cell-mean, z slices; residual≈0
  05 residual_head_fit        r̂ vs truth in σ_cosmo units (cosmology recovered)
  06 pred_vs_true_p1d         per-class P1D(k), held-out sims, log-log + frac resid
  07 perclass_frac_error      |frac error| vs k per class (the clean-limited story)
  08 theta_tracking           within-(z,τ₀)-cell θ-spread pred vs true (corr/ratio)
  09 cosmology_response       σ_cosmo·∂r̂/∂θ vs k for a few params (jax.jacfwd)

Reuses the data/model/train API verbatim (data.fit_baseline_residual_norm,
make_batch, reconstruct_P_filt; train.train_fold). All figures ≥150 dpi, titled,
axis-labeled. A trained checkpoint is cached under checkpoints/walkthrough_fold0 so
re-runs of the plotting alone are fast (--reuse).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_emulator_walkthrough.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log,
    reconstruct_P_filt, untransform_prediction, COARSE_NAMES,
    PARAM_LIMITS,
)
from hcd_analysis.emulator import train as T

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUTDIR = "/home/mfho/hcd_priya/figures/analysis/04_emulator/walkthrough"
CKPT = "/home/mfho/hcd_priya/checkpoints/walkthrough_fold0"
CLS = COARSE_NAMES
PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback"]


# ----------------------------------------------------------------------------- #
# Training / model helpers
# ----------------------------------------------------------------------------- #

# RESIDUAL-HEAD TUNING (the A_p low-k bias fix, scripts/diag_ap_fisher_bias.py
# sweep). The DEPLOYED production recipe: up-weight the cosmology (p_resid) term,
# emphasise the low/high-k band EDGES where the deployed residual was biased, mild
# weight-decay, and early-stop on the val RESIDUAL (cosmology) loss (the residual
# overfits past its val minimum on this 60-sim fold). This drove the A_p Fisher-bias
# from +0.26σ (untuned, overfit) to ~+0.03σ ROBUSTLY across seeds (uniform/early-stop
# -only was seed-unstable: A_p swung 0.03→1.2σ), with n_s <0.2σ and deployed median
# |P̂/P−1| <1% (clean/LLS/subDLA).
RESID_TUNE = dict(
    term_w={"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0, "p_resid": 8.0, "delta": 1.0},
    edge_gain=3.0, lowk_extra=2.0, weight_decay=3e-4, early_stop_metric="auto",
)


def train_or_load(d, *, fold=0, n_basis=24, epochs=180, lr=1e-3, batch=512,
                  seed=0, reuse=False):
    """Train fold-0 (staged=False, the tuned residual recipe) or reload the cache."""
    from hcd_analysis.emulator.data import edge_emphasis_k_weight
    n_k = d["P_tier_p"].shape[1]
    tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
    if reuse and Path(CKPT + ".eqx").exists():
        model, meta, norm = T.load_checkpoint(CKPT)
        with open(CKPT + ".hist.json") as f:
            history = {k: np.asarray(v) for k, v in json.load(f).items()}
        print(f"[reuse] loaded {CKPT} (n_basis={meta['arch_cfg']['n_basis']})")
        return model, norm, history, (tr, va, ho)
    print(f"[train] fold={fold} epochs={epochs} n_basis={n_basis} "
          f"train={len(tr)} val={len(va)} holdout={len(ho)}  (tuned residual recipe)")
    k_weight = edge_emphasis_k_weight(d["kfkms"][0], edge_gain=RESID_TUNE["edge_gain"],
                                      lowk_extra=RESID_TUNE["lowk_extra"])
    model, norm, history = T.train_fold(
        d, tr, va, n_basis=n_basis, lr=lr, epochs=epochs, batch_size=batch,
        seed=seed, key=jax.random.PRNGKey(seed), patience=30, n_k=n_k,
        staged=False, term_w=RESID_TUNE["term_w"], k_weight=k_weight,
        early_stop_metric=RESID_TUNE["early_stop_metric"],
        weight_decay=RESID_TUNE["weight_decay"])
    arch_cfg = {"in_dim": 10, "n_k": n_k, "n_basis": n_basis}
    T.save_checkpoint(CKPT, model, arch_cfg, norm, seed=seed,
                      kfkms=d["kfkms"], cache_path=CACHE)
    with open(CKPT + ".hist.json", "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in history.items()}, f)
    print(f"[train] {len(history['train_loss'])} epochs; checkpoint -> {CKPT}")
    return model, norm, history, (tr, va, ho)


def predict_phys(model, d, idx, norm):
    """Reconstructed LINEAR P_filt + standardized head outputs for rows idx."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pred = jax.vmap(model)(x, tau0)
    pred = {k: np.asarray(v) for k, v in pred.items()}
    P_lin = reconstruct_P_filt(pred["P_filt_base"], pred["P_filt_resid"],
                               norm["P_filt"])
    return pred, P_lin


# ----------------------------------------------------------------------------- #
# Per-term loss recomputation (for the loss-curve figure)
# ----------------------------------------------------------------------------- #

def per_term_losses(model, batch):
    """Recompute the 5 per-term standardized-MSE values (model.joint_loss internals)."""
    from hcd_analysis.emulator.model import masked_mse
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    la_cddf = masked_mse(preds["f_nhi"], batch["t_f_nhi"], batch["t_f_nhi_mask"],
                         weight=batch["inv_nalpha"][:, None])
    la_dndx = masked_mse(preds["dndx"], batch["t_dndx"], batch["t_dndx_mask"],
                         weight=batch["inv_nalpha"][:, None])
    m3 = batch["mask"][:, None, :]
    wcls4 = batch["inv_nc"][:, :, None]
    wcls3 = batch["inv_nc"][:, 1:, None]
    mask4 = m3 & jnp.ones_like(batch["t_p_resid"], bool)
    lb_base = masked_mse(preds["P_filt_base"], batch["t_p_base"], mask4, weight=wcls4)
    lb_resid = masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=wcls4)
    lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                       m3 & jnp.ones_like(batch["t_delta"], bool), weight=wcls3)
    return {k: float(v) for k, v in
            dict(f_nhi=la_cddf, dndx=la_dndx, p_base=lb_base,
                 p_resid=lb_resid, delta=lb_dl).items()}


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #

def fig01_variance_decomposition(d, kf):
    """Per-k log-P1D variance split into cosmology(θ) vs (z,τ₀) fractions."""
    R, C, K = d["P_filt"].shape
    logP = safe_log(d["P_filt"])
    cells = cell_id(d)
    vt = np.nanvar(logP, axis=0)                         # (C,K) total
    vw = np.zeros((C, K))                                # within-cell (cosmology)
    for cl in np.unique(cells):
        m = cells == cl
        if m.sum() < 2:
            continue
        with np.errstate(invalid="ignore"):
            vw += np.nan_to_num(np.nanvar(logP[m], axis=0)) * m.sum()
    vw /= R
    frac = vw / np.where(vt > 0, vt, 1.0)                # cosmology fraction

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ci, nm in enumerate(CLS):
        axes[0].semilogx(kf, 100 * frac[ci], label=nm)
    axes[0].set_xlabel("k  [s/km]")
    axes[0].set_ylabel("cosmology fraction of log-variance  [%]")
    axes[0].set_title("Cosmology(θ) is <1% of the per-k log-variance")
    axes[0].axhline(0.5, color="grey", ls=":", lw=0.8)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # stacked bar: median over k of cosmology vs (z,τ₀) fractions per class
    med_cos = 100 * np.array([np.nanmedian(frac[ci]) for ci in range(C)])
    x = np.arange(C)
    axes[1].bar(x, 100 - med_cos, label="(z,τ₀) spread  [inputs]", color="C7")
    axes[1].bar(x, med_cos, bottom=100 - med_cos,
                label="cosmology θ  [signal]", color="C3")
    for i, v in enumerate(med_cos):
        axes[1].text(i, 100.5, f"{v:.2f}%", ha="center", fontsize=8, color="C3")
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLS)
    axes[1].set_ylabel("median fraction of log-variance  [%]")
    axes[1].set_ylim(0, 104)
    axes[1].set_title("Why a global σ buries cosmology")
    axes[1].legend(loc="lower center"); axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("01 — Variance decomposition: cosmology vs (z,τ₀) per-k log-variance")
    fig.tight_layout()
    p = f"{OUTDIR}/01_variance_decomposition.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, {nm: float(med_cos[ci]) for ci, nm in enumerate(CLS)}


def fig02_whitening_scales(d, norm, kf):
    """σ_marg (marginal) vs σ_cosmo (within-cell) per-k whitening scales + ratio."""
    pf = norm["P_filt"]
    sig_marg = pf["sig_marg"]            # (4,K)
    sig_cosmo = pf["sig_cosmo"]          # (4,K)
    ratio = sig_cosmo / np.where(sig_marg > 0, sig_marg, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ci, nm in enumerate(CLS):
        axes[0].loglog(kf, sig_marg[ci], color=f"C{ci}", ls="-",
                       label=f"{nm} σ_marg")
        axes[0].loglog(kf, sig_cosmo[ci], color=f"C{ci}", ls="--")
    axes[0].set_xlabel("k  [s/km]"); axes[0].set_ylabel("whitening scale (log-P units)")
    axes[0].set_title("σ_marg (solid, τ₀/z spread)  vs  σ_cosmo (dashed, cosmology)")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3, which="both")

    for ci, nm in enumerate(CLS):
        axes[1].semilogx(kf, ratio[ci], color=f"C{ci}", label=nm)
    rmed = np.nanmedian(ratio[ratio > 0])
    axes[1].axhline(rmed, color="k", ls=":", lw=1,
                    label=f"median ratio = {rmed:.3f}")
    axes[1].set_xlabel("k  [s/km]")
    axes[1].set_ylabel("σ_cosmo / σ_marg")
    axes[1].set_title("Ratio ≈ √(cosmology fraction) — the un-burying factor")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("02 — The two whitening scales (residual head trains in σ_cosmo units)")
    fig.tight_layout()
    p = f"{OUTDIR}/02_whitening_scales.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, float(rmed)


def fig03_loss_curves(model, d, va, norm, history):
    """Total loss + per-term breakdown vs epoch."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    # per-term snapshot on the val batch for the *final* model (the per-epoch term
    # split isn't logged, so we annotate the converged split alongside the totals).
    vb = {k: jnp.asarray(v) for k, v in make_batch(d, va, norm).items()}
    terms = per_term_losses(model, vb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].semilogy(ep, history["train_loss"], "-o", ms=3, label="train (total)")
    axes[0].semilogy(ep, history["val_loss"], "-s", ms=3, label="val (total)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("joint loss (standardized MSE)")
    axes[0].set_title("Total joint loss (train + val)")
    axes[0].legend(); axes[0].grid(alpha=0.3, which="both")

    names = ["p_base", "p_resid", "f_nhi", "dndx", "delta"]
    labels = ["baseline P_filt", "residual P_filt (cosmology)",
              "Head-A f_nhi", "Head-A dN/dX", "Δ_c"]
    vals = [terms[n] for n in names]
    colors = ["C0", "C3", "C2", "C4", "C5"]
    axes[1].bar(range(len(names)), vals, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("converged val per-term MSE (standardized)")
    axes[1].set_title("Per-term val loss at convergence")
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.2g}", ha="center", va="bottom", fontsize=7)
    axes[1].grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("03 — Loss curves: total descent + per-term budget")
    fig.tight_layout()
    p = f"{OUTDIR}/03_loss_curves.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, terms


def fig04_baseline_head_fit(model, d, norm, kf, idx_pool):
    """m̂ (baseline head, σ_marg-standardized) vs the (z,τ₀) cell-mean target."""
    pf = norm["P_filt"]
    cells = cell_id(d)
    # pick a handful of distinct z slices (one alpha each) present in idx_pool
    z_all = np.round(d["z_grid"], 4)
    zs = np.unique(z_all[idx_pool])
    z_pick = zs[np.linspace(0, len(zs) - 1, 4).astype(int)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    rms_resid = []
    for zi, zval in enumerate(z_pick):
        rows = idx_pool[np.isclose(z_all[idx_pool], zval)]
        # one representative row (median alpha) for the prediction
        r0 = rows[len(rows) // 2]
        cid = int(cell_id(d, np.array([r0]))[0])
        cm = pf["cell_mean"].get(cid, pf["mu_marg"])             # (4,K) target cell-mean
        t_base = (cm - pf["mu_marg"]) / pf["sig_marg"]           # standardized target
        pred = jax.vmap(model)(jnp.asarray(d["x"][[r0]]),
                               jnp.asarray(d["tau0"][[r0]]))
        m_hat = np.asarray(pred["P_filt_base"])[0]               # (4,K) standardized
        tau0 = float(d["tau0"][r0])
        ax = axes[0, zi]; axr = axes[1, zi]
        for ci, nm in enumerate(CLS):
            ok = np.isfinite(t_base[ci])
            ax.semilogx(kf[ok], t_base[ci][ok], color=f"C{ci}", ls="-",
                        label=nm if zi == 0 else None)
            ax.semilogx(kf[ok], m_hat[ci][ok], color=f"C{ci}", ls="--")
            axr.semilogx(kf[ok], m_hat[ci][ok] - t_base[ci][ok], color=f"C{ci}")
            rms_resid.append(np.sqrt(np.nanmean((m_hat[ci][ok] - t_base[ci][ok]) ** 2)))
        ax.set_title(f"z={zval:.2f}, τ₀={tau0:.2f}"); ax.grid(alpha=0.3)
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_xlabel("k  [s/km]")
        if zi == 0:
            ax.set_ylabel("m̂ , target  (σ_marg units)")
            axr.set_ylabel("m̂ − target")
            ax.legend(fontsize=7, title="solid=target, dashed=m̂")
    fig.suptitle("04 — θ-blind BASELINE head m̂(z,τ₀) vs the (z,τ₀) cell-mean "
                 f"(residual RMS ≈ {np.nanmean(rms_resid):.3f} σ_marg)")
    fig.tight_layout()
    p = f"{OUTDIR}/04_baseline_head_fit.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, float(np.nanmean(rms_resid))


def fig05_residual_head_fit(model, d, va, norm):
    """r̂ (residual head) vs truth in σ_cosmo units — the recovered cosmology signal."""
    b = make_batch(d, va, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    r_hat = np.asarray(pred["P_filt_resid"])             # (n,4,K) standardized
    r_true = np.asarray(b["t_p_resid"])                  # (n,4,K) σ_cosmo-whitened truth
    mask = np.asarray(b["mask"])[:, None, :] & np.isfinite(r_true)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ci, nm in enumerate(CLS):
        m = mask[:, ci, :]
        xt = r_true[:, ci, :][m]; yp = r_hat[:, ci, :][m]
        g = np.isfinite(xt) & np.isfinite(yp)
        xt, yp = xt[g], yp[g]
        if len(xt) < 2:
            axes[ci].set_title(f"{nm} (no data)"); continue
        corr = np.corrcoef(xt, yp)[0, 1]
        axes[ci].hexbin(xt, yp, gridsize=55, mincnt=1, cmap="viridis")
        lim = np.nanpercentile(np.abs(xt), 99)
        axes[ci].plot([-lim, lim], [-lim, lim], "r--", lw=1)
        axes[ci].set_xlim(-lim, lim); axes[ci].set_ylim(-lim, lim)
        axes[ci].set_title(f"{nm}  (corr={corr:.2f})")
        axes[ci].set_xlabel("true residual  (σ_cosmo units)")
        if ci == 0:
            axes[ci].set_ylabel("predicted r̂")
    fig.suptitle("05 — RESIDUAL head r̂ vs truth (held-out, σ_cosmo units): the cosmology signal recovered")
    fig.tight_layout()
    p = f"{OUTDIR}/05_residual_head_fit.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def _pick_class_rows(d, idx_pool, ci, n=3):
    """Pick up to n held-out rows where class ci has the most counts (well-sampled)."""
    cnt = d["coarse_counts"][idx_pool, ci]
    order = idx_pool[np.argsort(-cnt)]
    return order[:n]


def fig06_pred_vs_true_p1d(model, d, idx_pool, norm, kf):
    """Pred vs true P1D(k) per class, a few held-out sims, log-log + frac-resid subpanel."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for ci, nm in enumerate(CLS):
        rows = _pick_class_rows(d, idx_pool, ci, n=3)
        _, P_pred = predict_phys(model, d, rows, norm)
        P_true = d["P_filt"][rows]
        ax = axes[0, ci]; axr = axes[1, ci]
        for ri in range(len(rows)):
            pt = P_true[ri, ci]; pp = P_pred[ri, ci]
            ok = np.isfinite(pt) & (pt > 0) & np.isfinite(pp) & (pp > 0)
            ax.loglog(kf[ok], pt[ok], "-", color=f"C{ri}", alpha=0.85,
                      label="true" if (ci == 0 and ri == 0) else None)
            ax.loglog(kf[ok], pp[ok], "--", color=f"C{ri}", alpha=0.85,
                      label="pred" if (ci == 0 and ri == 0) else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                axr.semilogx(kf[ok], pp[ok] / pt[ok] - 1.0, "-", color=f"C{ri}", alpha=0.85)
        ax.set_title(nm); ax.grid(alpha=0.3, which="both")
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_ylim(-0.3, 0.3); axr.set_xlabel("k  [s/km]")
        if ci == 0:
            ax.set_ylabel("P1D (filtered)"); axr.set_ylabel("pred/true − 1")
            ax.legend(fontsize=8)
    fig.suptitle("06 — Predicted vs true per-class P1D (held-out sims); subpanel = fractional residual")
    fig.tight_layout()
    p = f"{OUTDIR}/06_pred_vs_true_p1d.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def fig07_perclass_frac_error(model, d, idx_pool, norm, kf):
    """Per-class absolute fractional error vs k (median + IQR over held-out rows)."""
    _, P_pred = predict_phys(model, d, idx_pool, norm)
    P_true = d["P_filt"][idx_pool]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.abs(P_pred / P_true - 1.0)             # (n,4,K)
    ok = np.isfinite(P_true) & (P_true > 0) & np.isfinite(P_pred) & (P_pred > 0)
    frac = np.where(ok, frac, np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    rms_overall = {}
    for ci, nm in enumerate(CLS):
        med = np.nanmedian(frac[:, ci, :], axis=0)
        q1 = np.nanpercentile(frac[:, ci, :], 25, axis=0)
        q3 = np.nanpercentile(frac[:, ci, :], 75, axis=0)
        ax.fill_between(kf, q1, q3, color=f"C{ci}", alpha=0.15)
        rms = np.sqrt(np.nanmean(frac[:, ci, :] ** 2))
        rms_overall[nm] = float(rms)
        ax.loglog(kf, med, color=f"C{ci}", label=f"{nm} (RMS={rms:.3f})")
    ax.set_xlabel("k  [s/km]"); ax.set_ylabel("|pred/true − 1|  (median, IQR shaded)")
    ax.set_title("07 — Per-class absolute fractional error vs k\n"
                 "(where in k the clean-limited error lives)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = f"{OUTDIR}/07_perclass_frac_error.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, rms_overall


def _theta_tracking(model, d, va, norm):
    """Within-(z,τ₀)-cell θ-deviation: pred vs true (the headline metric)."""
    b = make_batch(d, va, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    pred = {k: np.asarray(v) for k, v in pred.items()}
    P_pred = reconstruct_P_filt(pred["P_filt_base"], pred["P_filt_resid"], norm["P_filt"])
    lp = safe_log(P_pred); lt = safe_log(d["P_filt"][va])
    cv = cell_id(d, va)
    dev_t, dev_p = [], []
    for cl in np.unique(cv):
        m = cv == cl
        if m.sum() < 2:
            continue
        dev_t.append((lt[m] - np.nanmean(lt[m], 0)).ravel())
        dev_p.append((lp[m] - np.nanmean(lp[m], 0)).ravel())
    dev_t = np.concatenate(dev_t); dev_p = np.concatenate(dev_p)
    g = np.isfinite(dev_t) & np.isfinite(dev_p)
    corr = float(np.corrcoef(dev_t[g], dev_p[g])[0, 1])
    ratio = float(np.nanstd(dev_p[g]) / np.nanstd(dev_t[g]))
    return dev_t[g], dev_p[g], corr, ratio


def fig08_theta_tracking(model, d, va, norm):
    """Within-cell θ-tracking (AFTER, this model) next to the saved BEFORE figure."""
    dev_t, dev_p, corr, ratio = _theta_tracking(model, d, va, norm)
    before = "/home/mfho/hcd_priya/figures/analysis/04_emulator/fold0_cosmology_tracking.png"

    fig = plt.figure(figsize=(12.5, 5.6))
    ax1 = fig.add_subplot(1, 2, 2)
    hb = ax1.hexbin(dev_t, dev_p, gridsize=65, mincnt=1, cmap="viridis")
    lim = np.nanpercentile(np.abs(dev_t), 99)
    ax1.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("TRUE log-P1D deviation from (z,τ₀)-cell mean  [cosmology signal]")
    ax1.set_ylabel("PRED deviation from cell mean")
    ax1.set_title(f"AFTER redesign (this fold-0 model)\ncorr={corr:.2f}, spread-ratio={ratio:.2f}")
    ax1.legend(); fig.colorbar(hb, ax=ax1, label="count")

    ax0 = fig.add_subplot(1, 2, 1)
    if Path(before).exists():
        ax0.imshow(plt.imread(before)); ax0.axis("off")
        ax0.set_title("BEFORE (global-σ baseline; corr≈0.80)")
    else:
        ax0.text(0.5, 0.5, "BEFORE figure not found", ha="center")
        ax0.axis("off")
    fig.suptitle("08 — Within-(z,τ₀)-cell θ-tracking: global-σ (before) vs conditional σ_cosmo (after)")
    fig.tight_layout()
    p = f"{OUTDIR}/08_theta_tracking.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p, corr, ratio


def fig09_cosmology_response(model, d, idx_pool, norm, kf, params=("ns", "Ap")):
    """σ_cosmo·∂r̂/∂θ vs k for a few params via jax.jacfwd (the k-dependent sensitivity)."""
    pf = norm["P_filt"]
    sig_cosmo = jnp.asarray(pf["sig_cosmo"])             # (4,K)
    pidx = [PARAM_NAMES.index(pp) for pp in params]
    # response = ∂logP̂/∂θ_unit  (the baseline is θ-blind, so it is sig_cosmo·∂r̂/∂θ).
    # Pick a representative mid-grid row.
    r0 = idx_pool[len(idx_pool) // 2]
    x0 = jnp.asarray(d["x"][r0]); tau0 = float(d["tau0"][r0])

    def logP_of_x(x):
        pred = model(x, tau0)
        r_hat = pred["P_filt_resid"]                     # (4,K)
        return sig_cosmo * r_hat                         # ∂logP̂/∂x = sig_cosmo·∂r̂/∂x

    J = jax.jacfwd(logP_of_x)(x0)                        # (4,K,10)
    J = np.asarray(J)
    z_r0 = float(d["z_grid"][r0])

    fig, axes = plt.subplots(1, len(params), figsize=(6.5 * len(params), 4.8),
                             squeeze=False)
    for j, (pp, pj) in enumerate(zip(params, pidx)):
        ax = axes[0, j]
        for ci, nm in enumerate(CLS):
            ax.semilogx(kf, J[ci, :, pj], color=f"C{ci}", label=nm)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("k  [s/km]")
        ax.set_ylabel(r"$\partial \ln P / \partial \hat\theta$  (unit-cube θ)")
        ax.set_title(f"response to {pp}  (z={z_r0:.2f}, τ₀={tau0:.2f})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("09 — Learned cosmology response σ_cosmo·∂r̂/∂θ vs k (jax.jacfwd)")
    fig.tight_layout()
    p = f"{OUTDIR}/09_cosmology_response.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--n-basis", type=int, default=24)
    ap.add_argument("--reuse", action="store_true",
                    help="reload the cached walkthrough checkpoint instead of retraining")
    args = ap.parse_args()

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    d = load_cache(CACHE)
    kf = d["kfkms"][0]
    print(f"cache: {d['P_tier_p'].shape[0]} rows, n_k={len(kf)}")

    model, norm, history, (tr, va, ho) = train_or_load(
        d, epochs=args.epochs, n_basis=args.n_basis, reuse=args.reuse)

    produced = {}
    skipped = {}

    p, cos_frac = fig01_variance_decomposition(d, kf); produced["01"] = p
    p, ratio_med = fig02_whitening_scales(d, norm, kf); produced["02"] = p
    p, terms = fig03_loss_curves(model, d, va, norm, history); produced["03"] = p
    p, base_rms = fig04_baseline_head_fit(model, d, norm, kf, tr); produced["04"] = p
    p = fig05_residual_head_fit(model, d, va, norm); produced["05"] = p
    p = fig06_pred_vs_true_p1d(model, d, va, norm, kf); produced["06"] = p
    p, perclass_rms = fig07_perclass_frac_error(model, d, va, norm, kf); produced["07"] = p
    p, corr, spread = fig08_theta_tracking(model, d, va, norm); produced["08"] = p
    try:
        p = fig09_cosmology_response(model, d, va, norm, kf, params=("ns", "Ap"))
        produced["09"] = p
    except Exception as e:
        skipped["09"] = repr(e)

    # ---- headline numbers ----
    head = {
        "n_epochs": int(len(history["train_loss"])),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "theta_tracking_corr": corr,
        "theta_spread_ratio": spread,
        "perclass_abs_frac_rms": perclass_rms,
        "cosmology_logvar_fraction_pct": cos_frac,
        "sig_cosmo_over_sig_marg_median": ratio_med,
        "baseline_head_rms_sigmarg": base_rms,
        "val_per_term_loss": terms,
    }
    with open(f"{OUTDIR}/headline_numbers.json", "w") as f:
        json.dump(head, f, indent=2)

    print("\n==== HEADLINE NUMBERS ====")
    print(json.dumps(head, indent=2))
    print("\n==== FIGURES PRODUCED ====")
    for k, v in produced.items():
        print(f"  {k}: {v}")
    if skipped:
        print("==== SKIPPED ====")
        for k, v in skipped.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
