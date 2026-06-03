"""PUSH the residual head to KILL the coherent k-tilt the PI rejected.

The PI rejects the *coherent* (mean-over-cosmologies) residual at the failing
bands: clean/subDLA at LOW k (k<0.01) and DLA at HIGH k (k>0.03). The current
production tuning (commit 73009c5: term_w[p_resid]=8, edge-emphasis gain3/lowk2,
wd3e-4, n_basis24, early-stop) only HALVED that coherent tilt (~1.2% clean at
k<0.005 vs the 0.73% CV floor) — REDUCED, not killed.

The diagnosis (jax-traps #23): an unweighted/MSE-only residual leaves the
band-edge an UNCONSTRAINED COHERENT degree of freedom. MSE is dominated by the
per-sim CV SCATTER and does NOT directly penalize the coherent (θ-mean) offset —
the tilt is exactly such a θ-mean offset. This script adds the missing lever:

  THE COHERENT-RESIDUAL (DE-BIAS) LOSS TERM.
  Within each minibatch, group rows by (z,τ₀)-CELL and penalize the per-cell mean
  over the sims (cosmologies) of the residual fit error:
       L_coh = Σ_cell  Σ_(c,k)  k_w(k) · ⟨ r̂ − t_resid ⟩_θ²
  Because the training target t_resid = (logP − m_cell)/σ_cosmo has ⟨t_resid⟩_θ=0
  per cell BY CONSTRUCTION (m_cell is the train cell-mean), this drives the
  systematic θ-mean of the residual head toward zero — i.e. it FLATTENS the
  coherent tilt that the per-sim MSE leaves free. Using (r̂ − t_resid) rather than
  r̂ alone makes the estimator robust to minibatch subsampling of θ within a cell.

This is an EXPLORATION script (build on the train_fold knobs in-process; do NOT
edit production model/train). It sweeps coherent-weight × k-emphasis × capacity,
scores each on the fold-0 coherent-vs-CV split + A_p/ns Fisher-bias + deployed
medians, then MULTI-FOLD validates the winner. Reuses the diag scoring from
scripts/diag_ap_fisher_bias.py and diag_theta_tracking_honest.py.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/push_residual_refine.py [opts]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log,
    edge_emphasis_k_weight,
)
from hcd_analysis.emulator.model import (
    Emulator, masked_mse, _k_weight_from_batch,
)
from hcd_analysis.emulator import train as T

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
CKPT_DIR = "/home/mfho/hcd_priya/checkpoints"
CV_JSON = f"{OUT}/diag_lfhf_tilt_and_cv.json"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
N_CELLS = 360          # static segment count = 18 z * 20 alpha (cell ids in [0,360))


# ===========================================================================
# CUSTOM LOSS with the coherent de-bias term
# ===========================================================================
def coherent_debias_term(preds, batch, k_w_coh):
    """Σ_cell ⟨ r̂ − t_resid ⟩_θ² over (class,k), k-weighted (k_w_coh, (1,1,K) or None).

    Segment-sum the per-row (r̂ − t_resid) into the GLOBAL (z,τ₀)-cell bins
    (num_segments=N_CELLS, static), divide by the per-cell row count to get the
    per-cell mean over the sims in that cell, and average its square over the
    populated cells / valid (class,k). NaN-safe: above-Nyquist bins are zeroed via
    the per-row mask BEFORE the segment-sum and excluded from the per-cell count.

    This is the de-bias lever: it penalizes the SYSTEMATIC (cosmology-mean) residual
    that the per-sim MSE leaves unconstrained — exactly the coherent k-tilt.
    """
    rhat = preds["P_filt_resid"]                       # (B,4,K)
    t = jnp.nan_to_num(batch["t_p_resid"], nan=0.0)    # (B,4,K)
    m = batch["mask"][:, None, :]                       # (B,1,K) Nyquist mask
    err = jnp.where(m, rhat - t, 0.0)                   # (B,4,K) per-row resid error
    seg = batch["cell"]                                 # (B,) global cell id

    # per-cell sum of err and of the mask (the # finite sims contributing per (c,k))
    num = jax.ops.segment_sum(err, seg, num_segments=N_CELLS)         # (N_CELLS,4,K)
    cnt = jax.ops.segment_sum(
        jnp.broadcast_to(m.astype(err.dtype), err.shape), seg,
        num_segments=N_CELLS)                                        # (N_CELLS,4,K)
    cell_mean_err = jnp.where(cnt > 0, num / jnp.maximum(cnt, 1.0), 0.0)  # ⟨r̂−t⟩_θ
    sq = cell_mean_err ** 2
    if k_w_coh is not None:
        sq = sq * k_w_coh                              # (1,1,K) broadcast -> band emphasis
    valid = (cnt > 0).astype(sq.dtype)
    denom = jnp.maximum(jnp.sum(valid * (k_w_coh if k_w_coh is not None else 1.0)), 1.0)
    return jnp.sum(sq) / denom


def make_loss(term_w, k_w_resid, k_w_coh, w_coh):
    """joint_loss + w_coh·coherent_debias_term, with per-k emphasis on BOTH the
    per-sim residual MSE (k_w_resid, via the batch k_weight) and the coherent
    de-bias term (k_w_coh). Returns a closure loss_fn(model, batch).

    The per-sim MSE part reproduces production joint_loss exactly (so the baseline,
    Head-A, delta terms are unchanged); only the cosmology residual gets the extra
    coherent-mean penalty. k_w_resid is applied inside the batch (k_weight key);
    k_w_coh is a (1,1,K) array applied here to the de-bias term.
    """
    tw = term_w or {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0,
                    "p_resid": 1.0, "delta": 1.0}

    def loss_fn(model, batch):
        preds = jax.vmap(model)(batch["x"], batch["tau0"])
        la_cddf = masked_mse(preds["f_nhi"], batch["t_f_nhi"],
                             batch["t_f_nhi_mask"], weight=batch["inv_nalpha"][:, None])
        la_dndx = masked_mse(preds["dndx"], batch["t_dndx"],
                             batch["t_dndx_mask"], weight=batch["inv_nalpha"][:, None])
        m3 = batch["mask"][:, None, :]
        wcls4 = batch["inv_nc"][:, :, None]
        wcls3 = batch["inv_nc"][:, 1:, None]
        mask4 = m3 & jnp.ones_like(batch["t_p_resid"], bool)
        lb_base = masked_mse(preds["P_filt_base"], batch["t_p_base"], mask4, weight=wcls4)
        kw = _k_weight_from_batch(batch)                # the per-sim residual k-emphasis
        w_resid = wcls4 if kw is None else wcls4 * kw
        lb_resid = masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=w_resid)
        lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                           m3 & jnp.ones_like(batch["t_delta"], bool), weight=wcls3)
        base = (tw["f_nhi"]*la_cddf + tw["dndx"]*la_dndx + tw["p_base"]*lb_base
                + tw["p_resid"]*lb_resid + tw["delta"]*lb_dl)
        if w_coh > 0.0:
            base = base + w_coh * coherent_debias_term(preds, batch, k_w_coh)
        return base

    return loss_fn


def coherent_metric(model, batch, k_w_coh):
    """The val-side coherent de-bias term value (early-stop / monitor)."""
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    return float(coherent_debias_term(preds, batch, k_w_coh))


# ===========================================================================
# TRAIN one fold with the custom coherent-aware loss (mirrors train_fold's
# two-phase: deep θ-blind baseline pre-fit + freeze, then train the residual).
# ===========================================================================
def _add_cell(batch_np, d, idx):
    batch_np = dict(batch_np)
    batch_np["cell"] = cell_id(d, idx).astype(np.int32)
    return batch_np


def prefit_model(d, train_idx, *, n_basis, n_k, seed, baseline_n_layers,
                 baseline_width, prefit_epochs, prefit_lr=2e-3):
    """Build the SVD-warm-started Emulator and pre-fit + freeze its θ-blind baseline.

    Factored out so a sweep can reuse ONE prefit across recipes that share
    (fold, n_basis, seed, baseline geometry) — the prefit depends only on those,
    not on the coherent-term / k-weight knobs. Returns (model, norm_stats).
    """
    key = jax.random.PRNGKey(seed)
    train_idx = np.asarray(train_idx)
    norm_stats = T._fit_norm(d, train_idx)
    p_filt_basis_init, baseline_basis_init = T._warmstart_bases(
        d, train_idx, norm_stats, n_basis, n_k)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis,
                     p_filt_basis_init=p_filt_basis_init,
                     baseline_basis_init=baseline_basis_init,
                     baseline_n_layers=baseline_n_layers,
                     baseline_width=baseline_width, key=key)
    model = T._prefit_baseline(model, d, train_idx, norm_stats,
                               epochs=prefit_epochs, lr=prefit_lr, seed=seed)
    return model, norm_stats


def train_fold_coh(d, train_idx, val_idx, *, n_basis, lr, epochs, batch_size,
                   seed, patience, n_k, term_w, k_w_resid, k_w_coh, w_coh,
                   baseline_n_layers=3, baseline_width=256,
                   prefit_epochs=8000, prefit_lr=2e-3, weight_decay=3e-4,
                   early_stop="coh", prefit=None):
    """Two-phase train with the coherent-aware loss.

    Phase 1: pre-fit + FREEZE the deep θ-blind baseline (production recipe; reuse
      train._prefit_baseline). Pass ``prefit=(model, norm_stats)`` to REUSE a cached
      prefit (the baseline depends only on fold/n_basis/seed/geometry, not on the
      coherent knobs) — saves the ~60s baseline fit across a sweep.
    Phase 2: train encoder/HeadA/HeadB on the custom loss (per-sim MSE + coherent
      de-bias). Early-stop on the val COHERENT term ("coh"), the val residual MSE
      ("resid"), or their sum ("both").
    Returns (best_model, norm_stats, history).
    """
    train_idx = np.asarray(train_idx); val_idx = np.asarray(val_idx)
    rng = np.random.default_rng(seed)
    if prefit is not None:
        model, norm_stats = prefit
    else:
        model, norm_stats = prefit_model(
            d, train_idx, n_basis=n_basis, n_k=n_k, seed=seed,
            baseline_n_layers=baseline_n_layers, baseline_width=baseline_width,
            prefit_epochs=prefit_epochs, prefit_lr=prefit_lr)

    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    total_steps = steps_per_epoch * epochs
    opt = T.make_optimizer(lr=lr, steps=total_steps, weight_decay=weight_decay)

    join_mask = T._trainable_mask(model, train_baseline=False, train_rest=True)
    diff_model, static_model = eqx.partition(model, join_mask)
    opt_state = opt.init(eqx.filter(diff_model, eqx.is_array))
    loss_fn = make_loss(term_w, k_w_resid, k_w_coh, w_coh)

    @eqx.filter_jit
    def step(dm, sm, st, batch):
        def wrapped(m):
            return loss_fn(eqx.combine(m, sm), batch)
        l, g = eqx.filter_value_and_grad(wrapped)(dm)
        u, st = opt.update(g, st, eqx.filter(dm, eqx.is_array))
        return eqx.apply_updates(dm, u), st, l

    def _mb(idx):
        b = make_batch(d, idx, norm_stats, k_weight=k_w_resid)
        return _add_cell(b, d, idx)

    val_batch = T._to_jnp_batch(_mb(val_idx))
    history = {"train_loss": [], "val_resid": [], "val_coh": []}
    best_metric = np.inf
    best_model = model
    stall = 0
    for ep in range(epochs):
        ep_losses = []
        for mb in T._iter_minibatches(rng, train_idx, batch_size):
            batch = T._pad_batch(T._to_jnp_batch(_mb(mb)), batch_size)
            # padded rows get cell -> a valid bin but mask False, so they add 0 to
            # both the MSE and the coherent-segment-sum numerator/count. Belt: clamp.
            batch["cell"] = jnp.clip(batch["cell"], 0, N_CELLS - 1)
            diff_model, opt_state, loss = step(diff_model, static_model, opt_state, batch)
            ep_losses.append(float(loss))
        model_now = eqx.combine(diff_model, static_model)
        v_resid = float(_val_resid_mse(model_now, val_batch, k_w_resid is not None))
        v_coh = coherent_metric(model_now, val_batch, k_w_coh)
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_resid"].append(v_resid)
        history["val_coh"].append(v_coh)
        if early_stop == "coh":
            metric = v_coh
        elif early_stop == "resid":
            metric = v_resid
        else:                       # "both"
            metric = v_resid + v_coh
        if metric < best_metric - 1e-12:
            best_metric = metric
            best_model = model_now
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    history = {k: np.asarray(v) for k, v in history.items()}
    return best_model, norm_stats, history


def _val_resid_mse(model, batch, use_kw):
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    mask4 = batch["mask"][:, None, :] & jnp.ones_like(batch["t_p_resid"], bool)
    wcls4 = batch["inv_nc"][:, :, None]
    if use_kw:
        kw = _k_weight_from_batch(batch)
        if kw is not None:
            wcls4 = wcls4 * kw
    return masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=wcls4)


# ===========================================================================
# SCORING (reuse the diag math): coherent-vs-CV split + A_p/ns Fisher-bias +
# deployed medians + honest (a)/(b).
# ===========================================================================
def _load_cv_floor():
    cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
    cv = {}
    for band, v in cvj.items():
        lo, hi = (float(x) for x in band.split("-"))
        cv[(lo, hi)] = float(v["median_pct"]) / 100.0
    return cv


def coherent_vs_cv(model, d, va, norm, z_fid=3.0):
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    resid = np.asarray(pred["P_filt_resid"])
    logP_hat = base + sc * resid
    logP_true = safe_log(d["P_filt"][va][vz])
    frac = np.exp(logP_hat - logP_true) - 1.0
    m = np.isfinite(logP_true)
    frac = np.where(m, frac, np.nan)
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"][va][vz]),
                               d["kfkms"][va][vz], np.nan), 0)
    with np.errstate(invalid="ignore"):
        coherent = np.nanmean(frac, axis=0)
        scatter = np.nanstd(frac, axis=0)
    cv = _load_cv_floor()

    def band_stat(lo, hi):
        kb = (kf >= lo) & (kf < hi) & np.isfinite(kf)
        out = {}
        for ci, nm in enumerate(CLS):
            c = coherent[ci, kb]; s = scatter[ci, kb]
            out[nm] = dict(coherent_rms=float(np.sqrt(np.nanmean(c ** 2))),
                           coherent_mean=float(np.nanmean(c)),
                           scatter_rms=float(np.sqrt(np.nanmean(s ** 2))),
                           n_kbins=int(np.isfinite(c).sum()))
        best = min(cv, key=lambda b: abs((b[0]+b[1])/2 - (lo+hi)/2))
        out["cv_floor_frac"] = float(cv[best])
        return out

    bands = {"lowk_0-0.005": band_stat(0.0, 0.005),
             "lowk_0.005-0.01": band_stat(0.005, 0.01),
             "mid_0.01-0.03": band_stat(0.01, 0.03),
             "high_0.03-0.07": band_stat(0.03, 0.07)}
    return dict(kf=kf, coherent=coherent, scatter=scatter, bands=bands, cv=cv)


def percell_coherent(model, d, va, norm, z_fid=3.0):
    """The PI's EXACT metric: the coherent residual per (z,τ₀)-CELL.

    coherent[cell,c,k] = ⟨P̂/P−1⟩_θ over the val sims SHARING that (z,τ₀) cell;
    band coherent_rms = RMS over (cells, k) in the band. This is ``Σ_cell ⟨P̂−P⟩_θ²``
    (the systematic, per-cell), NOT the pooled-over-cells z=3 mean in coherent_vs_cv
    (which mixes τ₀ cells). z restricted to z_fid (≥2 sims/cell). Returns per-class
    per-band coherent_rms.
    """
    pf = norm["P_filt"]; sc = pf["sig_cosmo"]
    sub = va[np.isclose(d["z_grid"][va], z_fid, atol=0.05)]
    x = jnp.asarray(d["x"][sub]); tau = jnp.asarray(d["tau0"][sub])
    pr = jax.vmap(model)(x, tau)
    base = np.asarray(pr["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    res = np.asarray(pr["P_filt_resid"])
    logPhat = base + sc * res
    logPt = safe_log(d["P_filt"][sub])
    frac = np.exp(logPhat - logPt) - 1.0
    m = np.isfinite(logPt)
    frac = np.where(m, frac, np.nan)
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"][sub]), d["kfkms"][sub], np.nan), 0)
    cells = cell_id(d, sub)
    # (n_cells, 4, K) per-cell mean over the sims in that cell
    uc = np.unique(cells)
    with np.errstate(invalid="ignore"):
        pc = np.stack([np.nanmean(frac[cells == c], axis=0) for c in uc])  # (Nc,4,K)
    bands = {"lowk_0-0.005": (0.0, 0.005), "lowk_0.005-0.01": (0.005, 0.01),
             "mid_0.01-0.03": (0.01, 0.03), "high_0.03-0.07": (0.03, 0.07)}
    out = {}
    for bn, (lo, hi) in bands.items():
        kb = (kf >= lo) & (kf < hi) & np.isfinite(kf)
        out[bn] = {}
        for ci, nm in enumerate(CLS):
            block = pc[:, ci, :][:, kb]
            out[bn][nm] = float(np.sqrt(np.nanmean(block ** 2)))
    return out


def fisher_bias(model, d, va, norm, z_fid=3.0):
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]; n_k = sc.shape[1]
    fiducial = np.full(9, 0.5)
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
    kvalid = (mask_va[vz].mean(0) > 0.5)
    z_j = jnp.asarray(z_unit_fid)

    def rhat(theta9_tau):
        theta9 = theta9_tau[:9]; tau = theta9_tau[9]
        x = jnp.concatenate([theta9, z_j[None]])
        return model(x, tau)["P_filt_resid"]

    p0 = jnp.asarray(np.concatenate([fiducial, [tau0_fid]]), dtype=jnp.float64)
    Jr = np.asarray(jax.jacfwd(rhat)(p0))
    J_logP = Jr * sc[:, :, None]
    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    resid = np.asarray(pred["P_filt_resid"])
    logP_hat = base + sc * resid
    logP_true = safe_log(d["P_filt"][va][vz])
    m_slice = np.isfinite(logP_true)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j], (logP_hat - logP_true)[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])
    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
               or not np.isfinite(delta[ci, j]):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    J = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    J9 = J[:, :9]
    F = (J9.T * Cinv) @ J9
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J9.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    return dict(
        n_modes=int(len(dv)), fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
        bias_in_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        dtheta_over_prior={PARAMS[i]: float(abs(dtheta[i])) for i in range(9)},
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)})


def deployed_medians(model, d, va, norm):
    """Deployed median |P̂/P−1| per class over ALL val rows (not just z=3)."""
    pf = norm["P_filt"]; sc = pf["sig_cosmo"]
    x_va = jnp.asarray(d["x"][va]); tau_va = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    resid = np.asarray(pred["P_filt_resid"])
    logP_hat = base + sc * resid
    logP_true = safe_log(d["P_filt"][va])
    frac = np.abs(np.exp(logP_hat - logP_true) - 1.0)
    m = np.isfinite(logP_true)
    out = {}
    for ci, nm in enumerate(CLS):
        f = frac[:, ci, :][m[:, ci, :]]
        out[nm] = float(np.median(f)) if f.size else np.nan
    return out


def honest_ab(model, d, va, norm):
    """Deployed whitened (a) residual-fit + (b) baseline-misfit per class."""
    pf = norm["P_filt"]
    sig_marg, mu_marg, sc = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    b = make_batch(d, va, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    base = np.asarray(pred["P_filt_base"]); resid = np.asarray(pred["P_filt_resid"])
    m_hat = base * sig_marg + mu_marg
    lt = safe_log(d["P_filt"][va])
    cv = cell_id(d, va)
    m_cell = np.stack([pf["cell_mean"].get(int(c), mu_marg) for c in cv])
    r_true = (lt - m_hat) / sc
    t_resid = (lt - m_cell) / sc
    err_resid = resid - t_resid
    err_base = (m_hat - m_cell) / sc
    out = {}
    for ci, nm in enumerate(CLS):
        st = np.nanstd(r_true[:, ci, :][np.isfinite(r_true[:, ci, :])])
        a = np.sqrt(np.nanmean(err_resid[:, ci, :][np.isfinite(err_resid[:, ci, :])] ** 2)) / st
        bb = np.sqrt(np.nanmean(err_base[:, ci, :][np.isfinite(err_base[:, ci, :])] ** 2)) / st
        out[nm] = dict(resid_fit=float(a), base_misfit=float(bb))
    return out


def score_model(model, d, va, norm, tag):
    split = coherent_vs_cv(model, d, va, norm)
    fish = fisher_bias(model, d, va, norm)
    med = deployed_medians(model, d, va, norm)
    ab = honest_ab(model, d, va, norm)
    pc = percell_coherent(model, d, va, norm)        # the PI's per-cell metric
    return dict(tag=tag, split={b: {nm: split["bands"][b][nm] for nm in CLS} |
                                {"cv_floor_frac": split["bands"][b]["cv_floor_frac"]}
                                for b in split["bands"]},
                percell=pc,
                fisher=fish, medians=med, honest_ab=ab,
                _kf=split["kf"], _coh=split["coherent"], _sca=split["scatter"],
                _cv=split["cv"])


def print_score(s):
    print(f"\n--- {s['tag']} ---")
    f = s["fisher"]
    print(f"  Fisher cond={f['fisher_cond']:.2e}  "
          f"A_p bias={f['bias_in_sigma']['Ap']:+.3f}σ  "
          f"ns bias={f['bias_in_sigma']['ns']:+.3f}σ")
    print(f"  deployed median |P̂/P−1|: " +
          "  ".join(f"{nm}={s['medians'][nm]*100:.2f}%" for nm in CLS))
    print(f"  POOLED coherent (mixes τ₀ cells):")
    print(f"  {'band':18} {'CVfl%':>6} | " +
          "  ".join(f"{nm} coh%" for nm in CLS))
    for b in s["split"]:
        cvf = s["split"][b]["cv_floor_frac"]
        vals = "  ".join(f"{s['split'][b][nm]['coherent_rms']*100:5.2f}" for nm in CLS)
        print(f"  {b:18} {cvf*100:6.2f} | {vals}")
    print(f"  PER-CELL coherent (the PI's metric, Σ_cell ⟨P̂/P−1⟩_θ²):")
    for b in s["percell"]:
        vals = "  ".join(f"{s['percell'][b][nm]*100:5.2f}" for nm in CLS)
        print(f"  {b:18} {'':>6} | {vals}")


# ===========================================================================
# Figure: coherent residual vs k per class (before vs after overlay)
# ===========================================================================
def fig_coherent_overlay(scores, path, title):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax = ax.ravel()
    cv = scores[0]["_cv"]
    for ci, nm in enumerate(CLS):
        a = ax[ci]
        for s in scores:
            kf = s["_kf"]; coh = s["_coh"][ci]
            g = np.isfinite(kf) & np.isfinite(coh)
            a.semilogx(kf[g], coh[g] * 100, lw=1.6, label=s["tag"])
        # CV floor band overlay
        for (lo, hi), v in cv.items():
            a.hlines(v * 100, lo, hi, color="k", ls="--", lw=1.0, alpha=0.6)
            a.hlines(-v * 100, lo, hi, color="k", ls="--", lw=1.0, alpha=0.6)
        a.axhline(0, color="grey", lw=0.6)
        a.axvline(0.009, color="purple", ls=":", lw=1, alpha=0.7)
        a.set_title(nm)
        a.set_ylabel("coherent ⟨P̂/P−1⟩ [%]")
        a.grid(alpha=0.3, which="both")
        a.set_ylim(-3, 3)
        if ci == 0:
            a.legend(fontsize=7, ncol=2)
    for a in ax[2:]:
        a.set_xlabel("k [s/km]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def _strip(s):
    """Drop the heavy arrays before JSON dump."""
    return {k: v for k, v in s.items() if not k.startswith("_")}


def run_recipe(d, fold, n_k, *, n_basis, term_w, edge_gain, lowk_extra,
               coh_edge_gain, coh_lowk_extra, w_coh, epochs, patience,
               weight_decay, early_stop, baseline_n_layers, baseline_width,
               seed, lr=1e-3, batch_size=512, prefit_epochs=8000, prefit=None,
               coh_flat=False):
    tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
    k_w_resid = None
    if edge_gain != 0.0 or lowk_extra != 0.0:
        k_w_resid = edge_emphasis_k_weight(d["kfkms"][0], edge_gain=edge_gain,
                                           lowk_extra=lowk_extra)
    k_w_coh = None
    if w_coh > 0 and coh_flat:
        # FLAT (uniform-k) coherent de-bias — the winning k-shape: it de-biases
        # ALL bands evenly, so the low-k coherent comes down WITHOUT the edge
        # weight's mid-band trade (see the k-shape sweep). None -> joint_loss treats
        # it as flat anyway; an explicit ones array keeps the intent legible.
        k_w_coh = jnp.ones(n_k)[None, None, :]
    elif w_coh > 0 and (coh_edge_gain != 0.0 or coh_lowk_extra != 0.0):
        kwc = edge_emphasis_k_weight(d["kfkms"][0], edge_gain=coh_edge_gain,
                                     lowk_extra=coh_lowk_extra)
        k_w_coh = jnp.asarray(kwc)[None, None, :]
    t0 = time.time()
    model, norm, hist = train_fold_coh(
        d, tr, va, n_basis=n_basis, lr=lr, epochs=epochs, batch_size=batch_size,
        seed=seed, patience=patience, n_k=n_k, term_w=term_w,
        k_w_resid=k_w_resid, k_w_coh=k_w_coh, w_coh=w_coh,
        baseline_n_layers=baseline_n_layers, baseline_width=baseline_width,
        prefit_epochs=prefit_epochs, weight_decay=weight_decay,
        early_stop=early_stop, prefit=prefit)
    dt = time.time() - t0
    n_ep = len(hist["train_loss"])
    return model, norm, (tr, va, ho), dt, n_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="sweep",
                    choices=["sweep", "multifold", "single"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--prefit-epochs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    # the WINNING recipe knobs (defaults = production 73009c5 + coherent term off)
    ap.add_argument("--n-basis", type=int, default=24)
    ap.add_argument("--p-resid-w", type=float, default=8.0)
    ap.add_argument("--edge-gain", type=float, default=3.0)
    ap.add_argument("--lowk-extra", type=float, default=2.0)
    ap.add_argument("--coh-edge-gain", type=float, default=3.0)
    ap.add_argument("--coh-lowk-extra", type=float, default=2.0)
    ap.add_argument("--coh-flat", action="store_true",
                    help="FLAT (uniform-k) coherent de-bias weight (the WINNER) — "
                         "de-biases all bands evenly, no edge-weight mid-band trade")
    ap.add_argument("--w-coh", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--early-stop", default="both", choices=["coh", "resid", "both"])
    ap.add_argument("--baseline-n-layers", type=int, default=3)
    ap.add_argument("--baseline-width", type=int, default=256)
    ap.add_argument("--tag", default="coh")
    args = ap.parse_args()

    d = load_cache(CACHE)
    n_k = d["P_tier_p"].shape[1]

    def tw(w):
        if w == 1.0:
            return None
        return {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0, "p_resid": w, "delta": 1.0}

    if args.mode == "single":
        model, norm, (tr, va, ho), dt, n_ep = run_recipe(
            d, args.fold, n_k, n_basis=args.n_basis, term_w=tw(args.p_resid_w),
            edge_gain=args.edge_gain, lowk_extra=args.lowk_extra,
            coh_edge_gain=args.coh_edge_gain, coh_lowk_extra=args.coh_lowk_extra,
            w_coh=args.w_coh, epochs=args.epochs, patience=args.patience,
            weight_decay=args.weight_decay, early_stop=args.early_stop,
            baseline_n_layers=args.baseline_n_layers, baseline_width=args.baseline_width,
            seed=args.seed, prefit_epochs=args.prefit_epochs, coh_flat=args.coh_flat)
        s = score_model(model, d, va, norm, args.tag)
        print(f"\n[{args.tag}] trained {n_ep} ep in {dt:.0f}s")
        print_score(s)
        json.dump(_strip(s), open(f"{OUT}/push_resid_{args.tag}.json", "w"), indent=2)
        print(f"wrote {OUT}/push_resid_{args.tag}.json")
        return

    if args.mode == "sweep":
        # Sweep the coherent-weight ladder (and a no-coherent baseline) on fold-0.
        recipes = [
            dict(tag="A_prod_baseline", w_coh=0.0),                      # = 73009c5
            dict(tag="B_coh0.5", w_coh=0.5),
            dict(tag="C_coh2", w_coh=2.0),
            dict(tag="D_coh8", w_coh=8.0),
            dict(tag="E_coh20", w_coh=20.0),
            dict(tag="F_coh8_nb32", w_coh=8.0, n_basis=32),
            dict(tag="G_coh8_nb48", w_coh=8.0, n_basis=48),
        ]
        tr0, va0, ho0 = make_splits(d, args.fold, n_folds=8, holdout_frac=0.15)
        prefit_cache = {}        # n_basis -> (model, norm) reused across recipes
        scores = []
        for r in recipes:
            nb = r.get("n_basis", args.n_basis)
            if nb not in prefit_cache:
                prefit_cache[nb] = prefit_model(
                    d, tr0, n_basis=nb, n_k=n_k, seed=args.seed,
                    baseline_n_layers=args.baseline_n_layers,
                    baseline_width=args.baseline_width,
                    prefit_epochs=args.prefit_epochs)
            model, norm, (tr, va, ho), dt, n_ep = run_recipe(
                d, args.fold, n_k, n_basis=nb, term_w=tw(args.p_resid_w),
                edge_gain=args.edge_gain, lowk_extra=args.lowk_extra,
                coh_edge_gain=args.coh_edge_gain, coh_lowk_extra=args.coh_lowk_extra,
                w_coh=r["w_coh"], epochs=args.epochs, patience=args.patience,
                weight_decay=args.weight_decay,
                early_stop=("resid" if r["w_coh"] == 0 else args.early_stop),
                baseline_n_layers=args.baseline_n_layers,
                baseline_width=args.baseline_width, seed=args.seed,
                prefit_epochs=args.prefit_epochs, prefit=prefit_cache[nb])
            s = score_model(model, d, va, norm, r["tag"])
            print(f"\n[{r['tag']}] w_coh={r['w_coh']} nb={nb} {n_ep}ep {dt:.0f}s")
            print_score(s)
            scores.append(s)
        fig_coherent_overlay(scores, f"{OUT}/push_resid_sweep_coherent_vs_k.png",
                             "Coherent residual vs k — coherent-weight sweep (fold-0, z=3)")
        json.dump([_strip(s) for s in scores],
                  open(f"{OUT}/push_resid_sweep.json", "w"), indent=2)
        print(f"\nwrote {OUT}/push_resid_sweep.json")
        return

    if args.mode == "multifold":
        folds = [int(x) for x in args.folds.split(",")]
        all_scores = {}
        per_fold_for_fig = []
        for fold in folds:
            model, norm, (tr, va, ho), dt, n_ep = run_recipe(
                d, fold, n_k, n_basis=args.n_basis, term_w=tw(args.p_resid_w),
                edge_gain=args.edge_gain, lowk_extra=args.lowk_extra,
                coh_edge_gain=args.coh_edge_gain, coh_lowk_extra=args.coh_lowk_extra,
                w_coh=args.w_coh, epochs=args.epochs, patience=args.patience,
                weight_decay=args.weight_decay, early_stop=args.early_stop,
                baseline_n_layers=args.baseline_n_layers,
                baseline_width=args.baseline_width, seed=args.seed,
                prefit_epochs=args.prefit_epochs, coh_flat=args.coh_flat)
            s = score_model(model, d, va, norm, f"fold{fold}")
            print(f"\n[fold{fold}] w_coh={args.w_coh} {n_ep}ep {dt:.0f}s")
            print_score(s)
            all_scores[f"fold{fold}"] = _strip(s)
            if fold in (0, 1, 2):
                per_fold_for_fig.append(s)
        # summary table across folds (PER-CELL coherent = the PI's metric)
        print("\n=== MULTI-FOLD SUMMARY (w_coh=%.1f, coh_flat=%s) ===" % (args.w_coh, args.coh_flat))
        print(f"{'fold':6} {'A_p σ':>7} {'ns σ':>7} | "
              f"{'cl lowk':>8} {'LLS lowk':>8} {'sub lowk':>8} {'cl mid':>7} {'cl med%':>7} (per-cell coh%)")
        for fk, s in all_scores.items():
            f = s["fisher"]
            pc = s["percell"]
            cl = pc["lowk_0-0.005"]["clean"] * 100
            ll = pc["lowk_0-0.005"]["LLS"] * 100
            sd = pc["lowk_0-0.005"]["subDLA"] * 100
            md = pc["mid_0.01-0.03"]["clean"] * 100
            med = s["medians"]["clean"] * 100
            print(f"{fk:6} {f['bias_in_sigma']['Ap']:+7.3f} {f['bias_in_sigma']['ns']:+7.3f} | "
                  f"{cl:8.2f} {ll:8.2f} {sd:8.2f} {md:7.2f} {med:7.2f}")
        if per_fold_for_fig:
            fig_coherent_overlay(per_fold_for_fig,
                                 f"{OUT}/push_resid_multifold_coherent_vs_k.png",
                                 f"Coherent residual vs k — winning recipe per fold (w_coh={args.w_coh})")
        json.dump(all_scores, open(f"{OUT}/push_resid_multifold_{args.tag}.json", "w"), indent=2)
        print(f"\nwrote {OUT}/push_resid_multifold_{args.tag}.json")
        return


if __name__ == "__main__":
    main()
