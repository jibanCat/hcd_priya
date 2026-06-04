#!/usr/bin/env python3
"""Diagnostic: TRAIN vs VAL convergence on the SAME (uniform joint) loss.

THE ISSUE this answers. The standard convergence figure (run_loso_sweep
``fig_perfold_convergence`` / ``loso_loss_curves``) plots two DIFFERENT
functions on one axis:

  * ``train_loss`` = the WEIGHTED+REGULARIZED training OBJECTIVE that AdamW
    actually minimizes — ``joint_loss`` with ``term_w[p_resid]=8``, the
    edge k-weight (gain3/lowk2), the FLAT coherent de-bias ``w_coh=80``, and
    the data-range soft down-weight (see ``train.py`` ``_loss_with_term_w`` +
    ``make_batch(k_weight=..., datarange=True)``);
  * ``val_loss`` = ``evaluate(model, val_batch)`` = the PLAIN UNIFORM
    ``joint_loss`` (uniform term_w, w_coh=0, no k-weight, no data-range).

Two different scalars -> train >> val even at epoch 0 (e.g. fold-0 train 0.42
vs val 0.18, a ~5x gap) — a WEIGHTED-vs-UNIFORM artifact, NOT a generalization
gap. The PI's sweet-spot check ("do train and val converge to similar?") is
only meaningful when train and val are the SAME loss.

WHAT THIS SCRIPT DOES. It re-trains a few LOSO folds (default 0, 3, 5) with the
FINALIZED recipe (``run_loso_sweep.FINAL_RECIPE``). The TRAINING STEP is left
exactly as production: the full weighted objective is what's optimized (we do
NOT change what's trained). But per epoch we additionally log BOTH

  * ``train_uniform`` = ``evaluate(model, train_eval_batch)`` — the UNIFORM
    joint_loss on a FIXED ~2000-row train subsample (same subsample every
    epoch); and
  * ``val_uniform``   = ``evaluate(model, val_eval_batch)`` — the UNIFORM
    joint_loss on the full val fold (this IS the existing ``val_loss``).

Both ``*_eval_batch`` are built by ``make_batch(d, idx, norm_stats)`` with NO
``k_weight`` and NO ``datarange`` -> the batch lacks those keys -> ``joint_loss``
falls back to uniform term_w, w_coh=0. So ``evaluate`` is the SAME function on
both, and ``train_uniform`` vs ``val_uniform`` is the genuine generalization
gap. Expect the ratio near 1 at the val-min (the healthy-convergence sweet
spot).

DELTA-CHANNEL DEGENERACY (the PRIMARY metric excludes it). The HCD Δ_c "delta"
channel has a degenerate train-split standardization under LOSO: in LLS low-k
bins the train sims' Δ_c is near-constant (std down to ~1e-7), so a held-out
sim's signed-log(delta) there standardizes to ~1e4-1e5 σ. On some folds (e.g.
fold 3) this makes the uniform DELTA term a huge, model-INDEPENDENT constant
(~1.7e5, unchanged across all epochs) that swamps the full uniform loss — a
normalization artifact of a minor nuisance channel, NOT a generalization signal,
and present in-range too (a data-range cut does not remove it). So the PRIMARY
same-loss metric is the uniform joint loss EXCLUDING the delta term
(``train_nodelta`` / ``val_nodelta`` — the cosmology-bearing p_resid/p_base + the
CDDF/dN/dX Head-A terms, all well-conditioned). The full (all-term) uniform loss,
the delta-only term, and the p_resid-only (pure cosmology) term are all logged as
references.

NOTE — fully-uniform vs production's ``val_loss``. ``joint_loss`` reads
``k_weight``/``datarange_weight`` FROM THE BATCH (via
``_k_weight_from_batch`` / ``_datarange_weight_from_batch``) regardless of
``term_w``. Production ``train_fold`` builds its val batch WITH those keys
(``_mb(val_idx)``), so its logged ``val_loss`` = ``evaluate`` still applies the
edge k-weight + data-range down-weight (only term_w is uniform and w_coh=0).
This script's eval batches OMIT those keys, so ``val_uniform`` here is FULLY
uniform — the cleaner same-as-train function. ``val_uniform`` therefore differs
slightly from production's ``val_loss``; that is intended (a fully-uniform loss
is the correct apples-to-apples reference for the train/val comparison).

The loop is a faithful REPLICA of ``train.train_fold`` (frozen pre-fit baseline,
partitioned AdamW, cosine LR, same minibatching/padding, the SAME early-stop
metric ``val_resid + w_coh*val_coh``), reusing the production helpers so the
trajectory matches. It does NOT edit model.py/data.py/likelihood.py/
multifidelity.py or train.py.

Figure: figures/analysis/04_emulator/convergence_same_loss.png
  one panel per fold: train (solid) and val (dashed) on the SAME uniform loss
  EXCLUDING the degenerate delta channel (left axis, directly comparable) + the
  pure-cosmology p_resid-only term (faint dotted) + the weighted training
  OBJECTIVE on a twin axis (context, clearly labeled). The val-min epoch is
  marked and the train/val ratio at the val-min is annotated.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_convergence_same_loss.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

# JAX HYGIENE: enable x64 BEFORE any array is created. Importing
# hcd_analysis.emulator also sets it (its __init__), but assert here so a future
# import-order change can't silently drop us to float32 (the structural P1D
# identities are bit-level and break in float32).
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, edge_emphasis_k_weight, DATA_RANGE,
)
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.model import p_resid_loss, coherent_resid_loss

# Reuse the FINALIZED recipe verbatim (single source of truth).
from scripts.run_loso_sweep import FINAL_RECIPE

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
FIGDIR = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
DEFAULT_FOLDS = (0, 3, 5)

# Term-selecting uniform weights. The eval batch carries NO k_weight / datarange
# weight, so joint_loss(model, batch, term_w=...) with w_coh=0 gives the PLAIN
# UNIFORM per-element-mean MSE of exactly the term(s) with weight 1 (others 0).
# PRIMARY same-loss metric = uniform joint loss EXCLUDING the delta channel
# (``W_NODELTA``). WHY EXCLUDE DELTA: the HCD Δ_c "delta" channel has a DEGENERATE
# train-split standardization under LOSO — in LLS low-k bins the train sims' Δ_c is
# near-constant (std down to ~1e-7), so a held-out sim's signed-log(delta) there
# standardizes to ~1e4-1e5 σ. That makes the uniform delta MSE a huge,
# model-INDEPENDENT constant on some folds (e.g. fold 3: delta term ~1.7e5,
# unchanged across 126 epochs) that swamps the comparison. It is a normalization
# artifact of a minor nuisance channel, NOT a generalization signal, and it is
# present in-range too (data-range restriction does not remove it). The
# cosmology-bearing P1D terms (p_resid, p_base) + the CDDF/dN/dX Head-A terms are
# all well-conditioned; the convergence question is about THOSE. We keep the full
# (all-term) uniform loss and the delta-only / p_resid-only terms as references.
_TERMS = ("f_nhi", "dndx", "p_base", "p_resid", "delta")
W_FULL = {t: 1.0 for t in _TERMS}
W_NODELTA = {**W_FULL, "delta": 0.0}
W_DELTA = {t: 0.0 for t in _TERMS}; W_DELTA["delta"] = 1.0
W_PRESID = {t: 0.0 for t in _TERMS}; W_PRESID["p_resid"] = 1.0
W_PBASE = {t: 0.0 for t in _TERMS}; W_PBASE["p_base"] = 1.0


def _uniform_term(model, batch, w):
    """Uniform joint_loss with term-selecting weights ``w`` (w_coh=0, no k/dr
    weight in batch) -> plain uniform MSE of the selected term(s)."""
    from hcd_analysis.emulator.model import joint_loss
    return float(joint_loss(model, batch, term_w=w, w_coh=0.0))


def train_fold_logging_uniform(d, train_idx, val_idx, *, recipe, term_w, k_weight,
                               seed, train_eval_subsample, rng_eval):
    """REPLICA of train.train_fold that ALSO logs uniform train/val joint_loss.

    Identical to ``train.train_fold`` in everything that affects the optimizer
    trajectory (norm fit, SVD warm-start, frozen pre-fit baseline, partitioned
    AdamW, cosine LR, minibatch shuffle+pad, the SAME weighted loss closure, the
    SAME early-stop metric). The ONLY addition: per epoch we evaluate the UNIFORM
    ``joint_loss`` (``T.evaluate``) on a fixed train subsample and on full val.

    Returns a history dict with, per epoch (all uniform metrics are the SAME
    function on train & val — train on a fixed ~2000-row subsample, val on the
    full fold):
      train_obj                  : the WEIGHTED+REGULARIZED objective AdamW
                                   minimizes (mean over minibatches) ==
                                   train.train_fold's ``train_loss``;
      train_nodelta/val_nodelta  : PRIMARY same-loss metric — uniform joint loss
                                   EXCLUDING the degenerate delta channel (the
                                   apples-to-apples train/val comparison);
      train_uniform/val_uniform  : full (all-term) uniform joint loss (reference;
                                   val == production ``val_loss``, here w/o the
                                   k-weight/data-range the production batch carries);
      val_loss                   : alias of val_uniform;
      train_delta/val_delta      : delta-only uniform term (the degenerate channel);
      train_presid/val_presid    : p_resid-only uniform term (the cosmology signal);
      val_resid_loss, val_coh_loss, grad_norm, lr.
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    n_k = d["P_tier_p"].shape[1]
    epochs = recipe["epochs"]
    batch_size = recipe["batch"]
    lr = recipe["lr"]
    patience = recipe["patience"]
    w_coh = recipe["w_coh"]
    weight_decay = recipe["weight_decay"]
    datarange = recipe["datarange"]

    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)          # minibatch shuffler (matches train_fold)
    norm_stats = T._fit_norm(d, train_idx)

    # SVD warm-start both P_filt bases (identical to train_fold).
    p_filt_basis_init, baseline_basis_init = T._warmstart_bases(
        d, train_idx, norm_stats, recipe["n_basis"], n_k)
    from hcd_analysis.emulator.model import Emulator
    model = Emulator(in_dim=10, n_k=n_k, n_basis=recipe["n_basis"],
                     p_filt_basis_init=p_filt_basis_init,
                     baseline_basis_init=baseline_basis_init, key=key)

    # Pre-fit + FREEZE the theta-blind baseline (the finalized recipe default:
    # prefit_baseline_epochs=8000, freeze on).
    prefit_ep = 8000
    model = T._prefit_baseline(model, d, train_idx, norm_stats,
                               epochs=prefit_ep, lr=2e-3, seed=seed)
    freeze_baseline = True
    stop_on_resid = True   # auto -> resid when baseline frozen (train_fold logic)

    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    total_steps = steps_per_epoch * epochs
    opt = T.make_optimizer(lr=lr, steps=total_steps, weight_decay=weight_decay)
    lr_sched = optax.cosine_decay_schedule(lr, total_steps)

    # WEIGHTED training batch closure (k_weight + data-range) — what AdamW sees.
    def _mb(idx):
        return make_batch(d, idx, norm_stats, k_weight=k_weight, datarange=datarange)

    # UNIFORM eval batch (NO k_weight, NO datarange weight) -> joint_loss is
    # uniform, w_coh=0: the SAME function for train and val. Built ONCE.
    def _uniform_batch(idx):
        return T._to_jnp_batch(make_batch(d, idx, norm_stats))

    # fixed train subsample for the uniform train eval (same rows every epoch).
    n_sub = min(train_eval_subsample, len(train_idx))
    sub = rng_eval.choice(len(train_idx), size=n_sub, replace=False)
    train_eval_idx = train_idx[np.sort(sub)]
    train_eval_batch = _uniform_batch(train_eval_idx)
    val_eval_batch = _uniform_batch(val_idx)

    # WEIGHTED training loss closure (term_w + w_coh) — identical to train_fold.
    loss_fn = T._loss_with_term_w(term_w, w_coh=w_coh)

    # Freeze the baseline via partition (train_fold's frozen path).
    join_mask = T._trainable_mask(model, train_baseline=False, train_rest=True)
    diff_model, static_model = eqx.partition(model, join_mask)
    opt_state = opt.init(eqx.filter(diff_model, eqx.is_array))

    # The early-stop residual / coh val metrics use the SAME k-emphasis +
    # data-range weight the WEIGHTED loss optimizes (train_fold uses the WEIGHTED
    # val batch for these). Build a weighted val batch for the stop metric only.
    val_batch_weighted = T._to_jnp_batch(_mb(val_idx))
    use_kw_es = k_weight is not None
    use_dr_es = datarange

    # PRIMARY same-loss metric: train_nodelta / val_nodelta (uniform joint loss
    # EXCLUDING the degenerate delta channel). Plus references: full (all-term)
    # uniform, delta-only, and p_resid-only (cosmology) — all on the SAME batches.
    history = {"train_obj": [],
               "train_nodelta": [], "val_nodelta": [],     # PRIMARY same-loss pair
               "train_uniform": [], "val_uniform": [],     # full (all-term) reference
               "val_loss": [],                             # alias of full val (production)
               "train_delta": [], "val_delta": [],         # delta-only (the degenerate term)
               "train_presid": [], "val_presid": [],       # p_resid-only (cosmology)
               "val_resid_loss": [], "val_coh_loss": [], "grad_norm": [], "lr": []}
    best_metric = np.inf
    best_epoch = 0
    stall = 0
    step = 0
    for ep in range(epochs):
        ep_losses, ep_gnorms = [], []
        for mb in T._iter_minibatches(rng, train_idx, batch_size):
            batch = T._pad_batch(T._to_jnp_batch(_mb(mb)), batch_size)
            diff_model, opt_state, loss, gnorm = T.train_step_partitioned(
                diff_model, static_model, opt, opt_state, batch, loss_fn)
            model = eqx.combine(diff_model, static_model)
            ep_losses.append(float(loss))
            ep_gnorms.append(float(gnorm))
            step += 1
        ep_lr = float(lr_sched(step))

        # SAME uniform function on train & val, decomposed by term.
        tr_nodelta = _uniform_term(model, train_eval_batch, W_NODELTA)
        va_nodelta = _uniform_term(model, val_eval_batch, W_NODELTA)
        tr_full = float(T.evaluate(model, train_eval_batch))   # all terms (uniform)
        va_full = float(T.evaluate(model, val_eval_batch))
        tr_delta = _uniform_term(model, train_eval_batch, W_DELTA)
        va_delta = _uniform_term(model, val_eval_batch, W_DELTA)
        tr_presid = _uniform_term(model, train_eval_batch, W_PRESID)
        va_presid = _uniform_term(model, val_eval_batch, W_PRESID)
        # WEIGHTED early-stop metrics (match train_fold exactly).
        val_resid = float(p_resid_loss(model, val_batch_weighted,
                                       use_k_weight=use_kw_es, use_datarange=use_dr_es))
        val_coh = (float(coherent_resid_loss(model, val_batch_weighted))
                   if w_coh > 0.0 else np.nan)

        history["train_obj"].append(float(np.mean(ep_losses)))
        history["train_nodelta"].append(tr_nodelta)
        history["val_nodelta"].append(va_nodelta)
        history["train_uniform"].append(tr_full)
        history["val_uniform"].append(va_full)
        history["val_loss"].append(va_full)
        history["train_delta"].append(tr_delta)
        history["val_delta"].append(va_delta)
        history["train_presid"].append(tr_presid)
        history["val_presid"].append(va_presid)
        history["val_resid_loss"].append(val_resid)
        history["val_coh_loss"].append(val_coh)
        history["grad_norm"].append(float(np.mean(ep_gnorms)))
        history["lr"].append(ep_lr)

        # SAME early-stop metric as train_fold: resid + w_coh*coh (frozen+w_coh>0).
        metric = val_resid
        if w_coh > 0.0 and stop_on_resid:
            metric = val_resid + w_coh * val_coh
        if metric < best_metric - 1e-9:
            best_metric = metric
            best_epoch = ep
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    history = {k: np.asarray(v) for k, v in history.items()}
    # 1-based early-stop epoch (restore-best), matching run_loso_sweep convention.
    history_es = int(best_epoch) + 1
    return history, history_es, len(train_eval_idx)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_convergence(results, figpath):
    """Per fold: train & val on the SAME uniform loss (left axis) + the weighted
    training objective (twin axis, context). val-min marked; train/val ratio
    annotated.

    PRIMARY curves (bold solid/dashed) = the uniform joint loss EXCLUDING the
    degenerate delta channel (``train_nodelta`` / ``val_nodelta``) — the
    apples-to-apples train/val comparison (same function on both). These are the
    cosmology-bearing P1D + CDDF terms (p_resid, p_base, f_nhi, dN/dX). FAINT
    dotted blue/orange = the p_resid-only (pure cosmology) term, for context.
    The delta-only term is NOT plotted (its degenerate ~1e4-1e5 standardized
    values on some folds would blow the axis; see the table / script findings).
    The val-min epoch + the annotated ratio use the PRIMARY (no-delta) metric.
    """
    plt = _plt()
    folds = sorted(results)
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.8), squeeze=False)
    axes = axes[0]
    for ai, fold in enumerate(folds):
        h = results[fold]["history"]
        es = results[fold]["val_min_epoch"]          # 1-based no-delta val-min epoch
        ax = axes[ai]
        ep = np.arange(1, len(h["train_nodelta"]) + 1)
        # PRIMARY: no-delta uniform loss (the headline same-loss curves).
        ax.semilogy(ep, h["train_nodelta"], "-", color="C0", lw=1.8,
                    label="train (uniform, no-delta)")
        ax.semilogy(ep, h["val_nodelta"], "--", color="C1", lw=1.8,
                    label="val (uniform, no-delta)")
        # FAINT: the pure cosmology (p_resid-only) term, train vs val, for context.
        ax.semilogy(ep, h["train_presid"], ":", color="C0", lw=1.0, alpha=0.5,
                    label="train (p_resid only)")
        ax.semilogy(ep, h["val_presid"], ":", color="C1", lw=1.0, alpha=0.5,
                    label="val (p_resid only)")

        tr_at = results[fold]["train_nodelta_at_min"]
        va_at = results[fold]["val_nodelta_at_min"]
        ratio = results[fold]["ratio_at_min"]
        ax.axvline(es, color="k", ls="-.", lw=1.0, alpha=0.8,
                   label=f"val-min ep {es}")
        ax.plot([es], [tr_at], "o", color="C0", ms=6)
        ax.plot([es], [va_at], "o", color="C1", ms=6)
        ax.annotate(f"@val-min\ntrain={tr_at:.3g}\nval={va_at:.3g}\nratio={ratio:.2f}",
                    xy=(es, va_at), xytext=(0.52, 0.70), textcoords="axes fraction",
                    fontsize=8, ha="left",
                    bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.85))
        ax.set_title(f"fold {fold}: train/val (uniform, no-delta)\n"
                     f"ratio @val-min = {ratio:.2f}")
        ax.set_xlabel("epoch")
        if ai == 0:
            ax.set_ylabel("uniform joint_loss, delta-excluded (log)")
        ax.grid(alpha=0.3, which="both")

        # robust y-limits from the plotted (well-conditioned) curves.
        both = np.concatenate([h["train_nodelta"], h["val_nodelta"],
                               h["train_presid"], h["val_presid"]])
        both = both[np.isfinite(both) & (both > 0)]
        if both.size:
            lo = min(both.min(), tr_at, va_at) * 0.7
            hi = np.percentile(both, 97) * 2.0
            hi = max(hi, tr_at * 1.3, va_at * 1.3)
            if hi > lo * 1.05:
                ax.set_ylim(lo, hi)

        # twin axis: the WEIGHTED training objective (context only).
        ax2 = ax.twinx()
        ax2.semilogy(ep, h["train_obj"], ":", color="C3", lw=1.2, alpha=0.8,
                     label="train OBJECTIVE (weighted+reg, twin axis)")
        ax2.set_ylabel("weighted train objective (log)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, fontsize=6.5, loc="upper right")
    fig.suptitle("Train vs Val on the SAME (uniform joint) loss — the PI's "
                 "apples-to-apples convergence check\n"
                 "bold = uniform loss EXCLUDING the degenerate delta channel "
                 "(train & val sit together, ratio ~1); dotted red (twin axis) = "
                 "the weighted OBJECTIVE AdamW minimizes (NOT comparable to val)")
    fig.tight_layout()
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    return figpath


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-eval-subsample", type=int, default=2000,
                    help="fixed #train rows for the uniform train eval (same every epoch)")
    ap.add_argument("--figpath", default=str(Path(FIGDIR) / "convergence_same_loss.png"))
    ap.add_argument("--histdir", default="/home/mfho/hcd_priya/checkpoints",
                    help="dir for per-fold per-epoch uniform-loss history JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, few epochs — quick end-to-end check")
    args = ap.parse_args()

    assert jax.config.jax_enable_x64, "x64 must be enabled (structural P1D identities are bit-level)"
    print("jax.devices():", jax.devices(), "| x64:", jax.config.jax_enable_x64)

    recipe = dict(FINAL_RECIPE)
    if args.smoke:
        args.folds = args.folds[:2]
        recipe["epochs"] = min(recipe["epochs"], 8)
        recipe["patience"] = min(recipe["patience"], 6)
        print(f"[SMOKE] folds={args.folds} epochs={recipe['epochs']} "
              f"patience={recipe['patience']}")

    t0 = time.time()
    d = load_cache(args.cache)
    n_k = d["P_tier_p"].shape[1]
    kgrid = d["kfkms"][0]
    print(f"loaded cache: {d['P_tier_p'].shape[0]} rows, n_k={n_k}, "
          f"{len(set(d['sim_name']))} sims ({time.time()-t0:.1f}s)")

    # FINALIZED recipe knobs -> term_w + k_weight (matching run_loso_sweep main()).
    term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0,
              "p_resid": recipe["p_resid_w"], "delta": 1.0}
    k_weight = edge_emphasis_k_weight(kgrid, edge_gain=recipe["edge_gain"],
                                      lowk_extra=recipe["lowk_extra"])
    print(f"recipe: n_basis={recipe['n_basis']} p_resid_w={recipe['p_resid_w']} "
          f"edge_gain={recipe['edge_gain']} lowk_extra={recipe['lowk_extra']} "
          f"w_coh={recipe['w_coh']} wd={recipe['weight_decay']} "
          f"datarange={recipe['datarange']} epochs={recipe['epochs']} "
          f"patience={recipe['patience']}")
    print(f"DATA_RANGE: z in [{DATA_RANGE['z_lo']},{DATA_RANGE['z_hi']}], "
          f"k>={DATA_RANGE['k_min']:.0e}")

    rng_eval = np.random.default_rng(args.seed + 1000)   # fixed train-subsample picker

    results = {}
    for fold in args.folds:
        tr, va, ho = make_splits(d, fold, n_folds=args.n_folds,
                                 holdout_frac=args.holdout_frac)
        print(f"\n=== fold {fold}: train={len(tr)} val={len(va)} "
              f"(holdout {len(ho)} excluded) ===")
        tf = time.time()
        history, es, n_sub = train_fold_logging_uniform(
            d, tr, va, recipe=recipe, term_w=term_w, k_weight=k_weight,
            seed=args.seed, train_eval_subsample=args.train_eval_subsample,
            rng_eval=rng_eval)
        n_ep = len(history["train_nodelta"])

        # The APPLES-TO-APPLES point: val-MIN on the SAME uniform loss. PRIMARY =
        # the uniform joint loss EXCLUDING the degenerate delta channel.
        val_u = history["val_nodelta"]
        train_u = history["train_nodelta"]
        val_min_epoch = int(np.argmin(val_u)) + 1            # 1-based (no-delta val-min)
        i = val_min_epoch - 1
        train_at = float(train_u[i])
        val_at = float(val_u[i])
        ratio = train_at / val_at if val_at > 0 else np.nan
        train_final = float(train_u[-1])
        val_final = float(val_u[-1])
        ratio_final = train_final / val_final if val_final > 0 else np.nan

        # FULL-footprint (all-term) uniform loss + delta-only + p_resid-only at the
        # SAME (no-delta) val-min epoch, for the table / transparency.
        val_uf = history["val_uniform"]; train_uf = history["train_uniform"]
        train_full_at = float(train_uf[i]); val_full_at = float(val_uf[i])
        ratio_full_at = train_full_at / val_full_at if val_full_at > 0 else np.nan
        train_full_final = float(train_uf[-1]); val_full_final = float(val_uf[-1])
        val_delta_at = float(history["val_delta"][i])
        train_delta_at = float(history["train_delta"][i])
        val_presid_at = float(history["val_presid"][i])
        train_presid_at = float(history["train_presid"][i])
        ratio_presid = train_presid_at / val_presid_at if val_presid_at > 0 else np.nan

        results[fold] = dict(
            history=history, early_stop_epoch=es, val_min_epoch=val_min_epoch,
            train_nodelta_at_min=train_at, val_nodelta_at_min=val_at,
            ratio_at_min=ratio, train_nodelta_final=train_final,
            val_nodelta_final=val_final, ratio_final=ratio_final, n_sub=n_sub,
            # references:
            train_full_at_min=train_full_at, val_full_at_min=val_full_at,
            ratio_full_at_min=ratio_full_at, train_full_final=train_full_final,
            val_full_final=val_full_final,
            train_delta_at_min=train_delta_at, val_delta_at_min=val_delta_at,
            train_presid_at_min=train_presid_at, val_presid_at_min=val_presid_at,
            ratio_presid_at_min=ratio_presid)
        # dump per-fold per-epoch uniform-loss history (PI review + debugging).
        import json as _json
        Path(args.histdir).mkdir(parents=True, exist_ok=True)
        hp = Path(args.histdir) / f"diag_same_loss_fold{fold}.hist.json"
        with open(hp, "w") as f:
            _json.dump({k: np.asarray(v).astype(float).tolist()
                        for k, v in history.items()}, f, indent=2)

        print(f"  trained {n_ep} epochs in {time.time()-tf:.1f}s; "
              f"early-stop(resid+coh) ep {es}; no-delta val-min ep {val_min_epoch}")
        print(f"  [SAME LOSS, no-delta] @val-min: train={train_at:.5g} "
              f"val={val_at:.5g}  ratio(train/val)={ratio:.3f}")
        print(f"  [SAME LOSS, no-delta] final:    train={train_final:.5g} "
              f"val={val_final:.5g}  ratio(train/val)={ratio_final:.3f}")
        print(f"  [p_resid only]        @val-min: train={train_presid_at:.5g} "
              f"val={val_presid_at:.5g}  ratio={ratio_presid:.3f}")
        print(f"  [full footprint]      @val-min: train={train_full_at:.5g} "
              f"val={val_full_at:.5g}  ratio={ratio_full_at:.3f}"
              + ("" if val_full_at < 1e3 else
                 f"   (delta-only val term = {val_delta_at:.4g}; degenerate LLS low-k standardization)"))

    figpath = fig_convergence(results, args.figpath)

    # ---- summary tables ------------------------------------------------------
    print("\n" + "=" * 86)
    print("SAME-LOSS (uniform joint_loss) train vs val — the PI's sweet-spot check")
    print("PRIMARY = uniform loss EXCLUDING the degenerate delta channel; val-min on this loss")
    print("=" * 86)
    print(f"{'fold':>4} {'val-min ep':>10} {'train(no-delta)':>16} "
          f"{'val(no-delta)':>14} {'ratio tr/val':>13}")
    for fold in sorted(results):
        r = results[fold]
        print(f"{fold:>4} {r['val_min_epoch']:>10} {r['train_nodelta_at_min']:>16.5g} "
              f"{r['val_nodelta_at_min']:>14.5g} {r['ratio_at_min']:>13.3f}")
    print("\nfinal-state (last logged epoch) — no-delta uniform train vs val:")
    print(f"{'fold':>4} {'#ep':>5} {'train(no-delta)':>16} {'val(no-delta)':>14} "
          f"{'ratio tr/val':>13}")
    for fold in sorted(results):
        r = results[fold]
        print(f"{fold:>4} {len(r['history']['train_nodelta']):>5} "
              f"{r['train_nodelta_final']:>16.5g} {r['val_nodelta_final']:>14.5g} "
              f"{r['ratio_final']:>13.3f}")

    print("\np_resid (pure cosmology) uniform term — train vs val @val-min:")
    print(f"{'fold':>4} {'train(p_resid)':>15} {'val(p_resid)':>14} {'ratio':>10}")
    for fold in sorted(results):
        r = results[fold]
        print(f"{fold:>4} {r['train_presid_at_min']:>15.5g} {r['val_presid_at_min']:>14.5g} "
              f"{r['ratio_presid_at_min']:>10.3f}")

    print("\nfull-footprint uniform loss (ALL terms incl. delta) + delta-only term "
          "@val-min (reference):")
    print(f"{'fold':>4} {'train(full)':>12} {'val(full)':>12} {'val(delta-only)':>16}  note")
    for fold in sorted(results):
        r = results[fold]
        note = ("" if r["val_delta_at_min"] < 1e3
                else "<- delta term degenerate (LLS low-k near-zero-var standardization)")
        print(f"{fold:>4} {r['train_full_at_min']:>12.5g} {r['val_full_at_min']:>12.5g} "
              f"{r['val_delta_at_min']:>16.5g}  {note}")

    # contrast: the OLD plot's weighted-vs-uniform artifact at epoch 0 + val-min.
    print("\ncontext — the OLD figure's weighted train OBJECTIVE vs the uniform (no-delta) val")
    print("(these are DIFFERENT functions; their gap is NOT a generalization gap):")
    print(f"{'fold':>4} {'obj@ep0':>10} {'val_nd@ep0':>11} {'obj@valmin':>11} "
          f"{'val_nd@valmin':>13}")
    for fold in sorted(results):
        h = results[fold]["history"]
        vm = results[fold]["val_min_epoch"] - 1
        print(f"{fold:>4} {h['train_obj'][0]:>10.4g} {h['val_nodelta'][0]:>11.4g} "
              f"{h['train_obj'][vm]:>11.4g} {h['val_nodelta'][vm]:>13.4g}")

    print(f"\nfigure -> {figpath}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
