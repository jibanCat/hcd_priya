#!/usr/bin/env python3
"""Phase-2b emulator training CLI (plan Task 13).

Trains ONE LOSO fold of the JAX/Equinox emulator on a merged v3.3 cache, with the
τ₀-edge holdout excluded from training, an early-stopped AdamW loop, a round-
trippable checkpoint, and review figures (train/val loss, grad-norm+LR, predicted-
vs-cache per-class P1D). ``--profile`` runs a single short fold and prints device +
timing so the budget can be sized before the full k-fold sweep (cavestru0 is tight).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/train_emulator.py [args]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import (
    load_cache, make_splits, fit_target_norm,
    untransform_prediction, COARSE_NAMES,
)
from hcd_analysis.emulator import train as T

DEFAULT_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def _peak_mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _fig_loss(history, fold, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ep, history["train_loss"], "-o", ms=3, label="train")
    ax.semilogy(ep, history["val_loss"], "-s", ms=3, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("joint loss (log)")
    ax.set_title(f"Fold {fold}: train/val loss"); ax.legend(); ax.grid(alpha=0.3)
    p = Path(figdir) / f"train_val_loss_fold{fold}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return str(p)


def _fig_grad_lr(history, fold, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(history["grad_norm"]) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ep, history["grad_norm"], "-o", ms=3, color="C0", label="grad norm")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("global grad L2", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ep, history["lr"], "-s", ms=3, color="C1", label="lr")
    ax2.set_ylabel("learning rate", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title(f"Fold {fold}: grad-norm + LR")
    p = Path(figdir) / f"grad_norm_lr_fold{fold}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return str(p)


def _fig_pred_vs_true(model, d, val_idx, norm_stats, fold, figdir, n_rows=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = np.asarray(val_idx)[:n_rows]
    x = jnp.asarray(d["x"][rows]); tau0 = jnp.asarray(d["tau0"][rows])
    pred = jax.vmap(model)(x, tau0)
    # REDESIGN: reconstruct linear P_filt from the θ-blind baseline + the residual.
    P_pred = untransform_prediction(
        {"P_filt_base": np.asarray(pred["P_filt_base"]),
         "P_filt_resid": np.asarray(pred["P_filt_resid"])}, norm_stats)["P_filt"]
    P_true = d["P_filt"][rows]   # (n,4,K) linear cache
    kf = d["kfkms"][rows]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for ci, cname in enumerate(COARSE_NAMES):
        ax = axes[0, ci]; axr = axes[1, ci]
        for ri in range(len(rows)):
            k = kf[ri]; pt = P_true[ri, ci]; pp = P_pred[ri, ci]
            ok = np.isfinite(pt) & (pt > 0) & np.isfinite(pp) & (pp > 0)
            ax.loglog(k[ok], pt[ok], "-", color=f"C{ri}", alpha=0.8,
                      label=f"row{rows[ri]} cache" if ci == 0 else None)
            ax.loglog(k[ok], pp[ok], "--", color=f"C{ri}", alpha=0.8,
                      label=f"row{rows[ri]} pred" if ci == 0 else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                resid = pp[ok] / pt[ok] - 1.0
            axr.semilogx(k[ok], resid, "-", color=f"C{ri}", alpha=0.8)
        ax.set_title(cname); ax.grid(alpha=0.3)
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_xlabel("k [s/km]")
        if ci == 0:
            ax.set_ylabel("P1D (filtered)"); axr.set_ylabel("pred/cache - 1")
            ax.legend(fontsize=7)
    fig.suptitle(f"Fold {fold}: predicted vs cache per-class P1D (held-out rows)")
    p = Path(figdir) / f"pred_vs_true_p1d_fold{fold}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--n-basis", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints/emu_fold0")
    ap.add_argument("--figdir", default="figures/analysis/04_emulator")
    ap.add_argument("--profile", action="store_true",
                    help="single short fold; print device + timing")
    ap.add_argument("--staged", action="store_true",
                    help="3-stage training (baseline -> freeze+residual -> joint); "
                         "early-stop on val residual (cosmology) loss")
    args = ap.parse_args()

    print("jax.devices():", jax.devices())
    if args.profile:
        # keep it cheap: cap epochs so the timing is representative but quick.
        args.epochs = min(args.epochs, 30)
        print(f"[PROFILE] fold={args.fold} epochs={args.epochs} "
              f"batch={args.batch} n_basis={args.n_basis}")

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d = load_cache(args.cache)
    n_k = d["P_tier_p"].shape[1]
    print(f"loaded cache {args.cache}: {d['P_tier_p'].shape[0]} rows, n_k={n_k} "
          f"({time.time()-t0:.1f}s)")

    # Exclude the tau0-edge holdout from EVERYTHING (extrapolation probe stays unseen).
    # make_splits owns the tau0-holdout x LOSO composition (see data.make_splits).
    try:
        tr, va, holdout = make_splits(
            d, args.fold, n_folds=args.n_folds, holdout_frac=args.holdout_frac)
    except IndexError as e:
        raise SystemExit(f"--fold {e}")
    print(f"fold {args.fold}/{args.n_folds}: train={len(tr)} val={len(va)} "
          f"(holdout {len(holdout)} excluded)")

    # epoch timing via a lightweight per-epoch wrapper: train_fold owns the loop, so
    # time the whole fold then report mean per-epoch.
    t_train = time.time()
    model, norm_stats, history = T.train_fold(
        d, tr, va, n_basis=args.n_basis, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch, seed=args.seed, key=jax.random.PRNGKey(args.seed),
        patience=args.patience, n_k=n_k, staged=args.staged,
    )
    train_wall = time.time() - t_train
    n_ep = len(history["train_loss"])
    print(f"trained {n_ep} epochs in {train_wall:.1f}s "
          f"({train_wall/max(n_ep,1):.2f}s/epoch)")
    print(f"final train_loss={history['train_loss'][-1]:.6g} "
          f"val_loss={history['val_loss'][-1]:.6g} "
          f"best_val_loss={float(np.min(history['val_loss'])):.6g}")
    print(f"peak RSS: {_peak_mem_mb():.0f} MB")

    arch_cfg = {"in_dim": 10, "n_k": n_k, "n_basis": args.n_basis}
    T.save_checkpoint(args.out, model, arch_cfg, norm_stats, seed=args.seed,
                      kfkms=d["kfkms"], cache_path=args.cache)
    print(f"checkpoint -> {args.out}.eqx / .meta.json / .norm.pkl")

    # figures + history
    fp_loss = _fig_loss(history, args.fold, args.figdir)
    fp_grad = _fig_grad_lr(history, args.fold, args.figdir)
    fp_pred = _fig_pred_vs_true(model, d, va, norm_stats, args.fold, args.figdir)
    hist_path = Path(args.figdir) / f"train_history_fold{args.fold}.npz"
    np.savez(hist_path, **history)
    print("figures:")
    for p in (fp_loss, fp_grad, fp_pred, str(hist_path)):
        print("  ", p)

    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
