#!/usr/bin/env python3
"""Phase-2b Task 14: k-fold LOSO sweep driver + emulator error-vector producer.

Runs the full LOSO cross-validation sweep over the merged v3.3 cache, training one
emulator per fold (REUSING ``hcd_analysis.emulator.train.train_fold``), collecting
per-fold fractional P1D residuals stratified into (class, k, z-band) cells, then
aggregates them into the σ error vector + DLA high-k shot-noise flag that feeds the
likelihood covariance (``aggregate_error_vector``). Emits ``error_vector.npz`` and
three review figures.

Does NOT modify any ``hcd_analysis/emulator/*.py`` (other work reads them).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_loso_sweep.py [args]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import (
    load_cache, make_splits, untransform_prediction, COARSE_NAMES,
)
from hcd_analysis.emulator import train as T

DEFAULT_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


# --- z-band scheme ------------------------------------------------------------

def make_z_bands(z_grid, n_bands):
    """Partition rows into ``n_bands`` z-bands by quantile edges of ``z_grid``.

    Returns ``(z_band_of_row (R,) int in [0,n_bands), edges (n_bands+1,))``. Quantile
    edges adapt to the (discrete, 18-value) z sampling so each band holds a roughly
    equal share of rows; the outer edges are nudged to ±inf so every row lands in a
    band even at the extremes."""
    z = np.asarray(z_grid, float)
    qs = np.linspace(0.0, 1.0, n_bands + 1)
    edges = np.quantile(z, qs)
    edges = np.unique(edges)                      # collapse ties (discrete z grid)
    if len(edges) < n_bands + 1:
        # ties collapsed bands; fall back to linear edges over the z range.
        edges = np.linspace(z.min(), z.max(), n_bands + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    band = np.digitize(z, edges[1:-1], right=False)   # 0..n_bands-1
    band = np.clip(band, 0, n_bands - 1)
    return band.astype(int), edges


# --- per-fold residual / neff stratification ----------------------------------

def fold_resid_neff(d, model, val_idx, norm_stats, z_band_of_row, n_bands):
    """Per-fold RMS fractional residual + effective sightline count, (4,K,Zb).

    Residual definition (per val-row r, class c, k-bin k):
        rfrac[r,c,k] = (P_filt_pred[r,c,k] - P_filt_true[r,c,k]) / P_filt_true[r,c,k]
    in LINEAR per-class P1D space (predictions are untransformed from the emulator's
    standardized-log space via ``untransform_prediction``). A residual is kept only
    where BOTH pred & true are finite, the row's k-mask is set (drops NaN/Nyquist
    bins), and the true value is non-zero (drops structural/empty-class zeros).

    Per (class,k,z-band) cell we reduce the kept residuals to their RMS over the val
    rows in that z-band — ``aggregate_error_vector`` then RMS-combines these per-fold
    per-cell values over folds. Empty cells (no kept rows) are NaN (nan-safe later).

    neff definition: effective sightline count per (class,k,z-band) =
        Σ_{r in z-band} coarse_counts[r, class]
    broadcast over k (k-independent — counts are per-row, not per-k). DLA high-k cells
    that are shot-noise-limited surface via the small DLA coarse_counts.
    """
    val_idx = np.asarray(val_idx)
    x = jnp.asarray(d["x"][val_idx])
    tau0 = jnp.asarray(d["tau0"][val_idx])
    pred = jax.vmap(model)(x, tau0)
    # REDESIGN: reconstruct linear P_filt from the θ-blind baseline + the residual.
    P_pred = untransform_prediction(
        {"P_filt_base": np.asarray(pred["P_filt_base"]),
         "P_filt_resid": np.asarray(pred["P_filt_resid"])},
        norm_stats)["P_filt"]                                          # (Nval,4,K) linear
    P_true = d["P_filt"][val_idx]                                        # (Nval,4,K) linear
    mask_k = d["mask"][val_idx]                                          # (Nval,K) finite Tier-P
    coarse = d["coarse_counts"][val_idx].astype(float)                  # (Nval,4)
    zband = z_band_of_row[val_idx]                                      # (Nval,)

    Nval, n_cls, K = P_true.shape
    keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true != 0.0)
            & mask_k[:, None, :])                                        # (Nval,4,K)
    with np.errstate(invalid="ignore", divide="ignore"):
        rfrac = (P_pred - P_true) / P_true
    rfrac = np.where(keep, rfrac, np.nan)

    sigma = np.full((n_cls, K, n_bands), np.nan)
    neff = np.zeros((n_cls, K, n_bands))
    for zb in range(n_bands):
        sel = (zband == zb)
        if not sel.any():
            continue
        rb = rfrac[sel]                                                  # (nb,4,K)
        with np.errstate(invalid="ignore"):
            sigma[:, :, zb] = np.sqrt(np.nanmean(rb ** 2, axis=0))       # (4,K) RMS over rows
        neff[:, :, zb] = coarse[sel].sum(axis=0)[:, None]               # (4,1)->(4,K)
    return sigma, neff, P_pred, P_true, mask_k


# --- figures ------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_loss_curves(histories, figdir):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for fold, h in sorted(histories.items()):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        c = f"C{fold % 10}"
        ax.semilogy(ep, h["train_loss"], "-", color=c, alpha=0.9,
                    label=f"fold {fold} train")
        ax.semilogy(ep, h["val_loss"], "--", color=c, alpha=0.9,
                    label=f"fold {fold} val")
    ax.set_xlabel("epoch"); ax.set_ylabel("joint loss (log)")
    ax.set_title("LOSO sweep: train/val loss per fold")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    p = Path(figdir) / "loso_loss_curves.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return str(p)


def fig_error_vs_k(sigma, kfkms, z_band_edges, figdir):
    """σ error vector vs k, one line per class, one panel per z-band, log-log."""
    plt = _plt()
    n_cls, K, Zb = sigma.shape
    k = np.asarray(kfkms)
    fig, axes = plt.subplots(1, Zb, figsize=(4.2 * Zb, 4.2), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for zb in range(Zb):
        ax = axes[zb]
        lo, hi = z_band_edges[zb], z_band_edges[zb + 1]
        for ci, cname in enumerate(COARSE_NAMES):
            s = sigma[ci, :, zb]
            ok = np.isfinite(s) & (s > 0)
            if ok.any():
                ax.loglog(k[ok], s[ok], "-o", ms=2.5, color=f"C{ci}", label=cname)
        ax.set_xlabel("k [s/km]")
        ax.set_title(f"z-band {zb}  [{lo:.2f}, {hi:.2f})")
        ax.grid(alpha=0.3, which="both")
        if zb == 0:
            ax.set_ylabel(r"$\sigma$ (RMS frac. P1D resid)")
            ax.legend(fontsize=8)
    fig.suptitle("LOSO emulator error vector vs k (per class, per z-band)")
    p = Path(figdir) / "loso_error_vs_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return str(p)


def fig_dla_shotflag(dla_shot_flag, kfkms, figdir):
    plt = _plt()
    k = np.asarray(kfkms)
    flag = np.asarray(dla_shot_flag).astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(np.arange(len(k)), flag, width=1.0, color="C3", alpha=0.85)
    ax.set_xlabel("k-bin index")
    ax.set_ylabel("DLA shot-limited")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["ok", "flagged"])
    n_flag = int(flag.sum())
    ax.set_title(f"DLA high-k shot-noise flag ({n_flag}/{len(k)} k-bins flagged)")
    # secondary axis: k values at a few ticks
    idx = np.linspace(0, len(k) - 1, min(8, len(k))).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{k[i]:.2g}" for i in idx], rotation=45, fontsize=7)
    ax.set_xlabel("k [s/km] (at ticks)")
    p = Path(figdir) / "loso_dla_shotflag.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
    return str(p)


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--n-basis", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--z-bands", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints/loso")
    ap.add_argument("--figdir", default="figures/analysis/04_emulator")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, few epochs — quick end-to-end check")
    args = ap.parse_args()

    if args.smoke:
        args.n_folds = 2
        args.epochs = min(args.epochs, 8)
        print(f"[SMOKE] n_folds={args.n_folds} epochs={args.epochs} "
              f"batch={args.batch} n_basis={args.n_basis} z_bands={args.z_bands}")

    print("jax.devices():", jax.devices())
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d = load_cache(args.cache)
    n_k = d["P_tier_p"].shape[1]
    kgrid = d["kfkms"][0]                                  # shared k-grid (R,n_k)
    print(f"loaded cache {args.cache}: {d['P_tier_p'].shape[0]} rows, n_k={n_k}, "
          f"{len(set(d['sim_name']))} sims ({time.time()-t0:.1f}s)")

    z_band_of_row, z_band_edges = make_z_bands(d["z_grid"], args.z_bands)
    print(f"z-bands ({args.z_bands}) edges (quantile of z_grid): "
          f"{np.array2string(z_band_edges, precision=3)}")
    for zb in range(args.z_bands):
        sel = z_band_of_row == zb
        zz = d["z_grid"][sel]
        print(f"  band {zb}: {sel.sum()} rows, z in "
              f"[{zz.min():.2f}, {zz.max():.2f}]")

    resid_folds, neff_folds, histories = [], [], {}
    per_fold_summary = []
    for fold in range(args.n_folds):
        tr, va, holdout = make_splits(
            d, fold, n_folds=args.n_folds, holdout_frac=args.holdout_frac)
        print(f"\n=== fold {fold}/{args.n_folds}: train={len(tr)} val={len(va)} "
              f"(holdout {len(holdout)} excluded) ===")
        tf = time.time()
        model, norm_stats, history = T.train_fold(
            d, tr, va, n_basis=args.n_basis, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch, seed=args.seed, key=jax.random.PRNGKey(args.seed),
            patience=args.patience, n_k=n_k,
        )
        n_ep = len(history["train_loss"])
        print(f"  trained {n_ep} epochs in {time.time()-tf:.1f}s; "
              f"best_val_loss={float(np.min(history['val_loss'])):.6g}")

        arch_cfg = {"in_dim": 10, "n_k": n_k, "n_basis": args.n_basis}
        ckpt = f"{args.out}_fold{fold}"
        T.save_checkpoint(ckpt, model, arch_cfg, norm_stats, seed=args.seed,
                          kfkms=d["kfkms"], cache_path=args.cache)
        print(f"  checkpoint -> {ckpt}.eqx / .meta.json / .norm.pkl")

        sigma, neff, P_pred, P_true, mask_k = fold_resid_neff(
            d, model, va, norm_stats, z_band_of_row, args.z_bands)
        resid_folds.append(sigma)
        neff_folds.append(neff)
        histories[fold] = history

        # per-fold val RMS (overall + per class), over kept bins only.
        keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true != 0.0)
                & mask_k[:, None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            rfrac = (P_pred - P_true) / P_true
        rfrac = np.where(keep, rfrac, np.nan)
        overall = float(np.sqrt(np.nanmean(rfrac ** 2)))
        per_cls = [float(np.sqrt(np.nanmean(rfrac[:, ci] ** 2)))
                   for ci in range(len(COARSE_NAMES))]
        per_fold_summary.append((fold, overall, per_cls))
        print(f"  val RMS overall={overall:.4g}  " +
              "  ".join(f"{c}={v:.4g}" for c, v in zip(COARSE_NAMES, per_cls)))

    # --- aggregate error vector + DLA shot flag -------------------------------
    ev = T.aggregate_error_vector(resid_folds, neff_folds)
    sigma = ev["sigma"]                       # (4,K,Zb)
    dla_shot_flag = ev["dla_shot_flag"]       # (K,)

    out_npz = Path(args.out).parent / "error_vector.npz"
    np.savez(
        out_npz,
        sigma=sigma,
        dla_shot_flag=dla_shot_flag,
        z_band_edges=z_band_edges,
        class_names=np.array(COARSE_NAMES),
        kfkms=kgrid,
    )
    print(f"\nerror vector -> {out_npz}")

    # --- figures --------------------------------------------------------------
    fp_loss = fig_loss_curves(histories, args.figdir)
    fp_evk = fig_error_vs_k(sigma, kgrid, z_band_edges, args.figdir)
    fp_flag = fig_dla_shotflag(dla_shot_flag, kgrid, args.figdir)
    print("figures:")
    for p in (fp_loss, fp_evk, fp_flag):
        print("  ", p)

    # --- summary --------------------------------------------------------------
    total_wall = time.time() - t0
    print("\n========== SUMMARY ==========")
    print(f"folds={args.n_folds} z_bands={args.z_bands} n_basis={args.n_basis} "
          f"epochs={args.epochs}")
    print("per-fold val RMS (fractional P1D residual):")
    for fold, overall, per_cls in per_fold_summary:
        print(f"  fold {fold}: overall={overall:.4g}  " +
              "  ".join(f"{c}={v:.4g}" for c, v in zip(COARSE_NAMES, per_cls)))
    print("error-vector median σ per class (over k & z-bands):")
    for ci, cname in enumerate(COARSE_NAMES):
        med = float(np.nanmedian(sigma[ci]))
        print(f"  {cname}: median σ = {med:.4g}")
    n_flag = int(np.asarray(dla_shot_flag).sum())
    print(f"DLA shot-flagged k-bins: {n_flag}/{len(kgrid)}")
    print(f"total wall: {total_wall:.1f}s "
          f"({total_wall/max(args.n_folds,1):.1f}s/fold)")


if __name__ == "__main__":
    main()
