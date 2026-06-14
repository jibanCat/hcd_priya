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
    datarange_mask, DATA_RANGE, make_tau0_bands,
)
from hcd_analysis.emulator import train as T

DEFAULT_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"

# FINALIZED LF emulator recipe (the two productionized wins + the residual tuning
# from commit 73009c5): deep frozen θ-blind baseline, term_w[p_resid]=8, edge
# k-weight (gain3/lowk2), FLAT coherent de-bias w_coh=80, data-range soft
# down-weight, wd3e-4, n_basis24, early-stop on resid+coh.
FINAL_RECIPE = dict(
    n_basis=24, p_resid_w=8.0, edge_gain=3.0, lowk_extra=2.0,
    w_coh=80.0, weight_decay=3e-4, datarange=True,
    epochs=180, patience=25, lr=1e-3, batch=512,
)


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

def fold_resid_neff(d, model, val_idx, norm_stats, z_band_of_row, n_bands,
                    tau0_band_of_row=None, n_tb=1,
                    datarange=True, z_lo=None, z_hi=None, k_min=None):
    """Per-fold RMS fractional residual + effective sightline count, (4,K,Zb,Tb).

    Residual definition (per val-row r, class c, k-bin k):
        rfrac[r,c,k] = (P_filt_pred[r,c,k] - P_filt_true[r,c,k]) / P_filt_true[r,c,k]
    in LINEAR per-class P1D space (predictions are untransformed from the emulator's
    standardized-log space via ``untransform_prediction``). A residual is kept only
    where BOTH pred & true are finite, the row's k-mask is set (drops NaN/Nyquist
    bins), and the true value is non-zero (drops structural/empty-class zeros).

    DATA-RANGE SCOPING (``datarange=True``, default): the kept residuals are ALSO
    restricted to the DESI data range (z∈[z_lo,z_hi], k≥k_min; defaults from
    ``DATA_RANGE``) via ``datarange_mask`` — so the emulator-error budget (the σ error
    vector / C_emu) only covers modes the data constrain. Out-of-range (z,k) bins are
    dropped (their residual is set NaN and excluded from the RMS).

    Per (class,k,z-band,τ₀-band) cell we reduce the kept residuals to their RMS over the
    val rows in that cell — ``aggregate_error_vector`` then RMS-combines these per-fold
    per-cell values over folds. The τ₀-band axis (``tau0_band_of_row``, ``n_tb``; the
    outer bands isolate the ladder extremes, ``data.make_tau0_bands``) makes the error
    vector τ₀-AWARE for the C_emu consumer. ``tau0_band_of_row=None``/``n_tb=1`` reduces
    to the old z-only behaviour (trailing singleton). Empty cells are NaN (nan-safe later).

    neff definition: effective sightline count per (class,k,z-band) =
        Σ_{r in z-band} coarse_counts[r, class]
    broadcast over k (k-independent — counts are per-row, not per-k). DLA high-k cells
    that are shot-noise-limited surface via the small DLA coarse_counts.

    CAVEAT (review I1): the catalog is tau0-invariant, so all n_alpha tau0 rows of a
    given snap carry IDENTICAL coarse_counts. neff is therefore inflated by the alpha
    multiplicity (~20x) AND summed across snaps — read it as a RELATIVE shot-noise
    proxy, not a literal independent-sightline count. With the full grid the per-row
    DLA count (~8000+) sits far above any reasonable shot threshold, so the DLA shot
    flag is expected-INERT -- which is the DESIRED state: the DLA TEMPLATE is calibrated
    from the simulation (many DLA sightlines), while the observed P1D is measured on
    DLA-MASKED spectra, so the emulator's DLA sector fits only the masking-residual
    amplitude (alpha_DLA), not raw DLA absorption. The flag guards sim-side template
    shot noise, not a data-side concern. Divide by n_alpha (and use a per-sightline
    floor) before reading neff as an absolute count.
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
    tband = (np.asarray(tau0_band_of_row)[val_idx] if tau0_band_of_row is not None
             else np.zeros(len(val_idx), int))                         # (Nval,) τ₀-band

    Nval, n_cls, K = P_true.shape
    keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true != 0.0)
            & mask_k[:, None, :])                                        # (Nval,4,K)
    if datarange:
        # restrict the error budget to the DESI data range (z∈[z_lo,z_hi], k≥k_min).
        in_range = datarange_mask(d, z_lo=z_lo, z_hi=z_hi, k_min=k_min)[val_idx]  # (Nval,K)
        keep = keep & in_range[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rfrac = (P_pred - P_true) / P_true
    rfrac = np.where(keep, rfrac, np.nan)

    sigma = np.full((n_cls, K, n_bands, n_tb), np.nan)
    neff = np.zeros((n_cls, K, n_bands, n_tb))
    for zb in range(n_bands):
        for tb in range(n_tb):
            sel = (zband == zb) & (tband == tb)
            if not sel.any():
                continue
            rb = rfrac[sel]                                              # (nb,4,K)
            with np.errstate(invalid="ignore"):
                sigma[:, :, zb, tb] = np.sqrt(np.nanmean(rb ** 2, axis=0))  # (4,K)
            neff[:, :, zb, tb] = coarse[sel].sum(axis=0)[:, None]       # (4,1)->(4,K)
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


def _earlystop_epoch(h):
    """The early-stop (restore-best) epoch the train loop selected for this fold.

    Mirrors train_fold's stop metric: when the coherent de-bias is active
    (val_coh_loss finite) the stop is on (val_resid + w_coh·val_coh); the proxy here
    is argmin of (val_resid + val_coh) when coh is present, else argmin(val_resid)
    when present, else argmin(val_loss). Returns a 1-based epoch."""
    vr = np.asarray(h.get("val_resid_loss", []), float)
    vc = np.asarray(h.get("val_coh_loss", []), float)
    vl = np.asarray(h.get("val_loss", []), float)
    if vr.size and np.isfinite(vr).any():
        if vc.size and np.isfinite(vc).all():
            metric = vr + vc
        else:
            metric = vr
    else:
        metric = vl
    return int(np.nanargmin(metric)) + 1


def fig_perfold_convergence(histories, figdir,
                            fname="final_perfold_convergence.png"):
    """Per-fold train/val convergence: train & val loss vs epoch for all 8 folds,
    each fold its own panel, with the early-stop (restore-best) epoch marked.

    The PI's explicit ask: 'how converged are the folds?' Shows train (solid) vs val
    (dashed) joint loss per fold + a vertical line at the restored-best epoch and the
    val-residual curve (the inference-relevant stop metric)."""
    plt = _plt()
    folds = sorted(histories)
    n = len(folds)
    ncol = 4 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ai, fold in enumerate(folds):
        h = histories[fold]
        ax = axes[ai]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.semilogy(ep, h["train_loss"], "-", color="C0", lw=1.4, label="train (joint)")
        ax.semilogy(ep, h["val_loss"], "--", color="C1", lw=1.4, label="val (joint)")
        if "val_resid_loss" in h and np.isfinite(h["val_resid_loss"]).any():
            ax.semilogy(ep, h["val_resid_loss"], ":", color="C3", lw=1.2,
                        label="val resid")
        es = _earlystop_epoch(h)
        ax.axvline(es, color="k", ls="-.", lw=1.0, alpha=0.8,
                   label=f"early-stop ep {es}")
        ax.set_title(f"fold {fold} ({len(ep)} ep run)")
        ax.grid(alpha=0.3, which="both")
        if ai == 0:
            ax.legend(fontsize=7)
        if ai % ncol == 0:
            ax.set_ylabel("loss (log)")
        if ai >= (nrow - 1) * ncol:
            ax.set_xlabel("epoch")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-fold train/val convergence (finalized recipe): "
                 "de-bias w_coh=80 + data-range down-weight")
    p = Path(figdir) / fname
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
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
    ap.add_argument("--n-basis", type=int, default=FINAL_RECIPE["n_basis"])
    ap.add_argument("--epochs", type=int, default=FINAL_RECIPE["epochs"])
    ap.add_argument("--lr", type=float, default=FINAL_RECIPE["lr"])
    ap.add_argument("--batch", type=int, default=FINAL_RECIPE["batch"])
    ap.add_argument("--z-bands", type=int, default=3)
    ap.add_argument("--tau0-bands", type=int, default=4,
                    help="τ₀-ladder bands for the error vector (outer bands isolate "
                         "the ladder extremes); 1 = τ₀-flat (old behaviour)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=FINAL_RECIPE["patience"])
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints/loso")
    ap.add_argument("--figdir", default="figures/analysis/04_emulator")
    ap.add_argument("--histdir", default="checkpoints",
                    help="dir for per-fold per-epoch history JSON (final_fold{f}.hist.json)")
    # FINALIZED RECIPE knobs (defaults = the two productionized wins; see FINAL_RECIPE)
    ap.add_argument("--p-resid-w", type=float, default=FINAL_RECIPE["p_resid_w"])
    ap.add_argument("--edge-gain", type=float, default=FINAL_RECIPE["edge_gain"])
    ap.add_argument("--lowk-extra", type=float, default=FINAL_RECIPE["lowk_extra"])
    ap.add_argument("--w-coh", type=float, default=FINAL_RECIPE["w_coh"],
                    help="FLAT coherent de-bias weight (the flat_w80 winner)")
    ap.add_argument("--weight-decay", type=float, default=FINAL_RECIPE["weight_decay"])
    ap.add_argument("--no-datarange", action="store_true",
                    help="disable the data-range soft down-weight + error-vector restriction")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, few epochs — quick end-to-end check")
    args = ap.parse_args()

    if args.smoke:
        args.n_folds = 2
        args.epochs = min(args.epochs, 8)
        args.w_coh = min(args.w_coh, 80.0)
        print(f"[SMOKE] n_folds={args.n_folds} epochs={args.epochs} "
              f"batch={args.batch} n_basis={args.n_basis} z_bands={args.z_bands}")

    datarange = not args.no_datarange
    term_w = None
    if args.p_resid_w != 1.0:
        term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0,
                  "p_resid": args.p_resid_w, "delta": 1.0}

    print("jax.devices():", jax.devices())
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d = load_cache(args.cache)
    n_k = d["P_tier_p"].shape[1]
    # Representative k-grid for labelling the error vector + figures. kfkms is NOT
    # identical across rows (per-snap vmax differs; max dev ~3% LF / ~9% HR at high
    # k). Training/residuals use each row's OWN grid per-index; only this saved label
    # is row-0's grid, so downstream likelihood must treat error_vector["kfkms"] as a
    # representative grid, not exact per-mode k. (Review I2.)
    kgrid = d["kfkms"][0]                                  # representative k-grid (n_k,)
    print(f"loaded cache {args.cache}: {d['P_tier_p'].shape[0]} rows, n_k={n_k}, "
          f"{len(set(d['sim_name']))} sims ({time.time()-t0:.1f}s)")

    # FINALIZED recipe: edge-emphasis k-weight on the per-sim residual MSE + the FLAT
    # coherent de-bias (w_coh) + data-range soft down-weight. Report the config.
    from hcd_analysis.emulator.data import edge_emphasis_k_weight
    k_weight = None
    if args.edge_gain != 0.0 or args.lowk_extra != 0.0:
        k_weight = edge_emphasis_k_weight(kgrid, edge_gain=args.edge_gain,
                                          lowk_extra=args.lowk_extra)
    print(f"recipe: n_basis={args.n_basis} p_resid_w={args.p_resid_w} "
          f"edge_gain={args.edge_gain} lowk_extra={args.lowk_extra} "
          f"w_coh={args.w_coh} wd={args.weight_decay} datarange={datarange} "
          f"(z∈[{DATA_RANGE['z_lo']},{DATA_RANGE['z_hi']}], k≥{DATA_RANGE['k_min']:.0e})")
    Path(args.histdir).mkdir(parents=True, exist_ok=True)

    z_band_of_row, z_band_edges = make_z_bands(d["z_grid"], args.z_bands)
    print(f"z-bands ({args.z_bands}) edges (quantile of z_grid): "
          f"{np.array2string(z_band_edges, precision=3)}")
    for zb in range(args.z_bands):
        sel = z_band_of_row == zb
        zz = d["z_grid"][sel]
        print(f"  band {zb}: {sel.sum()} rows, z in "
              f"[{zz.min():.2f}, {zz.max():.2f}]")

    # τ₀-LADDER bands (Phase-C T2): the error vector becomes τ₀-aware. Outer bands
    # isolate the ladder extremes (data.make_tau0_bands); centres are in α=τ₀/Kim(z).
    tau0_band_of_row, tau0_band_centres = make_tau0_bands(
        d["tau0"], d["z_grid"], args.tau0_bands)
    print(f"τ₀-bands ({args.tau0_bands}) α-centres: "
          f"{np.array2string(tau0_band_centres, precision=3)}")
    for tb in range(args.tau0_bands):
        sel = tau0_band_of_row == tb
        print(f"  τ₀-band {tb}: {sel.sum()} rows")

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
            term_w=term_w, k_weight=k_weight, w_coh=args.w_coh,
            datarange=datarange, weight_decay=args.weight_decay,
        )
        n_ep = len(history["train_loss"])
        es = _earlystop_epoch(history)
        print(f"  trained {n_ep} epochs in {time.time()-tf:.1f}s; "
              f"early-stop ep {es}; best_val_loss={float(np.min(history['val_loss'])):.6g}")

        # FINAL_RECIPE training knobs (frozen with the checkpoint AND the hist.json so
        # the LF backbone the MF/likelihood load is fully reproducible — w_coh, term_w,
        # edge_gain/lowk, datarange, …).
        recipe = dict(
            n_basis=args.n_basis, p_resid_w=args.p_resid_w, edge_gain=args.edge_gain,
            lowk_extra=args.lowk_extra, w_coh=args.w_coh, weight_decay=args.weight_decay,
            datarange=datarange, patience=args.patience, epochs=args.epochs,
            term_w=term_w)

        # arch_cfg here is the MINIMAL caller view; save_checkpoint COMPLETES it from
        # the model's static fields (adds baseline_n_layers/baseline_width).
        arch_cfg = {"in_dim": 10, "n_k": n_k, "n_basis": args.n_basis}
        ckpt = f"{args.out}_fold{fold}"
        T.save_checkpoint(ckpt, model, arch_cfg, norm_stats, seed=args.seed,
                          kfkms=d["kfkms"], cache_path=args.cache, recipe=recipe)
        print(f"  checkpoint -> {ckpt}.eqx / .meta.json / .norm.pkl")

        # SAVE per-fold per-epoch history (+ per-term metrics) to disk for the
        # convergence figure / PI review (checkpoints/final_fold{f}.hist.json).
        import json as _json
        hist_path = Path(args.histdir) / f"final_fold{fold}.hist.json"
        hist_json = {k: np.asarray(v).astype(float).tolist() for k, v in history.items()}
        hist_json["early_stop_epoch"] = es
        hist_json["recipe"] = recipe
        with open(hist_path, "w") as f:
            _json.dump(hist_json, f, indent=2)
        print(f"  history -> {hist_path}")

        sigma, neff, P_pred, P_true, mask_k = fold_resid_neff(
            d, model, va, norm_stats, z_band_of_row, args.z_bands,
            tau0_band_of_row=tau0_band_of_row, n_tb=args.tau0_bands, datarange=datarange)
        resid_folds.append(sigma)
        neff_folds.append(neff)
        histories[fold] = history

        # per-fold val RMS (overall + per class), over kept bins only — restricted
        # to the data range when datarange (matches the error-vector scoping).
        keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true != 0.0)
                & mask_k[:, None, :])
        if datarange:
            in_range = datarange_mask(d)[va]                    # (Nval,K)
            keep = keep & in_range[:, None, :]
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
    sigma = ev["sigma"]                       # (4,K,Zb,Tb) τ₀-aware
    dla_shot_flag = ev["dla_shot_flag"]       # (K,)
    sigma_zonly = np.sqrt(np.nanmean(sigma ** 2, axis=3))   # (4,K,Zb) τ₀-marginalized

    out_npz = Path(args.out).parent / "error_vector.npz"
    np.savez(
        out_npz,
        sigma=sigma,                          # (4,K,Zb,Tb) — the τ₀-aware vector
        dla_shot_flag=dla_shot_flag,
        z_band_edges=z_band_edges,
        tau0_band_centres=tau0_band_centres,  # (Tb,) α=τ₀/Kim(z) band centres
        class_names=np.array(COARSE_NAMES),
        kfkms=kgrid,
    )
    print(f"\nerror vector -> {out_npz}  (sigma shape {sigma.shape})")

    # --- figures --------------------------------------------------------------
    fp_loss = fig_loss_curves(histories, args.figdir)
    fp_conv = fig_perfold_convergence(histories, args.figdir)
    fp_evk = fig_error_vs_k(sigma_zonly, kgrid, z_band_edges, args.figdir)
    fp_flag = fig_dla_shotflag(dla_shot_flag, kgrid, args.figdir)
    print("figures:")
    for p in (fp_loss, fp_conv, fp_evk, fp_flag):
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
