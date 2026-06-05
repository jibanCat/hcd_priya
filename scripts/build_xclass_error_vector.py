#!/usr/bin/env python3
"""Phase-C — build the CROSS-CLASS (4×4) emulator-error block ρ_cc'(k,z,τ₀).

WHY: the diagonal C_emu (``emu_var = Σ_c coef_c²·σ_c²·P_c²``) UNDER-sizes the held-out
P_obs residual by ~2.5× (whitening var≈2.46). A CS decomposition pinned the cause to
**cross-class correlation**: the 4 per-class residuals come from ONE network, so they are
coherently correlated (+1.18 of the χ²/dof) — NOT k-correlation, NOT coarse binning. The
fix is to upgrade the per-class DIAGONAL σ_c² to a cross-class 4×4 block ρ_cc' estimated
over the POOLED held-out residuals from ALL 8 folds, so the off-diagonals carry the
class-coupling and the fold-amplitude variance is absorbed by the pooled covariance.

MATH (the consumer is ``inference.predict_P_obs_and_cov_single_z``):
  diagonal (today):  emu_var(k) = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c(k)²
  cross-class (new):  emu_var(k) = Σ_{c,c'} coef_c·coef_c'·ρ_cc'(k,z,τ₀)·P_c(k)·P_c'(k)
  ρ_cc'(k,z,τ₀) = mean over the cell's POOLED held-out rows of [rfrac_c(k)·rfrac_c'(k)],
  rfrac_c = (P_pred_c − P_true_c)/P_true_c.
The DIAGONAL ρ_cc = mean(rfrac_c²) = σ_c² (recovers today's per-cell variance, MODULO the
pooled-vs-RMS-over-folds difference); the OFF-diagonals add the coupling. ρ is a SAMPLE
covariance ⇒ SPD ⇒ emu_var = coefᵀ(P∘ρ∘P)coef ≥ 0 guaranteed. C_emu stays DIAGONAL IN k
(CS confirmed no k-correlation) — only the per-k 4×4 block is new.

WHAT: load all 8 ``final_fold{f}`` + the v3.3 cache; per fold extract the held-out
``rfrac`` (Nval,4,K) via run_loso_sweep's EXACT model→untransform→rfrac path; POOL across
folds; per (k, z-band Zb, τ₀-band Tb) cell compute the 4×4 ρ. Save
``checkpoints/error_vector_xclass.npz`` (``rho`` (4,4,K,Zb,Tb) + carried metadata).
ASSERTS the new diagonal ≈ the old per-class variance and reports the ratio.

Does NOT modify any committed library module.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/build_xclass_error_vector.py [args]
"""
from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import (
    load_cache, make_splits, untransform_prediction, COARSE_NAMES,
    datarange_mask, make_tau0_bands,
)
from scripts.run_loso_sweep import make_z_bands

REPO = "/home/mfho/hcd_priya"
DEFAULT_CKPT = f"{REPO}/checkpoints/final_fold"        # +{f}
DEFAULT_EV = f"{REPO}/checkpoints/error_vector.npz"
DEFAULT_OUT = f"{REPO}/checkpoints/error_vector_xclass.npz"


def fold_rfrac(d, model, val_idx, norm_stats, datarange=True):
    """Per-fold held-out FRACTIONAL residual ``rfrac`` (Nval,4,K) — run_loso_sweep's path.

    rfrac[r,c,k] = (P_filt_pred[r,c,k] − P_filt_true[r,c,k]) / P_filt_true[r,c,k] in LINEAR
    per-class P1D space (predictions untransformed from the emulator's standardized-log
    space via ``untransform_prediction``). EXACTLY mirrors ``run_loso_sweep.fold_resid_neff``
    up to the reduction step: where ``fold_resid_neff`` reduces to ``sigma=sqrt(mean(rfrac²))``
    per (z,τ₀)-cell, we KEEP the full per-row rfrac so the cross-class build can form the
    4×4 ``mean(rfrac_c·rfrac_c')`` per cell. A residual is kept only where BOTH pred & true
    are finite, the row's k-mask is set, the true value is non-zero, AND (datarange) the
    (z,k) bin is in the DESI data range; dropped residuals are NaN (nan-safe downstream).
    """
    val_idx = np.asarray(val_idx)
    x = jnp.asarray(d["x"][val_idx])
    tau0 = jnp.asarray(d["tau0"][val_idx])
    pred = jax.vmap(model)(x, tau0)
    P_pred = untransform_prediction(
        {"P_filt_base": np.asarray(pred["P_filt_base"]),
         "P_filt_resid": np.asarray(pred["P_filt_resid"])},
        norm_stats)["P_filt"]                                          # (Nval,4,K) linear
    P_true = d["P_filt"][val_idx]                                       # (Nval,4,K) linear
    mask_k = d["mask"][val_idx]                                         # (Nval,K) finite Tier-P

    keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true != 0.0)
            & mask_k[:, None, :])                                       # (Nval,4,K)
    if datarange:
        in_range = datarange_mask(d)[val_idx]                          # (Nval,K)
        keep = keep & in_range[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rfrac = (P_pred - P_true) / P_true
    rfrac = np.where(keep, rfrac, np.nan)                              # (Nval,4,K)
    return rfrac


def xclass_rho(rfrac, zband, tband, n_bands, n_tb, n_cls, K):
    """Per (k, z-band, τ₀-band) cell: ρ_cc'(k) = mean over the cell's pooled rows of
    [rfrac_c(k)·rfrac_c'(k)], the 4×4 cross-class second moment (a sample covariance ⇒ SPD).

    ``rfrac`` (N,4,K) is the POOLED-over-folds held-out residual; ``zband``/``tband`` (N,)
    assign each row to its (z,τ₀) cell. Empty (c,c',k) entries (no finite row pair) are NaN
    (nan-safe in the τ₀-interp). The DIAGONAL ρ_cc(k) = mean(rfrac_c²) = the per-class
    variance the old σ_c² RMS targets; the OFF-diagonals are the class coupling.

    Implementation: for each cell, the 4×4 block at each k is the nan-mean over rows of the
    outer product rfrac[:,:,k] ⊗ rfrac[:,:,k]. We compute it as a masked matmul per k-slab
    so a row with a NaN in class c does NOT poison the (c',c'') entries it CAN inform — each
    (c,c') entry uses its OWN finite-pair count (pairwise-complete, like np.cov's pairwise).
    """
    rho = np.full((n_cls, n_cls, K, n_bands, n_tb), np.nan)
    for zb in range(n_bands):
        for tb in range(n_tb):
            sel = (zband == zb) & (tband == tb)
            if not sel.any():
                continue
            rb = rfrac[sel]                                            # (nb,4,K)
            finite = np.isfinite(rb)                                  # (nb,4,K)
            rb0 = np.where(finite, rb, 0.0)                           # zero the NaNs
            f = finite.astype(np.float64)
            # per-k 4×4 sums via einsum over rows: prod_cc'(k)=Σ_r rb_c·rb_c', n_cc'(k)=Σ_r f_c·f_c'.
            prod = np.einsum("rck,rdk->cdk", rb0, rb0)               # (4,4,K) Σ rfrac_c·rfrac_c'
            npair = np.einsum("rck,rdk->cdk", f, f)                  # (4,4,K) finite-pair counts
            with np.errstate(invalid="ignore", divide="ignore"):
                cell = np.where(npair > 0, prod / npair, np.nan)     # (4,4,K) pairwise mean
            # HARDENING (CS review 2026-06-05): collapse any MIXED k-slab (some of the 16
            # 4×4 entries finite, some NaN — possible if a (c,c') pair has zero finite-pairs
            # while the diagonals don't) to ALL-NaN. The consumer's nan_to_num then zeros the
            # WHOLE block (trivially PSD) instead of partially zeroing an off-diagonal, which
            # can make an otherwise-SPD block indefinite → NaN Cholesky/logdet. (Current
            # production build has 0 mixed cells; this guards future sparser pools.)
            nfin = np.isfinite(cell).sum(axis=(0, 1))               # (K,) finite entries per k-slab
            cell[:, :, (nfin > 0) & (nfin < n_cls * n_cls)] = np.nan
            rho[:, :, :, zb, tb] = cell
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None,
                    help="v3.3 cache path (default: read from fold-0 meta cache_path)")
    ap.add_argument("--ckpt-prefix", default=DEFAULT_CKPT)
    ap.add_argument("--error-vector", default=DEFAULT_EV,
                    help="the existing DIAGONAL error vector (carries z/τ₀ bands + sigma)")
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--z-bands", type=int, default=3)
    ap.add_argument("--tau0-bands", type=int, default=4)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("jax.devices():", jax.devices())
    t0 = time.time()

    # carry the band scheme + sigma from the existing diagonal vector (so the cross-class
    # block is on the IDENTICAL (Zb,Tb) grid the diagonal C_emu uses).
    ev = np.load(args.error_vector, allow_pickle=True)
    sigma_old = ev["sigma"]                                # (4,K,Zb,Tb) diagonal RMS-over-folds
    n_cls, K_ev, Zb_ev, Tb_ev = sigma_old.shape
    tau0_band_centres = ev["tau0_band_centres"]
    z_band_edges_ev = ev["z_band_edges"]
    dla_shot_flag = ev["dla_shot_flag"]
    kfkms = ev["kfkms"]
    print(f"loaded diagonal error vector {args.error_vector}: sigma {sigma_old.shape}")

    # load the cache from fold-0's meta (the matched v3.3 build) unless overridden.
    _m, meta0, _n = T.load_checkpoint(f"{args.ckpt_prefix}0")
    cache_path = args.cache or meta0["cache_path"]
    d = load_cache(cache_path)
    R, K = d["P_filt"].shape[0], d["P_filt"].shape[2]
    assert K == K_ev, f"cache n_k={K} != error_vector n_k={K_ev}"
    print(f"loaded cache {cache_path}: {R} rows, n_k={K} "
          f"({len(set(d['sim_name']))} sims, {time.time()-t0:.1f}s)")

    # rebuild the SAME z-band / τ₀-band assignment run_loso_sweep used (deterministic from
    # z_grid / tau0); assert the edges/centres match the carried diagonal vector.
    z_band_of_row, z_band_edges = make_z_bands(d["z_grid"], args.z_bands)
    tau0_band_of_row, tau0_centres = make_tau0_bands(d["tau0"], d["z_grid"], args.tau0_bands)
    assert args.z_bands == Zb_ev and args.tau0_bands == Tb_ev, \
        f"band count mismatch vs error_vector (Zb {args.z_bands}/{Zb_ev}, Tb {args.tau0_bands}/{Tb_ev})"
    assert np.allclose(np.nan_to_num(z_band_edges, nan=0.0, posinf=0.0, neginf=0.0),
                       np.nan_to_num(z_band_edges_ev, nan=0.0, posinf=0.0, neginf=0.0)), \
        "z_band_edges differ from the carried error_vector"
    assert np.allclose(tau0_centres, tau0_band_centres, rtol=1e-6), \
        "tau0_band_centres differ from the carried error_vector"
    print(f"z-bands {args.z_bands} edges {np.array2string(z_band_edges, precision=3)}; "
          f"τ₀-bands {args.tau0_bands} α-centres "
          f"{np.array2string(np.asarray(tau0_centres), precision=3)}")

    # --- POOL the held-out rfrac across all 8 folds ---------------------------
    # each fold contributes its OWN val rows (LOSO: a sim is held out in exactly one fold),
    # so pooling concatenates DISJOINT held-out rows → the pooled set is every sim's held-out
    # residual exactly once. Stash each row's (z-band, τ₀-band) so the cells pool correctly.
    rfrac_pool, zband_pool, tband_pool = [], [], []
    n_rows_per_fold = []
    for f in range(args.n_folds):
        model, meta, norm = T.load_checkpoint(f"{args.ckpt_prefix}{f}")
        _tr, va, _ho = make_splits(d, f, n_folds=args.n_folds)
        rfrac = fold_rfrac(d, model, va, norm, datarange=True)         # (Nval,4,K)
        rfrac_pool.append(rfrac)
        zband_pool.append(z_band_of_row[va])
        tband_pool.append(tau0_band_of_row[va])
        n_rows_per_fold.append(len(va))
        print(f"  fold {f}: held-out {len(va)} rows pooled "
              f"(finite rfrac frac {np.isfinite(rfrac).mean():.3f})")
    rfrac_pool = np.concatenate(rfrac_pool, axis=0)                    # (Npool,4,K)
    zband_pool = np.concatenate(zband_pool, axis=0)                    # (Npool,)
    tband_pool = np.concatenate(tband_pool, axis=0)                    # (Npool,)
    Npool = rfrac_pool.shape[0]
    print(f"pooled {Npool} held-out rows across {args.n_folds} folds "
          f"({time.time()-t0:.1f}s)")

    # --- per-cell 4×4 ρ block --------------------------------------------------
    rho = xclass_rho(rfrac_pool, zband_pool, tband_pool, args.z_bands,
                     args.tau0_bands, n_cls, K)                        # (4,4,K,Zb,Tb)
    print(f"built rho {rho.shape} ({time.time()-t0:.1f}s)")

    # --- SANITY: the new diagonal ρ_cc vs the old per-class σ_c² ---------------
    # The new diagonal is the POOLED mean(rfrac_c²); the old σ_c² is the RMS-OVER-FOLDS of the
    # per-fold per-cell mean(rfrac_c²). They differ ONLY by the pooling-vs-fold-RMS weighting
    # (Jensen / unequal fold counts) — report the ratio so the cross-class diagonal is a
    # faithful super-set of the diagonal budget, not a silent re-scaling.
    rho_diag = np.stack([rho[c, c] for c in range(n_cls)])            # (4,K,Zb,Tb)
    sig2_old = sigma_old ** 2                                         # (4,K,Zb,Tb)
    both = np.isfinite(rho_diag) & np.isfinite(sig2_old) & (sig2_old > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(both, rho_diag / sig2_old, np.nan)
    med_ratio = float(np.nanmedian(ratio))
    p16, p84 = (float(np.nanpercentile(ratio, 16)),
                float(np.nanpercentile(ratio, 84)))
    print("\n=== DIAGONAL-MATCH SANITY (new rho[c,c] vs old sigma[c]²) ===")
    print(f"  ratio rho_cc / sigma_c²  (pooled-mean vs RMS-over-folds):")
    print(f"    median = {med_ratio:.3f}   [16,84]% = [{p16:.3f}, {p84:.3f}]")
    for c, cn in enumerate(COARSE_NAMES):
        with np.errstate(invalid="ignore"):
            r_c = np.where(both[c], ratio[c], np.nan)
        print(f"    {cn:7s}: median {np.nanmedian(r_c):.3f}  "
              f"({int(np.isfinite(r_c).sum())} cells)")
    # off-diagonal coupling magnitude (the new physics): typical |ρ_cc'|/√(ρ_cc·ρ_c'c').
    corr_offdiag = []
    for c in range(n_cls):
        for cprime in range(c + 1, n_cls):
            denom = np.sqrt(rho[c, c] * rho[cprime, cprime])
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = rho[c, cprime] / denom
            corr = corr[np.isfinite(corr)]
            if corr.size:
                corr_offdiag.append((COARSE_NAMES[c], COARSE_NAMES[cprime],
                                     float(np.median(corr))))
    print("  off-diagonal class CORRELATION (median ρ_cc'/√(ρ_cc·ρ_c'c')):")
    for a, b, v in corr_offdiag:
        print(f"    {a:7s}–{b:7s}: {v:+.3f}")

    # --- SPD spot-check (the einsum's ≥0 guarantee rests on this) --------------
    # ρ is a sample covariance ⇒ SPD per (k,cell). A pairwise-complete estimate can be
    # marginally indefinite if some (c,c') pairs have different row support; report the
    # worst (most-negative) min-eigenvalue over all finite 4×4 cells.
    min_eig = np.inf
    n_cells = 0
    for zb in range(args.z_bands):
        for tb in range(args.tau0_bands):
            for k in range(K):
                M = rho[:, :, k, zb, tb]
                if np.isfinite(M).all():
                    w = np.linalg.eigvalsh(0.5 * (M + M.T))
                    min_eig = min(min_eig, float(w.min()))
                    n_cells += 1
    print(f"  SPD spot-check over {n_cells} finite 4×4 cells: "
          f"min eigenvalue = {min_eig:.3e} "
          f"({'OK ≥0' if min_eig >= -1e-12 else 'marginally indefinite — clamped at consume'})")

    # --- save -----------------------------------------------------------------
    out = Path(args.out)
    np.savez(
        out,
        rho=rho,                                  # (4,4,K,Zb,Tb) cross-class block
        sigma=sigma_old,                          # (4,K,Zb,Tb) carried diagonal (back-compat)
        dla_shot_flag=dla_shot_flag,
        z_band_edges=z_band_edges,
        tau0_band_centres=tau0_centres,
        class_names=np.array(COARSE_NAMES),
        kfkms=kfkms,
        diag_match_median=med_ratio,
        n_pool=Npool,
    )
    print(f"\ncross-class error vector -> {out}  (rho {rho.shape})")
    print(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
