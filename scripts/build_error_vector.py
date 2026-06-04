"""Build the τ₀-AWARE emulator error vector by REUSING trained LOSO checkpoints.

Fast alternative to a full run_loso_sweep (which retrains): loads
``checkpoints/<prefix>_fold{0..N-1}`` (already trained), recomputes each fold's banded
residual via ``fold_resid_neff`` with the τ₀-band axis, aggregates over folds, and
writes ``error_vector.npz`` with sigma (4,K,Zb,Tb) + the τ₀-band α-centres (Phase-C T2).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/build_error_vector.py \
      [--prefix checkpoints/decomp_nb24] [--z-bands 3] [--tau0-bands 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import numpy as np

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_tau0_bands, COARSE_NAMES,
)
from hcd_analysis.emulator import train as T

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")

# fold_resid_neff + make_z_bands live in run_loso_sweep
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from run_loso_sweep import fold_resid_neff, make_z_bands  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(ROOT / "checkpoints/decomp_nb24"))
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--z-bands", type=int, default=3)
    ap.add_argument("--tau0-bands", type=int, default=4)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "checkpoints/error_vector.npz"))
    args = ap.parse_args()

    d = load_cache(args.cache)
    kgrid = d["kfkms"][0]
    z_band_of_row, z_band_edges = make_z_bands(d["z_grid"], args.z_bands)
    tau0_band_of_row, tau0_band_centres = make_tau0_bands(
        d["tau0"], d["z_grid"], args.tau0_bands)
    print(f"τ₀-band α-centres: {np.array2string(tau0_band_centres, precision=3)}")

    resid_folds, neff_folds = [], []
    for fold in range(args.n_folds):
        ckpt = f"{args.prefix}_fold{fold}"
        if not Path(ckpt + ".eqx").exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt}.eqx")
        model, meta, norm = T.load_checkpoint(ckpt)
        _, va, _ = make_splits(d, fold, n_folds=args.n_folds,
                               holdout_frac=args.holdout_frac)
        sigma, neff, *_ = fold_resid_neff(
            d, model, va, norm, z_band_of_row, args.z_bands,
            tau0_band_of_row=tau0_band_of_row, n_tb=args.tau0_bands, datarange=True)
        resid_folds.append(sigma)
        neff_folds.append(neff)
        print(f"  fold {fold}: val={len(va)} sigma{sigma.shape} "
              f"median={np.nanmedian(sigma):.4g}")

    ev = T.aggregate_error_vector(resid_folds, neff_folds)
    sigma = ev["sigma"]                                  # (4,K,Zb,Tb)
    np.savez(
        args.out,
        sigma=sigma,
        dla_shot_flag=ev["dla_shot_flag"],
        z_band_edges=z_band_edges,
        tau0_band_centres=tau0_band_centres,
        class_names=np.array(COARSE_NAMES),
        kfkms=kgrid,
    )
    print(f"\nwrote {args.out}  sigma shape {sigma.shape}")
    # per (class, τ₀-band) median σ — does it grow toward the ladder extremes?
    print("median σ per class × τ₀-band (RMS over k,z):")
    for ci, cn in enumerate(COARSE_NAMES):
        meds = [float(np.nanmedian(sigma[ci, :, :, tb])) for tb in range(args.tau0_bands)]
        print(f"  {cn:7s} " + "  ".join(f"tb{tb}={m:.4g}" for tb, m in enumerate(meds)))


if __name__ == "__main__":
    main()
