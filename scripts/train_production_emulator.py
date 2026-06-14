#!/usr/bin/env python3
"""Train ONE member of the PRODUCTION (all-sims, no LOSO holdout) emulator ensemble.

Baseline config (PI-decided 2026-06-14): the VALIDATED FINAL_RECIPE architecture + hyperparameters
(byte-for-byte from the 8 LOSO folds), trained on ALL 60 sims + ALL τ₀ rungs, with a FIXED random
~10% ROW-val held out ONLY for restore-best early-stop (in-distribution — no sim or τ₀ rung is fully
excluded; the architecture's GENERALIZATION is already certified by the LOSO folds + the 5-review GO).
The production model is a MULTI-SEED ENSEMBLE: run this once per seed (--seed S → final_prod_seed{S});
all members share the SAME row-val split (VAL_SEED fixed) so the ensemble isolates the single-draw
optimisation/init scatter. C_emu = the existing 8-fold error_vector.npz (the LOSO generalisation budget,
conservative for an all-sims model trained on MORE data).

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/train_production_emulator.py --seed 0
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import jax

import hcd_analysis.emulator  # noqa: F401  x64 before jax
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache, edge_emphasis_k_weight

REPO = "/home/mfho/hcd_priya"
CACHE = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = f"{REPO}/checkpoints/final_prod"          # → final_prod_seed{S}.{eqx,meta.json,norm.pkl}
VAL_SEED = 12345                                 # FIXED row-val split (shared across ALL ensemble members)
VAL_FRAC = 0.10                                  # ~10% random rows held out for restore-best early-stop ONLY

# the VALIDATED recipe (run_loso_sweep.FINAL_RECIPE) — reused byte-for-byte
RECIPE = dict(n_basis=24, p_resid_w=8.0, edge_gain=3.0, lowk_extra=2.0,
              w_coh=80.0, weight_decay=3e-4, datarange=True,
              epochs=180, patience=25, lr=1e-3, batch=512)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, help="ensemble-member seed (→ final_prod_seed{S})")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    d = load_cache(args.cache)
    n_rows = d["P_tier_p"].shape[0]; n_k = d["P_tier_p"].shape[1]
    n_sims = len(set(d["sim_name"]))
    # ALL-SIMS split: fixed random row-val (same for every member), train on the rest. ALL τ₀ rungs
    # included (NO τ₀-edge holdout — the production model should be accurate everywhere it's evaluated).
    perm = np.random.default_rng(VAL_SEED).permutation(n_rows)
    n_val = int(VAL_FRAC * n_rows)
    val_idx = np.sort(perm[:n_val]); train_idx = np.sort(perm[n_val:])
    print(f"[prod] cache {args.cache}: {n_rows} rows, n_k={n_k}, {n_sims} sims  ({time.time()-t0:.1f}s)")
    print(f"[prod] ALL-SIMS split: train={len(train_idx)} val={len(val_idx)} (row-val frac {VAL_FRAC}, "
          f"VAL_SEED={VAL_SEED}); member seed={args.seed}")

    # FINAL_RECIPE weighting (identical to run_loso_sweep)
    term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0, "p_resid": RECIPE["p_resid_w"], "delta": 1.0}
    k_weight = edge_emphasis_k_weight(d["kfkms"][0], edge_gain=RECIPE["edge_gain"],
                                      lowk_extra=RECIPE["lowk_extra"])

    tf = time.time()
    model, norm_stats, history = T.train_fold(
        d, train_idx, val_idx, n_basis=RECIPE["n_basis"], lr=RECIPE["lr"], epochs=RECIPE["epochs"],
        batch_size=RECIPE["batch"], seed=args.seed, key=jax.random.PRNGKey(args.seed),
        patience=RECIPE["patience"], n_k=n_k, term_w=term_w, k_weight=k_weight,
        w_coh=RECIPE["w_coh"], datarange=RECIPE["datarange"], weight_decay=RECIPE["weight_decay"])
    n_ep = len(history["train_loss"]); best_val = float(np.min(history["val_loss"]))
    print(f"[prod] seed {args.seed}: {n_ep} epochs in {time.time()-tf:.0f}s; best_val={best_val:.6g}")

    arch_cfg = {"in_dim": 10, "n_k": n_k, "n_basis": RECIPE["n_basis"]}
    recipe = dict(RECIPE, all_sims=True, val_frac=VAL_FRAC, val_seed=VAL_SEED,
                  n_train=int(len(train_idx)), n_val=int(len(val_idx)))
    ckpt = f"{args.out}_seed{args.seed}"
    T.save_checkpoint(ckpt, model, arch_cfg, norm_stats, seed=args.seed,
                      kfkms=d["kfkms"], cache_path=args.cache, recipe=recipe)
    hist_json = {k: np.asarray(v).astype(float).tolist() for k, v in history.items()}
    hist_json["recipe"] = recipe
    Path(f"{ckpt}.hist.json").write_text(json.dumps(hist_json, indent=2))
    print(f"[prod] checkpoint → {ckpt}.eqx/.meta.json/.norm.pkl/.hist.json  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
