#!/usr/bin/env python3
"""Train ONE member of the matched leave-one-simulation-out comparison (hcd_analysis/emulator/loo_matched.py).

Identical to scripts/train_production_emulator.py (FINAL_RECIPE weighting, the same train_fold call and checkpoint
writer) except for the rows: the fold's matched-design rows of the other simulations (the PRIYA rungs and redshifts),
early stopping on the production row split of those rows. ``--step-matched`` is the pre-registered secondary arm: the
epoch count that restores the production optimizer-step budget. The per-fold delta_c coefficients are written next to
the checkpoint. Writes only into a NEW ``--out`` directory.

  python scripts/run_matched_loo_member.py --cache LF_CACHE.h5 --holdout-sim NAME --seed S --out NEW_DIR [--step-matched]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import hcd_analysis.emulator  # noqa: E402,F401  (x64 before jax arrays exist)
import jax  # noqa: E402

from hcd_analysis.emulator import loo_matched as L  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import edge_emphasis_k_weight, load_cache  # noqa: E402

RECIPE = dict(n_basis=24, p_resid_w=8.0, edge_gain=3.0, lowk_extra=2.0, w_coh=80.0, weight_decay=3e-4,
              datarange=True, epochs=180, patience=25, lr=1e-3, batch=512)   # = train_production_emulator.RECIPE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout-sim", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="NEW directory (refused if it exists)")
    ap.add_argument("--step-matched", action="store_true")
    a = ap.parse_args()
    if os.path.lexists(a.out):
        sys.exit(f"refusing: {a.out} exists")
    t0 = time.time()
    d = load_cache(a.cache)
    L.check_matched_design(d)
    fit, val, held = L.fold_split(d, a.holdout_sim)
    n_k = d["P_tier_p"].shape[1]
    epochs = L.step_matched_epochs(fit.size, batch_size=RECIPE["batch"]) if a.step_matched else RECIPE["epochs"]
    term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0, "p_resid": RECIPE["p_resid_w"], "delta": 1.0}
    k_weight = edge_emphasis_k_weight(d["k_com_hmpc"], edge_gain=RECIPE["edge_gain"], lowk_extra=RECIPE["lowk_extra"])
    model, norm_stats, history = T.train_fold(
        d, fit, val, n_basis=RECIPE["n_basis"], lr=RECIPE["lr"], epochs=epochs, batch_size=RECIPE["batch"],
        seed=a.seed, key=jax.random.PRNGKey(a.seed), patience=RECIPE["patience"], n_k=n_k, term_w=term_w,
        k_weight=k_weight, w_coh=RECIPE["w_coh"], datarange=RECIPE["datarange"], weight_decay=RECIPE["weight_decay"])
    train_sims = sorted(set(np.asarray(d["sim_name"])[fit]) | set(np.asarray(d["sim_name"])[val]))
    coeffs, resid_std = L.fit_delta_c(d, train_sims)
    os.makedirs(a.out)
    recipe = dict(RECIPE, epochs=epochs, matched_design=True, step_matched=a.step_matched, holdout_sim=a.holdout_sim,
                  val_frac=L.VAL_FRAC, val_seed=L.VAL_SEED, n_fit=int(fit.size), n_val=int(val.size),
                  n_held=int(held.size))
    ckpt = os.path.join(a.out, f"matched_{a.seed}")
    T.save_checkpoint(ckpt, model, {"in_dim": 10, "n_k": n_k, "n_basis": RECIPE["n_basis"]}, norm_stats, seed=a.seed,
                      cache=d, cache_path=a.cache, recipe=recipe)
    with open(os.path.join(a.out, "fold.json"), "x") as f:
        json.dump({"holdout_sim": a.holdout_sim, "fit_rows": fit.tolist(), "val_rows": val.tolist(),
                   "held_rows": held.tolist(), "delta_c_coeffs": coeffs.tolist(), "delta_c_resid_std": resid_std.tolist(),
                   "n_epochs_run": len(history["train_loss"]), "epochs": epochs,
                   "best_val": float(np.min(history["val_loss"])), "seconds": time.time() - t0}, f)
    print(f"[matched-loo] {a.holdout_sim} seed {a.seed}: {len(history['train_loss'])} epochs, "
          f"{time.time() - t0:.0f} s -> {ckpt}")


if __name__ == "__main__":
    main()
