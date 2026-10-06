#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 section 3, step 4: the T3 product from all 60 simulations with the recorded comparison
(``--comparison``, the step-3 JSON: per leg the chosen representation and its CV-best rank). Representation P: each
simulation's coherent clean-class residual placed on the leg's bins (kept, k >= 0.01 s/km, z on the 2.0-4.4 grid) at
its own coordinate with the production log-k weights; F = mean_s v_s v_s^T truncated to the rank; stored as U (N_leg,
r) (rows of other bins zero) and w (r,), fractional; production algebra (amplitude P_data, off-diagonal only, diagonal
absorbed by max). Production applies T3 on DESI and KS. A leg whose recorded choice is M is refused (not wired).
Write-once ``cemu_t3`` product.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/build_gate_e_t3.py --eval-dir <..> --comparison <t3_comparison.json> --out <product.npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator import cemu_t3 as T3
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.products import save_product
from hcd_analysis.emulator.schema import L_BOX_HMPC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gate_e_t3_compare as CMP  # noqa: E402  (the step-3 definitions: z grid, k cut, coherent residual, bins)

PROD_LEGS = ("DESI", "KS")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def legs():
    return {"DESI": DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
            "KS": DL.load_ks_leg(resolution_float=True, k_max=0.065)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rec = json.load(open(a.comparison))["legs"]
    d = load_cache(a.cache)
    k_com = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC
    R = CB.load_loo_ensemble_residuals(a.eval_dir, d)
    sims, coh, theta = CMP.coh_per_sim(R, d)
    arrays, ranks = {}, {}
    for name, leg in legs().items():
        if rec[name]["choice"] != "P" or rec[name].get("stop_2"):
            raise SystemExit(f"{name}: recorded T3 choice {rec[name]['choice']} (stop_2 {rec[name].get('stop_2')}); "
                             "only P is wired")
        r = int(rec[name]["decision_stats"]["best_rank"]["P"])
        bins, B, kept, pos = CMP.leg_bins(leg)
        used = np.unique(bins["iz"])
        if not np.all(np.isfinite(coh[:, used])):
            raise SystemExit(f"{name}: a used z cell lacks rows")
        c = np.nan_to_num(coh)
        V = np.stack([T3.weights(k_com, theta[s], bins["z"], bins["iz"], bins["k"]) @ c[s].ravel()
                      for s in range(len(sims))])
        S = V.T @ V / V.shape[0]
        w, U = np.linalg.eigh(0.5 * (S + S.T))
        w, U = w[::-1][:r], U[:, ::-1][:, :r]
        Uleg = np.zeros((np.asarray(leg.k).size, r))
        Uleg[B] = U
        arrays.update({f"U_{name}": Uleg, f"w_{name}": np.maximum(w, 0.0), f"k_{name}": np.asarray(leg.k),
                       f"z_{name}": np.asarray(leg.z_row, float)})
        ranks[name] = r
        print(f"{name}: rank {r}, {B.size} bins, captured trace {w.sum() / np.trace(S):.4f}")
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    prov = dict(code_commit=commit, cache_sha256=_sha(a.cache), representation="P", ranks=ranks,
                inputs={"eval_dir": a.eval_dir, "comparison": a.comparison, "comparison_sha256": _sha(a.comparison)},
                row_rule=("coh_s(z, mode) = mean over simulation s's rows at z (all rungs) of the clean-class LOO ensemble "
                          "residual; placed on the leg's kept bins with k >= 0.01 and z on 2.0-4.4 at s's own coordinate "
                          "(production log-k weights); F = mean_s v v^T over all 60 simulations, top-r"),
                amendment="GATE_E_AMENDMENT_A1 rev 1 section 3 (PU-0068)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_product(a.out, "cemu_t3", k_com_hmpc=k_com, provenance=prov, **arrays)
    print(f"wrote {a.out} sha256 {_sha(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
