"""Merge Leg-B coverage shards -> aggregate -> print coverage + bias + emit the figure.

Leg-B verdict basis = coverage >= nominal + bias ~ 0 ONLY (the loglik-rank ECDF is
DIAGNOSTIC-ONLY here; Leg-B's null is not rank-uniform).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import numpy as np
import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator.closure_legb import _aggregate_legb, L_FLOOR
from hcd_analysis.emulator.closure_legb_figs import fig_smoke_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--q-levels", default="0.68,0.95")
    ap.add_argument("--figdir", default="/home/mfho/hcd_priya/figures/analysis/05_likelihood")
    a = ap.parse_args()
    q_levels = tuple(float(x) for x in a.q_levels.split(","))

    per_mock, seen = [], set()
    for fn in sorted(glob.glob(os.path.join(a.shard_dir, "shard_*.pkl"))):
        with open(fn, "rb") as f:
            dd = pickle.load(f)
        for rec, mi in zip(dd["per_mock"], dd["idxs"]):
            if mi in seen:
                continue
            seen.add(mi); per_mock.append(rec)
    if not per_mock:
        raise SystemExit(f"no shards in {a.shard_dir}")
    res = _aggregate_legb(per_mock, q_levels=q_levels)

    print(f"\n===== Leg-B coverage (de-circularised rho) =====")
    print(f"N_mocks={res['n_mocks']}  L(thinned)={res['L']}  "
          f"gate_valid(L>={L_FLOOR})={res['gate_valid']}  divergent={res['n_divergent']}")
    for q in res["q_levels"]:
        print(f"\ncoverage @ {int(q*100)}% CR (target >= {q:.2f}):")
        for nm in res["names"]:
            c = res["coverage"][q][nm]
            flag = "" if c["ci_high"] >= q else "  <-- below nominal"
            print(f"  {nm:14s}: {c['coverage']:.2f}  [{c['ci_low']:.2f},{c['ci_high']:.2f}]"
                  f"  (k={c['k']}/{c['n']}){flag}")
    print("\nper-param normalized bias (truth-mean)/std, mean over mocks (~0 = unbiased):")
    for nm in res["names"]:
        b = res["bias"][nm]
        print(f"  {nm:14s}: {b['mean']:+.2f} +/- {b['se']:.2f} sigma  (n={b['n']})")
    print("\n[verdict basis] coverage >= nominal + bias~0 ONLY (loglik-rank ECDF is diagnostic).")

    p = fig_smoke_coverage(
        res, figdir=a.figdir,
        suptitle=f"Leg-B coverage pilot (N={res['n_mocks']}, de-circularised rho) — "
                 "coverage + bias are the verdict; loglik-rank ECDF DIAGNOSTIC-ONLY")
    print(f"[fig] {p}")


if __name__ == "__main__":
    main()
