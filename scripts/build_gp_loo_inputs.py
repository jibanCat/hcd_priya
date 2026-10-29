#!/usr/bin/env python3
"""Write the GP side of one matched leave-one-simulation-out fold (loo_matched.gp_fold_inputs) to a NEW .npz and print
its sha256 for the refit request of hcd_analysis/comparison/adapters/lyaemu_loo_refit.py. Runs in the emulator
environment (the hcd_analysis.emulator package imports JAX).

  python scripts/build_gp_loo_inputs.py --cache LF_CACHE.h5 --holdout-sim NAME --n-modes N --out NEW.npz
"""
import argparse
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis.emulator import loo_matched as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout-sim", required=True)
    ap.add_argument("--n-modes", type=int, required=True, help="modes 1..N (the reference training file's count)")
    ap.add_argument("--out", required=True, help="NEW .npz (refused if it exists)")
    a = ap.parse_args()
    c = L.read_cache_for_gp(a.cache)
    L.check_matched_design(c)
    L.write_gp_fold_inputs(a.out, L.gp_fold_inputs(c, a.holdout_sim, a.n_modes))
    with open(a.out, "rb") as f:
        print(hashlib.sha256(f.read()).hexdigest(), a.out)


if __name__ == "__main__":
    main()
