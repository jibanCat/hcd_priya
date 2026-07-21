"""Run ONE shard of the Leg-B coverage pilot: the mocks m with m % n_shards == shard.

Per-mock RNG is fold_in(seed, m), so shards are disjoint + reproducible and the merged set
equals a single full run. Writes the raw per-mock list to {out_dir}/shard_{shard}.pkl for
merge_legb_shards.py. Uses the DE-CIRCULARISED rho (folds 1-7) by default.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import os
import pickle

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax  # noqa: F401
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb

HOLDOUT0_EV = "/home/mfho/hcd_priya/checkpoints/error_vector_xclass_holdout0.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--xclass-ev", default=HOLDOUT0_EV,
                    help="cross-class error vector (default: de-circularised folds-1-7 rho)")
    ap.add_argument("--cemu-inflate", type=float, default=1.0)
    ap.add_argument("--n-warmup", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=250)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    ctx, d = build_legb_ctx(xclass_error_vector=a.xclass_ev,
                            cemu_inflate=a.cemu_inflate, use_xclass=True)
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] mocks={idxs}  rho={os.path.basename(a.xclass_ev)}  "
          f"legs={[l.name for l in ctx.legs]}  (warmup={a.n_warmup} samples={a.n_samples} "
          f"mtd={a.max_tree_depth})")
    per_mock = run_legb(ctx, d, n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True,
                        n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed,
                        dense_mass=True, max_tree_depth=a.max_tree_depth, verbose=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"shard_{a.shard:03d}.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(idxs=idxs, per_mock=per_mock, meta=vars(a)), f)
    print(f"[shard {a.shard}] wrote {len(per_mock)} mocks -> {out}")


if __name__ == "__main__":
    main()
