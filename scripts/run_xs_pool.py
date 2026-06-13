#!/usr/bin/env python3
"""Launch ONLY the cosmology-safety arm chains (XS_* mocks: subDLA prior-CENTER shift on JOINT
DESI+KS) as a pool. Thin wrapper over run_stepA.dispatch, filtered to the XS chains, writing to a
SEPARATE health file (health_xs.json) so it can run concurrently with another run_stepA pool.

The #9 referee panel must-do: verify a ±1σ subDLA prior-center mis-specification does NOT drag
n_s/A_p (the residual risk behind corr(subDLA,n_s)=+0.82). Production NUTS knobs (PROD). Usage:

  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 python3 scripts/run_xs_pool.py --workers 16
"""
import argparse
import sys

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import run_stepA as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    R._force_single_thread_env()
    R.HEALTH_JSON = f"{R.CKPT_DIR}/health_xs.json"
    R.HEALTH_TXT = f"{R.CKPT_DIR}/health_xs.txt"

    cfg = [c for c in R.build_config() if c["mock_id"].startswith("XS_")]
    mocks = sorted(set(c["mock_id"] for c in cfg))
    print(f"[xs-pool] {len(cfg)} chains over {len(mocks)} mocks: {mocks}", flush=True)
    nuts = dict(R.PROD)
    print(f"[xs-pool] NUTS knobs: {nuts}; workers={args.workers}", flush=True)
    R.dispatch(cfg, workers=args.workers, nuts_kwargs=nuts, smoke=False)


if __name__ == "__main__":
    main()
