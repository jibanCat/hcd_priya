#!/usr/bin/env python3
"""Launch ONLY the v2 submanifold (hier + z-slope marginalized) chains (HZ_* mocks) as a pool. Thin
wrapper over run_stepA.dispatch, filtered to HZ_*, separate health_hz.json so it can run
concurrently with another run_stepA pool.

The Option-B validation gate (joint DESI+KS): HB0=legacy OFF / HB1=hierarchical ON / HBp,HBm =
ON with the A_HCD prior center ±1σ. Production NUTS knobs. Usage:

  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 python3 scripts/run_hb_pool.py --workers 16
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
    R.HEALTH_JSON = f"{R.CKPT_DIR}/health_hz.json"
    R.HEALTH_TXT = f"{R.CKPT_DIR}/health_hz.txt"
    cfg = [c for c in R.build_config() if c["mock_id"].startswith("HZ_")]
    mocks = sorted(set(c["mock_id"] for c in cfg))
    print(f"[hz-pool] {len(cfg)} chains over {len(mocks)} mocks: {mocks}", flush=True)
    nuts = dict(R.PROD)
    print(f"[hz-pool] NUTS knobs: {nuts}; workers={args.workers}", flush=True)
    R.dispatch(cfg, workers=args.workers, nuts_kwargs=nuts, smoke=False)


if __name__ == "__main__":
    main()
