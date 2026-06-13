#!/usr/bin/env python3
"""Launch ONLY the Phase-5a EMUCOH validation chains (the *_EC0 / *_EC1 mocks) as a pool.

A thin wrapper over run_stepA.dispatch that filters build_config() to just the 64 EC chains and
writes to a SEPARATE health file (health_emucoh.json), so it can run CONCURRENTLY with another
run_stepA pool on a different node without (a) re-running that pool's in-flight chains — the EC
checkpoint files are disjoint from every other mock id, and dispatch skips-done — or (b) racing on
health.json. Production NUTS knobs (PROD). Usage:

  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 python3 scripts/run_emucoh_pool.py --workers 20
"""
import argparse
import sys

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import run_stepA as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    R._force_single_thread_env()
    # separate health log so we never clobber a concurrent run_stepA pool's health.json
    R.HEALTH_JSON = f"{R.CKPT_DIR}/health_emucoh.json"
    R.HEALTH_TXT = f"{R.CKPT_DIR}/health_emucoh.txt"

    cfg = [c for c in R.build_config() if c["mock_id"].endswith(("_EC0", "_EC1", "_EC2"))]
    mocks = sorted(set(c["mock_id"] for c in cfg))
    print(f"[emucoh-pool] {len(cfg)} chains over {len(mocks)} mocks: {mocks}", flush=True)
    nuts = dict(R.PROD)
    print(f"[emucoh-pool] NUTS knobs: {nuts}; workers={args.workers}", flush=True)
    R.dispatch(cfg, workers=args.workers, nuts_kwargs=nuts, smoke=False)


if __name__ == "__main__":
    main()
