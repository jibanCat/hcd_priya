#!/usr/bin/env python3
"""Launch ONLY the eBOSS DR14 closure-cert chains (survey=="eBOSS") as a pool, pinned to the FREE
high cores so it runs CONCURRENTLY with other run_stepA pools without oversubscription.

A concurrent EC pool uses the low cores (sched_getaffinity[:workers]); by restricting THIS process's
affinity to allowed[core_offset:] before dispatch, the eBOSS workers taskset onto the high cores only.
Writes a SEPARATE health_eboss.json. Usage:
  ... python3 scripts/run_eboss_cert.py --core-offset 14 --workers 10
"""
import argparse
import os
import sys

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import run_stepA as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-offset", type=int, default=14,
                    help="skip the first N allowed cores (used by a concurrent EC/diag pool)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    R._force_single_thread_env()
    R.HEALTH_JSON = f"{R.CKPT_DIR}/health_eboss.json"
    R.HEALTH_TXT = f"{R.CKPT_DIR}/health_eboss.txt"

    allowed = sorted(os.sched_getaffinity(0))
    free = allowed[args.core_offset:] or allowed
    try:
        os.sched_setaffinity(0, set(free))    # dispatch will read this restricted set
    except OSError:
        pass
    cfg = [c for c in R.build_config() if c.get("survey") == "eBOSS"]
    mocks = sorted(set(c["mock_id"] for c in cfg))
    nw = min(args.workers, len(free), len(cfg))
    print(f"[eboss-cert] {len(cfg)} chains over {len(mocks)} mocks {mocks}; "
          f"pinned to {len(free)} free cores (offset {args.core_offset}); workers={nw}", flush=True)
    R.dispatch(cfg, workers=nw, nuts_kwargs=dict(R.PROD), smoke=False)


if __name__ == "__main__":
    main()
