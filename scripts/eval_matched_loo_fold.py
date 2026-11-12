#!/usr/bin/env python3
"""Paired errors of ONE matched leave-one-simulation-out fold: C1 (the fold's members from
scripts/run_matched_loo_member.py) and the GP refit record (adapters/lyaemu_loo_refit.py) on the held-out rows.

The member directories must be one ensemble: one fold (holdout simulation, rows, per-fold delta_c), one training arm,
trained on the given cache, no directory twice. A member with a non-finite prediction on a compared mode is dropped,
and a member that is missing (fewer directories than ``--expected-members``, for example one stopped at its time
limit) counts as dropped; fewer than 4 usable members, or a GP fold without a usable record (``--gp-not-run REASON``
when its job left none), makes the fold NOT_RUN for both methods. The GP's training rows must be exactly C1's fit and
early-stopping rows. Writes a NEW json for scripts/matched_loo_readout.py.

  python scripts/eval_matched_loo_fold.py --cache LF_CACHE.h5 --holdout-sim NAME --member-dirs D1 D2 ...
      (--gp-record REC.json | --gp-not-run REASON) --out NEW.json [--expected-members 5] [--truth-weights]
"""

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import hcd_analysis.emulator  # noqa: E402,F401  (x64 before jax arrays exist)

from hcd_analysis.emulator import loo_matched as L  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import load_cache  # noqa: E402
from hcd_analysis.emulator.safe_io import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout-sim", required=True)
    ap.add_argument("--member-dirs", nargs="*", default=[])
    ap.add_argument("--expected-members", type=int, default=5)
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument("--gp-record")
    gp.add_argument(
        "--gp-not-run",
        metavar="REASON",
        help="the GP job left no record (for example its time limit)",
    )
    ap.add_argument("--out", required=True, help="NEW json (refused if it exists)")
    ap.add_argument(
        "--truth-weights",
        action="store_true",
        help="context only: the held-out row's class fractions",
    )
    a = ap.parse_args()
    if os.path.lexists(a.out):
        sys.exit(f"refusing: {a.out} exists")
    cache_sha = sha256_file(a.cache)
    d = load_cache(a.cache)
    members = []
    for md in a.member_dirs:
        with open(os.path.join(md, "fold.json")) as f:
            fj = json.load(f)
        (meta_path,) = glob.glob(os.path.join(md, "*.meta.json"))
        model, meta, norm = T.load_checkpoint(meta_path[: -len(".meta.json")])
        members.append((md, fj, meta, (model, norm)))
    if a.gp_not_run is not None:
        if not a.gp_not_run.strip():
            sys.exit("--gp-not-run needs a reason")
        rec = {"fold_sim": a.holdout_sim, "status": "NOT_RUN", "reason": a.gp_not_run}
    else:
        with open(a.gp_record) as f:
            rec = json.load(f)
    try:
        out = L.assemble_fold(
            d,
            a.holdout_sim,
            members,
            rec,
            cache_sha256=cache_sha,
            expected_members=a.expected_members,
            truth_weights=a.truth_weights,
        )
    except ValueError as e:
        sys.exit(str(e))
    out.update(member_dirs=a.member_dirs, gp_record=a.gp_record)
    with open(a.out, "x") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
