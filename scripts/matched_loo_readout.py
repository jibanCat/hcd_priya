#!/usr/bin/env python3
"""The pre-registered paired read-out of the matched leave-one-simulation-out comparison (comparison/loo_readout.py)
over the fold files of scripts/eval_matched_loo_fold.py. NumPy and SciPy only. Writes a NEW json.

``--expected-folds`` lists every held-out simulation of the campaign (one name per line): a fold without a file (a
job stopped at its time limit, or never run) is reported as NOT_RUN and counts against completeness, never dropped.

  python scripts/matched_loo_readout.py --expected-folds SIMS.txt --folds F1.json F2.json ... --out NEW.json
      [--n-boot 10000]
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis.comparison import loo_readout as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--expected-folds", required=True, help="text file: every held-out simulation, one per line"
    )
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="NEW json (refused if it exists)")
    ap.add_argument("--n-boot", type=int, default=R.N_BOOT)
    ap.add_argument(
        "--truth-weights-context",
        action="store_true",
        help="read out fold files made with --truth-weights (context only: uses held-out truth)",
    )
    a = ap.parse_args()
    want = "truth (context only)" if a.truth_weights_context else "head_a_per_fold"
    if os.path.lexists(a.out):
        sys.exit(f"refusing: {a.out} exists")
    folds = []
    for p in a.folds:
        with open(p) as f:
            rec = json.load(f)
        if rec.get("weights") != want:
            sys.exit(f"{p}: weights {rec.get('weights')!r}, expected {want!r}")
        folds.append(
            {
                k: (np.asarray(v, dtype=np.float64) if k in ("e_c1", "e_gp", "k_skm", "z") else v)
                for k, v in rec.items()
            }
        )
    with open(a.expected_folds) as f:
        expected = [x.strip() for x in f if x.strip()]
    if len(set(expected)) != len(expected):
        sys.exit("the expected-fold list names a simulation twice")
    out = R.readout(folds, expected=expected, n_boot=a.n_boot)
    out["fold_files"] = a.folds
    out["weights"] = want
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    print(out["verdict"], out["band_averaged"])


if __name__ == "__main__":
    main()
