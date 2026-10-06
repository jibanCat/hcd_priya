#!/usr/bin/env python3
"""(Re)build or --check the product manifest (gate E amendment A1 rev 1 section 7): the sha256 of the MF, DLA-core, T1,
T3 (and, once amendment A2 registers it, T2) products, against the production training cache and the committed
production ensemble manifest. Never hand-edited; regenerated here and committed with the analysis.lock.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/gen_product_manifest.py --mf <..> --dla-core <..> --t1 <..> --t3 <..> [--t2 <..>] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from hcd_analysis.emulator import prod_ensemble as PE  # noqa: E402
from hcd_analysis.emulator import product_manifest as PM  # noqa: E402

DEFAULT_OUT = os.path.join(_REPO_ROOT, "checkpoints", "product_manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mf")
    ap.add_argument("--dla-core")
    ap.add_argument("--t1")
    ap.add_argument("--t3")
    ap.add_argument("--t2", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    ens = json.load(open(PE.DEFAULT_MANIFEST_PATH))
    cache_sha = ens["cache_sha256"]
    ens_sha = PM.sha256_file(PE.DEFAULT_MANIFEST_PATH)
    if a.check:
        paths = PM.verify(a.out, cache_sha256=cache_sha, ensemble_manifest_sha256=ens_sha)
        print(f"[gen_product_manifest --check] OK: {json.dumps(paths)}")
        return 0
    m = PM.build({"mf": a.mf, "dla_core": a.dla_core, "cemu_t1": a.t1, "cemu_t3": a.t3, "cemu_t2": a.t2},
                 cache_sha256=cache_sha, ensemble_manifest_sha256=ens_sha)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    print(f"[gen_product_manifest] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
