#!/usr/bin/env python3
"""Refit the dN/dX-to-w_c map (delta_c(z) and Xbar(z)) from a release training cache and write it as a stamped product.

The estimators are those of scripts/calibrate_delta_c.py (delta_c) and closure_legb.hcd_xbar_polyfit (Xbar), applied to
a format-3.4 cache read through the strict loader (hcd_analysis/emulator/wc_map.py). The output is a small JSON file
with a hcd_priya.lineage/1 record naming the cache's sha256; a model bundle (format hcd_priya.h5/2) ships it. Runs from a
clean committed checkout only (otherwise the record is not release-eligible and nothing is written); never overwrites.

  python scripts/refit_wc_map.py --cache CACHE.h5 --cache-sha256 HEX --out NEW_FILE.json
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis.emulator import wc_map as W  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="the release training cache (format 3.4)")
    ap.add_argument("--cache-sha256", required=True, help="its expected sha256 (checked at load)")
    ap.add_argument("--out", required=True, help="the NEW map file (refused if it exists)")
    a = ap.parse_args(argv)
    from hcd_analysis.tau0_cache import CacheLineageError

    try:
        rec, sha = W.build(a.cache, a.out, expected_cache_sha256=a.cache_sha256)
    except (W.WcMapError, CacheLineageError) as e:  # the map, or the strict cache loader, refused
        print(f"refused: {e}", file=sys.stderr)
        return 2
    m = W.read_wc_map(a.out, expected_sha256=sha)
    print(W.dumps_summary(m))
    print(f"wrote {a.out} sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
