#!/usr/bin/env python3
"""Promote a staged carried cache (scripts/carry_validated_cache.py) once the pre-registered fresh-row comparison
(scripts/fresh_row_sample.py compare) has passed: the only step that makes carried rows readable by the release reader.

The staged cache is read with tau0_cache.read_cache_staged (every release check except the promotion). The comparison
report must hash to the sha256 given on the command line, say PASSED for every group, and name the staged file by its
sha256. The promoted cache holds the staged datasets and attributes unchanged (so the same payload digest) and the
staged record, plus product.carried.promoted = {passed, comparison_sha256, staged_sha256, prereg_sha256}. Runs from a
clean, committed checkout; never overwrites; the staged file is left in place.

  python scripts/promote_carried_cache.py --staged STAGED.h5 --comparison REPORT.json --comparison-sha256 HEX \\
      --output PROMOTED.h5
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402


class PromotionRefused(RuntimeError):
    pass


def promote(staged, comparison, comparison_sha256, output):
    staged, comparison, output = Path(staged), Path(comparison), Path(output)
    if output.exists():
        raise PromotionRefused(f"{output} exists; a cache is never overwritten")
    code = L.code_identity()
    if not (code.get("commit") and code.get("dirty") is False):
        raise PromotionRefused(f"promotion runs from a clean, committed checkout only (code identity {code})")
    got = L.sha256_file(comparison)
    if got != comparison_sha256:
        raise PromotionRefused(f"{comparison}: sha256 {got} is not the pinned {comparison_sha256}")
    rep = json.loads(comparison.read_text())
    ident = L.file_identity(staged)
    d, attrs, rec = T.read_cache_staged(staged)
    if os.stat(staged).st_mtime_ns != ident["mtime_ns"] or L.sha256_file(staged) != ident["sha256"]:
        raise PromotionRefused(f"{staged} changed while it was read")
    car = rec["product"].get("carried")
    if not car or any(g["rows"] != T.ROWS_CARRIED for g in rec["product"]["groups"]):
        raise PromotionRefused(f"{staged}: not a carried cache")
    if car.get("promoted"):
        raise PromotionRefused(f"{staged}: already promoted")
    groups = rep.get("groups") or []
    if not (rep.get("passed") is True and groups and all(g.get("ok") is True for g in groups)):
        raise PromotionRefused(f"{comparison}: the fresh-row comparison did not pass")
    if (rep.get("carried") or {}).get("sha256") != ident["sha256"]:
        raise PromotionRefused(f"{comparison}: compares the carried cache {(rep.get('carried') or {}).get('sha256')},"
                               f" not {staged} ({ident['sha256']})")
    staged_groups = {(g["sim"], int(g["snap"])) for g in rec["product"]["groups"]}
    if not {(g["sim"], int(g["snap"])) for g in groups} <= staged_groups:
        raise PromotionRefused(f"{comparison}: compares groups that are not in {staged}")
    prereg = (rep.get("prereg") or {}).get("sha256")
    promoted = {"passed": True, "comparison_sha256": got, "staged_sha256": ident["sha256"], "prereg_sha256": prereg,
                "n_groups_compared": len(groups)}
    product = rec["product"]
    out = T.write_cache(output, d, {k: v for k, v in attrs.items() if k != "cache_version"},
                        inputs=rec["inputs"], settings=rec["settings"], groups=product["groups"],
                        carried=dict(car, promoted=promoted), validation=rec.get("validation"),
                        selection=product.get("selection"))
    if out["product"]["payload_sha256"] != product["payload_sha256"]:
        output.unlink()
        raise PromotionRefused("the promoted payload differs from the staged one (nothing kept)")
    if not out["release_eligible"]:
        output.unlink()
        raise PromotionRefused("the promoted cache would not be release-eligible: " + "; ".join(out["ineligible_reasons"]))
    T.read_cache(output)                                      # the release reader accepts it
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True, help="fresh_row_sample.py compare's report")
    ap.add_argument("--comparison-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        promote(a.staged, a.comparison, a.comparison_sha256, a.output)
    except (PromotionRefused, T.CacheLineageError) as e:
        print(f"PROMOTION REFUSED: {e}")
        return 2
    print(f"wrote {a.output} sha256 {L.sha256_file(a.output)} (promoted from {a.staged})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
