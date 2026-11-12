#!/usr/bin/env python3
"""Promote a staged carried cache (scripts/carry_validated_cache.py) once the pre-registered fresh-row comparison
(scripts/fresh_row_sample.py compare) has passed: the only step that makes carried rows readable by the release reader.

The staged cache is read with tau0_cache.read_cache_staged (every release check except the promotion). The comparison
report must hash to the sha256 given on the command line and be bound to the pre-registration of record
(PREREG_OF_RECORD_SHA256): the report names that pre-registration; the file given with --prereg hashes to it and is
re-read; it names the carried cache's source (``historical_cache.sha256`` equal to ``carried.source_sha256``; a report
that records ``source_sha256`` must agree too); the report's criterion is the pre-registration's (rtol 1e-12); its
groups are exactly the pre-registered groups, each passed and built from the same rebuilt catalogue; it says PASSED,
names the staged file by its sha256, lists every fresh file with its sha256 (each file re-hashed here), and was written
from a clean, committed code identity. The promoted cache holds the staged datasets and attributes unchanged (so the
same payload digest) and the staged record, plus product.carried.promoted = {passed, comparison_sha256, staged_sha256,
prereg_sha256, fresh_sha256, n_groups_compared}. Runs from a clean, committed checkout; never overwrites; the staged
file is left in place.

  python scripts/promote_carried_cache.py --staged STAGED.h5 --comparison REPORT.json --comparison-sha256 HEX \\
      --prereg PREREGISTRATION.json --output PROMOTED.h5
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fresh_row_sample as FR  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402

# The pre-registration of record of the fresh-row sample (written by fresh_row_sample.py select before any fresh
# rebuild): its 8 groups and its criterion are what a promotion must have been compared on.
PREREG_OF_RECORD_SHA256 = "d26b210c1a1249eb60e475ab9c9d28f9851c57e35bc2536c9e78506b6c132ee0"


class PromotionRefused(RuntimeError):
    pass


def _hex64(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def check_report(rep, prereg_path, car, staged_sha256):
    """Refuse (PromotionRefused) a comparison report that is not bound to the pre-registration of record and to this
    staged carry (module docstring). Returns ``(pre-registration sha256, sorted fresh sha256s)``."""
    pre_ref = rep.get("prereg") or {}
    if pre_ref.get("sha256") != PREREG_OF_RECORD_SHA256:
        raise PromotionRefused(
            f"the report compares against the pre-registration {pre_ref.get('sha256')!r}, not the "
            f"one of record {PREREG_OF_RECORD_SHA256}"
        )
    if L.sha256_file(prereg_path) != PREREG_OF_RECORD_SHA256:
        raise PromotionRefused(f"{prereg_path} is not the pre-registration of record")
    pre = json.loads(Path(prereg_path).read_text())
    if pre.get("format") != FR.FORMAT or (pre.get("criterion") or {}).get("rtol") != FR.RTOL:
        raise PromotionRefused(f"{prereg_path}: not a {FR.FORMAT} file with rtol {FR.RTOL}")
    source = (pre.get("historical_cache") or {}).get("sha256")
    if source != car.get("source_sha256"):
        raise PromotionRefused(
            f"the pre-registration names the source {source}, the carry's source is "
            f"{car.get('source_sha256')}"
        )
    if "source_sha256" in rep and rep["source_sha256"] != car.get("source_sha256"):
        raise PromotionRefused(
            f"the report names the source {rep['source_sha256']}, not the carry's"
        )
    if rep.get("criterion") != pre["criterion"]:
        raise PromotionRefused("the report's criterion is not the pre-registered one")
    want = sorted((str(g["sim"]), int(g["snap"])) for g in pre.get("groups") or [])
    groups = rep.get("groups") or []
    got = sorted((str(g.get("sim")), int(g.get("snap", -1))) for g in groups)
    if not want or got != want:
        raise PromotionRefused(
            f"the report compares {len(got)} groups, not exactly the {len(want)} pre-registered"
        )
    if not (
        rep.get("passed") is True
        and all(g.get("ok") is True and g.get("same_rebuilt_catalogue") is True for g in groups)
    ):
        raise PromotionRefused(
            "the fresh-row comparison did not pass for every pre-registered group"
        )
    if (rep.get("carried") or {}).get("sha256") != staged_sha256:
        raise PromotionRefused(
            f"the report compares the carried cache {(rep.get('carried') or {}).get('sha256')}, "
            f"not the staged file ({staged_sha256})"
        )
    fresh = rep.get("fresh") or []
    if not fresh or not all(
        isinstance(f, dict) and _hex64(f.get("sha256")) and f.get("path") for f in fresh
    ):
        raise PromotionRefused("the report does not list its fresh files with their sha256")
    listed = {str(f["path"]) for f in fresh}
    if any(str(g.get("fresh_file")) not in listed for g in groups):
        raise PromotionRefused("a compared group's fresh file is not in the report's fresh list")
    for f in fresh:
        if not os.path.isfile(f["path"]) or L.sha256_file(f["path"]) != f["sha256"]:
            raise PromotionRefused(
                f"the fresh file {f['path']} is missing or not the one the report lists"
            )
    code = rep.get("code") or {}
    if (
        not (isinstance(code.get("commit"), str) and len(code["commit"]) == 40)
        or code.get("dirty") is not False
    ):
        raise PromotionRefused(
            f"the report was not written from a clean, committed checkout (code {code})"
        )
    return PREREG_OF_RECORD_SHA256, sorted(f["sha256"] for f in fresh)


def promote(staged, comparison, comparison_sha256, output, prereg):
    staged, comparison, output = Path(staged), Path(comparison), Path(output)
    if output.exists():
        raise PromotionRefused(f"{output} exists; a cache is never overwritten")
    code = L.code_identity()
    if not (code.get("commit") and code.get("dirty") is False):
        raise PromotionRefused(
            f"promotion runs from a clean, committed checkout only (code identity {code})"
        )
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
    prereg_sha, fresh_shas = check_report(rep, prereg, car, ident["sha256"])
    groups = rep["groups"]
    staged_groups = {(g["sim"], int(g["snap"])) for g in rec["product"]["groups"]}
    if not {(g["sim"], int(g["snap"])) for g in groups} <= staged_groups:
        raise PromotionRefused(f"{comparison}: compares groups that are not in {staged}")
    promoted = {
        "passed": True,
        "comparison_sha256": got,
        "staged_sha256": ident["sha256"],
        "prereg_sha256": prereg_sha,
        "fresh_sha256": fresh_shas,
        "n_groups_compared": len(groups),
    }
    product = rec["product"]
    out = T.write_cache(
        output,
        d,
        {k: v for k, v in attrs.items() if k != "cache_version"},
        inputs=rec["inputs"],
        settings=rec["settings"],
        groups=product["groups"],
        carried=dict(car, promoted=promoted),
        validation=rec.get("validation"),
        selection=product.get("selection"),
    )
    if out["product"]["payload_sha256"] != product["payload_sha256"]:
        output.unlink()
        raise PromotionRefused("the promoted payload differs from the staged one (nothing kept)")
    if not out["release_eligible"]:
        output.unlink()
        raise PromotionRefused(
            "the promoted cache would not be release-eligible: "
            + "; ".join(out["ineligible_reasons"])
        )
    T.read_cache(output)  # the release reader accepts it
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument(
        "--comparison", type=Path, required=True, help="fresh_row_sample.py compare's report"
    )
    ap.add_argument("--comparison-sha256", required=True)
    ap.add_argument(
        "--prereg", type=Path, required=True, help="the pre-registration of record (re-read here)"
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        promote(a.staged, a.comparison, a.comparison_sha256, a.output, a.prereg)
    except (PromotionRefused, T.CacheLineageError) as e:
        print(f"PROMOTION REFUSED: {e}")
        return 2
    print(f"wrote {a.output} sha256 {L.sha256_file(a.output)} (promoted from {a.staged})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
