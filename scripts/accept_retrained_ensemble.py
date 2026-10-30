#!/usr/bin/env python3
"""The acceptance of a retrained ensemble before it is exported: the pre-registered member checks.

For every member checkpoint prefix (``.eqx``, ``.meta.json``, ``.norm.pkl``, ``.hist.json``):
  * the meta records a full commit (``git_sha``, 40 hex), the same for every member, and ``code_dirty`` false;
  * ``cache_sha256`` is the release cache's (``--cache-sha256``) and ``cache_lineage_sha256`` the digest of the lineage
    record stored in that cache file (read from the file whose sha256 is checked; the record is parsed and checked);
  * ``cache_path`` is a file name, the seeds are distinct, the recipe and architecture are the same for every member;
  * the basename and directory are not the historical ensembles' (``--historical-prefix``, repeatable);
  * training completed under the recipe's stopping rule: the history holds finite losses, all three stages, and at most
    the recipe's epochs;
  * the validation error is not worse than the old ensemble's beyond its seed spread: for each ``--metric`` (a history
    key; the best, i.e. smallest, value per member), mean(new) <= mean(old) + std(old) (ddof 1) over the members, with
    the old ensemble's histories given by ``--old-hist`` (repeatable). The old-to-new change is reported per metric.

Writes a JSON report; exit 0 when every check passes, 1 otherwise.

    python scripts/accept_retrained_ensemble.py --member DIR/NAME_seed0 [...] --cache CACHE.h5 --cache-sha256 HEX \\
        --old-hist OLD_seed0.hist.json [...] --metric val_resid_loss --metric val_loss \\
        --historical-prefix OLD_DIR/prod_repaired --out REPORT.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.catalog import RELEASE_FINDER_IDS  # noqa: E402

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def cache_record_digest(path, sha256):
    """The digest of the lineage record stored in the cache file whose sha256 is ``sha256`` (record parsed, checked
    and eligible; the payload itself was checked by the strict loader when the members were trained)."""
    import h5py
    if L.sha256_file(path) != sha256:
        raise SystemExit(f"refusing: {path} does not hash to {sha256}")
    with h5py.File(path, "r") as f:
        rec = L.check_record(L.parse_record(f["lineage_json"].asstr()[()]), RELEASE_FINDER_IDS)
    if rec["kind"] != "tau0_cache" or not rec["release_eligible"]:
        raise SystemExit(f"refusing: {path} does not hold an eligible tau0_cache record")
    return L.record_sha256(rec)


def best(hist, metric):
    v = np.asarray(hist.get(metric, []), dtype=float)
    return float(np.min(v)) if v.size and np.all(np.isfinite(v)) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cache-sha256", required=True)
    ap.add_argument("--old-hist", action="append", required=True)
    ap.add_argument("--metric", action="append", required=True)
    ap.add_argument("--historical-prefix", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing: {a.out} exists")
    rec_sha = cache_record_digest(a.cache, a.cache_sha256)
    checks, members = {}, []
    hist_names = {os.path.basename(p) for p in a.historical_prefix}
    hist_dirs = {os.path.realpath(os.path.dirname(p)) for p in a.historical_prefix}
    for p in a.member:
        meta = json.load(open(p + ".meta.json"))
        hist = json.load(open(p + ".hist.json"))
        members.append((p, meta, hist))
    metas = [m for _, m, _ in members]
    m0 = metas[0]
    checks["git_sha_full_and_one"] = all(isinstance(m.get("git_sha"), str) and _HEX40.match(m["git_sha"])
                                         for m in metas) and len({m["git_sha"] for m in metas}) == 1
    checks["code_clean"] = all(m.get("code_dirty") is False for m in metas)
    checks["cache_sha256_is_the_release_cache"] = all(m.get("cache_sha256") == a.cache_sha256 for m in metas)
    checks["cache_lineage_record"] = all(m.get("cache_lineage_sha256") == rec_sha for m in metas)
    checks["cache_path_is_a_file_name"] = all(isinstance(m.get("cache_path"), str) and os.sep not in m["cache_path"]
                                              for m in metas)
    checks["seeds_distinct"] = len({m.get("seed") for m in metas}) == len(metas)
    checks["one_recipe_and_architecture"] = all(m.get("recipe") == m0.get("recipe") and m.get("arch_cfg") ==
                                                m0.get("arch_cfg") for m in metas)
    stems = {re.sub(r"_seed\d+\Z", "", os.path.basename(p)) for p in a.member}
    checks["distinct_basename_and_directory"] = not (stems & {re.sub(r"_seed\d+\Z", "", n) for n in hist_names}) and \
        not ({os.path.realpath(os.path.dirname(p)) for p in a.member} & hist_dirs)
    epochs = (m0.get("recipe") or {}).get("epochs")
    done = []
    for p, meta, hist in members:
        losses = [np.asarray(hist.get(k, []), dtype=float) for k in ("train_loss", "val_loss", "val_resid_loss")]
        done.append(all(v.size and np.all(np.isfinite(v)) for v in losses)
                    and set(int(s) for s in hist.get("stage", [])) == {1, 2, 3}
                    and (epochs is None or len(hist["train_loss"]) <= int(epochs)))
    checks["training_completed_under_the_stopping_rule"] = all(done)
    old = [json.load(open(h)) for h in a.old_hist]
    comparison = {}
    for metric in a.metric:
        new_b = [best(h, metric) for _, _, h in members]
        old_b = [best(h, metric) for h in old]
        ok = None not in new_b and None not in old_b and len(old_b) > 1
        if ok:
            mo, so, mn = float(np.mean(old_b)), float(np.std(old_b, ddof=1)), float(np.mean(new_b))
            ok = mn <= mo + so
            comparison[metric] = {"new_best": new_b, "old_best": old_b, "new_mean": mn, "old_mean": mo,
                                  "old_std": so, "new_std": float(np.std(new_b, ddof=1)) if len(new_b) > 1 else None,
                                  "change_new_minus_old": mn - mo, "change_in_old_std": (mn - mo) / so if so else None,
                                  "ok": ok}
        else:
            comparison[metric] = {"ok": False, "error": "a history lacks the metric, holds non-finite values, or the "
                                                        "old ensemble has fewer than two members"}
        checks[f"validation_{metric}_within_old_seed_spread"] = bool(ok)
    passed = all(checks.values())
    report = {"format": "hcd_priya.ensemble_acceptance/1", "passed": passed, "checks": checks,
              "members": [os.path.basename(p) for p in a.member], "cache_sha256": a.cache_sha256,
              "cache_lineage_sha256": rec_sha, "git_sha": m0.get("git_sha"), "validation": comparison,
              "rule": "mean over members of the best value of each metric: new <= old + std(old, ddof 1)"}
    with open(a.out, "x") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    for k, v in comparison.items():
        if "new_mean" in v:
            print(f"  {k}: old {v['old_mean']:.6g} +- {v['old_std']:.3g}, new {v['new_mean']:.6g} "
                  f"(change {v['change_new_minus_old']:+.3g})")
    print(f"verdict: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
