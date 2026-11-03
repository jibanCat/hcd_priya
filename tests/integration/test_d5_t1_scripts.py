"""T1 (b) and (c) on the accepted production members (integration, real data): the acceptance tool reproduces the
recorded report at leaf level (every pre-existing leaf identical except ``format``; the new content-check entries
present and true), and a re-export of the members reproduces the published manifest once its created_utc and
export commits are substituted back and the added ``provenance.eligibility_checks`` is removed; every member file and
the map are byte-identical."""
import json
import os

from tests.integration._config import CFG, REPO, flat, needs_data, path, require_clean_checkout, run, sha256

pytestmark = needs_data
NEW_CHECKS = {"training_set_declared_complete", "split_recomputed_equal", "normaliser_fingerprint_equal",
              "wc_map_refit_equal"}


def test_t1b_the_acceptance_report_at_leaf_level(tmp_path):
    p = CFG["production"]
    ref_path = path(p["acceptance_report"])
    assert sha256(ref_path) == p["acceptance_report_sha256"]
    args = [f"{REPO}/scripts/accept_retrained_ensemble.py"]
    for n in p["member_names"]:
        args += ["--member", os.path.join(path(p["members_dir"]), n)]
    args += ["--cache", path(CFG["cache"]), "--cache-sha256", CFG["cache_sha256"], "--wc-map", path(p["wc_map"]),
             "--wc-map-sha256", p["wc_map_sha256"]]
    for h in p["old_hists"]:
        args += ["--old-hist", path(h)]
    args += ["--old-source", path(p["old_source"]), "--old-source-sha256", p["old_source_sha256"]]
    for h in p["historical_prefixes"]:
        args += ["--historical-prefix", path(h)]
    out = tmp_path / "ensemble_acceptance.json"
    r = run(args + ["--out", str(out)], cwd=tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    old, new = json.load(open(ref_path)), json.load(open(out))
    a, b = flat(old), flat(new)
    differing = sorted(k for k in a if k != "/format" and a[k] != b.get(k, "<absent>"))
    assert not differing, differing
    assert old["format"] == "hcd_priya.ensemble_acceptance/1" and new["format"] == "hcd_priya.ensemble_acceptance/2"
    assert set(new["checks"]) - set(old["checks"]) == NEW_CHECKS and all(new["checks"][k] for k in NEW_CHECKS)
    assert set(new) - set(old) == {"content_check_environment", "content_checks"} and new["passed"]
    assert {c["component"]: c["outcome"] for c in new["content_checks"]} == {
        "cache": "PASS", "split": "PASS", "normaliser": "PASS", "wc_map": "PASS"}


def test_t1c_the_re_export_reproduces_the_published_manifest(tmp_path):
    code = require_clean_checkout()
    p = CFG["production"]
    args = [f"{REPO}/scripts/export_safe_bundle.py", "--out", str(tmp_path / "out")]
    for n in p["member_names"]:
        args += ["--member", os.path.join(path(p["members_dir"]), n)]
    args += ["--sha256sums", path(p["sha256sums"]), "--sha256sums-sha256", p["sha256sums_sha256"], "--cache",
             path(CFG["cache"]), "--cache-sha256", CFG["cache_sha256"], "--wc-map", path(p["wc_map"]),
             "--wc-map-sha256", p["wc_map_sha256"]]
    r = run(args, cwd=tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    bundle = tmp_path / "out" / "bundle"
    ref_dir = path(p["bundle"])
    ref = json.load(open(os.path.join(ref_dir, "manifest.json")))
    new = json.load(open(bundle / "manifest.json"))
    ec = new["provenance"].pop("eligibility_checks")
    assert ec["passed"] and ec["read_back"] and {c["outcome"] for c in ec["components"]} == {"PASS"}
    assert new["provenance"]["export_code"] == code and new["lineage"]["export_code"] == code
    new["provenance"]["created_utc"] = ref["provenance"]["created_utc"]
    new["provenance"]["export_code"]["commit"] = ref["provenance"]["export_code"]["commit"]
    new["lineage"]["export_code"]["commit"] = ref["lineage"]["export_code"]["commit"]
    raw = json.dumps(new, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == p["manifest_sha256"]
    for f in ref["files"]:
        assert sha256(bundle / f["path"]) == f["sha256"] == sha256(os.path.join(ref_dir, f["path"])), f["path"]
