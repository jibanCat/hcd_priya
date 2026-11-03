"""T4 (integration, real data): a real W-1 fold pair through scripts/export_research_bundle.py into a temporary
directory: the research loaders load it, its members are bitwise the training checkpoints, its w_c map is the
recorded W-1 fold map, every content check passes, and every production loader refuses it."""
import json
import os

import pytest

from tests.integration._config import CFG, REPO, needs_data, path, require_clean_checkout, run, sha256

pytestmark = needs_data


@pytest.mark.parametrize("k", CFG["w1"]["t4_folds"] if CFG else [0])
def test_t4_a_real_fold_pair_through_the_research_export(tmp_path, k):
    require_clean_checkout()
    w1 = CFG["w1"]
    names = [f"w1_b0_fold{k}_seed{s}" for s in (0, 1)]
    sums = tmp_path / "SHA256SUMS"                                   # the recorded digests, checked here first
    with open(sums, "w") as fh:
        for n in names:
            for ext, want in w1["members"][n].items():
                assert sha256(os.path.join(path(w1["b0_dir"]), f"{n}.{ext}")) == want
                fh.write(f"{want}  {n}.{ext}\n")
    held = sorted(json.load(open(w1["fold_file"]))["folds"][f"fold{k}"])
    args = [f"{REPO}/scripts/export_research_bundle.py", "--out", str(tmp_path / "out"), "--purpose",
            f"W-1 fold {k} cross-validation (integration test)", "--split-rule", "multi_holdout_row_split/1",
            "--fold", str(k), "--folds-digest", w1["fold_file_sha256"], "--fold-file", w1["fold_file"],
            "--fold-file-sha256", w1["fold_file_sha256"], "--sha256sums", str(sums), "--sha256sums-sha256",
            sha256(sums), "--cache", path(CFG["cache"]), "--cache-sha256", CFG["cache_sha256"]]
    for s in held:
        args += ["--held-out-sim", s]
    for n in names:
        args += ["--member", os.path.join(path(w1["b0_dir"]), n)]
    r = run(args, cwd=tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s = json.load(open(tmp_path / "out" / "evidence" / "research_export_summary.json"))
    assert s["verdict"] == "PASS" and all(s["checks"].values())
    assert s["checks"]["ensemble_P_filt"] and s["checks"]["production_loaders_refuse"]
    from hcd_analysis.emulator import safe_h5 as H
    rb = H.load_research(str(tmp_path / "out" / "bundle"), expected_manifest_sha256=s["bundle_manifest_sha256"],
                         purpose="integration check")
    assert {c["component"]: c["outcome"] for c in rb.research["content_checks"]} == {
        "cache": "PASS", "split": "PASS", "normaliser": "PASS", "wc_map": "PASS"}
    rec = json.load(open(path(w1["summary"])))["folds_maps"][str(k)]
    assert rb.wc_map.doc["delta_c"]["coeffs"] == rec["delta_c_coeffs"]
    assert rb.wc_map.doc["xbar"]["coeffs"] == rec["xbar_coeffs"] and rb.wc_map.doc["delta_c"]["n_sims"] == rec["n_sims"]
    assert rb.research["wc_map"]["fitted_on_sims_sha256"] == CFG["step0"]["w1_fold_maps"][str(k)][
        "fitted_on_sims_sha256"]
    assert rb.split["held_out_sims"] == held and set(rb.research["restored_epoch"]) == set(names)
