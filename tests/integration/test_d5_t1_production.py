"""T1 (a) and (d) on the accepted production bundle (integration, real data; tests/integration/_config.py): the
public predictor reproduces the pre-change predictions bitwise in fresh processes (the preserved D5-R2 probe and the
widened probe, both pinned by digest), and the production content checks PASS on every component with the golden
digests of the pre-change recomputation (T31)."""

import json

import numpy as np

from tests.integration._config import CFG, REPO, needs_data, path, run, sha256

pytestmark = needs_data


def _probe(tmp_path, which, probe):
    p = CFG["production"]
    base = CFG["step0"][which]
    assert sha256(probe) == base["probe_sha256"], f"{probe} is not the step-0 probe"
    npz, js = tmp_path / f"{which}.npz", tmp_path / f"{which}.json"
    r = run([probe, REPO, path(p["bundle"]), p["manifest_sha256"], str(npz), str(js)], cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    return sha256(npz), json.load(open(js)), base


def test_t1a_the_preserved_probe_is_bitwise_the_baseline(tmp_path):
    npz, s, base = _probe(
        tmp_path, "t1_preserved", f"{REPO}/tests/integration/t1_probe_preserved.py"
    )
    assert npz == base["npz_sha256"] and s["arrays_sha256"] == base["arrays_sha256"]
    assert s["lineage_canonical_sha256"] == base["lineage_canonical_sha256"] and s["n_arrays"] == 36


def test_t1a_the_widened_probe_is_bitwise_the_baseline(tmp_path):
    npz, s, base = _probe(tmp_path, "t1_widened", f"{REPO}/tests/integration/t1_probe_widened.py")
    assert npz == base["npz_sha256"] and s["arrays_sha256"] == base["arrays_sha256"]
    assert s["stable_canonical_sha256"] == base["stable_canonical_sha256"]


def test_t1d_the_production_content_check_passes_with_the_golden_digests():
    from hcd_analysis.emulator import eligibility as E
    from hcd_analysis.emulator import safe_h5 as H
    from hcd_analysis.emulator.data import load_cache

    p, g = CFG["production"], CFG["step0"]["production_content"]
    rb = H.load_release(path(p["bundle"]), expected_manifest_sha256=p["manifest_sha256"])
    members = H.load_bundle(path(p["bundle"]), expected_manifest_sha256=p["manifest_sha256"])
    d = load_cache(path(CFG["cache"]))
    c = E.production_content_check(
        [m for _, m, _ in members],
        [n for _, _, n in members],
        rb.wc_map,
        cache=d,
        cache_path=path(CFG["cache"]),
        expected_cache_sha256=CFG["cache_sha256"],
    )
    by = {x["component"]: x for x in c["components"]}
    assert c.passed and {k: v["outcome"] for k, v in by.items()} == {
        "cache": "PASS",
        "split": "PASS",
        "normaliser": "PASS",
        "wc_map": "PASS",
    }
    sp = by["split"]["detail"]
    assert (sp["n_train"], sp["n_val"], sp["train_idx_sha256"], sp["val_idx_sha256"]) == (
        g["n_train"],
        g["n_val"],
        g["train_idx_sha256"],
        g["val_idx_sha256"],
    )
    assert by["normaliser"]["detail"]["recomputed_tensors_sha256"] == g["normaliser_tensors_sha256"]
    assert (
        by["normaliser"]["n_tensors"] == by["normaliser"]["n_bitwise"]
        and len(by["normaliser"]["detail"]["members"]) == 5
    )
    assert by["wc_map"]["detail"]["refit_payload_sha256"] == g["wc_refit_payload_sha256"]
    assert by["wc_map"]["detail"]["fitted_on_sims_sha256"] == g["all_sims_sha256"]
    assert np.isfinite(by["normaliser"]["max_rel"]) and json.dumps(dict(c), allow_nan=False)
