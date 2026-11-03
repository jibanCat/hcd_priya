"""The real metas and the 20 W-1 fold members (integration, real data): the reserved-key rule hits every W-1 fold
member and none of the production metas; the library multi-held-out split recomputes every W-1 member's recorded
sizes and fold lists (T28); every member's normaliser is the recomputation on its training rows (the fingerprint);
the library training-set refit reproduces each of the ten recorded W-1 fold maps bitwise (T33); and the members are
admitted, as they are, to a research bundle of their fold (T32)."""
import json
import os

import numpy as np
import pytest

from tests.integration._config import CFG, needs_data, path, sha256

pytestmark = needs_data


@pytest.fixture(scope="module")
def cache():
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(path(CFG["cache"]))
    assert d["cache_sha256"] == CFG["cache_sha256"]
    return d


def _w1(name):
    """A W-1 B0 member, its files checked against the recorded digests before anything is unpickled."""
    from hcd_analysis.emulator import train as T
    pre = os.path.join(path(CFG["w1"]["b0_dir"]), name)
    for ext, want in CFG["w1"]["members"][name].items():
        assert sha256(f"{pre}.{ext}") == want, (name, ext)
    return T.load_checkpoint(pre)


def test_the_reserved_keys_hit_every_fold_member_and_no_production_meta():
    from hcd_analysis.emulator import eligibility as E
    from hcd_analysis.emulator import safe_h5 as H
    p = CFG["production"]
    for _, meta, _ in H.load_bundle(path(p["bundle"]), expected_manifest_sha256=p["manifest_sha256"]):
        assert E.production_recipe_reasons(meta) == []
    for n in p["member_names"]:
        assert E.production_recipe_reasons(json.load(open(os.path.join(path(p["members_dir"]), n + ".meta.json")))) == []
    for name in CFG["w1"]["members"]:
        meta = json.load(open(os.path.join(path(CFG["w1"]["b0_dir"]), name + ".meta.json")))
        codes = {E.reason_code(r) for r in E.production_recipe_reasons(meta)}
        assert {"RSV_KEY:recipe.holdout_sims", "RSV_KEY:recipe.fold", "RSV_KEY:recipe.folds_sha256",
                "ALL_SIMS_NOT_TRUE"} <= codes, name


def test_every_w1_member_split_normaliser_map_and_admission(cache):
    from hcd_analysis.emulator import data as D
    from hcd_analysis.emulator import eligibility as E
    from hcd_analysis.emulator import wc_map as W
    w1 = CFG["w1"]
    assert sha256(w1["fold_file"]) == w1["fold_file_sha256"] and sha256(path(w1["summary"])) == w1["summary_sha256"]
    folds = json.load(open(w1["fold_file"]))["folds"]
    maps = json.load(open(path(w1["summary"])))["folds_maps"]
    sim = np.asarray(cache["sim_name"]).astype(str)
    from hcd_analysis.emulator import safe_h5 as H
    p = CFG["production"]
    rb = H.load_release(path(p["bundle"]), expected_manifest_sha256=p["manifest_sha256"])
    n_groups, doc_full_blocks = rb.lineage["cache"]["n_groups"], rb.wc_map.doc["delta_c"]["n_blocks"]
    assert H.wc_map_production_reasons(rb.lineage, rb.wc_map) == []
    for k in range(len(folds)):
        held = sorted(folds[f"fold{k}"])
        pair = [_w1(f"w1_b0_fold{k}_seed{s}") for s in (0, 1)]
        metas = [m for _, m, _ in pair]
        r = metas[0]["recipe"]
        tr, va, ho = D.multi_holdout_row_split(sim, r["holdout_sims"], val_seed=r["val_seed"], val_frac=r["val_frac"])
        for s, meta in zip((0, 1), metas):                                           # T28
            rr = meta["recipe"]
            assert (rr["n_train"], rr["n_val"]) == (tr.size, va.size) and sorted(rr["holdout_sims"]) == held
            assert rr["fold"] == k and rr["folds_sha256"] == w1["fold_file_sha256"]
            base = CFG["step0"]["w1_splits"][f"w1_b0_fold{k}_seed{s}"]
            assert (E.index_sha256(tr), E.index_sha256(va)) == (base["train_idx_sha256"], base["val_idx_sha256"])
        nc = E.normaliser_component(cache, [n for _, _, n in pair], tr, training_environment=None,
                                    current_environment=E.environment())
        assert nc["outcome"] == "PASS" and nc["detail"]["recomputed_tensors_sha256"] == \
            CFG["step0"]["w1_splits"][f"w1_b0_fold{k}_seed0"]["normaliser_tensors_sha256"]
        train = sorted(set(sim.tolist()) - set(held))
        doc = W.refit_subset(cache, train)                                          # T33
        rec = maps[str(k)]
        assert doc["delta_c"]["coeffs"] == rec["delta_c_coeffs"] and doc["xbar"]["coeffs"] == rec["xbar_coeffs"]
        assert (doc["delta_c"]["n_blocks"], doc["xbar"]["n_groups"], doc["delta_c"]["n_sims"]) == (
            rec["n_blocks"], rec["n_xbar_groups"], rec["n_sims"])
        assert W.payload_sha256(doc) == CFG["step0"]["w1_fold_maps"][str(k)]["refit_payload_sha256"]
        # D5 review R1 M1: a fold map never counts the whole cache's groups, the production map does
        assert rec["n_blocks"] < n_groups == doc_full_blocks
        rs = {"schema": E.RESEARCH_SCHEMA, "purpose": f"W-1 fold {k}", "split_rule": E.RULE_MULTI_HOLDOUT,
              "split": {"held_out_sims": held, "fold": k, "val_seed": r["val_seed"], "val_frac": r["val_frac"],
                        "n_train": r["n_train"], "n_val": r["n_val"]},
              "folds_digest": w1["fold_file_sha256"], "content_checks": [],
              "wc_map": {"rule": E.RULE_WCMAP_SUBSET, "fitted_on_sims_sha256": E.sims_sha256(train),
                         "n_sims": len(train)}}
        E.check_research_marking(rs, check_recorded=False)
        assert E.research_member_reasons(metas, rs) == [] and all("research" not in m["recipe"] for m in metas)
        del pair
