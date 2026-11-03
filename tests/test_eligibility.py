"""hcd_analysis.emulator.eligibility: the production metadata rule (reserved held-out and fold keys, the declared full
training set, research markings, the cv_selection/1 schema) and its stable reason codes. Synthetic metas only; the
values mean nothing, the tests check the rules (D5 plan v2 3.1, v3 2.1, v3.1 E2; T5-T14, T14b, T26, T29)."""
import copy
import json
import os

import pytest

from hcd_analysis.emulator import eligibility as E

HEX = "a" * 64


def _meta(**recipe_over):
    """A production-shaped synthetic meta (made-up values)."""
    recipe = {"epochs": 12, "patience": 3, "w_coh": 2.0, "all_sims": True, "holdout_sim": None, "val_seed": 1,
              "val_frac": 0.1, "n_train": 90, "n_val": 10}
    recipe.update(recipe_over)
    return {"schema_version": "2.0", "seed": 0, "arch_cfg": {"in_dim": 10}, "k_com_hmpc": [0.1, 0.2],
            "param_limits": [[0.0, 1.0]], "git_sha": "5" * 40, "code_dirty": False, "cache_sha256": HEX,
            "cache_lineage_sha256": HEX, "cache_path": "c.h5", "recipe": recipe}


def _codes(meta):
    return {E.reason_code(r) for r in E.production_recipe_reasons(meta)}


def _fold_meta():
    """A fold member's meta shaped like a grouped cross-validation member (made-up names and sizes)."""
    return _meta(all_sims=False, holdout_sims=["simA", "simB"], fold=3, folds_sha256=HEX)


# ------------------------------------------------------------------------------------------------ reason codes
def test_reason_codes_are_stable_machine_codes_with_a_human_text():
    r = E.reason("TIE_CACHE", "member 1 was trained on another cache")
    assert r == "[TIE_CACHE] member 1 was trained on another cache"
    assert E.reason_code(r) == "TIE_CACHE" and E.code_class(r) == "tie"
    assert E.reason_code("[RSV_KEY:recipe.split.fold] text") == "RSV_KEY:recipe.split.fold"
    assert E.code_class("[RSV_KEY:recipe.fold] text") == "production"
    assert E.code_class("[LIN_EXPORT_CODE] the export ran ...") == "lineage"
    assert E.code_class(E.research_reason("a purpose")) == "research"
    assert E.research_reason("p") == "[RESEARCH_BUNDLE] research bundle: p"
    for uncoded in ("the export ran from a dirty tree", "[lowercase] x", "[CODE]no space", "", None, 3):
        assert E.reason_code(uncoded) is None and E.code_class(uncoded) is None


# ------------------------------------------------------------------------------------- the production metadata rule
def test_a_production_member_has_no_reason():
    assert E.production_recipe_reasons(_meta()) == []


@pytest.mark.parametrize("over, code", [
    ({"fold": 0}, "RSV_KEY:recipe.fold"),                                  # T5: a zero fold is present
    ({"fold": []}, "RSV_KEY:recipe.fold"),                                 # T6: an empty list is present
    ({"holdout_sims": []}, "RSV_KEY:recipe.holdout_sims"),
    ({"holdout_sim": ""}, "RSV_KEY:recipe.holdout_sim"),                   # T7: an empty string is present
    ({"held_out_sims": ["s"]}, "RSV_KEY:recipe.held_out_sims"),            # T8: W-1's other spelling
    ({"heldOutSims": ["s"]}, "RSV_KEY:recipe.heldOutSims"),                # camelCase normalises to a reserved form
    ({"Fold-ID": 1}, "RSV_KEY:recipe.Fold-ID"),
    ({"nHeld": 6}, "RSV_KEY:recipe.nHeld"),
    ({"kFold": 10}, "RSV_KEY:recipe.kFold"),
    ({"cv_fold": 2}, "RSV_KEY:recipe.cv_fold"),
    ({"LOSO": True}, "RSV_KEY:recipe.LOSO"),
    ({"excluded_sims": ["s"]}, "RSV_KEY:recipe.excluded_sims"),
    ({"n_held_rows": 3}, "RSV_KEY:recipe.n_held_rows"),
    ({"split": {"holdout_sims": ["s"]}}, "RSV_KEY:recipe.split.holdout_sims"),      # T13: nested
    ({"blocks": [{"x": 1}, {"fold": 2}]}, "RSV_KEY:recipe.blocks.1.fold"),           # nested in a list of dicts
])
def test_reserved_keys_are_refused_by_whole_normalised_name_at_any_depth(over, code):
    assert code in _codes(_meta(**over))


@pytest.mark.parametrize("value", ["true", 1, None, False, 1.0, "yes"])
def test_all_sims_must_be_true_itself(value):                               # T9-T11
    assert _codes(_meta(all_sims=value)) == {"ALL_SIMS_NOT_TRUE"}
    m = _meta()
    del m["recipe"]["all_sims"]
    assert _codes(m) == {"ALL_SIMS_NOT_TRUE"}


@pytest.mark.parametrize("recipe", [None, [], "all_sims", 3, ["all_sims", True]])
def test_a_recipe_that_is_not_an_object_gives_a_reason_not_an_exception(recipe):       # T12
    m = _meta()
    m["recipe"] = recipe
    assert "RECIPE_NOT_DICT" in _codes(m)
    assert E.reason_code(E.production_recipe_reasons("not a meta")[0]) == "RECIPE_NOT_DICT"


def test_a_reserved_key_outside_the_recipe_is_refused_too():
    m = _meta()
    m["fold"] = 4
    m["training"] = {"loso": "simX"}
    assert _codes(m) == {"RSV_KEY:fold", "RSV_KEY:training.loso"}


def test_a_fold_member_is_refused_with_every_reason_named():                 # T14: the reasons name each key
    why = E.production_recipe_reasons(_fold_meta(), 2)
    assert {E.reason_code(r) for r in why} == {"ALL_SIMS_NOT_TRUE", "RSV_KEY:recipe.holdout_sims",
                                               "RSV_KEY:recipe.fold", "RSV_KEY:recipe.folds_sha256"}
    assert all("member 2" in r for r in why)


def test_a_research_marking_refuses_production():
    assert _codes(_meta(research={"purpose": "an ablation"})) == {"RECIPE_RESEARCH"}


def test_a_relabelled_fold_member_with_its_keys_stripped_passes_the_metadata_rule():
    """The documented limit (T15): metadata can be mislabelled or forged; only the content checks see it."""
    m = _fold_meta()
    for k in ("holdout_sims", "fold", "folds_sha256"):
        del m["recipe"][k]
    m["recipe"]["all_sims"] = True
    assert E.production_recipe_reasons(m) == []


@pytest.mark.parametrize("over", [{"threshold": 0.5}, {"lr_floor": 1e-5}, {"manifold_dim": 3}, {"loop": 2},
                                  {"scaffold": "x"}, {"unfold_k": 1}, {"n_train": 5}, {"val_frac": 0.2},
                                  {"holdout_sim": None}, {"fold": None}, {"holdout_sims": None},
                                  {"folding": 1}, {"heldout_sims_count_note": "none"}, {"bloom_w": 1.0}])
def test_no_false_refusal_of_ordinary_training_knobs(over):                   # T14b
    assert E.production_recipe_reasons(_meta(**over)) == []


def test_a_key_path_with_spaces_or_brackets_still_gives_a_parseable_code():
    m = _meta(**{"my block]": {"fold": 1}})
    why = E.production_recipe_reasons(m)
    assert [E.reason_code(r) for r in why] == ["RSV_KEY:recipe.my_block_.fold"] and "'recipe.my block].fold'" in why[0]


def test_key_normalisation():
    assert E.normalise_key("Held_Out-Sims") == "heldoutsims" and E.normalise_key(3) == "3"
    assert {"holdoutsims", "heldoutsims", "fold", "foldssha256", "kfold", "cvfold", "loso"} <= E.RESERVED_FORMS
    assert E.RULE_RESERVED_KEYS == "reserved_keys/2"
    assert E.find_reserved({"a": [{"b": {"Fold": 0}}], "fold": None}) == ["a.0.b.Fold"]


# ------------------------------------------------------------------------------------------ cv_selection/1 (T29)
def _cv(**over):
    rec = {"schema": "cv_selection/1", "source_cv_set_id": "a synthetic cv set", "source_folds_digest": HEX,
           "derived": {"epochs": {"value": 12, "rule": "median restored epoch over the fold members"}}}
    rec.update(over)
    return rec


def test_an_honest_selection_record_passes_a_scan_that_exempts_nothing():
    assert E.production_recipe_reasons(_meta(cv_selection=_cv())) == []
    assert E.production_recipe_reasons(_meta(cv_selection=_cv(source_record_sha256="b" * 64))) == []


@pytest.mark.parametrize("cv, code", [
    (dict(_cv(), holdout_sims=["s"]), "RSV_KEY:recipe.cv_selection.holdout_sims"),   # smuggled under the namespace
    (dict(_cv(), fold=1), "RSV_KEY:recipe.cv_selection.fold"),
    (dict(_cv(), split={"held_out": ["s"]}), "RSV_KEY:recipe.cv_selection.split.held_out"),
    (_cv(derived={"fold": {"value": 1, "rule": "r"}}), "RSV_KEY:recipe.cv_selection.derived.fold"),
])
def test_a_smuggled_split_under_cv_selection_is_refused(cv, code):
    m = _meta(cv_selection=cv, fold=1) if "derived" in cv and "fold" in cv["derived"] else _meta(cv_selection=cv)
    assert code in _codes(m) and "CV_SELECTION" in _codes(m)


@pytest.mark.parametrize("cv, text", [
    (dict(_cv(), note="x"), "keys outside"),                                   # an unknown key
    ({k: v for k, v in _cv().items() if k != "source_folds_digest"}, "lacks"),
    (_cv(schema="cv_selection/2"), "schema"),
    (_cv(source_cv_set_id=""), "source_cv_set_id"),
    (_cv(source_folds_digest="xyz"), "source_folds_digest"),
    (_cv(source_record_sha256="0"), "source_record_sha256"),
    (_cv(derived={}), "non-empty"),
    (_cv(derived={"epochs": {"value": 12}}), "exactly {value, rule}"),
    (_cv(derived={"epochs": {"value": [12, 13], "rule": "r"}}), "JSON scalar"),       # F-m1: scalars only
    (_cv(derived={"nonexistent": {"value": 1, "rule": "r"}}), "names no recipe key"),
    (_cv(derived={"epochs": {"value": 13, "rule": "r"}}), "is not the recipe's"),     # f2: equals the recipe
    (_cv(derived={"epochs": {"value": 12.0, "rule": "r"}}), "is not the recipe's"),
    ("a string", "not a cv_selection/1 object"),
])
def test_cv_selection_is_a_closed_schema(cv, text):
    why = E.production_recipe_reasons(_meta(cv_selection=cv))
    assert any(E.reason_code(r) == "CV_SELECTION" and text in r for r in why), why


# ------------------------------------------------------------------- T26: every clause has its own killing test
@pytest.mark.parametrize("make, only", [
    (lambda m: m["recipe"].update(all_sims=None), {"ALL_SIMS_NOT_TRUE"}),
    (lambda m: m.update(recipe=None), {"RECIPE_NOT_DICT"}),
    (lambda m: m["recipe"].update(research={"purpose": "p"}), {"RECIPE_RESEARCH"}),
    (lambda m: m["recipe"].update(cv_selection=dict(_cv(), extra=1)), {"CV_SELECTION"}),
    (lambda m: m["recipe"].update(withheld=["s"]), {"RSV_KEY:recipe.withheld"}),
])
def test_each_clause_alone_gives_exactly_its_code(make, only):
    m = copy.deepcopy(_meta())
    make(m)
    assert _codes(m) == only


# ------------------------------------------------------------------------------------ the split rules (T28, T31)
import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from hcd_analysis.emulator import data as D  # noqa: E402


def _w1_reference_split(sim_name, held_sims, *, val_seed=12345, val_frac=0.10):
    """W-1's production_split_multi (scripts/w1/common.py at 5067bb4), copied as the reference."""
    sim_name = np.asarray(sim_name).astype(str)
    held_sims = [str(s) for s in held_sims]
    is_held = np.isin(sim_name, held_sims)
    avail, held = np.flatnonzero(~is_held), np.flatnonzero(is_held)
    perm = np.random.default_rng(val_seed).permutation(avail.size)
    n_val = int(val_frac * avail.size)
    return np.sort(avail[perm[n_val:]]), np.sort(avail[perm[:n_val]]), held


def _sims(counts):
    return np.repeat([f"s{i}" for i in range(len(counts))], counts)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="<i8")).tobytes())
    return h.hexdigest()


SIMS7 = _sims([10, 12, 9, 11, 10, 13, 8])
# sha256 of (train, val, held) int64 bytes on SIMS7, val_seed 12345, val_frac 0.1, computed with the W-1 reference
GOLDEN_PRODUCTION_SPLIT = "1cc37f31cd886126f83b618b1b88893018c9de4d059feca458f7c3307ed9017a"     # nothing held out
GOLDEN_MULTI_SPLIT = "f9088d03215622f2b738b4cfa01303a04b111d0b5513d99886ebe2b2bc83d2a5"          # s2, s5 held out


@pytest.mark.parametrize("held", [[], ["s3"], ["s2", "s5"], ["s0", "s1", "s6"], ["s6", "s0"]])
@pytest.mark.parametrize("val_seed, val_frac", [(12345, 0.1), (7, 0.25)])
def test_the_multi_holdout_split_is_numerically_the_w1_splitter(held, val_seed, val_frac):
    got = D.multi_holdout_row_split(SIMS7, held, val_seed=val_seed, val_frac=val_frac)
    ref = _w1_reference_split(SIMS7, held, val_seed=val_seed, val_frac=val_frac)
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(got, ref))
    tr, va, ho = got
    assert np.array_equal(np.sort(np.concatenate([tr, va, ho])), np.arange(SIMS7.size))
    assert set(SIMS7[ho]) == set(held) and not set(SIMS7[tr]) & set(held) and not set(SIMS7[va]) & set(held)


def test_with_none_or_one_held_out_it_is_the_production_split():
    for held, one in (([], None), (["s4"], "s4")):
        a = D.multi_holdout_row_split(SIMS7, held, val_seed=3, val_frac=0.2)
        b = D.production_row_split(SIMS7, val_seed=3, val_frac=0.2, holdout_sim=one)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="not in the cache"):
        D.multi_holdout_row_split(SIMS7, ["s9"])


def test_the_number_of_simulations_is_the_caches_own():
    for counts in ([4, 4, 4], [3, 5, 4, 6, 2, 7, 5, 3, 4, 6, 5]):          # 3 and 11 simulations
        sims = _sims(counts)
        tr, va, _ = D.multi_holdout_row_split(sims, [])
        assert tr.size + va.size == sims.size and va.size == int(0.1 * sims.size)


def test_golden_digests_of_the_split_rules():                               # T31: a change forces a new rule version
    assert _digest(*D.production_row_split(SIMS7)) == _digest(*_w1_reference_split(SIMS7, []))
    assert _digest(*D.production_row_split(SIMS7)) == GOLDEN_PRODUCTION_SPLIT
    assert _digest(*D.multi_holdout_row_split(SIMS7, ["s2", "s5"])) == GOLDEN_MULTI_SPLIT


# ---------------------------------------------------- the content rules (v2 3.4, v3 2.3-2.4, v3.1 E1/E2; T15-T17b, T27,
#                                                      T30, T31, T35, T37) on synthetic release-format caches
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.emulator import safe_io as S  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator import wc_map as W  # noqa: E402
from tests import tau0_fixtures as tf  # noqa: E402
from tests.emulator._fixture import write_synthetic_cache_v34  # noqa: E402

# tensors_sha256 of train._fit_norm on the 5-simulation synthetic cache (seed 0) under its production split
# (val_seed 12345, val_frac 0.1), computed with the pre-D5 trainer
GOLDEN_NORM_FINGERPRINT = "c60b1ac36059e32837f0b55e6657c9e05d00c4fbdbe8c2cd10a0f48eae921d54"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A 5-simulation synthetic release cache, its w_c map, and honest production member metas and normaliser."""
    tf.clean_code(monkeypatch)
    cache = tmp_path / "cache5_v34.h5"
    write_synthetic_cache_v34(cache, n_sims=5, snaps_per_sim=4)
    d = D.load_cache(cache)
    wc = tmp_path / "wc.json"
    W.build(cache, wc)
    tr, va, _ = D.production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1)
    recipe = {"synthetic": 1, "all_sims": True, "holdout_sim": None, "val_seed": 12345, "val_frac": 0.1,
              "n_train": int(tr.size), "n_val": int(va.size)}
    metas = [{"cache_sha256": d["cache_sha256"], "recipe": dict(recipe), "seed": s} for s in (0, 1)]
    return {"cache": str(cache), "d": d, "wcm": W.read_wc_map(wc), "metas": metas, "norm": T._fit_norm(d, tr),
            "tr": tr, "va": va, "tmp": tmp_path}


def _prod(w, *, metas=None, norms=None, wcm=None, env=None, d=None, sha=None):
    return E.production_content_check(metas or w["metas"], norms or [w["norm"], w["norm"]], wcm or w["wcm"],
                                      cache=d or w["d"], cache_path=w["cache"],
                                      expected_cache_sha256=sha or w["d"]["cache_sha256"], training_environment=env)


def _oc(check):
    return {c["component"]: (c["outcome"], c["sub_reason"]) for c in check["components"]}


def test_honest_production_members_pass_every_component(world):
    c = _prod(world)
    assert c.passed and _oc(c) == {"cache": ("PASS", "tied"), "split": ("PASS", "recomputed"),
                                   "normaliser": ("PASS", "bitwise"), "wc_map": ("PASS", "bitwise")}
    sp = next(x for x in c["components"] if x["component"] == "split")["detail"]
    assert sp["n_sims_in_cache"] == 5 and sp["n_train_sims"] == 5 and sp["n_train"] == world["tr"].size
    nm = next(x for x in c["components"] if x["component"] == "normaliser")
    assert nm["n_tensors"] == nm["n_bitwise"] == len(S.norm_tensors(world["norm"]))
    assert nm["detail"]["recomputed_tensors_sha256"] == GOLDEN_NORM_FINGERPRINT                       # T31
    env = c["environment"]
    assert {"python", "numpy", "jax", "scipy", "hcd_priya"} <= set(env) and env["hcd_priya"]["commit"] == "c" * 40
    json.dumps(dict(c), allow_nan=False)                                       # recorded as is: finite JSON


def test_a_split_size_or_membership_that_differs_is_mismatch(world):          # T30(e), T16
    bad = [dict(m, recipe=dict(m["recipe"], n_train=m["recipe"]["n_train"] - 1)) for m in world["metas"]]
    c = _prod(world, metas=bad)
    assert not c.passed and _oc(c)["split"] == ("MISMATCH", "content") and _oc(c)["normaliser"][1] == "not-run"
    no_sizes = [dict(m, recipe={k: v for k, v in m["recipe"].items() if k not in ("n_train", "n_val")})
                for m in world["metas"]]
    assert _oc(_prod(world, metas=no_sizes))["split"] == ("MISMATCH", "content")    # sizes required in production
    two = [world["metas"][0], dict(world["metas"][1], recipe=dict(world["metas"][1]["recipe"], val_seed=7))]
    assert "different splits" in next(x for x in _prod(world, metas=two)["components"]
                                      if x["component"] == "split")["detail"]["why"]


def test_a_normaliser_not_fitted_on_the_declared_training_rows_is_mismatch(world):      # T16, T17
    d = world["d"]
    all_rows = T._fit_norm(d, np.arange(d["sim_name"].size))
    fold_tr, _, _ = D.multi_holdout_row_split(d["sim_name"], ["sim1"])
    one_row = np.array(world["tr"], copy=True)
    one_row[0] = world["va"][0]                                                # one wrong training row
    for norm in (all_rows, T._fit_norm(d, fold_tr), T._fit_norm(d, np.sort(one_row))):
        c = _prod(world, norms=[world["norm"], norm])
        assert not c.passed and _oc(c)["normaliser"] == ("MISMATCH", "content")
        nm = next(x for x in c["components"] if x["component"] == "normaliser")
        assert nm["max_rel"] > E.DRIFT_BOUND and [m["outcome"] for m in nm["detail"]["members"]] == ["PASS",
                                                                                                    "MISMATCH"]


def test_a_relabelled_fold_member_passes_the_loader_rule_but_not_the_content_check(world):      # T15
    d = world["d"]
    fold_tr, fold_va, _ = D.multi_holdout_row_split(d["sim_name"], ["sim2"])
    relabelled = dict(world["metas"][0], recipe=dict(world["metas"][0]["recipe"], n_train=int(fold_tr.size),
                                                     n_val=int(fold_va.size)))
    assert E.production_recipe_reasons(dict(relabelled, recipe=relabelled["recipe"])) == []
    c = _prod(world, metas=[relabelled, relabelled], norms=[T._fit_norm(d, fold_tr)] * 2)
    assert _oc(c)["split"] == ("MISMATCH", "content")                          # its sizes are a fold's
    forged = _prod(world, norms=[T._fit_norm(d, fold_tr)] * 2)                 # sizes forged to the production ones
    assert _oc(forged)["split"][0] == "PASS" and _oc(forged)["normaliser"] == ("MISMATCH", "content")


def _drifted(norm, rel=1e-14):
    out = copy.deepcopy(norm)
    out["P_filt"]["mu_marg"] = np.asarray(out["P_filt"]["mu_marg"]) * (1.0 + rel)
    return out


def test_outcome_classification(world):                                       # T30, T17b
    other_env = dict(E.environment(), numpy="0.0")                            # a complete, differing record
    assert _oc(_prod(world))["normaliser"] == ("PASS", "bitwise")                                     # (a)
    c = _prod(world, norms=[_drifted(world["norm"])] * 2, env=other_env)
    assert _oc(c)["normaliser"] == ("UNAVAILABLE", "env-drift") and not c.passed                      # (b), T17b
    assert _oc(_prod(world, norms=[_drifted(world["norm"])] * 2))["normaliser"] == ("MISMATCH", "strict-env")   # (c)
    assert _oc(_prod(world, norms=[_drifted(world["norm"], 6.8e-5)] * 2,
                     env=other_env))["normaliser"] == ("MISMATCH", "content")                            # (d)
    same = _prod(world, norms=[_drifted(world["norm"])] * 2, env=E.environment())
    assert _oc(same)["normaliser"] == ("MISMATCH", "content")                 # a recorded identical environment


def test_the_relative_difference_has_a_per_tensor_scale_floor():
    a = {"t": np.array([1.0, 1e-20, -3.0])}
    b = {"t": np.array([1.0, 1e-20 + 1e-26, -3.0])}
    diff = E.tensor_differences(a, b)
    assert diff["max_rel"] == pytest.approx(1e-26 / 3e-12) and diff["max_rel"] <= E.DRIFT_BOUND
    assert E.tensor_differences({"t": np.zeros(3)}, {"t": np.zeros(3)})["n_bitwise"] == 1
    assert E.classify(E.tensor_differences({"t": np.zeros(2)}, {"t": np.zeros(3)})) == ("MISMATCH", "content")
    assert E.classify(E.tensor_differences({"t": np.zeros(2)}, {"u": np.zeros(2)})) == ("MISMATCH", "content")


def _map_doc_record(w, doc=None, code=None, cache=None):
    d = cache if cache is not None else w["d"]
    doc = doc if doc is not None else W.refit(d)
    rec = W.make_record(doc, d["lineage"], {"sha256": d["cache_sha256"], "size": os.path.getsize(w["cache"])},
                        code=code or {"commit": "d" * 40, "dirty": False}, cache_name=os.path.basename(w["cache"]))
    return doc, rec


def test_the_map_comparison_ignores_the_run_that_wrote_it_and_nothing_else(world):          # T37
    doc, rec = _map_doc_record(world)
    rec = dict(rec, created_utc="2031-01-01T00:00:00Z", run_settings_sha256="b" * 64)
    other_run = W.parse_wc_map(W.to_bytes(doc, rec))                          # another commit, another time
    assert other_run.record["code"]["commit"] != world["wcm"].record["code"]["commit"]
    assert _oc(_prod(world, wcm=other_run))["wc_map"] == ("PASS", "bitwise")
    last_bit = copy.deepcopy(doc)
    last_bit["delta_c"]["coeffs"][1][2] = float(np.nextafter(doc["delta_c"]["coeffs"][1][2], np.inf))
    bit = W.parse_wc_map(W.to_bytes(*_map_doc_record(world, last_bit)))
    assert _oc(_prod(world, wcm=bit))["wc_map"] == ("MISMATCH", "strict-env")
    subset = W.parse_wc_map(W.to_bytes(*_map_doc_record(world, W.refit_subset(world["d"], ["sim0", "sim1", "sim2"]))))
    c = _prod(world, wcm=subset)
    assert _oc(c)["wc_map"] == ("MISMATCH", "content") and "payload delta_c.n_sims" in next(
        x for x in c["components"] if x["component"] == "wc_map")["detail"]["why"]
    zfit = copy.deepcopy(doc)
    zfit["xbar"]["z_fit"] = [2.1, 4.5]                                          # dossier section 6: xbar.z_fit too
    assert _oc(_prod(world, wcm=W.parse_wc_map(W.to_bytes(*_map_doc_record(world, zfit)))))["wc_map"][0] == "MISMATCH"


def test_another_cache_fails_the_tie_first(world, monkeypatch):                # T27
    tf.clean_code(monkeypatch)
    other = world["tmp"] / "cache3_v34.h5"
    write_synthetic_cache_v34(other, n_sims=3, snaps_per_sim=4, seed=3)
    d3 = D.load_cache(other)
    c = _prod(world, d=d3, sha=d3["cache_sha256"])
    assert _oc(c)["cache"] == ("MISMATCH", "content") and {_oc(c)[k][1] for k in ("split", "normaliser",
                                                                                   "wc_map")} == {"not-run"}
    assert _oc(_prod(world, sha="0" * 64))["cache"] == ("MISMATCH", "content")


def test_complete_means_every_row_of_this_cache_whatever_its_simulation_count(world, monkeypatch):    # T27
    tf.clean_code(monkeypatch)
    p = world["tmp"] / "cache3_v34.h5"
    write_synthetic_cache_v34(p, n_sims=3, snaps_per_sim=4, seed=3)
    d3 = D.load_cache(p)
    wc3 = world["tmp"] / "wc3.json"
    W.build(p, wc3)
    tr, va, _ = D.production_row_split(d3["sim_name"])
    metas = [{"cache_sha256": d3["cache_sha256"], "recipe": dict(world["metas"][0]["recipe"], n_train=int(tr.size),
                                                                 n_val=int(va.size))}]
    c = E.production_content_check(metas, [T._fit_norm(d3, tr)], W.read_wc_map(wc3), cache=d3, cache_path=str(p),
                                   expected_cache_sha256=d3["cache_sha256"])
    sp = next(x for x in c["components"] if x["component"] == "split")["detail"]
    assert c.passed and sp["n_sims_in_cache"] == sp["n_train_sims"] == 3


# ------------------------------------------------------------------------------- research content checks
def _research(world, held=("sim3",), rule=E.RULE_MULTI_HOLDOUT, **split_over):
    d = world["d"]
    tr, va, _ = D.multi_holdout_row_split(d["sim_name"], list(held))
    train_sims = sorted(set(d["sim_name"].tolist()) - set(held))
    split = {"held_out_sims": sorted(held), "fold": 0, "val_seed": 12345, "val_frac": 0.1, "n_train": int(tr.size),
             "n_val": int(va.size)}
    split.update(split_over)
    rs = {"schema": E.RESEARCH_SCHEMA, "purpose": "synthetic", "split_rule": rule, "split": split,
          "folds_digest": HEX, "content_checks": [],
          "wc_map": {"rule": E.RULE_WCMAP_SUBSET, "fitted_on_sims_sha256": E.sims_sha256(train_sims),
                     "n_sims": len(train_sims)}}
    recipe = {"all_sims": False, "holdout_sim": None, "holdout_sims": sorted(held), "fold": 0, "folds_sha256": HEX,
              "val_seed": 12345, "val_frac": 0.1, "n_train": int(tr.size), "n_val": int(va.size)}
    metas = [{"cache_sha256": d["cache_sha256"], "recipe": recipe}]
    wcm = W.parse_wc_map(W.to_bytes(*_map_doc_record(world, W.refit_subset(d, train_sims))))
    return rs, metas, [T._fit_norm(d, tr)], wcm


def _res(world, rs, metas, norms, wcm):
    return E.research_content_check(metas, norms, wcm, rs, cache=world["d"], cache_path=world["cache"],
                                    expected_cache_sha256=world["d"]["cache_sha256"])


def test_an_honest_fold_member_passes_the_research_content_check(world):
    rs, metas, norms, wcm = _research(world)
    c = _res(world, rs, metas, norms, wcm)
    assert c.passed and E.research_outcome_ok(c) and _oc(c)["wc_map"] == ("PASS", "bitwise")
    assert next(x for x in c["components"] if x["component"] == "wc_map")["rule"] == E.RULE_WCMAP_SUBSET


def test_research_content_mismatches(world):
    rs, metas, norms, wcm = _research(world)
    assert _oc(_res(world, rs, metas, [world["norm"]], wcm))["normaliser"] == ("MISMATCH", "content")
    c = _res(world, rs, metas, norms, world["wcm"])                            # the full-cache map: a leak
    assert _oc(c)["wc_map"][0] == "MISMATCH" and not E.research_outcome_ok(c)
    bad = copy.deepcopy(rs)
    bad["wc_map"]["fitted_on_sims_sha256"] = HEX
    assert "fitted_on_sims_sha256" in next(x for x in _res(world, bad, metas, norms, wcm)["components"]
                                           if x["component"] == "wc_map")["detail"]["why"]
    missing = copy.deepcopy(rs)
    missing["split"]["held_out_sims"] = ["sim9"]
    assert _oc(_res(world, missing, metas, norms, wcm))["split"] == ("MISMATCH", "content")


def test_an_unverifiable_split_still_refits_the_map_on_the_declared_training_set(world):
    rs, metas, norms, wcm = _research(world)
    u = dict(copy.deepcopy(rs), split_rule=E.RULE_UNVERIFIABLE, split={"held_out_sims": ["sim3"],
                                                                        "description": "by hand"})
    c = _res(world, u, metas, norms, wcm)
    assert _oc(c)["split"] == ("UNAVAILABLE", "unverifiable") and _oc(c)["normaliser"] == ("UNAVAILABLE", "not-run")
    assert _oc(c)["wc_map"] == ("PASS", "bitwise") and E.research_outcome_ok(c) and not c.passed


# --------------------------------------------------------------------- T35: the trainer's normaliser rows
def test_the_trainer_fits_the_normaliser_on_the_training_rows_only(world, monkeypatch):
    d, tr, va = world["d"], world["tr"], world["va"]
    seen = []
    real = T._fit_norm

    def spy(dd, idx):
        seen.append(np.array(idx, copy=True))
        return real(dd, idx)

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    monkeypatch.setattr(T, "_fit_norm", spy)
    monkeypatch.setattr(T, "_warmstart_bases", stop)
    with pytest.raises(Stop):
        T.train_fold(d, tr, va, epochs=1, prefit_baseline_epochs=0)
    assert len(seen) == 1 and np.array_equal(seen[0], tr)
    poisoned = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    poisoned["P_filt"][va] *= 1000.0
    poisoned["delta"][va] += 5.0
    a, b = S.norm_tensors(real(d, tr)), S.norm_tensors(real(poisoned, tr))
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def test_the_map_records_whole_input_entry_is_compared_but_its_eligibility(world):        # dossier section 6
    d = world["d"]
    doc = W.refit(d)
    ident = {"sha256": d["cache_sha256"], "size": os.path.getsize(world["cache"])}
    renamed = W.make_record(doc, d["lineage"], ident, code={"commit": "d" * 40, "dirty": False},
                            cache_name="another_name.h5")
    c = _prod(world, wcm=W.parse_wc_map(W.to_bytes(doc, renamed)))
    assert _oc(c)["wc_map"] == ("MISMATCH", "content") and "inputs[0].info" in next(
        x for x in c["components"] if x["component"] == "wc_map")["detail"]["why"]
    same = W.make_record(doc, d["lineage"], ident, code={"commit": "d" * 40, "dirty": False},
                         cache_name=os.path.basename(world["cache"]))
    assert _oc(_prod(world, wcm=W.parse_wc_map(W.to_bytes(doc, same))))["wc_map"] == ("PASS", "bitwise")
    with pytest.raises(E.EligibilityError, match="cache_path"):
        E.production_content_check(world["metas"], [world["norm"]] * 2, world["wcm"], cache=d,
                                   expected_cache_sha256=d["cache_sha256"], cache_size=1)


def test_a_validation_split_that_removes_whole_simulations_is_not_complete(world):
    """Production means the TRAINING rows cover every simulation of the cache: a validation fraction so large that
    whole simulations fall out of training is MISMATCH even with consistent sizes (killed mutant
    split_membership_unchecked)."""
    d = world["d"]
    tr, va, _ = D.production_row_split(d["sim_name"], val_seed=12345, val_frac=0.95)
    assert len(set(d["sim_name"][tr].tolist())) < 5
    metas = [{"cache_sha256": d["cache_sha256"], "recipe": dict(world["metas"][0]["recipe"], val_frac=0.95,
                                                                n_train=int(tr.size), n_val=int(va.size))}]
    c = _prod(world, metas=metas, norms=[T._fit_norm(d, tr)])
    sp = next(x for x in c["components"] if x["component"] == "split")
    assert sp["outcome"] == "MISMATCH" and "the training rows cover" in sp["detail"]["why"]



# ------------------------------------------------------------------------- D5 review R1 m3, m4, m5
@pytest.mark.parametrize("env", [{}, {"python": None}, {"python": "3.0"},
                                 "a string", lambda: dict(E.environment(), numpy=None),
                                 lambda: dict(E.environment(), extra="x")])
def test_only_a_complete_environment_record_counts_as_recorded(world, env):
    """m3: an empty, partial, None-valued or foreign record is not a recorded training environment, so a drift
    within the bound stays the strict MISMATCH."""
    env = env() if callable(env) else env
    c = _prod(world, norms=[_drifted(world["norm"])] * 2, env=env)
    assert _oc(c)["normaliser"] == ("MISMATCH", "strict-env")
    assert not E.recorded_environment(env)
    assert E.recorded_environment(dict(E.environment(), numpy="0.0"))


def test_the_content_checks_refuse_mismatched_or_empty_inputs(world):
    """m4: as many normalisers as metas (at least one) and a map are required; zero tensors are never PASS."""
    d = world["d"]
    kw = dict(cache=d, cache_path=world["cache"], expected_cache_sha256=d["cache_sha256"])
    for metas, norms, wcm in ((world["metas"] + world["metas"][:1], [world["norm"]] * 2, world["wcm"]),
                              ([], [], world["wcm"]), (world["metas"], [world["norm"]] * 2, None)):
        with pytest.raises(E.EligibilityError, match="metas|w_c map"):
            E.production_content_check(metas, norms, wcm, **kw)
    rs, rmetas, rnorms, rwcm = _research(world)
    for metas, norms, wcm in ((rmetas * 2, rnorms, rwcm), ([], [], rwcm), (rmetas, rnorms, None)):
        with pytest.raises(E.EligibilityError, match="metas|w_c map"):
            E.research_content_check(metas, norms, wcm, rs, **kw)
    assert E.classify(E.tensor_differences({}, {})) == ("MISMATCH", "content")


def test_the_map_records_validation_entry_is_compared(world):                 # m5
    doc, rec = _map_doc_record(world)
    with_validation = dict(rec, validation={"note": "an entry the refit does not make"})
    c = _prod(world, wcm=W.parse_wc_map(W.to_bytes(doc, with_validation)))
    assert _oc(c)["wc_map"] == ("MISMATCH", "content") and "record validation" in next(
        x for x in c["components"] if x["component"] == "wc_map")["detail"]["why"]
