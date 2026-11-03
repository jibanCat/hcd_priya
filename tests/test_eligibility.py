"""hcd_analysis.emulator.eligibility: the production metadata rule (reserved held-out and fold keys, the declared full
training set, research markings, the cv_selection/1 schema) and its stable reason codes. Synthetic metas only; the
values mean nothing, the tests check the rules (D5 plan v2 3.1, v3 2.1, v3.1 E2; T5-T14, T14b, T26, T29)."""
import copy

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
