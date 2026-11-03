"""Research bundles (hcd_priya.h5/2 with provenance.research): the research marking, the research loader
(safe_h5.load_research) and its member admission, the loader modes, and the refusal of every research bundle by the
production loaders (D5 plan v2 3.3, v3 2.2 and 2.6, v3.1 E2; T3, T4b, T18, T19, T21b, T32, T36). Synthetic bundles
(release_testing) only; the values mean nothing, the tests check the rules."""
import copy
import json
import os
import warnings

import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import eligibility as E
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H

PURPOSE = "synthetic fold 0 cross-validation (tests)"


@pytest.fixture
def res(tmp_path):
    return RT.write_synthetic_research_bundle(str(tmp_path / "research"), purpose=PURPOSE)


def _rewrite(info, fn):
    mp = os.path.join(info.bundle_dir, "manifest.json")
    man = json.load(open(mp))
    fn(man)
    os.remove(mp)
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return H.sha256_file(mp)


def _production_refusals(bundle_dir, sha, match):
    for load in (lambda: H.load_release(bundle_dir, expected_manifest_sha256=sha),
                 lambda: H.load_ensemble(bundle_dir, expected_manifest_sha256=sha),
                 lambda: H.load_bundle(bundle_dir, expected_manifest_sha256=sha),
                 lambda: hp.load_bundle(bundle_dir, sha)):
        with pytest.raises(H.SafeFormatError, match=match):
            load()


def _load(info, purpose="a test"):
    return H.load_research(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256, purpose=purpose)


def _recipe_fn(edit, only=None):
    def fn(i, model, meta, nm):
        if only is not None and i != only:
            return model, meta, nm
        r = copy.deepcopy(meta["recipe"])
        edit(r)
        return model, dict(meta, recipe=r), nm
    return fn


# ----------------------------------------------------------------------------- T3: a fold research bundle
def test_a_research_bundle_is_refused_by_every_production_loader_and_loads_as_research(res):
    assert res.lineage["release_eligible"] is False
    assert E.research_reason(PURPOSE) in res.lineage["ineligible_reasons"]
    _production_refusals(res.bundle_dir, res.manifest_sha256, "never a production model")
    rb = _load(res, purpose="evaluate the held-out simulation")
    assert isinstance(rb, H.ResearchBundle) and not isinstance(rb, H.ReleaseBundle)
    assert rb.kind == "research" and rb.release_verified is False and rb.anchored is True
    assert rb.metas.kind == "research" and rb.metas.release_verified is False and rb.metas.lineage_verified
    assert rb.purpose == PURPOSE and rb.load_purpose == "evaluate the held-out simulation"
    assert rb.split == {"rule": E.RULE_MULTI_HOLDOUT, "held_out_sims": ["synthsim2"], "fold": 0,
                        "folds_digest": RT.SYNTHETIC_FOLDS_SHA256, "n_train": 36, "n_val": 4, "cv_set_id": None}
    assert rb.research == res.research and rb.wc_map.doc["delta_c"]["n_sims"] == 2
    assert len(rb.ensemble.members) == 2


def test_a_research_bundle_loads_unanchored_only_with_a_warning_naming_it_research(res):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ens, metas, _ = H.load_ensemble(res.bundle_dir, expected_manifest_sha256=None)
    assert [x.category for x in w] == [H.UnanchoredBundleWarning] and "RESEARCH" in str(w[0].message)
    assert metas.kind == "dev" and not metas.anchored and not metas.release_verified


def test_a_production_bundle_is_refused_by_the_research_loader(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "prod"))
    with pytest.raises(H.SafeFormatError, match="not a research bundle"):
        _load(info)


# ------------------------------------------------------------------ T4b: a full-simulation research model
def test_a_full_simulation_research_model_is_research_by_its_marking(tmp_path):
    full = RT.write_synthetic_research_bundle(str(tmp_path / "full"), purpose="a full-simulation ablation",
                                              held_out_sims=())
    assert full.research["split_rule"] == E.RULE_PRODUCTION_SPLIT and full.research["split"]["held_out_sims"] == []
    _production_refusals(full.bundle_dir, full.manifest_sha256, "never a production model")
    rb = _load(full)
    assert rb.split["held_out_sims"] == [] and rb.wc_map.doc["delta_c"]["n_sims"] == 3


def test_a_full_simulation_member_without_recipe_research_is_not_admitted(tmp_path):     # T32 (b)
    def strip(r):
        r.pop("research")
    full = RT.write_synthetic_research_bundle(str(tmp_path / "full"), purpose="ablation", held_out_sims=(),
                                              member_fn=_recipe_fn(strip, only=1))
    with pytest.raises(H.SafeFormatError, match="member 1 declares no exclusion and carries no recipe.research"):
        _load(full)


# --------------------------------------------------------------------------------- T32: member admission
@pytest.mark.parametrize("edit, match", [
    (lambda r: r.update(research={"purpose": "another purpose"}), "names another purpose"),            # (iii)
    (lambda r: r.update(all_sims=True), "declares an exclusion and all_sims true"),                     # (iv)
    (lambda r: r.update(holdout_sims=[]), "empty held-out set"),                                        # F-m6
    (lambda r: r.update(holdout_sims=["synthsim1"]), "holds out"),                                      # (ii)
    (lambda r: r.update(fold=1), "fold 1 is not the split's 0"),
    (lambda r: r.update(folds_sha256="0" * 64), "folds_sha256 is not the bundle's folds_digest"),
    (lambda r: r.update(n_train=35), "n_train 35 is not the split's 36"),
    (lambda r: r.update(val_frac=0.2), "val_frac"),
    (lambda r: r.update(withheld_sims=["synthsim2"]), "not one of the multi_holdout_row_split/1 rule"),
    (lambda r: r.update(holdout_sims=None, holdout_sim="synthsim2"), "not one of the multi_holdout_row_split/1"),
])
def test_a_member_the_declared_split_does_not_admit_is_refused(tmp_path, edit, match):
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "b"), purpose=PURPOSE, member_fn=_recipe_fn(edit, only=1))
    with pytest.raises(H.SafeFormatError, match="not admitted to this research bundle.*" + match):
        _load(bad)


def test_fold_members_without_recipe_research_are_admitted_as_they_are(res):              # T32 (a): W-1 members
    for p in res.prefixes:
        r = json.load(open(p + ".json"))["meta"]["recipe"]
        assert "research" not in r and r["holdout_sims"] == ["synthsim2"] and r["fold"] == 0
    assert _load(res).kind == "research"


def test_mixed_and_two_fold_ensembles_are_refused(tmp_path):                              # T18, T19
    def to_full(r):
        for k in ("holdout_sims", "fold", "folds_sha256"):
            r.pop(k)
        r.update(all_sims=True, research={"purpose": PURPOSE})
    mixed = RT.write_synthetic_research_bundle(str(tmp_path / "mixed"), purpose=PURPOSE,
                                               member_fn=_recipe_fn(to_full, only=1))
    with pytest.raises(H.SafeFormatError, match="mixes members that declare an exclusion with full-simulation"):
        _load(mixed)
    _production_refusals(mixed.bundle_dir, mixed.manifest_sha256, "never a production model")
    two = RT.write_synthetic_research_bundle(str(tmp_path / "two"), purpose=PURPOSE,
                                             member_fn=_recipe_fn(lambda r: r.update(holdout_sims=["synthsim0"],
                                                                                     fold=1), only=1))
    with pytest.raises(H.SafeFormatError, match="member 1 holds out"):
        _load(two)


def test_an_unverifiable_split_admits_its_declared_members(tmp_path):
    def unverifiable(rs):
        rs = copy.deepcopy(rs)
        rs.update(split_rule=E.RULE_UNVERIFIABLE, split={"held_out_sims": ["synthsim2"],
                                                         "description": "rows of synthsim2 withheld by hand"})
        return rs
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "u"), purpose=PURPOSE, research_fn=unverifiable,
                                             member_fn=_recipe_fn(lambda r: r.update(withheld_sims=["synthsim2"])))
    rb = _load(bad)
    assert rb.split["rule"] == E.RULE_UNVERIFIABLE and rb.split["n_train"] is None


# ---------------------------------------------------------------------------------- the research marking
@pytest.mark.parametrize("edit, match", [
    (lambda rs: rs.update(schema="research/2"), "schema"),
    (lambda rs: rs.update(purpose=""), "purpose"),
    (lambda rs: rs.update(purpose=" padded"), "purpose"),
    (lambda rs: rs.update(split_rule="explicit_rows/1"), "not supported yet"),
    (lambda rs: rs.update(split_rule="loso/9"), "split_rule"),
    (lambda rs: rs["split"].update(held_out_sims=["b", "a"]), "sorted list"),
    (lambda rs: rs["split"].pop("fold"), "split keys"),
    (lambda rs: rs["split"].update(n_train=True), "counts"),
    (lambda rs: rs.update(folds_digest="xyz"), "folds_digest"),
    (lambda rs: rs.update(extra=1), "keys"),
    (lambda rs: rs.update(content_checks="none"), "content_checks"),
    (lambda rs: rs["wc_map"].update(rule="other/1"), "wc_map"),
    (lambda rs: rs["wc_map"].update(n_sims=0), "wc_map"),
])
def test_a_malformed_research_marking_is_refused_by_the_writer(tmp_path, edit, match):
    def fn(rs):
        rs = copy.deepcopy(rs)
        edit(rs)
        return rs
    with pytest.raises(H.SafeFormatError, match=match):
        RT.write_synthetic_research_bundle(str(tmp_path / "b"), purpose=PURPOSE, research_fn=fn)


def test_the_marking_must_name_the_maps_simulation_count(tmp_path):
    def fn(rs):
        rs = copy.deepcopy(rs)
        rs["wc_map"]["n_sims"] = 3
        return rs
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "b"), purpose=PURPOSE, research_fn=fn)
    with pytest.raises(H.SafeFormatError, match="n_sims 3 is not the map's 2"):
        _load(bad)


# ---------------------------------------------------- T36: the writer keeps the reason and the marking together
def _relist(res, tmp_path, provenance, lineage):
    import shutil
    import tempfile
    bd = tmp_path / tempfile.mkdtemp(dir=tmp_path) / "copy"                 # a fresh copy each time
    shutil.copytree(res.bundle_dir, bd)
    os.remove(bd / "manifest.json")
    return H.write_manifest(str(bd), [str(bd / os.path.basename(p)) for p in res.prefixes], provenance,
                            lineage=lineage)


def test_write_manifest_refuses_a_research_reason_and_marking_that_disagree(res, tmp_path):
    block = json.loads(json.dumps(res.lineage))
    with pytest.raises(H.SafeFormatError, match="does not carry exactly its research reason|carries a research reason"):
        _relist(res, tmp_path, {"source": "x"}, block)                               # reason without the marking
    plain = dict(block, ineligible_reasons=[r for r in block["ineligible_reasons"]
                                            if E.reason_code(r) != "RESEARCH_BUNDLE"])
    with pytest.raises(H.SafeFormatError, match="does not carry exactly its research reason"):
        _relist(res, tmp_path, {"research": res.research}, plain)                   # marking without the reason
    other = dict(res.research, purpose="another purpose")
    with pytest.raises(H.SafeFormatError, match="does not carry exactly its research reason"):
        _relist(res, tmp_path, {"research": other}, block)                          # another purpose
    for bad in ("text", ["research"], None):
        with pytest.raises(H.SafeFormatError, match="provenance must be a JSON object"):
            _relist(res, tmp_path, bad, block)
    path = _relist(res, tmp_path, {"research": res.research}, block)                # consistent: written
    assert os.path.exists(path)


def test_an_h5_1_bundle_carries_no_research_marking(tmp_path):
    with pytest.raises(H.SafeFormatError, match="carries no research marking"):
        H.write_manifest_historical(str(tmp_path), [], {"research": {}})


# -------------------------------------------------------------- the research loader's lineage rules (v3 2.6)
@pytest.mark.parametrize("edit, match", [
    (lambda m: m["lineage"]["ineligible_reasons"].append("[LIN_EXPORT_CODE] the export ran from a dirty tree"),
     "lineage or tie reasons"),
    (lambda m: m["lineage"]["ineligible_reasons"].append("[TIE_CACHE] member 0 was trained on another cache"),
     "lineage or tie reasons"),
    (lambda m: m["lineage"]["ineligible_reasons"].append("an uncoded reason"), "without a code"),
    (lambda m: m["lineage"].update(ineligible_reasons=[r for r in m["lineage"]["ineligible_reasons"]
                                                       if not r.startswith("[RESEARCH_BUNDLE]")]),
     "research reason"),
    (lambda m: m["lineage"]["export_code"].update(dirty=True), "dirty tree"),
    (lambda m: m["provenance"]["research"].update(purpose="edited"), "research reason"),
    (lambda m: m["provenance"].pop("research"), "not a research bundle"),
    (lambda m: m.update(provenance="text"), "provenance is not a JSON object"),
])
def test_a_stale_or_inconsistent_research_block_is_refused(res, edit, match):
    sha = _rewrite(res, edit)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_research(res.bundle_dir, expected_manifest_sha256=sha, purpose="a test")


def test_stored_production_codes_are_not_compared(res):
    """g-2: a later reserved name that would hit a stored member adds a production code; it does not orphan the
    bundle (only the lineage and tie codes must be empty)."""
    sha = _rewrite(res, lambda m: m["lineage"]["ineligible_reasons"].append(
        "[RSV_KEY:recipe.some_future_name] member 0 declares the held-out or fold key 'recipe.some_future_name'"))
    assert H.load_research(res.bundle_dir, expected_manifest_sha256=sha, purpose="a test").kind == "research"


def test_a_release_loader_refuses_a_provenance_that_is_not_an_object(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "p"))
    sha = _rewrite(info, lambda m: m.update(provenance=["research"]))
    _production_refusals(info.bundle_dir, sha, "provenance is not a JSON object")


# ------------------------------------------------------------------------------- T21b: the loader modes
def test_the_mode_and_the_digest_must_agree(res, tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "prod"))
    with pytest.raises(ValueError, match="needs a non-empty purpose"):
        H.load_research(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256, purpose="  ")
    with pytest.raises(TypeError):
        H.load_research(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256)        # purpose required
    for bad in (None, "f" * 63, 3):
        with pytest.raises(H.SafeFormatError, match="needs the published manifest sha256"):
            H.load_research(res.bundle_dir, expected_manifest_sha256=bad, purpose="a test")
    with pytest.raises(H.SafeFormatError, match="a development load takes no manifest digest"):
        H._load_v2(info.bundle_dir, expected=info.manifest_sha256, mode="dev")
    with pytest.raises(H.SafeFormatError, match="needs the published manifest sha256"):
        H._load_v2(info.bundle_dir, expected=None, mode="release")
    with pytest.raises(ValueError, match="purpose belongs to a research load"):
        H._load_v2(info.bundle_dir, expected=info.manifest_sha256, mode="release", purpose="x")
    with pytest.raises(ValueError, match="mode must be one of"):
        H._load_v2(info.bundle_dir, expected=info.manifest_sha256, mode="production")
    with pytest.raises(TypeError):
        H._load_v2(info.bundle_dir, expected=info.manifest_sha256)                          # mode required


# ------------------------------------------------------------------------ the bundle objects and their marks
def test_the_objects_carry_a_mandatory_kind_and_fail_closed_defaults(res, tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "prod"))
    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    assert rb.kind == "release" and rb.release_verified is True and rb.metas.kind == "release"
    assert type(rb.metas[0:1]) is list and type(rb.metas[:]) is list                 # a slice drops the marks
    with pytest.raises(TypeError):
        H.MemberMetas([], anchored=True, lineage_verified=True, manifest_sha256="0" * 64, fmt=H.FORMAT)
    with pytest.raises(ValueError, match="kind must be one of"):
        H.MemberMetas([], anchored=True, lineage_verified=True, manifest_sha256="0" * 64, fmt=H.FORMAT, kind="prod")
    mm = H.MemberMetas([], anchored=True, lineage_verified=True, manifest_sha256="0" * 64, fmt=H.FORMAT, kind="dev",
                       release_verified=True)
    assert mm.release_verified is False                                              # only a release load is
    with pytest.raises(TypeError):
        H.ReleaseBundle(*rb[:9])                                                     # kind has no default
    assert H.ReleaseBundle(*rb[:9], kind="release").release_verified is False        # fail-closed default
    assert H.ResearchBundle._field_defaults == {"release_verified": False}


@pytest.mark.parametrize("key, value, code", [("cache_sha256", "1" * 64, "TIE_CACHE"),
                                              ("code_dirty", True, "TIE_DIRTY"),
                                              ("cache_lineage_sha256", "2" * 64, "TIE_CACHE_RECORD")])
def test_the_research_loader_recomputes_the_members_ties(tmp_path, key, value, code):
    """A research member not tied to the block's cache or commit is refused by the recomputed ties even when the
    stored block was edited to hide the tie reason (killed mutant research_ties_unchecked)."""
    def untie(i, model, meta, nm):
        return model, (dict(meta, **{key: value}) if i == 1 else meta), nm
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "b"), purpose=PURPOSE, member_fn=untie)
    assert any(E.reason_code(r) == code for r in bad.lineage["ineligible_reasons"])
    sha = _rewrite(bad, lambda m: m["lineage"].update(ineligible_reasons=[
        r for r in m["lineage"]["ineligible_reasons"] if E.code_class(r) != "tie"]))
    with pytest.raises(H.SafeFormatError, match=f"members are not the ones the lineage block describes.*{code}"):
        H.load_research(bad.bundle_dir, expected_manifest_sha256=sha, purpose="a test")


# ------------------------------- D5 review R2 M1: an unverifiable split must not contradict the members' exclusion
def _unverifiable(held, rule=E.RULE_WCMAP_SUBSET):
    def fn(rs):
        rs = copy.deepcopy(rs)
        rs.update(split_rule=E.RULE_UNVERIFIABLE, split={"held_out_sims": list(held), "description": "a test"})
        rs["wc_map"] = dict(rs["wc_map"], rule=rule if held else E.RULE_WCMAP)
        return rs
    return fn


@pytest.mark.parametrize("held, member_edit, match", [
    ((), None, "an unverifiable split must declare the held-out simulations"),                 # P1
    (("synthsim1",), None, "member 0 holds out \\['synthsim2'\\], the bundle's split \\['synthsim1'\\]"),   # P2
    ((), lambda r: [r.pop(k) for k in ("holdout_sims", "fold", "folds_sha256")] and None or r.update(
        withheld_sims=["synthsim2"]), "an unverifiable split must declare the held-out simulations"),  # P3
    (("synthsim1", "synthsim2"), None, "holds out"),
])
def test_an_unverifiable_split_must_declare_the_members_own_held_out_set(tmp_path, held, member_edit, match):
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "u"), purpose=PURPOSE, research_fn=_unverifiable(held),
                                             member_fn=_recipe_fn(member_edit) if member_edit else None)
    with pytest.raises(H.SafeFormatError, match="not admitted to this research bundle.*" + match):
        _load(bad)


# --------------------------------------------------------- D5 review R2 m1, m4, m8: the recorded checks and the codes
def _marking(edit):
    def fn(rs):
        rs = copy.deepcopy(rs)
        edit(rs)
        return rs
    return fn


@pytest.mark.parametrize("edit, match", [
    (lambda rs: rs["content_checks"][1].update(outcome="MISMATCH"), "recorded MISMATCH"),
    (lambda rs: rs.update(content_checks=[]), "content_checks must record"),
    (lambda rs: rs["content_checks"].pop(2), "content_checks must record"),
    (lambda rs: rs["content_checks"].append(dict(rs["content_checks"][0])), "content_checks must record"),
    (lambda rs: rs["wc_map"].update(rule=E.RULE_WCMAP), "wcmap_refit_subset/1"),
])
def test_a_research_marking_with_inconsistent_recorded_checks_is_refused(tmp_path, res, edit, match):
    """R2 m1: a recorded MISMATCH, a missing or repeated component, or a map rule that contradicts the held-out set
    is refused by the writer and by the research loader."""
    with pytest.raises(H.SafeFormatError, match=match):
        RT.write_synthetic_research_bundle(str(tmp_path / "w"), purpose=PURPOSE, research_fn=_marking(edit))

    def manifest_edit(m):
        edit(m["provenance"]["research"])
    sha = _rewrite(res, manifest_edit)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_research(res.bundle_dir, expected_manifest_sha256=sha, purpose="a test")


def test_a_full_simulation_marking_must_name_the_full_map_rule(tmp_path):
    with pytest.raises(H.SafeFormatError, match="wcmap_refit/1"):
        RT.write_synthetic_research_bundle(str(tmp_path / "f"), purpose=PURPOSE, held_out_sims=(),
                                           research_fn=_marking(lambda rs: rs["wc_map"].update(
                                               rule=E.RULE_WCMAP_SUBSET)))


def test_research_admission_refusals_carry_codes(tmp_path):                                    # R2 m4
    bad = RT.write_synthetic_research_bundle(str(tmp_path / "b"), purpose=PURPOSE,
                                             member_fn=_recipe_fn(lambda r: r.update(fold=1, n_train=35), only=1))
    rs = bad.research
    metas = [json.load(open(p + ".json"))["meta"] for p in bad.prefixes]
    why = E.research_member_reasons(metas, rs)
    assert {E.reason_code(r) for r in why} == {"RES_FOLD", "RES_KNOB"}
    assert {E.code_class(r) for r in why} == {"admission"}
    with pytest.raises(H.SafeFormatError, match=r"\[RES_FOLD\]"):
        _load(bad)


def test_a_digest_must_be_lowercase_hex(res):                                                   # R2 m8
    with pytest.raises(H.SafeFormatError, match="64 lowercase hex digits"):
        H.load_research(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256.upper(), purpose="x")
