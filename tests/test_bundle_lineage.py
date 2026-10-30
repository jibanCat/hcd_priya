"""The lineage checks of the h5/2 release loader (safe_h5.load_ensemble / load_bundle / load_release with a manifest
digest): a bundle loads only when its lineage block establishes a clean lineage and the w_c map and the members agree
with it; every refusal below is anchored on the edited manifest's own digest, so it is the lineage check that refuses,
not the digest. Synthetic bundles (hcd_analysis.emulator.release_testing) only."""
import json
import os
import warnings

import h5py
import pytest

from hcd_analysis import lineage as L
from hcd_analysis.catalog import FINDER_ID
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H


@pytest.fixture
def info(tmp_path):
    return RT.write_synthetic_bundle(str(tmp_path / "bundle"))


def _rewrite(info, fn):
    """Edit manifest.json with ``fn(manifest)``; return the new manifest's sha256 (the anchor of the edited bundle)."""
    mp = os.path.join(info.bundle_dir, "manifest.json")
    man = json.load(open(mp))
    fn(man)
    os.remove(mp)
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return H.sha256_file(mp)


def _forge_eligible(man):
    man["lineage"]["release_eligible"], man["lineage"]["ineligible_reasons"] = True, []


def test_a_synthetic_release_bundle_loads_and_is_marked_anchored(info):
    ens, metas, norm = H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    assert len(ens.members) == 2 and metas.anchored and metas.lineage_verified and metas.format == H.FORMAT
    assert metas.manifest_sha256 == info.manifest_sha256 and metas.lineage == info.lineage
    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    assert rb.lineage["release_eligible"] and rb.wc_map.cache_sha256 == rb.lineage["cache"]["sha256"]
    assert rb.lineage["finder_ids"] == [FINDER_ID] and rb.lineage["code_commit"] == RT.SYNTHETIC_COMMIT
    assert rb.lineage["cache"]["rows"] == {"built": 2, "carried_validated": 0} and rb.lineage["cache"]["carried"] is None
    man = json.load(open(os.path.join(info.bundle_dir, "manifest.json")))
    assert set(man) == {"format", "members", "files", "provenance", "wc_map", "lineage"}
    assert man["wc_map"] == H.WC_MAP_FILE and H.WC_MAP_FILE in {f["path"] for f in man["files"]}
    with h5py.File(info.prefixes[0] + ".h5", "r") as f:
        assert f.attrs["format"] == H.FORMAT
    for m in metas:
        assert m["cache_sha256"] == info.lineage["cache"]["sha256"] and m["git_sha"] == info.lineage["code_commit"]


def test_a_carried_cache_is_labelled_in_the_block(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "b"), carried=True)
    c = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256).lineage["cache"]
    assert c["rows"] == {"built": 0, "carried_validated": 2}
    assert c["carried"]["label"] == "carried_validated" and c["carried"]["promoted"]["passed"] is True
    assert c["carried"]["validation_all_passed"] is True and len(c["carried"]["source_sha256"]) == 64


def test_the_unanchored_development_path_loads_marked_and_warns(info):
    sha = _rewrite(info, lambda m: m["lineage"].update(release_eligible=False, ineligible_reasons=["dev"]))
    with pytest.raises(H.SafeFormatError, match="does not establish a clean release"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ens, metas, norm = H.load_ensemble(info.bundle_dir, expected_manifest_sha256=None)
        members = H.load_bundle(info.bundle_dir, expected_manifest_sha256=None)
    assert [x.category for x in w] == [H.UnanchoredBundleWarning] * 2
    assert not metas.anchored and not metas.lineage_verified and len(members) == 2
    with pytest.raises(TypeError):
        H.load_ensemble(info.bundle_dir)                          # the keyword stays required
    for bad in (None, "", "f" * 63, 1):
        with pytest.raises(H.SafeFormatError, match="needs the published manifest sha256"):
            H.load_release(info.bundle_dir, expected_manifest_sha256=bad)


@pytest.mark.parametrize("edit, match", [
    (lambda m: m["lineage"].update(finder_ids=["hcd_priya.finder/1"]), "not in the release allow-list"),
    (lambda m: m["lineage"].update(finder_ids=[]), "no finder id"),
    (lambda m: m["lineage"]["cache"].update(release_eligible=False), "not a release-eligible tau0_cache"),
    (lambda m: m["lineage"]["cache"].update(kind="catalog"), "not a release-eligible tau0_cache"),
    (lambda m: m["lineage"]["catalogues"].update(n=0), "names no catalogue"),
    (lambda m: m["lineage"].update(code_commit=None), "code commit is unknown"),
    (lambda m: m["lineage"].update(code_commit="abc"), "code commit is unknown"),
    (lambda m: m["lineage"]["export_code"].update(dirty=True), "dirty tree"),
    (lambda m: m["lineage"]["export_code"].update(commit=None), "unknown commit"),
    (lambda m: m["lineage"]["wc_map"].update(release_eligible=False), "w_c map is not release-eligible"),
    (lambda m: m["lineage"]["wc_map"].update(cache_sha256="0" * 64), "fitted to another cache"),
    (lambda m: m["lineage"].update(release_eligible=False, ineligible_reasons=["stated"]), "stated"),
    (lambda m: m["lineage"]["cache"]["rows"].update(carried_validated=1), "carried rows without the carry"),
])
def test_a_block_that_does_not_establish_a_clean_lineage_is_refused(info, edit, match):
    sha = _rewrite(info, edit)
    for load in (H.load_ensemble, H.load_bundle, H.load_release):
        with pytest.raises(H.SafeFormatError, match=match):
            load(info.bundle_dir, expected_manifest_sha256=sha)


def test_carried_rows_must_be_promoted_and_labelled(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "b"), carried=True)
    sha = _rewrite(info, lambda m: m["lineage"]["cache"]["carried"].update(promoted=None))
    with pytest.raises(H.SafeFormatError, match="not promoted"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)
    sha = _rewrite(info, lambda m: m["lineage"]["cache"]["carried"].update(label="carried"))
    with pytest.raises(H.SafeFormatError, match="carried rows without the carry's label"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)


@pytest.mark.parametrize("edit, match", [
    (lambda m: m.pop("lineage"), "manifest keys"),
    (lambda m: m.update(extra=1), "manifest keys"),
    (lambda m: m.pop("wc_map"), "manifest keys"),
    (lambda m: m["lineage"].pop("finder_ids"), "lineage block: keys"),
    (lambda m: m["lineage"].update(format="hcd_priya.bundle_lineage/0"), "lineage block: format"),
    (lambda m: m["lineage"]["cache"].update(sha256="xyz"), "malformed cache entry"),
    (lambda m: m["lineage"].update(ineligible_reasons=["x"]), "disagrees with ineligible_reasons"),
    (lambda m: m["lineage"]["wc_map"].update(sha256="0" * 64), "not the one the lineage block names"),
])
def test_a_malformed_block_or_manifest_is_refused(info, edit, match):
    sha = _rewrite(info, edit)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)


def test_the_w_c_map_must_be_the_listed_one_and_fitted_to_the_cache(info, tmp_path):
    wc = os.path.join(info.bundle_dir, H.WC_MAP_FILE)
    raw = open(wc, "rb").read()
    os.remove(wc)
    open(wc, "wb").write(raw.replace(b'"n_groups":4', b'"n_groups":5'))
    with pytest.raises(H.SafeFormatError, match="wc_map.json does not match the manifest"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    os.remove(wc)
    open(wc, "wb").write(raw)
    # a map of the same layout fitted to another cache, listed under its own digest: the block names another map
    from hcd_analysis.emulator import wc_map as W
    rec, ident = RT.synthetic_cache_record(n_groups=3)
    doc = RT.synthetic_wc_map_doc()
    other = tmp_path / "other.json"
    W.write_wc_map(other, doc, W.make_record(doc, rec, ident, code=RT.SYNTHETIC_CODE))
    os.remove(wc)
    open(wc, "wb").write(open(other, "rb").read())

    def relist(m):
        for f in m["files"]:
            if f["path"] == H.WC_MAP_FILE:
                f["sha256"], f["bytes"] = H.sha256_file(wc), os.path.getsize(wc)
    sha = _rewrite(info, relist)
    with pytest.raises(H.SafeFormatError, match="not the one the lineage block names"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)


@pytest.mark.parametrize("key, value, match", [
    ("cache_sha256", "1" * 64, "member 1 was trained on another cache"),
    ("cache_lineage_sha256", "1" * 64, "member 1 does not name the cache's lineage record"),
    ("git_sha", "6" * 40, "member 1 was trained at another or an unknown commit"),
    ("code_dirty", True, "member 1 was trained from a tree not known to be clean"),
    ("code_dirty", None, "member 1 was trained from a tree not known to be clean"),
])
def test_members_not_described_by_the_block_are_refused(tmp_path, key, value, match):
    def differ(i, model, meta, nm):
        return model, (dict(meta, **{key: value}) if i == 1 else meta), nm
    info = RT.write_synthetic_bundle(str(tmp_path / "b"), member_fn=differ)
    assert not info.lineage["release_eligible"] and any(match in r for r in info.lineage["ineligible_reasons"])
    with pytest.raises(H.SafeFormatError, match="does not establish a clean release"):   # the block as written says so
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)

    def forge(m):                                                  # a block forged eligible: the members refuse it
        _forge_eligible(m)
        m["lineage"]["code_commit"] = RT.SYNTHETIC_COMMIT
    sha = _rewrite(info, forge)
    with pytest.raises(H.SafeFormatError, match="members are not the ones the lineage block describes.*" + match):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)


def test_the_member_count_must_be_the_blocks(info):
    sha = _rewrite(info, lambda m: m["lineage"].update(n_members=3))
    with pytest.raises(H.SafeFormatError, match="2 members, the block states 3"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)


def test_the_block_eligibility_is_computed_never_passed_in(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "b"), export_code={"commit": RT.SYNTHETIC_COMMIT, "dirty": True})
    assert info.lineage["release_eligible"] is False and "the export ran from an unknown commit or a dirty tree" in \
        info.lineage["ineligible_reasons"]
    with pytest.raises(H.SafeFormatError, match="dirty tree"):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    rec, ident = RT.synthetic_cache_record(code={"commit": None, "dirty": None})       # an ineligible cache
    assert rec["release_eligible"] is False
    with pytest.raises(H.SafeFormatError, match="lineage block: keys"):
        H.check_lineage_block(dict(info.lineage, extra=1))


def test_an_h5_2_member_never_records_a_cache_path(tmp_path):
    import jax
    from hcd_analysis.emulator.model import Emulator
    rec, ident = RT.synthetic_cache_record()
    meta = dict(RT.synthetic_meta(0, rec, ident), cache_path="/some/where/cache.h5")
    with pytest.raises(H.SafeFormatError, match="records the cache's file name only"):
        H.export_member(str(tmp_path / "m"), Emulator(**RT.SYNTHETIC_ARCH, key=jax.random.PRNGKey(0)), meta,
                        RT.synthetic_norm())
    assert os.listdir(tmp_path) == []


def test_the_writer_refuses_a_block_naming_another_map(tmp_path, info):
    bd = tmp_path / "copy"
    import shutil
    shutil.copytree(info.bundle_dir, bd)
    os.remove(bd / "manifest.json")
    block = json.loads(json.dumps(info.lineage))
    block["wc_map"]["sha256"] = "0" * 64
    with pytest.raises(H.SafeFormatError, match="not the w_c map the lineage block names"):
        H.write_manifest(str(bd), [str(bd / os.path.basename(p)) for p in info.prefixes], {}, lineage=block)
    assert L.canonical_json(info.lineage)                                  # the block is plain canonical JSON
