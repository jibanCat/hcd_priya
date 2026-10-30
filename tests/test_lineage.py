"""The lineage record (hcd_analysis/lineage.py, format hcd_priya.lineage/1): building, the release-eligibility rule,
the form checks, file identity, and embedding in an .npz product. Synthetic values only."""
import json
import os

import numpy as np
import pytest

from hcd_analysis import lineage as L

ALLOWED = frozenset({"test.finder/2"})
CLEAN = {"commit": "a" * 40, "dirty": False}
H = "b" * 64


def _rec(**kw):
    args = dict(inputs=[], finder_ids=["test.finder/2"], settings={"x": 1}, code=CLEAN, allowed_finder_ids=ALLOWED)
    args.update(kw)
    return L.build_record("thing", 1, **args)


def test_a_clean_build_from_an_allowed_finder_is_eligible_and_canonical():
    r = _rec(settings={"b": (1.0, 2.0), "a": None}, run_settings_sha256=H, product={"n": 3},
             validation={"verdict": "PASS"})
    assert r["format"] == L.FORMAT and r["release_eligible"] is True and r["ineligible_reasons"] == []
    assert r["settings"] == {"a": None, "b": [1.0, 2.0]}                     # tuples become lists
    assert L.parse_record(L.canonical_json(r)) == r
    assert L.record_sha256(r) == L.record_sha256(json.loads(L.canonical_json(r)))


@pytest.mark.parametrize("change, why", [
    (dict(code={"commit": None, "dirty": False}), "commit is unknown"),
    (dict(code={"commit": "a" * 40, "dirty": True}), "dirty"),
    (dict(code={"commit": "a" * 40, "dirty": None}), "state is unknown"),
    (dict(finder_ids=[]), "no finder id"),
    (dict(finder_ids=["test.finder/1"]), "not in the release allow-list"),
    (dict(product={"historical_reason": "a reproduction"}), "historical"),
])
def test_each_condition_of_the_rule_makes_a_product_ineligible(change, why):
    r = _rec(**change)
    assert r["release_eligible"] is False and any(why in w for w in r["ineligible_reasons"])


def test_eligibility_propagates_from_lineage_bearing_inputs_only():
    good, bad = _rec(), _rec(code={"commit": None, "dirty": None})
    ident = {"sha256": H, "size": 10}
    raw = L.input_entry("raw", "raw_spectra", dict(ident, mtime_ns=5))          # no record: does not propagate
    ok = L.build_record("child", 1, inputs=[raw, L.input_entry("parent", "thing", ident, record=good)],
                        finder_ids=None, settings={}, code=CLEAN, allowed_finder_ids=ALLOWED)
    assert ok["release_eligible"] and ok["finder_ids"] == ["test.finder/2"]       # the union of the inputs' ids
    assert "finder_ids" not in ok["inputs"][0] and ok["inputs"][0]["mtime_ns"] == 5
    no = L.build_record("child", 1, inputs=[L.input_entry("parent", "thing", ident, record=bad)], finder_ids=None,
                        settings={}, code=CLEAN, allowed_finder_ids=ALLOWED)
    assert not no["release_eligible"] and "input 'parent' is not release-eligible" in no["ineligible_reasons"]
    none = L.build_record("child", 1, inputs=[raw], finder_ids=None, settings={}, code=CLEAN,
                          allowed_finder_ids=ALLOWED)
    assert not none["release_eligible"] and "no finder id" in none["ineligible_reasons"]


def test_an_input_without_a_record_is_eligible_only_as_a_raw_file_or_a_validated_input():
    """Opt-in lineage closed: an input listed without its record (an old product) makes the child ineligible unless it
    is a raw simulation file or the child's validation block names it."""
    ident = {"sha256": H, "size": 10}

    def child(inputs, validation=None):
        return L.build_record("child", 1, inputs=inputs, finder_ids=None, settings={}, code=CLEAN,
                              allowed_finder_ids=ALLOWED, validation=validation)
    parent = L.input_entry("parent", "thing", ident, record=_rec())
    for kind in sorted(L.RAW_INPUT_KINDS):
        assert child([parent, L.input_entry("raw", kind, ident)])["release_eligible"]
    old = L.input_entry("old_catalogue", "catalog", ident)                       # no record: not raw
    r = child([parent, old])
    assert not r["release_eligible"] and any("'old_catalogue' of kind 'catalog' carries no lineage" in w
                                             for w in r["ineligible_reasons"])
    assert child([parent, old], validation={"validated_inputs": ["old_catalogue"]})["release_eligible"]
    assert not child([parent, old], validation={"validated_inputs": ["other"]})["release_eligible"]
    with pytest.raises(L.LineageError, match="claims release eligibility"):     # a forged flag is refused on check
        L.check_record(dict(r, release_eligible=True, ineligible_reasons=[]), ALLOWED)


def test_a_record_a_historical_loader_handed_out_is_never_listed_as_lineage_bearing():
    rec = _rec(product={"historical_reason": "an audit"})
    with pytest.raises(L.LineageError, match="historical loader"):
        L.input_entry("parent", "thing", {"sha256": H, "size": 1}, record=rec)


def test_a_record_claiming_eligibility_it_does_not_have_is_refused():
    r = _rec()
    forged = dict(r, code={"commit": "a" * 40, "dirty": True})                 # eligible flag kept, tree dirty
    L.check_record(forged)                                                     # well-formed ...
    with pytest.raises(L.LineageError, match="claims release eligibility"):
        L.check_record(forged, ALLOWED)                                        # ... but the rule says no
    with pytest.raises(L.LineageError, match="claims release eligibility"):
        L.check_record(r, frozenset({"other/1"}))                              # judged under another allow-list
    with pytest.raises(L.LineageError, match="disagrees"):
        L.check_record(dict(r, ineligible_reasons=["x"]))


@pytest.mark.parametrize("mutate, why", [
    (lambda r: r.update(format="other/1"), "not a hcd_priya.lineage/1"),
    (lambda r: r.pop("settings"), "missing"),
    (lambda r: r.update(extra=1), "unknown"),
    (lambda r: r.update(kind=""), "kind"),
    (lambda r: r.update(kind_version=True), "kind_version"),
    (lambda r: r.update(finder_ids=["b", "a"]), "sorted"),
    (lambda r: r.update(run_settings_sha256="abc"), "run_settings_sha256"),
    (lambda r: r.update(code={"commit": "xyz", "dirty": False}), "code.commit"),
    (lambda r: r.update(inputs=[{"role": "r", "kind": "k", "sha256": "0" * 63, "size": 1}]), "sha256"),
    (lambda r: r.update(inputs=[{"role": "r", "kind": "k", "sha256": H}]), "no size"),
    (lambda r: r.update(inputs=[{"role": "r", "kind": "k", "sha256": H, "size": 1, "finder_ids": []}]), "or neither"),
    (lambda r: r.update(inputs=[{"role": "r", "kind": "k", "sha256": H, "size": -1}]), "non-negative"),
    (lambda r: r.update(inputs=[{"role": "r", "kind": "k", "sha256": H, "size": 1, "path": "/x"}]), "malformed"),
    (lambda r: r.update(validation=[1]), "validation"),
])
def test_malformed_records_are_refused(mutate, why):
    r = _rec()
    mutate(r)
    with pytest.raises(L.LineageError, match=why):
        L.check_record(r)


@pytest.mark.parametrize("text, why", [("not json", "not a readable"), ('{"a": NaN}', "not a readable"),
                                       ("[" * 100000 + "]" * 100000, "not a readable"), ("{}", "not a hcd_priya")])
def test_unreadable_record_text_is_refused(text, why):
    with pytest.raises(L.LineageError, match=why):
        L.parse_record(text)


def test_non_finite_settings_are_refused_when_the_record_is_built():
    with pytest.raises(ValueError):
        _rec(settings={"x": float("nan")})


def test_file_identity_and_same_file(tmp_path):
    p = tmp_path / "raw.bin"
    p.write_bytes(b"synthetic bytes" * 1000)
    ident = L.file_identity(p, chunk=7)                                         # streamed in small blocks
    import hashlib
    assert ident["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() == L.sha256_file(p)
    assert ident["size"] == p.stat().st_size and ident["mtime_ns"] == p.stat().st_mtime_ns
    assert L.same_file(p, ident)
    os.utime(p, ns=(1, 2))                                                      # a copy with another mtime: hashed
    assert L.same_file(p, ident)
    p.write_bytes(b"Synthetic bytes" * 1000)                                    # same size, other bytes
    os.utime(p, ns=(1, 2))
    assert not L.same_file(p, ident)
    p.write_bytes(b"short")
    assert not L.same_file(p, ident)


def test_verify_input_names_the_role(tmp_path):
    p = tmp_path / "raw.bin"
    p.write_bytes(b"abc")
    r = L.build_record("thing", 1, inputs=[L.input_entry("raw_spectra", "raw_spectra", L.file_identity(p))],
                       finder_ids=["test.finder/2"], settings={}, code=CLEAN, allowed_finder_ids=ALLOWED)
    assert L.verify_input(r, "raw_spectra", p)["size"] == 3
    with pytest.raises(L.LineageError, match="0 inputs with role 'other'"):
        L.verify_input(r, "other", p)
    p.write_bytes(b"abd")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, r["inputs"][0]["mtime_ns"]))             # rewritten in place, mtime restored
    assert L.verify_input(r, "raw_spectra", p)                                  # the size and mtime shortcut ...
    with pytest.raises(L.LineageError, match="not the raw_spectra input"):
        L.verify_input(r, "raw_spectra", p, rehash=True)                        # ... which rehash does not take
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(L.LineageError, match="not the raw_spectra input"):
        L.verify_input(r, "raw_spectra", p)


def test_the_payload_digest_sees_names_dtypes_shapes_and_bytes():
    base = {"a": np.arange(4, dtype=np.int32), "s": np.array("name")}
    d = L.arrays_sha256(base)
    assert d == L.arrays_sha256(dict(reversed(list(base.items()))))           # order of the mapping is irrelevant
    for other in ({"b": base["a"], "s": base["s"]}, {"a": base["a"].astype(np.int64), "s": base["s"]},
                  {"a": base["a"].reshape(2, 2), "s": base["s"]}, {"a": base["a"] + 1, "s": base["s"]}):
        assert L.arrays_sha256(other) != d
    with pytest.raises(L.LineageError, match="Python objects"):
        L.arrays_sha256({"o": np.array([object()])})


def _npz_record(arrays):
    return _rec(product={"payload_sha256": L.arrays_sha256(arrays)})


def test_npz_round_trip_without_pickle_and_atomic(tmp_path):
    arrays = {"x": np.linspace(0.0, 1.0, 5), "n": np.array(3), "name": np.array("synthetic")}
    rec = _npz_record(arrays)
    p = tmp_path / "product.npz"
    L.write_npz(p, arrays, rec)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["product.npz"]       # no temporary file left
    got, r = L.read_npz(p)
    assert r == rec and set(got) == set(arrays) and all(np.array_equal(got[k], arrays[k]) for k in arrays)
    with np.load(p, allow_pickle=False) as d:
        assert str(d[L.NPZ_KEY][()]) == L.canonical_json(rec)


def test_npz_refusals(tmp_path):
    arrays = {"x": np.arange(3.0)}
    rec = _npz_record(arrays)
    with pytest.raises(L.LineageError, match="payload_sha256"):
        L.write_npz(tmp_path / "a.npz", {"x": np.arange(4.0)}, rec)            # not the arrays the record digested
    with pytest.raises(L.LineageError, match="reserved"):
        L.write_npz(tmp_path / "a.npz", {**arrays, L.NPZ_KEY: np.array("x")}, rec)
    assert not any(tmp_path.iterdir())
    np.savez(tmp_path / "old.npz", **arrays)                                    # written before records existed
    with pytest.raises(L.LineageError, match="no lineage record"):
        L.read_npz(tmp_path / "old.npz")
    got, r = L.read_npz(tmp_path / "old.npz", require_record=False)
    assert r is None and np.array_equal(got["x"], arrays["x"])
    np.savez(tmp_path / "edited.npz", x=np.arange(3.0) + 1, **{L.NPZ_KEY: np.array(L.canonical_json(rec))})
    with pytest.raises(L.LineageError, match="edited after it was made"):
        L.read_npz(tmp_path / "edited.npz")
    np.savez(tmp_path / "list.npz", x=arrays["x"], **{L.NPZ_KEY: np.array([L.canonical_json(rec)])})
    with pytest.raises(L.LineageError, match="0-d string"):
        L.read_npz(tmp_path / "list.npz")


def test_code_identity_of_a_directory_that_is_not_the_top_of_a_checkout(tmp_path):
    assert L.code_identity(tmp_path) == {"commit": None, "dirty": None}
    ident = L.code_identity()                                                   # this checkout, or an installed copy
    assert set(ident) == {"commit", "dirty"}
    assert ident["commit"] is None or len(ident["commit"]) == 40
