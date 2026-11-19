"""Cache splicing and merging, their lineage, and the historical-vs-repaired delta audit (S4/F8 repair; internal spec
S4_REPAIR_SPEC sections 3-4). Synthetic caches with the production key layout (tests/tau0_fixtures.py); no
fake_spectra."""

import json
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import merge_tau0_cache as mg  # noqa: E402
import splice_tau0_cache as sp  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402
from tests import tau0_fixtures as tf  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_code(monkeypatch):
    tf.clean_code(monkeypatch)


def _write(path, groups, *, edges_shift=0.0, attrs=None, **kw):
    """A release-eligible v3.4 cache of synthetic groups [(sim, snap, z_grid, n_alpha, seed)]."""
    tf.write_v34(
        path,
        groups,
        edges_shift=edges_shift,
        attrs=dict({"merged_from_n_shards": 3}, **(attrs or {})),
        **kw,
    )
    return path


BASE = [
    ("simA", 16, 3.2, 3, 1),
    ("simA", 17, 2.8, 3, 2),
    ("simA", 19, 2.6, 3, 3),
    ("simB", 5, 4.0, 3, 4),
]
SHARD = [("simA", 17, 3.0, 3, 20), ("simA", 18, 2.8, 3, 21)]
COMMIT = "0123456789abcdef0123456789abcdef01234567"  # synthetic


def _splice(tmp_path, base=BASE, shard=SHARD, remove=(("simA", 17, 2.8),), **kw):
    b = _write(tmp_path / "base.h5", base)
    s = _write(tmp_path / "shard.h5", shard)
    out = tmp_path / "out.h5"
    kw.setdefault("code_commit", COMMIT)
    sp.splice_cache(b, [s], list(remove), out, record={"spec": "test"}, **kw)
    return b, s, out


def _read(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][...] for k in f.keys()}
        return d, dict(f.attrs)


def test_splice_replaces_and_inserts_in_canonical_order(tmp_path):
    b, s, out = _splice(tmp_path)
    d, a = _read(out)
    ids = [(x.decode(), int(n)) for x, n in zip(d["snap_sim_name"], d["snap_snap"])]
    assert ids == [("simA", 16), ("simA", 17), ("simA", 18), ("simA", 19), ("simB", 5)]
    assert a["n_rows"] == 15 and a["n_snaps"] == 5 and "merged_from_n_shards" not in a
    rec = json.loads(a["repair_record"])
    assert (
        rec["spec"] == "test"
        and rec["removed"] == [["simA", 17, 2.8]]
        and len(rec["base_sha256"]) == 64
    )
    assert rec["added"] == [["simA", 17, 3.0], ["simA", 18, 2.8]]
    sim, snap = d["sim_name"].astype(str), d["snap"]
    for r in range(sim.size):  # every row points at its own block
        g = d["snap_group_idx"][r]
        assert (sim[r], int(snap[r])) == (d["snap_sim_name"][g].decode(), int(d["snap_snap"][g]))
    zg = {(s_, int(n)): z for s_, n, z in zip(sim, snap, d["z_grid"])}
    assert zg[("simA", 17)] == 3.0 and zg[("simA", 18)] == 2.8


def test_splice_keeps_untouched_groups_bit_identical_and_the_key_set(tmp_path):
    b, s, out = _splice(tmp_path)
    rep = sp.cache_delta(b, out)
    assert rep["key_set_equal"] and rep["static_equal"] and rep["dtypes_equal"]
    # row identity = (sim, z_grid, alpha_idx): the z = 2.8 rows exist on both sides (rebuilt), the z = 3.0 rows are new
    assert rep["rows_only_hist"] == []
    assert rep["rows_only_new"] == [("simA", 3.0, a) for a in range(3)]
    assert set(rep["rows_changed"]) == {("simA", 2.8, a) for a in range(3)}
    assert (
        "snap" in rep["rows_changed"][("simA", 2.8, 0)]
        and "P_tier_c" in rep["rows_changed"][("simA", 2.8, 0)]
    )
    untouched = [i for i in rep["row_identities_common"] if i[0] == "simB" or i[1] in (3.2, 2.6)]
    assert len(untouched) == 9 and all(i not in rep["rows_changed"] for i in untouched)
    # block identity = (sim, snap): (simA, 17) now holds the shard's block, (simA, 18) is new
    assert rep["blocks_only_new"] == [("simA", 18)] and rep["blocks_only_hist"] == []
    assert set(rep["blocks_changed"]) == {("simA", 17)}


def test_cache_delta_catches_any_change_outside_the_footprint(tmp_path):
    b, s, out = _splice(tmp_path)
    with h5py.File(out, "r+") as f:  # perturb one untouched row's P_tier_c by 1e-9
        i = int(np.where((f["sim_name"].asstr()[...] == "simB"))[0][1])
        f["P_tier_c"][i, 3, 2] = f["P_tier_c"][i, 3, 2] * (1 + 1e-9)
    rep = sp.cache_delta(b, out)
    assert ("simB", 4.0, 1) in rep["rows_changed"] and "P_tier_c" in rep["rows_changed"][
        ("simB", 4.0, 1)
    ]


def test_cache_delta_catches_block_and_static_changes(tmp_path):
    b, s, out = _splice(tmp_path)
    with h5py.File(out, "r+") as f:
        f["snap_dNdX_DLA"][0] = f["snap_dNdX_DLA"][0] + 1e-12
        f["log_nhi_edges"][3] = f["log_nhi_edges"][3] + 1e-12
    rep = sp.cache_delta(b, out)
    assert ("simA", 16) in rep["blocks_changed"] and not rep["static_equal"]


def test_splice_refuses_a_group_present_in_both(tmp_path):
    with pytest.raises(ValueError, match="already present"):
        _splice(tmp_path, remove=())


def test_splice_refuses_unknown_removal(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _splice(tmp_path, remove=(("simA", 17, 2.8), ("simC", 1, 2.2)))


def test_splice_refuses_mismatched_static_tables(tmp_path):
    b = _write(tmp_path / "base.h5", BASE)
    s = _write(tmp_path / "shard.h5", SHARD, edges_shift=1e-9)
    with pytest.raises(ValueError, match="log_nhi_centres"):
        sp.splice_cache(
            b, [s], [("simA", 17, 2.8)], tmp_path / "out.h5", record={}, code_commit=COMMIT
        )


def test_splice_refuses_a_base_not_in_canonical_order(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        _splice(tmp_path, base=[BASE[1], BASE[0], BASE[2], BASE[3]])


def test_splice_refuses_to_overwrite_its_inputs(tmp_path):
    b = _write(tmp_path / "base.h5", BASE)
    s = _write(tmp_path / "shard.h5", SHARD)
    with pytest.raises(ValueError, match="overwrite"):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], b, record={}, code_commit=COMMIT)


# --- R1-2 (B-D9 #18): an existing output is refused before anything is opened; no "unknown" commit ----------------


def _inputs(tmp_path):
    return _write(tmp_path / "base.h5", BASE), _write(tmp_path / "shard.h5", SHARD)


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
def test_splice_refuses_an_existing_output_before_opening_anything(tmp_path, monkeypatch, kind):
    b, s = _inputs(tmp_path)
    out = tmp_path / "earlier_repaired.h5"
    if kind == "file":
        out.write_bytes(b"an earlier product")
    else:
        out.symlink_to(tmp_path / "nowhere.h5")

    def no_open(*a, **k):
        raise AssertionError("an input was opened before the output path was checked")

    monkeypatch.setattr(sp, "_load", no_open)
    with pytest.raises(FileExistsError, match="overwrite"):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], out, record={}, code_commit=COMMIT)
    if kind == "file":
        assert out.read_bytes() == b"an earlier product"
    else:
        assert out.is_symlink() and not (tmp_path / "nowhere.h5").exists()


def test_splice_opens_the_output_exclusively(tmp_path, monkeypatch):
    """Even if the path appears after the up-front check (a race), the write must not truncate it."""
    b, s = _inputs(tmp_path)
    out = tmp_path / "out.h5"
    real_load = sp._load

    def load_then_race(path, *a):
        if not out.exists():
            out.write_bytes(b"appeared meanwhile")
        return real_load(path, *a)

    monkeypatch.setattr(sp, "_load", load_then_race)
    with pytest.raises((FileExistsError, OSError)):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], out, record={}, code_commit=COMMIT)
    assert out.read_bytes() == b"appeared meanwhile"


@pytest.mark.parametrize("commit", [None, "", "unknown"])
def test_splice_refuses_without_a_code_commit(tmp_path, commit):
    b, s = _inputs(tmp_path)
    out = tmp_path / "out.h5"
    with pytest.raises(ValueError, match="code commit"):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], out, record={}, code_commit=commit)
    assert not out.exists()


def test_splice_records_the_code_commit(tmp_path):
    b, s, out = _splice(tmp_path)
    d, a = _read(out)
    assert a["git_sha"] == COMMIT and json.loads(a["repair_record"])["code_commit"] == COMMIT


def _main(monkeypatch, tmp_path, out, extra=()):
    b, s = _inputs(tmp_path)
    rj = tmp_path / "record.json"
    rj.write_text(json.dumps({"spec": "test"}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "splice_tau0_cache.py",
            "--base",
            str(b),
            "--shard",
            str(s),
            "--remove",
            "simA:17:2.8",
            "--record-json",
            str(rj),
            "--output",
            str(out),
            *extra,
        ],
    )
    return sp.main()


def test_main_without_code_commit_records_the_git_head(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_git_head", lambda: COMMIT)
    out = tmp_path / "out.h5"
    _main(monkeypatch, tmp_path, out)
    assert _read(out)[1]["git_sha"] == COMMIT


def test_main_refuses_when_no_commit_can_be_determined(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_git_head", lambda: None)
    out = tmp_path / "out.h5"
    with pytest.raises(SystemExit, match="code-commit"):
        _main(monkeypatch, tmp_path, out)
    assert not out.exists()


def test_main_refuses_an_existing_output_before_reading_its_inputs(tmp_path, monkeypatch):
    out = tmp_path / "out.h5"
    out.write_bytes(b"keep")
    monkeypatch.setattr(sp, "_git_head", lambda: COMMIT)
    monkeypatch.setattr(
        sp, "_load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("input opened"))
    )
    with pytest.raises(SystemExit, match="exists"):
        _main(monkeypatch, tmp_path, out)
    assert out.read_bytes() == b"keep"


# ----------------------------------------------------------------------------------------------------- lineage
def test_the_splice_output_is_eligible_and_lists_the_kept_and_added_inputs(tmp_path):
    b, s, out = _splice(tmp_path)
    d, attrs, rec = T.read_cache(out)  # the release reader accepts it
    roles = {i["role"] for i in rec["inputs"]}
    assert {"base", "shard:0"} <= roles and rec["release_eligible"]
    assert (
        T.group_role("catalog", "simA", 16) in roles
        and T.group_role("catalog", "simA", 18) in roles
    )
    shard_rec = T.read_cache(s)[2]
    base_rec = T.read_cache(b)[2]
    by_role = {i["role"]: i for i in rec["inputs"]}
    assert (
        by_role[T.group_role("catalog", "simA", 17)]
        == {i["role"]: i for i in shard_rec["inputs"]}[T.group_role("catalog", "simA", 17)]
    )  # the removed group's inputs are gone
    assert (
        by_role[T.group_role("catalog", "simA", 17)]
        != {i["role"]: i for i in base_rec["inputs"]}[T.group_role("catalog", "simA", 17)]
    )
    assert [(g["sim"], g["snap"]) for g in rec["product"]["groups"]] == [
        ("simA", 16),
        ("simA", 17),
        ("simA", 18),
        ("simA", 19),
        ("simB", 5),
    ]


def test_splice_refuses_historical_or_ineligible_inputs_unless_historical(tmp_path):
    tf.write_v33(tmp_path / "old.h5", BASE)
    s = _write(tmp_path / "shard.h5", SHARD)
    with pytest.raises(T.CacheLineageError, match="cache version '3.3'.*historical"):
        sp.splice_cache(
            tmp_path / "old.h5",
            [s],
            [("simA", 17, 2.8)],
            tmp_path / "o.h5",
            record={},
            code_commit=COMMIT,
        )
    bad = _write(tmp_path / "bad.h5", SHARD, eligible=False)
    b = _write(tmp_path / "base.h5", BASE)
    with pytest.raises(T.CacheLineageError, match="not release-eligible"):
        sp.splice_cache(
            b, [bad], [("simA", 17, 2.8)], tmp_path / "o.h5", record={}, code_commit=COMMIT
        )
    assert not (tmp_path / "o.h5").exists()
    sp.splice_cache(
        tmp_path / "old.h5",
        [s],
        [("simA", 17, 2.8)],
        tmp_path / "h.h5",
        record={},
        code_commit=COMMIT,
        historical_reason="a reproduction",
    )
    rec = T.read_cache_historical(tmp_path / "h.h5", reason="test")[2]
    assert not rec["release_eligible"] and any(
        "historical: a reproduction" in w for w in rec["ineligible_reasons"]
    )
    with pytest.raises(T.CacheLineageError, match="not release-eligible"):
        T.read_cache(tmp_path / "h.h5")


def test_merge_combines_lineage_and_refuses_what_it_must(tmp_path):
    a = _write(tmp_path / "a.h5", BASE[:2])
    b = _write(tmp_path / "b.h5", BASE[2:])
    rec = mg.merge_shards([a, b], tmp_path / "m.h5", expect_rows=12, expect_pairs=4)
    d, attrs, back = T.read_cache(tmp_path / "m.h5")
    assert back == rec and rec["release_eligible"] and int(attrs["merged_from_n_shards"]) == 2
    assert {"shard:0", "shard:1", T.group_role("cddf", "simB", 5)} <= {
        i["role"] for i in rec["inputs"]
    }
    assert [g["snap"] for g in rec["product"]["groups"]] == [16, 17, 19, 5]
    assert np.array_equal(d["snap_group_idx"], np.repeat(np.arange(4), 3))
    with pytest.raises(ValueError, match="more than one shard"):
        mg.merge_shards([a, a], tmp_path / "dup.h5")
    c = _write(tmp_path / "c.h5", BASE[2:], settings={"cache_version": T.CACHE_VERSION, "n_k": 99})
    with pytest.raises(ValueError, match="settings differ"):
        mg.merge_shards([a, c], tmp_path / "s.h5")
    old = tf.write_v33(tmp_path / "old.h5", BASE[2:])
    with pytest.raises(T.CacheLineageError, match="historical"):
        mg.merge_shards([a, old], tmp_path / "o.h5")
    hist = mg.merge_shards([a, old], tmp_path / "h.h5", historical_reason="a reproduction")
    assert not hist["release_eligible"]
    with pytest.raises(FileExistsError):
        mg.merge_shards([a, b], tmp_path / "m.h5")


def test_the_release_reader_refuses_an_edited_or_unrecorded_cache(tmp_path):
    p = _write(tmp_path / "a.h5", BASE)
    T.read_cache(p)
    with h5py.File(p, "r+") as f:
        f["P_tier_c"][0, 0, 0] = f["P_tier_c"][0, 0, 0] + 1.0
    with pytest.raises(T.CacheLineageError, match="payload digest"):
        T.read_cache(p)
    q = _write(tmp_path / "q.h5", BASE)
    with h5py.File(q, "r+") as f:
        f.attrs["tau_thresh"] = 1e5  # an attribute counts too
    with pytest.raises(T.CacheLineageError, match="payload digest"):
        T.read_cache(q)
    r = _write(tmp_path / "r.h5", BASE)
    with h5py.File(r, "r+") as f:
        del f[T.LINEAGE_KEY]
    with pytest.raises(T.CacheLineageError, match="no lineage record"):
        T.read_cache(r)
    with pytest.raises(ValueError, match="non-empty reason"):
        T.read_cache_historical(r, reason=" ")
    assert T.read_cache_historical(r, reason="audit")[2] is None


def test_carried_shards_are_never_merged_or_spliced_in(tmp_path):
    a = _write(tmp_path / "a.h5", BASE[:2])
    carried = _write(tmp_path / "carried.h5", BASE[2:], **tf.carry_kw())
    with pytest.raises(ValueError, match="carried rows is not merged"):
        mg.merge_shards([a, carried], tmp_path / "m.h5")
    base = _write(tmp_path / "base.h5", BASE[:3])
    shard = _write(tmp_path / "shard.h5", [("simC", 7, 3.4, 3, 30)], **tf.carry_kw())
    with pytest.raises(ValueError, match="carried rows is not spliced in"):
        sp.splice_cache(
            base, [shard], [], tmp_path / "s.h5", record={"spec": "test"}, code_commit=COMMIT
        )
    staged = _write(tmp_path / "staged.h5", BASE[2:], **tf.carry_kw(promoted=False))
    with pytest.raises(
        T.CacheLineageError, match="not yet promoted"
    ):  # a staged carry is read by nothing
        mg.merge_shards([a, staged], tmp_path / "m2.h5")


def test_splicing_into_a_carried_base_keeps_its_carry_and_validation(tmp_path):
    kw = tf.carry_kw()
    base = _write(tmp_path / "base.h5", BASE, **kw)
    shard = _write(tmp_path / "shard.h5", SHARD)
    out = tmp_path / "out.h5"
    sp.splice_cache(
        base, [shard], [("simA", 17, 2.8)], out, record={"spec": "test"}, code_commit=COMMIT
    )
    _d, _a, rec = T.read_cache(out)  # eligible, carry bound to its source
    _bd, _ba, brec = T.read_cache(base)
    assert rec["release_eligible"] and rec["product"]["carried"] == brec["product"]["carried"]
    assert rec["validation"] == brec["validation"]
    rows = {(g["sim"], g["snap"]): g["rows"] for g in rec["product"]["groups"]}
    assert rows[("simA", 17)] == rows[("simA", 18)] == T.ROWS_BUILT
    assert rows[("simA", 16)] == rows[("simB", 5)] == T.ROWS_CARRIED
    assert [i for i in rec["inputs"] if i["role"] == tf.SOURCE_ROLE] == [kw["extra_inputs"][0]]


def test_the_release_reader_refuses_a_record_whose_counts_are_not_the_files(tmp_path):
    p = _write(tmp_path / "a.h5", BASE)
    with h5py.File(p, "r+") as f:
        rec = json.loads(f[T.LINEAGE_KEY].asstr()[()])
        rec["product"]["n_rows"] += 1
        del f[T.LINEAGE_KEY]
        f.create_dataset(T.LINEAGE_KEY, data=json.dumps(rec), dtype=h5py.string_dtype())
    with pytest.raises(T.CacheLineageError, match="row or group count"):
        T.read_cache(p)
