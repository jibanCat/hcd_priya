"""The catalogue acceptance (scripts/accept_rebuilt_catalogues.py): a rebuilt catalogue is accepted only when it is
the old one minus its box-edge duplicates, with the trace's duplicate counts, identical labels, and counts and CDDF of
the old minus the duplicates. Synthetic world (tests/tau0_world.py)."""
import json
import shutil
import sys

import pytest

from hcd_analysis import lineage as L
from hcd_analysis.paths import REPO_ROOT
from tests import tau0_world as W

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import accept_rebuilt_catalogues as AC  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        w = W.make_world(tmp_path_factory.mktemp("world"), mp)
        (w["base"] / "groups.txt").write_text("".join(f"{s} {n}\n" for s, n, _z in W.GROUPS))
        yield w


def check(w, out, *, old=None, trace=None, extra=()):
    return AC.main([str(a) for a in ("check", "--groups", w["base"] / "groups.txt", "--old-root", old or w["old"],
                                     "--new-root", w["rebuilt"], "--raw-root", w["emu"], "--trace-rows",
                                     trace or w["trace"], "--out", out, *extra)])


def records(out):
    return {p.name: json.loads(p.read_text()) for p in sorted((out / "groups").iterdir())}


def summarize(w, out, *, trace=None, extra=()):
    return AC.main([str(a) for a in ("summarize", "--groups", w["base"] / "groups.txt", "--trace-rows",
                                     trace or w["trace"], "--out", out, *extra)])


def test_the_rebuilt_world_is_accepted_group_by_group_and_summarized(world, tmp_path):
    assert check(world, tmp_path, extra=["--rehash-raw"]) == 0
    recs = records(tmp_path)
    assert len(recs) == len(W.GROUPS) and all(r["accepted"] for r in recs.values())
    r = next(iter(recs.values()))
    assert r["checks"]["release_loader"]["rehashed"] and r["checks"]["entries_old_minus_duplicates"]["old_duplicates"] == 2
    assert r["checks"]["nhi_bitwise_or_within_one_ulp"]["bitwise"] == r["checks"]["entries_old_minus_duplicates"]["new"]
    assert r["info"]["old_cddf_equals_old_catalogue_histogram"]
    assert summarize(world, tmp_path) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["all_accepted"] and s["n_accepted"] == 3 and len(s["code_identities"]) == 1
    assert sum(s["duplicates_per_class_total"].values()) == 6
    assert s["trace_rows_sha256"] == L.sha256_file(world["trace"]) and len(s["acceptance_code_identities"]) == 1
    inp = r["inputs"]                                    # what the group was checked against
    sim, snap = r["sim"], r["snap"]
    assert inp["trace_sha256"] == L.sha256_file(world["trace"])
    assert inp["old_catalogue"]["sha256"] == L.sha256_file(world["old"] / sim / f"snap_{snap:03d}" / "catalog.npz")
    assert inp["old_catalogue"]["has_record"] is False and len(inp["old_meta_sha256"]) == 64
    assert inp["roots"] == {"old": str(world["old"]), "new": str(world["rebuilt"]), "raw": str(world["emu"])}


def test_one_array_task_checks_one_line(world, tmp_path):
    assert check(world, tmp_path, extra=["--index", "2"]) == 0
    sim, snap, _z = W.GROUPS[1]
    assert list(records(tmp_path)) == [f"{sim}__snap_{snap:03d}.json"]
    assert summarize(world, tmp_path) == 1
    assert len(json.loads((tmp_path / "summary.json").read_text())["missing"]) == 2


def test_label_trace_and_count_mismatches_are_refused(world, tmp_path):
    sim, snap, _z = W.GROUPS[0]
    old = W.copy_tree(world["old"], tmp_path / "old")
    shutil.rmtree(old / sim / f"snap_{snap:03d}")                      # a head above its own system
    W.write_old_snapshot(old, world["rebuilt"] / sim / f"snap_{snap:03d}", world["raws"][(sim, snap)],
                         head_factor=300.0)
    assert check(world, tmp_path / "a", old=old) == 1
    bad = records(tmp_path / "a")[f"{sim}__snap_{snap:03d}.json"]["checks"]
    assert not bad["sightline_labels_identical"]["ok"] and bad["integer_columns_exact_in_order"]["ok"]
    rows = json.loads(world["trace"].read_text())
    rows[1]["dups"]["LLS"] += 1
    (tmp_path / "trace.json").write_text(json.dumps(rows))
    assert check(world, tmp_path / "b", trace=tmp_path / "trace.json") == 1
    s2, n2, _z2 = W.GROUPS[1]
    assert not records(tmp_path / "b")[f"{s2}__snap_{n2:03d}.json"]["checks"]["duplicates_per_class_equal_trace"]["ok"]
    old2 = W.copy_tree(world["old"], tmp_path / "old2")
    m = old2 / sim / f"snap_{snap:03d}" / "meta.json"
    meta = json.loads(m.read_text())
    meta["n_absorbers"]["LLS"] += 1
    m.write_text(json.dumps(meta))
    assert check(world, tmp_path / "c", old=old2) == 1
    assert not records(tmp_path / "c")[f"{sim}__snap_{snap:03d}.json"]["checks"][
        "meta_counts_equal_old_minus_duplicates"]["ok"]


def test_a_rebuilt_snapshot_not_made_with_the_release_settings_is_refused(world, tmp_path):
    sim, snap, z = W.GROUPS[2]
    new = W.copy_tree(world["rebuilt"], tmp_path / "new")
    shutil.rmtree(new / sim)
    W.rebuild_snapshot(tmp_path / "other", world["emu"], sim, snap, z, min_pixels=3)   # its own root, then copied
    shutil.copytree(tmp_path / "other" / sim, new / sim)
    w = dict(world, rebuilt=new)
    assert check(w, tmp_path / "out") == 1
    rec = records(tmp_path / "out")[f"{sim}__snap_{snap:03d}.json"]
    assert not rec["accepted"] and "other finder settings" in rec["checks"]["release_loader"]["error"]


def _edit_record(out, i, fn):
    p = sorted((out / "groups").iterdir())[i]
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))


def test_summarize_requires_rehash_one_clean_checker_one_trace_and_the_expected_commit(world, tmp_path):
    assert check(world, tmp_path / "plain") == 0                     # accepted, but the raw files were not re-hashed
    assert summarize(world, tmp_path / "plain") == 1
    assert len(json.loads((tmp_path / "plain" / "summary.json").read_text())["not_rehashed"]) == 3
    out = tmp_path / "ok"
    assert check(world, out, extra=["--rehash-raw"]) == 0
    assert summarize(world, out, extra=["--expect-commit", "c" * 7]) == 0
    assert summarize(world, out, extra=["--expect-commit", "d" * 7]) == 1
    assert len(json.loads((out / "summary.json").read_text())["off_commit"]) == 3
    rows = json.loads(world["trace"].read_text())
    rows[0]["dups"]["LLS"] += 1
    (tmp_path / "other_trace.json").write_text(json.dumps(rows))
    assert summarize(world, out, trace=tmp_path / "other_trace.json") == 1             # another trace file
    assert json.loads((out / "summary.json").read_text())["trace_mismatch"]
    for i, fn in enumerate((lambda r: r.__setitem__("code", {"commit": "e" * 40, "dirty": False}),   # two checkers
                            lambda r: r.__setitem__("code", {"commit": "c" * 40, "dirty": True}))):
        o = tmp_path / f"id{i}"
        shutil.copytree(out, o)
        if i == 1:
            for j in range(3):
                _edit_record(o, j, fn)                                  # one identity, but dirty
        else:
            _edit_record(o, 0, fn)
        assert summarize(world, o) == 1, i
    o = tmp_path / "older"                                              # a record written before records named the
    shutil.copytree(out, o)                                             # trace: checked by its duplicate counts
    _edit_record(o, 0, lambda r: r.pop("inputs"))
    assert summarize(world, o) == 0
    _edit_record(o, 0, lambda r: r["checks"]["duplicates_per_class_equal_trace"]["trace"].__setitem__("LLS", 99))
    assert summarize(world, o) == 1
