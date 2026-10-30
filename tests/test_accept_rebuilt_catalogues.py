"""The catalogue acceptance (scripts/accept_rebuilt_catalogues.py): a rebuilt catalogue is accepted only when it is
the old one minus its box-edge duplicates, with the trace's duplicate counts, identical labels, and counts and CDDF of
the old minus the duplicates. Synthetic world (tests/tau0_world.py)."""
import json
import shutil
import sys

import pytest

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


def test_the_rebuilt_world_is_accepted_group_by_group_and_summarized(world, tmp_path):
    assert check(world, tmp_path, extra=["--rehash-raw"]) == 0
    recs = records(tmp_path)
    assert len(recs) == len(W.GROUPS) and all(r["accepted"] for r in recs.values())
    r = next(iter(recs.values()))
    assert r["checks"]["release_loader"]["rehashed"] and r["checks"]["entries_old_minus_duplicates"]["old_duplicates"] == 2
    assert r["checks"]["nhi_bitwise_or_within_one_ulp"]["bitwise"] == r["checks"]["entries_old_minus_duplicates"]["new"]
    assert r["info"]["old_cddf_equals_old_catalogue_histogram"]
    assert AC.main(["summarize", "--groups", str(world["base"] / "groups.txt"), "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["all_accepted"] and s["n_accepted"] == 3 and len(s["code_identities"]) == 1
    assert sum(s["duplicates_per_class_total"].values()) == 6


def test_one_array_task_checks_one_line(world, tmp_path):
    assert check(world, tmp_path, extra=["--index", "2"]) == 0
    sim, snap, _z = W.GROUPS[1]
    assert list(records(tmp_path)) == [f"{sim}__snap_{snap:03d}.json"]
    assert AC.main(["summarize", "--groups", str(world["base"] / "groups.txt"), "--out", str(tmp_path)]) == 1
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
