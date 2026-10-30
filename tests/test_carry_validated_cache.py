"""The R1 carry (scripts/carry_validated_cache.py): historical P1D rows enter a release cache only after every group's
label tie, raw identity and metadata checks pass, with every count field rebuilt; any failure writes no cache.
Synthetic world (tests/tau0_world.py): made-up optical depths, pipeline-built catalogue-only snapshots, pre-fix
snapshots with box-edge duplicates, a historical cache built from them."""
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as T
from hcd_analysis.paths import REPO_ROOT
from tests import tau0_world as W

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import carry_validated_cache as CV  # noqa: E402

IDENTICAL = {"identical": True, "items": [], "diff_sha256": "0" * 64}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        yield W.make_world(tmp_path_factory.mktemp("world"), mp)


@pytest.fixture
def same_code(monkeypatch):
    """The row-path diff against the builder commits of record, as an empty one (tested on a real repository below)."""
    monkeypatch.setattr(CV, "row_path_diff", lambda repo, a, b, spec=CV.ROW_PATH: (dict(IDENTICAL, old_commit=a,
                                                                                         new_commit=b), ""))


def carry(w, out_dir, *, hist=None, old=None, rebuilt=None, emu=None, extra=()):
    args = ["--historical-cache", hist or w["hist"], "--old-root", old or w["old"], "--rebuilt-root",
            rebuilt or w["rebuilt"], "--raw-root", emu or w["emu"], "--row-path-commit", "abc1234",
            "--validation-dir", out_dir / "val", "--output", out_dir / "carried.h5", *extra]
    return CV.main([str(a) for a in args])


def summary(out_dir):
    return json.loads((out_dir / "val" / "summary.json").read_text())


def failed_checks(out_dir):
    out = set()
    for p in (out_dir / "val" / "groups").iterdir():
        out |= {k for k, c in json.loads(p.read_text())["checks"].items() if not c["ok"]}
    return out


def test_the_carry_copies_validated_rows_and_rebuilds_every_count(world, tmp_path, same_code):
    assert carry(world, tmp_path) == 0
    out = tmp_path / "carried.h5"
    d, attrs, rec = T.read_cache(out)                                             # the release reader accepts it
    src, _sa, _ = T.read_cache_historical(world["hist"], reason="test")
    assert rec["release_eligible"] and attrs["cache_version"] == T.CACHE_VERSION
    assert [g["rows"] for g in rec["product"]["groups"]] == [T.ROWS_CARRIED] * 3
    for k in T.CARRIED_FIELDS:                                                    # copied bit for bit
        assert np.array_equal(d[k], src[k], equal_nan=True) and d[k].dtype == src[k].dtype
    for k in ("tier_c_counts", "params", "z_meta", "z_grid", "dv_kms", "alpha_slope", "snap_total_path_dX"):
        assert np.array_equal(d[k], src[k])                                       # rebuilt, equal to the stored
    # counts rebuilt from the rebuilt catalogues: the old ones less the box-edge duplicates
    for b, (sim, snap) in enumerate(zip(d["snap_sim_name"], d["snap_snap"])):
        cddf, _ = L.read_npz(world["rebuilt"] / sim / f"snap_{int(snap):03d}" / "cddf.npz")
        assert np.array_equal(d["snap_n_absorbers"][b], cddf["n_absorbers"])
    assert int(src["snap_n_absorbers"].sum()) - int(d["snap_n_absorbers"].sum()) == 6        # two heads per group
    car = rec["product"]["carried"]
    assert car["source_sha256"] == L.sha256_file(world["hist"]) and car["fields"] == list(T.CARRIED_FIELDS)
    src_in = [i for i in rec["inputs"] if i["role"] == CV.SOURCE_ROLE]
    assert len(src_in) == 1 and "release_eligible" not in src_in[0]              # named as validated, not as eligible
    assert rec["validation"]["validated_inputs"] == [CV.SOURCE_ROLE] and rec["validation"]["all_passed"]
    assert rec["validation"]["summary_sha256"] == L.sha256_file(tmp_path / "val" / "summary.json")
    s = summary(tmp_path)
    assert s["passed"] and s["n_groups"] == 3 and all(s["global_checks"].values())
    assert s["origins"]["spliced"] == [f"{W.SIMS[0]}:5"]
    for name, sha in s["group_records_sha256"].items():
        g = json.loads((tmp_path / "val" / name).read_text())
        assert L.sha256_file(tmp_path / "val" / name) == sha and g["ok"] and len(g["checks"]) == 12
    from hcd_analysis.emulator.data import load_cache
    loaded = load_cache(out)
    assert loaded["cache_sha256"] == L.sha256_file(out) and loaded["lineage"] == rec


def _copy(world, tmp_path, *names):
    """Copies of parts of the world, for damage."""
    w = dict(world)
    for n in names:
        w[n] = W.copy_tree(world[n], tmp_path / n)
    return w


def _v33_with(world, tmp_path, key, change):
    src, attrs, _ = T.read_cache_historical(world["hist"], reason="test")
    p = tmp_path / "hist_edit.h5"
    shutil.copy2(world["hist"], p)
    with h5py.File(p, "r+") as f:
        a = f[key][...]
        change(a)
        f[key][...] = a
    return p


@pytest.mark.parametrize("damage, check", [
    ("labels", "labels_old_equal_rebuilt"),
    ("tier_c_counts", "rebuilt_counts_equal_stored_tier_c_counts"),
    ("snap_counts", "stored_counts_equal_old_catalogue"),
    ("raw_touched", "raw_predates_old_catalogue_and_source"),
    ("raw_bytes", "inputs"),
    ("rebuilt_dirty", "inputs"),
    ("old_header", "raw_header_equals_old_meta"),
    ("params", "row_metadata_recomputed_equal_stored"),
])
def test_any_failed_check_writes_no_cache(world, tmp_path, same_code, monkeypatch, damage, check):
    kw = {}
    if damage == "labels":                       # a head above its own system changes a sightline's label
        w = _copy(world, tmp_path, "old")
        sim, snap, _z = W.GROUPS[0]
        nd = world["rebuilt"] / sim / f"snap_{snap:03d}"
        shutil.rmtree(w["old"] / sim / f"snap_{snap:03d}")
        W.write_old_snapshot(w["old"], nd, world["raws"][(sim, snap)], head_factor=300.0)
        kw["old"] = w["old"]
    elif damage == "tier_c_counts":
        kw["hist"] = _v33_with(world, tmp_path, "tier_c_counts", lambda a: a.__setitem__((5, 0), a[5, 0] + 1))
    elif damage == "snap_counts":
        kw["hist"] = _v33_with(world, tmp_path, "snap_n_absorbers", lambda a: a.__setitem__((1, 3), a[1, 3] + 1))
    elif damage == "params":
        kw["hist"] = _v33_with(world, tmp_path, "params", lambda a: a.__setitem__((0, 0), a[0, 0] * (1 + 1e-15)))
    elif damage in ("raw_touched", "raw_bytes"):
        w = _copy(world, tmp_path, "emu")
        sim, snap, _z = W.GROUPS[1]
        raw = w["emu"] / sim / "output" / f"SPECTRA_{snap:03d}" / "lya_forest_spectra_grid_480.hdf5"
        if damage == "raw_touched":
            os.utime(raw)                        # the same bytes, modified after the old catalogue was written
        else:
            with h5py.File(raw, "r+") as f:
                f["tau/H/1/1215"][3, 7] = f["tau/H/1/1215"][3, 7] + 0.01
            os.utime(raw, ns=(1, 1))
        kw["emu"] = w["emu"]
    elif damage == "rebuilt_dirty":
        w = _copy(world, tmp_path, "rebuilt")
        sim, snap, z = W.GROUPS[2]
        shutil.rmtree(w["rebuilt"] / sim)
        monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: {"commit": "c" * 40, "dirty": True})
        W.rebuild_snapshot(w["rebuilt"], world["emu"], sim, snap, z)
        monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(W.CLEAN))
        kw["rebuilt"] = w["rebuilt"]
    elif damage == "old_header":
        w = _copy(world, tmp_path, "old")
        sim, snap, _z = W.GROUPS[0]
        m = w["old"] / sim / f"snap_{snap:03d}" / "meta.json"
        meta = json.loads(m.read_text())
        meta["box_kpc_h"] = meta["box_kpc_h"] * 2
        m.write_text(json.dumps(meta))
        kw["old"] = w["old"]
    assert carry(world, tmp_path, **kw) == 2
    assert not (tmp_path / "carried.h5").exists()
    assert not summary(tmp_path)["passed"] and check in failed_checks(tmp_path)


def test_a_changed_row_path_passes_only_with_its_reviewed_digest(world, tmp_path, monkeypatch):
    changed = dict(IDENTICAL, identical=False, items=[{"item": "x", "status": "changed"}], diff_sha256="d" * 64)
    monkeypatch.setattr(CV, "row_path_diff", lambda repo, a, b, spec=CV.ROW_PATH: (dict(changed, old_commit=a,
                                                                                        new_commit=b), "- a\n+ b\n"))
    assert carry(world, tmp_path / "a") == 2
    s = summary(tmp_path / "a")
    assert not s["global_checks"]["row_path_code"] and (tmp_path / "a" / "val" / "row_path").is_dir()
    assert carry(world, tmp_path / "b", extra=["--accept-row-path-diff", "abc1234=" + "e" * 64]) == 2
    assert carry(world, tmp_path / "c", extra=["--accept-row-path-diff", "abc1234=" + "d" * 64]) == 0
    rec = T.read_cache(tmp_path / "c" / "carried.h5")[2]
    assert rec["product"]["carried"]["row_path"][0]["accepted_sha256"] == "d" * 64


def test_the_carry_runs_only_from_a_clean_tree_into_new_places(world, tmp_path, same_code, monkeypatch, capsys):
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: {"commit": "c" * 40, "dirty": True})
    assert carry(world, tmp_path) == 2 and "clean, committed" in capsys.readouterr().out
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(W.CLEAN))
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "x").write_text("")
    assert carry(world, tmp_path) == 2 and "not empty" in capsys.readouterr().out
    (tmp_path / "val" / "x").unlink()
    (tmp_path / "carried.h5").write_text("")
    assert carry(world, tmp_path) == 2 and "never overwritten" in capsys.readouterr().out


def test_a_release_cache_is_never_a_carry_source(world, tmp_path, same_code):
    assert carry(world, tmp_path / "a") == 0
    assert carry(world, tmp_path / "b", hist=tmp_path / "a" / "carried.h5") == 2
    assert not summary(tmp_path / "b")["global_checks"]["source_is_version_3.3"]


# ------------------------------------------------------------------------------------- the row-path diff, for real
def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_row_path_diff_on_a_real_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.org")
    _git(repo, "config", "user.name", "test")
    spec = (("pkg/b.py", ("rows", "K", "C.m")), ("pkg/whole.py", None))
    (repo / "pkg" / "b.py").write_text("K = 1\n\n\ndef rows(x):\n    return x + K\n\n\ndef other():\n    return 0\n\n\n"
                                       "class C:\n    def m(self):\n        return 1\n")
    (repo / "pkg" / "whole.py").write_text("A = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "pkg" / "b.py").write_text((repo / "pkg" / "b.py").read_text().replace("return 0", "return 9"))
    _git(repo, "commit", "-q", "-am", "two: outside the row path")
    res, text = CV.row_path_diff(repo, first[:7], "HEAD", spec)
    assert res["identical"] and text == "" and res["old_commit"] == first
    (repo / "pkg" / "b.py").write_text((repo / "pkg" / "b.py").read_text().replace("x + K", "x - K"))
    (repo / "pkg" / "whole.py").write_text("A = 2\n")
    _git(repo, "commit", "-q", "-am", "three: inside")
    res, text = CV.row_path_diff(repo, first, "HEAD", spec)
    status = {i["item"]: i["status"] for i in res["items"]}
    assert not res["identical"] and status == {"pkg/b.py:rows": "changed", "pkg/b.py:K": "identical",
                                               "pkg/b.py:C.m": "identical", "pkg/whole.py": "changed"}
    assert "-    return x + K" in text and "+    return x - K" in text and "+A = 2" in text
    again, _ = CV.row_path_diff(repo, first, "HEAD", spec)
    assert again["diff_sha256"] == res["diff_sha256"]
    res, _ = CV.row_path_diff(repo, first, "HEAD", (("pkg/b.py", ("gone",)),))
    assert res["items"] == [{"item": "pkg/b.py:gone", "status": "absent at both"}] and not res["identical"]


def test_the_row_path_spec_names_code_that_exists_at_head():
    for path, symbols in CV.ROW_PATH:
        src = (REPO_ROOT / path).read_text()
        for sym in symbols or ():
            assert CV._symbol_source(src, sym) is not None, (path, sym)
