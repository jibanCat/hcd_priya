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
import promote_carried_cache as PC  # noqa: E402

IDENTICAL = {"identical": True, "items": [], "diff_sha256": "0" * 64}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        yield W.make_world(tmp_path_factory.mktemp("world"), mp)


@pytest.fixture(autouse=True)
def of_record(world, monkeypatch):
    """The synthetic world's historical cache and builder commit as the tool's source and commits of record."""
    W.pin_of_record(monkeypatch, CV, world["hist"])


@pytest.fixture
def same_code(monkeypatch):
    """The row-path diff against the builder commits of record, as an empty one (tested on a real repository below)."""
    monkeypatch.setattr(CV, "row_path_diff", lambda repo, a, b, spec=CV.ROW_PATH: (dict(IDENTICAL, old_commit=a,
                                                                                         new_commit=b), ""))


def carry(w, out_dir, *, hist=None, old=None, rebuilt=None, emu=None, extra=(), rehash=True,
          commits=(W.BUILDER_COMMIT,)):
    args = ["--historical-cache", hist or w["hist"], "--old-root", old or w["old"], "--rebuilt-root",
            rebuilt or w["rebuilt"], "--raw-root", emu or w["emu"], *[x for c in commits for x in
                                                                      ("--row-path-commit", c)],
            "--validation-dir", out_dir / "val", "--output", out_dir / "carried.h5", *extra,
            *(["--rehash-raw"] if rehash else [])]
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
    with pytest.raises(T.CacheLineageError, match="not yet promoted"):          # staged: the release reader refuses
        T.read_cache(out)
    d, attrs, rec = T.read_cache_staged(out)
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
    assert s["of_record"]["source_sha256"] == L.sha256_file(world["hist"])
    assert s["of_record"]["repair_builder_commit"].split()[0] == W.BUILDER_COMMIT
    for name, sha in s["group_records_sha256"].items():
        g = json.loads((tmp_path / "val" / name).read_text())
        assert L.sha256_file(tmp_path / "val" / name) == sha and g["ok"] and len(g["checks"]) == 12
        dg = g["digests"]                                    # enough to audit the record without the catalogues
        assert set(dg["carried_slices_sha256"]) == set(T.CARRIED_FIELDS)
        assert dg["labels_old_sha256"] == dg["labels_rebuilt_sha256"]
        assert dg["old_meta_sha256"] == L.sha256_file(world["old"] / g["sim"] / f"snap_{g['snap']:03d}" / "meta.json")
        rows = np.arange(g["rows"][0], g["rows"][1] + 1)
        assert dg["carried_slices_sha256"]["scale"] == L.arrays_sha256({"scale": src["scale"][rows]})
        assert g["checks"]["raw_matches_rebuilt_record"]["evidence"] == "re-hashed by this run"
    from hcd_analysis.emulator.data import load_cache
    with pytest.raises(T.CacheLineageError, match="not yet promoted"):
        load_cache(out)
    assert not list(tmp_path.glob(".*unchecked"))


def _prereg(world, out, hist=None):
    """A pre-registration written by fresh_row_sample.py select (the strata of the synthetic historical cache)."""
    import fresh_row_sample as FR
    FR.main(["select", "--cache", str(hist or world["hist"]), "--seed", "5", "--n-total", "2", "--out", str(out)])
    return out, L.sha256_file(out)


def _report(tmp_path, staged, prereg, name="cmp", *, edit=None):
    """A comparison report as fresh_row_sample.py compare writes it (fresh files: stand-in bytes, listed by sha256);
    ``edit(report)`` damages it before it is written."""
    pre = json.loads(prereg.read_text())
    fresh, groups = [], []
    for i, g in enumerate(pre["groups"]):
        f = tmp_path / f"{name}_fresh{i}.h5"
        f.write_bytes(f"fresh rows {i}".encode())
        fresh.append({"path": str(f), "sha256": L.sha256_file(f)})
        groups.append({"sim": g["sim"], "snap": g["snap"], "ok": True, "same_rebuilt_catalogue": True,
                       "fresh_file": str(f), "n_rows": [20, 20], "fields": {}})
    rep = {"passed": True, "groups": groups, "fields": {}, "criterion": pre["criterion"],
           "source_sha256": pre["historical_cache"]["sha256"],
           "prereg": {"path": str(prereg), "sha256": L.sha256_file(prereg)},
           "carried": {"path": str(staged), "sha256": L.sha256_file(staged)}, "fresh": fresh, "code": dict(W.CLEAN)}
    if edit is not None:
        edit(rep)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(rep))
    return path, L.sha256_file(path)


def _promote(staged, rep, sha, out, prereg):
    return PC.main(["--staged", str(staged), "--comparison", str(rep), "--comparison-sha256", sha,
                    "--prereg", str(prereg), "--output", str(out)])


def test_the_pre_registration_of_record_is_pinned():
    import ast
    tree = ast.parse((REPO_ROOT / "scripts" / "promote_carried_cache.py").read_text())
    pins = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
            if isinstance(t, ast.Name) and t.id == "PREREG_OF_RECORD_SHA256"}
    assert pins == {"PREREG_OF_RECORD_SHA256": "d26b210c1a1249eb60e475ab9c9d28f9851c57e35bc2536c9e78506b6c132ee0"}


def test_a_forged_report_is_refused_by_the_pinned_pre_registration(world, tmp_path, same_code, capsys):
    """The four-key report: passed, one ok group, the staged sha256, a pre-registration digest of b's."""
    assert carry(world, tmp_path) == 0
    staged = tmp_path / "carried.h5"
    prereg, _ = _prereg(world, tmp_path / "prereg.json")
    forged = {"passed": True, "groups": [{"sim": W.GROUPS[0][0], "snap": W.GROUPS[0][1], "ok": True}],
              "carried": {"sha256": L.sha256_file(staged)}, "prereg": {"sha256": "b" * 64}}
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    assert _promote(staged, tmp_path / "forged.json", L.sha256_file(tmp_path / "forged.json"), tmp_path / "f.h5",
                    prereg) == 2
    assert "not the one of record" in capsys.readouterr().out and not (tmp_path / "f.h5").exists()


def test_promotion_needs_the_passed_comparison_bound_to_the_pre_registration(world, tmp_path, same_code, monkeypatch,
                                                                             capsys):
    assert carry(world, tmp_path) == 0
    staged = tmp_path / "carried.h5"
    prereg, pre_sha = _prereg(world, tmp_path / "prereg.json")
    monkeypatch.setattr(PC, "PREREG_OF_RECORD_SHA256", pre_sha)      # the synthetic pre-registration is of record
    rep, sha = _report(tmp_path, staged, prereg)
    final = tmp_path / "final.h5"
    assert _promote(staged, rep, sha, final, prereg) == 0
    d, _a, rec = T.read_cache(final)                                             # the release reader accepts it
    _s, _sa, srec = T.read_cache_staged(staged)
    assert rec["product"]["payload_sha256"] == srec["product"]["payload_sha256"]
    report = json.loads(rep.read_text())
    assert rec["product"]["carried"]["promoted"] == {
        "passed": True, "comparison_sha256": sha, "staged_sha256": L.sha256_file(staged), "prereg_sha256": pre_sha,
        "fresh_sha256": sorted(f["sha256"] for f in report["fresh"]), "n_groups_compared": len(report["groups"])}
    from hcd_analysis.emulator.data import load_cache
    assert load_cache(final, validate=False)["cache_sha256"] == L.sha256_file(final)
    capsys.readouterr()
    other_hist = tmp_path / "other_hist.h5"
    shutil.copy2(world["hist"], other_hist)
    with h5py.File(other_hist, "r+") as f:
        f.attrs["created_utc"] = "2026-01-01T00:00:00Z"                      # another source file
    other_prereg, _ = _prereg(world, tmp_path / "prereg_other.json", hist=other_hist)
    bad = {
        "four keys, the pre-registration's digest": (lambda r: [r.pop(k) for k in ("criterion", "fresh", "code",
                                                                                   "source_sha256")]
                                                     and r.update(groups=r["groups"][:1]), "criterion"),
        "one group": (lambda r: r.update(groups=r["groups"][:1]), "groups, not exactly the"),
        "another group": (lambda r: r["groups"][1].update(snap=99), "groups, not exactly the"),
        "failed": (lambda r: r.update(passed=False), "did not pass"),
        "group not ok": (lambda r: r["groups"][0].update(ok=False), "did not pass"),
        "another catalogue": (lambda r: r["groups"][0].update(same_rebuilt_catalogue=False), "did not pass"),
        "criterion": (lambda r: r["criterion"].update(rtol=1e-9), "criterion"),
        "another prereg": (lambda r: r["prereg"].update(sha256="b" * 64), "not the one of record"),
        "another cache": (lambda r: r["carried"].update(sha256="1" * 64), "not the staged file"),
        "source": (lambda r: r.update(source_sha256="2" * 64), "names the source"),
        "fresh unlisted": (lambda r: r.update(fresh=[]), "fresh files"),
        "fresh without sha": (lambda r: r["fresh"][0].pop("sha256"), "fresh files"),
        "fresh changed": (lambda r: r["fresh"][0].update(sha256="3" * 64), "missing or not the one"),
        "group fresh file": (lambda r: r["groups"][0].update(fresh_file="elsewhere.h5"), "fresh list"),
        "dirty report": (lambda r: r.update(code={"commit": "c" * 40, "dirty": True}), "clean, committed"),
        "no report code": (lambda r: r.pop("code"), "clean, committed"),
    }
    for name, (edit, match) in bad.items():
        key = name.replace(" ", "_").replace(",", "").replace("'", "")
        r, h = _report(tmp_path, staged, prereg, key, edit=edit)
        assert _promote(staged, r, h, tmp_path / f"{key}.h5", prereg) == 2, name
        out = capsys.readouterr().out
        assert match in out, (name, out)
        assert not (tmp_path / f"{key}.h5").exists()
    # the pre-registration file itself: edited, or one naming another source
    edited = tmp_path / "prereg_edited.json"
    edited.write_text(prereg.read_text().replace('"seed": 5', '"seed": 6'))
    assert _promote(staged, rep, sha, tmp_path / "e.h5", edited) == 2
    assert "not the pre-registration of record" in capsys.readouterr().out
    monkeypatch.setattr(PC, "PREREG_OF_RECORD_SHA256", L.sha256_file(other_prereg))
    r, h = _report(tmp_path, staged, other_prereg, "other_source")
    assert _promote(staged, r, h, tmp_path / "o.h5", other_prereg) == 2
    assert "names the source" in capsys.readouterr().out
    monkeypatch.setattr(PC, "PREREG_OF_RECORD_SHA256", pre_sha)
    assert _promote(final, rep, sha, tmp_path / "again.h5", prereg) == 2       # already promoted
    assert _promote(staged, rep, sha, final, prereg) == 2                      # never overwritten
    with pytest.raises(SystemExit):
        PC.main(["--staged", str(staged), "--comparison", str(rep), "--comparison-sha256", sha,
                 "--output", str(tmp_path / "noprereg.h5")])                    # the pre-registration is required
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: {"commit": "c" * 40, "dirty": True})
    assert _promote(staged, rep, sha, tmp_path / "dirty.h5", prereg) == 2


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
    if "hist" in kw:                                             # an edited source: make it the one of record
        W.pin_of_record(monkeypatch, CV, kw["hist"])
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
    assert s["n_groups"] == 0 and s["n_groups_in_source"] == 3 and not (tmp_path / "a" / "val" / "groups").exists()
    assert (tmp_path / "a" / "val" / s["row_path"][0]["diff_file"]).read_text() == "- a\n+ b\n"
    assert carry(world, tmp_path / "b", extra=["--accept-row-path-diff", "abc1234=" + "e" * 64]) == 2
    assert carry(world, tmp_path / "c", extra=["--accept-row-path-diff", "abc1234=" + "d" * 64]) == 0
    rec = T.read_cache_staged(tmp_path / "c" / "carried.h5")[2]
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


def test_a_release_cache_is_never_a_carry_source(world, tmp_path, same_code, monkeypatch):
    assert carry(world, tmp_path / "a") == 0
    W.pin_of_record(monkeypatch, CV, tmp_path / "a" / "carried.h5")
    assert carry(world, tmp_path / "b", hist=tmp_path / "a" / "carried.h5") == 2
    assert not summary(tmp_path / "b")["global_checks"]["source_is_version_3.3"]


# ------------------------------------------------------------------------------- the source and commits of record
def test_a_source_that_is_not_of_record_is_refused(world, tmp_path, same_code, monkeypatch):
    monkeypatch.setattr(CV, "SOURCE_OF_RECORD_SHA256", "d9c3783892872f8739f6c1a4039ee84cae5ab84803235a61d7d06ac119336cf2")
    assert carry(world, tmp_path) == 2 and not (tmp_path / "carried.h5").exists()
    s = summary(tmp_path)
    assert s["global_checks"]["source_of_record"] is False and s["n_groups"] == 0


def test_the_pinned_source_and_commits_are_the_reviewed_ones():
    import ast
    tree = ast.parse((REPO_ROOT / "scripts" / "carry_validated_cache.py").read_text())
    pins = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
            if isinstance(t, ast.Name) and t.id in ("SOURCE_OF_RECORD_SHA256", "COMMITS_OF_RECORD")}
    assert pins["SOURCE_OF_RECORD_SHA256"] == "d9c3783892872f8739f6c1a4039ee84cae5ab84803235a61d7d06ac119336cf2"
    assert sorted(c[:7] for c in pins["COMMITS_OF_RECORD"]) == ["03f03b8", "3557ae2", "964e935", "bce73bf",
                                                                 "ddd22ad"]
    assert all(len(c) == 40 for c in pins["COMMITS_OF_RECORD"])


def test_head_as_a_commit_of_record_is_refused(world, tmp_path, monkeypatch):
    """A commit equal to HEAD would make the code-lineage check vacuous: the real diff is empty, yet the commit is not
    one of record, so the carry is refused."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: {"commit": head, "dirty": False})
    assert carry(world, tmp_path, commits=("HEAD",)) == 2 and not (tmp_path / "carried.h5").exists()
    s = summary(tmp_path)
    assert s["row_path"][0]["identical"] and s["row_path"][0]["old_commit"] == head
    assert s["global_checks"]["row_path_commits_of_record"] is False and s["n_groups"] == 0


@pytest.mark.parametrize("commits", [(), ("abc1234", "def5678"), ("abc1234", "abc1234"), ("def5678",)])
def test_a_commit_set_other_than_the_record_is_refused(world, tmp_path, same_code, monkeypatch, commits):
    W.pin_of_record(monkeypatch, CV, world["hist"], commits=("abc1234", "fff0000") if not commits else ("abc1234",))
    assert carry(world, tmp_path, commits=commits or ("abc1234",)) == 2
    assert summary(tmp_path)["global_checks"]["row_path_commits_of_record"] is False


def test_a_repair_record_naming_another_builder_is_refused(world, tmp_path, same_code, monkeypatch):
    for i, bc in enumerate(("fff0000", "", "fff0000 (abc1234)", "abc12")):    # another, none, in the note, too short
        p = _v33_attr(world, tmp_path / str(i), "repair_record", json.dumps({"added": [[W.SIMS[0], 5, 2.6]],
                                                                             "builder_commit": bc}))
        W.pin_of_record(monkeypatch, CV, p)
        assert carry(world, tmp_path / str(i), hist=p) == 2, bc
        assert summary(tmp_path / str(i))["global_checks"]["repair_builder_commit_of_record"] is False


# ------------------------------------------------------------------------------------------------ raw re-hash evidence
def _acceptance(world, tmp_path, *, rehashed=True, all_accepted=True, raw_sha=None):
    acc = tmp_path / "acceptance"
    files = {}
    for sim, snap, _z in W.GROUPS:
        cat = L.read_npz(world["rebuilt"] / sim / f"snap_{snap:03d}" / "catalog.npz")[1]
        raw = L.find_input(cat, "raw_spectra")
        rec = {"sim": sim, "snap": snap, "accepted": True,
               "checks": {"release_loader": {"ok": True, "rehashed": rehashed,
                                             "raw": {"sha256": raw_sha or raw["sha256"], "size": raw["size"]}}}}
        name = f"groups/{sim}__snap_{snap:03d}.json"
        (acc / "groups").mkdir(parents=True, exist_ok=True)
        (acc / name).write_text(json.dumps(rec))
        files[name] = L.sha256_file(acc / name)
    (acc / "summary.json").write_text(json.dumps({"format": "hcd_priya.catalogue_acceptance/1",
                                                  "all_accepted": all_accepted, "group_files_sha256": files}))
    return acc / "summary.json"


def test_the_raw_files_are_rehashed_here_or_by_the_pinned_acceptance(world, tmp_path, same_code, capsys):
    assert carry(world, tmp_path / "none", rehash=False) == 2
    assert "re-hashed" in capsys.readouterr().out and not (tmp_path / "none" / "carried.h5").exists()
    acc = _acceptance(world, tmp_path / "a")
    ok = ["--acceptance-summary", acc, "--acceptance-summary-sha256", L.sha256_file(acc)]
    assert carry(world, tmp_path / "ok", rehash=False, extra=ok) == 0
    g = json.loads(next((tmp_path / "ok" / "val" / "groups").iterdir()).read_text())
    assert g["checks"]["raw_matches_rebuilt_record"]["evidence"].startswith("re-hashed by the catalogue acceptance")
    assert summary(tmp_path / "ok")["catalogue_acceptance"]["sha256"] == L.sha256_file(acc)
    assert carry(world, tmp_path / "pin", rehash=False, extra=ok[:3] + ["0" * 64]) == 2
    assert carry(world, tmp_path / "half", rehash=False, extra=ok[:2]) == 2
    for name, kw, check in (("notall", dict(all_accepted=False), None),
                            ("norehash", dict(rehashed=False), "raw_matches_rebuilt_record"),
                            ("otherraw", dict(raw_sha="1" * 64), "raw_matches_rebuilt_record")):
        a = _acceptance(world, tmp_path / name, **kw)
        out = tmp_path / f"run_{name}"
        assert carry(world, out, rehash=False,
                     extra=["--acceptance-summary", a, "--acceptance-summary-sha256", L.sha256_file(a)]) == 2
        if check is None:
            assert summary(out)["global_checks"]["catalogue_acceptance_passed"] is False
        else:
            assert check in failed_checks(out)


# --------------------------------------------------------------------------------- the global checks and inputs
def _v33_attr(world, tmp_path, name, value):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"hist_attr_{name}.h5"
    shutil.copy2(world["hist"], p)
    with h5py.File(p, "r+") as f:
        if value is None:
            del f.attrs[name]
        else:
            f.attrs[name] = value
    return p


@pytest.mark.parametrize("edit, check", [
    (("dataset", "alpha_idx", lambda a: a.__setitem__(slice(0, 2), a[[1, 0]])), "canonical_layout"),
    (("attr", "n_k", 7), "n_k_equal_builder"),
    (("attr", "tau_thresh", 2.0e6), "recipe_attributes_equal_builder"),
    (("dataset", "alpha_slope", lambda a: a.__setitem__(0, a[0] * 1.001)), "mean_flux_ladder_equal_builder"),
    (("dataset", "tier_c_nhi_edges", lambda a: a.__setitem__(0, a[0] + 0.1)), "static_tables_equal_builder"),
])
def test_each_global_check_refuses(world, tmp_path, same_code, monkeypatch, edit, check):
    kind, key, change = edit
    p = _v33_with(world, tmp_path, key, change) if kind == "dataset" else _v33_attr(world, tmp_path, key, change)
    W.pin_of_record(monkeypatch, CV, p)
    assert carry(world, tmp_path, hist=p) == 2 and not (tmp_path / "carried.h5").exists()
    s = summary(tmp_path)
    assert s["global_checks"][check] is False and s["n_groups"] == 0


def test_a_changed_absorption_path_is_refused(world, tmp_path, same_code, monkeypatch):
    p = _v33_with(world, tmp_path, "snap_total_path_dX", lambda a: a.__setitem__(1, a[1] * 1.01))
    W.pin_of_record(monkeypatch, CV, p)
    assert carry(world, tmp_path, hist=p) == 2
    assert "cddf_bins_and_absorption_path_equal_stored" in failed_checks(tmp_path)


@pytest.mark.parametrize("missing", ["raw", "old_catalogue", "rebuilt_group"])
def test_a_missing_input_fails_its_group(world, tmp_path, same_code, missing):
    sim, snap, _z = W.GROUPS[2]
    kw = {}
    if missing == "raw":
        w = _copy(world, tmp_path, "emu")
        shutil.rmtree(w["emu"] / sim / "output" / f"SPECTRA_{snap:03d}")
        kw["emu"] = w["emu"]
    elif missing == "old_catalogue":
        w = _copy(world, tmp_path, "old")
        (w["old"] / sim / f"snap_{snap:03d}" / "catalog.npz").unlink()
        kw["old"] = w["old"]
    else:
        w = _copy(world, tmp_path, "rebuilt")
        shutil.rmtree(w["rebuilt"] / sim / f"snap_{snap:03d}")
        kw["rebuilt"] = w["rebuilt"]
    assert carry(world, tmp_path, **kw) == 2 and not (tmp_path / "carried.h5").exists()
    assert "inputs" in failed_checks(tmp_path)


def test_an_unexpected_error_exits_2_with_a_summary_and_no_cache(world, tmp_path, same_code, monkeypatch, capsys):
    p = _v33_attr(world, tmp_path, "created_utc", None)
    W.pin_of_record(monkeypatch, CV, p)
    assert carry(world, tmp_path, hist=p) == 2
    assert "unexpected KeyError" in capsys.readouterr().out
    s = summary(tmp_path)
    assert s["passed"] is False and s["error"].startswith("KeyError") and not (tmp_path / "carried.h5").exists()


def test_an_ineligible_output_never_appears_under_the_output_name(world, tmp_path, same_code, monkeypatch):
    calls = []

    def code(repo_dir=None):                     # clean when the run starts, dirty when the cache is written
        calls.append(1)
        return {"commit": "c" * 40, "dirty": len(calls) > 1}
    monkeypatch.setattr(L, "code_identity", code)
    real = T.write_cache

    def spy(path, *a, **k):
        assert path.name.startswith(".") and path.name.endswith(".unchecked")
        return real(path, *a, **k)
    monkeypatch.setattr(T, "write_cache", spy)
    assert carry(world, tmp_path) == 2
    assert not (tmp_path / "carried.h5").exists() and not list(tmp_path.glob(".*unchecked"))
    s = summary(tmp_path)                                    # written as passed before the cache: now says failed
    assert s["passed"] is False and "would not be release-eligible" in s["error"] and s["n_groups"] == 3


def test_a_failure_after_the_summary_leaves_it_saying_failed(world, tmp_path, same_code, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(T, "write_cache", boom)
    assert carry(world, tmp_path) == 2 and "unexpected OSError" in capsys.readouterr().out
    s = summary(tmp_path)
    assert s["passed"] is False and s["error"] == "OSError: disk full" and s["n_groups"] == 3   # the record is kept
    assert not (tmp_path / "carried.h5").exists() and not (tmp_path / "val" / "failure.json").exists()


def test_an_output_published_by_another_run_after_the_check_is_never_replaced(world, tmp_path, same_code, monkeypatch,
                                                                             capsys):
    """R1-4: the output exists check runs first, the publish last; a cache another run publishes at the output name in
    between stays as it is, this run is refused and its summary says failed."""
    real = T.write_cache

    def plant(path, *a, **k):
        rec = real(path, *a, **k)
        (tmp_path / "carried.h5").write_bytes(b"another run's cache")
        return rec
    monkeypatch.setattr(T, "write_cache", plant)
    assert carry(world, tmp_path) == 2
    assert "never overwritten" in capsys.readouterr().out
    assert (tmp_path / "carried.h5").read_bytes() == b"another run's cache"
    assert not list(tmp_path.glob(".*unchecked"))
    s = summary(tmp_path)
    assert s["passed"] is False and "never overwritten" in s["error"]


@pytest.mark.parametrize("bad", ["abc1234=xyz", "=" + "d" * 64, "abc1234=" + "D" * 64, "abc1234"])
def test_a_malformed_accepted_digest_is_refused(world, tmp_path, same_code, bad, capsys):
    assert carry(world, tmp_path, extra=["--accept-row-path-diff", bad]) == 2
    assert "expected COMMIT=SHA256" in capsys.readouterr().out


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
    (repo / "pkg" / "c.py").write_text("x = 1\n")                             # another commit, the same code
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "four: outside")
    assert CV.row_path_diff(repo, first, "HEAD", spec)[0]["diff_sha256"] == res["diff_sha256"]
    res, _ = CV.row_path_diff(repo, first, "HEAD", (("pkg/b.py", ("gone",)),))
    assert res["items"] == [{"item": "pkg/b.py:gone", "status": "absent at both"}] and not res["identical"]


def test_the_row_path_spec_names_code_that_exists_at_head():
    for path, symbols in CV.ROW_PATH:
        src = (REPO_ROOT / path).read_text()
        for sym in symbols or ():
            assert CV._symbol_source(src, sym) is not None, (path, sym)
