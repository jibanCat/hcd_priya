"""Cache format 3.4 and its lineage: the builder reads catalogue-only snapshots only through the snapshot verifier and
records every input; load_cache is strict and load_cache_historical reads 3.3 as before; the fresh-row sample's
pre-registration rule and comparison. Synthetic world (tests/tau0_world.py) and synthetic caches only."""
import json
import os
import shutil
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis import catalog as C
from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as T
from hcd_analysis.paths import REPO_ROOT
from tests import tau0_fixtures as tf
from tests import tau0_world as W

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_emulator_cache_tau0 as bt0  # noqa: E402
import fresh_row_sample as FR  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        yield W.make_world(tmp_path_factory.mktemp("world"), mp)


def build(w, out, root=None, *extra):
    W.build(["--hcd-root", root or w["rebuilt"], "--emu-root", w["emu"], "--output", out, *extra])
    return out


# ---------------------------------------------------------------------------------------------------- the builder
def test_a_cache_built_from_catalogue_only_snapshots_is_eligible_and_names_every_input(world, tmp_path):
    out = build(world, tmp_path / "c.h5")
    d, attrs, rec = T.read_cache(out)
    assert rec["release_eligible"] and rec["finder_ids"] == [C.FINDER_ID] and attrs["cache_version"] == "3.4"
    assert rec["settings"] == T.cache_settings(fidelity="lf", n_k=172, alpha_refine=2,
                                               alpha_range=(float(d["alpha_slope"].min()),
                                                            float(d["alpha_slope"].max())))
    assert [(g["sim"], g["snap"], g["rows"]) for g in rec["product"]["groups"]] == \
        [(s, n, T.ROWS_BUILT) for s, n, _z in sorted(W.GROUPS)]
    by_role = {i["role"]: i for i in rec["inputs"]}
    for sim, snap, _z in W.GROUPS:
        sd = world["rebuilt"] / sim / f"snap_{snap:03d}"
        cat = by_role[T.group_role("catalog", sim, snap)]
        assert cat["sha256"] == L.sha256_file(sd / "catalog.npz") and cat["release_eligible"]
        assert by_role[T.group_role("snapshot", sim, snap)]["sha256"] == L.sha256_file(sd / "meta.json")
        assert by_role[T.group_role("cddf", sim, snap)]["sha256"] == L.sha256_file(sd / "cddf.npz")
        raw = by_role[T.group_role("raw_spectra", sim, snap)]
        assert {k: raw[k] for k in ("sha256", "size", "mtime_ns")} == L.file_identity(world["raws"][(sim, snap)])
    assert {f"sim_ics:{s}" for s in W.SIMS} <= set(by_role)
    assert len(rec["inputs"]) == 4 * len(W.GROUPS) + len(W.SIMS)
    with h5py.File(out, "r") as f:                                        # the record: a dataset, not an attribute
        assert f[T.LINEAGE_KEY].shape == () and T.LINEAGE_KEY not in f.attrs
    from hcd_analysis.emulator.data import load_cache, load_cache_historical
    strict = load_cache(out)
    assert strict["lineage"] == rec and strict["cache_sha256"] == L.sha256_file(out)
    hist = load_cache_historical(out, reason="test")
    for k, v in hist.items():                                             # the same arrays either way
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, strict[k], equal_nan=v.dtype.kind == "f"), k
    assert set(strict) - set(hist) == {"lineage", "cache_sha256"}


def test_old_snapshots_are_refused_and_a_historical_reproduction_is_stamped_ineligible(world, tmp_path):
    with pytest.raises(C.CatalogIntegrityError, match="no lineage record|completed snapshot refused"):
        build(world, tmp_path / "c.h5", world["old"])
    assert not (tmp_path / "c.h5").exists() and not list(tmp_path.glob(".c.h5*"))
    out = build(world, tmp_path / "h.h5", world["old"], "--historical-reproduction", "a reproduction")
    rec = T.read_cache_historical(out, reason="test")[2]
    assert not rec["release_eligible"] and any("historical: a reproduction" in w for w in rec["ineligible_reasons"])
    from hcd_analysis.emulator.data import load_cache
    with pytest.raises(T.CacheLineageError, match="not release-eligible"):
        load_cache(out)


def test_the_builder_needs_its_roots_and_never_overwrites(world, tmp_path):
    with pytest.raises(SystemExit):
        W.build(["--emu-root", world["emu"], "--output", tmp_path / "x.h5"])        # no snapshot root
    with pytest.raises(SystemExit):
        W.build(["--hcd-root", world["rebuilt"], "--output", tmp_path / "x.h5"])    # no raw root
    (tmp_path / "x.h5").write_text("")
    with pytest.raises(SystemExit, match="never overwritten"):
        build(world, tmp_path / "x.h5")


def test_other_finder_settings_a_dirty_tree_or_a_dry_run_give_no_release_cache(world, tmp_path, monkeypatch):
    other = tmp_path / "other"
    sim, snap, z = W.GROUPS[0]
    W.rebuild_snapshot(other, world["emu"], sim, snap, z, tau_threshold=50.0)
    with pytest.raises(C.CatalogProvenanceError, match="other finder settings: tau_threshold"):
        build(world, tmp_path / "o.h5", other)
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: {"commit": "c" * 40, "dirty": True})
    rec = T.read_cache_historical(build(world, tmp_path / "d.h5"), reason="test")[2]
    assert not rec["release_eligible"] and "the code tree was dirty" in rec["ineligible_reasons"]
    with pytest.raises(T.CacheLineageError, match="not release-eligible"):
        T.read_cache(tmp_path / "d.h5")
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(W.CLEAN))
    rec = T.read_cache_historical(build(world, tmp_path / "n.h5", None, "--n-skewers", "4"), reason="test")[2]
    assert not rec["release_eligible"] and any("dry run" in w for w in rec["ineligible_reasons"])


def test_discovery_needs_a_completion_marker_but_no_phase1_p1d_products(world, tmp_path):
    root = W.copy_tree(world["rebuilt"], tmp_path / "r")
    found = bt0.discover_snapshot_dirs(root)
    assert [(s, n) for s, n, _d in found] == sorted((s, n) for s, n, _z in W.GROUPS)
    sim, snap, _z = W.GROUPS[1]
    (root / sim / f"snap_{snap:03d}" / "done_catalogue").unlink()
    assert len(bt0.discover_snapshot_dirs(root)) == len(W.GROUPS) - 1


# ----------------------------------------------------------------------------------------------------- the loader
def test_load_cache_refuses_3_3_and_load_cache_historical_reads_it_as_before(tmp_path, monkeypatch):
    from hcd_analysis.emulator.data import load_cache, load_cache_historical
    tf.clean_code(monkeypatch)
    groups = [("simA", 16, 3.2, 3, 1), ("simB", 5, 4.0, 3, 4)]
    old, new = tf.write_v33(tmp_path / "old.h5", groups), tf.write_v34(tmp_path / "new.h5", groups)
    with pytest.raises(T.CacheLineageError, match=r"cache version '3.3'.*load_cache_historical"):
        load_cache(tmp_path / "old.h5", validate=False)
    for bad in ("", None):
        with pytest.raises(ValueError, match="non-empty reason"):
            load_cache_historical(old, reason=bad, validate=False)
    h = load_cache_historical(old, reason="test", validate=False)
    s = load_cache(tmp_path / "new.h5", validate=False)
    for k, v in h.items():
        assert np.array_equal(v, s[k], equal_nan=np.asarray(v).dtype.kind == "f"), k
    with h5py.File(tmp_path / "new.h5", "r+") as f:
        f["scale"][0] = f["scale"][0] * 2
    with pytest.raises(T.CacheLineageError, match="payload digest"):
        load_cache(tmp_path / "new.h5", validate=False)
    with h5py.File(old, "r+") as f:
        f.attrs["cache_version"] = "3.2"
    with pytest.raises(T.CacheLineageError, match="3.2"):
        load_cache_historical(old, reason="test", validate=False)


def test_a_carried_group_needs_the_carry_block_and_a_passed_validation(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    groups = [("simA", 16, 3.2, 3, 1)]
    tf.write_v34(tmp_path / "a.h5", groups, rows=T.ROWS_CARRIED)
    with pytest.raises(T.CacheLineageError, match="carried rows without the carry's source"):
        T.read_cache(tmp_path / "a.h5")
    kw = tf.carry_kw()
    kw["validation"] = dict(kw["validation"], all_passed=False)
    tf.write_v34(tmp_path / "b.h5", groups, **kw)
    with pytest.raises(T.CacheLineageError, match="without a passed validation"):
        T.read_cache(tmp_path / "b.h5")
    tf.write_v34(tmp_path / "c.h5", groups, **tf.carry_kw(promoted=False))
    with pytest.raises(T.CacheLineageError, match="not yet promoted"):          # staged: the release reader refuses
        T.read_cache(tmp_path / "c.h5")
    T.read_cache_staged(tmp_path / "c.h5")
    tf.write_v34(tmp_path / "e.h5", groups, **tf.carry_kw())
    T.read_cache(tmp_path / "e.h5")


def _probe(tmp_path, name, groups, **kw):
    rec = tf.write_v34(tmp_path / name, groups, **kw)
    assert rec["release_eligible"]                     # the writer computes eligibility; the reader must still refuse
    return tmp_path / name


def test_the_validated_input_escape_is_bound_to_the_carry_source(tmp_path, monkeypatch):
    """Inputs named as validated carry no lineage of their own: allowed only for carried rows, and only as the one
    historical cache the carry block names."""
    tf.clean_code(monkeypatch)
    groups = [("simA", 16, 3.2, 3, 1)]
    src = tf.source_entry("5" * 64, 7)
    # every group built, an unrecorded cache named as validated
    p = _probe(tmp_path, "p2.h5", groups, extra_inputs=[src],
               validation={"all_passed": True, "validated_inputs": [tf.SOURCE_ROLE]})
    with pytest.raises(T.CacheLineageError, match="validated inputs without carried rows"):
        T.read_cache(p)
    # carried rows whose carry block names another source than the validated input
    kw = tf.carry_kw(src)
    kw["carried"] = dict(kw["carried"], source_sha256="6" * 64)
    with pytest.raises(T.CacheLineageError, match="exactly the carry's source"):
        T.read_cache(_probe(tmp_path, "p2b.h5", groups, **kw))
    kw = tf.carry_kw(src)
    kw["carried"] = dict(kw["carried"], source_size=8)
    with pytest.raises(T.CacheLineageError, match="exactly the carry's source"):
        T.read_cache(_probe(tmp_path, "p2c.h5", groups, **kw))
    other = tf.source_entry("5" * 64, 7, kind="catalog")
    with pytest.raises(T.CacheLineageError, match="exactly the carry's source"):
        T.read_cache(_probe(tmp_path, "p2d.h5", groups, **tf.carry_kw(other)))
    kw = tf.carry_kw(src)
    kw["extra_inputs"] = [src, dict(src, role="second")]
    kw["validation"] = dict(kw["validation"], validated_inputs=[tf.SOURCE_ROLE, "second"])
    with pytest.raises(T.CacheLineageError, match="exactly the carry's source"):
        T.read_cache(_probe(tmp_path, "p2e.h5", groups, **kw))
    T.read_cache(_probe(tmp_path, "ok.h5", groups, **tf.carry_kw(src)))


def _with_entries(tmp_path, name, groups, edit):
    """A cache whose group inputs are edited by ``edit(entries)`` before it is written."""
    d = tf.datasets(groups)
    entries = [e for (sim, snap, *_r) in groups for e in tf.group_entries(sim, snap)]
    edit(entries)
    gl = [{"sim": sim, "snap": int(snap), "z_grid": float(zg), "rows": T.ROWS_BUILT} for (sim, snap, zg, *_r) in groups]
    a = dict(tf.ATTRS, n_rows=len(d["sim_name"]), n_snaps=len(d["snap_snap"]))
    rec = T.write_cache(tmp_path / name, d, a, inputs=entries, settings={"cache_version": T.CACHE_VERSION},
                        groups=gl)
    return tmp_path / name, rec


@pytest.mark.parametrize("edit, match", [
    # a catalogue role filled by an unrecorded raw file (eligible by the lineage rule, refused by the reader)
    (lambda es: es.__setitem__(1, {"role": es[1]["role"], "kind": "raw_spectra", "sha256": "7" * 64, "size": 1}),
     "not one input of kind catalog"),
    (lambda es: es[2].__setitem__("kind", "catalog"), "not one input of kind cddf"),
    (lambda es: es[3].__setitem__("kind", "sim_ics"), "not one input of kind raw_spectra"),
    (lambda es: es.__setitem__(0, {k: v for k, v in es[0].items() if k not in ("finder_ids", "release_eligible")}
                               | {"kind": "snapshot"}), "not a release-eligible product"),
    (lambda es: es.pop(2), r"no \['cddf'\] input"),
])
def test_each_group_input_has_its_kind_and_lineage(tmp_path, monkeypatch, edit, match):
    tf.clean_code(monkeypatch)
    from hcd_analysis import lineage as LL
    monkeypatch.setattr(LL, "RAW_INPUT_KINDS", LL.RAW_INPUT_KINDS | {"snapshot"})   # let the writer call it eligible
    p, _rec = _with_entries(tmp_path, "g.h5", [("simA", 16, 3.2, 3, 1)], edit)
    with pytest.raises(T.CacheLineageError, match=match):
        T.read_cache(p)


@pytest.mark.parametrize("edit, match", [
    (lambda g: g.reverse(), "not the file's snapshot blocks"),
    (lambda g: g[0].__setitem__("rows", "copied"), "unknown row origin"),
    (lambda g: g.pop(), "not the file's snapshot blocks"),
])
def test_the_record_groups_must_be_the_file_blocks(tmp_path, monkeypatch, edit, match):
    tf.clean_code(monkeypatch)
    groups = [("simA", 16, 3.2, 3, 1), ("simB", 5, 4.0, 3, 4)]
    d = tf.datasets(groups)
    entries = [e for (sim, snap, *_r) in groups for e in tf.group_entries(sim, snap)]
    gl = [{"sim": sim, "snap": int(snap), "z_grid": float(zg), "rows": T.ROWS_BUILT} for (sim, snap, zg, *_r) in groups]
    edit(gl)
    a = dict(tf.ATTRS, n_rows=len(d["sim_name"]), n_snaps=len(d["snap_snap"]))
    T.write_cache(tmp_path / "g.h5", d, a, inputs=entries, settings={}, groups=gl)
    with pytest.raises(T.CacheLineageError, match=match):
        T.read_cache(tmp_path / "g.h5")


# ------------------------------------------------------------------------------------------- the fresh-row sample
GRID = [(f"s{i}", n, z) for i in range(6) for n, z in ((4, 5.4), (10, 4.2), (16, 3.0), (22, 2.0))]


def test_the_preregistration_rule_is_deterministic_and_covers_its_strata():
    added = [("s3", 16), ("s3", 10)]
    a = FR.select_groups(GRID, added, seed=11, n_total=8, include_sim_prefix="s5")
    b = FR.select_groups(list(reversed(GRID)), added, seed=11, n_total=8, include_sim_prefix="s5")
    assert a == b and len(a) == 8 and len({(g[0], g[1]) for g in a}) == 8
    strata = [g[4] for g in a]
    assert strata[:2] == ["spliced origin"] * 2 and [g[:2] for g in a[:2]] == [("s3", 10), ("s3", 16)]
    assert a[2][2] == 2.0 and a[2][4] == "lowest z_grid" and a[3][2] == 5.4 and a[3][4] == "highest z_grid"
    assert any(g[0].startswith("s5") for g in a)
    rest = [g for g in a if g[4] == "remainder"]
    assert len({g[0] for g in rest}) == len(rest) and all(g[3] == "base" for g in a[2:])
    assert FR.select_groups(GRID, added, seed=12, n_total=8, include_sim_prefix="s5") != a


def test_the_select_command_writes_the_rule_before_any_rebuild_and_never_rewrites(world, tmp_path):
    out = tmp_path / "prereg.json"
    assert FR.main(["select", "--cache", str(world["hist"]), "--seed", "3", "--n-total", "3",
                    "--include-sim-prefix", "ns0.97", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["format"] == FR.FORMAT and doc["rule"] == FR.RULE and doc["criterion"]["rtol"] == FR.RTOL
    assert doc["historical_cache"]["sha256"] == L.sha256_file(world["hist"])
    assert doc["generator"]["script_sha256"] == L.sha256_file(REPO_ROOT / "scripts" / "fresh_row_sample.py")
    assert doc["groups"][0]["stratum"] == "spliced origin" and doc["groups"][0]["snap"] == 5
    with pytest.raises(SystemExit, match="never rewritten"):
        FR.main(["select", "--cache", str(world["hist"]), "--seed", "3", "--n-total", "3", "--out", str(out)])


@pytest.fixture(scope="module")
def carried(world, tmp_path_factory):
    import carry_validated_cache as CV
    base = tmp_path_factory.mktemp("carried")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CV, "row_path_diff", lambda repo, a, b, spec=None: ({"old_commit": a, "new_commit": b,
                                                                        "identical": True, "items": [],
                                                                        "diff_sha256": "0" * 64}, ""))
        W.pin_of_record(mp, CV, world["hist"])
        assert CV.main([str(x) for x in ("--historical-cache", world["hist"], "--old-root", world["old"],
                                         "--rebuilt-root", world["rebuilt"], "--raw-root", world["emu"],
                                         "--row-path-commit", W.BUILDER_COMMIT, "--rehash-raw",
                                         "--validation-dir", base / "val", "--output", base / "carried.h5")]) == 0
    pre = base / "prereg.json"
    FR.main(["select", "--cache", str(world["hist"]), "--seed", "5", "--n-total", "2", "--out", str(pre)])
    return base / "carried.h5", pre


def _fresh(world, pre, tmp_path):
    out = []
    for i, g in enumerate(json.loads(pre.read_text())["groups"]):
        out += ["--fresh", str(build(world, tmp_path / f"fresh{i}.h5", None, "--select", f"{g['sim']}:{g['snap']}"))]
    return out


def _compare(pre, car, fresh, out):
    return FR.main(["compare", "--prereg", str(pre), "--prereg-sha256", L.sha256_file(pre), "--carried", str(car),
                    *fresh, "--out", str(out)])


def test_fresh_rows_equal_the_carried_rows_bitwise(world, carried, tmp_path):
    car, pre = carried
    assert _compare(pre, car, _fresh(world, pre, tmp_path), tmp_path / "r.json") == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["passed"] and all(g["same_rebuilt_catalogue"] for g in rep["groups"])
    assert set(rep["fields"]) == (set(T.ROW_KEYS) - {"snap_group_idx"}) | set(T.SNAP_KEYS)
    assert all(f["bitwise_all"] and f["max_rel"] == 0.0 for f in rep["fields"].values())


def test_a_fresh_row_difference_or_a_wrong_sample_is_caught(world, carried, tmp_path, monkeypatch):
    car, pre = carried
    flux = sys.modules["fake_spectra.fluxstatistics"]
    orig = flux.mean_flux
    monkeypatch.setattr(flux, "mean_flux", lambda tau, t: orig(tau, t) * (1 + 1e-9))   # another mean-flux solve
    fresh = _fresh(world, pre, tmp_path)
    assert _compare(pre, car, fresh, tmp_path / "r.json") == 1
    rep = json.loads((tmp_path / "r.json").read_text())
    assert not rep["passed"] and rep["fields"]["scale"]["max_rel"] > FR.RTOL and not rep["fields"]["scale"][
        "bitwise_all"] and rep["fields"]["tier_c_counts"]["bitwise_all"]
    with pytest.raises(SystemExit, match="sha256 differs"):
        FR.main(["compare", "--prereg", str(pre), "--prereg-sha256", "0" * 64, "--carried", str(car), *fresh,
                 "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="not the pre-registered"):
        _compare(pre, car, fresh[:2], tmp_path / "x.json")


def test_the_comparison_criterion():
    a = np.array([[1.0, 2.0, np.nan], [0.0, 0.0, 0.0]])
    assert FR.compare_field(a, a.copy()) == (True, True, 0.0)
    b = a.copy()
    b[0, 1] *= 1 + 5e-13
    ok, bit, m = FR.compare_field(b, a)
    assert ok and not bit and 0 < m <= FR.RTOL
    b[0, 1] = 2.0 * (1 + 5e-12)
    assert not FR.compare_field(b, a)[0]
    c = a.copy()
    c[0, 2] = 1.0                                                  # a NaN moved
    assert not FR.compare_field(c, a)[0]
    d = a.copy()
    d[1, 0] = 1e-300                                               # an all-zero reference row: any change counts
    assert not FR.compare_field(d, a)[0]
    assert FR.compare_field(np.array([1, 2]), np.array([1, 3]))[0] is False
    assert FR.compare_field(np.array(["a"], dtype=object), np.array(["a"], dtype=object))[0]


def test_the_cache_file_is_fsynced_before_it_is_renamed_into_place(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    events, real_fsync, real_replace = [], T.os.fsync, T.os.replace

    def fsync(fd):
        events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}") if os.path.exists("/proc/self/fd") else fd))
        return real_fsync(fd)

    def replace(a, b):
        events.append(("replace", str(a)))
        return real_replace(a, b)
    monkeypatch.setattr(T.os, "fsync", fsync)
    monkeypatch.setattr(T.os, "replace", replace)
    tf.write_v34(tmp_path / "c.h5", [("simA", 16, 3.2, 3, 1)])
    kinds = [e[0] for e in events]
    i = kinds.index("replace")
    assert "fsync" in kinds[:i]
    if os.path.exists("/proc/self/fd"):
        assert any(e[0] == "fsync" and str(e[1]) == events[i][1] for e in events[:i])   # the temporary file itself
