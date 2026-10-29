"""The run-settings record (hcd_analysis/run_settings.py) and its use by the pipeline: an output directory holds the
results of one set of settings, and a rerun with different settings is refused before anything is read or written.
Synthetic optical depths and settings only."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import run_settings as RS

SETTINGS = {"absorber": {"tau_threshold": 100.0, "b_bounds": (1.0, 300.0)}, "debug": False}


def test_the_first_run_records_the_settings_and_the_info(tmp_path):
    p = RS.record_or_verify(tmp_path, SETTINGS, info=lambda: {"code_commit": "abc1234"})
    assert p == tmp_path / RS.RUN_SETTINGS_FILE
    doc = json.loads(p.read_text())
    assert doc["format"] == RS.FORMAT
    assert doc["settings"] == {"absorber": {"tau_threshold": 100.0, "b_bounds": [1.0, 300.0]}, "debug": False}
    assert doc["info"]["code_commit"] == "abc1234" and "created_utc" in doc["info"]
    assert sorted(x.name for x in tmp_path.iterdir()) == [RS.RUN_SETTINGS_FILE]       # no temporary file left


def test_a_rerun_with_the_same_settings_passes_and_rewrites_nothing(tmp_path):
    p = RS.record_or_verify(tmp_path, SETTINGS, info=lambda: {"code_commit": "abc1234"})
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    calls = []
    RS.record_or_verify(tmp_path, {"debug": False, "absorber": {"b_bounds": [1.0, 300.0], "tau_threshold": 100.0}},
                        info=lambda: calls.append(1) or {"code_commit": "other"})
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before
    assert calls == []                                    # info is only gathered when a record is written


def test_a_rerun_with_different_settings_is_refused_and_names_every_difference(tmp_path):
    p = RS.record_or_verify(tmp_path, SETTINGS)
    before = p.read_bytes()
    changed = {"absorber": {"tau_threshold": 200.0, "b_bounds": (1.0, 300.0)}, "debug": True}
    with pytest.raises(RS.RunSettingsMismatch) as e:
        RS.record_or_verify(tmp_path, changed)
    msg = str(e.value)
    assert "absorber.tau_threshold: recorded 100.0, this run 200.0" in msg and "debug: recorded False" in msg
    assert p.read_bytes() == before


@pytest.mark.parametrize("extra", [{"new_key": 1}, {}])
def test_an_added_or_missing_setting_is_a_difference(tmp_path, extra):
    RS.record_or_verify(tmp_path, {"a": 1, **({} if extra else {"gone": 2})})
    with pytest.raises(RS.RunSettingsMismatch, match="new_key" if extra else "gone"):
        RS.record_or_verify(tmp_path, {"a": 1, **extra})


@pytest.mark.parametrize("content, why", [
    (b"not json", "not a readable"), (b"[1, 2]", "not a hcd_priya.run_settings/1 record"),
    (b'{"format": "other/1", "settings": {"a": 1}}', "not a hcd_priya.run_settings/1 record"),   # equal settings
    (b'{"format": "hcd_priya.run_settings/1", "settings": [["a", 1]]}', "not a hcd_priya.run_settings/1 record"),
    (b'{"format": "hcd_priya.run_settings/1", "settings": {"a": 1}, "info": {"x": NaN}}', "not a readable"),
    (b'{"format": "hcd_priya.run_settings/1", "settings": {"a": 1}, "info": ' + b"[" * 100000 + b"]" * 100000 + b"}",
     "not a readable")])
def test_an_unreadable_record_is_refused_even_when_its_settings_are_equal(tmp_path, content, why):
    (tmp_path / RS.RUN_SETTINGS_FILE).write_bytes(content)
    with pytest.raises(RS.RunSettingsMismatch, match=why):
        RS.record_or_verify(tmp_path, {"a": 1})


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_a_record_that_is_not_a_regular_file_is_refused_without_blocking(tmp_path, kind):
    import threading
    p = tmp_path / RS.RUN_SETTINGS_FILE
    p.mkdir() if kind == "directory" else os.mkfifo(p)
    caught = []

    def call():
        try:
            RS.record_or_verify(tmp_path, {"a": 1})
        except RS.RunSettingsMismatch as e:
            caught.append(str(e))
    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "reading the record blocked"
    assert caught and ("not a regular file" in caught[0] or "cannot be read" in caught[0])


def test_the_record_is_world_readable_and_the_message_lists_at_most_twelve_keys(tmp_path):
    RS.record_or_verify(tmp_path, {f"k{i:02d}": i for i in range(13)})
    assert (tmp_path / RS.RUN_SETTINGS_FILE).stat().st_mode & 0o777 == 0o644
    with pytest.raises(RS.RunSettingsMismatch) as e:
        RS.record_or_verify(tmp_path, {f"k{i:02d}": -1 for i in range(12)})
    msg = str(e.value)
    assert "(and 1 more)" in msg and "k00: recorded 0, this run -1" in msg and "k12" not in msg
    with pytest.raises(RS.RunSettingsMismatch, match="k12: recorded 12, this run '<absent>'"):
        RS.record_or_verify(tmp_path, {f"k{i:02d}": i for i in range(12)})


def test_a_process_that_recorded_and_wrote_between_the_check_and_the_scan_is_not_mistaken_for_unknown_files(
        tmp_path, monkeypatch):
    """The re-check after the scan: another process records, then writes a product, after this one found no record."""
    real = RS._holds_files

    def other_process_first(out_dir, ignore=()):
        if not (tmp_path / RS.RUN_SETTINGS_FILE).exists():
            RS._publish(tmp_path, tmp_path / RS.RUN_SETTINGS_FILE, {"format": RS.FORMAT, "settings": {"a": 1},
                                                                    "info": {}})
            (tmp_path / "sim").mkdir()
            (tmp_path / "sim" / "done").write_text("")
        return real(out_dir, ignore)
    monkeypatch.setattr(RS, "_holds_files", other_process_first)
    RS.record_or_verify(tmp_path, {"a": 1})                                     # verified, not refused
    with pytest.raises(RS.RunSettingsMismatch, match="a: recorded 1, this run 2"):
        RS.record_or_verify(tmp_path, {"a": 2})


def test_the_scan_rules(tmp_path):
    """Only top-level entries can be ignored; a symlink counts as contents and is not followed; a loop ends."""
    nested = tmp_path / "n"
    (nested / "sim" / "hires").mkdir(parents=True)
    (nested / "sim" / "hires" / "x").write_text("")
    with pytest.raises(RS.RunSettingsMismatch, match="holds files"):
        RS.record_or_verify(nested, {"a": 1}, ignore=("hires",))
    (tmp_path / "empty").mkdir()
    link = tmp_path / "l"
    link.mkdir()
    (link / "to_empty").symlink_to(tmp_path / "empty")
    with pytest.raises(RS.RunSettingsMismatch, match="holds files"):
        RS.record_or_verify(link, {"a": 1})
    loop = tmp_path / "loop"
    (loop / "d").mkdir(parents=True)
    (loop / "d" / "back").symlink_to(loop)
    with pytest.raises(RS.RunSettingsMismatch, match="holds files"):
        RS.record_or_verify(loop, {"a": 1})


def test_a_stale_temporary_file_does_not_block_or_count_as_contents(tmp_path):
    (tmp_path / f".{RS.RUN_SETTINGS_FILE}.{os.getpid()}.tmp").write_text("left by a killed process")
    RS.record_or_verify(tmp_path, {"a": 1})
    assert json.loads((tmp_path / RS.RUN_SETTINGS_FILE).read_text())["settings"] == {"a": 1}


def test_a_directory_holding_files_but_no_record_is_refused_and_left_unchanged(tmp_path):
    (tmp_path / "sim" / "snap_001").mkdir(parents=True)
    RS.record_or_verify(tmp_path / "empty_tree_only", {"a": 1})                 # new directory: recorded
    (tmp_path / "sim" / "snap_001" / "done").write_text("")
    before = sorted(p.name for p in tmp_path.rglob("*"))
    with pytest.raises(RS.RunSettingsMismatch, match="holds files but no run_settings.json"):
        RS.record_or_verify(tmp_path, {"a": 1})
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_empty_directories_and_ignored_entries_do_not_count_as_contents(tmp_path):
    (tmp_path / "sim" / "snap_001").mkdir(parents=True)
    (tmp_path / "hires" / "x").mkdir(parents=True)
    (tmp_path / "hires" / "x" / "done").write_text("")
    RS.record_or_verify(tmp_path, {"a": 1}, ignore=("hires",))
    assert (tmp_path / RS.RUN_SETTINGS_FILE).exists()


def test_an_oversized_record_is_refused_without_reading_it_whole(tmp_path):
    (tmp_path / RS.RUN_SETTINGS_FILE).write_bytes(b" " * (RS.MAX_RECORD_BYTES + 1))
    with pytest.raises(RS.RunSettingsMismatch, match="larger than"):
        RS.record_or_verify(tmp_path, {"a": 1})


def test_a_symlinked_record_is_refused(tmp_path):
    real = tmp_path / "elsewhere.json"
    RS.record_or_verify(tmp_path / "x", {"a": 1})
    real.write_bytes((tmp_path / "x" / RS.RUN_SETTINGS_FILE).read_bytes())
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / RS.RUN_SETTINGS_FILE).symlink_to(real)
    with pytest.raises(RS.RunSettingsMismatch, match="cannot be read"):
        RS.record_or_verify(tmp_path / "y", {"a": 1})


def test_non_finite_settings_are_refused_before_anything_is_created(tmp_path):
    with pytest.raises(ValueError):
        RS.record_or_verify(tmp_path / "a" / "b", {"a": float("nan")})
    assert list(tmp_path.iterdir()) == []
    RS.record_or_verify(tmp_path / "a" / "b", {"a": 1})                         # nested new directories are made
    with pytest.raises(ValueError):
        RS.record_or_verify(tmp_path / "a" / "b", {"a": float("inf")})


def _race_worker(args):
    out, settings, barrier = args
    barrier.wait()
    try:
        RS.record_or_verify(out, settings)
        return "ok", settings["v"]
    except RS.RunSettingsMismatch:
        return "refused", settings["v"]


def test_processes_starting_together_agree_on_one_record(tmp_path):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    for trial in range(3):
        out = tmp_path / f"t{trial}"
        barrier = ctx.Manager().Barrier(8)
        with ctx.Pool(8) as pool:
            res = pool.map(_race_worker, [(out, {"v": i % 2}, barrier) for i in range(8)])
        won = json.loads((out / RS.RUN_SETTINGS_FILE).read_text())["settings"]["v"]
        assert sorted(res) == sorted(("ok" if v == won else "refused", v) for _, v in res)
        assert [p.name for p in out.iterdir()] == [RS.RUN_SETTINGS_FILE]


def test_a_record_written_first_by_another_process_is_verified_not_replaced(tmp_path, monkeypatch):
    """Two processes starting together: the loser of the link race verifies the winner's record."""
    other = {"absorber": {"tau_threshold": 300.0}}
    real_link, raced = os.link, []

    def racing_link(src, dst, *a, **k):
        if not raced:                                      # the other process links its record first
            raced.append(1)
            theirs = tmp_path / "theirs.json"
            theirs.write_text(json.dumps({"format": RS.FORMAT, "settings": other, "info": {}}))
            real_link(theirs, dst)
            theirs.unlink()
        return real_link(src, dst, *a, **k)
    monkeypatch.setattr(RS.os, "link", racing_link)
    with pytest.raises(RS.RunSettingsMismatch, match="tau_threshold: recorded 300.0, this run 100.0"):
        RS.record_or_verify(tmp_path, {"absorber": {"tau_threshold": 100.0}})
    assert json.loads((tmp_path / RS.RUN_SETTINGS_FILE).read_text())["settings"] == other
    assert sorted(x.name for x in tmp_path.iterdir()) == [RS.RUN_SETTINGS_FILE]


# ------------------------------------------------------------------------------------------------ the pipeline
def _pipeline(monkeypatch):
    """hcd_analysis.pipeline, importable without PyYAML (as in tests/test_catalog_systems.py)."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        stub = types.ModuleType("yaml")

        def _absent(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise ImportError(f"PyYAML is not installed (yaml.{name} requested; stub from tests/test_run_settings.py)")
        stub.__getattr__ = _absent
        monkeypatch.setitem(sys.modules, "yaml", stub)
    from hcd_analysis import pipeline as PL
    return PL


def _snapshot(root, nb=64, n_sk=4, z=3.0):
    """A tiny tau file with one strong system in the middle of skewer 1."""
    import h5py
    root.mkdir(parents=True, exist_ok=True)
    p = root / "s.hdf5"
    tau = np.random.default_rng(0).uniform(0.0, 0.3, size=(n_sk, nb)).astype(np.float32)
    tau[1, 20:26] = 1e5
    with h5py.File(p, "w") as f:
        g = f.create_group("Header")
        for k, v in dict(redshift=z, hubble=0.7, omegam=0.3, omegal=0.7, box=1200.0, nbins=nb, omegab=0.05).items():
            g.attrs[k] = v
        f.create_dataset("tau/H/1/1215", data=tau)
    return p


def _setup(tmp_path, monkeypatch):
    PL = _pipeline(monkeypatch)
    from hcd_analysis.config import PipelineConfig
    from hcd_analysis.io import SimInfo
    from hcd_analysis.snapshot_map import SimSnapshot, SnapEntry
    entry = SnapEntry(snap=1, a=0.25, z=3.0, path=_snapshot(tmp_path / "data"))
    ss = SimSnapshot(sim=SimInfo(name="synthsim", path=tmp_path / "simroot", params={}), entries=[entry])

    def cfg(**absorber):
        c = PipelineConfig(output_root=str(tmp_path / "out"), resume=True, n_workers=1)
        c.p1d.nhi_excl_thresholds = []
        for k, v in absorber.items():
            setattr(c.absorber, k, v)
        return c
    return PL, ss, entry, cfg


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns) for p in Path(root).rglob("*"))


def test_the_pipeline_records_its_product_settings_at_the_output_root(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    c = cfg(tau_threshold=100.0)
    assert PL.run_one_snap(ss, entry, c) is not None
    doc = json.loads((Path(c.output_root) / RS.RUN_SETTINGS_FILE).read_text())
    assert doc["settings"] == RS._canonical(PL.product_settings(c))
    assert doc["settings"]["absorber"]["tau_threshold"] == 100.0
    assert set(doc["settings"]) == {"data_root", "prefer_grid", "absorber_classes", "absorber", "p1d", "cddf",
                                    "benchmark", "debug"}


@pytest.mark.parametrize("done", [True, False])
def test_a_rerun_with_another_threshold_is_refused_before_anything_is_read_or_written(tmp_path, monkeypatch, done):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    c = cfg(tau_threshold=100.0)
    assert PL.run_one_snap(ss, entry, c) is not None
    snap_dir = PL.snap_output_dir(c, "synthsim", 1)
    if not done:
        (snap_dir / "done").unlink()
    before = _tree(c.output_root)
    for run in (lambda c2: PL.run_one_snap(ss, entry, c2), lambda c2: PL.run_sim_all_z(ss, c2)):
        with pytest.raises(RS.RunSettingsMismatch, match="absorber.tau_threshold: recorded 100.0, this run 200.0"):
            run(cfg(tau_threshold=200.0))
        assert _tree(c.output_root) == before


@pytest.mark.parametrize("key, change", [
    ("absorber.min_log_nhi", lambda c: setattr(c.absorber, "min_log_nhi", 17.5)),
    ("absorber.mask_fill_strategy", lambda c: setattr(c.absorber, "mask_fill_strategy", "zero_tau")),
    ("p1d.n_skewers", lambda c: setattr(c.p1d, "n_skewers", 1000)),
    ("cddf.A", lambda c: setattr(c.cddf, "A", 2.0)),
    ("absorber_classes", lambda c: setattr(c, "absorber_classes", ["LLS"])),
    ("data_root", lambda c: setattr(c, "data_root", "/elsewhere")),
    ("debug", lambda c: setattr(c, "debug", True))])
def test_a_rerun_into_a_new_snapshot_of_the_same_root_is_refused_on_each_product_setting(tmp_path, monkeypatch, key,
                                                                                           change):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    from hcd_analysis.snapshot_map import SnapEntry
    assert PL.run_one_snap(ss, entry, cfg(tau_threshold=100.0)) is not None
    entry2 = SnapEntry(snap=2, a=1 / 4.1, z=3.1, path=_snapshot(tmp_path / "data2", z=3.1))
    c2 = cfg(tau_threshold=100.0)
    change(c2)
    with pytest.raises(RS.RunSettingsMismatch, match=key.replace(".", r"\.") + ": recorded"):
        PL.run_one_snap(ss, entry2, c2)
    assert not (Path(tmp_path / "out") / "synthsim" / "snap_002").exists()


def test_meta_records_the_code_commit(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    c = cfg()
    PL.run_one_snap(ss, entry, c)
    meta = json.loads((PL.snap_output_dir(c, "synthsim", 1) / "meta.json").read_text())
    assert meta["code_commit"] == PL._repo_commit() and meta["code_dirty"] in (True, False, None)


def test_an_output_root_written_without_a_record_is_refused(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    c = cfg(tau_threshold=100.0)
    PL.run_one_snap(ss, entry, c)
    (Path(c.output_root) / RS.RUN_SETTINGS_FILE).unlink()                     # as written before records existed
    before = _tree(c.output_root)
    for tau in (100.0, 1000.0):
        with pytest.raises(RS.RunSettingsMismatch, match="holds files but no run_settings.json"):
            PL.run_one_snap(ss, entry, cfg(tau_threshold=tau))
    assert _tree(c.output_root) == before


def _stub_discovery(monkeypatch, PL, ss):
    """run_all / run_hires with the snapshot discovery replaced by the synthetic simulation (and joblib stubbed if
    absent; n_workers=1 never calls it)."""
    from hcd_analysis.snapshot_map import SimSnapshot
    monkeypatch.setattr(PL, "build_snapshot_map", lambda **kw: [SimSnapshot(sim=ss.sim, entries=list(ss.entries))])
    try:
        import joblib  # noqa: F401
    except ImportError:
        stub = types.ModuleType("joblib")
        stub.Parallel = stub.delayed = None
        monkeypatch.setitem(sys.modules, "joblib", stub)


@pytest.mark.parametrize("entry_point", ["run_all", "run_hires"])
def test_run_all_and_run_hires_are_refused_before_anything_is_written(tmp_path, monkeypatch, entry_point):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    _stub_discovery(monkeypatch, PL, ss)
    monkeypatch.setattr(PL, "save_config", lambda c, path: Path(path).write_text(f"tau {c.absorber.tau_threshold}\n"))
    run = getattr(PL, entry_point)
    run(cfg(tau_threshold=100.0))
    root = Path(tmp_path / "out")
    written = sorted(str(p.relative_to(root)) for p in root.rglob("*.yaml"))
    assert written == (["config_used.yaml"] if entry_point == "run_all" else ["hires/config_hires.yaml"])
    before = _tree(root)
    with pytest.raises(RS.RunSettingsMismatch, match="absorber.tau_threshold: recorded 100.0, this run 200.0"):
        run(cfg(tau_threshold=200.0))
    assert _tree(root) == before


def test_run_hires_then_run_all_into_one_root(tmp_path, monkeypatch):
    """The hires subdirectory has its own record and does not count as the low-resolution root's contents."""
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    _stub_discovery(monkeypatch, PL, ss)
    monkeypatch.setattr(PL, "save_config", lambda c, path: Path(path).write_text("x\n"))
    PL.run_hires(cfg(tau_threshold=100.0))
    PL.run_all(cfg(tau_threshold=100.0))
    root = Path(tmp_path / "out")
    assert (root / RS.RUN_SETTINGS_FILE).exists() and (root / "hires" / RS.RUN_SETTINGS_FILE).exists()


def test_convergence_ratios_check_the_records_first(tmp_path, monkeypatch, caplog):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    c = cfg(tau_threshold=100.0)
    root = Path(c.output_root)
    (root / "hires").mkdir(parents=True)
    assert PL.compute_convergence_ratios(c) == {}                            # two unrecorded roots: warned
    assert "not compared" in caplog.text
    RS.record_or_verify(root / "hires", PL.product_settings(cfg(tau_threshold=100.0)))
    with pytest.raises(RS.RunSettingsMismatch, match="low-resolution root has no run_settings.json"):
        PL.compute_convergence_ratios(c)                                     # the hires record only
    (root / "hires" / RS.RUN_SETTINGS_FILE).unlink()
    RS.record_or_verify(root, PL.product_settings(c), ignore=("hires",))
    with pytest.raises(RS.RunSettingsMismatch, match="hires root has no run_settings.json"):
        PL.compute_convergence_ratios(c)                                     # the low-resolution record only
    h = cfg(tau_threshold=300.0)
    RS.record_or_verify(root / "hires", PL.product_settings(h))
    with pytest.raises(RS.RunSettingsMismatch, match="absorber.tau_threshold: low-resolution 100.0, hires 300.0"):
        PL.compute_convergence_ratios(c)


def test_convergence_refuses_resolutions_made_with_different_settings(tmp_path, monkeypatch, caplog):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    lf, hf = tmp_path / "lf", tmp_path / "lf" / "hires"
    PL.check_convergence_settings(lf, hf)                                     # no records: warned, not refused
    assert "not compared" in caplog.text
    a, b = cfg(tau_threshold=100.0), cfg(tau_threshold=100.0)
    b.data_root = "/the/hires/suite"
    RS.record_or_verify(hf, PL.product_settings(b))
    RS.record_or_verify(lf, PL.product_settings(a), ignore=("hires",))
    PL.check_convergence_settings(lf, hf)                                     # data_root alone may differ
    other = tmp_path / "other_hires"
    c = cfg(tau_threshold=300.0)
    RS.record_or_verify(other, PL.product_settings(c))
    with pytest.raises(RS.RunSettingsMismatch, match="absorber.tau_threshold: low-resolution 100.0, hires 300.0"):
        PL.check_convergence_settings(lf, other)


def test_operational_settings_do_not_refuse_a_rerun(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    assert PL.run_one_snap(ss, entry, cfg()) is not None
    c = cfg()
    c.n_workers, c.n_workers_skewer, c.skewer_batch_size = 3, 2, 17
    c.z_min, c.z_max, c.sim_filter, c.debug_n_sims, c.debug_n_snaps = 2.5, 3.5, ["synth"], 5, 7
    assert PL.run_one_snap(ss, entry, c) is not None


def test_the_dirty_flag_is_null_when_git_cannot_tell(monkeypatch):
    PL = _pipeline(monkeypatch)
    import subprocess

    def fail(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(subprocess, "check_output", fail)
    assert PL._repo_dirty() is None
