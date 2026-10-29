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


@pytest.mark.parametrize("content", [b"not json", b"[1, 2]", b'{"format": "other/1", "settings": {}}',
                                     b'{"format": "hcd_priya.run_settings/1", "settings": 3}',
                                     b'{"format": "hcd_priya.run_settings/1", "settings": {"a": NaN}}'])
def test_an_unreadable_record_is_refused(tmp_path, content):
    (tmp_path / RS.RUN_SETTINGS_FILE).write_bytes(content)
    with pytest.raises(RS.RunSettingsMismatch):
        RS.record_or_verify(tmp_path, {"a": 1})


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


def test_non_finite_settings_are_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError):
        RS.record_or_verify(tmp_path, {"a": float("nan")})
    assert list(tmp_path.iterdir()) == []


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


def test_a_rerun_into_a_new_snapshot_of_the_same_root_is_refused_too(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    from hcd_analysis.snapshot_map import SnapEntry
    assert PL.run_one_snap(ss, entry, cfg(tau_threshold=100.0)) is not None
    entry2 = SnapEntry(snap=2, a=1 / 4.1, z=3.1, path=_snapshot(tmp_path / "data2", z=3.1))
    with pytest.raises(RS.RunSettingsMismatch):
        PL.run_one_snap(ss, entry2, cfg(min_log_nhi=17.5))
    assert not (Path(tmp_path / "out") / "synthsim" / "snap_002").exists()


def test_operational_settings_do_not_refuse_a_rerun(tmp_path, monkeypatch):
    PL, ss, entry, cfg = _setup(tmp_path, monkeypatch)
    assert PL.run_one_snap(ss, entry, cfg()) is not None
    c = cfg()
    c.n_workers, c.n_workers_skewer, c.skewer_batch_size = 3, 2, 17
    c.z_min, c.z_max, c.sim_filter, c.debug_n_sims, c.debug_n_snaps = 2.5, 3.5, ["synth"], 5, 7
    assert PL.run_one_snap(ss, entry, c) is not None
