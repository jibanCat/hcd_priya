"""scripts/build_mf_release.py: the refusals before any input is read (write once, clean tree for a release, the cache
pins, the argument rules) and the build itself on synthetic inputs (a synthetic h5/2 backbone, synthetic matched
caches): the record binds to the backbone, its targets use the backbone's LF prediction, folds hold out simulation
groups, and the HR catalogue-derived columns are never read. Synthetic values only."""
import importlib.util
import os

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import mf_release as MR
from hcd_analysis.emulator import release_testing as RT
from tests.test_mf_release import COMMIT, H64, synthetic_pairs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("build_mf_release", os.path.join(REPO, "scripts", "build_mf_release.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
K = RT.SYNTHETIC_ARCH["n_k"]
SIMS = ("synthsim0", "synthsim1", "synthsim2", "synthsim3")
CLEAN = lambda: {"commit": COMMIT, "dirty": False}               # noqa: E731
DIRTY = lambda: {"commit": COMMIT, "dirty": True}                # noqa: E731


def _argv(tmp_path, out="out", kind="release", extra=()):
    return ["--kind", kind, "--mf-version", "v", "--out-dir", str(tmp_path / out),
            "--lf-cache", str(tmp_path / "lf.h5"), "--lf-cache-sha256", H64("lf"),
            "--hr-cache", str(tmp_path / "hr.h5"), "--hr-cache-sha256", H64("hr"),
            "--backbone", str(tmp_path / "bundle"), "--backbone-manifest-sha256", H64("b"),
            "--composition-id", "c", "--ruling-id", "r", *extra]


# --------------------------------------------------------------------------------------------- refusals first
def test_existing_out_dir_is_refused_before_any_input_is_read(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        B.main(_argv(tmp_path), _code=CLEAN)                      # the caches do not even exist


def test_release_from_a_dirty_tree_is_refused_before_any_input_is_read(tmp_path):
    with pytest.raises(SystemExit, match="clean tree"):
        B.main(_argv(tmp_path), _code=DIRTY)
    assert not os.path.lexists(tmp_path / "out")


def test_cache_pins_are_checked(tmp_path):
    (tmp_path / "lf.h5").write_bytes(b"not the pinned lf")
    (tmp_path / "hr.h5").write_bytes(b"not the pinned hr")
    with pytest.raises(SystemExit, match="LF cache .* not the pinned"):
        B.main(_argv(tmp_path), _code=CLEAN)
    import hashlib
    argv = _argv(tmp_path)
    argv[argv.index("--lf-cache-sha256") + 1] = hashlib.sha256(b"not the pinned lf").hexdigest()
    with pytest.raises(SystemExit, match="HR cache .* not the pinned"):
        B.main(argv, _code=CLEAN)
    assert not os.path.lexists(tmp_path / "out")


def test_argument_rules(tmp_path):
    with pytest.raises(SystemExit):
        B.parse(_argv(tmp_path, extra=("--holdout-sim", "x")))    # a release takes no held-out group
    with pytest.raises(SystemExit):
        B.parse(_argv(tmp_path, kind="fold"))                      # a fold needs its id and group
    a = B.parse(_argv(tmp_path, kind="fold", extra=("--fold-id", "f", "--holdout-sim", "x")))
    assert a.holdout_sim == ["x"] and a.exclude_sim == []


# --------------------------------------------------------------------------------------------- the build
def _inputs(drop_catalogue=False):
    lf, hr = synthetic_pairs(sims=SIMS)
    from hcd_analysis.emulator.schema import L_BOX_HMPC
    k_ext = 2 * np.pi * np.arange(1, K + 5) / L_BOX_HMPC
    kf = np.stack([np.asarray(KC.k_skm_from_kcom(k_ext, z, p[5], p[6])) for z, p in zip(lf["z_grid"], lf["params"])])
    lf, hr = dict(lf, kfkms=kf, k_com_hmpc=k_ext), dict(hr, kfkms=kf, k_com_hmpc=k_ext)
    if drop_catalogue:
        hr = {k: v for k, v in hr.items() if not k.startswith("snap_")}
    return lf, hr


def _caches(bundle):
    return {"lf": {"file": RT.SYNTHETIC_CACHE_NAME, "sha256": bundle.lineage["cache"]["sha256"]},
            "hr": {"file": "synthetic_hr.h5", "sha256": H64("synthetic hr")}}


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("b") / "bundle"), n_members=2)


def test_release_build_binds_to_the_backbone_and_uses_its_lf_prediction(info, tmp_path):
    lf, hr = _inputs()
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {"id": "synthetic-x3", "excluded_sims": ["synthsim3"], "ruling_ids": ["synthetic ruling"]}
    path, sha, summ = B.build_from_inputs(str(tmp_path / "mf"), lf=lf, hr=hr, backbone=bb, kind="release",
                                          mf_version="v1", caches=_caches(bb), composition=comp, fold=None,
                                          fit_code=CLEAN())
    assert summ["train_sims"] == ["synthsim0", "synthsim1", "synthsim2"] and summ["n_pairs"] == 4 * 9
    rec = MR.load_mf_record(str(tmp_path / "mf"), expected_record_sha256=sha,
                            backbone_manifest_sha256=info.manifest_sha256, composition=comp)
    assert rec.record["backbone"]["manifest_sha256"] == info.manifest_sha256
    # the targets are log P_HR - log P_backbone: refit them independently and compare the tables bit for bit
    from hcd_analysis.emulator import mf_modes as MM
    pairs = MM.match_pairs(lf, hr)
    l_rows = np.array([p[1] for p in pairs])
    P_lf = bb.lf_P_filt_at(lf["x"][l_rows], lf["tau0"][l_rows])
    np.testing.assert_array_equal(P_lf[0], bb.predict({n: float(v) for n, v in zip(hp.PARAM_NAMES,
                                                                                      lf["params"][l_rows[0]])},
                                                      float(lf["z_grid"][l_rows[0]]),
                                                      tau_eff=float(lf["tau0"][l_rows[0]]), fidelity="LF").P_class)
    t = MM.measure_mode_targets(lf, MR.hr_view(hr), pairs, P_lf, x=lf["x"], n_modes=K)
    ref = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != "synthsim3")[0])
    for k in MM.TABLE_KEYS:
        np.testing.assert_array_equal(rec.tables[k], np.asarray(ref[k], float))
    emu = hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(tmp_path / "mf"), mf_record_sha256=sha)
    assert emu.has_mf


def test_build_never_reads_the_hr_catalogue_columns(info, tmp_path):
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {"id": "c", "excluded_sims": [], "ruling_ids": ["r"]}
    out = []
    for i, drop in enumerate((False, True)):
        lf, hr = _inputs(drop_catalogue=drop)
        _, sha, _ = B.build_from_inputs(str(tmp_path / f"mf{i}"), lf=lf, hr=hr, backbone=bb, kind="release",
                                        mf_version="v1", caches=_caches(bb), composition=comp, fold=None,
                                        fit_code=CLEAN())
        out.append(open(tmp_path / f"mf{i}" / MR.TABLES_FILE, "rb").read())
    assert out[0] == out[1]


def test_fold_build_on_a_research_backbone_holds_out_the_group(tmp_path):
    res = RT.write_synthetic_research_bundle(str(tmp_path / "res"), held_out_sims=("synthsim2",), n_groups=3)
    bb = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test")
    lf, hr = _inputs()
    comp = {"id": "synthetic-x3", "excluded_sims": ["synthsim3"], "ruling_ids": ["synthetic ruling"]}
    fold = {"fold_id": "f2", "holdout_sims": ["synthsim2"]}
    _, sha, summ = B.build_from_inputs(str(tmp_path / "fold"), lf=lf, hr=hr, backbone=bb, kind="fold",
                                       mf_version="v1", caches=_caches(bb), composition=comp, fold=fold,
                                       fit_code=DIRTY())
    rec = MR.load_mf_record(str(tmp_path / "fold"), expected_record_sha256=sha,
                            backbone_manifest_sha256=res.manifest_sha256, kind="fold")
    assert rec.record["train_sims"] == ["synthsim0", "synthsim1"]
    assert rec.record["fold"]["backbone_excludes_holdout"] is True
    assert rec.record["backbone"] == {"manifest_sha256": res.manifest_sha256, "kind": "research",
                                      "cache_sha256": res.lineage["cache"]["sha256"], "held_out_sims": ["synthsim2"]}
    rp = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(tmp_path / "fold"),
                                 mf_record_sha256=sha)
    assert rp.has_mf
