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
_spec = importlib.util.spec_from_file_location(
    "build_mf_release", os.path.join(REPO, "scripts", "build_mf_release.py")
)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
K = RT.SYNTHETIC_ARCH["n_k"]
SIMS = ("synthsim0", "synthsim1", "synthsim2", "synthsim3")
CLEAN = lambda: {"commit": COMMIT, "dirty": False}  # noqa: E731
DIRTY = lambda: {"commit": COMMIT, "dirty": True}  # noqa: E731


def _argv(tmp_path, out="out", kind="release", extra=()):
    return [
        "--kind",
        kind,
        "--mf-version",
        "v",
        "--out-dir",
        str(tmp_path / out),
        "--lf-cache",
        str(tmp_path / "lf.h5"),
        "--lf-cache-sha256",
        H64("lf"),
        "--hr-cache",
        str(tmp_path / "hr.h5"),
        "--hr-cache-sha256",
        H64("hr"),
        "--backbone",
        str(tmp_path / "bundle"),
        "--backbone-manifest-sha256",
        H64("b"),
        "--composition-id",
        "c",
        "--ruling-id",
        "r",
        *extra,
    ]


# --------------------------------------------------------------------------------------------- refusals first
def test_existing_out_dir_is_refused_before_any_input_is_read(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        B.main(_argv(tmp_path), _code=CLEAN)  # the caches do not even exist


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
        B.parse(_argv(tmp_path, extra=("--holdout-sim", "x")))  # a release takes no held-out group
    with pytest.raises(SystemExit):
        B.parse(_argv(tmp_path, kind="fold"))  # a fold needs its id and group
    a = B.parse(_argv(tmp_path, kind="fold", extra=("--fold-id", "f", "--holdout-sim", "x")))
    assert a.holdout_sim == ["x"] and a.exclude_sim == []
    assert a.validated_z_min == MR.VALIDATED_Z_MIN == 2.2  # the default
    assert B.parse(_argv(tmp_path, extra=("--validated-z-min", "2.6"))).validated_z_min == 2.6


# --------------------------------------------------------------------------------------------- the build
def _inputs(drop_catalogue=False):
    lf, hr = synthetic_pairs(sims=SIMS)
    from hcd_analysis.emulator.schema import L_BOX_HMPC

    k_ext = 2 * np.pi * np.arange(1, K + 5) / L_BOX_HMPC
    kf = np.stack(
        [
            np.asarray(KC.k_skm_from_kcom(k_ext, z, p[5], p[6]))
            for z, p in zip(lf["z_grid"], lf["params"])
        ]
    )
    lf, hr = dict(lf, kfkms=kf, k_com_hmpc=k_ext), dict(hr, kfkms=kf, k_com_hmpc=k_ext)
    if drop_catalogue:
        hr = {k: v for k, v in hr.items() if not k.startswith("snap_")}
    return lf, hr


def _caches(bundle):
    return {
        "lf": {"file": RT.SYNTHETIC_CACHE_NAME, "sha256": bundle.lineage["cache"]["sha256"]},
        "hr": {"file": "synthetic_hr.h5", "sha256": H64("synthetic hr")},
    }


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("b") / "bundle"), n_members=2)


def test_release_build_binds_to_the_backbone_and_uses_its_lf_prediction(info, tmp_path):
    lf, hr = _inputs()
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {
        "id": "synthetic-x3",
        "excluded_sims": ["synthsim3"],
        "ruling_ids": ["synthetic ruling"],
    }
    path, sha, summ = B.build_from_inputs(
        str(tmp_path / "mf"),
        lf=lf,
        hr=hr,
        backbone=bb,
        kind="release",
        mf_version="v1",
        caches=_caches(bb),
        composition=comp,
        fold=None,
        fit_code=CLEAN(),
    )
    assert (
        summ["train_sims"] == ["synthsim0", "synthsim1", "synthsim2"] and summ["n_pairs"] == 4 * 9
    )
    rec = MR.load_mf_record(
        str(tmp_path / "mf"),
        expected_record_sha256=sha,
        backbone_manifest_sha256=info.manifest_sha256,
        composition=comp,
    )
    assert rec.record["backbone"]["manifest_sha256"] == info.manifest_sha256
    # the targets are log P_HR - log P_backbone: refit them independently and compare the tables bit for bit
    from hcd_analysis.emulator import mf_modes as MM

    pairs = MM.match_pairs(lf, hr)
    l_rows = np.array([p[1] for p in pairs])
    P_lf = hp.lf_P_filt_at(bb, lf["x"][l_rows], lf["tau0"][l_rows])
    np.testing.assert_array_equal(
        P_lf[0],
        bb.predict(
            {n: float(v) for n, v in zip(hp.PARAM_NAMES, lf["params"][l_rows[0]])},
            float(lf["z_grid"][l_rows[0]]),
            tau_eff=float(lf["tau0"][l_rows[0]]),
            fidelity="LF",
        ).P_class,
    )
    t = MM.measure_mode_targets(lf, MR.hr_view(hr), pairs, P_lf, x=lf["x"], n_modes=K)
    ref = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != "synthsim3")[0])
    for k in MM.TABLE_KEYS:
        np.testing.assert_array_equal(rec.tables[k], np.asarray(ref[k], float))
    emu = hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=str(tmp_path / "mf"), mf_record_sha256=sha
    )
    assert "mf" in emu.provenance


def test_build_never_reads_the_hr_catalogue_columns(info, tmp_path):
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {"id": "c", "excluded_sims": [], "ruling_ids": ["r"]}
    out = []
    for i, drop in enumerate((False, True)):
        lf, hr = _inputs(drop_catalogue=drop)
        _, sha, _ = B.build_from_inputs(
            str(tmp_path / f"mf{i}"),
            lf=lf,
            hr=hr,
            backbone=bb,
            kind="release",
            mf_version="v1",
            caches=_caches(bb),
            composition=comp,
            fold=None,
            fit_code=CLEAN(),
        )
        out.append(open(tmp_path / f"mf{i}" / MR.TABLES_FILE, "rb").read())
    assert out[0] == out[1]


def test_fold_build_on_a_research_backbone_holds_out_the_group(tmp_path):
    res = RT.write_synthetic_research_bundle(
        str(tmp_path / "res"), held_out_sims=("synthsim2",), n_groups=3
    )
    bb = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test")
    lf, hr = _inputs()
    comp = {
        "id": "synthetic-x3",
        "excluded_sims": ["synthsim3"],
        "ruling_ids": ["synthetic ruling"],
    }
    fold = {"fold_id": "f2", "holdout_sims": ["synthsim2"]}
    _, sha, summ = B.build_from_inputs(
        str(tmp_path / "fold"),
        lf=lf,
        hr=hr,
        backbone=bb,
        kind="fold",
        mf_version="v1",
        caches=_caches(bb),
        composition=comp,
        fold=fold,
        fit_code=DIRTY(),
    )
    rec = MR.load_mf_record(
        str(tmp_path / "fold"),
        expected_record_sha256=sha,
        backbone_manifest_sha256=res.manifest_sha256,
        kind="fold",
    )
    assert rec.record["train_sims"] == ["synthsim0", "synthsim1"]
    assert rec.record["fold"]["backbone_excludes_holdout"] is True
    assert rec.record["backbone"] == {
        "manifest_sha256": res.manifest_sha256,
        "kind": "research",
        "cache_sha256": res.lineage["cache"]["sha256"],
        "held_out_sims": ["synthsim2"],
    }
    rp = hp.load_research_bundle(
        res.bundle_dir,
        res.manifest_sha256,
        purpose="a test",
        mf_dir=str(tmp_path / "fold"),
        mf_record_sha256=sha,
    )
    assert "mf" in rp.provenance


def test_build_takes_the_backbone_identity_from_a_loaded_predictor_only(info, tmp_path):
    lf, hr = _inputs()
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    fake = {
        "manifest_sha256": info.manifest_sha256,
        "kind": "research",
        "held_out_sims": ["synthsim2"],
    }
    with pytest.raises(SystemExit, match="loaded predictor"):
        B.build_from_inputs(
            str(tmp_path / "x"),
            lf=lf,
            hr=hr,
            backbone=fake,
            kind="release",
            mf_version="v1",
            caches=_caches(bb),
            composition={"id": "c", "excluded_sims": [], "ruling_ids": ["r"]},
            fold=None,
            fit_code=CLEAN(),
        )
    assert not os.path.lexists(tmp_path / "x")


# --------------------------------------------------------------------- the D1 stored-grid exception (one design point)
EXC = MR.D1_EXCEPTION
D1_SIMS = (EXC["lf_sim"], "synthsim1", "synthsim2", "synthsim3")


def _inputs_d1(err, group, *, hr_name=None, side="lf", name_group=None):
    """Synthetic matched caches whose stored grids of one LF group differ by a factor (1 + err) between LF and HR
    (side "lf"; side "hr" perturbs the HR grid, which the canonical check sees). The exception group's HR rows carry
    the recorded HR run name (or ``hr_name``), or ``name_group``'s HR rows do."""
    lf, hr = synthetic_pairs(sims=D1_SIMS)
    from hcd_analysis.emulator.schema import L_BOX_HMPC

    k_ext = 2 * np.pi * np.arange(1, K + 5) / L_BOX_HMPC
    kf = np.stack(
        [
            np.asarray(KC.k_skm_from_kcom(k_ext, z, p[5], p[6]))
            for z, p in zip(lf["z_grid"], lf["params"])
        ]
    )
    rows = np.asarray(lf["sim_name"]) == group
    kl, kh = kf.copy(), kf.copy()
    (kl if side == "lf" else kh)[rows] *= 1.0 + err
    names = np.asarray(hr["sim_name"]).astype(object)
    names[np.asarray(lf["sim_name"]) == (name_group or EXC["lf_sim"])] = hr_name or EXC["hr_sim"]
    return dict(lf, kfkms=kl, k_com_hmpc=k_ext), dict(
        hr, kfkms=kh, k_com_hmpc=k_ext, sim_name=names.astype(str)
    )


def _build_d1(info, tmp_path, lf, hr, name="mf"):
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {
        "id": "synthetic-x3",
        "excluded_sims": ["synthsim3"],
        "ruling_ids": ["synthetic ruling"],
    }
    return B.build_from_inputs(
        str(tmp_path / name),
        lf=lf,
        hr=hr,
        backbone=bb,
        kind="release",
        mf_version="v1",
        caches=_caches(bb),
        composition=comp,
        fold=None,
        fit_code=CLEAN(),
    )


def test_d1_exception_passes_the_recorded_case_and_is_recorded(info, tmp_path):
    err = 0.6 * EXC["bound"]
    _, sha, summ = _build_d1(info, tmp_path, *_inputs_d1(err, EXC["lf_sim"]))
    rec = MR.load_mf_record(
        str(tmp_path / "mf"),
        expected_record_sha256=sha,
        backbone_manifest_sha256=info.manifest_sha256,
    )
    mm = rec.record["mode_mapping"]
    assert mm["tol"] == B.D1_TOL == MR.D1_TOL == 1e-12
    assert mm["stored_max"] <= 1e-12 and mm["canonical_max"] <= 1e-12
    exc = mm["exception"]
    assert {k: exc[k] for k in EXC} == EXC and exc[
        "n_pairs"
    ] == 9  # every pair of the group, no other
    assert abs(exc["max_error"] / err - 1.0) < 1e-4
    assert summ["D1"] == mm
    assert rec.record["validated_z_min"] == 2.2 and summ["validated_z_min"] == 2.2
    emu = hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=str(tmp_path / "mf"), mf_record_sha256=sha
    )
    assert emu.provenance["mf"]["mode_mapping"] == mm
    _, sha0, _ = _build_d1(info, tmp_path, *_inputs_d1(0.0, EXC["lf_sim"]), name="none")
    assert (
        MR.load_mf_record(
            str(tmp_path / "none"),
            expected_record_sha256=sha0,
            backbone_manifest_sha256=info.manifest_sha256,
        ).record["mode_mapping"]["exception"]
        is None
    )


def test_build_records_a_raised_validated_z_min(info, tmp_path):
    lf, hr = _inputs()
    bb = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    comp = {"id": "c", "excluded_sims": [], "ruling_ids": ["r"]}
    _, sha, summ = B.build_from_inputs(
        str(tmp_path / "mf"),
        lf=lf,
        hr=hr,
        backbone=bb,
        kind="release",
        mf_version="v1",
        caches=_caches(bb),
        composition=comp,
        fold=None,
        fit_code=CLEAN(),
        validated_z_min=2.6,
    )
    rec = MR.load_mf_record(
        str(tmp_path / "mf"),
        expected_record_sha256=sha,
        backbone_manifest_sha256=info.manifest_sha256,
    )
    assert rec.record["validated_z_min"] == 2.6 == summ["validated_z_min"]
    with pytest.raises(MR.MFRecordError, match="validated_z_min"):  # never below the code floor
        B.build_from_inputs(
            str(tmp_path / "lo"),
            lf=lf,
            hr=hr,
            backbone=bb,
            kind="release",
            mf_version="v1",
            caches=_caches(bb),
            composition=comp,
            fold=None,
            fit_code=CLEAN(),
            validated_z_min=2.0,
        )
    assert not os.path.lexists(tmp_path / "lo")


def test_main_writes_the_validated_z_min_it_is_given(tmp_path, monkeypatch, capsys):
    """--validated-z-min reaches the written record through main() (the default 2.2, and a raised 2.6), on a synthetic
    build: real cache pins, a real synthetic backbone bound to the pinned LF bytes, the real build, write and reload.
    Only the two cache readers are stubbed (they return synthetic matched caches)."""
    import hashlib
    import json
    from hcd_analysis.emulator import data as D

    lf_bytes, hr_bytes = b"synthetic lf cache bytes", b"synthetic hr cache bytes"
    (tmp_path / "lf.h5").write_bytes(lf_bytes)
    (tmp_path / "hr.h5").write_bytes(hr_bytes)
    lf_sha, hr_sha = hashlib.sha256(lf_bytes).hexdigest(), hashlib.sha256(hr_bytes).hexdigest()
    crec, _ = RT.synthetic_cache_record()
    bb = RT.write_synthetic_bundle(
        str(tmp_path / "bundle"),
        n_members=2,
        cache=(crec, {"sha256": lf_sha, "size": len(lf_bytes)}),
    )
    lf, hr = _inputs()
    monkeypatch.setattr(D, "load_cache", lambda path, **k: dict(lf, cache_sha256=lf_sha))
    monkeypatch.setattr(D, "load_cache_historical", lambda path, *, reason, validate=True: hr)
    for out, extra, want in (
        ("mf_default", (), 2.2),
        ("mf_raised", ("--validated-z-min", "2.6"), 2.6),
    ):
        argv = _argv(tmp_path, out=out, extra=extra)
        for flag, value in (
            ("--lf-cache-sha256", lf_sha),
            ("--hr-cache-sha256", hr_sha),
            ("--backbone-manifest-sha256", bb.manifest_sha256),
        ):
            argv[argv.index(flag) + 1] = value
        B.main(argv, _code=CLEAN)
        printed = json.loads(capsys.readouterr().out)
        rec = MR.load_mf_record(
            str(tmp_path / out),
            expected_record_sha256=printed["record_sha256"],
            backbone_manifest_sha256=bb.manifest_sha256,
        )
        assert rec.record["validated_z_min"] == want == printed["validated_z_min"]
        assert rec.provenance()["validated_z_min"] == want


@pytest.mark.parametrize(
    "err, group, kw",
    [
        (0.6 * EXC["bound"], "synthsim1", {}),  # the same error on another group
        (1.2 * EXC["bound"], EXC["lf_sim"], {}),  # a larger error on the recorded group
        (
            0.6 * EXC["bound"],
            EXC["lf_sim"],
            {"hr_name": "synthsim0"},
        ),  # the recorded LF run with another HR run
        (
            0.6 * EXC["bound"],
            "synthsim1",
            {"name_group": "synthsim1"},
        ),  # the recorded HR run name on another LF run
        (0.6 * EXC["bound"], EXC["lf_sim"], {"side": "hr"}),  # the canonical mapping stays strict
    ],
)
def test_d1_exception_is_narrow(info, tmp_path, err, group, kw):
    with pytest.raises(SystemExit, match="do not map one to one"):
        _build_d1(info, tmp_path, *_inputs_d1(err, group, **kw))
    assert not os.path.lexists(tmp_path / "mf")


# --------------------------------------------------------------------- code identity and TOCTOU (C1 review MAJOR-2)
def _git(repo, *args):
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        env=env,
    )


def test_code_identity_reads_the_commit_and_the_dirty_state_of_a_real_checkout(
    tmp_path, monkeypatch
):
    """R13a, R13b: the real git check (main's default), on a temporary repository: clean, a modified tracked file and
    an untracked file; a directory that is not a checkout is refused."""
    for k in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(k)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "synthetic")
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert B.code_identity(str(repo)) == {"commit": head, "dirty": False}
    (repo / "a.txt").write_text("two\n")
    assert B.code_identity(str(repo)) == {"commit": head, "dirty": True}
    _git(repo, "checkout", "-q", "--", "a.txt")
    assert B.code_identity(str(repo))["dirty"] is False
    (repo / "untracked.txt").write_text("new\n")
    assert B.code_identity(str(repo))["dirty"] is True
    (tmp_path / "plain").mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(SystemExit, match="code identity"):
        B.code_identity(str(tmp_path / "plain"))


def _pinned_inputs(tmp_path):
    import hashlib

    (tmp_path / "lf.h5").write_bytes(b"synthetic lf bytes")
    (tmp_path / "hr.h5").write_bytes(b"synthetic hr bytes")
    lf_sha = hashlib.sha256(b"synthetic lf bytes").hexdigest()
    argv = _argv(tmp_path)
    argv[argv.index("--lf-cache-sha256") + 1] = lf_sha
    argv[argv.index("--hr-cache-sha256") + 1] = hashlib.sha256(b"synthetic hr bytes").hexdigest()
    return argv, lf_sha


def _stub_loaders(monkeypatch, lf_sha, *, lf_loaded_sha, during_hr_read=None):
    """main() with the backbone and cache readers stubbed (the pins and the re-checks are the code under test)."""
    from hcd_analysis.emulator import data as D
    from hcd_analysis.emulator import safe_h5 as H

    class Backbone:
        lineage = {"cache": {"sha256": lf_sha}}

    def hist(path, *, reason, validate=True):
        if during_hr_read:
            during_hr_read(path)
        return {}

    monkeypatch.setattr(H, "load_release", lambda *a, **k: object())
    monkeypatch.setattr(hp, "Bundle", lambda rb: Backbone())
    monkeypatch.setattr(D, "load_cache", lambda path, **k: {"cache_sha256": lf_loaded_sha})
    monkeypatch.setattr(D, "load_cache_historical", hist)
    monkeypatch.setattr(B, "build_from_inputs", lambda *a, **k: pytest.fail("reached the build"))


def test_lf_cache_digest_at_read_is_compared_with_the_pin(tmp_path, monkeypatch):
    """R3d: the LF cache's digest as loaded (data.load_cache hashes the file it reads) must be the pinned one."""
    argv, lf_sha = _pinned_inputs(tmp_path)
    _stub_loaders(monkeypatch, lf_sha, lf_loaded_sha=H64("changed after hashing"))
    with pytest.raises(SystemExit, match="LF cache changed after it was hashed"):
        B.main(argv, _code=CLEAN)
    assert not os.path.lexists(tmp_path / "out")


def test_hr_cache_is_rehashed_after_the_read(tmp_path, monkeypatch):
    """R3b: an HR cache that changes while it is read (between the pin check and the end of the read) is refused."""
    argv, lf_sha = _pinned_inputs(tmp_path)

    def change(path):
        with open(path, "ab") as fh:
            fh.write(b" changed")

    _stub_loaders(monkeypatch, lf_sha, lf_loaded_sha=lf_sha, during_hr_read=change)
    with pytest.raises(SystemExit, match="HR cache changed while it was read"):
        B.main(argv, _code=CLEAN)
    assert not os.path.lexists(tmp_path / "out")
