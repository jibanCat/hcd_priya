"""scripts/compare_emulators.py: the exit status follows the verdict, and the L1c headline carries the k-grid check.

A wrapper such as ``compare_emulators.py ... && record_pass`` must never record a FAIL, BLOCKED or NOT RUN level, or a
run whose reference tree changed while it ran, as a pass. Synthetic values only; no external data.
"""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runner():
    spec = importlib.util.spec_from_file_location(
        "compare_emulators_under_test", os.path.join(REPO, "scripts", "compare_emulators.py")
    )
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    return R


def _stub_main(monkeypatch, tmp_path, R, level_result, master_hashes=None):
    """Stub the reference design, the case list and the chosen level; return the argv for main()."""
    monkeypatch.setattr(
        R, "ks_design", lambda basedir: {"params": np.zeros((2, 10)), "rungs": np.array([0.0])}
    )
    monkeypatch.setattr(
        R, "build_cases", lambda cfg, ks: [{"case_id": "c0", "support": {"status": "COMPARABLE"}}]
    )
    monkeypatch.setattr(R, "_git_head", lambda: "0" * 40)
    hashes = iter(master_hashes or [])
    monkeypatch.setattr(R, "tree_hash", lambda root: next(hashes) if root else None)

    def level(a, cfg, cases_or_ks, res_out):
        res_out.update(level_result)

    for name in ("level_L1b", "level_L1c", "level_L2LF"):
        monkeypatch.setattr(R, name, level)
    argv = [
        "compare_emulators.py",
        "--level",
        "L1b",
        "--out",
        str(tmp_path / "out"),
        "--cache",
        "c.h5",
        "--ks-basedir",
        "b",
        "--lyaemu-parent",
        "p",
        "--ks-python",
        "py",
    ]
    if master_hashes:
        argv += ["--ks-master", str(tmp_path / "master")]
    monkeypatch.setattr(R.sys, "argv", argv)
    return tmp_path / "out"


def _exit_code(R):
    with pytest.raises(SystemExit) as e:
        R.main()
    return e.value.code


@pytest.mark.parametrize("verdict", ["FAIL", "BLOCKED", "NOT RUN", "NOT COMPARABLE"])
def test_every_non_pass_verdict_exits_nonzero(monkeypatch, tmp_path, verdict):
    R = _runner()
    out = _stub_main(monkeypatch, tmp_path, R, {"verdict": verdict})
    code = _exit_code(R)
    assert code not in (0, None)
    assert code == R.EXIT_CODES[verdict]
    assert (
        json.loads((out / "result.json").read_text())["verdict"] == verdict
    )  # the record is still written


def test_non_pass_exit_codes_are_distinct():
    R = _runner()
    assert R.EXIT_CODES["PASS"] == 0
    others = [v for k, v in R.EXIT_CODES.items() if k != "PASS"]
    assert 0 not in others and len(set(others)) == len(others)
    assert R.EXIT_MASTER_CHANGED not in [0] + others


def test_pass_exits_zero(monkeypatch, tmp_path):
    R = _runner()
    _stub_main(monkeypatch, tmp_path, R, {"verdict": "PASS"})
    assert _exit_code(R) == 0


def test_pass_with_an_unchanged_master_exits_zero(monkeypatch, tmp_path):
    R = _runner()
    out = _stub_main(monkeypatch, tmp_path, R, {"verdict": "PASS"}, master_hashes=["aa", "aa"])
    assert _exit_code(R) == 0
    res = json.loads((out / "result.json").read_text())
    assert res["ks_master_unchanged"] is True and res["verdict"] == "PASS"


def test_a_reference_tree_changed_during_the_run_is_a_failure(monkeypatch, tmp_path, capsys):
    R = _runner()
    out = _stub_main(monkeypatch, tmp_path, R, {"verdict": "PASS"}, master_hashes=["aa", "bb"])
    assert _exit_code(R) == R.EXIT_MASTER_CHANGED
    res = json.loads((out / "result.json").read_text())
    assert res["ks_master_unchanged"] is False
    assert res["verdict"] == "FAIL" and res["level_verdict"] == "PASS"
    assert "PASS" not in capsys.readouterr().out.splitlines()[0]


def test_a_missing_verdict_is_not_a_pass(monkeypatch, tmp_path):
    R = _runner()
    _stub_main(monkeypatch, tmp_path, R, {})
    assert _exit_code(R) not in (0, None)


def test_a_refusal_inside_a_level_exits_1_without_result_json(monkeypatch, tmp_path):
    """As the module docstring says: a refusal exits 1 (a message to sys.exit) and writes no result.json."""
    R = _runner()
    out = _stub_main(monkeypatch, tmp_path, R, {"verdict": "PASS"})

    def refusing_level(a, cfg, cases_or_ks, res_out):
        sys.exit("refusing: synthetic")

    monkeypatch.setattr(R, "level_L1b", refusing_level)
    code = _exit_code(R)
    assert isinstance(code, str) and code.startswith("refusing:")  # Python exits 1 for a message
    assert out.is_dir() and not (out / "result.json").exists()


def test_an_existing_out_dir_is_refused_with_exit_status_1(tmp_path):
    """The real process exit status of a refusal is 1, the same code as FAIL; no result.json is written."""
    out = tmp_path / "out"
    out.mkdir()
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            os.path.join(REPO, "scripts", "compare_emulators.py"),
            "--level",
            "L1b",
            "--out",
            str(out),
            "--cache",
            "c.h5",
            "--ks-basedir",
            "b",
            "--lyaemu-parent",
            "p",
            "--ks-python",
            "py",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1 and "refusing:" in p.stderr
    assert list(out.iterdir()) == []


# --- L1c: the k-grid label check is part of the headline verdict -------------------------------------------------


def _l1c_inputs(tmp_path, kfkms_scale=1.0, p_scale=1.0, bad_k=None):
    """A two-row synthetic cache and the matching synthetic reference design, one rung, two redshifts.

    ``bad_k`` puts one non-finite k label in the second redshift row: "cache" or "reference" sets a NaN on that side,
    "zero_over_zero" sets 0 on both sides (0/0), "inf_cache" sets +inf on the cache side, and "x_over_0" sets 0 on the
    reference side only (a finite label over zero)."""
    import h5py
    from hcd_analysis.comparison import conventions as C

    rng = np.random.default_rng(3)
    theta = rng.uniform(0.5, 1.5, size=len(C.PARAM_NAMES))
    zout, nk = np.array([3.0, 2.5]), 4
    kf = np.outer(1.0 + zout, np.arange(1, nk + 1) * 1e-3)
    fv = rng.uniform(1.0, 2.0, size=(1, zout.size * nk))
    kf_ref, kf_cache = kf.copy(), kf * kfkms_scale
    if bad_k in ("cache", "zero_over_zero"):
        kf_cache[1, 2] = np.nan if bad_k == "cache" else 0.0
    if bad_k in ("reference", "zero_over_zero"):
        kf_ref[1, 2] = np.nan if bad_k == "reference" else 0.0
    if bad_k == "inf_cache":
        kf_cache[1, 2] = np.inf
    if bad_k == "x_over_0":
        kf_ref[1, 2] = 0.0
    ks = {
        "params": np.concatenate([[0.0], theta])[None, :],
        "zout": zout,
        "kfmpc": np.arange(nk, dtype=float),
        "flux_vectors": fv,
        "kfkms": kf_ref[None, :, :],
    }
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as h:
        h["params"] = np.vstack([theta, theta])
        h["param_names"] = np.array([n.encode() for n in C.PARAM_NAMES])
        h["z_grid"], h["alpha_slope"] = zout, np.zeros(2)
        h["P_tier_p"] = fv.reshape(zout.size, nk) * p_scale
        h["kfkms"] = kf_cache
        h["sim_name"] = np.array([b"sim_a", b"sim_a"])
        h["snap"] = np.array([1, 2])
        h.attrs["repair_record"] = json.dumps({"added": []})
    return ks, path


def _run_l1c(tmp_path, **kw):
    R = _runner()
    from hcd_analysis.comparison import cases as CS

    ks, cache = _l1c_inputs(tmp_path, **kw)
    out = tmp_path / "l1c"
    out.mkdir()
    a = type("A", (), {"cache": str(cache), "out": str(out)})()
    res = {}
    R.level_L1c(a, CS.load_config("phase0_v1"), ks, res)
    return res


def test_l1c_passes_when_power_and_k_labels_both_agree(tmp_path):
    res = _run_l1c(tmp_path)
    assert (
        res["lineage_verdict"] == "PASS"
        and res["kfkms_verdict"] == "PASS"
        and res["verdict"] == "PASS"
    )


def test_l1c_k_label_mismatch_cannot_read_pass(tmp_path):
    """The 2026-09-30 incident class: the power matches, the k labels do not."""
    res = _run_l1c(tmp_path, kfkms_scale=1.001)
    assert res["lineage_verdict"] == "PASS"
    assert res["kfkms_verdict"] == "FAIL" and res["verdict"] == "FAIL"


def test_l1c_power_mismatch_still_fails(tmp_path):
    res = _run_l1c(tmp_path, p_scale=1.01)
    assert (
        res["lineage_verdict"] == "FAIL"
        and res["kfkms_verdict"] == "PASS"
        and res["verdict"] == "FAIL"
    )


@pytest.mark.parametrize("bad_k", ["cache", "reference", "zero_over_zero", "inf_cache", "x_over_0"])
def test_l1c_non_finite_k_label_cannot_read_pass(tmp_path, bad_k):
    """A non-finite k label (NaN on either side, 0/0, +inf, or a finite label over 0) is a k-label FAIL, never dropped."""
    res = _run_l1c(tmp_path, bad_k=bad_k)
    assert res["lineage_verdict"] == "PASS"
    assert res["kfkms_verdict"] == "FAIL" and res["verdict"] == "FAIL"
    assert not np.isfinite(res["kfkms_max_rel"])
    assert ["sim_a", 2.5] in res["post_hoc_failing_sim_z"]


def test_l1c_non_finite_k_label_exits_nonzero_end_to_end(monkeypatch, tmp_path, capsys):
    """Through main(), with the real L1c level: the headline is FAIL and the exit status is FAIL's."""
    R = _runner()
    ks, cache = _l1c_inputs(tmp_path, bad_k="cache")
    monkeypatch.setattr(R, "ks_design", lambda basedir: dict(ks, rungs=np.array([0.0])))
    monkeypatch.setattr(
        R, "build_cases", lambda cfg, k: [{"case_id": "c0", "support": {"status": "COMPARABLE"}}]
    )
    monkeypatch.setattr(R, "_git_head", lambda: "0" * 40)
    monkeypatch.setattr(
        R.sys,
        "argv",
        [
            "compare_emulators.py",
            "--level",
            "L1c",
            "--out",
            str(tmp_path / "out"),
            "--cache",
            str(cache),
            "--ks-basedir",
            "b",
            "--lyaemu-parent",
            "p",
            "--ks-python",
            "py",
        ],
    )
    assert _exit_code(R) == R.EXIT_CODES["FAIL"]
    assert capsys.readouterr().out.splitlines()[0] == "L1c: FAIL"
    assert json.loads((tmp_path / "out" / "result.json").read_text())["kfkms_verdict"] == "FAIL"


def test_l1c_terminal_prints_the_k_grid_verdict(monkeypatch, tmp_path, capsys):
    R = _runner()
    _stub_main(
        monkeypatch,
        tmp_path,
        R,
        {
            "verdict": "FAIL",
            "lineage_verdict": "PASS",
            "kfkms_verdict": "FAIL",
            "kfkms_max_rel": 1e-3,
        },
    )
    assert _exit_code(R) == R.EXIT_CODES["FAIL"]
    out = capsys.readouterr().out
    assert "kfkms_verdict=FAIL" in out and "lineage_verdict=PASS" in out
