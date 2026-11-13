"""scripts/build_incidence_mf.py: the refusals before any input is read, the digest pins, and one build from a
synthetic LF cache and a synthetic HR incidence product, read by the historical multifidelity.load_snap_cache.
Synthetic values only."""

import hashlib
import importlib.util
import json
import os

import h5py
import numpy as np
import pytest

from hcd_analysis.emulator import incidence_mf as IM
from hcd_analysis.emulator import multifidelity as MF
from tests.test_incidence_mf import CDDF_RATIO, RATIO, SIMS, ZS, snaps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "build_incidence_mf", os.path.join(REPO, "scripts", "build_incidence_mf.py")
)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
CLEAN = lambda: {"commit": "c" * 40, "dirty": False}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture()
def inputs(tmp_path):
    """A synthetic LF cache with the datasets load_snap_cache reads (one P1D row per snapshot group) and a synthetic
    HR incidence product. Returns their paths and digests."""
    lf, hr = snaps()
    p = tmp_path / "lf_cache.h5"
    with h5py.File(p, "w") as f:
        f["snap_group_idx"] = np.arange(len(lf["z"]))
        f["z_grid"] = lf["z"]
        f["params"] = lf["params"]
        for i, c in enumerate(IM.CLASS_ORDER):
            f[f"snap_dNdX_{c}"] = lf["dndx"][:, i]
        f["snap_f_nhi"] = lf["f_nhi"]
        f["log_nhi_centres"] = lf["log_nhi"]
        f["snap_sim_name"] = np.array([s.encode() for s in lf["sim"]])
    q = tmp_path / "hr_incidence.npz"
    np.savez(
        q,
        sim_name=hr["sim"],
        z=hr["z"],
        params=hr["params"],
        dndx=hr["dndx"],
        cddf=hr["f_nhi"],
        log_nhi_centres=hr["log_nhi"],
        class_names=np.array(IM.CLASS_ORDER),
    )
    return {"lf": str(p), "hr": str(q), "lf_sha": _sha(p), "hr_sha": _sha(q), "snaps": (lf, hr)}


def _argv(inputs, out, extra=(), **over):
    a = {
        "--version": "v",
        "--out-dir": str(out),
        "--lf-cache": inputs["lf"],
        "--lf-cache-sha256": inputs["lf_sha"],
        "--hr-incidence": inputs["hr"],
        "--hr-incidence-sha256": inputs["hr_sha"],
        "--composition-id": "synthetic-composition",
        "--ruling-id": "synthetic ruling",
    }
    a.update(over)
    return [x for kv in a.items() for x in kv] + list(extra)


def test_the_synthetic_cache_is_read_by_the_historical_loader(inputs):
    lf, _ = inputs["snaps"]
    got = MF.load_snap_cache(inputs["lf"])
    for k in ("dndx", "f_nhi", "z", "pu", "log_nhi"):
        np.testing.assert_array_equal(got[k], lf[k])
    assert list(got["sim"]) == list(lf["sim"])


def test_refusals_before_any_input_is_read(inputs, tmp_path):
    (tmp_path / "there").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        B.main(_argv(inputs, tmp_path / "there"), _code=CLEAN)
    with pytest.raises(SystemExit, match="clean tree"):
        B.main(_argv(inputs, tmp_path / "out"), _code=lambda: {"commit": "c" * 40, "dirty": True})
    for key, what in (
        ("--lf-cache-sha256", "LF cache"),
        ("--hr-incidence-sha256", "HR incidence product"),
    ):
        with pytest.raises(SystemExit, match=f"the {what} .* not the pinned"):
            B.main(_argv(inputs, tmp_path / "out", **{key: "0" * 64}), _code=CLEAN)
    assert not (tmp_path / "out").exists()
    with pytest.raises(
        SystemExit
    ):  # the composition has no default: its id and a ruling are required
        B.main(
            [
                x
                for x in _argv(inputs, tmp_path / "out")
                if x not in ("--ruling-id", "synthetic ruling")
            ]
        )


def test_a_build_writes_the_record_of_the_stated_composition(inputs, tmp_path, capsys):
    out = tmp_path / "all"
    B.main(
        _argv(inputs, out, extra=["--group-not-in-product", "synthsim9 snapshot 3"]), _code=CLEAN
    )
    s = json.loads(capsys.readouterr().out)
    assert s["train_sims"] == sorted(SIMS) and s["n_pairs"] == 9 and s["z"] == list(ZS)
    inc = IM.load_record(
        out, expected_record_sha256=s["record_sha256"], lf_cache_sha256=inputs["lf_sha"]
    )
    np.testing.assert_allclose(inc.dndx_ratio, RATIO, rtol=1e-14)
    np.testing.assert_allclose(inc.cddf_ratio, CDDF_RATIO, rtol=1e-14)
    r = inc.record
    assert r["lf_cache"] == {
        "file": "lf_cache.h5",
        "sha256": inputs["lf_sha"],
        "source": IM.LF_SOURCE,
    }
    assert (
        r["hr_incidence"]["file"] == "hr_incidence.npz"
        and r["hr_incidence"]["sha256"] == inputs["hr_sha"]
    )
    assert r["hr_incidence"]["groups_not_in_product"] == ["synthsim9 snapshot 3"]
    assert r["composition"] == {
        "id": "synthetic-composition",
        "excluded_sims": [],
        "ruling_ids": ["synthetic ruling"],
    }
    assert r["fit_code"] == CLEAN() and sorted(os.listdir(out)) == [IM.RECORD_FILE]
    # another composition: one simulation left out, and the record says so
    out2 = tmp_path / "two"
    B.main(_argv(inputs, out2, extra=["--exclude-sim", "synthsimC"]), _code=CLEAN)
    s2 = json.loads(capsys.readouterr().out)
    assert s2["train_sims"] == ["synthsimA", "synthsimB"] and s2["n_sims"] == [2, 2, 2]
    assert (
        s2["composition"]["excluded_sims"] == ["synthsimC"]
        and s2["record_sha256"] != s["record_sha256"]
    )


def test_a_build_refuses_what_the_fit_refuses_and_writes_nothing(inputs, tmp_path):
    with pytest.raises(SystemExit, match="refusing: unknown simulations"):
        B.main(_argv(inputs, tmp_path / "out", extra=["--exclude-sim", "no_such_sim"]), _code=CLEAN)
    assert not (tmp_path / "out").exists()
