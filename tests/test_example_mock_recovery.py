"""The example examples/mock_recovery.py: part one on a synthetic model and a synthetic mock, the settings it writes
out, and the input it hands to Cobaya (built here without Cobaya; the chain itself is not run in the test tiers).
Synthetic values only."""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import likelihood as L
from hcd_analysis.emulator.data import PARAM_LIMITS
from tests.synthetic_release import make_release
from tests.test_predictor_mf import PARAMS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THETA = dict(PARAMS, tau0_amp=1.0, dtau0=0.0)
ZS = (2.6, 3.0)


def _script():
    spec = importlib.util.spec_from_file_location(
        "mock_recovery", os.path.join(REPO, "examples", "mock_recovery.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A synthetic release, and a mock made from it at THETA with a made-up covariance, as a data-set file."""
    root = tmp_path_factory.mktemp("example")
    info = make_release(root / "release", incidence=True)
    emu = hp.load_release(str(info["root"]), info["sha"])
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    rows = [0.5 * (np.asarray(pred.k[i])[1:4] + np.asarray(pred.k[i])[2:5]) for i in range(len(ZS))]
    blank = L.P1DData.from_blocks(z=ZS, k=rows, p1d=[np.ones(3)] * 2, cov_blocks=[np.eye(3)] * 2)
    made_at = L.GaussianP1DLikelihood(emu, blank).model_vector(THETA)
    mock = L.P1DData.from_blocks(
        z=ZS,
        k=rows,
        p1d=[made_at[:3], made_at[3:]],
        cov_blocks=[1e-4 * np.eye(3) * made_at[:3] ** 2, 1e-4 * np.eye(3) * made_at[3:] ** 2],
        truth=THETA,
        provenance={"what": "a synthetic mock made in a test"},
    )
    path = root / "mock.json"
    sha = L.write_p1d_data(path, mock)
    return {
        "emu": emu,
        "release": [str(info["root"]), info["sha"]],
        "data": [str(path), sha],
        "root": root,
    }


def test_part_one_prints_labelled_lines_and_a_zero_chi_square_at_the_truth(setup, capsys):
    ex = _script()
    assert ex.main(["--release", *setup["release"], "--data", *setup["data"]]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "=== Part 1: the likelihood at the point the mock was made at"
    assert "bins in the data vector: 6" in lines and "redshifts: [2.6, 3.0]" in lines
    (chi2,) = [ln for ln in lines if ln.startswith("chi-square at the truth")]
    assert chi2.endswith(": 0.0")
    assert any(ln.startswith("flags set at the truth") and "mf_applied" in ln for ln in lines)
    assert "=== Part 2: no chain was asked for (--demo N, or --full)" in lines
    assert "=== Part 3: no chain, so no posterior predictive" in lines
    body = [ln for ln in lines if not ln.startswith("===")]
    assert all(": " in ln or ln.endswith(":") for ln in body), "every printed line is labelled"


def test_without_cobaya_the_chain_is_skipped_and_said_so(setup, capsys, monkeypatch):
    ex = _script()
    real = importlib.util.find_spec
    monkeypatch.setattr(
        ex.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "cobaya" else real(name, *a),
    )
    assert ex.main(["--release", *setup["release"], "--data", *setup["data"], "--demo", "5"]) == 0
    out = capsys.readouterr().out
    assert "=== Part 2: skipped. Cobaya is not installed" in out
    assert "=== Part 3: no chain, so no posterior predictive" in out


def test_the_settings_are_written_out_and_lie_inside_the_trained_box():
    ex = _script()
    assert list(ex.BOXES) == list(hp.PARAM_NAMES) + ["tau0_amp", "dtau0"]
    for name, (lo, hi) in zip(hp.PARAM_NAMES, PARAM_LIMITS):
        a, b = ex.BOXES[name]
        assert lo <= a < b <= hi, name  # never wider than the box the model was trained on
    assert ex.BOXES["herei"] == (3.5, 4.1) and ex.BOXES["heref"] == (2.6, 3.2)
    assert ex.BOXES["tau0_amp"] == (0.75, 1.25) and ex.BOXES["dtau0"] == (-0.4, 0.25)
    assert set(ex.GAUSSIAN_TERMS) == {"hub", "bhfeedback"}
    assert ex.GAUSSIAN_TERMS["hub"] == (0.70, pytest.approx(0.015 / math.sqrt(2)))
    assert ex.GAUSSIAN_TERMS["bhfeedback"] == (0.05, pytest.approx(0.005 / math.sqrt(2)))
    assert ex.FULL == {"Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.05, "learn_proposal": True}


def test_the_input_for_cobaya_is_flat_boxes_the_truth_and_the_sampler_log_probability(setup):
    ex = _script()
    data = L.read_p1d_data(setup["data"][0], expected_sha256=setup["data"][1])
    like = L.GaussianP1DLikelihood(setup["emu"], data, gaussian_terms=ex.GAUSSIAN_TERMS)
    info = ex.cobaya_info(like, THETA, demo=7, seed=3, out=str(setup["root"] / "out"))
    assert list(info["params"]) == list(like.param_names)
    for name, p in info["params"].items():
        lo, hi = ex.BOXES[name]
        assert p["prior"] == {"min": lo, "max": hi} and p["ref"] == THETA[name]
        assert p["proposal"] == pytest.approx(ex.PROPOSAL_FRACTION * (hi - lo))
    assert info["sampler"]["mcmc"]["max_samples"] == 7 and info["sampler"]["mcmc"]["seed"] == 3
    assert info["output"].endswith(os.path.join("out", "chain"))
    full = ex.cobaya_info(like, THETA, demo=0, seed=3, out="o")["sampler"]["mcmc"]
    assert full["Rminus1_stop"] == 0.01 and full["Rminus1_cl_stop"] == 0.05
    assert "max_samples" not in full
    logp = info["likelihood"]["p1d"]["external"]
    assert logp(**THETA) == like.logp(THETA)
    assert (
        logp(**dict(THETA, ns=float("nan"))) == -math.inf
        and sum(v[0] for v in like.refusals.values()) == 1
    )


def test_what_the_example_refuses(setup, capsys, tmp_path):
    ex = _script()
    for argv, why in (
        (["--data", *setup["data"]], "give the model once"),
        (
            ["--release", *setup["release"], "--research", "d", "s", "--data", *setup["data"]],
            "give the model once",
        ),
        (
            ["--release", *setup["release"], "--purpose", "x", "--data", *setup["data"]],
            "belong to --research",
        ),
        (["--research", "d", "s", "--data", *setup["data"]], "--research needs --purpose"),
        (
            ["--release", *setup["release"], "--data", *setup["data"], "--demo", "5", "--full"],
            "not both",
        ),
    ):
        with pytest.raises(SystemExit) as e:
            ex.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err
    # a data set that states no truth is not a mock
    d = L.read_p1d_data(setup["data"][0], expected_sha256=setup["data"][1])
    bare = L.P1DData(z=d.z, k=d.k, p1d=d.p1d, cov=d.cov)
    sha = L.write_p1d_data(tmp_path / "bare.json", bare)
    with pytest.raises(SystemExit, match="states no truth"):
        ex.main(["--release", *setup["release"], "--data", str(tmp_path / "bare.json"), sha])
