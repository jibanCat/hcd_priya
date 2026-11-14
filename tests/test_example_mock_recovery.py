"""The example examples/mock_recovery.py: part one on a synthetic model and a synthetic mock, the settings it writes
out, and the input it hands to Cobaya (built here without Cobaya; the chain itself is not run in the test tiers).
Synthetic values only."""

import importlib.util
import logging
import math
import os
import re
import sys
import types

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
    labelled = re.compile(r"[a-zA-Z(][^:]{8,}: \S")  # a label of words, a colon, then a value
    assert all(labelled.match(ln) for ln in body), [ln for ln in body if not labelled.match(ln)]


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


class _FakeSample:
    """What the example reads from Cobaya's sample: columns by name, and the weights."""

    def __init__(self, columns):
        self._c = columns

    def __getitem__(self, name):
        return self._c[name]


def _pretend_cobaya_is_installed(ex, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        ex.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "cobaya" else real(name, *a),
    )


def test_the_chain_is_read_with_its_weights_after_burn_in_and_the_predictive_is_drawn(
    setup, capsys, monkeypatch, tmp_path
):
    """Parts two and three with a stand-in for Cobaya: ten made-up rows with unequal weights. Checked: the logger is
    quiet while the sampler runs and restored after, the burn-in, the weighted percentiles, the figure.
    """
    ex = _script()
    seen = {}
    rows = 10
    columns = {n: THETA[n] * (1.0 + 1e-3 * np.arange(rows)) for n in THETA if THETA[n] != 0}
    columns["dtau0"] = 1e-3 * np.arange(rows)
    columns["weight"] = np.array([1.0, 1, 1, 5, 1, 1, 1, 1, 1, 7])

    def run(info):
        seen["level"] = logging.getLogger("hcd_analysis.predictor").level
        seen["logp"] = info["likelihood"]["p1d"]["external"](**THETA)
        seen["info"] = info
        return None, types.SimpleNamespace(products=lambda: {"sample": _FakeSample(columns)})

    fake = types.ModuleType("cobaya")
    fake.run = types.ModuleType("cobaya.run")
    fake.run.run = run
    monkeypatch.setitem(sys.modules, "cobaya", fake)
    monkeypatch.setitem(sys.modules, "cobaya.run", fake.run)
    _pretend_cobaya_is_installed(ex, monkeypatch)
    before = logging.getLogger("hcd_analysis.predictor").level
    out_dir = tmp_path / "out"
    argv = [
        "--release",
        *setup["release"],
        "--data",
        *setup["data"],
        "--demo",
        "10",
        "--draws",
        "4",
    ]
    assert ex.main(argv + ["--out", str(out_dir), "--seed", "5"]) == 0
    out = capsys.readouterr().out
    assert seen["level"] == logging.WARNING
    assert logging.getLogger("hcd_analysis.predictor").level == before
    assert np.isfinite(seen["logp"]) and seen["info"]["sampler"]["mcmc"]["max_samples"] == 10
    assert "=== Part 2: A DEMONSTRATION CHAIN, NOT CONVERGED: its numbers are not results" in out
    assert "the chain starts at: the truth of the mock" in out
    assert "accepted steps in the chain: 10" in out
    assert "rows kept after dropping the first 30% as burn-in: 7" in out
    # the weighted percentiles of the seven kept rows of one parameter, computed here by hand
    v, w = columns["ns"][3:], columns["weight"][3:]
    cdf = np.cumsum(w) / w.sum()
    want = [v[np.searchsorted(cdf, q)] for q in (0.16, 0.5, 0.84)]
    (got,) = [ln for ln in out.splitlines() if ln.startswith("ns: 16th, 50th, 84th percentile")]
    assert got == (
        f"ns: 16th, 50th, 84th percentile of the chain | truth: {want[0]} {want[1]} {want[2]} | {THETA['ns']}"
    )
    assert want[0] == v[0] and want[1] != np.median(
        v
    )  # the weights matter: the heavy rows pull the median
    assert "points of the chain used: 4" in out
    assert (out_dir / "posterior_predictive.png").stat().st_size > 1000


def test_the_logger_is_quiet_only_inside_the_block_and_a_chain_needs_the_truth_inside_the_boxes(
    setup, capsys, monkeypatch, tmp_path
):
    ex = _script()
    logger = logging.getLogger("hcd_analysis.predictor")
    before = logger.level
    with pytest.raises(RuntimeError):
        with ex.quiet_predictor_log():
            assert logger.level == logging.WARNING
            raise RuntimeError("the level is put back also when the sampler fails")
    assert logger.level == before
    # a truth outside a flat box: part one still runs and says so; a chain is refused, since it would start there
    d = L.read_p1d_data(setup["data"][0], expected_sha256=setup["data"][1])
    outside = dict(d.truth, hub=0.749)
    moved = L.P1DData(z=d.z, k=d.k, p1d=d.p1d, cov=d.cov, truth=outside)
    sha = L.write_p1d_data(tmp_path / "outside.json", moved)
    _pretend_cobaya_is_installed(ex, monkeypatch)
    monkeypatch.setitem(
        ex.BOXES, "hub", (0.65, 0.74)
    )  # a narrower box, so that this truth lies outside it
    with pytest.raises(SystemExit, match="the truth lies outside the flat box of \\['hub'\\]"):
        ex.main(
            [
                "--release",
                *setup["release"],
                "--data",
                str(tmp_path / "outside.json"),
                sha,
                "--demo",
                "5",
            ]
        )
    assert (
        "parameters of the truth outside this example's flat boxes: ['hub']"
        in capsys.readouterr().out
    )
    with pytest.raises(SystemExit) as e:
        ex.main(["--release", *setup["release"], "--data", *setup["data"], "--draws", "0"])
    assert e.value.code == 2 and "--draws must be at least 1" in capsys.readouterr().err
