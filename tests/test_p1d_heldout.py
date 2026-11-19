"""The held-out prediction residual of the class P1D (hcd_analysis/emulator/p1d_heldout.py): the reduction is the
stated one; the record round-trips, is strict, and is recomputed from its own evidence file; a release names it as an
optional part; and a prediction carries it for its own fidelity, as a number on the logarithm and as a range on the
spectrum. Synthetic values only: round numbers that no simulation gives."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import p1d_heldout as PH
from hcd_analysis.emulator import release_manifest as RM
from hcd_analysis.emulator import safe_io as S
from tests.synthetic_release import make_release, p1d_residuals, write_p1d_heldout
from tests.test_predictor_mf import PARAMS

E = PH.MFRecordError
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALL = dict(tau0_amp=1.0, dtau0=0.0)


@pytest.fixture(scope="module")
def rel(tmp_path_factory):
    """A synthetic release with the three optional records."""
    return make_release(
        tmp_path_factory.mktemp("p1d") / "release", incidence=True, heldout=True, p1d_heldout=True
    )


@pytest.fixture(scope="module")
def emu(rel):
    return hp.load_release(str(rel["root"]), rel["sha"])


@pytest.fixture(scope="module")
def held(rel):
    return PH.load_record(
        str(rel["root"] / "p1d_heldout"), expected_record_sha256=rel["p1d_heldout_sha"]
    )


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ----------------------------------------------------------------------------------------------- the reduction
def test_the_reduction_is_the_rms_about_zero_over_simulations_and_rungs():
    """Two simulations, one redshift, two rungs, one mode: d = +0.1, +0.1 and -0.3, +0.1 in every row."""
    d = np.empty((2, 1, 2, 5, 1), dtype=np.float32)
    d[0, 0, :, :, 0] = np.float32(0.1)
    d[1, 0, 0, :, 0], d[1, 0, 1, :, 0] = np.float32(-0.3), np.float32(0.1)
    red = PH.reduce_residuals(d)
    x = d.astype(np.float64)[:, 0, :, 0, 0].ravel()  # the four stored values of one row
    np.testing.assert_allclose(red["e"], np.sqrt(np.mean(x**2)), rtol=1e-14, atol=0)
    np.testing.assert_allclose(red["b"], np.mean(x), rtol=1e-13, atol=0)
    assert red["e"].shape == (1, 5, 1) and red["n"].tolist() == [[2] * 5]
    # about zero, so a bias counts: e is larger than the scatter about the mean
    assert float(red["e"][0, 0, 0]) > float(np.std(x))
    # per rung: over the simulations only
    np.testing.assert_allclose(
        red["e_rung"][0, :, 0, 0],
        [np.sqrt((0.1**2 + 0.3**2) / 2), 0.1],
        rtol=1e-6,
        atol=0,
    )
    assert red["e_rung"].dtype == np.float32


def test_a_simulation_without_a_value_is_left_out_and_not_counted():
    d = p1d_residuals(
        3, [2.0, 3.0], 4, n_rung=2, scale=0.01
    )  # simulation 0 has no value at the first redshift
    red = PH.reduce_residuals(d)
    assert red["n"][:, 0].tolist() == [2, 3]
    np.testing.assert_allclose(
        red["e"][0, 0], 0.01, rtol=1e-6, atol=0
    )  # |d| is 0.01 in row 0 at redshift 0
    np.testing.assert_allclose(
        red["e"][1, 4], 0.01 * 5 * 2, rtol=1e-6, atol=0
    )  # row 4 at redshift 1


def test_a_cell_without_any_value_is_nan_and_such_a_record_is_refused(tmp_path):
    d = p1d_residuals(2, [2.0, 3.0], 4)
    d[:, 0] = np.nan
    assert np.isnan(PH.reduce_residuals(d)["e"][0]).all()
    with pytest.raises(E, match="finite"):
        write_p1d_heldout(
            tmp_path / "rec",
            cache="ab" * 32,
            hr_cache="cd" * 32,
            hr_sims=["synthsim1", "synthsim2"],
            n_modes=4,
            residuals={
                "lf": {
                    "sims": ["synthsim1", "synthsim2"],
                    "z": [2.0, 3.0],
                    "d": d,
                    "target_F": np.full((2, 2, 3), 0.5),
                    "counts": np.full((2, 2, 3, 4), 10, dtype=np.int64),
                },
                "lfmf": {
                    "sims": ["synthsim1", "synthsim2"],
                    "z": [3.0],
                    "d": p1d_residuals(2, [3.0], 4),
                    "target_F": np.full((2, 1, 3), 0.5),
                    "counts": np.full((2, 1, 3, 4), 10, dtype=np.int64),
                },
            },
            lf_only_sims=(),
        )
    assert not (tmp_path / "rec").exists()  # nothing was written


def test_the_summary_keeps_the_regimes_apart():
    z = np.array([2.0, 2.2, 4.6, 4.8])
    e = np.zeros((4, 5, 6))
    e[0] = 0.5  # below z 2.2
    e[1:3, :, :2] = 0.2  # the largest modes, z 2.2 to 4.6
    e[1:3, :, 2:] = 0.01  # typical
    e[3, :, :2] = 0.4  # above z 4.6, the largest modes
    e[3, :, 2:] = 0.03
    s = PH.summarise(e, z)
    assert s == {
        "typical": {"median": 0.01, "max": 0.01},
        "largest_modes": {"median": 0.2, "max": 0.2},
        "above_z_4p6": {"median": 0.03, "max": 0.03},
        "above_z_4p6_largest_modes": {"median": 0.4, "max": 0.4},
        "below_z_2p2": {"median": 0.5, "max": 0.5},
    }
    assert "below_z_2p2" not in PH.summarise(e[1:], z[1:])
    assert PH.N_LARGEST_MODES == 2 and PH.Z_TYPICAL == (2.2, 4.6)


# ----------------------------------------------------------------------------------------------- the record
def test_a_record_round_trips_and_its_tables_are_what_its_residuals_give(rel, held):
    d = rel["root"] / "p1d_heldout"
    assert sorted(os.listdir(d)) == [PH.RECORD_FILE, PH.TABLES_FILE]
    r = held.record
    assert (
        r["format"] == PH.RECORD_FORMAT
        and r["kind"] == PH.KIND
        and r["definition"] == PH.DEFINITION
    )
    assert r["class_order"] == ["clean", "LLS", "subDLA", "DLA"] and PH.ROWS[4] == "mixture"
    assert r["lf"]["z"] == [2.0, 3.0, 5.0] and r["lfmf"]["z"] == [3.0, 5.0]
    assert (
        r["lf"]["sims"] == sorted(r["lf"]["sims"])
        and len(r["lf"]["sims"]) == len(r["lfmf"]["sims"]) + 1
    )
    K = r["n_modes"]
    assert held.e["lf"].shape == (3, 5, K) and held.e["lfmf"].shape == (2, 5, K)
    # the synthetic residuals: |d| = scale * (row + 1) * (1 + index of the redshift)
    np.testing.assert_allclose(held.e["lf"][1, :, 0], 0.01 * np.arange(1, 6) * 2, rtol=1e-6, atol=0)
    np.testing.assert_allclose(held.e["lfmf"][0, :, 0], 0.03 * np.arange(1, 6), rtol=1e-6, atol=0)
    assert (
        held.n["lf"][0].tolist() == [len(r["lf"]["sims"]) - 1] * 5
    )  # one simulation has no first redshift
    assert not held.e["lf"].flags.writeable and held.anchored is True
    n = PH.verify_tables(str(d), held)
    arrays, attrs = S.read_table_h5(str(d / PH.TABLES_FILE), PH.table_datasets(r))
    assert n == int(np.isfinite(arrays["lf_d"]).sum() + np.isfinite(arrays["lfmf_d"]).sum())
    assert arrays["lf_d"].dtype == np.float32 and attrs["record_format"] == PH.RECORD_FORMAT
    assert arrays["lf_counts"].shape == arrays["lf_d"].shape[:3] + (4,)
    PH.reverify(held)


def test_the_sensitivity_table_is_the_lf_table_without_one_simulation(rel, held):
    r = held.record
    s = r["sensitivity"]
    assert s["fidelity"] == "lf" and s["without_sim"] == r["lf"]["sims"][0]
    arrays, _ = S.read_table_h5(
        str(rel["root"] / "p1d_heldout" / PH.TABLES_FILE), PH.table_datasets(r)
    )
    want = PH.reduce_residuals(
        arrays["lf_d"][1:]
    )  # the simulations are stored in the sorted order of their names
    np.testing.assert_allclose(held.e_sens, want["e"], rtol=1e-13, atol=0)
    assert np.array_equal(held.n_sens, want["n"])
    assert r["summary"]["sensitivity"] == PH.summarise(held.e_sens, np.array(s["z"]))
    assert "SENSITIVITY TEST, not the table of a prediction" in held.label(False)
    assert "SENSITIVITY TEST" not in held.label(True)  # it is of the LF table


@pytest.mark.parametrize(
    "what, change, why",
    [
        (
            "a changed table",
            lambda r: r["lf"]["e"][1][0].__setitem__(0, 0.5),
            "summary.lf is not the reduction",
        ),
        (
            "another definition",
            lambda r: r.__setitem__("definition", "another statement"),
            "definition is not",
        ),
        ("another kind", lambda r: r.__setitem__("kind", "sigma"), "kind is"),
        (
            "a changed summary",
            lambda r: r["summary"]["lfmf"]["typical"].__setitem__("max", 9.0),
            "summary.lfmf",
        ),
        ("an unknown key", lambda r: r.__setitem__("extra", 1), "keys must be exactly"),
        (
            "a negative value",
            lambda r: r["lf"]["e"][0][0].__setitem__(0, -0.1),
            "finite numbers >= 0",
        ),
        ("a count of zero", lambda r: r["lf"]["n"][0].__setitem__(0, 0), "integers from 1"),
        (
            "a count above the simulations",
            lambda r: r["lfmf"]["n"][0].__setitem__(0, 99),
            "integers from 1",
        ),
        ("redshifts out of order", lambda r: r["lf"]["z"].reverse(), "strictly ascending"),
        ("a dirty tree", lambda r: r["code"].__setitem__("dirty", True), "clean tree"),
        (
            "a path as a file name",
            lambda r: r["inputs"]["lf_cache"].__setitem__("file", "/x/y.h5"),
            "plain file name",
        ),
        (
            "an empty note",
            lambda r: r["notes"].__setitem__("anomalous_rows", " "),
            "non-empty text",
        ),
        (
            "a sensitivity without a known simulation",
            lambda r: r["sensitivity"].__setitem__("without_sim", "x"),
            "one of the held-out",
        ),
        (
            "simulations that are not the folds'",
            lambda r: r["lf"]["sims"].__setitem__(0, "another"),
            "sorted|held-out simulations",
        ),
    ],
)
def test_a_record_that_says_something_else_is_refused(held, what, change, why):
    r = copy.deepcopy(held.record)
    change(r)
    with pytest.raises(E, match=why):
        PH.check_record(r)


def test_the_record_is_read_against_its_digest_and_the_directory_holds_nothing_else(rel, tmp_path):
    src = rel["root"] / "p1d_heldout"
    sha = rel["p1d_heldout_sha"]
    with pytest.raises(E, match="published sha256"):
        PH.load_record(str(src), expected_record_sha256=None)
    with pytest.raises(E, match="does not match the published record sha256"):
        PH.load_record(str(src), expected_record_sha256="0" * 64)
    d = tmp_path / "copy"
    shutil.copytree(src, d)
    (d / "notes.txt").write_text("x")
    with pytest.raises(E, match="unexpected files"):
        PH.load_record(str(d), expected_record_sha256=sha)
    os.remove(d / "notes.txt")
    with open(d / PH.TABLES_FILE, "r+b") as fh:  # one byte of the evidence file
        fh.seek(-1, os.SEEK_END)
        b = fh.read(1)
        fh.seek(-1, os.SEEK_END)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(E, match="does not have the sha256 the record states"):
        PH.load_record(str(d), expected_record_sha256=sha)
    os.remove(d / PH.TABLES_FILE)
    with pytest.raises(E, match="evidence file"):
        PH.load_record(str(d), expected_record_sha256=sha)


def test_an_evidence_file_that_does_not_give_the_tables_is_refused_by_the_recomputation(
    rel, tmp_path
):
    """A record whose evidence file was replaced by other residuals, with the record's digests brought along: a load
    passes (the digests agree), the recomputation does not."""
    src = rel["root"] / "p1d_heldout"
    d = tmp_path / "forged"
    os.makedirs(d)
    rec = json.loads((src / PH.RECORD_FILE).read_text())
    arrays, _ = S.read_table_h5(str(src / PH.TABLES_FILE), PH.table_datasets(rec))
    arrays = {k: np.array(v) for k, v in arrays.items()}
    assert np.isfinite(
        arrays["lfmf_d"][-1, -1, -1, 0, 0]
    )  # a value that exists (the first cell is NaN)
    arrays["lfmf_d"][-1, -1, -1, 0, 0] += np.float32(0.25)
    S.write_table_h5(str(d / PH.TABLES_FILE), arrays, {"record_format": PH.RECORD_FORMAT})
    rec["tables"]["sha256"] = _sha(d / PH.TABLES_FILE)
    rec["tables"]["bytes"] = os.path.getsize(d / PH.TABLES_FILE)
    raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode()
    (d / PH.RECORD_FILE).write_bytes(raw)
    forged = PH.load_record(str(d), expected_record_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(E, match="lfmf.e is not what the residuals give"):
        PH.verify_tables(str(d), forged)


def test_a_record_changed_in_memory_is_refused(held):
    e = {k: np.array(v) for k, v in held.e.items()}
    e["lf"][0, 0, 0] += 1.0
    with pytest.raises(E, match="changed in memory"):
        PH.reverify(held._replace(e=e))
    with pytest.raises(E, match="loaded by p1d_heldout.load_record"):
        PH.reverify(held._replace(anchored=False))
    with pytest.raises(E, match="loaded by p1d_heldout.load_record"):
        PH.reverify({"record": held.record})


def test_a_record_is_written_once(rel, tmp_path):
    with pytest.raises(FileExistsError):
        write_p1d_heldout(
            rel["root"] / "p1d_heldout",
            cache="ab" * 32,
            hr_cache="cd" * 32,
            hr_sims=["a"],
            n_modes=4,
        )


def test_a_record_may_be_shipped_without_its_evidence_file(rel, held, tmp_path):
    """Then its ``tables`` key is null, the directory holds the record alone, the tables a prediction uses are the
    same, and nothing can be recomputed: ``verify_tables`` refuses, and the label says the vectors are not shipped.
    """
    mf = json.loads((rel["root"] / "mf" / "mf_record.json").read_text())
    kw = dict(
        cache=rel["bundle"].lineage["cache"]["sha256"],
        hr_cache=mf["caches"]["hr"]["sha256"],
        hr_sims=mf["train_sims"],
        n_modes=mf["tables"]["n_modes"],
    )
    sha = write_p1d_heldout(tmp_path / "alone", with_evidence=False, **kw)
    assert os.listdir(tmp_path / "alone") == [PH.RECORD_FILE]
    alone = PH.load_record(str(tmp_path / "alone"), expected_record_sha256=sha)
    assert alone.record["tables"] is None and alone.provenance()["tables"] is None
    for fid in PH.FIDELITIES:
        assert np.array_equal(alone.e[fid], held.e[fid]) and np.array_equal(
            alone.n[fid], held.n[fid]
        )
    assert np.array_equal(alone.e_sens, held.e_sens)
    with pytest.raises(E, match="shipped without its evidence file"):
        PH.verify_tables(str(tmp_path / "alone"), alone)
    assert "not shipped with this record" in alone.label(True)
    assert "not shipped with this record" not in held.label(True)
    PH.reverify(alone)
    # an evidence file beside such a record is a stray file
    shutil.copyfile(
        rel["root"] / "p1d_heldout" / PH.TABLES_FILE, tmp_path / "alone" / PH.TABLES_FILE
    )
    with pytest.raises(E, match="unexpected files"):
        PH.load_record(str(tmp_path / "alone"), expected_record_sha256=sha)


def test_check_release_says_when_the_evidence_file_is_not_shipped(tmp_path):
    from hcd_analysis import check_release as CR
    import contextlib
    import io

    root, anchor = _release_with(tmp_path, with_evidence=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = CR.main([str(root), anchor])
    text = out.getvalue()
    assert code == 0, text
    assert "shipped without its evidence file; its tables were not recomputed" in text
    pred = hp.load_release(str(root), anchor).predict(PARAMS, z=3.0, **CALL)
    assert (
        pred.flags["p1d_validation_available"] is True
        and pred.P_class_validation_rms_log.shape[0] == 4
    )


# ----------------------------------------------------------------------------------------------- the words
def test_the_statements_use_the_agreed_term_and_keep_three_cases_apart(held):
    """The term is "held-out prediction residual"; it is called an empirical validation diagnostic and never a
    generalisation error or a calibrated uncertainty; the label gives the typical case, the largest modes and the
    redshifts above 4.6 as three separate statements."""
    for lfmf in (False, True):
        text = held.label(lfmf)
        assert "held-out prediction residual" in text
        assert "empirical validation diagnostic, not a calibrated predictive uncertainty" in text
        low = text.lower()
        for word in (
            "generalisation error",
            "generalization error",
            "sigma",
            "gaussian",
            "symmetric",
            "±",
        ):
            assert word not in low, word
        a, b, c = (
            text.index("Typical (z 2.2 to 4.6, above the 2 largest modes"),
            text.index("The 2 largest modes"),
            text.index("Above z 4.6"),
        )
        assert a < b < c
        assert "for other fractions it is an indication only" in text
        assert held.record["what_it_does_not_cover"] in text
        for note in held.record["notes"].values():
            assert note in text
    assert (
        "generalisation" not in PH.DEFINITION.lower()
        and "v * exp(-e) to v * exp(+e)" in PH.DEFINITION
    )
    assert "LF+MF" in held.label(True) and "(LF:" in held.label(False)


# ----------------------------------------------------------------------------------------------- the manifest
def test_a_release_names_the_record_as_an_optional_part(rel, tmp_path):
    m = json.loads((rel["root"] / "release.json").read_text())
    assert m["p1d_heldout"] == {"dir": "p1d_heldout", "record_sha256": rel["p1d_heldout_sha"]}
    assert "p1d_heldout" in RM.OPTIONAL_PARTS and "p1d_heldout" not in RM.KEYS
    _, parts = RM.read_release_manifest(str(rel["root"]), rel["sha"])
    assert os.path.basename(parts["p1d_heldout"]) == "p1d_heldout"
    with pytest.raises(
        RM.ReleaseManifestError, match="both p1d_heldout_dir and p1d_heldout_record_sha256"
    ):
        RM.write_release_manifest(
            str(tmp_path),
            release_version="v",
            backbone_dir="backbone",
            backbone_manifest_sha256="a" * 64,
            mf_dir="mf",
            mf_record_sha256="b" * 64,
            p1d_heldout_dir="p1d_heldout",
        )
    bad = dict(m, p1d_heldout={"dir": "../x", "record_sha256": "c" * 64})
    with pytest.raises(RM.ReleaseManifestError, match="plain path segment"):
        RM.check_release_manifest(bad)


def test_a_release_without_the_part_is_what_it_was(tmp_path):
    """No new key in the manifest, no new field on a prediction, no new flag."""
    plain = make_release(tmp_path / "plain", incidence=True, heldout=True)
    assert "p1d_heldout" not in json.loads((plain["root"] / "release.json").read_text())
    pred = hp.load_release(str(plain["root"]), plain["sha"]).predict(PARAMS, z=3.0, **CALL)
    for name in (
        "P_class_validation_rms_log",
        "P_mix_validation_rms_log",
        "P_class_validation_lower",
        "P_class_validation_upper",
    ):
        assert getattr(pred, name) is None
    assert "p1d_validation_available" not in pred.flags and "p1d_validation" not in pred.labels
    assert "p1d_heldout" not in pred.provenance


# ----------------------------------------------------------------------------------------------- a prediction
def test_a_prediction_carries_the_residual_of_its_own_fidelity(emu, held):
    mf = emu.predict(PARAMS, z=3.0, **CALL)
    lf = emu.predict(PARAMS, z=3.0, fidelity="LF", **CALL)
    K = mf.k.size
    assert mf.P_class_validation_rms_log.shape == (4, K) and mf.P_mix_validation_rms_log.shape == (
        K,
    )
    assert np.array_equal(mf.P_class_validation_rms_log, held.err(3.0, True)[:4])
    assert np.array_equal(mf.P_mix_validation_rms_log, held.err(3.0, True)[4])
    assert np.array_equal(lf.P_class_validation_rms_log, held.err(3.0, False)[:4])
    assert not np.array_equal(lf.P_class_validation_rms_log, mf.P_class_validation_rms_log)
    np.testing.assert_allclose(
        mf.P_class_validation_rms_log[:, 0], 0.03 * np.arange(1, 5), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        lf.P_class_validation_rms_log[:, 0], 0.01 * np.arange(1, 5) * 2, rtol=1e-6, atol=0
    )
    for pred in (mf, lf):
        assert pred.flags["p1d_validation_available"] is True
        assert (
            not pred.P_class_validation_rms_log.flags.writeable
            and not pred.P_class_validation_lower.flags.writeable
        )
    assert mf.labels["p1d_validation"] == held.label(True)
    assert lf.labels["p1d_validation"] == held.label(False)
    assert mf.provenance["p1d_heldout"]["record_sha256"] == held.record_sha256


def test_the_range_is_on_the_spectrum_each_end_from_the_logarithm_on_its_own(emu):
    pred = emu.predict(PARAMS, z=3.0, **CALL)
    np.testing.assert_allclose(
        pred.P_class_validation_lower,
        pred.P_class * np.exp(-pred.P_class_validation_rms_log),
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        pred.P_class_validation_upper,
        pred.P_class * np.exp(+pred.P_class_validation_rms_log),
        rtol=1e-14,
        atol=0,
    )
    assert np.all(pred.P_class_validation_lower < pred.P_class) and np.all(
        pred.P_class < pred.P_class_validation_upper
    )
    # the upper end lies farther from the value than the lower end
    assert np.all(
        pred.P_class_validation_upper - pred.P_class > pred.P_class - pred.P_class_validation_lower
    )


def test_a_redshift_without_an_estimate_gives_nan_and_no_flag(emu):
    pred = emu.predict(PARAMS, z=4.0, **CALL)  # the synthetic record has 3.0 and 5.0 for LF+MF
    assert (
        np.isnan(pred.P_class_validation_rms_log).all()
        and np.isnan(pred.P_mix_validation_rms_log).all()
    )
    assert (
        np.isnan(pred.P_class_validation_lower).all()
        and np.isnan(pred.P_class_validation_upper).all()
    )
    assert pred.flags["p1d_validation_available"] is False
    assert np.all(np.isfinite(pred.P_class))  # the spectrum itself is unaffected


def test_a_stack_and_a_requested_k_carry_it_too(emu, held):
    one = emu.predict(PARAMS, z=3.0, **CALL)
    stack = emu.predict_many(PARAMS, z=[3.0, 4.0, 5.0], **CALL)
    assert stack.P_class_validation_rms_log.shape == (
        3,
        4,
        one.k.size,
    ) and stack.P_mix_validation_rms_log.shape == (
        3,
        one.k.size,
    )
    assert np.array_equal(stack.P_class_validation_rms_log[0], one.P_class_validation_rms_log)
    assert (
        np.isnan(stack.P_class_validation_rms_log[1]).all()
        and np.isfinite(stack.P_class_validation_rms_log[2]).all()
    )
    assert (
        stack.flags["p1d_validation_available"] is False
    )  # one redshift of the stack has no estimate
    k = np.array([one.k[1] * 1.3, one.k[3] * 1.1])
    at = emu.predict(PARAMS, z=3.0, k=k, **CALL)
    assert at.P_class_validation_rms_log.shape == (4, 2) and at.P_mix_validation_rms_log.shape == (
        2,
    )
    # the synthetic residual is the same at every mode, so it is the same at any requested wavenumber
    np.testing.assert_allclose(
        at.P_class_validation_rms_log, one.P_class_validation_rms_log[:, :2], rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        at.P_class_validation_upper,
        at.P_class * np.exp(at.P_class_validation_rms_log),
        rtol=1e-14,
        atol=0,
    )


def test_the_spectra_and_the_other_fields_do_not_depend_on_the_record(rel, emu, tmp_path):
    """The record adds fields; it changes no number of a prediction."""
    plain = make_release(tmp_path / "plain", incidence=True, heldout=True)
    other = hp.load_release(str(plain["root"]), plain["sha"])
    a, b = emu.predict(PARAMS, z=3.0, **CALL), other.predict(PARAMS, z=3.0, **CALL)
    for name in ("k", "P_class", "template", "w_c", "dndx", "cddf", "dndx_err", "member_std"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name


# ----------------------------------------------------------------------------------------------- whose record
def _release_with(tmp_path, **over):
    """A synthetic release whose P1D held-out record is written with other arguments."""
    plain = make_release(tmp_path / "base", incidence=True, heldout=True)
    root = plain["root"]
    mf = json.loads((root / "mf" / "mf_record.json").read_text())
    kw = dict(
        cache=plain["bundle"].lineage["cache"]["sha256"],
        hr_cache=mf["caches"]["hr"]["sha256"],
        hr_sims=mf["train_sims"],
        n_modes=mf["tables"]["n_modes"],
    )
    kw.update(over)
    sha = write_p1d_heldout(root / "p1d_heldout", **kw)
    old = json.loads((root / "release.json").read_text())
    os.remove(root / "release.json")
    _, anchor = RM.write_release_manifest(
        str(root),
        release_version=old["release_version"],
        backbone_dir="backbone",
        backbone_manifest_sha256=old["backbone"]["manifest_sha256"],
        mf_dir="mf",
        mf_record_sha256=old["mf"]["record_sha256"],
        incidence_dir="incidence",
        incidence_record_sha256=old["incidence"]["record_sha256"],
        p1d_heldout_dir="p1d_heldout",
        p1d_heldout_record_sha256=sha,
    )
    return root, anchor


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(cache="ee" * 32), "not on this bundle's training cache"),
        (dict(hr_cache="ee" * 32), "another HR cache than the MF record's"),
        (dict(n_modes=5), "modes, this model has"),
        (dict(hr_sims=["synthsim1", "another_run"]), "not of this composition"),
    ],
)
def test_a_record_of_another_model_is_refused_when_the_release_is_loaded(tmp_path, over, why):
    root, anchor = _release_with(tmp_path, **over)
    with pytest.raises(E, match=why):
        hp.load_release(str(root), anchor)


def test_a_research_model_takes_no_such_record(held, tmp_path):
    from hcd_analysis.emulator import release_testing as RT

    b = RT.write_synthetic_research_bundle(str(tmp_path / "fold"))
    fold = hp.load_research_bundle(b.bundle_dir, b.manifest_sha256, purpose="a test of the refusal")
    with pytest.raises(E, match="a research model carries no P1D held-out record"):
        fold._attach_p1d_heldout(held)
    pred = fold.predict(PARAMS, z=3.0, **CALL)
    assert pred.P_class_validation_rms_log is None and "p1d_validation_available" not in pred.flags


# ----------------------------------------------------------------------------------------------- the tools
def test_check_release_recomputes_the_tables_and_fails_on_a_forged_evidence_file(rel, tmp_path):
    from hcd_analysis import check_release as CR

    def run(root, sha):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = CR.main([str(root), sha])
        return code, out.getvalue()

    code, text = run(rel["root"], rel["sha"])
    assert code == 0, text
    assert "its tables recomputed from" in text and "values of its evidence file: equal" in text
    # the same release with an evidence file that gives other tables, every digest brought along
    root = tmp_path / "forged"
    shutil.copytree(rel["root"], root)
    d = root / "p1d_heldout"
    rec = json.loads((d / PH.RECORD_FILE).read_text())
    arrays, _ = S.read_table_h5(str(d / PH.TABLES_FILE), PH.table_datasets(rec))
    arrays = {k: np.array(v) for k, v in arrays.items()}
    arrays["lf_d"][-1, -1, -1, 0, 0] += np.float32(0.5)
    os.remove(d / PH.TABLES_FILE)
    S.write_table_h5(str(d / PH.TABLES_FILE), arrays, {"record_format": PH.RECORD_FORMAT})
    rec["tables"] = {
        "file": PH.TABLES_FILE,
        "sha256": _sha(d / PH.TABLES_FILE),
        "bytes": os.path.getsize(d / PH.TABLES_FILE),
    }
    raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode()
    os.remove(d / PH.RECORD_FILE)
    (d / PH.RECORD_FILE).write_bytes(raw)
    m = json.loads((root / "release.json").read_text())
    m["p1d_heldout"]["record_sha256"] = hashlib.sha256(raw).hexdigest()
    os.remove(root / "release.json")
    man = json.dumps(m, indent=1, sort_keys=True, allow_nan=False).encode()
    (root / "release.json").write_bytes(man)
    code, text = run(root, hashlib.sha256(man).hexdigest())
    assert code == 1 and "FAILED" in text and "is not what the residuals give" in text


def test_the_builder_makes_the_record_from_pinned_inputs_and_refuses_another_file(
    rel, held, tmp_path
):
    """scripts/build_p1d_heldout.py: a residual file and a spec, each pinned by its sha256."""
    h5py = pytest.importorskip("h5py")
    script = os.path.join(REPO, "scripts", "build_p1d_heldout.py")
    if not os.path.isfile(script):
        pytest.skip("the builder script is not part of an installation: run from a clone")
    r = held.record
    arrays, _ = S.read_table_h5(
        str(rel["root"] / "p1d_heldout" / PH.TABLES_FILE), PH.table_datasets(r)
    )
    res = tmp_path / "residuals.h5"
    with h5py.File(res, "w") as f:
        for fid in PH.FIDELITIES:
            g = f.create_group(fid)
            g.create_dataset("d", data=arrays[f"{fid}_d"])
            g.create_dataset("z", data=np.array(r[fid]["z"]))
            g.create_dataset("sims", data=np.array(r[fid]["sims"], dtype="S"))
            g.create_dataset("target_F", data=arrays[f"{fid}_target_F"])
            g.create_dataset("counts", data=arrays[f"{fid}_counts"])
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "inputs": {k: r["inputs"][k] for k in ("lf_cache", "hr_cache", "folds")},
                "code": r["code"],
                "sensitivity": {k: r["sensitivity"][k] for k in ("without_sim", "statement")},
                "what_it_does_not_cover": r["what_it_does_not_cover"],
                "notes": r["notes"],
            }
        )
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base = [sys.executable, "-B", script, "--version", r["version"]]
    pins = [
        "--residuals",
        str(res),
        "--residuals-sha256",
        _sha(res),
        "--spec",
        str(spec),
        "--spec-sha256",
        _sha(spec),
    ]
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "built")] + pins,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "tables recomputed from" in out.stdout and ": equal" in out.stdout
    built = json.loads((tmp_path / "built" / PH.RECORD_FILE).read_text())
    for key in ("lf", "lfmf", "sensitivity", "summary", "notes", "what_it_does_not_cover", "code"):
        assert built[key] == r[key], key
    assert built["inputs"]["residuals"] == {"file": "residuals.h5", "sha256": _sha(res)}
    wrong = list(pins)
    wrong[3] = "0" * 64
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "refused")] + wrong,
        capture_output=True,
        text=True,
        env=env,
    )
    assert (
        out.returncode != 0
        and "not the pinned" in out.stderr
        and not (tmp_path / "refused").exists()
    )
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "built")] + pins,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode != 0 and "exists" in out.stderr
