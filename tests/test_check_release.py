"""python -m hcd_analysis.check_release: a release is loaded and predicted from in a fresh interpreter, and compared
with reference predictions. Synthetic release (untrained networks): the check is exercised, no model is validated.
"""

import json
import os
import subprocess
import sys

import pytest

import hcd_analysis
from hcd_analysis import check_release as CR
from tests.synthetic_release import make_release

ENV = {k: v for k, v in os.environ.items() if not k.startswith("HCD_PRIYA_")}


def _run(*args, isolated=True, cwd=None, switch=False):
    """The module run in a fresh interpreter. In isolated mode (-I) the package must be installed; from a source tree
    (no installation) the tests run it with the tree on the path instead."""
    cmd = (
        [sys.executable, "-B"]
        + (["-I"] if isolated else [])
        + ["-m", "hcd_analysis.check_release", *args]
    )
    env = dict(ENV, HCD_PRIYA_ALLOW_SYNTHETIC="1") if switch else ENV
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("rel") / "release")


@pytest.fixture(scope="module")
def run(release):
    """How to run the module here. Isolated (-I) when the isolated interpreter imports the very package these tests
    import (an installation, as in the wheel tier); otherwise from the directory that holds the package (a source
    tree), so that a stale installation is never tested in its place."""
    here = os.path.dirname(os.path.abspath(hcd_analysis.__file__))
    code = "import hcd_analysis, os; print(os.path.dirname(os.path.abspath(hcd_analysis.__file__)))"
    probe = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, env=ENV
    )
    if probe.returncode == 0 and probe.stdout.strip() == here:
        return lambda *a, **kw: _run(*a, **kw)
    return lambda *a, **kw: _run(*a, isolated=False, cwd=os.path.dirname(here), **kw)


def test_a_release_loads_predicts_and_agrees_with_its_own_reference(release, run, tmp_path):
    ref = tmp_path / "reference_predictions.json"
    out = run(str(release["root"]), release["sha"], "--write-reference", str(ref))
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.rstrip().splitlines()[-1] == "PASSED" and "imported from" in out.stdout
    assert (
        "SYNTHETIC BUILD" not in out.stdout
    )  # a release directory given by path and digest is not that route
    doc = json.loads(ref.read_text())
    assert doc["format"] == CR.FORMAT and doc["release_sha256"] == release["sha"]
    assert sorted({c["fidelity"] for c in doc["cases"]}) == ["LF", "LF+MF"]
    assert {c["product"] for c in doc["cases"] if c["fidelity"] == "LF+MF"} == {"LF+MF"}
    again = run(str(release["root"]), release["sha"], "--reference", str(ref))
    assert again.returncode == 0 and f"{len(doc['cases'])} cases agree" in again.stdout
    refused = run(str(release["root"]), release["sha"], "--write-reference", str(ref))
    assert refused.returncode == 1 and "FAILED: the reference was not written" in refused.stdout
    assert "PASSED" not in refused.stdout and ref.read_text() == json.dumps(doc)  # not replaced


def test_a_changed_number_a_missing_case_and_another_release_fail(release, run, tmp_path):
    ref = tmp_path / "ref.json"
    assert run(str(release["root"]), release["sha"], "--write-reference", str(ref)).returncode == 0
    doc = json.loads(ref.read_text())
    cases = {
        "a number": lambda d: d["cases"][0]["P_class"][0].__setitem__(
            0, d["cases"][0]["P_class"][0][0] * 1.001
        ),
        "a case": lambda d: d["cases"].pop(),
        "a shape": lambda d: d["cases"][0].__setitem__("w_c", d["cases"][0]["w_c"][:2]),
        "the release": lambda d: d.__setitem__("release_sha256", "0" * 64),
        "the format": lambda d: d.__setitem__("format", "other/1"),
    }
    for what, change in cases.items():
        bad = json.loads(json.dumps(doc))
        change(bad)
        path = tmp_path / f"{what.replace(' ', '_')}.json"
        path.write_text(json.dumps(bad))
        out = run(str(release["root"]), release["sha"], "--reference", str(path))
        assert out.returncode == 1 and "FAILED" in out.stdout and "PASSED" not in out.stdout, what


def test_a_case_of_the_reference_that_is_not_predicted_fails(release, run, tmp_path):
    """The synthetic MF record starts at z = 2.6. A reference lists what was served when it was written; a later run
    that serves less (fewer redshifts, or no LF+MF where the reference has it) fails."""
    ref = tmp_path / "ref.json"
    args = (str(release["root"]), release["sha"])
    wrote = run(*args, "--z", "2.6", "3.0", "--write-reference", str(ref))
    assert wrote.returncode == 0, wrote.stdout + wrote.stderr
    assert len(json.loads(ref.read_text())["cases"]) == 4
    fewer = run(*args, "--z", "3.0", "--reference", str(ref))
    assert fewer.returncode == 1 and "in the reference, not predicted" in fewer.stdout
    assert "PASSED" not in fewer.stdout
    low = run(*args, "--z", "2.2")
    assert low.returncode == 0 and "not served: z = 2.2, LF+MF" in low.stdout
    assert "predicted: z = 2.2, LF:" in low.stdout and "predicted: z = 2.2, LF+MF" not in low.stdout
    # a reference that has LF+MF at a redshift where the model does not serve it
    doc = json.loads(ref.read_text())
    for c in doc["cases"]:
        c["z"] = 2.2 if c["z"] == 2.6 else c["z"]
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(doc))
    out = run(*args, "--z", "2.2", "3.0", "--reference", str(moved))
    assert out.returncode == 1
    assert "case z = 2.2, LF+MF: in the reference, not predicted" in out.stdout


def test_a_reference_that_cannot_be_read_fails_with_a_message(release, run, tmp_path):
    args = (str(release["root"]), release["sha"])
    missing = run(*args, "--reference", str(tmp_path / "none.json"))
    assert missing.returncode == 1 and "FAILED: the reference could not be read" in missing.stdout
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    broken = run(*args, "--reference", str(bad))
    assert broken.returncode == 1 and "FAILED: the reference could not be read" in broken.stdout
    bad.write_text("[1, 2]")
    wrong = run(*args, "--reference", str(bad))
    assert wrong.returncode == 1 and "not a JSON object" in wrong.stdout
    assert "Traceback" not in wrong.stderr


def test_a_wrong_digest_fails_and_says_why(release, run):
    out = run(str(release["root"]), "0" * 64)
    assert out.returncode == 1 and "FAILED: ReleaseManifestError" in out.stdout


def test_a_published_model_is_checked_from_the_installed_model_package(run):
    """--published loads the model of the installed hcd-priya-models. In tests that package is a synthetic build:
    it is refused without the test switch, and checked with it."""
    import hcd_priya_models

    assert hcd_priya_models.manifest()["synthetic"], "this test expects a synthetic model package"
    refused = run("--published", "lfmf")
    assert refused.returncode == 1 and "FAILED: ModelPackageError" in refused.stdout
    assert "synthetic test build" in refused.stdout and "PASSED" not in refused.stdout
    out = run("--published", "lfmf", switch=True)
    assert out.returncode == 0, out.stdout + out.stderr
    # the parameter point is stated with its values, and no name is needed for the LF+MF model
    assert "parameters: the midpoint of the training range of each parameter" in out.stdout
    assert "ns = 0.925, Ap = 1.9e-09" in out.stdout and "bhfeedback = 0.05" in out.stdout
    assert "example_params()" not in out.stdout  # the check prints its own line, once
    unnamed = run("--published", switch=True)
    assert unnamed.returncode == 0 and unnamed.stdout == out.stdout
    last = out.stdout.rstrip().splitlines()[-1]
    assert last.startswith("PASSED (synthetic build") and "SYNTHETIC BUILD" in out.stdout
    assert last != "PASSED"  # a synthetic build is never reported as the check of a model
    missing = run("--published", "no-such-model", switch=True)
    assert missing.returncode == 1 and "FAILED: NoPublishedModelError" in missing.stdout


def test_the_arguments_are_one_form_or_the_other(run):
    for args in ((), ("only_a_directory",), ("a_dir", "0" * 64, "--published", "lfmf")):
        assert run(*args).returncode == 2


def test_compare_reports_each_kind_of_difference():
    case = {
        "z": 3.0,
        "fidelity": "LF",
        "k": [1.0, 2.0],
        "P_class": [[1.0, 2.0]],
        "w_c": [0.1],
        "dndx": [0.2],
    }
    ref = {"format": CR.FORMAT, "cases": [json.loads(json.dumps(case))]}
    assert CR.compare([case], ref, 1e-6) == []
    close = dict(case, P_class=[[1.0 + 1e-9, 2.0]])
    assert CR.compare([close], ref, 1e-6) == []
    (value,) = CR.compare([dict(case, P_class=[[1.01, 2.0]])], ref, 1e-6)
    assert "P_class differs" in value and "allowed 1e-06" in value
    (shape,) = CR.compare([dict(case, w_c=[0.1, 0.2])], ref, 1e-6)
    assert "w_c has shape (2,), the reference (1,)" in shape
    (missing,) = CR.compare([], ref, 1e-6)
    assert "in the reference, not predicted" in missing
    (extra,) = CR.compare([case, dict(case, z=3.2)], ref, 1e-6)
    assert "case z = 3.2, LF: predicted, not in the reference" in extra
    twice = {"format": CR.FORMAT, "cases": [case, case]}
    assert CR.compare([case], twice, 1e-6) == [
        "the reference lists a case (z, fidelity) more than once"
    ]
    assert "format" in CR.compare([case], {"format": "other/1", "cases": [case]}, 1e-6)[0]
