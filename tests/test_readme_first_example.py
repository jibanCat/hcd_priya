"""The first example of the README: it writes the nine parameters out, it prints labelled lines whose output the
README shows, and it runs as written (here on a stand-in model package with a synthetic model)."""

import importlib.util
import os
import shutil

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import published_models as PM
from tests.synthetic_release import make_release, model_package, use_model_package

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "readme_first_example", os.path.join(REPO, "tools", "ci", "readme_first_example.py")
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
README = os.path.join(REPO, "README.md")


def test_the_example_writes_the_nine_parameters_out():
    code, shown = R.blocks(open(README, encoding="utf-8").read())
    params = R.written_out_params(code)
    assert params is not None and tuple(params) == tuple(hp.PARAM_NAMES)
    assert (
        "example_params" not in code and "PARAM_LIMITS" not in code
    )  # no value is computed or hidden
    # the README says they are the midpoint of each training range: they are
    np.testing.assert_allclose(
        hp.params_vector(params), np.mean(np.asarray(hp.PARAM_LIMITS), axis=1), rtol=1e-12
    )
    assert "not a fiducial cosmology" in code


def test_every_print_has_a_label_and_the_readme_shows_its_output():
    code, shown = R.blocks(open(README, encoding="utf-8").read())
    prints = [ln for ln in code.splitlines() if ln.startswith("print(")]
    want = shown.splitlines()
    assert len(prints) == len(want) >= 4
    for ln, out in zip(prints, want):
        label = ln[len('print("') :].split('"')[0]
        assert ln.startswith('print("') and label.endswith(":") and out.startswith(label + " ")
    assert "classes, in the order of P_class: " + repr(hp.CLASSES) in want


SWITCH = "HCD_PRIYA_ALLOW_SYNTHETIC"


def _readme_copy(tmp_path, change=None):
    """A copy of the README outside the package's tree (so the runner's location check passes), optionally with
    ``change = (old, new)`` applied once."""
    text = open(README, encoding="utf-8").read()
    if change is not None:
        old, new = change
        assert text.count(old) >= 1, old
        text = text.replace(old, new, 1)
    copy = tmp_path / "elsewhere" / "README.md"
    copy.parent.mkdir()
    copy.write_text(text, encoding="utf-8")
    return str(copy)


@pytest.fixture()
def stand_in(tmp_path, monkeypatch):
    """A stand-in model package with a synthetic build, as CI installs it, and the test switch on."""
    rel = make_release(tmp_path / "release")
    model_package(tmp_path, rel)  # version 0+synthetic, synthetic
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [])
    monkeypatch.setenv(SWITCH, "1")
    return rel


@pytest.mark.filterwarnings("ignore:.*synthetic test build")
def test_the_example_runs_as_written_on_the_stand_in_model(stand_in, tmp_path, capsys):
    assert R.main([_readme_copy(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("PASSED") and "model: LF+MF" in out
    assert "a synthetic test build: labels compared" in out


def test_with_a_published_model_every_line_must_equal_the_readme(tmp_path, monkeypatch, capsys):
    """The stand-in release listed as a published model: it has another number of wavenumbers than the README shows,
    so the exact comparison that a published model gets fails, and says which line."""
    rel = make_release(tmp_path / "release")
    model_package(tmp_path, rel, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    pin = {"name": "lfmf", "version": "7.0.0", "format": "hcd_priya.release/1"}
    monkeypatch.setattr(PM, "MODELS", [dict(pin, anchor_sha256=rel["sha"])])
    assert R.main([_readme_copy(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAILED: the README shows the line 'shape of pred.P_class" in out and "(4, 172)" in out


@pytest.mark.filterwarnings("ignore:.*synthetic test build")
@pytest.mark.parametrize(
    "old, new, why",
    [
        # the names are not the model's (a wrong name, and a wrong order)
        ('"ns": 0.925,', '"n_s": 0.925,', "FAILED: the example's parameter names"),
        (
            '"hub": 0.70,           # dimensionless Hubble parameter h\n    "omegamh2": 0.143,     # Omega_m h^2\n',
            '"omegamh2": 0.143,\n    "hub": 0.70,\n',
            "FAILED: the example's parameter names",
        ),
        # values that are hidden or computed
        ("params = {", "params = dict(hp.example_params()) or {", "write the nine parameters out"),
        (
            "params = {",
            "params = hp.example_params(verbose=False); unused = {",
            "write the nine parameters out",
        ),
        (
            "\npred = emu.predict(params,",
            "\nparams = dict(zip(hp.PARAM_NAMES, hp.PARAM_LIMITS.mean(axis=1)))\npred = emu.predict(params,",
            "FAILED: the prediction was not made at the parameter values the example writes out",
        ),
        (
            "\npred = emu.predict(params,",
            "\nother = dict(params, ns=0.95)\npred = emu.predict(other,",
            "FAILED: the prediction was not made at the parameter values the example writes out",
        ),
        # what is printed and what is shown
        ('print("model:"', 'print("the model:"', "FAILED: the example does not print the labels"),
        (
            'print("classes, in the order of P_class:", hp.CLASSES)\n',
            'print("classes, in the order of P_class:", hp.CLASSES)\nprint("extra:", 1)\n',
            "FAILED: the example does not print the labels",
        ),
        ("```text\nmodel: LF+MF\n", "```text\n", "FAILED: the example does not print the labels"),
        (
            "classes, in the order of P_class: ('clean', 'LLS', 'subDLA', 'DLA')\n",
            "classes, in the order of P_class: ('clean', 'DLA')\n",
            "FAILED: the README shows the line",
        ),
        ("```text\nmodel: LF+MF\n", "```text\nthe model is LF+MF\n", "each as 'label: value'"),
    ],
)
def test_a_changed_example_fails(stand_in, tmp_path, capsys, old, new, why):
    assert R.main([_readme_copy(tmp_path, (old, new))]) == 1
    assert why in capsys.readouterr().out


def test_an_example_without_an_output_block_fails(stand_in, tmp_path):
    text = open(README, encoding="utf-8").read()
    start = text.index("## Quickstart\n")
    a = text.index("```text\n", start)
    b = text.index("```\n", a + 8) + 4
    copy = tmp_path / "elsewhere" / "README.md"
    copy.parent.mkdir()
    copy.write_text(text[:a] + text[b:].replace("```text", "```output"), encoding="utf-8")
    with pytest.raises(ValueError):  # no text block follows the example
        R.main([str(copy)])
