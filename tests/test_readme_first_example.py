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


def test_the_example_runs_as_written_on_an_installed_model(tmp_path, monkeypatch, capsys):
    rel = make_release(tmp_path / "release")
    model_package(tmp_path, rel, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(
        PM,
        "MODELS",
        [
            {
                "name": "lfmf",
                "version": "7.0.0",
                "format": "hcd_priya.release/1",
                "anchor_sha256": rel["sha"],
            }
        ],
    )
    copy = tmp_path / "elsewhere" / "README.md"  # not beside the package: the location check passes
    copy.parent.mkdir()
    shutil.copyfile(README, copy)
    assert R.main([str(copy)]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("PASSED") and "model: LF+MF" in out


@pytest.mark.parametrize(
    "old, new, why",
    [
        ('"ns": 0.925,', '"n_s": 0.925,', "are not the model's"),
        ("params = {", "params = dict(hp.example_params()) or {", "write the nine parameters out"),
        ('print("model:"', 'print("the model:"', "does not print the lines that the README shows"),
        (
            "classes, in the order of P_class: ('clean', 'LLS', 'subDLA', 'DLA')\n",
            "classes, in the order of P_class: ('clean', 'DLA')\n",
            "the README shows",
        ),
    ],
)
def test_a_changed_example_fails(tmp_path, monkeypatch, capsys, old, new, why):
    rel = make_release(tmp_path / "release")
    model_package(tmp_path, rel, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(
        PM,
        "MODELS",
        [
            {
                "name": "lfmf",
                "version": "7.0.0",
                "format": "hcd_priya.release/1",
                "anchor_sha256": rel["sha"],
            }
        ],
    )
    text = open(README, encoding="utf-8").read()
    assert text.count(old) >= 1
    copy = tmp_path / "elsewhere" / "README.md"
    copy.parent.mkdir()
    copy.write_text(text.replace(old, new, 1), encoding="utf-8")
    try:
        code = R.main([str(copy)])
    except Exception as exc:  # a broken example may also stop with its own error
        code, why = 1, type(exc).__name__
        print(why)
    assert code == 1 and why in capsys.readouterr().out
