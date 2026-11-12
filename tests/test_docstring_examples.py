"""The examples in the docstrings of the user entry points run and print what they say (doctest).

The examples that need a model use the names ``emu``, ``params``, ``release_dir``, ``release_sha256``,
``bundle_dir``, ``manifest_sha256``, ``research_dir`` and ``research_sha256``. Here they are a synthetic release and a synthetic research bundle
(untrained networks): the examples show the interface, not science."""

import doctest

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import dndx_wc, wc_map
from hcd_analysis.emulator import release_testing as RT
from tests.synthetic_release import make_release

OBJECTS = [
    hp.Prediction,
    hp.Bundle.predict,
    hp.Bundle.predict_many,
    hp.tau_eff_priya,
    hp.params_vector,
    hp.snapshot_z,
    hp.check_amplitudes,
    hp.load_release,
    hp.load_bundle,
    hp.load_research_bundle,
    hp.add_hcd,
    dndx_wc.w_c_from_mu,
    dndx_wc.w_c_corrected,
    dndx_wc.dndx_powerlaw,
    wc_map.WcMap.w_c,
]


@pytest.fixture(scope="module")
def names(tmp_path_factory):
    rel = make_release(tmp_path_factory.mktemp("rel") / "release")
    research = RT.write_synthetic_research_bundle(str(tmp_path_factory.mktemp("res") / "research"))
    emu = hp.load_release(str(rel["root"]), rel["sha"])
    params = {n: 0.5 * (lo + hi) for n, (lo, hi) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS)}
    return {
        "hp": hp,
        "np": np,
        "emu": emu,
        "params": params,
        "release_dir": str(rel["root"]),
        "release_sha256": rel["sha"],
        "bundle_dir": rel["bundle"].bundle_dir,
        "manifest_sha256": rel["bundle"].manifest_sha256,
        "research_dir": research.bundle_dir,
        "research_sha256": research.manifest_sha256,
    }


@pytest.mark.parametrize("obj", OBJECTS, ids=lambda o: o.__qualname__)
def test_the_docstring_examples_run_and_print_what_they_say(obj, names):
    tests = [
        t for t in doctest.DocTestFinder(recurse=False).find(obj, globs=dict(names)) if t.examples
    ]
    assert tests, f"{obj.__qualname__} has no example"
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for t in tests:
        runner.run(t)
    assert (
        runner.failures == 0 and runner.tries > 0
    ), f"{runner.failures} of {runner.tries} examples failed"
