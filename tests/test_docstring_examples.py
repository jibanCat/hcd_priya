"""The examples in the docstrings of the user entry points run and print what they say (doctest).

The examples that need a model use the names ``emu``, ``params``, ``release_dir``, ``release_sha256``,
``bundle_dir``, ``manifest_sha256``, ``research_dir`` and ``research_sha256``. Here they are a synthetic release and a synthetic research bundle
(untrained networks): the examples show the interface, not science. The example of ``load_published`` loads the
installed model package, which in tests is a synthetic build, so the test switch for synthetic builds is on.
"""

import doctest
import warnings

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import likelihood
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
    hp.example_params,
    hp.published_models,
    hp.load_published,
    dndx_wc.w_c_from_mu,
    dndx_wc.w_c_corrected,
    dndx_wc.dndx_powerlaw,
    wc_map.WcMap.w_c,
    likelihood.P1DData.from_blocks,
    likelihood.interp_to_bins,
    likelihood.GaussianP1DLikelihood,
    hp.same_run,
]


@pytest.fixture(scope="module")
def names(tmp_path_factory):
    import hcd_priya_models

    assert hcd_priya_models.manifest()["synthetic"], "the doctests expect a synthetic model package"
    rel = make_release(tmp_path_factory.mktemp("rel") / "release")
    research = RT.write_synthetic_research_bundle(str(tmp_path_factory.mktemp("res") / "research"))
    emu = hp.load_release(str(rel["root"]), rel["sha"])
    params = {  # written out, as in the examples themselves: the midpoint of each training range
        "ns": 0.925,
        "Ap": 1.9e-9,
        "herei": 4.0,
        "heref": 2.7,
        "alphaq": 2.15,
        "hub": 0.70,
        "omegamh2": 0.143,
        "hireionz": 7.25,
        "bhfeedback": 0.05,
    }
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
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        mp.setenv("HCD_PRIYA_ALLOW_SYNTHETIC", "1")
        warnings.simplefilter(
            "ignore", UserWarning
        )  # the synthetic-build warning of load_published
        for t in tests:
            runner.run(t)
    assert (
        runner.failures == 0 and runner.tries > 0
    ), f"{runner.failures} of {runner.tries} examples failed"
