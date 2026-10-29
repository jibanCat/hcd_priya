"""scripts/tau_threshold_sensitivity.py: the pure parts (class collapse, templates, comparison, reading rule) against the
emulator cache loader and hand-made cases. Synthetic arrays only; the run itself needs fake_spectra."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tau_threshold_sensitivity.py"
_spec = importlib.util.spec_from_file_location("tau_threshold_sensitivity", _PATH)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)


def test_the_class_ranges_and_the_collapse_are_the_emulator_loaders():
    from hcd_analysis.emulator import data as D
    assert tuple((s.start, s.stop) for s in D.COARSE_SLICES) == TS.COARSE_RANGES
    assert tuple(D.COARSE_NAMES) == TS.CLASSES
    rng = np.random.default_rng(1)
    p15 = rng.uniform(0.5, 2.0, size=(15, 9))
    counts = rng.integers(0, 50, size=15)
    counts[8:13] = 0                                              # an empty class stays 0, as in the loader
    np.testing.assert_array_equal(TS.collapse(p15, counts), D._collapse_p1d(p15[None], counts[None])[0])


def test_templates_are_the_forward_models_choice_of_pieces():
    pf = np.array([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    pu = pf * 1.5
    t = TS.templates(pf, pu)
    np.testing.assert_allclose(t["LLS"], [0.5, 0.5])
    np.testing.assert_allclose(t["subDLA"], [1.0, 1.0])
    np.testing.assert_allclose(t["DLA"], [2.75, 2.75])               # unfiltered DLA over filtered clean


def test_compare_separates_an_amplitude_change_from_a_shape_change():
    ref = np.array([1.0, 0.5, 0.25, 0.1])
    assert TS.compare(ref, ref) == (1.0, 0.0)
    a, s = TS.compare(1.2 * ref, ref)
    assert a == pytest.approx(1.2) and s == pytest.approx(0.0, abs=1e-15)
    a, s = TS.compare(ref + np.array([0.0, 0.0, 0.0, 0.1]), ref)
    assert s > 0.05
    assert np.isnan(TS.compare(ref, np.zeros(4))[0])


def test_class_counts_use_the_production_edges():
    assert TS.class_counts([17.19, 17.2, 18.99, 19.0, 20.29, 20.3, 21.0]) == {"LLS": 2, "subDLA": 2, "DLA": 2}


def _result(scale_lls=1.0, n_lls=100, k=6):
    p15 = np.ones((15, k))
    p15[1:8] *= 1.0 + 0.2 * scale_lls * np.linspace(1.0, 0.2, k)    # an LLS excess over clean
    p15[8:13] *= 1.5
    p15[13:15] *= 2.0
    counts = np.full(15, 10)
    return {"pf15": p15, "pu15": p15 * 1.1, "counts15": counts,
            "abs_counts": {"LLS": n_lls, "subDLA": 50, "DLA": 20}}


def test_summary_and_the_pre_registered_reading_rule():
    res = {("z", 100.0): _result(), ("z", 50.0): _result(1.02, 103), ("z", 200.0): _result(0.97, 98),
           ("z", 30.0): _result(1.5, 150)}
    s = TS.summarize(res, 100.0, n_k=6)
    assert s["z|T100"]["LLS"]["amplitude_ratio"] == pytest.approx(1.0)
    assert s["z|T50"]["LLS"]["count_ratio"] == pytest.approx(1.03)
    assert s["z|T50"]["LLS"]["amplitude_ratio"] == pytest.approx(1.02, rel=1e-12)
    v = TS.verdict(s, {50.0, 200.0}, 100.0)
    assert v["reading"] == "negligible" and v["per_class"]["LLS"]["reading"] == "negligible"
    assert TS.verdict(s, {30.0, 50.0}, 100.0)["reading"] == "important"        # a 50% amplitude change
    res[("z", 50.0)] = _result(1.2, 120)
    assert TS.verdict(TS.summarize(res, 100.0, n_k=6), {50.0, 200.0}, 100.0)["reading"] == "modest"


def test_the_summary_needs_the_reference_threshold():
    with pytest.raises(KeyError, match="no reference"):
        TS.summarize({("z", 50.0): _result()}, 100.0, n_k=6)
