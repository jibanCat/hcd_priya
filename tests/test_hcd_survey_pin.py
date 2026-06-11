"""Guard the per-survey LLS pin + the lls_truth_boost mock injection (CS referee 2026-06-11 gap).

(1) inference.hcd_incidence_prior(survey=) applies the per-survey LLS center boost + width;
    survey=None reproduces the pre-pin prior; subDLA/DLA are survey-agnostic.
(2) closure_legb.make_truth_from_sim(lls_truth_boost=) scales the MOCK's α_LLS in BOTH the
    returned truth w_c AND the contaminated P_obs_true (so bias_z uses the boosted truth).
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax.numpy as jnp

from hcd_analysis.emulator import inference as INF
from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator.data import load_cache

_CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
W = jnp.array([0.27, 0.09, 0.04])   # toy w_c (LLS, subDLA, DLA)


def test_survey_pin_center_and_width():
    muN, sdN = INF.hcd_incidence_prior(W, z=3.0)                 # survey=None (default lit_over_sim)
    muD, sdD = INF.hcd_incidence_prior(W, z=3.0, survey="DESI")
    muK, sdK = INF.hcd_incidence_prior(W, z=3.0, survey="KS")
    muD, sdD, muK, sdK, muN, sdN = map(np.asarray, (muD, sdD, muK, sdK, muN, sdN))
    # DESI LLS center == no boost (boost 1.0); KS center == 2.5 x DESI
    np.testing.assert_allclose(muD[0], muN[0], rtol=1e-6)
    np.testing.assert_allclose(muK[0] / muD[0], INF.HCD_LLS_SURVEY_BOOST["KS"], rtol=1e-6)
    # widths sigma/mu: DESI 0.30, KS 0.40, None = HCD_PRIOR_FRAC_SIGMA[0] (0.15)
    assert sdD[0] / muD[0] == pytest.approx(INF.HCD_LLS_SURVEY_FRAC_SIGMA["DESI"])
    assert sdK[0] / muK[0] == pytest.approx(INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"])
    assert sdN[0] / muN[0] == pytest.approx(INF.HCD_PRIOR_FRAC_SIGMA[0])
    # subDLA/DLA centers are survey-agnostic (only LLS is boosted)
    np.testing.assert_allclose(muD[1:], muN[1:], rtol=1e-6)
    np.testing.assert_allclose(muK[1:], muN[1:], rtol=1e-6)


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="LF cache not present")
def test_lls_truth_boost_propagates_to_mock_and_truth():
    d = load_cache(_CACHE)
    sims, _ = C.held_out_sims(d, fold=0)
    s = str(np.asarray(sims)[0])
    t1 = C.make_truth_from_sim(d, s, fold=0, lls_truth_boost=1.0)
    t2 = C.make_truth_from_sim(d, s, fold=0, lls_truth_boost=2.0)
    # returned truth w_c: LLS x2, subDLA/DLA unchanged
    assert t2["w_c"][0] / t1["w_c"][0] == pytest.approx(2.0, rel=1e-6)
    np.testing.assert_allclose(t2["w_c"][1:], t1["w_c"][1:], rtol=1e-6)
    # the contaminated mock DATA changes (more LLS power), and clean fraction drops
    assert not np.allclose(np.asarray(t2["P_obs_true"]), np.asarray(t1["P_obs_true"]))


if __name__ == "__main__":
    test_survey_pin_center_and_width()
    if os.path.exists(_CACHE):
        test_lls_truth_boost_propagates_to_mock_and_truth()
    print("[hcd-survey-pin] per-survey center/width + lls_truth_boost propagation OK.")
