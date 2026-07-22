"""Guard the per-survey LLS pin + the lls_truth_boost mock injection (CS referee 2026-06-11 gap),
REWRITTEN for the W2 KS dN/dX-mapped semantics (2026-07-22).

(1) inference.hcd_incidence_prior(survey=) applies the per-survey LLS center boost + width;
    survey=None reproduces the pre-pin prior; subDLA/DLA are survey-agnostic. NOTE (W2): the
    KS row of this ALPHA-SPACE machinery is retained byte-exact but is consumed only by the
    ks_legacy_alpha_param override arm and the boost-1.0 sub/DLA slots of the mapped
    reference — the deployed KS prior is the dN/dX-MAPPED parameterization.
(2) closure_legb.make_truth_from_sim(lls_truth_boost=) scales the MOCK's α_LLS in BOTH the
    returned truth w_c AND the contaminated P_obs_true (so bias_z uses the boosted truth).
(3) NEW per-survey semantics pins (W2): HCD_ALPHA_PARAMETERIZATION (which parameterization
    each survey deploys), HCD_LLS_BOOST_SPACE (where the LLS boost acts), and the KS mapped
    width constants KS_DNDX_SIGMA_{EPS,KAPPA,MSUB} + KS_DNDX_DLA_RAW_MU0.
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
    # widths sigma/mu: DESI 0.287 (PI 2026-07-18 corrected-law width, 1x = measurement +
    # kernel-common-mode on the corrected K1a points), KS 0.40 (broad, its OWN selection-
    # driven rule, not the lit-error machinery), None = HCD_PRIOR_FRAC_SIGMA[0] (0.15,
    # the closure/SBC sim-truth width — deliberately NOT cascaded)
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA["DESI"] == pytest.approx(0.287)  # the corrected 1x knob
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"] == pytest.approx(0.40)     # KS stays broad
    assert sdD[0] / muD[0] == pytest.approx(INF.HCD_LLS_SURVEY_FRAC_SIGMA["DESI"])
    assert sdK[0] / muK[0] == pytest.approx(INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"])
    assert sdN[0] / muN[0] == pytest.approx(INF.HCD_PRIOR_FRAC_SIGMA[0])
    # subDLA/DLA centers are survey-agnostic (only LLS is boosted)
    np.testing.assert_allclose(muD[1:], muN[1:], rtol=1e-6)
    np.testing.assert_allclose(muK[1:], muN[1:], rtol=1e-6)


def test_eboss_and_dk_survey_pins():
    """The per-survey LLS pin keys eBOSS + DESI+KS: both at the cosmic-average center
    (boost 1.0) and the 1x corrected-law width 0.287 (same as DESI; PI 2026-07-18)."""
    for sv in ("eBOSS", "DESI+KS"):
        assert INF.HCD_LLS_SURVEY_BOOST[sv] == pytest.approx(1.0)            # cosmic-average center
        assert INF.HCD_LLS_SURVEY_FRAC_SIGMA[sv] == pytest.approx(0.287)     # 1x corrected width
        mu, sd = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey=sv))
        muN, _ = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0))
        np.testing.assert_allclose(mu[0], muN[0], rtol=1e-6)                 # no boost
        assert sd[0] / mu[0] == pytest.approx(0.287)


def test_lls_width_hedge_2x_toggle():
    """The 2x cosmic-variance hedge (use_lls_width_hedge2x=True) doubles the DESI/eBOSS/DESI+KS
    LLS width to 0.574 (2 x the corrected 1x knob 0.287); KS is unchanged at 0.40 (its width is
    the selection-driven PI rule, NOT the lit-error 1x/2x machinery — note the hedge now EXCEEDS
    the KS width, a flagged consequence of the corrected kernel-common-mode width)."""
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X["DESI"] == pytest.approx(0.574)
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X["eBOSS"] == pytest.approx(0.574)
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X["KS"] == pytest.approx(0.40)   # KS unchanged
    for sv in ("DESI", "eBOSS", "DESI+KS"):
        mu1, sd1 = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey=sv))
        mu2, sd2 = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey=sv,
                                                           use_lls_width_hedge2x=True))
        np.testing.assert_allclose(mu1, mu2, rtol=1e-6)            # center unchanged
        assert sd1[0] / mu1[0] == pytest.approx(0.287)            # 1x primary
        assert sd2[0] / mu2[0] == pytest.approx(0.574)            # 2x hedge
    # KS: hedge is a no-op (already 0.40)
    _, sdK1 = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey="KS"))
    _, sdK2 = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey="KS",
                                                      use_lls_width_hedge2x=True))
    np.testing.assert_allclose(sdK1, sdK2, rtol=1e-6)


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


# --------------------------------------------------------------------------------------------- #
#  F1 (adversarial backfill 2026-07-19): the survey-key silent .get() fallback is FAIL-LOUD.
#  Post-width-swap, an unknown survey string silently got width HCD_PRIOR_FRAC_SIGMA[0]=0.15 =
#  1.9x TIGHTER than the deployed DESI 0.287, on a real-data path, with no guard firing.
# --------------------------------------------------------------------------------------------- #
def test_unknown_survey_key_fails_loud():
    """hcd_incidence_prior(survey=<unknown>) must RAISE (never a silent fallback width/boost);
    the message lists the valid keys. 'DESI_DR2' (a plausible future key) and 'desi' (the
    case-typo class: run_real_fit CLI names are lowercase, dict keys are not)."""
    for bad in ("DESI_DR2", "desi"):
        with pytest.raises(AssertionError, match="valid"):
            INF.hcd_incidence_prior(W, z=3.0, survey=bad)
        with pytest.raises(AssertionError, match="valid"):
            INF.hcd_incidence_prior(W, z=3.0, survey=bad, use_lls_width_hedge2x=True)


def test_survey_none_keeps_closure_constants():
    """survey=None (closure/SBC) stays on HCD_PRIOR_FRAC_SIGMA[0] with boost 1.0 — the
    correct path F1 must NOT touch."""
    mu, sd = map(np.asarray, INF.hcd_incidence_prior(W, z=3.0, survey=None))
    assert sd[0] / mu[0] == pytest.approx(INF.HCD_PRIOR_FRAC_SIGMA[0])


def test_survey_dict_keysets_congruent():
    """The per-survey dicts must key the SAME survey set — a key added to one but not
    the others is exactly the state the silent .get() fallback used to paper over.
    Extended (W2): the two new semantics dicts key the same set too."""
    assert set(INF.HCD_LLS_SURVEY_BOOST) == set(INF.HCD_LLS_SURVEY_FRAC_SIGMA) \
        == set(INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X) \
        == set(INF.HCD_ALPHA_PARAMETERIZATION) == set(INF.HCD_LLS_BOOST_SPACE)


# --------------------------------------------------------------------------------------------- #
#  W2 (2026-07-22): the NEW per-survey semantics — parameterization id, boost space, and the
#  KS mapped width constants. These are freeze-payload constants (hcd_prior_signature covers
#  them); the pins here catch a silent value edit between freezes.
# --------------------------------------------------------------------------------------------- #
def test_per_survey_parameterization_ids():
    """KS deploys the dN/dX-mapped parameterization; every other survey key stays on the
    legacy alpha-pivot power-law. IDs are the forward_stamp 'hcd_parameterization' values."""
    assert INF.HCD_ALPHA_PARAMETERIZATION == {
        "DESI": "alpha_pivot_powerlaw_v1", "eBOSS": "alpha_pivot_powerlaw_v1",
        "DESI+KS": "alpha_pivot_powerlaw_v1", "KS": "dndx_mapped_v2"}


def test_per_survey_boost_space():
    """The KS 2.5x acts in dN/dX space PRE-map (dndx_premap); DESI/eBOSS/DESI+KS keep the
    alpha-postmap convention (inert at boost 1.0). The boost VALUE itself is unchanged."""
    assert INF.HCD_LLS_BOOST_SPACE == {
        "DESI": "alpha_postmap(inert at 1.0)", "eBOSS": "alpha_postmap(inert at 1.0)",
        "DESI+KS": "alpha_postmap(inert at 1.0)", "KS": "dndx_premap"}
    assert INF.HCD_LLS_SURVEY_BOOST["KS"] == pytest.approx(2.5)   # value unchanged, space moved


def test_ks_mapped_width_constants():
    """The pinned mapped-branch width literals + the DLA latent centre (see the convention
    strings in INF.KS_DNDX_WIDTH_CONVENTION, carried in the freeze payload)."""
    import math
    assert INF.KS_DNDX_SIGMA_EPS == 0.5310
    assert INF.KS_DNDX_SIGMA_KAPPA == 0.6681
    assert INF.KS_DNDX_SIGMA_MSUB == 0.4130
    assert INF.KS_DNDX_DLA_RAW_MU0 == math.log(math.expm1(1.0))
    assert set(INF.KS_DNDX_WIDTH_CONVENTION) == {
        "eps_lls", "kappa_lls", "m_sub", "t_sub", "dla_raw", "t_dla"}
    # kappa width == the corrected-law free-gamma GLS sigma (4 dp) from the derivation JSON
    import json
    with open("/home/mfho/hcd_priya/hcd_analysis/emulator/hcd_lit_dndx_corrected.json") as fh:
        j = json.load(fh)
    assert INF.KS_DNDX_SIGMA_KAPPA == pytest.approx(
        j["adopted_law"]["free_fit_evidence"]["sigma_gamma"], abs=5e-5)


def test_every_run_real_fit_leg_is_a_known_survey_key():
    """Every leg name the run_real_fit SURVEY table passes as build_legb_ctx(survey=...) must be
    a key of the per-survey dicts (else the real-data driver dies at ctx build — fail-loud is
    correct, but the table and the dicts must never drift apart silently)."""
    from scripts.run_real_fit import SURVEY
    for sv, info in SURVEY.items():
        assert info["leg"] in INF.HCD_LLS_SURVEY_BOOST, (sv, info["leg"])
        assert info["leg"] in INF.HCD_LLS_SURVEY_FRAC_SIGMA, (sv, info["leg"])


def test_closure_legb_rebind_identity_tripwire():
    """closure_legb's from-import must alias the SAME dict objects as inference — a partial
    rebind (e.g. a module-level override in one module only) would make the build_legb_ctx
    lookups and the hcd_incidence_prior lookups disagree silently."""
    assert C.HCD_LLS_SURVEY_BOOST is INF.HCD_LLS_SURVEY_BOOST
    assert C.HCD_LLS_SURVEY_FRAC_SIGMA is INF.HCD_LLS_SURVEY_FRAC_SIGMA
    assert C.HCD_PRIOR_FRAC_SIGMA is INF.HCD_PRIOR_FRAC_SIGMA


if __name__ == "__main__":
    test_survey_pin_center_and_width()
    test_eboss_and_dk_survey_pins()
    test_lls_width_hedge_2x_toggle()
    test_unknown_survey_key_fails_loud()
    test_survey_none_keeps_closure_constants()
    test_survey_dict_keysets_congruent()
    test_per_survey_parameterization_ids()
    test_per_survey_boost_space()
    test_ks_mapped_width_constants()
    test_closure_legb_rebind_identity_tripwire()
    if os.path.exists(_CACHE):
        test_lls_truth_boost_propagates_to_mock_and_truth()
    print("[hcd-survey-pin] per-survey center/width (1x=0.287) + 2x hedge + lls_truth_boost "
          "+ W2 parameterization/boost-space/width pins OK.")
