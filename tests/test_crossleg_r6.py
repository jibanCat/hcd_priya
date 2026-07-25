"""Tests for the cross-leg r6x paired prior-sensitivity tooling (2026-07-24, PI decisions #7).

Covers, per the build order:
  1. OVERRIDE MECHANICS: the in-place dict item assignment IS seen by the frozen ctx-build
     path (_survey_alpha_prior, the exact consumer build_legb_ctx deploys), while a
     module-attribute rebind is NOT (the documented from-import trap) yet still moves the
     signature: the desync the mechanism exists to exclude. Unit-level on the real modules
     (a full build_legb_ctx needs checkpoints + hours; _survey_alpha_prior is the verbatim
     extracted survey block, PI decision C2 2026-07-20, so it IS the build path).
  2. Pre-registered signature-pair logic + restoration guarantee (incl. on exception).
  3. TILT VERDICT mechanics: no freeze-safe z-slope-centre knob (immutable float from-import;
     rebind desyncs; HCD_INCIDENCE_SLOPE / ZSLOPE_PRIOR_SIGMA signature-uncovered).
  4. Analyzer refusals on synthetic pkls (pair identity, signature pair, stamps, seed) +
     the either-channel extension rule.

No frozen module is touched; every mutation here restores and re-verifies.
"""
import os
import pickle

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import hcd_analysis.emulator.inference as INF
import hcd_analysis.emulator.closure_legb as CL
import scripts.crossleg_r6_common as CC
from scripts.analyze_crossleg_r6 import load_r6x, pair_report

# The frozen cache's z=3 pivot inputs (hcd_pivot_wc_and_xbar of record, full precision) so the
# _survey_alpha_prior build path runs without the h5 cache load. The LLS centre the survey
# path realizes depends only on Xbar_z3 (lit-law direct), matching R6X_ALPHA_LLS_MU_B1.
W_C_MED = np.array([0.18906395143855034, 0.06218315845676444, 0.03290798796827177])
XBAR_Z3 = 0.6316034425658955


def _survey_lls(survey):
    """(mu0, sd0, zslope_mu) floats from the frozen build path for the LLS slot."""
    import jax.numpy as jnp
    mu, sd, zmu = CL._survey_alpha_prior(jnp.asarray(W_C_MED), XBAR_Z3, survey)
    return float(np.asarray(mu)[0]), float(np.asarray(sd)[0]), zmu


# --------------------------------------------------------------------------------------------
# 1. Override mechanics: in-place mutation vs rebind, at the frozen build path.
# --------------------------------------------------------------------------------------------
def test_dicts_are_shared_objects():
    assert CL.HCD_LLS_SURVEY_BOOST is INF.HCD_LLS_SURVEY_BOOST
    assert CL.HCD_LLS_SURVEY_FRAC_SIGMA is INF.HCD_LLS_SURVEY_FRAC_SIGMA


def test_base_centre_matches_preregistered_pin():
    mu0, sd0, _ = _survey_lls("eBOSS")
    mu_e, sd_e = CC.expected_lls_centre_width("eBOSS", "deployed")
    assert mu0 == mu_e == CC.R6X_ALPHA_LLS_MU_B1
    assert sd0 == sd_e


@pytest.mark.parametrize("leg", ["eBOSS", "DESI"])
def test_inplace_mutation_is_seen_by_the_build_path(leg):
    mu0, sd0, _ = _survey_lls(leg)
    with CC.r6x_override(leg):
        # visible through closure_legb's from-imported binding (same object)
        assert CL.HCD_LLS_SURVEY_BOOST[leg] == CC.R6X_DEPLOYED_BOOST[leg] * CC.R6X_FACTOR
        assert CL.HCD_LLS_SURVEY_FRAC_SIGMA[leg] == CC.R6X_FRAC_DISP
        mu1, sd1, _ = _survey_lls(leg)
        mu_e, sd_e = CC.expected_lls_centre_width(leg, "dispprior")
        assert mu1 == mu_e, f"displaced centre {mu1!r} != expected {mu_e!r}"
        assert sd1 == sd_e, f"displaced width {sd1!r} != expected {sd_e!r}"
        # fixed ABSOLUTE width: bit-equal to the deployed width
        assert sd1 == sd0
    mu2, sd2, _ = _survey_lls(leg)
    assert mu2 == mu0 and sd2 == sd0, "build path did not return to deployed after restore"


def test_ks_override_touches_boost_only():
    with CC.r6x_override("KS"):
        assert INF.HCD_LLS_SURVEY_BOOST["KS"] == 2.5 * CC.R6X_FACTOR == 3.2175
        assert INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"] == 0.40, \
            "KS FRAC_SIGMA must stay untouched (inert on the mapped branch)"
    assert INF.HCD_LLS_SURVEY_BOOST["KS"] == 2.5


def test_module_attribute_rebind_is_NOT_seen_by_the_build_path():
    """The documented from-import trap, demonstrated: a rebind moves the signature but NOT
    the built prior -- the desync the in-place mechanism exists to exclude."""
    mu0, sd0, _ = _survey_lls("eBOSS")
    orig = INF.HCD_LLS_SURVEY_BOOST
    try:
        INF.HCD_LLS_SURVEY_BOOST = dict(orig, eBOSS=9.9)     # REBIND (the forbidden mechanism)
        # closure_legb still reads the OLD object: the build path is UNMOVED...
        assert CL.HCD_LLS_SURVEY_BOOST["eBOSS"] == 1.0
        mu1, sd1, _ = _survey_lls("eBOSS")
        assert mu1 == mu0 and sd1 == sd0
        # ...while the signature MOVED (hcd_prior_constants_payload reads inference live):
        assert INF.hcd_prior_signature() != CC.R6X_DEPLOYED_HEX, \
            "expected the rebind to move the signature (the desync direction)"
    finally:
        INF.HCD_LLS_SURVEY_BOOST = orig
    assert INF.hcd_prior_signature() == CC.R6X_DEPLOYED_HEX
    assert CL.HCD_LLS_SURVEY_BOOST is INF.HCD_LLS_SURVEY_BOOST


# --------------------------------------------------------------------------------------------
# 2. Signature pair + restoration guarantee.
# --------------------------------------------------------------------------------------------
def test_preregistered_hexes_per_leg():
    assert INF.hcd_prior_signature() == CC.R6X_DEPLOYED_HEX
    for leg in CC.R6X_LEGS:
        with CC.r6x_override(leg):
            assert INF.hcd_prior_signature() == CC.R6X_DISPLACED_HEX[leg]
        assert INF.hcd_prior_signature() == CC.R6X_DEPLOYED_HEX
    # the three displaced hexes are mutually distinct and distinct from deployed
    hexes = [CC.R6X_DEPLOYED_HEX] + [CC.R6X_DISPLACED_HEX[l] for l in CC.R6X_LEGS]
    assert len(set(hexes)) == 4


def test_restoration_on_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with CC.r6x_override("DESI"):
            assert INF.HCD_LLS_SURVEY_BOOST["DESI"] == CC.R6X_FACTOR
            raise RuntimeError("boom")
    assert INF.HCD_LLS_SURVEY_BOOST["DESI"] == 1.0
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA["DESI"] == 0.287
    assert INF.hcd_prior_signature() == CC.R6X_DEPLOYED_HEX


def test_payload_membership_tracks_the_live_dicts():
    pay = INF.hcd_prior_constants_payload()
    assert "HCD_LLS_SURVEY_BOOST" in pay and "HCD_LLS_SURVEY_FRAC_SIGMA" in pay
    with CC.r6x_override("eBOSS"):
        pay2 = INF.hcd_prior_constants_payload()
        assert pay2["HCD_LLS_SURVEY_BOOST"]["eBOSS"] == CC.R6X_FACTOR
        assert pay2["HCD_LLS_SURVEY_FRAC_SIGMA"]["eBOSS"] == CC.R6X_FRAC_DISP


def test_verify_deployed_state_refuses_drift():
    b0 = INF.HCD_LLS_SURVEY_BOOST["DESI"]
    try:
        INF.HCD_LLS_SURVEY_BOOST["DESI"] = 1.01
        with pytest.raises(AssertionError):
            CC.verify_deployed_prior_state("DESI")
    finally:
        INF.HCD_LLS_SURVEY_BOOST["DESI"] = b0


# --------------------------------------------------------------------------------------------
# 3. Tilt mechanism verification (design deliverable 2): NO freeze-safe knob.
# --------------------------------------------------------------------------------------------
def test_tilt_zslope_centre_has_no_freeze_safe_knob():
    assert isinstance(INF.HCD_LLS_REALFIT_ZSLOPE, float), \
        "z-slope centre is an immutable float: no in-place mutation path exists"
    _, _, zmu0 = _survey_lls("eBOSS")
    assert float(np.asarray(zmu0)[0]) == 2.127
    orig = INF.HCD_LLS_REALFIT_ZSLOPE
    try:
        INF.HCD_LLS_REALFIT_ZSLOPE = 2.647                      # rebind (the only option)
        # the build path keeps closure_legb's load-time binding: ctx zslope centre UNMOVED
        assert CL.HCD_LLS_REALFIT_ZSLOPE == orig
        _, _, zmu1 = _survey_lls("eBOSS")
        assert float(np.asarray(zmu1)[0]) == 2.127, \
            "rebinding inference.HCD_LLS_REALFIT_ZSLOPE must NOT move the built centre"
        # ...but the signature MOVES: the worst-case desync direction. Tilt arms DROPPED.
        assert INF.hcd_prior_signature() != CC.R6X_DEPLOYED_HEX
    finally:
        INF.HCD_LLS_REALFIT_ZSLOPE = orig
    assert INF.hcd_prior_signature() == CC.R6X_DEPLOYED_HEX


def test_tilt_slope_constants_not_signature_carried():
    pay = INF.hcd_prior_constants_payload()
    assert "HCD_INCIDENCE_SLOPE" not in pay, \
        "HCD_INCIDENCE_SLOPE (closure_legb) is covered by NO signature (payload docstring)"
    assert "ZSLOPE_PRIOR_SIGMA" not in pay, \
        "ZSLOPE_PRIOR_SIGMA (closure_legb) is covered by NO signature"
    assert CC.R6X_TILT_VERDICT.startswith("DROPPED")


# --------------------------------------------------------------------------------------------
# 4. Helpers: naming, hex lookup, extension rule.
# --------------------------------------------------------------------------------------------
def test_pkl_name_and_lookup_helpers():
    assert CC.pkl_name("eBOSS", "deployed", 3) == "r6x_eboss_deployed_shard_003.pkl"
    assert CC.pkl_name("KS", "dispprior", 11, smoke=True) == "r6x_ks_dispprior_shard_011.smoke.pkl"
    assert CC.expected_hex("DESI", "deployed") == CC.R6X_DEPLOYED_HEX
    assert CC.expected_hex("DESI", "dispprior") == CC.R6X_DISPLACED_HEX["DESI"]
    assert CC.displaced_dict_values("KS") == (3.2175, None)
    assert CC.displaced_dict_values("eBOSS") == (CC.R6X_FACTOR, CC.R6X_FRAC_DISP)
    with pytest.raises(AssertionError):
        CC.pkl_name("SDSS", "deployed", 0)


def test_extension_rule_either_channel():
    assert CC.extension_verdict(0.5, 0.5)[0] is False
    assert CC.extension_verdict(1.6, 0.0)[0] is True          # n_s channel
    assert CC.extension_verdict(0.0, -2.9)[0] is True         # A_p channel, sign-agnostic
    assert CC.extension_verdict(3.1, 0.0)[0] is False         # >= 3: resolved, no extension
    assert CC.extension_verdict(-1.5, 0.0)[0] is True         # boundary 1.5 included
    assert CC.extension_verdict(0.0, 3.0)[0] is False         # boundary 3.0 excluded
    assert CC.extension_verdict(float("nan"), 0.2)[0] is False


# --------------------------------------------------------------------------------------------
# 5. Analyzer: synthetic-pkl refusals + readout mechanics.
# --------------------------------------------------------------------------------------------
_NAMES = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"]
_RUN_KW = dict(n_warmup=250, n_samples=300, seed=CC.R6X_SEED, max_tree_depth=10,
               dense_mass=True)


def _mk_rec(mock, *, bias=0.0, rng=None, data_sha=None, n_div=0):
    """A minimal per-mock record with the fields the analyzer consumes. The truth is a pure
    function of the mock index (so the two arms of a pair agree by construction)."""
    tr = np.random.default_rng(1000 + mock)                    # truth: shared across arms
    truth_vec = tr.uniform(0.2, 0.8, size=len(_NAMES))
    truth_z = tr.uniform(0.0, 0.3, size=(4, 3))
    rng = rng or np.random.default_rng()
    draws = truth_vec[None, :] + bias + 0.01 * rng.standard_normal((50, len(_NAMES)))
    sites_extra = {}
    for nm, tv in (("tau0_amp", 1.0), ("dtau0", 0.0)):
        sites_extra[nm] = dict(draws=tv + bias + 0.01 * rng.standard_normal(50), truth=tv)
    sha = data_sha if data_sha is not None else CC.sha256_of_arrays(truth_vec, truth_z)
    return dict(truth_vec=truth_vec, truth_alpha_hcd_z=truth_z, draws=draws, names=_NAMES,
                sites_extra=sites_extra, n_div=n_div, r6x_data_sha256=sha,
                L=50, ll_true=0.0)


def _write_pkl(dirpath, leg, arm, mock, rec, *, hex_override=None, seed=CC.R6X_SEED,
               override_stamp=True, truth_source=CC.R6X_TRUTH_SOURCE, fwd_sig="fwdsig"):
    hx = hex_override if hex_override is not None else CC.expected_hex(leg, arm)
    pc = dict(r6x_override=(True if override_stamp else None), r6x_arm=arm, r6x_leg=leg,
              r6x_seed=seed, r6x_truth_source=truth_source,
              hcd_parameterization=CC.R6X_PARAMETERIZATION[leg], hcd_prior_signature=hx,
              r6x_factor=CC.R6X_FACTOR)
    if not override_stamp:
        pc.pop("r6x_override")
    meta = dict(run_kw=dict(_RUN_KW),
                forward=dict(forward_signature=fwd_sig, hcd_prior_signature=hx),
                prior_constants=pc, seed=seed)
    out = os.path.join(dirpath, CC.pkl_name(leg, arm, mock))
    with open(out, "wb") as f:
        pickle.dump(dict(arm=arm, leg=leg, survey=leg, idxs=[mock], per_mock=[rec],
                         meta=meta), f)
    return out


def _write_pair_set(dirpath, leg="eBOSS", mocks=(0, 1, 2), disp_bias=0.0):
    rng = np.random.default_rng(7)
    for m in mocks:
        _write_pkl(dirpath, leg, "deployed", m, _mk_rec(m, bias=0.0, rng=rng))
        _write_pkl(dirpath, leg, "dispprior", m, _mk_rec(m, bias=disp_bias, rng=rng))


def test_analyzer_happy_path(tmp_path):
    _write_pair_set(str(tmp_path), "eBOSS", mocks=(0, 1, 2, 3))
    out = pair_report(str(tmp_path), "eBOSS")
    assert set(out["mocks"]) == {0, 1, 2, 3}
    assert "delta_ns" in out and "delta_Ap" in out and "delta_tau0_amp" in out
    assert out["signature_pair"] == [CC.R6X_DEPLOYED_HEX, CC.R6X_DISPLACED_HEX["eBOSS"]]


def test_analyzer_extension_flag_fires_in_window(tmp_path):
    # a displaced-arm bias sized to land |t| in [1.5, 3) on every theta param with high
    # probability: delta ~ 0.006 with per-pair noise sd ~ 0.01/sqrt(50)*sqrt2 ~ 0.002
    # over 4 pairs -> SE ~ 0.001, t ~ 6 -- too big; use bias 2.5e-3 -> t ~ 2.5.
    _write_pair_set(str(tmp_path), "eBOSS", mocks=(0, 1, 2, 3), disp_bias=2.5e-3)
    out = pair_report(str(tmp_path), "eBOSS")
    t_ns = out["t_ns"]
    if 1.5 <= abs(t_ns) < 3.0 or 1.5 <= abs(out["t_Ap"]) < 3.0:
        assert bool(out["extend"]) is True
    else:
        assert bool(out["extend"]) is False


def test_analyzer_refuses_truth_mismatch(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng))
    bad = _mk_rec(0, rng=rng)
    bad["truth_vec"] = bad["truth_vec"] + 1e-9                 # NOT bit-identical
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, bad)
    with pytest.raises(AssertionError, match="truth_vec"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_data_sha_mismatch(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng))
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0,
               _mk_rec(0, rng=rng, data_sha="deadbeef"))
    with pytest.raises(AssertionError, match="data"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_signature_equality_across_arms(tmp_path):
    """The displaced arm carrying the DEPLOYED hex = the override never happened (or the
    R6-analyzer equality convention leaked in). Must refuse."""
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng))
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng),
               hex_override=CC.R6X_DEPLOYED_HEX)
    with pytest.raises(AssertionError, match="signature"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_wrong_displaced_hex(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng))
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng),
               hex_override=CC.R6X_DISPLACED_HEX["DESI"])      # another leg's displaced hex
    with pytest.raises(AssertionError, match="signature"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_missing_override_stamp(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng),
               override_stamp=False)
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng))
    with pytest.raises(AssertionError, match="r6x_override"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_wrong_seed(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng), seed=20260615)
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng))
    with pytest.raises(AssertionError, match="seed"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_foreign_truth_source(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng),
               truth_source="mapped-selfdraw")
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng))
    with pytest.raises(AssertionError, match="truth_source"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_duplicate_mock(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng))
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng))
    # a second deployed pkl claiming the same mock under a different shard id
    p = os.path.join(str(tmp_path), "r6x_eboss_deployed_shard_009.pkl")
    with open(os.path.join(str(tmp_path), CC.pkl_name("eBOSS", "deployed", 0)), "rb") as f:
        d = pickle.load(f)
    with open(p, "wb") as f:
        pickle.dump(d, f)
    with pytest.raises(AssertionError, match="duplicate"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_refuses_cross_arm_forward_drift(tmp_path):
    rng = np.random.default_rng(7)
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 0, _mk_rec(0, rng=rng), fwd_sig="fwdA")
    _write_pkl(str(tmp_path), "eBOSS", "dispprior", 0, _mk_rec(0, rng=rng), fwd_sig="fwdB")
    with pytest.raises(AssertionError, match="forward_signature"):
        pair_report(str(tmp_path), "eBOSS")


def test_analyzer_load_reports_orphans(tmp_path):
    rng = np.random.default_rng(7)
    _write_pair_set(str(tmp_path), "eBOSS", mocks=(0, 1))
    _write_pkl(str(tmp_path), "eBOSS", "deployed", 5, _mk_rec(5, rng=rng))  # orphan
    out = pair_report(str(tmp_path), "eBOSS")
    assert list(out["orphans"]) == [5]
    assert set(out["mocks"]) == {0, 1}
