"""TDD tests for the DEPLOYED corrected literature dN/dX laws (spec sec 7, all 10 cases;
PI adoption 2026-07-18: kernel K1a, deployed LLS = constrained-slope refit at gamma=2.127).

CI consumes ONLY the committed JSON (hcd_analysis/emulator/hcd_lit_dndx_corrected.json);
cache-dependent cases sit behind the _HAS_CACHE skipif pattern.

Case list (spec sec 7):
  1. Reproduction: refit the JSON's input points with the estimator RECORDED per class,
     bit-level equality with HCD_LIT_DNDX_LAW (anti-hand-tuning; incl. the CONSTRAINED
     LLS estimator as recorded).
  2. Estimand-regression vs the tombstoned wrong-object arrays (subDLA > 3 combined
     sigma; LLS — see the documented K1a deviation in the test).
  3. Poisson-consistency: deployed subDLA deviance/dof < 2; internal consistency < 5%.
  4. Telescoping: K3-closure pinned where it applies; K1a pulls RECORDED as
     expected-values (+1.7..+2.4 at z>=3 BY DESIGN, PI-accepted tension).
  5. Kernel reproducibility (cache-gated): live r(3) vs the frozen JSON to 1e-3.
  6. Transcription: module self-checks raise on any drift.
  7. Estimand machinery: dict equality module-vs-JSON; assert_dndx_law_estimand fires.
  8. Freeze coverage: hcd_prior_constants_payload round-trip + derivation_json_sha256.
  9. DLA-law regression: shared WLS reproduces (0.0076, 1.592) (machinery anchor).
 10. atol=0 discipline on CDDF-scale comparisons.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_lit_dndx_corrected.py -q
"""
import copy
import hashlib
import json
import os

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 before jax
from hcd_analysis.emulator import lit_dndx as LD
from hcd_analysis.emulator import inference as INF

_JSON = "/home/mfho/hcd_priya/hcd_analysis/emulator/hcd_lit_dndx_corrected.json"
_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_HAS_CACHE = os.path.exists(_CACHE)


@pytest.fixture(scope="module")
def J():
    with open(_JSON) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
#  Case 1 — Reproduction (bit-level, cache-free, anti-hand-tuning)             #
# --------------------------------------------------------------------------- #
def test_case1_lls_law_reproduced_by_recorded_constrained_estimator(J):
    """Deployed LLS law == the CONSTRAINED GLS (gamma_fixed=2.127, the recorded
    estimator) re-run on the JSON's recorded K1a-corrected points with the recorded
    kernel-covariance budget — bit-level equality (PI decision 1c)."""
    ad = J["adopted_law"]
    assert ad["kernel"] == "K1a" and ad["estimator"] == "gls_log_gamma_fixed"
    pts = ad["corrected_points"]
    fit = LD.gls_powerlaw_log(np.asarray(pts["z_bar"]), np.asarray(pts["lx"]),
                              np.asarray(pts["sig_log"]),
                              s_common=ad["budget"]["s_r3"],
                              sigma_eta=ad["budget"]["sigma_eta"],
                              gamma_fixed=2.127)
    A_dep, g_dep = INF.HCD_LIT_DNDX_LAW["LLS"]
    assert fit["A"] == A_dep                          # bit-level
    assert fit["gamma"] == g_dep == 2.127
    # the free-gamma fit recorded in the JSON is the consistency evidence for the
    # constraint (PI decision 1c): free gamma 2.137 within a small fraction of its sigma
    free = ad["free_fit_evidence"]
    assert abs(free["gamma"] - 2.127) < 0.5 * free["sigma_gamma"]


def test_case1_subdla_law_reproduced_by_poisson_glm(J):
    """Deployed subDLA law == the Poisson GLM (the recorded estimator) re-run on the
    JSON's verified Zafar counts — bit-level equality."""
    a = J["input_arrays"]["ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3"]
    g = LD.poisson_glm_powerlaw(np.asarray(a["z_bar"]), np.asarray(a["n"]),
                                np.asarray(a["dX"]))
    A_dep, g_dep = INF.HCD_LIT_DNDX_LAW["subDLA"]
    assert g["A"] == A_dep                            # bit-level
    assert g["gamma"] == g_dep


def test_case1_dla_law_is_rounded_wls_anchor(J):
    """Deployed DLA law (0.0076, 1.592) == the shared-WLS refit of the PW09 points at the
    deployed rounding (the refit is documented in the JSON even though numerically the
    constant is unchanged)."""
    a = J["input_arrays"]["ELL_X_DLA_GE20P3_PW09_T1"]
    fit = LD.wls_powerlaw(np.asarray(a["z_bar"]), np.asarray(a["lx"]),
                          np.asarray(a["err_hi"]))
    A_dep, g_dep = INF.HCD_LIT_DNDX_LAW["DLA"]
    assert round(float(fit["A"]), 4) == A_dep == 0.0076
    assert round(float(fit["gamma"]), 3) == g_dep == 1.592


def test_case1_json_laws_match_deployed_dict(J):
    """The JSON's laws_deployed triple equals HCD_LIT_DNDX_LAW exactly (the committed
    artifact and the deployed module cannot diverge)."""
    for cls in ("LLS", "subDLA", "DLA"):
        A_j, g_j = J["adopted_law"]["laws_deployed"][cls]
        A_d, g_d = INF.HCD_LIT_DNDX_LAW[cls]
        assert A_j == A_d and g_j == g_d, cls


# --------------------------------------------------------------------------- #
#  Case 2 — Estimand-regression vs the tombstoned wrong-object arrays          #
# --------------------------------------------------------------------------- #
def test_case2_subdla_law_3sigma_from_tombstoned_dla_block():
    """The deployed subDLA law differs from the WLS of the tombstoned Zafar-DLA-column
    array by > 3 combined sigma: the wrong-object bug cannot be silently recommitted.

    'Combined sigma' = the 2D Mahalanobis distance of the tombstone law from the deployed
    law in the RECORDED estimator's (lnA_pivot, gamma) covariance (the deployed fit's own
    metric; a naive hypot with the tombstone WLS's slope error is diluted by that fit's
    huge sigma_gamma ~0.79 on 5 noisy points and understates the separation). Measured:
    ~6 combined sigma (the amplitude-at-pivot separation dominates; the slope alone is
    ~2.4 sigma of the GLM slope error 0.62 — few points, narrow ln(1+z) lever arm)."""
    t = LD.ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL
    wrong = LD.wls_powerlaw(t["z"], t["lx"], t["err"])
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    right = LD.poisson_glm_powerlaw(a["z_bar"], a["n"], a["dX"])
    A_dep, g_dep = INF.HCD_LIT_DNDX_LAW["subDLA"]
    # 2D distance in the recorded estimator's covariance (pivot parametrization)
    lnAp_wrong = np.log(float(wrong["A"])) + float(wrong["gamma"]) * np.log(4.0)
    lnAp_dep = np.log(A_dep) + g_dep * np.log(4.0)
    delta = np.array([lnAp_wrong - lnAp_dep, float(wrong["gamma"]) - g_dep])
    d2 = float(delta @ np.linalg.inv(np.asarray(right["cov"])) @ delta)
    assert np.sqrt(d2) > 3.0, f"tombstone only {np.sqrt(d2):.2f} combined sigma away"
    # the slope alone is separated at the ~2.4-sigma level (documented; not the gate)
    assert abs(g_dep - float(wrong["gamma"])) / right["sigma_gamma"] > 2.0
    # amplitude object is also distinct (0.0048 vs 0.0211)
    assert abs(A_dep / float(wrong["A"]) - 1.0) > 0.5


def test_case2_lls_law_distinct_from_tombstoned_cumulative_wls():
    """The deployed LLS law is NOT the WLS of the tombstoned cumulative compilation
    (0.0201, 2.127).

    DOCUMENTED DEVIATION from the spec's generic '>3 sigma' criterion: with the
    PI-adopted K1 kernel (-11% at z=3) AND the slope constrained to the same 2.127, the
    parameter-space separation is ~1.1 sigma of the measurement-only amplitude error (and
    ~0.4 sigma of the kernel-inflated error) BY CONSTRUCTION — the 3-sigma criterion was
    written for the K3 (-31%) candidate and is unachievable for K1a. The regression guard
    here is therefore (a) the amplitude moved by the full kernel correction (>5%,
    measured -8.3%), (b) exact distinctness from the tombstone WLS at reproduction
    precision, and (c) the bit-level case-1 pin, which a silent revert to 0.0201 trips
    immediately. subDLA carries the 3-sigma criterion (where it is meaningful)."""
    t = LD.LLS_TAU2_OLD_DEFECTS
    wrong = LD.wls_powerlaw(t["z"], t["lx"], t["err"])
    assert round(float(wrong["A"]), 4) == 0.0201      # the tombstone reproduces the old law
    A_dep, g_dep = INF.HCD_LIT_DNDX_LAW["LLS"]
    assert abs(A_dep / 0.0201 - 1.0) > 0.05           # the kernel correction moved A
    assert A_dep != round(float(wrong["A"]), 4)
    assert A_dep != float(wrong["A"])
    assert A_dep < 0.0201                             # cap-removal is a DOWNWARD correction
    assert g_dep == 2.127                             # slope kept BY PI DECISION (1c)


# --------------------------------------------------------------------------- #
#  Case 3 — Poisson-consistency + internal-consistency gates                   #
# --------------------------------------------------------------------------- #
def test_case3_subdla_deviance_healthy_and_internal_consistency(J):
    """Deployed subDLA law: deviance/dof < 2 against the verified counts (the old
    wrong-object law showed chi2/dof = 5.41); every deployed law within 5% of the
    recorded-estimator refit of its own input points (the transplant-killer gate)."""
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    g = LD.poisson_glm_powerlaw(a["z_bar"], a["n"], a["dX"])
    assert g["deviance_dof"] < 2.0
    z_eval = np.array([2.4, 3.0, 3.6, 4.2])
    A, gam = INF.HCD_LIT_DNDX_LAW["subDLA"]
    assert LD.law_consistency_vs_refit(A, gam, g, z_eval)["max_frac_dev"] < 0.05
    # LLS: deployed vs the recorded constrained refit of the recorded corrected points
    ad = J["adopted_law"]
    pts = ad["corrected_points"]
    con = LD.gls_powerlaw_log(np.asarray(pts["z_bar"]), np.asarray(pts["lx"]),
                              np.asarray(pts["sig_log"]),
                              s_common=ad["budget"]["s_r3"],
                              sigma_eta=ad["budget"]["sigma_eta"], gamma_fixed=2.127)
    A_l, g_l = INF.HCD_LIT_DNDX_LAW["LLS"]
    assert LD.law_consistency_vs_refit(A_l, g_l, con, z_eval)["max_frac_dev"] < 0.05
    # DLA: deployed (rounded anchor) vs the WLS refit
    d = J["input_arrays"]["ELL_X_DLA_GE20P3_PW09_T1"]
    wls = LD.wls_powerlaw(np.asarray(d["z_bar"]), np.asarray(d["lx"]),
                          np.asarray(d["err_hi"]))
    A_d, g_d = INF.HCD_LIT_DNDX_LAW["DLA"]
    assert LD.law_consistency_vs_refit(A_d, g_d, wls, z_eval)["max_frac_dev"] < 0.05


# --------------------------------------------------------------------------- #
#  Case 4 — Telescoping: K3 closes; K1a pulls are RECORDED expected-values     #
# --------------------------------------------------------------------------- #
def test_case4_telescoping_k3_closure_and_k1a_expected_pulls(J):
    """K3 (the fit-based-cap kernel) is telescoping-consistent BY CONSTRUCTION: its pulls
    stay < 1 — the closure property is pinned where it applies. The ADOPTED K1a law's
    pulls are +1.7..+2.4 sigma at z in [3.0, 4.0] BY DESIGN (the PRIYA-CDDF class shares
    vs the literature class decomposition are in real tension; PI-accepted 2026-07-18,
    recorded in the JSON pi_adoption.telescoping_note) — so for K1a the test pins the
    RECORDED expected values, NOT closure."""
    k3_pull = np.asarray(J["telescoping"]["K3"]["pull"], float)
    assert np.all(np.abs(k3_pull) < 1.0)              # closure where it applies
    # adopted (constrained K1a) pulls: recorded expected-values, PI-accepted tension
    pull = np.asarray(J["adopted_law"]["telescoping"]["pull"], float)
    z = np.asarray(J["adopted_law"]["telescoping"]["z"], float)
    exp = np.array([0.87, 1.70, 2.39, 2.10])          # the recorded 2026-07-18 values
    np.testing.assert_allclose(pull, exp, rtol=0, atol=0.05)
    hi = pull[z >= 3.0]
    assert np.all((hi > 1.5) & (hi < 2.6)), \
        "K1a z>=3 pulls left the documented +1.7..+2.4 design window — re-open PI review"
    assert "PI-accepted" in J["pi_adoption"]["telescoping_note"] or \
        "PI-accepted" in J["pi_adoption"]["telescoping_note"].replace("; ", " ")


# --------------------------------------------------------------------------- #
#  Case 5 — Kernel reproducibility (cache-gated)                               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_case5_kernel_r3_reproducible_from_cache(J):
    """r(3.0) recomputed from the h5 under the pinned K1 recipe matches the frozen JSON
    value to 1e-3 (else the committed kernel is unreproducible)."""
    from hcd_analysis.emulator import lit_dndx_kernel as LK
    inputs = LK.load_kernel_inputs(_CACHE)
    r3_live = LK.k1_r_at(inputs, np.array([3.0]))[0]
    z_vals = np.asarray(J["kernel_table"]["z_vals"], float)
    r_json = np.asarray(J["kernel_table"]["K1_r"], float)
    r3_json = LK.interp_ln1pz(z_vals, r_json, np.array([3.0]))[0]
    assert abs(r3_live - r3_json) < 1e-3
    assert abs(r3_live - J["kernel_table"]["budget"]["r3_k1"]) < 1e-3


# --------------------------------------------------------------------------- #
#  Case 6 — Transcription self-checks raise on drift                           #
# --------------------------------------------------------------------------- #
def test_case6_transcription_self_checks_raise_on_drift():
    bad = copy.deepcopy(LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)
    bad["n"] = np.asarray(bad["n"]).copy()
    bad["n"][3] += 1                                   # 89 -> 90
    with pytest.raises(AssertionError):
        LD._check_zafar_subdla(bad)
    bad2 = copy.deepcopy(LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)
    bad2["lx_gt203"] = np.asarray(bad2["lx_gt203"]).copy()
    bad2["lx_gt203"][1] += 0.05                        # breaks the two-route <= 0.025
    with pytest.raises(AssertionError):
        LD._check_zafar_subdla(bad2)


# --------------------------------------------------------------------------- #
#  Case 7 — Estimand machinery                                                 #
# --------------------------------------------------------------------------- #
def test_case7_estimand_dict_module_vs_json(J):
    assert INF.HCD_LIT_DNDX_ESTIMAND == {"LLS": "binned_17.2_19.0",
                                         "subDLA": "binned_19.0_20.3",
                                         "DLA": "binned_ge20.3"}
    assert INF.HCD_LIT_DNDX_ESTIMAND == J["hcd_lit_dndx_estimand"]


def test_case7_assert_dndx_law_estimand_fires_on_mismatch(monkeypatch):
    """The runtime estimand guard (first line of hcd_lls_realfit_alpha_center, on the
    CONSTANT dict — trace-safe, mirrors the _btilt_site guard pattern) passes the true
    estimand and fires on a wrong-object expectation or a drifted dict."""
    INF.assert_dndx_law_estimand("LLS", "binned_17.2_19.0", "test")
    with pytest.raises(AssertionError):
        INF.assert_dndx_law_estimand("LLS", "cumulative_ge17.5_tau2", "test")
    # a future re-labeling of the dict (wrong-object reinstatement) trips the consumer
    monkeypatch.setitem(INF.HCD_LIT_DNDX_ESTIMAND, "LLS", "cumulative_ge17.5_tau2")
    with pytest.raises(AssertionError):
        INF.hcd_lls_realfit_alpha_center(0.632, z=3.0)


def test_case7_alpha_center_consumes_corrected_law():
    """hcd_lls_realfit_alpha_center picks up the corrected constrained law: alpha(z=3,
    Xbar=0.632) == the JSON-adopted center (~0.172), no consumer-code change (alt-(b))."""
    with open(_JSON) as fh:
        j = json.load(fh)
    a = INF.hcd_lls_realfit_alpha_center(0.632, z=3.0, boost=1.0)
    assert a == pytest.approx(j["alpha_lls_z3"]["adopted"], rel=1e-9)
    assert a == pytest.approx(0.172, abs=5e-4)         # human-blessed tripwire (3 dp)


# --------------------------------------------------------------------------- #
#  Case 8 — Freeze coverage: payload + signature                               #
# --------------------------------------------------------------------------- #
_PAYLOAD_KEYS = ("HCD_LIT_DNDX_LAW", "HCD_LIT_DNDX_ESTIMAND", "HCD_LIT_OVER_SIM",
                 "HCD_LIT_OVER_SIM_SLOPE", "HCD_LLS_REALFIT_ZSLOPE",
                 "HCD_PIVOT_LLS_ALPHA_Z3_BAND", "HCD_PIVOT_LLS_ALLZ_MEDIAN",
                 "HCD_PIVOT_GUARD_REL", "HCD_PRIOR_FRAC_SIGMA", "HCD_LLS_SURVEY_BOOST",
                 "HCD_LLS_SURVEY_FRAC_SIGMA", "HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X",
                 "HCD_DLA_RESIDUAL_FRAC", "HCD_DLA_Z_RELIABLE", "HCD_Z_PIVOT", "adopted_kernel",
                 "adopted_kernel_r3", "adopted_alpha_lls_z3", "bracket_alpha_lls_z3",
                 "derivation_json_sha256")


def test_case8_payload_roundtrip_and_sha256():
    p = INF.hcd_prior_constants_payload()
    for k in _PAYLOAD_KEYS:
        assert k in p, k
    # JSON-native round-trip (the freeze task inserts this into analysis.lock)
    p2 = json.loads(json.dumps(p, sort_keys=True))
    assert p2 == p
    # the payload mirrors the deployed module constants exactly
    assert p["HCD_LIT_DNDX_LAW"] == {c: list(v) for c, v in INF.HCD_LIT_DNDX_LAW.items()}
    assert p["HCD_LLS_REALFIT_ZSLOPE"] == INF.HCD_LLS_REALFIT_ZSLOPE
    assert p["HCD_PIVOT_LLS_ALPHA_Z3_BAND"] == list(INF.HCD_PIVOT_LLS_ALPHA_Z3_BAND)
    assert p["HCD_DLA_Z_RELIABLE"] == INF.HCD_DLA_Z_RELIABLE
    assert p["adopted_kernel"] == "K1a"
    # derivation_json_sha256 matches the committed artifact bytes
    with open(_JSON, "rb") as fh:
        assert p["derivation_json_sha256"] == hashlib.sha256(fh.read()).hexdigest()


def test_case8b_deploy_slots_pinned_to_json():
    """The LIT_OVER_SIM slots and the band must equal the JSON's adopted deploy values — a
    silent revert of slot 0 to 1.06 previously passed every test (review meta finding 2; this
    failure class recurred 3+ times in this campaign's history)."""
    with open(_JSON) as fh:
        J = json.load(fh)
    alos = J["adopted_lit_over_sim"]
    for i, cls in enumerate(("LLS", "subDLA", "DLA")):
        assert INF.HCD_LIT_OVER_SIM[i] == alos[cls]["deploy_ratio_zp"], cls
        assert INF.HCD_LIT_OVER_SIM_SLOPE[i] == alos[cls]["deploy_ratio_slope"], cls
    assert list(INF.HCD_PIVOT_LLS_ALPHA_Z3_BAND) == list(J["adopted_band"]["band"])


def test_case8_signature_is_sha256_and_tracks_constants(monkeypatch):
    """hcd_prior_signature = sha256 over the canonical-JSON payload (forward_signature
    pattern, closure_legb.py); deterministic, and it MOVES when a prior constant moves —
    the tripwire gap forward_signature's docstring flags is closed."""
    s1 = INF.hcd_prior_signature()
    assert len(s1) == 64 and int(s1, 16) >= 0
    assert s1 == hashlib.sha256(
        json.dumps(INF.hcd_prior_constants_payload(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert INF.hcd_prior_signature() == s1             # deterministic
    monkeypatch.setattr(INF, "HCD_LLS_REALFIT_ZSLOPE", 2.128)
    assert INF.hcd_prior_signature() != s1             # a constant change moves it


# --------------------------------------------------------------------------- #
#  Case 9 — DLA-law regression anchor                                          #
# --------------------------------------------------------------------------- #
def test_case9_dla_wls_anchor():
    a = LD.ELL_X_DLA_GE20P3_PW09_T1
    fit = LD.wls_powerlaw(a["z_bar"], a["lx"], a["err_hi"])
    assert round(float(fit["A"]), 4) == 0.0076
    assert round(float(fit["gamma"]), 3) == 1.592


# --------------------------------------------------------------------------- #
#  Case 10 — atol=0 discipline                                                 #
# --------------------------------------------------------------------------- #
def test_case10_atol0_discipline_on_cddf_scale():
    good = np.array([3.0e-21, 5.0e-22, 8.0e-23])
    bad = good * 1.5                                   # 50% wrong; np.allclose says fine
    assert np.allclose(good, bad)                      # the footgun is real
    with pytest.raises(AssertionError):
        LD.assert_close_cddf(good, bad, rtol=1e-3, where="case10")
    LD.assert_close_cddf(good, good * (1 + 5e-4), rtol=1e-3, where="case10-ok")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
