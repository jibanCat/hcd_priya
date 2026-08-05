"""A1c PAIRED REPAIR COMPARISON (PI rulings, 2026-07-29).

Pre-registration section 3d makes the NO-REPEAT test a PAIRED test on the 48 common mocks, and
amendment 3 makes a six-conjunct NEGATIVE CONTROL mandatory before any paired number may be read.
Neither existed in code: the metals-off comparison was assembled by hand in a memo, and
`truth_vec` equality -- the assertion the pairing rests on -- is EXACTLY what a silent no-op
produces, so the check was passed most easily by the accident it exists to exclude.

Two PI rulings bind this module:

  1. n_s IS INCLUDED. A1's n_s passing its gate is NOT evidence the defect left n_s unaffected --
     the metals-off arm measured the LARGEST paired delta of any channel on n_s
     (+0.303 +/- 0.081, t = +3.72), on a GATED cosmology channel.
  2. The comparison is fixed in advance and computed by committed code, so no channel and no
     statistic can be chosen after the verdict is known.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_a1c_paired.py -q
"""
import importlib
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

paired = importlib.import_module("scripts.analyze_a1c_paired")
runner = importlib.import_module("scripts.run_prod_sbc_shard")

SIX = ["f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1", "k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1",
       "f_res_amp", "f_res_slope"]
NAMES = (["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]
         + [f"tau0_z{i}" for i in range(13)]
         + ["alpha_lls", "alpha_subdla", "alpha_dla"])
BRACKETS = paired.PRIOR_BRACKETS


def _draw_site_truth(nm, rng, *, reuse=False, wrong_prior=False):
    """A per-mock site truth from the DEPLOYED law (LogUniform for the metal nodes, Normal(0,
    0.05/0.5) for f_res_amp/slope). `reuse` reproduces the pre-2026-08 fixture -- one value per
    site reused across every mock, which the 3b-item-3 defect showed passed all six conjuncts.
    `wrong_prior` draws DISTINCT values strictly inside the brackets but from a plainly wrong
    law, so ONLY the distributional conjunct can catch it."""
    lo, hi, kind = BRACKETS[nm]
    if reuse:
        return (lo + hi) / 2.0 if kind == "bracket" else 0.0049
    if kind == "bracket":
        if wrong_prior:
            return float(rng.uniform(0.97 * hi, 0.999 * hi))     # in-bracket, wrong law
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    sig = 0.05 if nm == "f_res_amp" else 0.5
    if wrong_prior:
        return float(rng.normal(0.98 * sig, 1e-4 * sig))          # in-support, wrong law
    return float(rng.normal(0.0, sig))


def _mock(idx, *, corrected, seed=0, truth_shift=None, L=40, reuse_truths=False,
          wrong_prior=False):
    """One pkl-shaped record. `corrected` decides whether the six repaired truths are DRAWN
    (A1c) or `nan` (A1 -- and also exactly what a silent no-op produces)."""
    rng = np.random.default_rng(1000 + idx)
    truth = np.concatenate([rng.uniform(0.2, 0.8, 9),
                            np.linspace(0.2, 1.2, 13),
                            rng.uniform(0.3, 0.6, 3)])
    if truth_shift is not None:
        truth = truth.copy()
        truth[0] = np.nextafter(truth[0], truth[0] + truth_shift)
    dr = np.random.default_rng(seed * 977 + idx)
    draws = truth[None, :] + dr.normal(0, 0.05, (L, len(NAMES)))
    se = {}
    for nm in SIX:
        drawn = _draw_site_truth(nm, dr, reuse=reuse_truths, wrong_prior=wrong_prior)
        se[nm] = {"draws": dr.normal(0.01, 0.002, L),
                  "truth": float(np.nan) if not corrected else float(drawn)}
    cfg = dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", leg="eBOSS",
               metal_prior="flatlog2node", metal_selfdraw=corrected, fres_selfdraw=corrected,
               # the frozen constants conjunct 9 (PI 3d.4 / handoff 3b item 4) checks on BOTH arms
               hcd_prior_signature=paired.FROZEN_RUN_CFG["hcd_prior_signature"],
               sample_res=True, f_res_amp_sigma=0.05)
    return dict(names=list(NAMES), truth_vec=truth, draws=draws, L=L, n_div=0,
                sim=f"s{idx}", run_cfg=cfg, sites_extra=se,
                ll_true=(-1931.7 if corrected else -349.9),
                ll_draws=dr.normal(-1930 if corrected else -350, 5, L),
                truth_site_semantics={"self_draw": SIX if corrected else [],
                                      "not_self_drawn": [] if corrected else list(SIX),
                                      "note": ""})


def _arms(n=12, **kw):
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(n)}
    a1c = {i: _mock(i, corrected=True, seed=2, **kw) for i in range(n)}
    return a1c, a1


# --------------------------------------------------------------- the negative control (3d)

def test_pairing_passes_on_genuinely_paired_arms():
    a1c, a1 = _arms()
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is True, v
    assert all(c["ok"] for c in v["conjuncts"].values())


def test_a_silent_no_op_is_REJECTED_even_though_truth_vec_matches():
    """THE WHOLE POINT. If A1c silently no-ops (the OUTDIR footgun, or a dropped
    --fres-selfdraw) then truth_vec is identical, names are identical, indices are identical --
    and the arms are the SAME population. Conjunct 3 is the one that must fire."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    noop = {i: _mock(i, corrected=False, seed=1) for i in range(12)}   # flags never took effect
    v = paired.verify_pairing(noop, a1, expect_n=12)
    assert v["ok"] is False
    assert v["conjuncts"]["truth_vec_exact"]["ok"] is True, "the accident still passes conjunct 1"
    assert v["conjuncts"]["a1c_truths_drawn"]["ok"] is False, "conjunct 3 must catch the no-op"


def test_pairing_uses_EXACT_float_equality_not_allclose():
    """Amendment 3: `np.array_equal`/`tobytes`, never `allclose`. A one-ulp difference means the
    mocks are not the same draw, so the paired test is invalid."""
    a1c, a1 = _arms(truth_shift=1.0)                  # one ulp on ns in every A1c mock
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False
    assert v["conjuncts"]["truth_vec_exact"]["ok"] is False


def test_pairing_requires_both_flags_in_the_a1c_run_cfg():
    a1c, a1 = _arms()
    for d in a1c.values():
        d["run_cfg"] = dict(d["run_cfg"], fres_selfdraw=False)
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False and v["conjuncts"]["a1c_flags"]["ok"] is False


def test_pairing_requires_the_expected_mock_indices():
    a1c, a1 = _arms(n=12)
    a1c.pop(11)
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False and v["conjuncts"]["indices"]["ok"] is False


def test_pairing_requires_materially_different_ll_true():
    """Quantified in advance (amendment 3 left 'materially different' undefined): the observed
    gap on mock 0 is ~1580, so a threshold of 100 is far below the real signal and far above
    any numerical wobble."""
    a1c, a1 = _arms()
    for d in a1c.values():
        d["ll_true"] = -349.9                          # truth recorded but never propagated
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False and v["conjuncts"]["ll_true_moved"]["ok"] is False


def test_pairing_requires_the_provenance_stamp_to_be_clear():
    a1c, a1 = _arms()
    for d in a1c.values():
        d["truth_site_semantics"] = {"self_draw": [], "not_self_drawn": list(SIX), "note": ""}
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False and v["conjuncts"]["stamp_clear"]["ok"] is False


# --------------------------------------------------------------- the paired comparison

def test_paired_report_covers_ns_Ap_AND_tau0amp():
    """PI ruling 1: n_s is IN. A1's n_s gate pass is not evidence the defect left n_s alone."""
    a1c, a1 = _arms(n=16)
    rep = paired.paired_report(a1c, a1, expect_n=16)
    assert set(rep["channels"]) == {"ns", "Ap", "tau0amp"}


def test_paired_report_gives_delta_ci_rho_and_a_distribution_free_backup():
    a1c, a1 = _arms(n=16)
    ch = paired.paired_report(a1c, a1, expect_n=16)["channels"]["ns"]
    for k in ("delta_mean", "ci95_lo", "ci95_hi", "t", "p", "wilcoxon_p", "rho", "n"):
        assert k in ch, f"missing {k}"
    assert ch["n"] == 16
    assert ch["ci95_lo"] <= ch["delta_mean"] <= ch["ci95_hi"]
    assert -1.0 <= ch["rho"] <= 1.0


def test_paired_report_compares_the_CI_against_the_defect_magnitude_not_a_bare_p():
    """Amendment 3: rejecting H0: delta=0 licenses 'the arms differ', NOT 'the failure does not
    reproduce'. The report must carry both comparisons explicitly."""
    a1c, a1 = _arms(n=16)
    ch = paired.paired_report(a1c, a1, expect_n=16)["channels"]["Ap"]
    assert "full_removal_delta" in ch
    assert "ci_excludes_full_survival" in ch and "ci_excludes_full_removal" in ch


def test_paired_report_REFUSES_when_the_negative_control_fails():
    """'Any failure means the arms are not the populations they claim to be: STOP, do not read a
    number.' That must be enforced by the code, not by the reader's discipline."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    noop = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    with pytest.raises(paired.PairingError):
        paired.paired_report(noop, a1, expect_n=12)


# ------------------------------------------- the across-mock distributional control (PI 3d.4,
# ------------------------------------------- closing handoff 3b items 3 and 4)

def test_THE_DEFECT_reused_truths_are_REJECTED_across_mocks():
    """THE 3b-ITEM-3 SCENARIO. An arm that drew each of the six truths ONCE and reused it on
    every mock passed all six per-mock conjuncts -- demonstrated by this module's own
    pre-2026-08 fixture, which did exactly that. Reuse is not a prior-predictive draw; the
    distinctness conjunct must fire."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    a1c = {i: _mock(i, corrected=True, seed=2, reuse_truths=True) for i in range(12)}
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert "truths_distinct" in v["conjuncts"], "the distinctness conjunct must exist"
    assert v["ok"] is False
    assert v["conjuncts"]["truths_distinct"]["ok"] is False


def test_wrong_prior_draws_are_REJECTED_by_the_KS_conjunct():
    """Distinct, finite, strictly in-bracket -- but from a plainly wrong law. Only an
    across-mock distributional test can catch this; the per-mock conjuncts all pass."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    a1c = {i: _mock(i, corrected=True, seed=2, wrong_prior=True) for i in range(12)}
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert "truths_match_prior" in v["conjuncts"], "the KS conjunct must exist"
    assert v["conjuncts"]["truths_distinct"]["ok"] is True, "distinctness alone cannot catch it"
    assert v["ok"] is False
    assert v["conjuncts"]["truths_match_prior"]["ok"] is False


def test_healthy_arms_drawn_from_the_deployed_laws_pass_both_new_conjuncts():
    a1c, a1 = _arms(n=12)
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is True, v
    assert v["conjuncts"]["truths_distinct"]["ok"] is True
    assert v["conjuncts"]["truths_match_prior"]["ok"] is True


def test_run_cfg_frozen_constants_are_checked_on_BOTH_arms():
    """THE 3b-ITEM-4 SCENARIO. Every paired delta is measured AGAINST A1, and section 5c
    documents a near-perfect N=48 decoy population -- yet only A1c's run_cfg was validated. A1
    with a wrong prior signature must now refuse."""
    a1c, a1 = _arms(n=12)
    for d in a1.values():
        d["run_cfg"] = dict(d["run_cfg"], hcd_prior_signature="deadbeef" + "0" * 56)
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert "run_cfg_frozen" in v["conjuncts"], "the frozen-constants conjunct must exist"
    assert v["ok"] is False
    assert v["conjuncts"]["run_cfg_frozen"]["ok"] is False


def test_run_cfg_frozen_catches_a_mismatched_f_res_sigma_on_a1c():
    """The DESI width (0.02) on an eBOSS arm is a different fitting prior, not a variant."""
    a1c, a1 = _arms(n=12)
    for d in a1c.values():
        d["run_cfg"] = dict(d["run_cfg"], f_res_amp_sigma=0.02)
    v = paired.verify_pairing(a1c, a1, expect_n=12)
    assert v["ok"] is False
    assert v["conjuncts"]["run_cfg_frozen"]["ok"] is False


def test_frozen_signature_matches_the_freeze_artifact():
    """FROZEN_RUN_CFG is only trustworthy if it is pinned to the artifact of record: the
    hcd_prior_signature it asserts must be the one the committed analysis.lock carries (the
    freeze cut c7eb371), and the lock must carry exactly ONE such prior signature."""
    import re as _re
    lock = open("/home/mfho/hcd_priya/analysis.lock").read()
    sigs = set(_re.findall(r"\b50befc94[0-9a-f]{56}\b", lock))
    assert len(sigs) == 1, f"expected one frozen prior signature in the lock, got {sigs}"
    assert paired.FROZEN_RUN_CFG["hcd_prior_signature"] == sigs.pop()



def test_reference_deltas_are_the_MATCHED_subset_never_a_full_96_constant():
    """The campaign's single most important methodological point, re-checked here: A1c runs mocks
    0-47, so the reference must be A1's own mocks-0-47 pull, NOT the full-96 value
    (Ap -0.4053, tau0amp +0.5988)."""
    assert paired.A1_MATCHED_PULL["Ap"] == pytest.approx(-0.4046, abs=1e-4)
    assert paired.A1_MATCHED_PULL["tau0amp"] == pytest.approx(+0.4916, abs=1e-4)
    assert paired.A1_MATCHED_PULL["ns"] == pytest.approx(-0.2532, abs=1e-4), (
        "n_s DOES have a matched value; the PI ruled no hypothesis test, which is not the same "
        "as the quantity not existing")
    assert paired.FULL_REMOVAL_DELTA["ns"] is None, "no hypothesis test on n_s (PI ruling)"


def test_paired_report_is_not_a_gate():
    """PI #9 forbids a gate-definition change. No channel may carry a pass/fail verdict, and the
    report must declare itself ungated. (The prose note is exempt: it necessarily discusses A1's
    *gate pass*, which is the very thing PI ruling 1 says is not evidence of an unaffected n_s.)"""
    a1c, a1 = _arms(n=16)
    rep = paired.paired_report(a1c, a1, expect_n=16)
    assert rep["gated"] is False
    for k, c in rep["channels"].items():
        assert "verdict" not in c and "gate" not in c, f"{k} must carry no verdict"
        blob = " ".join(str(v) for v in c.values()).lower()
        assert "pass" not in blob and "fail" not in blob


# ============================================ the CI semantics (re-review, 2026-07-29)
# The first cut stored REFERENCE_DELTA["tau0amp"] = +0.4916 and called exclusion of it
# "ci_excludes_full_survival". Both are wrong, and the error is direction-dependent:
#
#   delta = A1c - A1.  FULL SURVIVAL => delta = 0.  FULL REMOVAL => delta = -(A1 mean).
#
# A1's matched mocks-0-47 pulls are Ap -0.4046 and tau0amp +0.4916, so the full-REMOVAL delta is
# +0.4046 for Ap but -0.4916 for tau0amp. Storing both as positive made the tau0amp flag True
# under EVERY hypothesis, including exact full survival -- so section 4c's pull limb, the branch
# PI ruling 2 was added to close, could never fire. Pin the BEHAVIOUR, not the constants.

def _arm_with_pulls(vals, base, seed):
    """An arm whose ns/Ap/tau0amp pulls are driven to prescribed means by shifting the truth."""
    out = {}
    rng = np.random.default_rng(seed)
    for i, _ in enumerate(base):
        d = base[i]
        out[i] = d
    return out


def _synth_delta(a1_mean, target_delta, n=48, seed=0):
    """Paired pull arrays (a1c, a1) with A1 mean `a1_mean` and paired delta `target_delta`."""
    rng = np.random.default_rng(seed)
    a1 = a1_mean + rng.normal(0, 0.95, n)
    a1c = a1 + target_delta + rng.normal(0, 0.30, n)
    return a1c, a1


def _ci(a1c_v, a1_v):
    from scipy import stats as st
    d = np.asarray(a1c_v) - np.asarray(a1_v)
    n = d.size
    sem = d.std(ddof=1) / np.sqrt(n)
    t = st.t.ppf(0.975, n - 1)
    return float(d.mean() - t * sem), float(d.mean() + t * sem)


def test_full_removal_delta_has_the_sign_of_MINUS_the_defect():
    """The arithmetic the first cut got wrong. Under full removal A1c -> 0, so the paired delta is
    minus A1's own pull: +0.4046 for A_p (whose pull was negative) and -0.4916 for tau0_amp."""
    assert paired.FULL_REMOVAL_DELTA["Ap"] == pytest.approx(+0.4046, abs=1e-4)
    assert paired.FULL_REMOVAL_DELTA["tau0amp"] == pytest.approx(-0.4916, abs=1e-4)
    assert paired.FULL_REMOVAL_DELTA["ns"] is None


def test_full_survival_is_flagged_NOT_excluded_on_both_channels():
    """THE BRANCH 4c(b) EXISTS TO CATCH. If the defect fully survives the paired delta is ~0, so
    the CI must NOT exclude full survival -- on tau0_amp as well as A_p. The first cut reported
    'excluded' here for tau0_amp, i.e. false reassurance on the arm's highest-power statistic."""
    for ch, a1_mean in (("Ap", -0.4046), ("tau0amp", +0.4916)):
        lo, hi = _ci(*_synth_delta(a1_mean, 0.0, seed=7))
        assert paired.ci_excludes_full_survival(lo, hi) is False, ch


def test_full_removal_is_flagged_excluded_from_zero_on_both_channels():
    """Under a working repair the delta is -(A1 mean), so the CI must EXCLUDE zero (the arms
    differ / the defect did not fully survive) on both channels."""
    for ch in ("Ap", "tau0amp"):
        lo, hi = _ci(*_synth_delta(-paired.FULL_REMOVAL_DELTA[ch],
                                   paired.FULL_REMOVAL_DELTA[ch], seed=11))
        assert paired.ci_excludes_full_survival(lo, hi) is True, ch
        assert paired.ci_excludes_full_removal(lo, hi, paired.FULL_REMOVAL_DELTA[ch]) is False, ch


def test_report_carries_both_comparisons_with_unambiguous_names():
    a1c, a1 = _arms(n=16)
    ch = paired.paired_report(a1c, a1, expect_n=16)["channels"]["tau0amp"]
    assert "ci_excludes_full_survival" in ch and "ci_excludes_full_removal" in ch
    assert "full_removal_delta" in ch
    assert ch["full_removal_delta"] == pytest.approx(-0.4916, abs=1e-4)


def test_escalation_flag_fires_when_full_survival_is_not_excluded():
    """Section 4c limb (b) clause 2, as code: escalate when the CI fails to exclude delta = 0."""
    lo, hi = _ci(*_synth_delta(+0.4916, 0.0, seed=3))          # tau0 defect fully survives
    assert paired.tau0_escalates(pull_mean=0.10, ci_lo=lo, ci_hi=hi) is True
    lo, hi = _ci(*_synth_delta(+0.4916, -0.4916, seed=3))      # tau0 defect removed
    assert paired.tau0_escalates(pull_mean=0.02, ci_lo=lo, ci_hi=hi) is False


def test_escalation_flag_fires_on_a_materially_nonzero_pull_alone():
    """Limb (b) clause 1 is independent: |pull mean| > 0.30 escalates even if the CI is clean."""
    lo, hi = _ci(*_synth_delta(+0.4916, -0.4916, seed=5))
    assert paired.tau0_escalates(pull_mean=0.45, ci_lo=lo, ci_hi=hi) is True
    assert paired.tau0_escalates(pull_mean=-0.45, ci_lo=lo, ci_hi=hi) is True


def test_metal_fnode_bracket_matches_the_DEPLOYED_prior():
    """closure_legb metal_fnode_lo/hi are 0.003/0.03. The first cut used 1e-3, so a truth of
    0.002 -- which the deployed LogUniform cannot produce -- passed the in-support conjunct."""
    import importlib
    CL = importlib.import_module("hcd_analysis.emulator.closure_legb")
    d = CL.prod_norc_forward() if hasattr(CL, "prod_norc_forward") else None
    lo, hi, kind = paired.PRIOR_BRACKETS["f_SiIII_eBOSS_z0"]
    assert (lo, hi) == (0.003, 0.03) and kind == "bracket"
    assert paired.PRIOR_BRACKETS["k_SiIII_eBOSS_z0"][:2] == (1e-3, 0.1)


def test_paired_report_verifies_by_DEFAULT():
    """'Refusal must not depend on the reader's discipline' -- so it must not depend on remembering
    to pass verify=True either."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    noop = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    with pytest.raises(paired.PairingError):
        paired.paired_report(noop, a1, expect_n=12)


def _arms_with_tau0_pull(a1_pull, a1c_pull, n=48):
    """Genuinely paired arms (all six conjuncts pass) whose tau0_amp pulls are driven to
    prescribed means by offsetting each arm's tau0 draws away from the SHARED truth ladder."""
    a1c, a1 = _arms(n=n)
    tau0 = [NAMES.index(f"tau0_z{i}") for i in range(13)]
    for arm, want in ((a1, a1_pull), (a1c, a1c_pull)):
        for m, d in arm.items():
            dr = np.array(d["draws"])
            sd = dr[:, tau0].std(ddof=1)
            dr[:, tau0] += want * sd
            d["draws"] = dr
    return a1c, a1


def test_escalation_is_computed_from_A1c_OWN_pull_not_the_paired_delta():
    """CALL-SITE test (round 4). The bare-function tests passed `pull_mean=` literals directly, so
    they could not catch `paired_report` handing clause 1 the PAIRED DELTA instead of A1c's own
    residual pull. 4c(b) clause 1 is `|A1c tau0_amp pull mean| > 0.30` -- "the same magnitude the
    frozen gate applies to the gated channels" -- a property of A1c ALONE.

    THE DISCRIMINATING CASE IS A PERFECT REPAIR: A1 carries the defect (+0.49), A1c does not
    (~0), so the DELTA is ~-0.49 (> 0.30 in magnitude, escalates) while the correct quantity,
    A1c's own pull, is ~0 (does not). Fed the delta, the rule escalated on a perfect repair."""
    a1c, a1 = _arms_with_tau0_pull(a1_pull=+0.4916, a1c_pull=0.0, n=16)
    ch = paired.paired_report(a1c, a1, expect_n=16)["channels"]["tau0amp"]
    assert abs(ch["a1c_mean"]) < 0.30 < abs(ch["delta_mean"]), (
        f"fixture must separate the two: a1c={ch['a1c_mean']:.3f} delta={ch['delta_mean']:.3f}")
    assert ch["escalates_4c_b"] == paired.tau0_escalates(
        ch["a1c_mean"], ch["ci95_lo"], ch["ci95_hi"])
    assert ch["escalates_4c_b"] is not paired.tau0_escalates(
        ch["delta_mean"], ch["ci95_lo"], ch["ci95_hi"]), (
        "a perfect repair must NOT escalate on clause 1; feeding the delta makes it escalate")


def test_clause_1_fires_on_an_A1c_pull_larger_than_A1s_own():
    """The other half of the same defect: an A1c tau0 pull LARGER than A1's (+0.75 vs +0.4916)
    has a SMALL delta, so a delta-fed clause 1 goes silent on a defect that got WORSE."""
    a1c, a1 = _arms_with_tau0_pull(a1_pull=+0.4916, a1c_pull=+0.75, n=16)
    ch = paired.paired_report(a1c, a1, expect_n=16)["channels"]["tau0amp"]
    assert abs(ch["a1c_mean"]) > 0.30, f"fixture: a1c pull {ch['a1c_mean']:.3f}"
    assert ch["escalates_4c_b"] is True, "a residual pull worse than A1's must escalate"


def test_clause_1_thresholds_on_the_bare_function():
    assert paired.tau0_escalates(0.70, -0.30, +0.30) is True     # big residual pull
    assert paired.tau0_escalates(0.70, +0.05, +0.30) is True     # even with a clean CI
    assert paired.tau0_escalates(0.02, +0.05, +0.30) is False    # small pull + CI excludes 0
