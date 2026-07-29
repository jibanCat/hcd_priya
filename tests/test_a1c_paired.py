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


def _mock(idx, *, corrected, seed=0, truth_shift=None, L=40):
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
        lo, hi, kind = BRACKETS[nm]
        # a DRAWN truth: mid-support for the bracketed metal nodes, a finite NON-ZERO value for
        # the two Normal-prior f_res sites (whose pinned value is exactly the prior centre 0)
        drawn = (lo + hi) / 2.0 if kind == "bracket" else 0.0049
        se[nm] = {"draws": dr.normal(0.01, 0.002, L),
                  "truth": float(np.nan) if not corrected else float(drawn)}
    cfg = dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", leg="eBOSS",
               metal_prior="flatlog2node", metal_selfdraw=corrected, fres_selfdraw=corrected)
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
    rep = paired.paired_report(a1c, a1)
    assert set(rep["channels"]) == {"ns", "Ap", "tau0amp"}


def test_paired_report_gives_delta_ci_rho_and_a_distribution_free_backup():
    a1c, a1 = _arms(n=16)
    ch = paired.paired_report(a1c, a1)["channels"]["ns"]
    for k in ("delta_mean", "ci95_lo", "ci95_hi", "t", "p", "wilcoxon_p", "rho", "n"):
        assert k in ch, f"missing {k}"
    assert ch["n"] == 16
    assert ch["ci95_lo"] <= ch["delta_mean"] <= ch["ci95_hi"]
    assert -1.0 <= ch["rho"] <= 1.0


def test_paired_report_compares_the_CI_against_the_defect_magnitude_not_a_bare_p():
    """Amendment 3: rejecting H0: delta=0 licenses 'the arms differ', NOT 'the failure does not
    reproduce'. The report must carry both comparisons explicitly."""
    a1c, a1 = _arms(n=16)
    ch = paired.paired_report(a1c, a1)["channels"]["Ap"]
    assert "reference_delta" in ch
    assert "ci_excludes_zero" in ch and "ci_excludes_full_survival" in ch


def test_paired_report_REFUSES_when_the_negative_control_fails():
    """'Any failure means the arms are not the populations they claim to be: STOP, do not read a
    number.' That must be enforced by the code, not by the reader's discipline."""
    a1 = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    noop = {i: _mock(i, corrected=False, seed=1) for i in range(12)}
    with pytest.raises(paired.PairingError):
        paired.paired_report(noop, a1, verify=True, expect_n=12)


def test_reference_deltas_are_the_MATCHED_subset_never_a_full_96_constant():
    """The campaign's single most important methodological point, re-checked here: A1c runs mocks
    0-47, so the reference must be A1's own mocks-0-47 value."""
    assert paired.REFERENCE_DELTA["Ap"] == pytest.approx(0.4046, abs=1e-4)
    assert paired.REFERENCE_DELTA["tau0amp"] == pytest.approx(0.4916, abs=1e-4)
    assert paired.REFERENCE_DELTA["ns"] is None, "n_s has no defective-arm reference; report only"


def test_paired_report_is_not_a_gate():
    """PI #9 forbids a gate-definition change. No channel may carry a pass/fail verdict, and the
    report must declare itself ungated. (The prose note is exempt: it necessarily discusses A1's
    *gate pass*, which is the very thing PI ruling 1 says is not evidence of an unaffected n_s.)"""
    a1c, a1 = _arms(n=16)
    rep = paired.paired_report(a1c, a1)
    assert rep["gated"] is False
    for k, c in rep["channels"].items():
        assert "verdict" not in c and "gate" not in c, f"{k} must carry no verdict"
        blob = " ".join(str(v) for v in c.values()).lower()
        assert "pass" not in blob and "fail" not in blob
