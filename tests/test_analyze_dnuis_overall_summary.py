"""PR#14 panel FIX 3: the OVERALL summary line must acknowledge a FLAGGED non-LLS arm.

Before the fix, the summary flipped only on ``any_fail`` -- a flagged resolution_oos non-LLS arm
(0.30<=ub<0.50, e.g. a bstar member) left ``any_fail`` False and the summary printed "all non-LLS
arms within the confidence-bound bias gate", which over-claims a clean pass when the arm actually
sits inside the documented adversarial FLAG budget. This does NOT change any gate decision (FAIL is
still FAIL, PASS-with-no-flags is still a plain PASS) -- it only corrects a wording gap in the
FLAG-but-not-FAIL case.
"""
from scripts.analyze_dnuis_bias import _overall_summary


def test_overall_summary_fail_wins_over_flag():
    s = _overall_summary(any_fail=True, any_flag=True)
    assert s.startswith("FAIL")


def test_overall_summary_plain_pass_no_flags():
    """The original wording is preserved byte-for-byte when nothing was flagged (no change to the
    common case)."""
    s = _overall_summary(any_fail=False, any_flag=False)
    assert s == "PASS — all non-LLS arms within the confidence-bound bias gate."


def test_overall_summary_flagged_non_lls_arm_is_acknowledged():
    """FIX 3: a flagged (but not failed) non-LLS arm must NOT be summarized as a clean 'all within
    the gate' pass -- the wording must say something distinct from the plain-pass string."""
    s = _overall_summary(any_fail=False, any_flag=True)
    assert s != "PASS — all non-LLS arms within the confidence-bound bias gate."
    assert s.startswith("PASS")
    assert "FLAG" in s
