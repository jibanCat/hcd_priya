"""Analyzer Tier-A FLAG disposition for the resolution_oos arm (Task 2C). ADVERSARIAL basis members
(bres1/bres2/bstar) get the SAME [0.30,0.50]sigma FLAG band as lls_excess (a documented, not
auto-failed, budget); bres_real (the measured, realistic residual -- Tier-R) keeps the HARD 0.30
gate. Non-resolution_oos arms (metal_misspec/resolution/lls_excess/metal_matched) are unchanged.

PR#14 panel FIX 2: the FLAG band is n_s-ONLY (2026-07-07-gate-b-consolidated-review.md gate spec).
A_p must HARD-gate at 0.30 even for lls_excess / resolution_oos-adversarial cells -- a breach there
needs a PI-signed waiver, not an auto-FLAG. ``_dnuis_verdict`` takes a ``param`` argument; every
call below is updated to pass it explicitly."""
import os
import pickle

import pytest

from scripts.analyze_dnuis_bias import GATE, LLS_BUDGET, _dnuis_verdict, load_shards


def test_dnuis_verdict_resolution_oos_adversarial_members_flag_band():
    """bres1/bres2/bstar sit in [GATE, LLS_BUDGET) -> FLAG for n_s, mirroring lls_excess."""
    for member in ("bres1", "bres2", "bstar"):
        v = _dnuis_verdict("resolution_oos", member, 0.40, "ns")
        assert v.startswith("FLAG"), f"member={member} verdict={v!r}"
        assert f"{LLS_BUDGET:.2f}" in v


def test_dnuis_verdict_flag_band_is_ns_only():
    """FIX 2: the FLAG band applies ONLY to param=='ns'. For A_p (param='Ap') a resolution_oos
    adversarial cell HARD-gates at GATE like any other arm -- no auto-FLAG, a breach needs a
    PI-signed waiver."""
    v_ap_fail = _dnuis_verdict("resolution_oos", "bstar", 0.40, "Ap")
    assert v_ap_fail == "FAIL", f"A_p ub=0.40 must hard-FAIL (not FLAG), got {v_ap_fail!r}"
    v_ns_flag = _dnuis_verdict("resolution_oos", "bstar", 0.40, "ns")
    assert v_ns_flag.startswith("FLAG"), f"n_s ub=0.40 must FLAG, got {v_ns_flag!r}"
    v_ap_pass = _dnuis_verdict("resolution_oos", "bstar", 0.25, "Ap")
    assert v_ap_pass == "PASS", f"A_p ub=0.25 must PASS, got {v_ap_pass!r}"


def test_dnuis_verdict_lls_excess_flag_band_is_ns_only():
    """FIX 2 applies the same n_s-only restriction to lls_excess (the other FLAG-band arm)."""
    assert _dnuis_verdict("lls_excess", None, 0.40, "Ap") == "FAIL"
    assert _dnuis_verdict("lls_excess", None, 0.40, "ns").startswith("FLAG")


def test_dnuis_verdict_resolution_oos_bres_real_hard_gate():
    """bres_real (Tier-R, realistic) is NOT flagged -- it keeps the hard GATE like any other arm,
    for BOTH params."""
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.40, "ns") == "FAIL"
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.10, "ns") == "PASS"
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.40, "Ap") == "FAIL"
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.10, "Ap") == "PASS"


def test_dnuis_verdict_resolution_oos_above_lls_budget_still_fails():
    """Even an adversarial member hard-FAILs once ub clears the LLS_BUDGET ceiling (n_s)."""
    for member in ("bres1", "bres2", "bstar"):
        assert _dnuis_verdict("resolution_oos", member, 0.60, "ns") == "FAIL"


def test_dnuis_verdict_resolution_oos_pass_branch():
    """FIX 7: a flag-arm member (bres1/bres2/bstar) with ub < GATE returns PASS -- the shared PASS
    return, exercised here for a FLAG-band arm (previously only hit by non-flag arms)."""
    for member in ("bres1", "bres2", "bstar"):
        assert _dnuis_verdict("resolution_oos", member, 0.10, "ns") == "PASS"
        assert _dnuis_verdict("resolution_oos", member, 0.10, "Ap") == "PASS"


def test_dnuis_verdict_other_arms_unchanged():
    """Golden-safety: non-resolution_oos arms keep their EXISTING dispositions."""
    assert _dnuis_verdict("metal_misspec", None, 0.10, "ns") == "PASS"
    assert _dnuis_verdict("metal_misspec", None, 0.40, "ns") == "FAIL"
    assert _dnuis_verdict("resolution", None, 0.10, "ns") == "PASS"
    assert _dnuis_verdict("resolution", None, 0.40, "ns") == "FAIL"           # in-span resolution stays HARD-gated
    assert _dnuis_verdict("lls_excess", None, 0.40, "ns").startswith("FLAG")  # pre-existing FLAG band, unaffected
    assert _dnuis_verdict("lls_excess", None, 0.60, "ns") == "FAIL"
    assert _dnuis_verdict("metal_matched", None, 0.10, "ns") == "PASS"


def _write_fake_shard(path, *, arm, survey, treatment, oos_member, n_records=1):
    rec = {"n_div": 0}
    with open(path, "wb") as f:
        pickle.dump(dict(arm=arm, treatment=treatment, survey=survey,
                         clean_per_mock=[rec] * n_records, inj_per_mock=[rec] * n_records,
                         meta=dict(b_res_oos_member=oos_member)), f)


def test_load_shards_exposes_oos_member_from_meta(tmp_path):
    """load_shards must surface meta['b_res_oos_member'] per (arm,treatment,survey) group so the
    analyzer can pick the FLAG vs hard-gate disposition without re-reading pkls."""
    _write_fake_shard(tmp_path / "resolution_oos_b_desi_shard_000.pkl",
                      arm="resolution_oos", survey="desi", treatment="b", oos_member="bstar")
    _write_fake_shard(tmp_path / "resolution_b_desi_shard_000.pkl",
                      arm="resolution", survey="desi", treatment="b", oos_member=None)
    groups = load_shards(str(tmp_path))
    assert set(groups) == {("resolution_oos", "b", "desi"), ("resolution", "b", "desi")}
    _cl, _inj, _nd, om = groups[("resolution_oos", "b", "desi")]
    assert om[0] == "bstar"
    _cl2, _inj2, _nd2, om2 = groups[("resolution", "b", "desi")]
    assert om2[0] is None
