"""Analyzer Tier-A FLAG disposition for the resolution_oos arm (Task 2C). ADVERSARIAL basis members
(bres1/bres2/bstar) get the SAME [0.30,0.50]sigma FLAG band as lls_excess (a documented, not
auto-failed, budget); bres_real (the measured, realistic residual -- Tier-R) keeps the HARD 0.30
gate. Non-resolution_oos arms (metal_misspec/resolution/lls_excess/metal_matched) are unchanged."""
import os
import pickle

import pytest

from scripts.analyze_dnuis_bias import GATE, LLS_BUDGET, _dnuis_verdict, load_shards


def test_dnuis_verdict_resolution_oos_adversarial_members_flag_band():
    """bres1/bres2/bstar sit in [GATE, LLS_BUDGET) -> FLAG, mirroring lls_excess."""
    for member in ("bres1", "bres2", "bstar"):
        v = _dnuis_verdict("resolution_oos", member, 0.40)
        assert v.startswith("FLAG"), f"member={member} verdict={v!r}"
        assert f"{LLS_BUDGET:.2f}" in v


def test_dnuis_verdict_resolution_oos_bres_real_hard_gate():
    """bres_real (Tier-R, realistic) is NOT flagged -- it keeps the hard GATE like any other arm."""
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.40) == "FAIL"
    assert _dnuis_verdict("resolution_oos", "bres_real", 0.10) == "PASS"


def test_dnuis_verdict_resolution_oos_above_lls_budget_still_fails():
    """Even an adversarial member hard-FAILs once ub clears the LLS_BUDGET ceiling."""
    for member in ("bres1", "bres2", "bstar"):
        assert _dnuis_verdict("resolution_oos", member, 0.60) == "FAIL"


def test_dnuis_verdict_other_arms_unchanged():
    """Golden-safety: non-resolution_oos arms keep their EXISTING dispositions."""
    assert _dnuis_verdict("metal_misspec", None, 0.10) == "PASS"
    assert _dnuis_verdict("metal_misspec", None, 0.40) == "FAIL"
    assert _dnuis_verdict("resolution", None, 0.10) == "PASS"
    assert _dnuis_verdict("resolution", None, 0.40) == "FAIL"           # in-span resolution stays HARD-gated
    assert _dnuis_verdict("lls_excess", None, 0.40).startswith("FLAG")  # pre-existing FLAG band, unaffected
    assert _dnuis_verdict("lls_excess", None, 0.60) == "FAIL"
    assert _dnuis_verdict("metal_matched", None, 0.10) == "PASS"


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
