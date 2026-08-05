"""A1c DISPOSITION LOGIC, in code, with an exhaustive synthetic truth table (PI ruling 3d.4).

The pre-registration's section-4 table, the 4b roll-up and the 4c escalation existed only as
TEXT, and two disposition defects survived four review rounds inside that text:

  1. Row 6 OUTRANKED row 3 in the 4b ordering, so a channel with NON-UNIFORM ranks that also
     failed on sd was PROMOTED from "a pathology survives the correction" to "the most likely
     non-ideal outcome" -- adding a second failure improved the disposition. PI ruling 3d.2
     (2026-08-05): any non-uniform rank result on a gated channel escalates the ARM to at least
     row 3, whatever that channel's gate row; mean-failure rows 4/5 remain more severe.
  2. The row-1/row-2 split rested on a bare "rejected" with no rule. PI ruling 3d.3: row 1
     requires `ci_excludes_full_survival AND NOT ci_excludes_full_removal`; otherwise a
     gate-pass, rank-uniform, boundary-adjacent result is row 2.

Both rules are implemented by `scripts/a1c_disposition.py` and pinned here BEFORE any A1c
result exists. The table below is the frozen section-4 table; nothing here changes the gate.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_a1c_disposition.py -q
"""
import importlib
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

disp = importlib.import_module("scripts.a1c_disposition")

DP, BA, DF = "DECISIVE-PASS", "BOUNDARY-ADJACENT", "DECISIVE-FAIL"

# severity index: position in the 4b ordering, worst first
SEV = {r: i for i, r in enumerate(disp.ROW_SEVERITY_ORDER)}


def NR(surv, rem):
    return {"ci_excludes_full_survival": surv, "ci_excludes_full_removal": rem}


# --------------------------------------------------------------------- the qualifier (sec 3c)

def test_qualifier_reproduces_the_preregistered_decisive_thresholds_at_n48():
    """Section 2: a decisive pass at N=48 needs |mean| <= 0.0096 (t(0.975,47) = 2.0117,
    sd such that 2.0117*sd/sqrt(48) = 0.29037); impossible once sd > 1.0332."""
    sd = 1.0
    sem = sd / np.sqrt(48)
    # half-width 2.0117*0.14434 = 0.29037 -> decisive-pass iff |mean| <= 0.00963
    assert disp.qualifier(0.0090, sem, 48) == DP
    assert disp.qualifier(0.0100, sem, 48) == BA
    # at sd just above 1.0332 even mean = 0 is boundary-adjacent
    assert disp.qualifier(0.0, 1.0340 / np.sqrt(48), 48) == BA
    assert disp.qualifier(0.0, 1.0320 / np.sqrt(48), 48) == DP


def test_qualifier_decisive_fail_requires_excluding_compliance():
    sem = 1.0 / np.sqrt(48)                       # half-width 0.29037
    assert disp.qualifier(0.65, sem, 48) == DF    # 0.65 - 0.290 > 0.30
    assert disp.qualifier(0.55, sem, 48) == BA    # straddles
    assert disp.qualifier(-0.65, sem, 48) == DF   # sign-symmetric


# --------------------------------------------------------------- per-channel rows (sec 4 table)

def test_row1_requires_survival_excluded_AND_removal_not_excluded():
    """PI ruling 3d.3, verbatim. Everything else in the PASS/UNIFORM/BOUNDARY-ADJACENT cell is
    row 2 -- including the previously-unruled combos, and None (CI unreadable) conservatively."""
    base = dict(pull_mean=0.10, pull_sd=1.00, rank_ks_p=0.50, qual=BA)
    assert disp.channel_row(**base, no_repeat=NR(True, False)) == 1
    assert disp.channel_row(**base, no_repeat=NR(True, True)) == 2
    assert disp.channel_row(**base, no_repeat=NR(False, False)) == 2
    assert disp.channel_row(**base, no_repeat=NR(False, True)) == 2
    assert disp.channel_row(**base, no_repeat=NR(True, None)) == 2, \
        "an unreadable full-removal CI must not be treated as 'not excluded'"


def test_ns_with_no_hypothesis_test_takes_row1_on_pass_uniform_boundary():
    """Amendment 4 (PI ruling 1): n_s carries NO no-repeat criterion; it takes row 1 whenever
    gate PASS and ranks UNIFORM and qualifier BOUNDARY-ADJACENT."""
    assert disp.channel_row(pull_mean=-0.25, pull_sd=1.05, rank_ks_p=0.30, qual=BA,
                            no_repeat=None) == 1


def test_nonuniform_ranks_with_gate_pass_is_row3_dominating_row7():
    """Section 4 precedence: row 3 dominates row 7 (a decisive-pass interval does not rescue
    non-uniform ranks)."""
    for qual in (DP, BA):
        assert disp.channel_row(pull_mean=0.005, pull_sd=0.95, rank_ks_p=0.01, qual=qual,
                                no_repeat=NR(True, False)) == 3


def test_mean_fail_rows_4_and_5_split_on_the_qualifier():
    assert disp.channel_row(pull_mean=0.65, pull_sd=1.0, rank_ks_p=0.5, qual=DF,
                            no_repeat=NR(True, False)) == 4
    assert disp.channel_row(pull_mean=0.40, pull_sd=1.0, rank_ks_p=0.5, qual=BA,
                            no_repeat=NR(True, False)) == 5
    # ranks do not change the per-channel mean-fail row (the escalation is ARM-level)
    assert disp.channel_row(pull_mean=0.40, pull_sd=1.0, rank_ks_p=0.01, qual=BA,
                            no_repeat=NR(True, False)) == 5


def test_sd_only_fail_is_row6_whatever_the_ranks():
    """Row 6 (PI Q2): |mean| <= 0.30 AND sd > 1.1 is a DISPERSION-ONLY FAIL. The per-channel row
    is 6 even with non-uniform ranks -- ruling 3d.2 handles that at the ARM level, so the
    channel's own verbatim record stays what the frozen gate reports."""
    assert disp.channel_row(pull_mean=0.10, pull_sd=1.20, rank_ks_p=0.50, qual=BA,
                            no_repeat=NR(True, False)) == 6
    assert disp.channel_row(pull_mean=0.10, pull_sd=1.20, rank_ks_p=0.01, qual=BA,
                            no_repeat=NR(True, False)) == 6


def test_decisive_pass_uniform_is_row7():
    assert disp.channel_row(pull_mean=0.005, pull_sd=0.95, rank_ks_p=0.50, qual=DP,
                            no_repeat=NR(True, False)) == 7


def test_channel_row_refuses_inconsistent_qualifier():
    """A DECISIVE-PASS qualifier with a gate-failing sd is arithmetically impossible at N<=48
    (2.0117*1.1/sqrt(48) = 0.319 > 0.30). Reaching it means the inputs are corrupted; the row
    logic must fail loud, not pick a row."""
    with pytest.raises(ValueError):
        disp.channel_row(pull_mean=0.10, pull_sd=1.20, rank_ks_p=0.5, qual=DP,
                         no_repeat=NR(True, False))


# --------------------------------------------------------------- arm roll-up (4b + PI 3d.2)

def test_THE_DEFECT_sd_fail_with_nonuniform_ranks_escalates_the_arm_to_row3():
    """THE 3b-ITEM-1 SCENARIO. Before ruling 3d.2 this arm rolled up to row 6 ("most likely
    non-ideal outcome") BECAUSE the channel also failed on sd -- adding a second failure improved
    the disposition. Reachable: A1's own A_p had KS p 2.3e-4 at sd 0.97."""
    r = disp.arm_rollup(channel_rows={"ns": 6, "Ap": 1},
                        rank_uniform={"ns": False, "Ap": True},
                        tau0_escalates=False)
    assert r["arm_row"] == 3
    assert r["base_row"] == 6
    assert any("3d.2" in e or "rank" in e.lower() for e in r["escalations"])


def test_mean_fail_rows_stay_more_severe_than_the_escalation():
    """Ruling 3d.2, second sentence: rows 4/5 remain more severe. Escalation can only WORSEN a
    disposition, never improve one."""
    for base_row in (4, 5):
        r = disp.arm_rollup(channel_rows={"ns": base_row, "Ap": 1},
                            rank_uniform={"ns": False, "Ap": True},
                            tau0_escalates=True)
        assert r["arm_row"] == base_row, "escalation must never downgrade a mean-fail"


def test_tau0_escalation_lifts_a_healthy_looking_arm_to_row3():
    """The 4c hole: n_s row 1 + A_p row 1 + tau0_amp still pathological must NOT roll up to
    'the expected healthy outcome'."""
    r = disp.arm_rollup(channel_rows={"ns": 1, "Ap": 1},
                        rank_uniform={"ns": True, "Ap": True},
                        tau0_escalates=True)
    assert r["arm_row"] == 3 and r["base_row"] == 1


def test_healthy_arm_rolls_up_to_row1_with_no_escalations():
    r = disp.arm_rollup(channel_rows={"ns": 1, "Ap": 1},
                        rank_uniform={"ns": True, "Ap": True},
                        tau0_escalates=False)
    assert r["arm_row"] == 1 and r["escalations"] == []


def test_rollup_base_is_the_worst_channel_row_by_the_4b_ordering():
    # 4b ordering: 4 > 5 > 3 > 6 > 2 > 1 > 7
    cases = [({"ns": 6, "Ap": 2}, 6), ({"ns": 1, "Ap": 7}, 1),
             ({"ns": 3, "Ap": 6}, 3), ({"ns": 5, "Ap": 3}, 5)]
    for rows, want in cases:
        uni = {k: rows[k] not in (3,) for k in rows}
        r = disp.arm_rollup(channel_rows=rows, rank_uniform=uni, tau0_escalates=False)
        assert r["base_row"] == want, rows


def test_EXHAUSTIVE_truth_table_invariants():
    """Every consistent combination of per-channel rows, uniformity flags and the tau0 flag.
    Invariants (PI rulings 3d.2 + the 4b ordering):
      (a) base_row is the worst channel row;
      (b) the arm is never LESS severe than base_row;
      (c) any non-uniform gated channel, or a tau0 escalation, makes the arm at least row 3;
      (d) with no escalation trigger the arm IS base_row;
      (e) rows 4/5 are never downgraded.
    """
    rows = (1, 2, 3, 4, 5, 6, 7)
    n_checked = 0
    for r_ns, r_ap, tau0 in itertools.product(rows, rows, (False, True)):
        # consistent uniformity assignments: row 3 forces NON-UNIFORM; rows 1/2/7 force UNIFORM
        def uni_options(row):
            if row == 3:
                return (False,)
            if row in (1, 2, 7):
                return (True,)
            return (True, False)
        for u_ns, u_ap in itertools.product(uni_options(r_ns), uni_options(r_ap)):
            r = disp.arm_rollup(channel_rows={"ns": r_ns, "Ap": r_ap},
                                rank_uniform={"ns": u_ns, "Ap": u_ap},
                                tau0_escalates=tau0)
            base, arm = r["base_row"], r["arm_row"]
            worst = min((r_ns, r_ap), key=lambda x: SEV[x])
            assert base == worst                                            # (a)
            assert SEV[arm] <= SEV[base]                                    # (b)
            if (not u_ns) or (not u_ap) or tau0:
                assert SEV[arm] <= SEV[3]                                   # (c)
            else:
                assert arm == base                                          # (d)
            if base in (4, 5):
                assert arm == base                                          # (e)
            n_checked += 1
    assert n_checked >= 200, "the sweep must actually be exhaustive"


# --------------------------------------------------------------- dispose(): wiring the inputs

def _leg(ns=(-0.10, 1.00), ap=(-0.12, 1.02), ns_ks=0.40, ap_ks=0.35, tau_ks=0.60, n=48):
    """A minimal gate-JSON leg dict, as analyze_sbc_perleg writes it."""
    def pulls(mean, sd):
        return {"mean": mean, "std": sd, "sem": sd / np.sqrt(n), "n": n}
    def gate(mean, sd):
        return "PASS" if (abs(mean) <= 0.30 and sd <= 1.1) else "FAIL"
    return {
        "n_mocks": n,
        "gate_ns": gate(*ns), "gate_Ap": gate(*ap),
        "pulls": {"ns": pulls(*ns), "Ap": pulls(*ap)},
        "rank_uniformity": {
            "ns": {"ks_p": ns_ks, "verdict": "UNIFORM" if ns_ks > 0.05 else "NON-UNIFORM"},
            "Ap": {"ks_p": ap_ks, "verdict": "UNIFORM" if ap_ks > 0.05 else "NON-UNIFORM"},
            "tau0amp": {"ks_p": tau_ks,
                        "verdict": "UNIFORM" if tau_ks > 0.05 else "NON-UNIFORM"},
        },
    }


def _paired(ap_flags=(True, False), tau_esc=False):
    return {"channels": {
        "ns": {"ci_excludes_full_survival": False, "ci_excludes_full_removal": None},
        "Ap": {"ci_excludes_full_survival": ap_flags[0],
               "ci_excludes_full_removal": ap_flags[1]},
        "tau0amp": {"ci_excludes_full_survival": True, "ci_excludes_full_removal": False,
                    "escalates_4c_b": tau_esc},
    }, "gated": False}


def test_dispose_healthy_scenario_reads_row1():
    d = disp.dispose(_leg(), _paired(), expect_n=48)
    assert d["channel_rows"] == {"ns": 1, "Ap": 1}
    assert d["arm"]["arm_row"] == 1


def test_dispose_wires_tau0_rank_limb_from_the_arm_itself():
    """4c limb (a) reads A1c's OWN tau0amp ranks from the gate JSON -- non-uniform there must
    escalate even when the paired pull limb is quiet."""
    d = disp.dispose(_leg(tau_ks=0.01), _paired(tau_esc=False), expect_n=48)
    assert d["tau0"]["limb_a_ranks"] is True
    assert d["arm"]["arm_row"] == 3


def test_dispose_wires_tau0_pull_limb_from_the_paired_report():
    d = disp.dispose(_leg(), _paired(tau_esc=True), expect_n=48)
    assert d["tau0"]["limb_b_pull"] is True
    assert d["arm"]["arm_row"] == 3


def test_dispose_the_3b1_scenario_end_to_end():
    """n_s sd-only-fail (1.2) WITH non-uniform n_s ranks: per-channel row 6, arm row 3."""
    d = disp.dispose(_leg(ns=(0.10, 1.20), ns_ks=0.001), _paired(), expect_n=48)
    assert d["channel_rows"]["ns"] == 6
    assert d["arm"]["arm_row"] == 3


def test_dispose_REFUSES_a_partial_arm_without_the_explicit_flag():
    """Completion condition (4c): the verdict is computed ONLY at 48/48. A partial arm must
    refuse by default and be loudly labeled PARTIAL when explicitly allowed."""
    with pytest.raises(disp.PartialArmError):
        disp.dispose(_leg(n=40), _paired(), expect_n=48)
    d = disp.dispose(_leg(n=40), _paired(), expect_n=48, allow_partial=True)
    assert d["partial"] is True


def test_dispose_REFUSES_on_gate_string_inconsistency():
    """The JSON's own gate strings are recomputed from the pulls; a mismatch means the inputs
    were edited or mixed and no disposition may be read."""
    leg = _leg()
    leg["gate_ns"] = "FAIL"                        # contradicts pulls (-0.10, 1.00)
    with pytest.raises(ValueError):
        disp.dispose(leg, _paired(), expect_n=48)


def test_dispose_never_emits_extension_or_promotion_language():
    """Every row: no autonomous extension, no promotion out of DIAGNOSTIC (PI acts)."""
    d = disp.dispose(_leg(ns=(0.5, 1.0)), _paired(), expect_n=48)
    txt = disp.format_disposition(d).lower()
    assert "extend" not in txt or "no extension" in txt or "not extend" in txt
    assert "diagnostic" in txt
