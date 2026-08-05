#!/usr/bin/env python3
"""A1c DISPOSITION -- the pre-registered section-4 table, 4b roll-up and 4c escalation, in code.

WHY THIS EXISTS. The disposition logic lived only in pre-registration TEXT, and two defects
survived four review rounds inside that text (caught 2026-07-29, ruled 2026-08-05):

  1. Row 6 OUTRANKED row 3 in the 4b ordering (4 > 5 > 3 > 6 > 2 > 1 > 7), so a channel with
     NON-UNIFORM ranks that also failed on sd matched row 6 (row 3 requires gate PASS) and was
     thereby PROMOTED from "a pathology survives the correction" to "the most likely non-ideal
     outcome". Adding a second failure improved the disposition. Reachable: A1's own A_p had
     KS p 2.3e-4 at sd 0.97.
  2. The row-1 / row-2 split rested on a bare "rejected" with no rule, so the arm's headline
     label would have been a post-hoc judgement inside a pre-registration.

TWO PI RULINGS (3d, 2026-08-05) BIND THIS MODULE:

  3d.2  Any non-uniform rank result on a GATED channel escalates the ARM to at least row 3,
        regardless of whether that channel also has an sd-only failure. The escalation is
        ARM-level, in the 4c idiom -- the per-channel row stays exactly what the frozen table
        assigns, so the channel's verbatim record is untouched. Mean-failure rows 4/5 remain
        more severe and are never downgraded.
  3d.3  Row 1 requires `ci_excludes_full_survival AND NOT ci_excludes_full_removal`; otherwise
        a gate-pass, rank-uniform, boundary-adjacent result is row 2. An unreadable CI (None)
        is conservatively row 2.

NOTHING HERE CHANGES THE GATE. The frozen gate (|mean| <= 0.30 AND sd <= 1.1; ranks KS p >
0.05) is computed by analyze_sbc_perleg.py from A1c alone; this module only maps its outputs,
plus the paired report's CI flags, onto the pre-registered rows. Escalations can only WORSEN a
disposition. In every row: NO autonomous extension to N=96, NO promotion out of DIAGNOSTIC --
both are PI acts.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/a1c_disposition.py GATE_JSON A1C_DIR A1_DIR [EXPECT_N]
"""
import json
import sys

import numpy as np

GATE_MEAN = 0.30                    # the frozen gate, unchanged
GATE_SD = 1.10
RANK_ALPHA = 0.05

# The 4b ordering, worst first. Note row 6 sits BELOW row 3 here by design -- that is the frozen
# text -- and ruling 3d.2 corrects the consequence at the ARM level rather than by re-ordering,
# which would let a non-uniform mean-FAIL escape rows 4/5.
ROW_SEVERITY_ORDER = (4, 5, 3, 6, 2, 1, 7)
_SEV = {r: i for i, r in enumerate(ROW_SEVERITY_ORDER)}

DP, BA, DF = "DECISIVE-PASS", "BOUNDARY-ADJACENT", "DECISIVE-FAIL"

ROW_MEANING = {
    1: "the expected healthy outcome (residual below 0.30 NOT established at this N)",
    2: "UNDER-RESOLVED: consistent both with a corrected arm and an unchanged one",
    3: "a pathology survives the correction: escalate; do not treat as a pass",
    4: "the correction did not fix it (the PI-#9-decision-4 reopening condition)",
    5: "FAIL per the frozen gate, resolution stated; PI decision required",
    6: "DISPERSION-ONLY FAIL (PI Q2): verbatim record; NOT evidence of mean bias; "
       "returned for PI disposition, no post-hoc threshold reinterpretation",
    7: "decisive pass: surprising (sec 2); double-check for an sd anomaly first",
}


class PartialArmError(RuntimeError):
    """The completion condition (4c): the verdict is computed ONLY at 48/48 landed."""


def qualifier(mean, sem, n):
    """Section-3c resolution qualifier, with t(0.975, n-1) recomputed from the realized n
    (the completion condition requires exactly that when n != 48)."""
    from scipy import stats as st
    if not (np.isfinite(mean) and np.isfinite(sem) and n > 1):
        raise ValueError(f"unreadable qualifier inputs: mean={mean} sem={sem} n={n}")
    half = float(st.t.ppf(0.975, n - 1)) * float(sem)
    if abs(mean) + half <= GATE_MEAN:
        return DP
    if abs(mean) - half > GATE_MEAN:
        return DF
    return BA


def channel_row(*, pull_mean, pull_sd, rank_ks_p, qual, no_repeat):
    """The frozen section-4 table, matched top to bottom, first match wins.

    `no_repeat` is None for n_s (PI ruling: NO hypothesis test against a defect magnitude --
    which is NOT a claim that no defect magnitude exists; A1's matched pull -0.2532 is context,
    never a threshold) or a dict with the two CI flags for A_p.
    """
    mean_fail = abs(float(pull_mean)) > GATE_MEAN
    sd_fail = float(pull_sd) > GATE_SD
    gate_pass = not (mean_fail or sd_fail)
    uniform = float(rank_ks_p) > RANK_ALPHA

    # A DECISIVE-PASS qualifier is arithmetically impossible with a gate-failing sd at N <= 48
    # (2.0117*1.1/sqrt(48) = 0.319 > 0.30). Seeing it means the inputs are corrupted.
    if sd_fail and qual == DP:
        raise ValueError(f"inconsistent inputs: sd={pull_sd} > {GATE_SD} with a {DP} qualifier")

    # rows 1 and 2 (PASS / UNIFORM / BOUNDARY-ADJACENT), split by PI ruling 3d.3
    if gate_pass and uniform and qual == BA:
        if no_repeat is None:
            return 1                       # n_s: amendment-4 collapse, no no-repeat criterion
        row1 = (no_repeat.get("ci_excludes_full_survival") is True
                and no_repeat.get("ci_excludes_full_removal") is False)
        return 1 if row1 else 2            # None (unreadable CI) lands in row 2, conservatively
    if gate_pass and not uniform:
        return 3                           # dominates row 7 (precedence rule, section 4)
    if mean_fail and qual == DF:
        return 4
    if mean_fail:
        return 5
    if sd_fail:
        return 6                           # any ranks: the ARM-level 3d.2 rule handles those
    if gate_pass and uniform and qual == DP:
        return 7
    raise ValueError(f"unreachable cell: mean={pull_mean} sd={pull_sd} "
                     f"ks_p={rank_ks_p} qual={qual}")


def arm_rollup(channel_rows, rank_uniform, tau0_escalates):
    """The 4b roll-up plus the two ARM-level escalations (PI 3d.2 and section 4c).

    base_row = the WORST channel row by the 4b ordering. Escalations lift the arm to AT LEAST
    row 3 and can only WORSEN the disposition: a base of 4 or 5 is already more severe and is
    never touched.
    """
    base = min(channel_rows.values(), key=lambda r: _SEV[r])
    row, esc = base, []
    nonuni = sorted(k for k, u in rank_uniform.items() if not u)
    if nonuni and _SEV[row] > _SEV[3]:
        row = 3
        esc.append(f"PI 3d.2: non-uniform ranks on gated channel(s) {','.join(nonuni)} "
                   f"escalate the ARM to at least row 3 (the per-channel rows are unchanged)")
    if tau0_escalates and _SEV[row] > _SEV[3]:
        row = 3
        esc.append("section 4c: tau0_amp escalation (calibration warning; NOT a cosmology-bias "
                   "claim; licenses NO extension)")
    return {"arm_row": row, "base_row": base, "escalations": esc}


def dispose(leg, paired, expect_n=48, allow_partial=False):
    """Map one gate-JSON leg dict + the paired report onto the pre-registered disposition.

    Refuses a partial arm by default (completion condition: the verdict is computed ONLY at
    expect_n landed; with `allow_partial` the result is loudly labeled PARTIAL and t is
    recomputed from the realized n by `qualifier`).
    """
    n = int(leg["n_mocks"])
    if n != int(expect_n) and not allow_partial:
        raise PartialArmError(
            f"{n} mocks landed, {expect_n} expected: the verdict is computed ONLY at "
            f"{expect_n}/{expect_n} (pre-registration 4c). Pass allow_partial=True only for a "
            f"deliberately-labeled PARTIAL read; do not compare a partial arm against A1's "
            f"48-mock subset without re-matching the mock indices.")

    rows, quals, uni = {}, {}, {}
    for ch, gate_key in (("ns", "gate_ns"), ("Ap", "gate_Ap")):
        p = leg["pulls"][ch]
        mean, sd, sem, pn = p["mean"], p["std"], p["sem"], int(p["n"])
        # cross-check the JSON's own gate string against the frozen thresholds: a mismatch
        # means the inputs were edited or mixed, and no disposition may be read from them
        want = "PASS" if (abs(mean) <= GATE_MEAN and sd <= GATE_SD) else "FAIL"
        got = leg.get(gate_key)
        if got not in (want,):
            raise ValueError(f"gate-string inconsistency on {ch}: JSON says {got!r}, the "
                             f"frozen thresholds on its own pulls say {want!r}")
        ks_p = leg["rank_uniformity"][ch]["ks_p"]
        quals[ch] = qualifier(mean, sem, pn)
        uni[ch] = float(ks_p) > RANK_ALPHA
        no_rep = None
        if ch == "Ap":
            pc = paired["channels"]["Ap"]
            no_rep = {"ci_excludes_full_survival": pc.get("ci_excludes_full_survival"),
                      "ci_excludes_full_removal": pc.get("ci_excludes_full_removal")}
        rows[ch] = channel_row(pull_mean=mean, pull_sd=sd, rank_ks_p=ks_p,
                               qual=quals[ch], no_repeat=no_rep)

    # section 4c, both limbs: (a) A1c's OWN tau0_amp ranks; (b) the paired-report pull limb.
    # HARD INDEX, not .get (round-6 panel): this limb has carried a defect three rounds running,
    # and a schema drift that renames the key must be a KeyError here, never a quiet False.
    tau_ks = leg["rank_uniformity"]["tau0amp"]["ks_p"]
    limb_a = float(tau_ks) <= RANK_ALPHA
    limb_b = bool(paired["channels"]["tau0amp"]["escalates_4c_b"])
    tau0 = {"limb_a_ranks": limb_a, "limb_b_pull": limb_b, "ks_p": float(tau_ks)}

    arm = arm_rollup(rows, uni, limb_a or limb_b)
    return {"channel_rows": rows, "qualifiers": quals, "rank_uniform": uni,
            "tau0": tau0, "arm": arm, "n_mocks": n,
            "partial": n != int(expect_n)}


def format_disposition(d):
    """Human-readable disposition. States the binding constraints in every output: no
    autonomous extension, no promotion out of DIAGNOSTIC, escalation != cosmology-bias claim."""
    L = ["", "=" * 92,
         "A1c DISPOSITION (pre-registration sec 4 / 4b / 4c + PI rulings 3d.2-3d.3) -- "
         "artifacts stay DIAGNOSTIC",
         "=" * 92]
    if d.get("partial"):
        L.append("  *** PARTIAL ARM: fewer mocks than pre-registered -- NOT a verdict. ***")
    for ch, row in d["channel_rows"].items():
        L.append(f"  {ch:8s} row {row}  [{d['qualifiers'][ch]}, ranks "
                 f"{'UNIFORM' if d['rank_uniform'][ch] else 'NON-UNIFORM'}]  "
                 f"-> {ROW_MEANING[row]}")
    t = d["tau0"]
    L.append(f"  tau0_amp  4c limb (a) ranks: {'FIRES' if t['limb_a_ranks'] else 'quiet'} "
             f"(KS p={t['ks_p']:.4g});  limb (b) pull: "
             f"{'FIRES' if t['limb_b_pull'] else 'quiet'}"
             "   [calibration warning only; not a gate; licenses NO extension]")
    a = d["arm"]
    L.append(f"  ARM: row {a['arm_row']} (base {a['base_row']})  -> {ROW_MEANING[a['arm_row']]}")
    for e in a["escalations"]:
        L.append(f"    escalated: {e}")
    L.append("  Binding in every row: NO autonomous extension to N=96; NO promotion out of "
             "DIAGNOSTIC; both are PI acts.")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("usage: a1c_disposition.py GATE_JSON A1C_DIR A1_DIR [EXPECT_N]")
    from scripts.analyze_a1c_paired import load_arm, paired_report
    _gate = json.load(open(sys.argv[1]))
    _leg = _gate["legs"]["eBOSS"]
    if _leg.get("status") == "no_pkls_yet":
        raise SystemExit("the gate JSON carries no eBOSS leg yet")
    _n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    # paired_report RAISES PairingError if any negative-control conjunct fails: the disposition
    # is unreachable unless the arms are the populations they claim to be
    _rep = paired_report(load_arm(sys.argv[2]), load_arm(sys.argv[3]), expect_n=_n)
    print(format_disposition(dispose(_leg, _rep, expect_n=_n)))
