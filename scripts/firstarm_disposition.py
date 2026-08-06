#!/usr/bin/env python3
"""FIRST-ARM disposition roll-up (A3c pre-registration section 4; serves A2c unchanged).

Reads the deployed per-leg gate JSON (analyze_sbc_perleg output) + the first-arm selfdraw
JSON (analyze_firstarm_selfdraw output) and applies the pre-registered rows:

  row 4  any gated-channel MEAN fail (|mean| > 0.30)                    (worst)
  row 3  any gated-channel NON-UNIFORM ranks (PI 3d.2), OR the unpaired 4c limbs on
         tau0_amp (non-uniform ranks or |pull| > 0.30)                  (escalation -> PI)
  row 6  an sd-only gate failure (Q2: dispersion-only FAIL, verbatim)   (-> PI)
  row 2  gates/ranks/tau0 healthy but the self-drawn HEALTH BLOCK fails (-> PI question)
  row 1  everything healthy: the expected healthy outcome (promotion = PI act)

Precedence 4 > 3 > 6 > 2 > 1 (mean fail is worst; rank non-uniformity escalates even when
the same channel also fails on sd -- the row-6-outranks-row-3 defect must not recur).
Row 5 is reserved/unused for a first arm (the A1c rows 4/5 split was repair-specific).

REFUSES (DispositionError) on: a failed/absent selfdraw conjunct record; a partial arm
(n mismatch); a survey mismatch between the two inputs; missing channels; a gate string
inconsistent with the criteria recomputed from the same JSON's pulls; a health-rank
cross-check mismatch against the gate JSON's own repaired_sectors (two independent
implementations of the same statistic must agree).

Usage:
  ... scripts/firstarm_disposition.py GATE_JSON SELFDRAW_JSON
"""
import json
import sys

MEAN_GATE = 0.30
SD_GATE = 1.1
TAU0_PULL_LIMB = 0.30
XCHECK_TOL = 1e-3

GATED = ("ns", "Ap")


class DispositionError(RuntimeError):
    """Partial or inconsistent inputs: no disposition is emitted."""


def _fail(msg):
    raise DispositionError(msg + " -- REFUSING to emit a disposition.")


def run(gate_json_path, selfdraw_json_path):
    with open(gate_json_path) as f:
        gate = json.load(f)
    with open(selfdraw_json_path) as f:
        sd = json.load(f)

    if sd.get("conjuncts_ok") is not True:
        _fail(f"selfdraw conjunct record is {sd.get('conjuncts_ok')!r}, not True")
    survey = sd.get("survey")
    leg = (gate.get("legs") or {}).get(survey)
    if leg is None:
        _fail(f"survey {survey!r} absent from the gate JSON legs "
              f"{sorted((gate.get('legs') or {}))}")
    if int(leg.get("n_mocks", -1)) != int(sd.get("n", -2)):
        _fail(f"n_mocks mismatch: gate {leg.get('n_mocks')} vs selfdraw {sd.get('n')} "
              f"(partial arm?)")
    gsurvey = (leg.get("prior") or {}).get("survey")
    if gsurvey != survey:
        _fail(f"survey mismatch: gate leg prior says {gsurvey!r}, selfdraw says {survey!r}")

    pulls = leg.get("pulls") or {}
    ranks = leg.get("rank_uniformity") or {}
    for ch in GATED + ("tau0amp",):
        if ch not in pulls or ch not in ranks:
            _fail(f"channel {ch} missing from the gate JSON")

    # recompute the gate criteria and refuse an inconsistent record
    crit = {}
    for ch in GATED:
        mean, std = float(pulls[ch]["mean"]), float(pulls[ch]["std"])
        verdict = str(ranks[ch]["verdict"])
        mean_ok, sd_ok = abs(mean) <= MEAN_GATE, std <= SD_GATE
        expect = "PASS" if (mean_ok and sd_ok) else "FAIL"
        got = str(leg.get(f"gate_{ch}"))
        if got != expect:
            _fail(f"gate string inconsistent for {ch}: JSON says {got!r} but the same "
                  f"JSON's pulls (mean {mean:+.4f}, sd {std:.4f}) imply {expect!r}")
        grank = str(leg.get(f"gate_rank_{ch}"))
        if grank != verdict:
            _fail(f"gate string inconsistent for {ch} ranks: {grank!r} vs rank_uniformity "
                  f"verdict {verdict!r}")
        crit[ch] = dict(mean=mean, std=std, mean_ok=mean_ok, sd_ok=sd_ok,
                        rank_ok=(verdict == "UNIFORM"))

    # cross-check the health ranks against the gate JSON's own repaired_sectors
    gsites = ((leg.get("repaired_sectors") or {}).get("sites") or {})
    for k, h in (sd.get("health") or {}).items():
        if k in gsites and "rank_ks_p" in gsites[k]:
            a, b = float(h["rank_ks_p"]), float(gsites[k]["rank_ks_p"])
            if abs(a - b) > XCHECK_TOL:
                _fail(f"health cross-check mismatch on {k}: selfdraw rank KS-p {a:.6f} vs "
                      f"gate repaired_sectors {b:.6f} (tol {XCHECK_TOL})")

    # G2 REVIEW FINDING 7b: refuse verdict strings outside the two the table understands --
    # an "underpowered"/"scipy-unavailable" verdict must be a refusal, never an escalation.
    for ch in GATED + ("tau0amp",):
        v = str(ranks[ch]["verdict"])
        if v not in ("UNIFORM", "NON-UNIFORM"):
            _fail(f"rank verdict for {ch} is {v!r}, outside {{UNIFORM, NON-UNIFORM}}")
    # G2 REVIEW FINDING 7a: a degenerate-fit mock must not silently drop out of the gate
    # statistic while the census reads complete.
    for ch in GATED + ("tau0amp",):
        n_ch = int(pulls[ch].get("n", -1))
        if n_ch != int(sd["n"]):
            _fail(f"channel {ch} pull n = {n_ch} != arm n = {sd['n']} (degenerate fits "
                  f"dropped from the statistic?)")

    tau0_mean = float(pulls["tau0amp"]["mean"])
    tau0_rank_ok = str(ranks["tau0amp"]["verdict"]) == "UNIFORM"
    tau0_esc = (not tau0_rank_ok) or (abs(tau0_mean) > TAU0_PULL_LIMB)
    tau0_limbs = []
    if not tau0_rank_ok:
        tau0_limbs.append("ranks non-uniform")
    if abs(tau0_mean) > TAU0_PULL_LIMB:
        tau0_limbs.append(f"|pull| {abs(tau0_mean):.3f} > {TAU0_PULL_LIMB}")

    mean_fail = [ch for ch in GATED if not crit[ch]["mean_ok"]]
    rank_fail = [ch for ch in GATED if not crit[ch]["rank_ok"]]
    sd_only = [ch for ch in GATED if crit[ch]["mean_ok"] and not crit[ch]["sd_ok"]]

    # Precedence 4 > 3 > 6 > 2 > 1. The tau0 escalation limbs sit INSIDE row 3 and therefore
    # BEAT an sd-only row 6 (G2 review MUST-FIX 1: the A1c machinery of record lifts a row-6
    # base to row 3 on tau0 escalation -- the row-6-outranks-row-3 defect class of 3d.2 must
    # not recur through the other limb). When both fire, both reasons are reported.
    if mean_fail:
        row, why = 4, f"gated mean FAIL on {mean_fail} (worst row; -> PI)"
    elif rank_fail:
        row, why = 3, f"non-uniform ranks on gated {rank_fail} (3d.2; -> PI)"
    elif tau0_esc:
        why3 = f"tau0_amp escalation limb ({'; '.join(tau0_limbs)}) (unpaired 4c; -> PI)"
        if sd_only:
            why3 += f"; ALSO an sd-only failure on {sd_only} (Q2, reported alongside)"
        row, why = 3, why3
    elif sd_only:
        row, why = 6, f"sd-only failure on {sd_only} (Q2 dispersion-only FAIL, verbatim; -> PI)"
    elif not sd.get("health_ok"):
        bad = [k for k, h in sd["health"].items() if not h["healthy"]]
        row, why = 2, (f"gates/ranks/tau0 healthy but the self-drawn health block fails on "
                       f"{bad} (-> PI question; NOT a certification recommendation)")
    else:
        row, why = 1, ("the expected healthy outcome (recommendation only; promotion is a "
                       "PI act)")

    return dict(row=row, why=why, survey=survey, n=int(sd["n"]), crit=crit,
                tau0=dict(mean=tau0_mean, rank_ok=tau0_rank_ok, escalated=tau0_esc),
                health_ok=bool(sd.get("health_ok")))


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        sys.exit(2)
    out = run(argv[1], argv[2])
    print(f"=== FIRST-ARM DISPOSITION ({out['survey']}, n={out['n']}) ===")
    for ch, c in out["crit"].items():
        print(f"  {ch:8s} mean {c['mean']:+.4f} (<= {MEAN_GATE}: {c['mean_ok']}), "
              f"sd {c['std']:.4f} (<= {SD_GATE}: {c['sd_ok']}), ranks uniform: {c['rank_ok']}")
    t = out["tau0"]
    print(f"  tau0amp  mean {t['mean']:+.4f}, ranks uniform: {t['rank_ok']}, "
          f"escalation limbs fired: {t['escalated']}")
    print(f"  health block: {'PASS' if out['health_ok'] else 'FAIL'}")
    print(f"\nARM ROW {out['row']}: {out['why']}")


if __name__ == "__main__":
    main(sys.argv)
