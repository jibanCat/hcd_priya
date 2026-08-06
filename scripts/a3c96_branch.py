#!/usr/bin/env python3
"""A3c N=96 adjudication: outcome-branch classification (PI #10 3.4; N=96 prereg sec 10).

Maps the CUMULATIVE N=96 readout to the authorized outcome branch, precedence
D > C > B > A. The deployed ``firstarm_disposition.run`` stays the single authority on
disposition rows; this module adds ONLY the branch mapping, the divergence criterion and
the tranche-1 integrity checks pinned in the pre-registration. It makes no measurement
and contains no threshold not stated there.

  D  mechanical/provenance failure: a disposition refusal (partial/inconsistent inputs),
     a wrong adjudication population (n != 96 or survey != KS), a tranche-1 sha256
     mismatch, or a wrong OUTDIR mock-file census (missing/stray/smoke pkls)
  C  potentially material anomaly: disposition rows 2/3/4, any cumulative n_div > 0, or
     a row-6 sd-only channel set other than exactly {ns}
  B  row 6 with sd-only set exactly {ns}, all else healthy
  A  row 1 (all frozen conditions pass)

Usage:
  ... scripts/a3c96_branch.py GATE_JSON SELFDRAW_JSON \
        --tranche1-sha256 SHA_FILE --outdir OUTDIR
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

N_TOTAL = 96
N_T1 = 48
SURVEY = "KS"

# The immutable N=48 result of record (2026-08-06-A3c-READOUT.md; PI #10 sec 2). Reported
# verbatim in every classification printout; never recomputed, never averaged away.
IMMUTABLE_N48 = (
    "IMMUTABLE N=48 RESULT (job 56590250, formal preregistered verdict FAIL,\n"
    "dispersion-only): n_s mean -0.1331 (inside 0.30), sd 1.1251 > 1.1, ranks uniform\n"
    "KS p 0.781; A_p PASS (-0.0511 +/- 1.0804, uniform); tau0_amp +0.0203 uniform\n"
    "p 0.996; f_res health block PASS; 0 divergences. Not evidence of mean bias."
)

ACTIONS = {
    "A": ("the KS certificate may proceed (PI #10 3.4-A); returns to the PI with the "
          "mandated permanent-record statements (N=48 dispersion-only formal FAIL not "
          "rewritten or erased; no mean-bias/rank evidence; separately authorized N=96 "
          "adjudication passed). Promotion wording is a PI act."),
    "B": ("STOP; return to the PI. No waiver, threshold change, reinterpretation or "
          "extension."),
    "C": ("potentially material anomaly: STOP the KS certification path; return for "
          "scientific review before any further repair, diagnostic or production "
          "action."),
    "D": ("STOP. Do not substitute a population, silently repair the record, or proceed "
          "under an assumed-equivalent configuration. Return to the PI."),
}


def _load_disposition():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "firstarm_disposition.py")
    spec = importlib.util.spec_from_file_location("firstarm_disposition", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(outdir, sha_file):
    """Section-4 checks: exact mock census 0000..0095 (no strays, no smoke pkls) and a
    byte-level sha256 match of all 48 tranche-1 pkls against the pre-launch inventory.
    Returns a list of failure strings (empty = intact)."""
    fails = []
    smoke = sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "*.smoke.pkl")))
    if smoke:
        fails.append(f"smoke pkl(s) present in OUTDIR: {smoke}")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "mock_*.pkl"))
                   if not p.endswith(".smoke.pkl"))
    expect = [f"mock_{m:04d}.pkl" for m in range(N_TOTAL)]
    if names != expect:
        missing = sorted(set(expect) - set(names))
        stray = sorted(set(names) - set(expect))
        fails.append(f"mock census != 0000..{N_TOTAL - 1:04d}: missing {missing or 'none'}, "
                     f"stray {stray or 'none'}")
    recorded = {}
    try:
        with open(sha_file) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    digest, name = line.split(None, 1)
                except ValueError:
                    return fails + [f"malformed sha256 inventory line {lineno}: {line!r}"]
                base = os.path.basename(name.strip().lstrip("*"))
                if base in recorded:
                    return fails + [f"duplicate sha256 inventory entry for {base}"]
                recorded[base] = digest
    except OSError as e:
        return fails + [f"cannot read tranche-1 sha256 inventory: {e}"]
    expect_t1 = {f"mock_{m:04d}.pkl" for m in range(N_T1)}
    if set(recorded) != expect_t1:
        fails.append(f"sha256 inventory names != tranche-1 census: has {len(recorded)} "
                     f"entries, symmetric difference {sorted(set(recorded) ^ expect_t1)[:4]}")
        return fails
    for name in sorted(recorded):
        path = os.path.join(outdir, name)
        try:
            got = _sha256(path)
        except OSError as e:
            fails.append(f"tranche-1 pkl unreadable: {name}: {e}")
            continue
        if got != recorded[name]:
            fails.append(f"tranche-1 pkl CHANGED since the pre-launch inventory: {name}")
    return fails


def classify_row(row, sd_only, why):
    """The pure row -> branch mapping of prereg section 10 (PI #10 3.4). An unrecognized
    row is a branch-D refusal, never a silent certificate-proceeds classification."""
    if row in (2, 3, 4):
        return "C", f"disposition row {row}: {why}"
    if row == 6:
        if sd_only == ["ns"]:
            return "B", f"row 6 with sd-only set exactly ['ns']: {why}"
        return "C", (f"row 6 with sd-only set {sd_only} != ['ns'] -- a failure mode "
                     f"absent at N=48 (other substantive failure)")
    if row == 1:
        return "A", f"disposition row 1: {why}"
    return "D", (f"unrecognized disposition row {row!r} -- refusing to classify (the "
                 f"branch table covers rows 1/2/3/4/6 only)")


def run(gate_json, selfdraw_json, sha_file, outdir):
    """Classify; returns dict(branch, reasons, disposition, integrity_fails, n_div_total)."""
    reasons = []
    integrity_fails = check_integrity(outdir, sha_file)
    if integrity_fails:
        return dict(branch="D", reasons=integrity_fails, disposition=None,
                    integrity_fails=integrity_fails, n_div_total=None)

    dp = _load_disposition()
    try:
        out = dp.run(gate_json, selfdraw_json)
        with open(selfdraw_json) as f:
            sd = json.load(f)
    except dp.DispositionError as e:
        return dict(branch="D", reasons=[f"disposition refusal: {e}"], disposition=None,
                    integrity_fails=[], n_div_total=None)
    except (OSError, json.JSONDecodeError) as e:
        return dict(branch="D", reasons=[f"unreadable/malformed readout input: {e}"],
                    disposition=None, integrity_fails=[], n_div_total=None)
    if out["n"] != N_TOTAL or out["survey"] != SURVEY:
        return dict(branch="D",
                    reasons=[f"wrong adjudication population: survey {out['survey']!r} "
                             f"n {out['n']} (expected {SURVEY!r} n {N_TOTAL}) -- is this "
                             f"the committed N=48 readout?"],
                    disposition=out, integrity_fails=[], n_div_total=None)
    n_div = sd.get("n_div_total")
    if not isinstance(n_div, int) or n_div < 0:
        return dict(branch="D",
                    reasons=[f"selfdraw JSON carries no usable n_div_total "
                             f"(got {n_div!r})"],
                    disposition=out, integrity_fails=[], n_div_total=None)

    row = out["row"]
    sd_only = [ch for ch in ("ns", "Ap")
               if out["crit"][ch]["mean_ok"] and not out["crit"][ch]["sd_ok"]]
    branch, reason = classify_row(row, sd_only, out["why"])
    reasons.append(reason)
    if branch != "D" and n_div > 0:
        reasons.append(f"cumulative n_div_total = {n_div} > 0 (pipeline norm is 0 across "
                       f"all landed production fits): potentially material anomaly")
        if branch in ("A", "B"):
            branch = "C"
    return dict(branch=branch, reasons=reasons, disposition=out,
                integrity_fails=[], n_div_total=n_div)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("gate_json")
    ap.add_argument("selfdraw_json")
    ap.add_argument("--tranche1-sha256", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv[1:])
    out = run(a.gate_json, a.selfdraw_json, a.tranche1_sha256, a.outdir)
    print("=== A3c N=96 ADJUDICATION BRANCH (PI #10 3.4; precedence D > C > B > A) ===\n")
    print(IMMUTABLE_N48 + "\n")
    if out["disposition"] is not None:
        d = out["disposition"]
        print(f"cumulative disposition: ARM ROW {d['row']} ({d['survey']}, n={d['n']}); "
              f"health block {'PASS' if d['health_ok'] else 'FAIL'}")
    print(f"tranche-1 integrity: "
          f"{'INTACT (48/48 sha256 match, census exact)' if not out['integrity_fails'] else 'FAILED'}")
    if out["n_div_total"] is not None:
        print(f"cumulative n_div_total: {out['n_div_total']}")
    print("\nreasons:")
    for r in out["reasons"]:
        print(f"  - {r}")
    print(f"\nBRANCH {out['branch']}: {ACTIONS[out['branch']]}")


if __name__ == "__main__":
    main(sys.argv)
