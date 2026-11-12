"""The ``hcd_priya.compare/1`` prediction record, its validator, and record-vs-record comparison (numpy only).

A record is a JSON object written by one adapter for one comparator:

  schema       "hcd_priya.compare/1"
  comparator   identity: id, name, repo, commit, implementation, artifacts [{path, sha256}], fidelity (LF/HF/MF),
               training (free text: sims, lineage), env
  observable   an id from OBSERVABLES (what P means: estimator, filtering, units)
  L_box_hmpc, modes   the native k nodes: k_com = 2 pi n / L_box for each integer mode n
  cases        [{case_id, family, state: {params: {9 named}, mean_flux: {tau0_amp, dtau0, z_pivot}}, z: [...],
                 support: {status, reason}, P: [n_z][n_k] (km/s), k_skm: [n_z][n_k], native: {...}}]

``native`` keeps the comparator's own outputs and inputs (per-class arrays, GP standard deviations, the exact input
vector) beside the harmonized ``P``. ``validate_record`` refuses a record whose k_skm does not follow from its modes
and state (catches h/Mpc labelled as s/km, a missing 2 pi, a wrong cosmology in an adapter that reports k), any
parameter that is not one of the nine named ones (``tau0`` is not a cosmological parameter here), NaN or non-positive
power, and shape mismatches. It cannot see a comparator's internal k: each adapter checks its own native nodes
against the box modes.
``compare_records`` refuses records whose observables, nodes or case states differ instead of intersecting them.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import conventions as C
from . import metrics as M

SCHEMA = "hcd_priya.compare/1"
OUTCOMES = ("PASS", "FAIL", "NOT COMPARABLE", "BLOCKED", "NOT RUN")
SUPPORT = ("COMPARABLE", "NOT COMPARABLE", "BLOCKED")
FIDELITIES = ("LF", "HF", "MF")
OBSERVABLES = {
    "O1": {
        "description": "1D flux power of F/<F>-1 over all sightlines after the tau>1e6 trough fill "
        "(fake_spectra _filter_single_tau_complex, thresh2 0.25), <F> the global target "
        "exp(-alpha(z) 0.0023 (1+z)^3.65), no window, no resolution correction",
        "P_unit": "km/s",
        "k_unit": "s/km (angular)",
        "native_k": "comoving box mode n, k = 2 pi n / L_box h/Mpc",
    },
}
UNDEFINED = "UNDEFINED"
OBSERVABLES[UNDEFINED] = {
    "description": "no matched observable is defined yet for this comparator (a blocker): a "
    "record with it has every case BLOCKED",
    "P_unit": "",
    "k_unit": "",
    "native_k": "",
}
K_REL_TOL = 1e-12
STATE_REL_TOL = 1e-12


class ComparisonSchemaError(ValueError):
    """A record or a pair of records that the harness refuses to compare."""


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _identity_problems(c):
    out = []
    if not isinstance(c, dict):
        return ["comparator is not an object"]
    for k in (
        "id",
        "name",
        "repo",
        "commit",
        "implementation",
        "artifacts",
        "fidelity",
        "training",
        "env",
    ):
        if k not in c:
            out.append(f"comparator.{k} missing")
    if c.get("fidelity") not in FIDELITIES:
        out.append(f"comparator.fidelity {c.get('fidelity')!r} not in {FIDELITIES}")
    if not isinstance(c.get("commit"), str) or not c.get("commit"):
        out.append("comparator.commit must be a non-empty string")
    for a in c.get("artifacts") or []:
        h = a.get("sha256") if isinstance(a, dict) else None
        if not (isinstance(h, str) and len(h) == 64 and all(ch in "0123456789abcdef" for ch in h)):
            out.append(f"artifact {a!r} lacks a 64-hex sha256")
    return out


def validate_record(rec):
    """Raise ComparisonSchemaError listing every problem; return the record unchanged when valid."""
    p = []
    if not isinstance(rec, dict):
        raise ComparisonSchemaError("record is not an object")
    if rec.get("schema") != SCHEMA:
        p.append(f"schema {rec.get('schema')!r}, expected {SCHEMA}")
    p += _identity_problems(rec.get("comparator"))
    if rec.get("observable") not in OBSERVABLES:
        p.append(f"observable {rec.get('observable')!r} not in {sorted(OBSERVABLES)}")
    L = rec.get("L_box_hmpc")
    modes = rec.get("modes")
    if not _is_num(L) or L <= 0:
        p.append("L_box_hmpc must be a positive number")
    if (
        not isinstance(modes, list)
        or not modes
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in modes)
        or modes != sorted(set(modes))
    ):
        p.append("modes must be a non-empty strictly increasing list of positive ints")
        modes = None
    cases = rec.get("cases")
    if not isinstance(cases, list) or not cases:
        p.append("cases must be a non-empty list")
        cases = []
    if rec.get("observable") == UNDEFINED and any(
        isinstance(c, dict) and (c.get("support") or {}).get("status") != "BLOCKED" for c in cases
    ):
        p.append("a record with an UNDEFINED observable must have every case BLOCKED")
    seen = set()
    for c in cases:
        cid = c.get("case_id") if isinstance(c, dict) else None
        if not isinstance(cid, str) or not cid:
            p.append(f"case without a case_id: {c!r:.80}")
            continue
        if cid in seen:
            p.append(f"duplicate case_id {cid}")
        seen.add(cid)
        p += [f"{cid}: {m}" for m in _case_problems(c, L, modes)]
    if p:
        raise ComparisonSchemaError("; ".join(p))
    return rec


def _case_problems(c, L, modes):
    p = []
    st = c.get("state") or {}
    try:
        C.check_params(st.get("params"))
        C.check_mean_flux(st.get("mean_flux"))
    except C.ConventionError as e:
        p.append(str(e))
    z = c.get("z")
    if not isinstance(z, list) or not z or not all(_is_num(v) for v in z) or len(set(z)) != len(z):
        p.append("z must be a non-empty list of distinct finite numbers")
        z = None
    sup = c.get("support") or {}
    if sup.get("status") not in SUPPORT:
        p.append(f"support.status {sup.get('status')!r} not in {SUPPORT}")
    elif sup["status"] != "COMPARABLE":
        if not isinstance(sup.get("reason"), str) or not sup["reason"].strip():
            p.append(f"support {sup['status']} needs a reason")
        return p
    if p or z is None or modes is None or not _is_num(L):
        return p + (["COMPARABLE case lacks valid z, modes or L_box"] if not p else [])
    P = np.asarray(c.get("P"), dtype=float) if c.get("P") is not None else None
    K = np.asarray(c.get("k_skm"), dtype=float) if c.get("k_skm") is not None else None
    want = (len(z), len(modes))
    if P is None or P.shape != want:
        p.append(f"P shape {None if P is None else P.shape}, expected {want}")
    elif not np.all(np.isfinite(P)) or not np.all(P > 0):
        p.append(
            "P has non-finite or non-positive values (masking is not allowed; declare the case instead)"
        )
    if K is None or K.shape != want:
        p.append(f"k_skm shape {None if K is None else K.shape}, expected {want}")
    else:
        prm = st["params"]
        kc = C.k_com_modes(np.asarray(modes), L)
        for i, zi in enumerate(z):
            ref = C.k_skm_from_kcom(kc, zi, prm["hub"], prm["omegamh2"])
            if not np.allclose(K[i], ref, rtol=K_REL_TOL, atol=0):
                p.append(
                    f"k_skm at z={zi} does not follow from the modes and (hub, omegamh2): wrong unit, "
                    f"missing 2 pi or wrong cosmology (max rel {np.max(np.abs(K[i] / ref - 1)):.3g})"
                )
                break
    if not isinstance(c.get("native", {}), dict):
        p.append("native must be an object")
    return p


def dumps(rec):
    """JSON text of a validated record (floats round-trip exactly through repr)."""
    return json.dumps(validate_record(rec), sort_keys=True, allow_nan=False)


def loads(text):
    return validate_record(json.loads(text))


def _same_state(a, b):
    pa, pb = a["params"], b["params"]
    if any(not math.isclose(pa[n], pb[n], rel_tol=STATE_REL_TOL, abs_tol=0) for n in C.PARAM_NAMES):
        return False
    return a["mean_flux"] == b["mean_flux"]


def level_verdict(outcomes):
    """The one precedence for any set of case outcomes: FAIL, then NOT RUN, then BLOCKED, then PASS, else NOT
    COMPARABLE (also for an empty set). Used for the level and for every group (family) of cases."""
    vals = set(outcomes)
    for v in ("FAIL", "NOT RUN", "BLOCKED", "PASS"):
        if v in vals:
            return v
    return "NOT COMPARABLE"


def group_verdicts(case_outcomes, group_of):
    """``{group: {"verdict", "complete", "outcomes": {outcome: count}}}`` over the cases of each group, with the level
    precedence: a group with a missing or blocked case can never PASS, and a group whose cases are all blocked is
    reported (BLOCKED), not dropped."""
    out = {}
    for cid, g in group_of.items():
        out.setdefault(g, []).append(case_outcomes.get(cid, "NOT RUN"))
    return {
        g: {
            "verdict": level_verdict(v),
            "complete": "NOT RUN" not in v and "BLOCKED" not in v,
            "outcomes": {o: v.count(o) for o in sorted(set(v))},
        }
        for g, v in out.items()
    }


def compare_records(test, ref, threshold, *, declared=None):
    """Compare two validated records case by case.

    Returns ``{"verdict", "complete", "max_abs", "cases": {case_id: outcome}, "points": n, "summary": {...},
    "r": {case_id: r}}`` (``r`` has the shape of ``P``: n_z by n_modes).

    Case outcomes: missing from either side NOT RUN; BLOCKED on either side BLOCKED; NOT COMPARABLE on BOTH sides NOT
    COMPARABLE (contributes no points); COMPARABLE on both PASS or FAIL. Support is declared by the runner for both
    sides, so one side calling a case NOT COMPARABLE that the other calls COMPARABLE is refused: an adapter cannot turn
    its own failure into an exclusion. A record whose observable is UNDEFINED makes every case BLOCKED.

    Level verdict: FAIL if any comparable point exceeds the threshold; otherwise NOT RUN if any case is missing
    (incomplete), BLOCKED if any case is blocked, PASS if every remaining case passed, NOT COMPARABLE if none was
    comparable.

    ``declared`` ({case_id: support status}, the runner's case list) binds the comparison to it: a declared case absent
    from both records is NOT RUN, a record case that was not declared or whose support differs from the declared one
    is refused. With ``declared``, PASS requires every declared case to be present and every comparable case to pass.
    """
    validate_record(test)
    validate_record(ref)
    undefined = UNDEFINED in (test["observable"], ref["observable"])
    if not undefined and test["observable"] != ref["observable"]:
        raise ComparisonSchemaError(
            f"observables differ: {test['observable']} vs {ref['observable']}"
        )
    if test["modes"] != ref["modes"] or test["L_box_hmpc"] != ref["L_box_hmpc"]:
        raise ComparisonSchemaError(
            "native k nodes differ (modes or L_box): refusing to intersect or interpolate"
        )
    tc = {c["case_id"]: c for c in test["cases"]}
    rc = {c["case_id"]: c for c in ref["cases"]}
    ids = set(tc) | set(rc)
    if declared is not None:
        stray = ids - set(declared)
        if stray:
            raise ComparisonSchemaError(f"cases not declared by the runner: {sorted(stray)[:5]}")
        for side in (tc, rc):
            for cid, c in side.items():
                if c["support"]["status"] != declared[cid] and not undefined:
                    raise ComparisonSchemaError(
                        f"{cid}: record support {c['support']['status']} differs from the "
                        f"declared {declared[cid]}"
                    )
        ids |= set(declared)
    outcomes, rs = {}, {}
    for cid in sorted(ids):
        a, b = tc.get(cid), rc.get(cid)
        if a is None or b is None:
            outcomes[cid] = "NOT RUN"
            continue
        if (
            not _same_state(a["state"], b["state"])
            or a["z"] != b["z"]
            or a.get("family") != b.get("family")
        ):
            raise ComparisonSchemaError(
                f"{cid}: the two records describe different physical states or redshifts"
            )
        sa, sb = a["support"]["status"], b["support"]["status"]
        if undefined or "BLOCKED" in (sa, sb):
            outcomes[cid] = "BLOCKED"
            continue
        if sa != sb:
            raise ComparisonSchemaError(
                f"{cid}: support differs between the records ({sa} vs {sb}); support is "
                "declared once by the runner, never by one adapter"
            )
        if sa == "NOT COMPARABLE":
            outcomes[cid] = "NOT COMPARABLE"
            continue
        r = M.ratio_residual(a["P"], b["P"])
        rs[cid] = r
        outcomes[cid] = M.verdict(r, threshold)
    all_r = np.concatenate([r.ravel() for r in rs.values()]) if rs else np.array([])
    vals = set(outcomes.values())
    level = level_verdict(vals)
    return {
        "verdict": level,
        "complete": "NOT RUN" not in vals and "BLOCKED" not in vals,
        "threshold": threshold,
        "points": int(all_r.size),
        "max_abs": float(np.max(np.abs(all_r))) if all_r.size else None,
        "cases": outcomes,
        "summary": M.summarize(all_r, threshold),
        "r": rs,
    }
