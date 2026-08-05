#!/usr/bin/env python3
"""A1c PAIRED REPAIR COMPARISON — the pre-registered NO-REPEAT test, in committed code.

WHY THIS EXISTS. Pre-registration section 3d makes the NO-REPEAT test a PAIRED test on the 48
common mocks, and amendment 3 makes a six-conjunct NEGATIVE CONTROL mandatory before any paired
number may be read. Neither was implemented: the metals-off comparison was assembled by hand in a
memo, and the pairing assertion (`truth_vec` equality) is EXACTLY what a silent no-op produces, so
as written it was passed most easily by the accident it exists to exclude. Doing this by hand
after the gate verdict is on screen is precisely the post-hoc statistic selection the
pre-registration forbids, so it is fixed in advance and executed by this module.

TWO PI RULINGS (2026-07-29) BIND IT:

  1. **n_s IS INCLUDED.** A1's n_s passing its gate is NOT evidence that the defect left n_s
     unaffected. The metals-off diagnostic measured the LARGEST paired delta of any channel on
     n_s (+0.303 +/- 0.081, t = +3.72, Wilcoxon 0.0024, 10/12) — larger and more significant than
     A_p's +0.273 — on a GATED cosmology channel. Reporting only A_p and tau0_amp discarded the
     single most significant piece of mechanism evidence.
  2. **This is NOT a gate and emits no verdict.** The frozen gate covers n_s and A_p from A1c
     ALONE (PI #9 decision 3). This test is an explicitly comparative, SECONDARY diagnostic that
     qualifies the REPORT and never the gate verdict. A tau0_amp result here is a calibration
     warning, not a cosmology-bias claim, and licenses no extension.

WHAT IT REFUSES TO DO. If any negative-control conjunct fails, the arms are not the populations
they claim to be, and the module raises `PairingError` rather than returning a number. That
enforcement is the point: "STOP, do not read a number" must not depend on the reader's discipline.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/analyze_a1c_paired.py A1C_DIR A1_DIR [EXPECT_N]
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

# The six sites Option A made prior-predictive, with the DEPLOYED prior support used for the
# in-bracket conjunct. The metal brackets are the deployed LogUniform support exactly
# (closure_legb `metal_fnode_lo/hi` = 0.003/0.03, `metal_knode_lo/hi` = 1e-3/0.1) -- do not widen
# them, or a truth the deployed prior cannot produce passes the conjunct. The two f_res sites have
# Normal priors and so have no bracket: amendment 3 left "strictly inside their prior brackets"
# undefined for them, so they are checked as FINITE and NON-ZERO (a pinned f_res truth is exactly
# 0, the prior centre), which is the discriminating property. `kind` records which test applies.
PRIOR_BRACKETS = {
    "f_SiIII_eBOSS_z0": (0.003, 0.03, "bracket"),
    "f_SiIII_eBOSS_z1": (0.003, 0.03, "bracket"),
    "k_SiIII_eBOSS_z0": (1e-3, 1e-1, "bracket"),
    "k_SiIII_eBOSS_z1": (1e-3, 1e-1, "bracket"),
    "f_res_amp":        (-np.inf, np.inf, "finite_nonzero"),
    "f_res_slope":      (-np.inf, np.inf, "finite_nonzero"),
}

# A1's own pull on the MATCHED subset (mocks 0-47), never a full-96 constant: testing against the
# full-96 mean is the matched-control error the metals-off readout identified as its single most
# important methodological point (full-96 would be Ap -0.4053, tau0amp +0.5988).
A1_MATCHED_PULL = {"ns": -0.2532, "Ap": -0.4046, "tau0amp": +0.4916}

# THE TWO HYPOTHESES THE CI IS READ AGAINST. `delta = A1c - A1`, so:
#   FULL SURVIVAL of the defect  =>  A1c reproduces A1  =>  delta = 0
#   FULL REMOVAL of the defect   =>  A1c pull is 0      =>  delta = -(A1's own pull)
# Hence the full-removal delta is +0.4046 for A_p (whose pull was NEGATIVE) but **-0.4916** for
# tau0_amp. The first cut of this module stored both as positive and called exclusion of that
# value "excludes full survival"; for tau0_amp that flag was then True under EVERY hypothesis,
# including exact full survival, so section 4c's pull limb could never fire -- the precise branch
# PI ruling 2 was added to close. n_s is None BY DESIGN: the PI ruled that no hypothesis test is
# run on n_s. That is NOT because no such quantity exists -- A1's matched n_s pull is -0.2532, a
# perfectly well-defined magnitude -- and that value is context, never a test threshold.
FULL_REMOVAL_DELTA = {"ns": None, "Ap": +0.4046, "tau0amp": -0.4916}

CHANNELS = ("ns", "Ap", "tau0amp")

# THE FROZEN RUN-CFG CONSTANTS (PI ruling 3d.4, closing handoff 3b item 4). Every paired delta
# is measured AGAINST A1, and section 5c documents a near-perfect N=48 decoy population, yet
# only A1c's run_cfg was validated -- A1's was never checked at all. Both arms must carry these
# EXACT values: the full frozen hcd_prior_signature (analysis.lock, freeze cut c7eb371; pinned
# by test_frozen_signature_matches_the_freeze_artifact) and the eBOSS deployed fitting-prior
# identity fields. A mismatch on EITHER arm means a wrong or mixed population: STOP.
FROZEN_RUN_CFG = {
    "hcd_prior_signature":
        "50befc941edfc4c789286d2054d0107f1eb19a5672bcb86131426fb101eea216",
    "metal_prior": "flatlog2node",
    "survey": "eBOSS",
    "leg": "eBOSS",
    "sample_res": True,
    "f_res_amp_sigma": 0.05,
}

# Quantified in advance (amendment 3 left "materially different" undefined). The observed gap on
# mock 0 is |-1931.72 - (-349.93)| = 1581.8, so 100 sits far below the real signal and far above
# any numerical wobble.
LL_TRUE_MIN_GAP = 100.0


TAU0_PULL_ESCALATION = 0.30          # section 4c limb (b) clause 1, the frozen gate's magnitude

# The eBOSS deployed f_res prior widths (closure_legb: F_RES_AMP_SIGMA leg-matched 0.05 for
# eBOSS, F_RES_SLOPE_SIGMA 0.5), used by the across-mock distributional conjunct below.
FRES_PRIOR_SIGMA = {"f_res_amp": 0.05, "f_res_slope": 0.5}

# Across-mock KS threshold (PI 3d.4, closing handoff 3b item 3). This is a GROSS-DEFECT
# TRIPWIRE, not a calibration test: it exists to catch an arm whose truths were not drawn from
# the deployed laws at all (reused, pinned, or a wrong distribution), for which the KS p is
# astronomically small. 0.001 per site keeps the family false-STOP over the six sites at ~0.6%
# while retaining essentially unit power against the defect class. The calibration EVIDENCE for
# the repaired sectors is 5c-bis (pull/rank statistics), never this conjunct.
KS_PRIOR_ALPHA = 1e-3


class PairingError(RuntimeError):
    """The two arms are not the populations they claim to be. No number may be read."""


# ------------------------------------------------------- the two CI comparisons, named for what
# ------------------------------------------------------- they actually test

def ci_excludes_full_survival(lo, hi):
    """Does the paired 95% CI exclude `delta = 0`? Excluding 0 means the arms differ, i.e. **the
    defect did NOT fully survive**. This is the comparison the disposition needs; a bare p-value
    is not (amendment 3)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return bool(lo > 0.0 or hi < 0.0)


def ci_excludes_full_removal(lo, hi, full_removal_delta):
    """Does the CI exclude the delta implied by COMPLETE removal of the defect? Excluding it means
    full removal is not established. Reported alongside, never instead."""
    if full_removal_delta is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return bool(lo > full_removal_delta or hi < full_removal_delta)


def tau0_escalates(pull_mean, ci_lo, ci_hi):
    """PRE-REGISTRATION SECTION 4c LIMB (b), as code. The ARM escalates to at least row 3 if the
    `tau0_amp` residual pull is materially non-zero:

      clause 1: |pull mean| > 0.30 (the magnitude the frozen gate applies to the gated channels);
      clause 2: the paired 95% CI FAILS TO EXCLUDE delta = 0, i.e. full survival of the defect is
                not ruled out.

    Both are conservative and can only ESCALATE. This is a CALIBRATION WARNING: `tau0_amp` is not
    a gated channel, so it can produce neither a PASS nor a FAIL, it is not by itself a
    cosmology-bias claim, and it licenses NO extension. Escalation means return to the PI.
    """
    if np.isfinite(pull_mean) and abs(float(pull_mean)) > TAU0_PULL_ESCALATION:
        return True
    return not ci_excludes_full_survival(ci_lo, ci_hi)


# ------------------------------------------------------------------ loading

def load_arm(d):
    """{mock_index: record} from a directory of mock_XXXX.pkl."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "mock_*.pkl"))):
        try:
            rec = pickle.load(open(f, "rb"))
        except (EOFError, pickle.UnpicklingError):
            print(f"[paired] SKIP partial {os.path.basename(f)}")
            continue
        out[int(os.path.basename(f)[5:9])] = rec
    return out


# ------------------------------------------------------------------ negative control (section 3d)

def _truth_ok(nm, t):
    lo, hi, kind = PRIOR_BRACKETS[nm]
    if not np.isfinite(t):
        return False
    if kind == "bracket":
        return bool(lo < t < hi)
    return t != 0.0                       # a PINNED f_res truth is exactly the prior centre 0


def verify_pairing(a1c, a1, expect_n=48):
    """The six conjuncts of amendment 3 PLUS the three of amendment 5 (PI ruling 3d.4), ALL of
    which must hold across ALL mocks.

    Conjunct 3 is the load-bearing one: `truth_vec` equality (conjunct 1) is ALSO exactly what a
    silent no-op produces, so conjunct 1 alone is passed most easily by the accident it exists to
    exclude. Conjunct 6 is NOT independent evidence — an absent site yields `not_self_drawn: []`
    vacuously — so it corroborates conjunct 3 rather than replacing it.

    The amendment-5 conjuncts close two review-found holes: (7)+(8) the per-mock conjuncts are
    all PER-MOCK, so an arm that drew each of the six truths ONCE and reused it on every mock
    passed all of them (demonstrated by this module's own pre-2026-08 test fixture) — the 48
    truths per site must be pairwise DISTINCT and KS-consistent with the deployed law; (9) every
    paired delta is measured AGAINST A1, yet only A1c's `run_cfg` was validated — BOTH arms must
    carry the frozen constants (FROZEN_RUN_CFG), or the comparison is against a decoy population.
    """
    c = {}
    common = sorted(set(a1c) & set(a1))

    want = set(range(expect_n))
    c["indices"] = {"ok": set(a1c) == want and set(a1) >= want,
                    "detail": f"a1c={len(a1c)} a1={len(a1)} expected 0..{expect_n - 1}",
                    "missing_a1c": sorted(want - set(a1c)), "extra_a1c": sorted(set(a1c) - want)}

    bad_names, bad_vec = [], []
    for m in common:
        if list(a1c[m]["names"]) != list(a1[m]["names"]):
            bad_names.append(m)
            continue
        x, y = np.asarray(a1c[m]["truth_vec"]), np.asarray(a1[m]["truth_vec"])
        # EXACT, never allclose (amendment 3): a one-ulp difference means a different draw.
        if x.shape != y.shape or not np.array_equal(x, y) or x.tobytes() != y.tobytes():
            bad_vec.append(m)
    c["names_equal"] = {"ok": not bad_names, "detail": f"mismatched names on {bad_names[:5]}"}
    c["truth_vec_exact"] = {"ok": not bad_vec,
                            "detail": f"non-identical truth_vec on {bad_vec[:5]} "
                                      f"({len(bad_vec)} of {len(common)})"}

    drawn_bad, a1_bad = [], []
    for m in common:
        se = a1c[m].get("sites_extra") or {}
        for nm in PRIOR_BRACKETS:
            if not _truth_ok(nm, float((se.get(nm) or {}).get("truth", np.nan))):
                drawn_bad.append((m, nm))
        se1 = a1[m].get("sites_extra") or {}
        for nm in PRIOR_BRACKETS:
            if np.isfinite(float((se1.get(nm) or {}).get("truth", np.nan))):
                a1_bad.append((m, nm))
    c["a1c_truths_drawn"] = {
        "ok": not drawn_bad,
        "detail": f"{len(drawn_bad)} A1c site-truths not finite/in-support, e.g. {drawn_bad[:4]}"
                  " -- THIS IS THE CONJUNCT A SILENT NO-OP FAILS"}
    c["a1_truths_pinned"] = {"ok": not a1_bad,
                             "detail": f"{len(a1_bad)} A1 site-truths unexpectedly finite, "
                                       f"e.g. {a1_bad[:4]}"}

    flag_bad = [m for m in a1c
                if not (a1c[m].get("run_cfg", {}).get("metal_selfdraw") is True
                        and a1c[m].get("run_cfg", {}).get("fres_selfdraw") is True)]
    c["a1c_flags"] = {"ok": not flag_bad,
                      "detail": f"metal_selfdraw/fres_selfdraw not both True on {flag_bad[:5]}"}

    ll_bad = [m for m in common
              if not abs(float(a1c[m]["ll_true"]) - float(a1[m]["ll_true"])) > LL_TRUE_MIN_GAP]
    c["ll_true_moved"] = {"ok": not ll_bad,
                          "detail": f"|dll_true| <= {LL_TRUE_MIN_GAP} on {ll_bad[:5]} "
                                    "-- truth recorded but seemingly not PROPAGATED"}

    stamp_bad = [m for m in a1c
                 if (a1c[m].get("truth_site_semantics") or {}).get("not_self_drawn") != []]
    c["stamp_clear"] = {"ok": not stamp_bad,
                        "detail": f"not_self_drawn non-empty on {stamp_bad[:5]} (corroborates "
                                  "conjunct 3; vacuous when sites are absent)"}

    # ---- conjuncts 7+8: the ACROSS-MOCK distributional control (amendment 5, PI 3d.4) ----
    # Everything above is per-mock, so an arm that drew each truth ONCE and reused it passes
    # them all. (7) prior-predictive draws are pairwise distinct almost surely, so ANY exact
    # repeat means reuse; (8) distinct-but-wrong-law draws (e.g. clustered at a bracket edge)
    # only an across-mock KS against the DEPLOYED law can catch.
    site_vals = {nm: np.array([float(((a1c[m].get("sites_extra") or {}).get(nm) or {})
                                     .get("truth", np.nan)) for m in sorted(a1c)])
                 for nm in PRIOR_BRACKETS}
    dup = [nm for nm, v in site_vals.items()
           if (lambda f: np.unique(f).size != f.size)(v[np.isfinite(v)])]
    c["truths_distinct"] = {
        "ok": not dup,
        "detail": f"repeated truth values across mocks on {dup} -- a value drawn once and "
                  "REUSED is not a prior-predictive draw"}

    ks_bad = []
    try:
        from scipy import stats as st
        for nm, v in site_vals.items():
            fin = v[np.isfinite(v)]
            if fin.size < 5:
                continue                    # conjunct a1c_truths_drawn already fails hard here
            lo, hi, kind = PRIOR_BRACKETS[nm]
            if kind == "bracket":           # deployed LogUniform: uniform in log space
                u = (np.log(fin) - np.log(lo)) / (np.log(hi) - np.log(lo))
                p = float(st.kstest(u, "uniform").pvalue)
            else:                           # deployed Normal(0, sigma), eBOSS leg-matched
                p = float(st.kstest(fin, "norm", args=(0.0, FRES_PRIOR_SIGMA[nm])).pvalue)
            if p <= KS_PRIOR_ALPHA:
                ks_bad.append((nm, p))
        c["truths_match_prior"] = {
            "ok": not ks_bad,
            "detail": f"across-mock KS vs the deployed law at alpha {KS_PRIOR_ALPHA}: "
                      f"{[(nm, f'{p:.2e}') for nm, p in ks_bad]} -- the truths were not drawn "
                      "from the deployed fitting priors"}
    except ImportError:                     # scipy is present everywhere this runs; fail loud
        c["truths_match_prior"] = {"ok": False, "detail": "scipy unavailable -- cannot verify "
                                                          "the distributional conjunct"}

    # ---- conjunct 9: the FROZEN RUN-CFG constants, on BOTH arms (handoff 3b item 4) ----
    # Every delta is measured against A1, and section 5c documents a near-perfect N=48 decoy
    # population; a stale/mixed A1 passes every conjunct above (its six truths are nan either
    # way). Three lines of assertion close it.
    cfg_bad = []
    for arm_name, arm in (("A1c", a1c), ("A1", a1)):
        for m in arm:
            rc = arm[m].get("run_cfg") or {}
            for key, want in FROZEN_RUN_CFG.items():
                if rc.get(key) != want:
                    cfg_bad.append((arm_name, m, key, rc.get(key)))
    c["run_cfg_frozen"] = {
        "ok": not cfg_bad,
        "detail": f"{len(cfg_bad)} frozen-constant mismatches, e.g. {cfg_bad[:4]} -- one or "
                  "both arms are not the frozen-eBOSS population the pairing claims"}

    return {"ok": all(v["ok"] for v in c.values()), "conjuncts": c, "n_common": len(common)}


# ------------------------------------------------------------------ pulls

def _tau0_amp(ladder, lx_c):
    """amp = exp(intercept at zbar), the SAME convention as analyze_sbc_perleg.tau0_amp_slope --
    verified bitwise identical on all 96 real A1 pkls. `lx_c` is unused here because the centred
    regression makes the intercept exactly the mean of log(tau0); it is kept in the signature so
    the two call sites read alike."""
    y = np.log(np.clip(np.asarray(ladder, float), 1e-8, None))
    return float(np.exp(y.mean()))


def channel_pulls(rec):
    """The three channels' pulls for one mock, on the SAME definition the gated readout uses:
    (post_mean - truth)/post_sd."""
    names = list(rec["names"])
    dr = np.asarray(rec["draws"])
    tv = np.asarray(rec["truth_vec"])
    out = {}
    for k in ("ns", "Ap"):
        j = names.index(k)
        col = dr[:, j]
        sd = col.std(ddof=1) if col.size > 1 else np.nan
        out[k] = float((col.mean() - tv[j]) / sd) if (np.isfinite(sd) and sd > 0) else np.nan
    idx = [names.index(f"tau0_z{i}") for i in range(13)]
    z = np.array([2.2 + 0.2 * i for i in range(13)])
    lx = np.log1p(z)
    lx_c = lx - lx.mean()
    amps = np.array([_tau0_amp(dr[i, idx], lx_c) for i in range(dr.shape[0])])
    at = _tau0_amp(tv[idx], lx_c)
    sa = amps.std(ddof=1) if amps.size > 1 else np.nan
    out["tau0amp"] = float((amps.mean() - at) / sa) if (np.isfinite(sa) and sa > 0) else np.nan
    return out


# ------------------------------------------------------------------ the paired report

def paired_report(a1c, a1, verify=True, expect_n=48):
    """Paired delta (A1c - A1) per channel, with the 95% CI, the realized arm-to-arm correlation,
    and a distribution-free backup. Emits NO verdict.

    The CI comparison, not the p-value, is what the disposition reads (amendment 3): rejecting
    `H0: delta = 0` licenses "the arms differ", NOT "the defective arm's failure does not
    reproduce". Both comparisons are reported explicitly so neither can be chosen later.
    """
    if verify:
        v = verify_pairing(a1c, a1, expect_n=expect_n)
        if not v["ok"]:
            bad = {k: d["detail"] for k, d in v["conjuncts"].items() if not d["ok"]}
            raise PairingError(
                "NEGATIVE CONTROL FAILED -- the arms are not the populations they claim to be. "
                "Do not read a number. Failing conjuncts: " + json.dumps(bad, indent=2))

    from scipy import stats as st
    common = sorted(set(a1c) & set(a1))
    pc = {m: channel_pulls(a1c[m]) for m in common}
    pa = {m: channel_pulls(a1[m]) for m in common}

    chans = {}
    for k in CHANNELS:
        x = np.array([pc[m][k] for m in common], float)
        y = np.array([pa[m][k] for m in common], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = int(x.size)
        d = x - y
        mean = float(d.mean()) if n else np.nan
        sd = float(d.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = float(st.t.ppf(0.975, n - 1)) if n > 1 else np.nan
        lo, hi = (mean - tcrit * sem, mean + tcrit * sem) if n > 1 else (np.nan, np.nan)
        tstat, p = (st.ttest_rel(x, y) if n > 1 else (np.nan, np.nan))
        try:
            wp = float(st.wilcoxon(x, y).pvalue) if n > 5 else np.nan
        except ValueError:                      # all-zero differences
            wp = np.nan
        rho = float(np.corrcoef(x, y)[0, 1]) if n > 2 else np.nan
        rem = FULL_REMOVAL_DELTA[k]
        rec = dict(
            n=n, delta_mean=mean, delta_sd=sd, sem=float(sem) if n > 1 else np.nan,
            ci95_lo=float(lo), ci95_hi=float(hi), t=float(tstat), p=float(p),
            wilcoxon_p=wp, rho=rho,
            a1c_mean=float(x.mean()) if n else np.nan,
            a1_mean=float(y.mean()) if n else np.nan,
            a1_matched_pull=A1_MATCHED_PULL[k],
            full_removal_delta=rem,
            # delta = 0 excluded => the arms differ => the defect did NOT fully survive
            ci_excludes_full_survival=ci_excludes_full_survival(lo, hi),
            # the full-removal delta excluded => full removal is NOT established
            ci_excludes_full_removal=ci_excludes_full_removal(lo, hi, rem),
        )
        if k == "tau0amp":
            # Section 4c limb (b). Recorded here so the escalation is read off a committed rule
            # rather than reconstructed by eye once the numbers are on screen.
            # CLAUSE 1 TAKES A1c's OWN RESIDUAL PULL (`x.mean()`), NOT the paired delta. 4c(b)
            # says "|pull mean| > 0.30, the same magnitude the frozen gate applies to the gated
            # channels" -- a property of A1c ALONE. Fed the delta (the round-4 defect), the rule
            # escalated under a PERFECT REPAIR (delta ~ -0.49) and went silent on an A1c pull
            # LARGER than A1's own (+0.75 -> delta ~ +0.26): the branch PI ruling 2 exists to
            # close, reopened at the call site. Pinned by a CALL-SITE test, not a bare-function one.
            rec["escalates_4c_b"] = tau0_escalates(float(x.mean()), lo, hi)
        chans[k] = rec

    return {
        "channels": chans,
        "n_common": len(common),
        "gated": False,
        "note": ("SECONDARY comparative diagnostic. The GATE and RANK verdicts are computed from "
                 "A1c ALONE (PI #9 decision 3); this qualifies the REPORT only and emits no "
                 "verdict. n_s is included per PI ruling 2026-07-29: A1's n_s gate pass is NOT "
                 "evidence the defect left n_s unaffected (metals-off measured the largest paired "
                 "delta of any channel on n_s). A tau0_amp result here is a calibration warning, "
                 "NOT a cosmology-bias claim, and licenses no extension. Read the CI against the "
                 "reference delta, never a bare p-value."),
    }


def _fmt(rep):
    L = ["", "=" * 92,
         "A1c PAIRED REPAIR COMPARISON (A1c - A1) -- pre-registered 3d, NOT GATED",
         "=" * 92]
    for k, c in rep["channels"].items():
        rem = ("n/a -- PI ruled NO hypothesis test on n_s; A1's matched pull "
               f"{c['a1_matched_pull']:+.4f} is CONTEXT, never a threshold"
               if c["full_removal_delta"] is None else f"{c['full_removal_delta']:+.4f}")
        L.append(f"  {k:8s} n={c['n']:3d}  delta={c['delta_mean']:+.4f} +/- {c['delta_sd']:.4f} "
                 f"(sem {c['sem']:.4f})  95% CI [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}]")
        L.append(f"           t={c['t']:+.3f} p={c['p']:.4g}  Wilcoxon p={c['wilcoxon_p']:.4g}  "
                 f"realized rho={c['rho']:+.3f}")
        L.append(f"           A1c mean={c['a1c_mean']:+.4f}  A1 mean={c['a1_mean']:+.4f}  "
                 f"delta under FULL REMOVAL={rem}")
        L.append(f"           CI excludes FULL SURVIVAL (delta=0, i.e. the arms differ): "
                 f"{c['ci_excludes_full_survival']}")
        L.append(f"           CI excludes FULL REMOVAL (full removal not established): "
                 f"{c['ci_excludes_full_removal']}")
        if "escalates_4c_b" in c:
            L.append(f"           >>> section 4c limb (b) ESCALATES: {c['escalates_4c_b']}"
                     "  (calibration warning; NOT a cosmology-bias claim; licenses NO extension)")
    L.append("")
    L.append("  " + rep["note"])
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: analyze_a1c_paired.py A1C_DIR A1_DIR [EXPECT_N]")
    _a1c = load_arm(sys.argv[1])
    _a1 = load_arm(sys.argv[2])
    _n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    _v = verify_pairing(_a1c, _a1, expect_n=_n)
    print("\n-- NEGATIVE CONTROL (pre-registration 3d, amendment 3) --")
    for _k, _d in _v["conjuncts"].items():
        # detail strings describe the FAILURE mode, so only print them when the conjunct failed
        print(f"  {'OK  ' if _d['ok'] else 'FAIL'}  {_k:20s} {'' if _d['ok'] else _d['detail']}")
    if not _v["ok"]:
        raise SystemExit("\nNEGATIVE CONTROL FAILED -- STOP, do not read a number.")
    print(_fmt(paired_report(_a1c, _a1, verify=True, expect_n=_n)))
