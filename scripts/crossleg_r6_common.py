"""Cross-leg r6x paired prior-sensitivity campaign: the SINGLE-AUTHORITY pre-registered
constants + override mechanics (PI decisions #7 execution annex, 2026-07-24; design (2) of
PROPOSAL-crossleg-R6.md with its PANEL REVISIONS applied).

SCOPE (PI amendment #6): r6x is an LLS-amplitude PRIOR-CENTRE SENSITIVITY test only, one
displacement direction, never "the HCD degeneracy" and never the legacy-vs-mapped migration
test. Mock fits only; no real-data path exists in any consumer of this module.

THE OVERRIDE MECHANISM (panel revision 4, CS MUST-FIX): the displaced arm moves the LLS prior
centre by IN-PLACE DICT ITEM ASSIGNMENT on the two signature-carried knobs
``inference.HCD_LLS_SURVEY_BOOST[leg]`` / ``inference.HCD_LLS_SURVEY_FRAC_SIGMA[leg]`` —
NEVER module-attribute rebinding. Why: closure_legb from-imports both names at load
(closure_legb.py:56-57), so its module globals are REFERENCES TO THE SAME DICT OBJECTS;
item assignment propagates through every from-import, while rebinding
``inference.HCD_LLS_SURVEY_BOOST = {...}`` leaves closure_legb's binding on the OLD object —
the built prior would then stay deployed while hcd_prior_signature() (which reads inference's
namespace live) moved: the documented from-import desync trap. Both dicts are in
``hcd_prior_constants_payload()``, so the in-place mutation moves ``hcd_prior_signature()``
COHERENTLY with the effective prior. Empirically verified by tests/test_crossleg_r6.py.

PER-LEG DISPLACEMENT (the ×1.287 common yardstick; deployed boosts VERIFIED per leg):
  eBOSS / DESI (deployed boost 1.0, frac sigma 0.287; alpha_pivot_powerlaw_v1):
      boost 1.0 -> 1.287 and frac 0.287 -> 0.287/1.287, i.e. the LLS prior centre moves
      +1.0 deployed prior sigma AT FIXED ABSOLUTE WIDTH (sigma_disp = (0.287/1.287) x
      (1.287 mu) = 0.287 mu exactly, bit-equal in IEEE double — asserted).
  KS (deployed boost 2.5, dndx_mapped_v2): boost 2.5 -> 2.5*1.287 = 3.2175. The KS boost
      acts in dN/dX space PRE-map (HCD_LLS_BOOST_SPACE["KS"]="dndx_premap"), so the dN/dX
      reference LLS curve displaces x1.287 at all z = +ln(1.287)/0.5310 = +0.4752 eps_lls
      prior sigma (NOT +1 KS prior sigma; the yardstick is the SAME multiplicative centre
      displacement, pre-registered). HCD_LLS_SURVEY_FRAC_SIGMA["KS"] is NOT touched: on the
      mapped branch the widths are the pinned KS_DNDX_SIGMA_* literals (fixed absolute width
      holds trivially), and mutating an inert dict entry would move the signature without
      moving the effective prior — exactly the desync class the mechanism must exclude.
      The displaced mapped pivot centre 0.471945 stays INSIDE the frozen guard band
      (0.326, 0.476) — verified, no frozen guard is touched.

Conventions: no em-dashes; fail-loud; stamps everywhere; frozen modules untouched.
"""
import contextlib
import hashlib

import numpy as np

# ---------------------------------------------------------------------------------------------
# Pre-registered campaign constants (execution annex, 2026-07-24).
# ---------------------------------------------------------------------------------------------
R6X_SEED = 20260724                     # FRESH seed (panel revision 6: 20260615 on DESI would
                                        # bit-replay the dispcenter clean arm's low truth set)
R6X_FACTOR = 1.287                      # the multiplicative LLS prior-centre displacement
R6X_FRAC_DISP = 0.287 / 1.287           # displaced frac sigma (eBOSS/DESI): fixed absolute width
R6X_LEGS = ("eBOSS", "DESI", "KS")
R6X_ARMS = ("deployed", "dispprior")
R6X_TRUTH_SOURCE = "deployed-selfdraw"  # per-mock truths from the DEPLOYED per-leg ctx's own
                                        # prior sites (run_legb truth_fn; interior points of BOTH
                                        # arms' support -> fair pairing)
R6X_N_PAIRS = 8                         # n=8 pairs; extension to 12 iff the rule below fires

# Deployed per-leg dict state this campaign was designed against (verified 2026-07-24 on the
# freeze-2026-07-23-gate-b tree; the driver refuses to run if the live dicts differ).
R6X_DEPLOYED_BOOST = {"eBOSS": 1.0, "DESI": 1.0, "KS": 2.5}
R6X_DEPLOYED_FRAC = {"eBOSS": 0.287, "DESI": 0.287, "KS": 0.40}
R6X_PARAMETERIZATION = {"eBOSS": "alpha_pivot_powerlaw_v1", "DESI": "alpha_pivot_powerlaw_v1",
                        "KS": "dndx_mapped_v2"}

# PRE-REGISTERED hcd_prior_signature hexes (computed ONCE from the frozen tree 2026-07-24 and
# pinned; the deployed hex equals the analysis.lock pin). The analyzer refuses any pkl whose
# signature is not the exact (deployed, displaced[leg]) PAIR — never signature EQUALITY across
# arms (panel revision 5: analyze_r6_pairs.py:86 asserts equality; r6x arms differ by design).
R6X_DEPLOYED_HEX = "50befc941edfc4c789286d2054d0107f1eb19a5672bcb86131426fb101eea216"
R6X_DISPLACED_HEX = {
    "eBOSS": "84d38e16c2de95ceb54909c404cc5c8689c2cfe629092c7e9a65d73976a03ad2",
    "DESI": "db30cc39abed8979e9ef4f14ee1b550fbcf5f4d4ce5934cb1113de729e0010fc",
    "KS": "b822af6e0c373b1668ad50f1c66fafe213078def0599829d113d089e54959898",
}

# The boost-1.0 LLS alpha-pivot centre (hcd_lls_realfit_alpha_center at the frozen cache's
# Xbar(z=3)=0.6316034425658955; the documented 0.17211216 at full precision). Every realized
# centre expectation is stored_boost * THIS value, replicating the frozen float arithmetic
# (hcd_lls_realfit_alpha_center returns float(boost) * float(base)).
R6X_ALPHA_LLS_MU_B1 = 0.17211216067355312

# KS mapped LLS pivot centres (w_c_corrected of the boosted dN/dX reference at z=3), computed
# once from the frozen construction and pinned; both INSIDE the frozen band (0.326, 0.476).
R6X_KS_MAPPED_PIVOT = {"deployed": 0.39323548008485276, "dispprior": 0.47194501237050596}

# KS displacement in eps_lls prior-sigma units: ln(1.287) / KS_DNDX_SIGMA_EPS(=0.5310).
R6X_KS_EPS_SIGMA_DISP = float(np.log(R6X_FACTOR) / 0.5310)   # = +0.4752

# Per-leg displacement semantics (stamped verbatim into every displaced pkl).
R6X_DISP_SPEC = {
    "eBOSS": ("LLS prior centre +1.0 deployed prior sigma at fixed absolute width: "
              "HCD_LLS_SURVEY_BOOST['eBOSS'] 1.0->1.287, HCD_LLS_SURVEY_FRAC_SIGMA['eBOSS'] "
              "0.287->0.287/1.287 (in-place dict item assignment)"),
    "DESI": ("LLS prior centre +1.0 deployed prior sigma at fixed absolute width: "
             "HCD_LLS_SURVEY_BOOST['DESI'] 1.0->1.287, HCD_LLS_SURVEY_FRAC_SIGMA['DESI'] "
             "0.287->0.287/1.287 (in-place dict item assignment)"),
    "KS": ("dN/dX-premap LLS reference centre x1.287 at all z (= +0.4752 eps_lls prior sigma; "
           "the same multiplicative centre displacement as eBOSS/DESI, pre-registered as the "
           "common yardstick): HCD_LLS_SURVEY_BOOST['KS'] 2.5->3.2175 (in-place dict item "
           "assignment). FRAC_SIGMA['KS'] deliberately untouched: inert on the mapped branch "
           "(widths are the pinned KS_DNDX_SIGMA_* literals -> fixed absolute width holds), "
           "and mutating an inert signature-carried knob would desync signature from prior."),
}

# Pre-registered readout scales (the KS R6 record, 2026-07-23 readout memo): the sensitivity
# target this campaign must resolve or exclude. theta-unit -> physical: n_s box 0.25, A_p box
# 1.4e-9 (the R6 memo conventions). Under-resolution rule (panel revision 7): if a leg's paired
# SE exceeds these theta-unit scales, the readout must SAY "under-resolves", never
# "consistent with zero" alone.
R6X_THETA_BOX = {"ns": 0.25, "Ap": 1.4e-9}
R6X_TARGET_SE_THETA = {"ns": 0.012, "Ap": 0.0265}   # KS-R6-scale effects in theta units

# TILT VERDICT (design deliverable 2, verified 2026-07-24; tests/test_crossleg_r6.py pins the
# mechanics): tilt arms are DROPPED. No freeze-safe signature-carried knob exists for the LLS
# z-slope prior CENTRE:
#   * The realized centre is closure_legb's from-imported HCD_LLS_REALFIT_ZSLOPE (an IMMUTABLE
#     float, closure_legb.py:56 + :801) -- there is no shared mutable container, so no in-place
#     mutation path exists. Rebinding inference.HCD_LLS_REALFIT_ZSLOPE moves the signature (it
#     is in the payload) but NOT the built ctx (closure_legb keeps its load-time binding): the
#     worst-case desync direction. Rebinding closure_legb's binding moves the ctx but NOT the
#     signature. A dual rebind is exactly the module-attribute-rebinding mechanism panel
#     revision 4 forbids, and on KS the same constant also feeds the mapped dN/dX reference
#     construction (_ks_dndx_reference, closure_legb.py:843), so a "slope-prior-only" dual
#     rebind would silently reshape the KS mapped reference too (incoherent).
#   * HCD_INCIDENCE_SLOPE and ZSLOPE_PRIOR_SIGMA live in closure_legb and are covered by NO
#     signature (hcd_prior_constants_payload docstring, explicit) -- moving them can never move
#     the signature coherently.
# Displacement convention that WOULD have applied (recorded for the future migration study):
# +1 slope-prior-sigma = +0.52 on the 2.127 centre (ZSLOPE_PRIOR_SIGMA[0]); the alternative
# migration-amplitude-matched choice is the displacement that reproduces the measured migration
# extreme (+1.41 sigma_prior amplitude-equivalent at eBOSS z=4.6, i.e. ~+0.85 on the slope for
# the eBOSS z-range). Per the execution annex this is RECORDED AS A LIMITATION, not escalated:
# the amplitude arms + the deferred migration study still answer the r6x scope.
R6X_TILT_VERDICT = "DROPPED: no freeze-safe signature-carried z-slope-centre override knob"


def pkl_name(leg, arm, mock, *, smoke=False):
    """Distinct pkl prefix r6x_<leg>_<arm>_shard_NNN[.smoke].pkl. The r6x_ prefix matches NO
    existing analyzer glob (defense in depth on top of the r6x_override stamp refusal)."""
    assert leg in R6X_LEGS, f"unknown leg {leg!r}"
    assert arm in R6X_ARMS, f"unknown arm {arm!r}"
    suffix = ".smoke" if smoke else ""
    return f"r6x_{leg.lower()}_{arm}_shard_{int(mock):03d}{suffix}.pkl"


def expected_hex(leg, arm):
    """The pre-registered hcd_prior_signature hex for one (leg, arm)."""
    assert arm in R6X_ARMS, f"unknown arm {arm!r}"
    if arm == "deployed":
        return R6X_DEPLOYED_HEX
    return R6X_DISPLACED_HEX[leg]


def displaced_dict_values(leg):
    """(boost, frac) the displaced arm's dicts must carry for ``leg``; frac is None for KS
    (deliberately untouched on the mapped branch, see R6X_DISP_SPEC)."""
    assert leg in R6X_LEGS, f"unknown leg {leg!r}"
    boost = R6X_DEPLOYED_BOOST[leg] * R6X_FACTOR
    frac = None if leg == "KS" else R6X_FRAC_DISP
    return boost, frac


def verify_deployed_prior_state(leg):
    """Refuse to run unless the live dicts + signature are the exact deployed state this
    campaign pre-registered (the --expect / K1a-tripwire pattern). Module-attribute reads."""
    from hcd_analysis.emulator import inference as INF
    from hcd_analysis.emulator import closure_legb as CL
    # the mechanism's load-bearing precondition: closure_legb's from-imported names are THE
    # SAME OBJECTS (item assignment propagates); a refactor that breaks this must fail here.
    assert CL.HCD_LLS_SURVEY_BOOST is INF.HCD_LLS_SURVEY_BOOST, \
        "closure_legb.HCD_LLS_SURVEY_BOOST is no longer the inference dict object: the " \
        "in-place override mechanism is broken (from-import identity lost) -- refusing"
    assert CL.HCD_LLS_SURVEY_FRAC_SIGMA is INF.HCD_LLS_SURVEY_FRAC_SIGMA, \
        "closure_legb.HCD_LLS_SURVEY_FRAC_SIGMA is no longer the inference dict object -- refusing"
    b = float(INF.HCD_LLS_SURVEY_BOOST[leg])
    f = float(INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg])
    assert b == R6X_DEPLOYED_BOOST[leg], (
        f"deployed HCD_LLS_SURVEY_BOOST[{leg!r}] = {b} != pre-registered "
        f"{R6X_DEPLOYED_BOOST[leg]} -- the prior state drifted; refusing to run")
    assert f == R6X_DEPLOYED_FRAC[leg], (
        f"deployed HCD_LLS_SURVEY_FRAC_SIGMA[{leg!r}] = {f} != pre-registered "
        f"{R6X_DEPLOYED_FRAC[leg]} -- refusing to run")
    sig = INF.hcd_prior_signature()
    assert sig == R6X_DEPLOYED_HEX, (
        f"live hcd_prior_signature {sig[:12]}... != the pre-registered deployed hex "
        f"{R6X_DEPLOYED_HEX[:12]}... -- prior constants drifted since pre-registration; "
        f"refusing to run")


@contextlib.contextmanager
def r6x_override(leg):
    """Apply the displaced-arm override for ``leg`` by IN-PLACE DICT ITEM ASSIGNMENT (never a
    rebind), assert the signature moved to the pre-registered displaced hex, and RESTORE (with
    re-verification, including on exception) in the finally block."""
    from hcd_analysis.emulator import inference as INF
    verify_deployed_prior_state(leg)
    boost_disp, frac_disp = displaced_dict_values(leg)
    b0 = INF.HCD_LLS_SURVEY_BOOST[leg]
    f0 = INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg]
    try:
        INF.HCD_LLS_SURVEY_BOOST[leg] = boost_disp            # in-place item assignment
        if frac_disp is not None:
            INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg] = frac_disp    # in-place item assignment
        sig = INF.hcd_prior_signature()
        assert sig != R6X_DEPLOYED_HEX, "override applied but hcd_prior_signature did NOT move"
        assert sig == R6X_DISPLACED_HEX[leg], (
            f"displaced hcd_prior_signature {sig[:12]}... != pre-registered "
            f"{R6X_DISPLACED_HEX[leg][:12]}... for {leg} -- the override realized a different "
            f"prior state than pre-registered; refusing")
        yield dict(boost=boost_disp, frac=(f0 if frac_disp is None else frac_disp))
    finally:
        INF.HCD_LLS_SURVEY_BOOST[leg] = b0                    # restore, in place
        INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg] = f0
        # RESTORE-AND-VERIFY (panel revision 4): values AND signature must be deployed again.
        assert float(INF.HCD_LLS_SURVEY_BOOST[leg]) == R6X_DEPLOYED_BOOST[leg], \
            f"restore failed: HCD_LLS_SURVEY_BOOST[{leg!r}] != deployed"
        assert float(INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg]) == R6X_DEPLOYED_FRAC[leg], \
            f"restore failed: HCD_LLS_SURVEY_FRAC_SIGMA[{leg!r}] != deployed"
        assert INF.hcd_prior_signature() == R6X_DEPLOYED_HEX, \
            "restore failed: hcd_prior_signature != the deployed hex after restoration"


def expected_lls_centre_width(leg, arm):
    """(mu, sigma) the BUILT ctx's LEGACY alpha-space LLS prior slot must realize, replicating
    the frozen float arithmetic exactly (float equality asserted on the built ctx):
    mu = stored_boost * base (hcd_lls_realfit_alpha_center), sigma = stored_frac * mu
    (_survey_alpha_prior). On KS these are the DORMANT legacy vectors (mapped era) but the
    boost still scales them, so they remain a valid override witness."""
    boost = R6X_DEPLOYED_BOOST[leg] if arm == "deployed" else R6X_DEPLOYED_BOOST[leg] * R6X_FACTOR
    frac = R6X_DEPLOYED_FRAC[leg]
    if arm == "dispprior" and leg != "KS":
        frac = R6X_FRAC_DISP
    mu = boost * R6X_ALPHA_LLS_MU_B1
    return mu, frac * mu


def extension_verdict(t_ns, t_ap):
    """The pre-registered n=12 extension rule (panel revision 3, either-channel): extend iff
    n_s OR A_p lands at 1.5 <= |t| < 3. Returns (extend: bool, verdict_line: str)."""
    hits = [nm for nm, t in (("n_s", t_ns), ("A_p", t_ap))
            if np.isfinite(t) and 1.5 <= abs(t) < 3.0]
    extend = bool(hits)
    line = ("EXTENSION RULE: EXTEND to n=12 pairs (pre-registered: {} at 1.5 <= |t| < 3; "
            "all-12 pooled mean to be quoted with the extension flagged)".format(", ".join(hits))
            if extend else
            "EXTENSION RULE: no extension (neither n_s nor A_p in 1.5 <= |t| < 3; "
            "t_ns={:+.2f}, t_Ap={:+.2f})".format(t_ns, t_ap))
    return extend, line


def sha256_of_arrays(*arrays):
    """Stable sha256 over the concatenated float64 bytes of the given arrays (the data / truth
    bit-identity certificate stamped per mock and asserted across the arm pair)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    return h.hexdigest()
