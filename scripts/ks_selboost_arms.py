"""KS selection-function mock-challenge ARM REGISTRY (spec 2026-07-18 + Amendment 2, PI A2.8
sign-offs 2026-07-19: "A-F all default"; second ask APPROVED => the RESTORED-N battery).

The registry is the SINGLE python-side source of the campaign arm matrix (no JSON through bash
quoting): arm_id -> {class-key: z-profile spec} in the closure_legb scalar-or-dict hook units
(truth boost B(z) applied MULTIPLICATIVELY to the class truth column, RELATIVE to the deployed
post-K1a KS prior-center curve, which already carries HCD_LLS_SURVEY_BOOST["KS"]=2.5 x the
corrected lit center x the lit z-slope 2.127). Log-quadratic family (Amendment 1):

    B(z) = b_pivot * exp(eta1*x + eta2*x^2),   x = ln((1+z)/(1+Z_PIVOT)),  Z_PIVOT = 3.0.

ENVELOPE (A2.1, machine-checked in tests/test_ks_selboost_arms.py): every envelope="in" arm
satisfies 0.8 <= B(z) <= 3.2 (relative units == cosmic 2x-8x over the deployed 2.5 center) for
ALL z in [2.4, 4.6]. K1 (the PI's truth-2.5-vs-center-3.5 displacement MIRROR) and C2 (the
adversarial corner) are explicitly labeled envelope="out": exempt from the envelope check and
from the Part-1 in-envelope gate set (C2 still counts in the binding Part-2 max if triggered,
flagged separately, per the recorded D4).

K5_joint_meas (D3, A2.5.1): the truth vector is the PRIYA-MEASURED DLA-conditional incidence
multiplier B_c(z) = l_c(X | segment has a DLA)/l_c(X), LLS + subDLA columns at L=120 Mpc/h,
pooled-suite rows (the single-sim rows 2.01/2.39/2.67/3.27 carry no suite scatter and are
excluded per the table caveat), loglog-interpolated, provenance sha256-PINNED at ingest (a
silently swapped measurement fails loud). Rows span z 2.0..5.4 so the table covers the full
ctx.z_global union grid (DESI rows reach below the KS band) with MEASURED values — no clamping
needed. zQSO caveat (PI 2026-07-19): the z-TREND of B_c(z) is a fixed-z in-box conditional and
must not be read as the real selection function's shape; K5 uses it as the clustering-channel
magnitude scale only (null-bound arm).

Pure numpy on purpose (login-node safe; the analyzer imports this). ``eval_profile`` here must
agree with the deployed hook evaluator closure_legb.eval_boost_profile to float precision —
enforced by test_registry_evaluator_matches_hook_evaluator.
"""
import hashlib
import json

import numpy as np

Z_PIVOT = 3.0                 # convention B; == inference.HCD_Z_PIVOT (test-pinned)
KS_Z_RANGE = (2.4, 4.6)
ENVELOPE_REL = (0.8, 3.2)     # A2.1: relative-boost envelope == cosmic 2x-8x over the 2.5 center
KS_DEPLOYED_LLS_BOOST = 2.5   # the deployed KS survey boost (cosmic-units conversion reference)

K5_TABLE_PATH = ("/home/mfho/hcd_priya_notes/docs/superpowers/corrected-dndx-artifacts/"
                 "conditional_incidence_table.txt")
# provenance pin (D3 measurement of record, 2026-07-19): sha256 of the committed table file.
# An updated measurement silently swapped under a running campaign fails loud here AND at the
# analyzer ingest re-assert (same philosophy as the runner prior-state tripwire).
K5_TABLE_SHA256 = "40931a5dc62c968150f39184f6191d5159863935bf0ad24167fddf9b3306edc9"

_PLAN_CPUH_PER_FIT = 5.0      # Sec-7 planning anchor (measured KS ~3.8 at 550 steps; plan at 5)
_CONTINGENCY = 1.5
_PILOT_CPUH = 40.0

# --------------------------------------------------------------------------------------------- #
#  The signed arm matrix (A2.2 + A2.8-b restored Ns). quad = (b_pivot, eta1, eta2), or None
#  (clean control), or "measured" (K5, resolved from the provenance-pinned table at spec build).
#  part1: membership in the binding Part-1 in-envelope gate set ("if_in_envelope" = K5's
#  pre-registered readout rule, A2.8-e). surface_fit: included in the pooled 3-vector
#  (S_A, S_eta1, S_eta2) response-surface fit (K5 excluded — subDLA component would contaminate;
#  C2 excluded — out-of-envelope corner, the surface interpolates the interior).
# --------------------------------------------------------------------------------------------- #
ARMS = {
    "K0_clean": dict(
        quad=None, n_mocks=16, envelope="in", default_run=True, part1=False, surface_fit=False,
        purpose="control, matched center; shared pairing baseline; SBC/rank reference"),
    "K1_flat_lo": dict(
        quad=(0.714, 0.0, 0.0), n_mocks=12, envelope="out", default_run=True, part1=False,
        surface_fit=True,
        purpose="D1 low-side amplitude (truth 2.5x vs center 3.5x displacement MIRROR); "
                "cosmic 1.79 flat sits below the 2.0 floor -> envelope='out', NOT in Part 1"),
    "K2_flat_hi": dict(
        quad=(1.667, 0.0, 0.0), n_mocks=16, envelope="in", default_run=True, part1=True,
        surface_fit=True,
        purpose="D1 high-side amplitude (truth 2.5x vs center 1.5x); DESI-comparable; "
                "Part-1 gate arm"),
    "K3_rising": dict(
        quad=(1.25, 2.7, 0.0), n_mocks=16, envelope="in", default_run=True, part1=True,
        surface_fit=True,
        purpose="strong-rise corner, calibrated Ho slope +2.7 (b0=1.25 keeps the low-z end on "
                "the 2.0-cosmic floor); anchors S_eta1; Part-1 gate arm"),
    "K4_u_paper": dict(
        quad=(0.81, -0.35, 13.0), n_mocks=16, envelope="in", default_run=True, part1=True,
        surface_fit=True,
        purpose="paper-like U (eta2>0), THE realistic criterion-bearing member: cosmic 3.0 at "
                "z=2.4, trough 2.0 near z=3.05, 4.5 at z=4.2, 7.8 at z=4.6; Part-1 gate arm"),
    "K6_inv_u": dict(
        quad=(2.67, 3.83, -22.0), n_mocks=12, envelope="in", default_run=True, part1=True,
        surface_fit=True,
        purpose="inverted-U corner (eta2<0), the marginalization-projection alternative; "
                "two-sides eta2 (identification + e2-sign linearity test); Part-1 gate arm"),
    "K7_falling": dict(
        quad=(2.00, -2.7, 0.0), n_mocks=12, envelope="in", default_run=True, part1=True,
        surface_fit=True,
        purpose="falling corner (cosmic 7.8 -> 2.0), promoted from contingency C1 to a DEFAULT "
                "arm by the approved second ask (A2.8-b); two-sides eta1; Part-1 gate arm"),
    "K5_joint_meas": dict(
        quad="measured", n_mocks=8, envelope="in", default_run=True, part1="if_in_envelope",
        surface_fit=False, part2_binding=False,   # FLAG A RESOLUTION (spec launch log,
        # PI-signed A2.8-c): the null-bound arm is EXCLUDED from the binding Part-2 max — at
        # its measured displacement D~0.13 the N=8 noise floor 2SE/D~1.3 makes it mechanically
        # unfailable-or-unpassable on noise alone; it reports a separate null-bound line.
        purpose="D3 joint-class NULL-BOUND arm at the measured DLA-conditional incidence "
                "(B_LLS ~1.12->1.03, B_subDLA ~1.29->1.08); detects gross non-additivity only "
                "(linear prediction ~0.05-0.07 sigma_post); OUT of the LLS surface fit"),
    # ---- pre-registered contingencies (trigger-gated, NOT run by default) -------------------- #
    "C1_falling": dict(
        quad=(2.00, -2.7, 0.0), n_mocks=12, envelope="in", default_run=False, part1=True,
        surface_fit=True,
        trigger="collapse-check failure, K1/K2 sign-asymmetry, or K3-vs-surface residual "
                "> 2 SE. VOID while K7_falling runs as a default arm (A2.4: the approved "
                "second ask absorbs C1 entirely) — registered for the ladder-descope branch "
                "in which K7 is dropped.",
        purpose="envelope-legal falling contingency (re-specced from the old b0=1 form)"),
    "C2_adversarial": dict(
        quad=(1.30, 6.5, 0.0), n_mocks=8, envelope="out", default_run=False, part1=False,
        surface_fit=False,
        trigger="K3 and K4 disagree qualitatively (slope-prior partial-absorption ambiguity), "
                "or the PI rules an adversarial corner INTO the Part-B max.",
        purpose="OUT-OF-ENVELOPE adversarial corner (cosmic ~29x at z=4.6; emulator-span "
                "caveat applies with force); counts in the binding Part-2 max if run (recorded "
                "D4), flagged separately in the readout"),
}

_CLS_KEYS = ("lls_truth_boost", "subdla_truth_boost")


def arm_ids():
    """All registered arm ids (default + contingency), registry order."""
    return list(ARMS)


def default_arm_ids():
    """The arms of the signed default campaign (second-ask matrix; 108 fits)."""
    return [a for a, e in ARMS.items() if e["default_run"]]


def campaign_cost_cpuh():
    """Cost ledger at the registered Ns and the Sec-7 planning anchor (5 CPU-h/fit):
    nominal, and worst-case = nominal x 1.5 contingency + the 40 CPU-h pilot (== the approved
    850 CPU-h second-ask envelope)."""
    fits = sum(ARMS[a]["n_mocks"] for a in default_arm_ids())
    nominal = fits * _PLAN_CPUH_PER_FIT
    return dict(fits=fits, nominal=nominal,
                worst_case=nominal * _CONTINGENCY + _PILOT_CPUH)


# --------------------------------------------------------------------------------------------- #
#  Profile evaluation (pure numpy; must track closure_legb.eval_boost_profile exactly).
# --------------------------------------------------------------------------------------------- #
def eval_profile(profile, z):
    """Evaluate a z-profile spec (log-quadratic or loglog-tabulated) at ``z`` — the login-node
    numpy twin of closure_legb.eval_boost_profile (agreement test-enforced). Scalar z -> float."""
    z_arr = np.atleast_1d(np.asarray(z, float))
    keys = frozenset(profile)
    if keys == frozenset({"b_pivot", "eta1", "eta2"}):
        x = np.log((1.0 + z_arr) / (1.0 + Z_PIVOT))
        B = float(profile["b_pivot"]) * np.exp(float(profile["eta1"]) * x
                                               + float(profile["eta2"]) * x * x)
    elif keys == frozenset({"z", "boost", "interp"}):
        assert profile["interp"] == "loglog", profile["interp"]
        zt = np.asarray(profile["z"], float)
        bt = np.asarray(profile["boost"], float)
        assert np.all(np.diff(zt) > 0) and np.all(bt > 0)
        assert z_arr.min() >= zt[0] and z_arr.max() <= zt[-1], "table does not cover the grid"
        B = np.exp(np.interp(np.log1p(z_arr), np.log1p(zt), np.log(bt)))
    else:
        raise ValueError(f"unknown profile key-set {sorted(keys)}")
    return B if np.ndim(z) else float(B[0])


def load_k5_measured_tables(path=K5_TABLE_PATH):
    """Ingest the D3 measured conditional-incidence table: pooled-suite rows (nsnap > 1) of the
    header table's L=120 columns. Returns dict(z, B_lls, B_subdla, sha256, path, n_rows).
    PROVENANCE: ALWAYS asserts the file's sha256 equals the pinned K5_TABLE_SHA256 — the
    campaign must never run on an unpinned measurement file, whatever path it came from."""
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == K5_TABLE_SHA256, (
        f"K5 measurement table sha256 {sha[:16]}... != pinned {K5_TABLE_SHA256[:16]}... "
        f"({path}); the D3 measured vector of record has been swapped/edited — re-pin "
        f"deliberately or restore the file")
    z, b_lls, b_sub = [], [], []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        # header-table rows: z nsnap B_LLS(L=120)+-e(s) B_subDLA(L=120)+-e(s) B_DLA ... — keyed
        # by the 2nd column being an integer nsnap (the section tables below lack it).
        if len(parts) >= 5 and parts[1].isdigit():
            nsnap = int(parts[1])
            if nsnap <= 1:
                continue                    # single-sim rows: no suite scatter (table caveat)
            z.append(float(parts[0]))
            b_lls.append(float(parts[2].split("+-")[0]))
            b_sub.append(float(parts[3].split("+-")[0]))
    z = np.asarray(z, float)
    order = np.argsort(z)
    out = dict(z=z[order], B_lls=np.asarray(b_lls, float)[order],
               B_subdla=np.asarray(b_sub, float)[order], sha256=sha, path=str(path),
               n_rows=int(z.size))
    assert out["n_rows"] >= 10 and np.all(np.diff(out["z"]) > 0), "K5 table parse failed"
    assert out["z"][0] <= 2.2 and out["z"][-1] >= 4.6, \
        "K5 table must cover the z_global union grid (no silent extrapolation in the hook)"
    return out


def arm_inject_spec(arm_id, table_path=K5_TABLE_PATH):
    """The run_legb inject_spec for one arm: None for the clean control, else
    {class-key: profile-spec}. K5 resolves BOTH class tables from the provenance-pinned
    measurement file. KeyError on an unknown arm (fail loud)."""
    e = ARMS[arm_id]
    if e["quad"] is None:
        return None
    if e["quad"] == "measured":
        t = load_k5_measured_tables(table_path)
        return {
            "lls_truth_boost": {"z": list(t["z"]), "boost": list(t["B_lls"]),
                                "interp": "loglog"},
            "subdla_truth_boost": {"z": list(t["z"]), "boost": list(t["B_subdla"]),
                                   "interp": "loglog"},
        }
    b0, e1, e2 = e["quad"]
    return {"lls_truth_boost": {"b_pivot": float(b0), "eta1": float(e1), "eta2": float(e2)}}


def arm_boost_B(arm_id, z, table_path=K5_TABLE_PATH):
    """Evaluate every class profile of ``arm_id`` on ``z``: {class-key: B array}. The clean
    control returns unit boosts for both classes."""
    spec = arm_inject_spec(arm_id, table_path)
    if spec is None:
        return {k: np.ones_like(np.atleast_1d(np.asarray(z, float))) for k in _CLS_KEYS}
    return {cls: np.atleast_1d(eval_profile(prof, z)) for cls, prof in spec.items()}


def _registry_payload(table_path=K5_TABLE_PATH):
    """JSON-native payload behind registry_signature: the resolved profiles (K5 tables
    included), Ns, envelope labels, default flags, and the K5 provenance hash."""
    payload = {}
    for aid, e in ARMS.items():
        payload[aid] = dict(
            profiles=arm_inject_spec(aid, table_path), n_mocks=e["n_mocks"],
            envelope=e["envelope"], default_run=e["default_run"], part1=e["part1"],
            surface_fit=e["surface_fit"])
    payload["_meta"] = dict(z_pivot=Z_PIVOT, envelope_rel=list(ENVELOPE_REL),
                            ks_z_range=list(KS_Z_RANGE),
                            k5_table_sha256=K5_TABLE_SHA256)
    return payload


def registry_signature(table_path=K5_TABLE_PATH):
    """Stable sha256 hex digest over the canonical-JSON registry payload — stamped into every
    shard pkl; the analyzer ingest asserts homogeneity (mixed-registry campaigns must not
    pool)."""
    return hashlib.sha256(
        json.dumps(_registry_payload(table_path), sort_keys=True).encode("utf-8")).hexdigest()


def batch_cells():
    """SLURM array-id -> (arm_id, mock) mapping over the signed default matrix (registry
    order, K0 first): 108 cells at the second-ask Ns. Numpy-light on purpose — the batch
    script resolves its task id through THIS module without importing the JAX-heavy runner."""
    return [(aid, m) for aid in default_arm_ids() for m in range(ARMS[aid]["n_mocks"])]
