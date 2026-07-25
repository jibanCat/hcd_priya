"""KS X-battery ARM REGISTRY (campaign "X-battery-2"; PROPOSAL-extreme-battery-v2 as adopted
by PI decisions of record #7, 2026-07-24, execution annex OQ1-11).

NEW module by design (K6-redesign Sec 8 freeze classification): the frozen legacy registry
`scripts/ks_selboost_arms.py` and its pinning test are UNTOUCHED. This registry carries its
OWN `registry_signature` over a canonical payload that INCLUDES the truth-table sha, so an
X-battery pkl can never pool with the legacy 108-fit campaign, with K-battery pkls, or with a
battery run against a different truth table.

Arms (annex OQ4 commissions X4; OQ7 adopts the naming; n = 16 per X arm, 8 per K8 sub-arm):

  X1_dla100   100% DLA-selected sightlines, DLAs masked. Data-side truth-P1D substitution
              (no alpha coefficient reaches the masked conditional DLA direction on KS,
              KS_DLA_FORWARD_FRAC = 0). PRIMARY fork = dilution-CORRECTED (annex OQ1);
              the diluted fork X1b is a READOUT OVERLAY only (no fits).
  X2_sub100   100% subDLA-selected (highest-class partition, annex OQ10; deployed trough-fill
              convention, annex OQ9). Mixture corner alpha = (0,1,0) at all z via the frozen
              run_legb truth_fn hook.
  X3_lls100   100% LLS-selected. Mixture corner alpha = (1,0,0) at all z.
  X4_prof     profiled LLS selection sub-arm: inverted-U f_sel(z) (eta1=3.83, eta2=-22.0,
              peak-normalized: f=1 at z=3.364, 0.254 at the band edges), applied as mixture
              weights alpha_LLS(z) = f_sel(z), other classes 0. NON-binding.
  K8a/b/c     the surviving K6-redesign Sec 7 in-manifold calibration battery: truths
              displaced in dN/dX space by exactly +/-1 DEPLOYED prior sigma
              (eps_lls +/- sigma_eps, kappa_lls + sigma_kappa) through the frozen
              w_c_corrected occupancy map (read-only import in the driver). NON-binding.

Part-1 binding membership (proposal Sec 6 + panel gate reform): X1/X2/X3 binding for KS
unblinding; X4 and K8 non-binding diagnostics. Gate: |mean paired delta| + 2 SE
< 0.30 sigma_post per parameter (ns, Ap), sigma_post = the K0-POOLED posterior sd
(round-2 revision 4a, pre-registered). Part-2 S is DISCLOSURE-ONLY, printed with its
pre-registered P(fail|null) from the stage-V gate-power inputs.

K0 pairing (annex OQ5): the R4/R5 corrected-geometry K0 shards are REUSED across registry
signatures under a PRE-REGISTERED allowance; the analyzer asserts the pair identity
(shared clean truth draw + stamps) per pair. Pair identity REQUIRES the K0 shards' seed
(20260615, stamped in their run_kw): see PAIR_SEED below.

Pre-registered readout caveats (runbook v2.1): the runner's emulator-span WARNING fires BY
DESIGN at the corners (truth alpha above the training-cache structural w_c ceiling); alpha
ranks are degenerate at the corners and rank uniformity is NOT expected on any displaced-
truth arm; the mock-noise convention at the X1 corner keeps clean-composition C_emu weights
(disclosed, not repaired). The gate is n_s/A_p bias, never alpha recovery.

Truth tables: NOT available until the stage-V validation pass emits them
(scripts/analyze_xsel_truth_tables.py, parallel build). Contract:
scripts/xsel_truth_contract.md. The loader below fails LOUD on a missing pin, missing file,
sha mismatch, missing keys, or a drifted convention stamp.

Pure numpy on purpose (login-node safe; the analyzer and the batch script import this).
No em-dashes in this file (repo convention).
"""
import hashlib
import json
import os

import numpy as np

CAMPAIGN = "X-battery-2"
Z_PIVOT = 3.0                    # convention B; == inference.HCD_Z_PIVOT
KS_Z_RANGE = (2.4, 4.6)
GATE = 0.30                      # Part-1: |mean paired delta| + 2 SE < GATE * sigma_post
SIGMA_POST_CONVENTION = "K0_pooled_posterior_sd"   # round-2 revision 4a, pre-registered

# PAIRING SEED OF RECORD. The adopted proposal (Sec 6 + round-2 revision 4b) pairs every arm
# 1:1 against the reused R4/R5 corrected-geometry K0 baseline with SHARED truth-theta and
# noise keys: that is only realizable at the K0 shards' stamped seed (20260615, the
# cross-campaign constant; verified on cert_2026-07/ks_rerun shard 000). The build brief's
# "fresh seed 20260724 + fold_in domain tag" wording is therefore NOT adopted for the
# truth/noise keys (it would break the pair-identity assertions that OQ5's K0 reuse is
# conditioned on); domain separation from the r6x campaign (seed 20260724) is automatic
# because the seeds differ. DEVIATION RECORDED in the build memo.
PAIR_SEED = 20260615
K0_DIR_DEFAULT = "/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/ks_rerun"
K0_ARM_ID = "K0_clean"
K0_N_MOCKS = 16

_PLAN_CPUH_PER_FIT = 5.0         # registry planning anchor (round-2 revision 7)
_CONTINGENCY = 1.5

# ---------------------------------------------------------------------------------------- #
#  X4 inverted-U selection profile (proposal Sec 4): the retired K6 shape in SELECTION space,
#  peak-normalized so the benchmark reaches 100% at the profile peak.
#      f_sel(z) = exp(eta1*x + eta2*x^2 - eta1^2/(4*|eta2|)),  x = ln((1+z)/(1+Z_PIVOT)).
#  f_sel in (0, 1] by construction; f = 1 at z = 3.364; f(2.4) = f(4.6) = 0.254.
# ---------------------------------------------------------------------------------------- #
X4_ETA1 = 3.83
X4_ETA2 = -22.0
X4_PEAK_Z = float((1.0 + Z_PIVOT) * np.exp(-X4_ETA1 / (2.0 * X4_ETA2)) - 1.0)  # 3.3639


def f_sel(z, eta1=X4_ETA1, eta2=X4_ETA2):
    """Peak-normalized inverted-U selected fraction f_sel(z) in (0, 1]. Scalar in, float out."""
    assert eta2 < 0.0, f"f_sel is only peak-normalizable for eta2 < 0, got {eta2}"
    z_arr = np.atleast_1d(np.asarray(z, float))
    assert np.all(np.isfinite(z_arr)) and np.all(z_arr > -1.0), "bad z grid for f_sel"
    x = np.log((1.0 + z_arr) / (1.0 + Z_PIVOT))
    f = np.exp(eta1 * x + eta2 * x * x - eta1 * eta1 / (4.0 * abs(eta2)))
    assert np.all(f > 0.0) and np.all(f <= 1.0 + 1e-12), "f_sel left (0, 1] (bug)"
    f = np.minimum(f, 1.0)
    return f if np.ndim(z) else float(f[0])


# ---------------------------------------------------------------------------------------- #
#  Mixture arithmetic (alpha rows are (nZ, 3) in class order LLS, subDLA, DLA).
# ---------------------------------------------------------------------------------------- #
def corner_alpha_rows(n_z, cls_idx):
    """100%-selected corner: alpha_cls = 1, other classes 0, at every z. (nZ,3)."""
    assert cls_idx in (0, 1), \
        f"mixture corners exist only for LLS (0) / subDLA (1); DLA (X1) is data-side, got {cls_idx}"
    rows = np.zeros((int(n_z), 3), float)
    rows[:, cls_idx] = 1.0
    return rows


def profile_alpha_rows(z, cls_idx):
    """Profiled selection: alpha_cls(z) = f_sel(z), other classes 0. (nZ,3)."""
    assert cls_idx in (0, 1), f"profiled mixture is LLS/subDLA only, got {cls_idx}"
    z_arr = np.asarray(z, float)
    rows = np.zeros((z_arr.size, 3), float)
    rows[:, cls_idx] = np.atleast_1d(f_sel(z_arr))
    return rows


def mixture_ratio(f, ratio_cond):
    """Exact mixture identity for a selected fraction f against a conditional/clean ratio R:
    P/(P_clean) = 1 + f*(R - 1). Used for the analyzer-side X4 derived curve and the X1
    data-side blend bookkeeping."""
    return 1.0 + np.asarray(f, float) * (np.asarray(ratio_cond, float) - 1.0)


def truth_admissible(alpha_rows, *, strict_interior=False, atol=1e-9):
    """Per-mock truth-admissibility tripwire: every row has alpha >= 0 and sum(alpha) <= 1
    (the mixture simplex; the corner sum == 1 is ADMISSIBLE under Decision 3).
    strict_interior=True additionally requires sum(alpha) < 1 (the K8 dN/dX-image arms are
    interior by construction; a boundary row there means the map recompute broke)."""
    a = np.asarray(alpha_rows, float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        return False
    s = a.sum(axis=1)
    ok = bool(np.all(a >= -atol) and np.all(s <= 1.0 + atol))
    if strict_interior:
        ok = ok and bool(np.all(s < 1.0 - 1e-12))
    return ok


# ---------------------------------------------------------------------------------------- #
#  The signed arm matrix. kind:
#    data_swap        X1: data-side truth-P1D substitution (driver reimplements the run_legb
#                     per-mock loop from frozen callables; swap-off byte-identity gated).
#    mixture_corner   X2/X3: truth_fn corner override (alpha_cls = 1 at all z).
#    mixture_profile  X4: truth_fn profiled override (alpha_LLS(z) = f_sel(z)).
#    dndx_displaced   K8: truth_fn dN/dX-space displacement through the frozen map.
#  sigma_expect pins the DEPLOYED width the displacement resolves against (driver asserts
#  the live inference constant equals it: a drifted prior fails loud, selboost precedent).
# ---------------------------------------------------------------------------------------- #
ARMS = {
    "X1_dla100": dict(
        kind="data_swap", cls=2, n_mocks=16, part1=True,
        table_key="ratio_rows_X1_dla100", overlay_key="ratio_rows_X1b_dla100_diluted",
        purpose="100% DLA-selected, masked; out-of-model-space stress of the reduced nuisance "
                "basis (no DLA direction on KS); dilution-CORRECTED primary fork (annex OQ1); "
                "X1b diluted fork = readout overlay only"),
    "X2_sub100": dict(
        kind="mixture_corner", cls=1, n_mocks=16, part1=True,
        table_key="ratio_rows_X2_sub100", overlay_key=None,
        purpose="100% subDLA-selected corner (deployed trough-fill truth, annex OQ9; "
                "highest-class partition, OQ10); largest prior-truth displacement; expected "
                "strongest n_s stress (UNSIGNED)"),
    "X3_lls100": dict(
        kind="mixture_corner", cls=0, n_mocks=16, part1=True,
        table_key="ratio_rows_X3_lls100", overlay_key=None,
        purpose="100% LLS-selected corner; mapped prior saturates near the simplex edge, so "
                "the arm measures the clamped-residual leakage through the LLS-excess/"
                "forest-amplitude degeneracy"),
    "X4_prof": dict(
        kind="mixture_profile", cls=0, n_mocks=16, part1=False,
        table_key=None, overlay_key=None, profile=dict(eta1=X4_ETA1, eta2=X4_ETA2),
        purpose="profiled LLS selection (inverted-U f_sel, peak z=3.364, edges 0.254): the "
                "retired K6 z-evolution stress living where the whole range is physical; "
                "COMMISSIONED non-binding (annex OQ4)"),
    "K8a_eps_hi": dict(
        kind="dndx_displaced", cls=0, n_mocks=8, part1=False,
        site="eps_lls", n_sigma=+1.0, sigma_expect=0.5310,
        table_key=None, overlay_key=None,
        purpose="in-manifold calibration: eps_lls = +1 deployed sigma through the frozen "
                "occupancy map (K6-redesign Sec 7; in-simplex for every draw by construction)"),
    "K8b_eps_lo": dict(
        kind="dndx_displaced", cls=0, n_mocks=8, part1=False,
        site="eps_lls", n_sigma=-1.0, sigma_expect=0.5310,
        table_key=None, overlay_key=None,
        purpose="in-manifold calibration: eps_lls = -1 deployed sigma"),
    "K8c_kap_hi": dict(
        kind="dndx_displaced", cls=0, n_mocks=8, part1=False,
        site="kappa_lls", n_sigma=+1.0, sigma_expect=0.6681,
        table_key=None, overlay_key=None,
        purpose="in-manifold calibration: kappa_lls = +1 deployed sigma (tilt displacement)"),
}

OVERLAY_ARMS = ("X1b_dla100_diluted",)     # readout overlays: gate-power entries, NO fits

# Pre-registered expected-direction notes (proposal Sec 6 + round-2 revision 8): physics
# expectations for the readout, NEVER pass criteria. n_s directions are UNSIGNED (the NUTS
# sign-reversal precedent, metal-ns-mechanism memo); magnitudes-only expectations. KS is
# metal-free (no a_SiIII confound).
EXPECTED_DIRECTIONS = {
    "X1_dla100": "broadband suppression (~10% corrected fork; ~35-38% in the X1b overlay): "
                 "expect tau0_amp HIGH, A_p LOW, alpha_LLS/alpha_sub toward lower support; "
                 "n_s residual UNSIGNED (magnitude-only)",
    "X2_sub100": "tilt-like signature (low-k excess ~1.3-1.6x, high-k suppression ~0.87-0.89): "
                 "strongest expected n_s stress via the n_s<->subDLA coupling, direction "
                 "UNSIGNED; amplitude shared between A_p and tau0",
    "X3_lls100": "low-k LLS excess (1.06-1.53x falling with z): expect A_p HIGH-side response "
                 "with an n_s residual, UNSIGNED; smallest displacement of the three",
    "X4_prof":   "z-profiled LLS selection: z-evolution stress; UNSIGNED n_s expectation",
    "K8a_eps_hi": "in-manifold +1 sigma: biases expected within budget; rank non-uniformity "
                  "expected (displaced truth)",
    "K8b_eps_lo": "in-manifold -1 sigma: as K8a, opposite displacement",
    "K8c_kap_hi": "in-manifold +1 sigma tilt: as K8a in the exponent direction",
}

# Corner-failure protocol OF RECORD (PI decision 8, record #7; annex OQ11): printed verbatim
# by the analyzer next to any Part-1 corner FAIL. It LIMITS composition claims and names the
# follow-up study; it is NOT a hard unblinding block.
CORNER_FAILURE_PROTOCOL = (
    "CORNER-FAILURE PROTOCOL (PI decision 8, record #7): a pure-corner Part-1 FAIL first "
    "LIMITS arbitrary-composition claims (no claims beyond the stated selection-contamination "
    "levels) and TRIGGERS a realistic selection-fraction / redshift-profile threshold study; "
    "it is NOT automatically a hard rejection or an unblinding block. The X2 corner is "
    "pre-registered as physically likely to stress n_s hardest (round-2 revision 4c); a FAIL "
    "there is an expected-sensitivity bound at an extreme composition unless the threshold "
    "study says otherwise.")


def arm_ids():
    return list(ARMS)


def part1_arm_ids():
    return [a for a, e in ARMS.items() if e["part1"]]


def campaign_cost_cpuh():
    """Cost ledger at the registered Ns and the 5 CPU-h planning anchor: X1-X3 240, X4 80,
    K8 120 nominal (proposal Sec 7 as corrected by round-2 revision 7)."""
    fits = sum(e["n_mocks"] for e in ARMS.values())
    x123 = sum(ARMS[a]["n_mocks"] for a in ("X1_dla100", "X2_sub100", "X3_lls100"))
    return dict(
        fits=fits,
        nominal=fits * _PLAN_CPUH_PER_FIT,
        worst_case=fits * _PLAN_CPUH_PER_FIT * _CONTINGENCY,
        nominal_x123=x123 * _PLAN_CPUH_PER_FIT,
        nominal_x4=ARMS["X4_prof"]["n_mocks"] * _PLAN_CPUH_PER_FIT,
        nominal_k8=sum(ARMS[a]["n_mocks"] for a in ARMS if a.startswith("K8"))
        * _PLAN_CPUH_PER_FIT)


def batch_cells():
    """SLURM array-id -> (arm_id, mock) over the registry (registry order): 88 cells.
    Numpy-light on purpose: the batch script resolves its task id through THIS module
    without importing the JAX-heavy runner, and WITHOUT needing the truth table."""
    return [(aid, m) for aid in ARMS for m in range(ARMS[aid]["n_mocks"])]


def shard_pkl_name(arm_id, mock, *, smoke=False):
    """ks_xsel_{arm}_shard_{m:03d}[.smoke].pkl. DISTINCT prefix: the legacy campaign analyzer
    globs ks_selboost_* and can never see an X pkl; analyze_xsel.py globs ks_xsel_* and can
    never see a legacy/K/R6 pkl (defense in depth on top of the signature refusals)."""
    assert arm_id in ARMS, f"unknown arm {arm_id!r}"
    return f"ks_xsel_{arm_id}_shard_{int(mock):03d}{'.smoke' if smoke else ''}.pkl"


# ---------------------------------------------------------------------------------------- #
#  Truth-table loader (contract: scripts/xsel_truth_contract.md). FAIL-LOUD in every branch.
# ---------------------------------------------------------------------------------------- #
XSEL_TRUTH_TABLE_PATH = ("/home/mfho/hcd_priya_notes/docs/superpowers/xsel-truth-artifacts/"
                         "xsel_truth_tables.npz")
# PIN: None until the stage-V validation pass emits the table AND its figures pass the
# delegated review (record #7); then the sha256 hex of the file bytes is set here
# DELIBERATELY. Every loader call before that fails loud. NEVER default this to a computed
# value: the pin is the provenance statement.
XSEL_TRUTH_TABLE_SHA256 = None

REQUIRED_CONVENTION = {
    "frame": "deployed",
    "mean_flux": "global",
    "tau0_rescale": True,
    "k_convention": "leg_native_angular",
    "x1_fork": "dilution_corrected",
    "x1b_fork": "dilution_included",
    "x2_trough_fill": "deployed",
    "partition": "highest_class",
    "selection_unit": "per_sightline_120mpch",
    "mask_width": "nominal_band_overlay",
    "metal_free_ks": True,
}

_RATIO_KEYS = ("ratio_rows_X1_dla100", "ratio_rows_X1b_dla100_diluted",
               "ratio_rows_X2_sub100", "ratio_rows_X3_lls100")
GATE_POWER_ARMS = tuple(ARMS) + OVERLAY_ARMS
_GATE_POWER_FIELDS = ("D", "sigma_pair_expected", "p_part1_fail_null")


def load_truth_tables(path=None, expect_sha=None):
    """Load + validate the sha-pinned stage-V truth tables. Returns dict(path, sha256, leg_k,
    leg_z, ratios {name: (N,)}, band (lo, hi) or None, convention dict, gate_power dict).

    FAIL-LOUD contract: unpinned sha -> RuntimeError; missing file -> FileNotFoundError;
    sha mismatch / missing key / bad shape / non-positive ratio / drifted convention /
    incomplete gate power -> AssertionError. There is NO soft path: the battery must never
    run or read out against an unvalidated table."""
    path = XSEL_TRUTH_TABLE_PATH if path is None else str(path)
    expect_sha = XSEL_TRUTH_TABLE_SHA256 if expect_sha is None else expect_sha
    if expect_sha is None:
        raise RuntimeError(
            "X-battery truth tables are NOT PINNED yet (ks_xsel_arms.XSEL_TRUTH_TABLE_SHA256 "
            "is None). The stage-V validation pass (scripts/analyze_xsel_truth_tables.py) "
            "must emit the table per scripts/xsel_truth_contract.md, its figures must pass "
            "the delegated review, and the sha must then be pinned DELIBERATELY in "
            "scripts/ks_xsel_arms.py. Refusing to run/read out before that.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"X-battery truth table missing: {path} (the stage-V validation pass has not "
            f"emitted it; contract scripts/xsel_truth_contract.md). Refusing to proceed.")
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == expect_sha, (
        f"truth-table sha256 {sha[:16]}... != pinned {str(expect_sha)[:16]}... ({path}); the "
        f"truth tables of record were swapped/edited. Re-pin deliberately or restore the file.")
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        need = {"leg_k", "leg_z", "convention_json", "gate_power_json", *_RATIO_KEYS}
        missing = sorted(need - keys)
        assert not missing, f"truth table {path} missing required keys {missing} (contract)"
        leg_k = np.asarray(f["leg_k"], float)
        leg_z = np.asarray(f["leg_z"], float)
        assert leg_k.ndim == 1 and leg_k.shape == leg_z.shape and leg_k.size > 0, \
            "leg_k/leg_z must be matching non-empty 1-D row grids"
        assert np.all(np.isfinite(leg_k)) and np.all(leg_k > 0), "non-finite/non-positive leg_k"
        assert np.all(np.isfinite(leg_z)) and np.all(leg_z > 0), "non-finite/non-positive leg_z"
        ratios = {}
        for kname in _RATIO_KEYS:
            r = np.asarray(f[kname], float)
            assert r.shape == leg_k.shape, f"{kname} shape {r.shape} != leg grid {leg_k.shape}"
            assert np.all(np.isfinite(r)) and np.all(r > 0), f"{kname} non-finite/non-positive"
            ratios[kname] = r
        band = None
        if "band_lo_rows_X1_dla100" in keys or "band_hi_rows_X1_dla100" in keys:
            assert {"band_lo_rows_X1_dla100", "band_hi_rows_X1_dla100"} <= keys, \
                "mask-width band must ship BOTH lo and hi rows"
            lo = np.asarray(f["band_lo_rows_X1_dla100"], float)
            hi = np.asarray(f["band_hi_rows_X1_dla100"], float)
            assert lo.shape == hi.shape == leg_k.shape and np.all(lo <= hi), \
                "band rows malformed (shape or lo > hi)"
            band = (lo, hi)
        convention = json.loads(str(f["convention_json"][()]))
        gate_power = json.loads(str(f["gate_power_json"][()]))
    for ck, cv in REQUIRED_CONVENTION.items():
        assert ck in convention and convention[ck] == cv, (
            f"truth-table convention[{ck!r}] = {convention.get(ck)!r} != required {cv!r} "
            f"(round-2 revision 1: only DEPLOYED-convention tables are admissible)")
    for aid in GATE_POWER_ARMS:
        assert aid in gate_power, f"gate_power_json missing arm {aid!r} (revision 4b inputs)"
        gp = gate_power[aid]
        for fld in _GATE_POWER_FIELDS:
            assert fld in gp and np.isfinite(float(gp[fld])), \
                f"gate_power[{aid}][{fld}] missing/non-finite"
        assert float(gp["D"]) > 0 and float(gp["sigma_pair_expected"]) > 0, \
            f"gate_power[{aid}]: D and sigma_pair_expected must be > 0"
        assert 0.0 <= float(gp["p_part1_fail_null"]) <= 1.0, \
            f"gate_power[{aid}]: p_part1_fail_null outside [0,1]"
    return dict(path=path, sha256=sha, leg_k=leg_k, leg_z=leg_z, ratios=ratios, band=band,
                convention=convention, gate_power=gate_power)


# ---------------------------------------------------------------------------------------- #
#  Registry signature: canonical payload INCLUDING the truth-table sha. Refuses pooling with
#  the legacy campaign (different module, different payload), with K-battery pkls, and across
#  truth-table swaps.
# ---------------------------------------------------------------------------------------- #
def _registry_payload(table_path=None, expect_sha=None):
    tt = load_truth_tables(table_path, expect_sha)
    payload = {}
    for aid, e in ARMS.items():
        payload[aid] = dict(
            kind=e["kind"], cls=int(e["cls"]), n_mocks=int(e["n_mocks"]),
            part1=bool(e["part1"]), table_key=e.get("table_key"),
            overlay_key=e.get("overlay_key"),
            profile=(dict(eta1=float(e["profile"]["eta1"]), eta2=float(e["profile"]["eta2"]))
                     if "profile" in e else None),
            displacement=(dict(site=e["site"], n_sigma=float(e["n_sigma"]),
                               sigma_expect=float(e["sigma_expect"]))
                          if e["kind"] == "dndx_displaced" else None))
    payload["_meta"] = dict(
        campaign=CAMPAIGN, z_pivot=Z_PIVOT, ks_z_range=list(KS_Z_RANGE), gate=GATE,
        sigma_post_convention=SIGMA_POST_CONVENTION, pair_seed=PAIR_SEED,
        k0_reuse="cross-signature R4/R5 corrected-geometry K0 (record #7 annex, OQ5)",
        x4=dict(eta1=X4_ETA1, eta2=X4_ETA2, peak_normalized=True),
        overlay_arms=list(OVERLAY_ARMS),
        truth_table_sha256=tt["sha256"],
        convention=REQUIRED_CONVENTION)
    return payload


def registry_signature(table_path=None, expect_sha=None):
    """Stable sha256 over the canonical-JSON registry payload (truth-table sha INCLUDED).
    Stamped into every shard pkl; the analyzer asserts homogeneity AND equality with the
    live registry + pinned table."""
    return hashlib.sha256(
        json.dumps(_registry_payload(table_path, expect_sha),
                   sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------- #
#  Swap-off per-mock-record identity harness (round-2 revision 6): shard pkls embed
#  wall-clock META fields, so the pre-registered X1 equivalence gate pins byte-identity of
#  the PER-MOCK RECORDS (pure functions of seed and mock id) with the swap disabled.
#  Numpy-pure and mockable: unit-tested on synthetic records, run for real by
#  run_xsel_shard.py --swap-off-check.
# ---------------------------------------------------------------------------------------- #
def compare_per_mock_records(rec_a, rec_b, _path="rec"):
    """Byte-level comparison of two run_legb-style per-mock records. Returns a list of
    mismatch descriptions (empty == byte-identical content). Arrays compare by dtype, shape
    and raw bytes; floats by exact equality (NaN == NaN allowed); dicts/lists recurse."""
    mism = []

    def _cmp(a, b, path):
        if isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict)):
                mism.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
                return
            ka, kb = set(a), set(b)
            for k in sorted(ka ^ kb):
                mism.append(f"{path}[{k!r}]: present in only one record")
            for k in sorted(ka & kb):
                _cmp(a[k], b[k], f"{path}[{k!r}]")
            return
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a_arr, b_arr = np.asarray(a), np.asarray(b)
            if a_arr.dtype != b_arr.dtype:
                mism.append(f"{path}: dtype {a_arr.dtype} vs {b_arr.dtype}")
            elif a_arr.shape != b_arr.shape:
                mism.append(f"{path}: shape {a_arr.shape} vs {b_arr.shape}")
            elif a_arr.tobytes() != b_arr.tobytes():
                mism.append(f"{path}: array bytes differ")
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                mism.append(f"{path}: sequence type/length differs")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(x, y, f"{path}[{i}]")
            return
        if isinstance(a, float) and isinstance(b, float):
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                mism.append(f"{path}: float {a!r} vs {b!r}")
            return
        if a != b:
            mism.append(f"{path}: {a!r} vs {b!r}")

    _cmp(rec_a, rec_b, _path)
    return mism
