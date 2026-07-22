#!/usr/bin/env python3
"""analysis.lock GENERATOR + lock-vs-live verifier (freeze decision 5, W4).

FREEZE-STEP CONTRACT
====================
* Regeneration of the REAL committed ``analysis.lock`` happens ONLY at the freeze cut, AFTER
  the independent review panel, ON THE MERGED TREE, via ``--i-am-the-freeze-step``. Until that
  step the committed lock is EXPECTED to be the stale June one and ``--check`` is EXPECTED to
  exit 1 (loudly, informatively) -- that is a statement about the repo being mid-flight (KS
  prior reparameterization), not a generator bug.
* This generator is the ONLY writer of ``analysis.lock``. Hand-edits are PROHIBITED: any
  intentional change lands in LIVE code first, then the lock is REGENERATED here.
* ``blind.lock`` is OUT OF SCOPE: a READ-ONLY input here and everywhere. It must never be
  written by anyone; this script hard-refuses any --out named ``blind.lock`` (no override).
* EVERY pinned value derives from LIVE code/artifacts. The old lock is NEVER read as an input
  to generation (regression of record: the June lock carries the retired all-z-median LLS
  centre 0.2908612702546296 and the retired wrong-object ratio z-slope [0.95, 0.15, 0.4];
  carrying either forward is a poisoning bug -- see ``assert_not_poisoned``).

MODES
=====
Generate (default)   : build the lock from live code and write it to --out. The default --out
                       is a DATED path under the system tmp dir (for review); writing the repo
                       ``analysis.lock`` requires ``--i-am-the-freeze-step`` AND running from
                       the tree that owns that lock.
--check              : compare the COMMITTED ``analysis.lock`` against a fresh live generation,
                       ignoring the volatile fields (created_utc, git_commit, generator sha256).
                       Exit 0 = identical-modulo-volatile; exit 1 = drift, with a readable
                       per-field diff. TODAY this exits 1 by design (stale June lock).
--emit-leg-summary S : INTERNAL subprocess mode -- build ONE survey's leg summary (the heavy
                       step: full build_real_ctx) and dump JSON to --leg-out.

Leg summaries are the slow part (a full production ctx build per survey). ``--legs-cache FILE``
reuses a previously saved summary set; ``--save-legs-cache FILE`` persists one. A cached
summary built under a DIFFERENT forward/prior signature than live is REFUSED (fail-loud)
unless ``--allow-stale-legs-cache`` is passed (machinery testing / review only; the freeze
step must never pass it).

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<tree> JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/gen_analysis_lock.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# ---------------------------------------------------------------------------------------------
# Tree layout. _REPO_ROOT is the tree THIS generator file lives in (worktree-safe): the lock it
# generates/checks belongs to the SAME tree, so worktree runs never touch the main tree's lock.
# The ensemble BINARIES are gitignored and live only in the main tree (same authority as
# run_real_fit.PROD_PREFIX).
# ---------------------------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_CHECKPOINTS_DIR = "/home/mfho/hcd_priya/checkpoints"
NOTES_REPO = "/home/mfho/hcd_priya_notes"

# PI decision records of record (freeze provenance; READ-ONLY -- we record path + sha256 only).
PI_DECISION_RECORDS = (
    "docs/superpowers/2026-07-21-PI-DECISIONS-OF-RECORD.md",
    "docs/superpowers/2026-07-21-PI-DECISIONS-02-LLS-REPARAM.md",
    "docs/superpowers/2026-07-22-PI-DECISIONS-03-TAKEOVER.md",
)

# ---------------------------------------------------------------------------------------------
# THE UNCOVERED-CONSTANTS REGISTRY (ONE registry, here, at the top).
#
# These are the closure-side / data-side constants covered by NO signature (the
# closure_legb.forward_signature docstring's "NOT covered" list + the
# inference.hcd_prior_constants_payload SCOPE note). Each entry is pulled LIVE by attribute at
# generation time; a missing attribute is recorded as a TODO_AT_FREEZE marker (never guessed).
#
# FREEZE-STEP RE-AUDIT CONTRACT: the freeze step MUST re-audit this list against (a) the
# then-current forward_signature docstring, (b) the hcd_prior_constants_payload SCOPE note, and
# (c) any constant added by in-flight work (e.g. the KS prior reparameterization) -- a new
# uncovered constant that is not added here rides into the freeze silently. Whole payloads/dicts
# (prior_constants, PROD_FORWARD_BY_LEG, the ensemble manifest) are pulled programmatically so
# new KEYS inside them are picked up automatically; this registry exists for the constants that
# live OUTSIDE any payload.
# ---------------------------------------------------------------------------------------------
UNCOVERED_CONSTANT_REGISTRY = (
    # (lock key, module, attribute, note)
    ("closure_legb.HCD_INCIDENCE_SLOPE", "hcd_analysis.emulator.closure_legb",
     "HCD_INCIDENCE_SLOPE",
     "per-class incidence z-slope d ln w_c/d ln(1+z), 60-sim median; the closure z-slope "
     "prior center (NOT the lit/sim ratio slope)"),
    ("closure_legb.ZSLOPE_PRIOR_SIGMA", "hcd_analysis.emulator.closure_legb",
     "ZSLOPE_PRIOR_SIGMA", "HCD z-slope prior width (lit WLS 1-sigma), class order LLS/sub/DLA"),
    ("closure_legb.SIGMA_A0", "hcd_analysis.emulator.closure_legb", "SIGMA_A0",
     "alpha_res restore-arm amplitude prior width (TruncatedNormal scale); inert under NORC"),
    ("closure_legb.SIGMA_S", "hcd_analysis.emulator.closure_legb", "SIGMA_S",
     "alpha_res restore-arm z-slope prior width; inert under NORC"),
    ("closure_legb.F_RES_PIVOT_Z", "hcd_analysis.emulator.closure_legb", "F_RES_PIVOT_Z",
     "option-b spectral-resolution b_res(z) pivot"),
    ("closure_legb.F_RES_AMP_SIGMA", "hcd_analysis.emulator.closure_legb", "F_RES_AMP_SIGMA",
     "option-b f_res_amp default prior width (tight physics width, not cup1d's)"),
    ("closure_legb.F_RES_SLOPE_SIGMA", "hcd_analysis.emulator.closure_legb",
     "F_RES_SLOPE_SIGMA", "option-b f_res_slope prior width"),
    ("closure_legb.PROD_FORWARD_BY_LEG", "hcd_analysis.emulator.closure_legb",
     "PROD_FORWARD_BY_LEG",
     "certified per-leg data-nuisance forward (single source for real fit AND SBC); pulled "
     "WHOLE so new per-leg keys ride in automatically"),
    ("closure_legb.PROD_RES_CORR_ON", "hcd_analysis.emulator.closure_legb", "PROD_RES_CORR_ON",
     "the GLOBAL NORC reversal knob (False = deployed NORC forward)"),
    ("data_likelihood.DESI_DLA_COV_REDUCE", "hcd_analysis.emulator.data_likelihood",
     "DESI_DLA_COV_REDUCE",
     "reduced-DESI-covariance authority (drop syst_e_dla_completeness while alpha_DLA floats; "
     "PI disposition 2026-07-17)"),
    ("seeding.SEED_DERIVATION", "hcd_analysis.emulator.seeding", "SEED_DERIVATION",
     "the deterministic NUTS seed derivation (crc32, P0-A fix); changing it is a new lock era"),
)

# The retired POISONED values the June lock carries. Generation and the tests both tripwire on
# them: they must never re-enter a generated lock as prior centers.
POISON_LLS_ALLZ_MEDIAN = 0.2908612702546296     # retired all-z-median LLS alpha-pivot centre
POISON_RATIO_ZSLOPE = (0.95, 0.15, 0.4)         # retired wrong-object lit/sim ratio slope
_POISON_RTOL = 1e-6

# Fields that legitimately differ between two correct generations of the SAME live state.
VOLATILE_PATHS = frozenset({
    "created_utc",
    "git_commit",
    "provenance.git_commit",
    "provenance.generator.sha256",
})

SURVEY_ORDER = ("desi", "eboss", "ks")           # deterministic build/emit order


# ---------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------
def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path):
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _git(*args):
    out = subprocess.run(["git", "-C", _REPO_ROOT] + list(args),
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def _jn(x, where="value"):
    """Coerce to JSON-native, fail-loud on anything exotic (no silent stringification)."""
    import numpy as np
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jn(v, f"{where}.{k}") for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jn(v, f"{where}[{i}]") for i, v in enumerate(x)]
    if hasattr(x, "tolist"):                      # np/jnp arrays and scalars
        return _jn(x.tolist(), where)
    raise TypeError(f"non-JSON-native value at {where}: {type(x).__name__} = {x!r}")


def canonical_dumps(lock):
    """Deterministic serialization: sorted keys, 2-space indent, trailing newline. The ONLY
    volatility in a generated lock is created_utc/git_commit/generator-sha (VOLATILE_PATHS)."""
    return json.dumps(lock, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _flatten(obj, prefix=""):
    """Flatten nested dicts to {dotted.path: leaf}; lists stay leaves (compared whole)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, p))
            else:
                out[p] = v
    else:
        out[prefix] = obj
    return out


def _close(a, b, rtol=_POISON_RTOL):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30)


def assert_not_poisoned(lock):
    """Generation-time tripwire (defense in depth alongside the live pivot/zslope guards):
    the retired June-lock values must never appear in a generated lock AS PRIOR CENTERS.

    * No HCD alpha prior center (any legs.*.prior.alpha_hcd_mu / priors.hcd by-survey mu) may
      sit at the retired all-z-median LLS centre 0.29086... (x any live survey boost).
    * No z-slope prior center (priors.zslope.mu / legs.*.prior.zslope_mu) may equal the retired
      wrong-object ratio slope (0.95, 0.15, 0.4).

    NOTE: prior_constants.HCD_PIVOT_LLS_ALLZ_MEDIAN legitimately RECORDS the retired value --
    it is the live guard's reference constant (assert_hcd_pivot_z3), not a prior center; it is
    exempt by construction (only *_mu paths are scanned here).
    """
    flat = _flatten(lock)
    boosts = [1.0]
    surv_boost = (lock.get("prior_constants") or {}).get("HCD_LLS_SURVEY_BOOST") or {}
    boosts += [float(v) for v in surv_boost.values()]
    for path, val in flat.items():
        leaf = path.rsplit(".", 1)[-1]
        if leaf in ("alpha_hcd_mu",) and isinstance(val, (list, tuple)) and len(val) >= 1:
            for b in boosts:
                if _close(val[0], POISON_LLS_ALLZ_MEDIAN * b, rtol=1e-4):
                    raise AssertionError(
                        f"POISONED LOCK: {path}[0]={val[0]} == retired all-z-median LLS centre "
                        f"{POISON_LLS_ALLZ_MEDIAN} x boost {b} -- values must derive from live "
                        f"code, never the old lock")
        if leaf in ("mu", "zslope_mu") and "zslope" in path and \
                isinstance(val, (list, tuple)) and len(val) == 3:
            if all(_close(v, p) for v, p in zip(val, POISON_RATIO_ZSLOPE)):
                raise AssertionError(
                    f"POISONED LOCK: {path}={val} == the retired wrong-object lit/sim ratio "
                    f"z-slope {list(POISON_RATIO_ZSLOPE)} -- the z-slope center is the "
                    f"INCIDENCE slope (~2.4), see hcd-dndx-zslope-bug")


# ---------------------------------------------------------------------------------------------
# live-source pulls
# ---------------------------------------------------------------------------------------------
def _load_run_real_fit():
    """scripts/ is not a package; load run_real_fit by path from THIS tree."""
    path = os.path.join(_REPO_ROOT, "scripts", "run_real_fit.py")
    spec = importlib.util.spec_from_file_location("run_real_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_uncovered_constants():
    """Resolve the registry against live modules. Missing attributes (in-flight work) become
    TODO_AT_FREEZE markers -- recorded, warned about, never guessed."""
    values, missing = {}, []
    for key, modname, attr, note in UNCOVERED_CONSTANT_REGISTRY:
        try:
            mod = importlib.import_module(modname)
            val = getattr(mod, attr)
        except (ImportError, AttributeError) as e:
            missing.append({"key": key, "todo": f"TODO_AT_FREEZE: {modname}.{attr} not "
                                                f"resolvable at generation base ({e}); "
                                                f"re-audit at the freeze step"})
            continue
        values[key] = {"value": _jn(val, key), "note": note}

    # Two live constants that exist only as a LITERAL / a SIGNATURE DEFAULT (no module attr):
    import hcd_analysis.emulator.closure_legb as CL
    src = inspect.getsource(CL.build_legb_ctx)
    m = re.search(r'_ks_kw\["k_max"\]\s*=\s*([0-9.]+)', src)
    if m:
        values["closure_legb.build_legb_ctx.KS_NORC_KMAX_CAP"] = {
            "value": float(m.group(1)),
            "note": "the KS k_max auto-cap literal applied when res_corr_on=False (Gate-A "
                    "NORC); parsed from the build_legb_ctx source (no named constant)"}
    else:
        missing.append({"key": "closure_legb.build_legb_ctx.KS_NORC_KMAX_CAP",
                        "todo": "TODO_AT_FREEZE: the NORC KS k_max cap literal was not found "
                                "in build_legb_ctx source; re-audit (moved/renamed?)"})
    p = inspect.signature(CL.build_legb_ctx).parameters.get("mf_anchor_mult")
    if p is not None and p.default is not inspect.Parameter.empty:
        values["closure_legb.build_legb_ctx.mf_anchor_mult_default"] = {
            "value": float(p.default),
            "note": "MF res_corr low-k anchor multiplier default (5.0 = production anchor; "
                    "0.0 disables, diagnostic only)"}
    else:
        missing.append({"key": "closure_legb.build_legb_ctx.mf_anchor_mult_default",
                        "todo": "TODO_AT_FREEZE: build_legb_ctx has no mf_anchor_mult default; "
                                "re-audit"})
    for miss in missing:
        print(f"!!! [gen_analysis_lock] {miss['todo']}", file=sys.stderr)
    return {
        "registry_note": (
            "ONE registry: UNCOVERED_CONSTANT_REGISTRY at the top of "
            "scripts/gen_analysis_lock.py. These constants are covered by NO signature "
            "(forward_signature 'NOT covered' list + the hcd_prior_constants_payload SCOPE "
            "note). THE FREEZE STEP RE-AUDITS THIS LIST against the then-current docstrings "
            "and any in-flight additions before regenerating the lock."),
        "values": values,
        "missing_at_generation": missing,
    }


def build_leg_summary(survey):
    """Build ONE survey's production ctx (run_real_fit.build_real_ctx, the deployed path) and
    reduce it to a JSON-native leg summary. HEAVY (ensemble + caches + data npz)."""
    import numpy as np
    import hcd_analysis.emulator.closure_legb as CL
    from hcd_analysis.emulator import inference as INF

    RF = _load_run_real_fit()
    ctx, _d, members = RF.build_real_ctx(survey)
    assert len(ctx.legs) == 1, f"build_real_ctx({survey!r}) returned {len(ctx.legs)} legs"
    leg = ctx.legs[0]
    stamp = CL.forward_stamp(ctx, leg)

    z = np.asarray(leg.z, dtype=float)
    k = np.asarray(leg.k, dtype=float)
    zsl_mu = ctx.zslope_mu if ctx.zslope_mu is not None else CL.HCD_INCIDENCE_SLOPE
    zsl_sd = ctx.zslope_sigma if ctx.zslope_sigma is not None else CL.ZSLOPE_PRIOR_SIGMA
    return _jn({
        "name": leg.name,
        "survey": survey,
        "z": [round(float(v), 6) for v in z],
        "k_min": float(k.min()),
        "k_max": float(k.max()),
        "n_rows": int(np.asarray(leg.P_data).size),
        "n_z": int(leg.n_z),
        "n_per_z": [int(v) for v in np.asarray(leg.n_per_z)],
        "dla_forward_frac": float(leg.dla_forward_frac),
        "metals_on": bool(leg.metals_on),
        "mf_floor_on": bool(leg.mf_floor_on),
        "resolution_on": bool(leg.resolution_on),
        "resolution_ready": bool(getattr(leg, "resolution_ready", False)),
        "resolution_coherent_on": bool(getattr(leg, "resolution_coherent_on", False)),
        "n_ensemble_members": len(members),
        # the SINGLE-AUTHORITY forward stamp (res_corr/f_res/metals knobs + data-selection
        # flags + BOTH freeze signatures) -- everything forward_stamp carries, verbatim:
        "forward": stamp,
        "prior": {
            "alpha_hcd_mu": ctx.alpha_hcd_mu,
            "alpha_hcd_sigma": ctx.alpha_hcd_sigma,
            "marginalize_zslope": bool(ctx.marginalize_zslope),
            "zslope_mu": zsl_mu,
            "zslope_sigma": zsl_sd,
            "tau0_amp_range": ctx.tau0_amp_range,
            "dtau0_range": ctx.dtau0_range,
            "tau0_pivot_z": float(ctx.tau0_pivot_z),
        },
        "metals": {
            "metal_prior": str(ctx.metal_prior),
            "sample_metals": bool(ctx.sample_metals),
            "a_siiii_max": float(ctx.a_siiii_max),
            "metal_node_z": ctx.metal_node_z,
            "metal_fnode_lo": float(ctx.metal_fnode_lo),
            "metal_fnode_hi": float(ctx.metal_fnode_hi),
            "metal_one_minus_F_ref": float(ctx.metal_one_minus_F_ref),
            "metal_siII_legs": ctx.metal_siII_legs,
        },
        "ctx_cov": {
            "cemu_inflate": float(ctx.cemu_inflate),
            "use_xclass": ctx.rho_zb_per_leg is not None,
            "mf_on": ctx.mf is not None,
            "mf_floor_attached": ctx.mf_floor is not None,
            "emucoh_on": ctx.mf_emucoh_per_leg is not None,
            "emucoh_offdiag_only": bool(ctx.mf_emucoh_offdiag_only),
            "mf_shape_on": ctx.mf_shape_per_leg is not None,
            "dla_cov_reduced": bool(getattr(leg, "dla_cov_reduced", False)),
        },
    }, where=f"leg_summary[{survey}]")


def _leg_build_cwd():
    """The leg build reads the MF backbone from the CWD-RELATIVE ``checkpoints`` dir
    (multifidelity.load_lf_backbone(ckpt_dir="checkpoints")); the fold binaries are gitignored
    and live only in the main tree. From the main tree, cwd = the tree itself; from a worktree,
    build from a scratch cwd whose ``checkpoints/`` symlinks the main-tree dir (READ-ONLY --
    nothing is ever written outside the calling tree/scratch)."""
    if os.path.exists(os.path.join(_REPO_ROOT, "checkpoints", "final_fold0.eqx")):
        return _REPO_ROOT
    d = tempfile.mkdtemp(prefix="gen_analysis_lock_cwd.")
    os.symlink(MAIN_CHECKPOINTS_DIR, os.path.join(d, "checkpoints"))
    return d


def collect_leg_summaries(legs_cache=None, save_legs_cache=None, in_process=False):
    """All three survey leg summaries, keyed by survey. Default: one SUBPROCESS per survey
    (bounds memory; a fresh interpreter per production ctx build). ``legs_cache`` reuses a
    saved JSON set instead of building."""
    if legs_cache:
        with open(legs_cache) as f:
            summaries = json.load(f)
        missing = [s for s in SURVEY_ORDER if s not in summaries]
        if missing:
            raise SystemExit(f"--legs-cache {legs_cache} is missing surveys {missing}")
        return summaries
    summaries = {}
    for survey in SURVEY_ORDER:
        if in_process:
            summaries[survey] = build_leg_summary(survey)
            continue
        with tempfile.NamedTemporaryFile(mode="r", suffix=f".leg_{survey}.json",
                                         delete=False) as tf:
            leg_out = tf.name
        try:
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--emit-leg-summary", survey, "--leg-out", leg_out]
            print(f"[gen_analysis_lock] building leg summary for {survey!r} (subprocess)...")
            r = subprocess.run(cmd, text=True, cwd=_leg_build_cwd())
            if r.returncode != 0:
                raise SystemExit(f"leg-summary subprocess for {survey!r} failed "
                                 f"(exit {r.returncode})")
            with open(leg_out) as f:
                summaries[survey] = json.load(f)
        finally:
            if os.path.exists(leg_out):
                os.unlink(leg_out)
    if save_legs_cache:
        with open(save_legs_cache, "w") as f:
            json.dump(summaries, f, sort_keys=True, indent=2)
            f.write("\n")
        print(f"[gen_analysis_lock] saved legs cache -> {save_legs_cache}")
    return summaries


def _assert_leg_signatures_live(leg_summaries, allow_stale=False):
    """A cached leg summary built under a DIFFERENT forward/prior signature than live would
    embed stale signature strings into the lock -- refuse unless explicitly allowed (machinery
    testing only; the freeze step must NEVER pass --allow-stale-legs-cache)."""
    import hcd_analysis.emulator.closure_legb as CL
    from hcd_analysis.emulator.inference import hcd_prior_signature
    live_fwd, live_prior = CL.forward_signature(), hcd_prior_signature()
    stale = {}
    for survey, s in leg_summaries.items():
        fwd = s.get("forward", {})
        if (fwd.get("forward_signature") != live_fwd
                or fwd.get("hcd_prior_signature") != live_prior):
            stale[survey] = {"cached_forward": fwd.get("forward_signature"),
                             "cached_prior": fwd.get("hcd_prior_signature")}
    if stale and not allow_stale:
        raise SystemExit(
            f"STALE LEGS CACHE: cached leg summaries were built under different signatures "
            f"than live (live forward={live_fwd[:12]}..., live prior={live_prior[:12]}...): "
            f"{json.dumps(stale, indent=2)}\nRebuild the legs cache (drop --legs-cache), or "
            f"pass --allow-stale-legs-cache for MACHINERY TESTING ONLY.")
    return bool(stale)


# ---------------------------------------------------------------------------------------------
# lock assembly
# ---------------------------------------------------------------------------------------------
def generate_lock(leg_summaries, allow_stale_legs=False):
    """Assemble the full lock dict from live sources + the given leg summaries."""
    import hcd_analysis.emulator.closure_legb as CL
    from hcd_analysis.emulator import inference as INF
    from hcd_analysis.emulator import data as ED
    from hcd_analysis.emulator import seeding as SEED
    from hcd_analysis.emulator import meanflux_prior as MFP

    _assert_leg_signatures_live(leg_summaries, allow_stale=allow_stale_legs)
    RF = _load_run_real_fit()

    # --- signatures + the FULL prior-constants payload (its docstring REQUIRES this insert) --
    signatures = {
        "forward_signature": CL.forward_signature(),
        "hcd_prior_signature": INF.hcd_prior_signature(),
        "note": ("forward_signature = sha256 over {PROD_FORWARD_BY_LEG, PROD_RES_CORR_ON, "
                 "DESI_DLA_COV_REDUCE}; hcd_prior_signature = sha256 over "
                 "hcd_prior_constants_payload(). Constants covered by NEITHER live in "
                 "uncovered_constants."),
    }
    prior_constants = _jn(INF.hcd_prior_constants_payload(), "prior_constants")

    # --- blinding: VERBATIM from blind.lock (READ-ONLY input; never recomputed, never
    # regenerated here). analysis.lock's prior_sigma MUST equal blind.lock's -- a desync would
    # mean two different blinding transforms are in circulation (= a blinding bug). ------------
    blind_path = os.path.join(_REPO_ROOT, "blind.lock")
    with open(blind_path, "rb") as f:
        blind_raw = f.read()
    bl = json.loads(blind_raw)
    blinding = {
        "params": bl["blind_params"],
        "offset_sigma_multiple": bl["offset_sigma_multiple"],
        "prior_sigma": bl["prior_sigma"],
        "seed_ref": ("blind.lock (seed_str = project_string@git_commit); offset applied at "
                     "VIEW time. REFERENCE ONLY -- the seed_str itself is deliberately not "
                     "duplicated here."),
        "blind_lock_git_commit": bl["git_commit"],
        "blind_lock_sha256": _sha256_bytes(blind_raw),
        "note": ("COPIED VERBATIM from blind.lock (read-only; the generator never writes or "
                 "recomputes it). prior_sigma desync between the two locks = blinding bug."),
    }
    assert blinding["prior_sigma"] == bl["prior_sigma"] and \
        blinding["offset_sigma_multiple"] == bl["offset_sigma_multiple"], \
        "blinding block desynced from blind.lock (blinding bug)"

    # --- emulator: reuse the W3 verify() machinery (digests + pairing + count + order + stray
    # tripwire) rather than re-hashing independently; then embed the verified manifest. --------
    from hcd_analysis.emulator import prod_ensemble as PE
    manifest_path = os.path.join(_REPO_ROOT, "checkpoints", PE.MANIFEST_BASENAME)
    manifest, member_prefixes = PE.verify_manifest(checkpoints_dir=MAIN_CHECKPOINTS_DIR,
                                                   manifest_path=manifest_path)
    emulator = {
        "forward": "ensemble-mean over members",
        "ensemble_members": [m["name"] for m in manifest["members"]],
        "n_members": manifest["n_members"],
        "normalizers_byte_identical": manifest["normalizers_byte_identical"],
        "manifest_schema_version": manifest["schema_version"],
        "manifest_path": os.path.join("checkpoints", PE.MANIFEST_BASENAME),
        "manifest_sha256": _sha256_file(manifest_path),
        "checkpoints_dir": MAIN_CHECKPOINTS_DIR,
        "members": _jn(manifest["members"], "emulator.members"),
        "verified": ("verify_manifest PASS at generation (sha256 digests + structural "
                     "pairing + count + order + stray-member tripwire)"),
    }
    assert len(emulator["members"]) == emulator["n_members"], "manifest member count desync"
    n_digests = sum(1 for m in emulator["members"] for k in ("eqx_sha256", "norm_sha256")
                    if m.get(k))
    assert n_digests == 2 * emulator["n_members"], \
        f"expected {2 * emulator['n_members']} eqx+norm digests, found {n_digests}"

    # --- legs -----------------------------------------------------------------------------
    legs = {s["name"]: s for s in (leg_summaries[k] for k in SURVEY_ORDER)}
    assert set(legs) == {"DESI", "KS", "eBOSS"}, f"unexpected leg set {sorted(legs)}"

    # --- covariance (ctx-level flags; must agree across surveys) ---------------------------
    covs = {sv: leg_summaries[sv]["ctx_cov"] for sv in SURVEY_ORDER}
    ref = covs[SURVEY_ORDER[0]]
    for f_ in ("cemu_inflate", "use_xclass", "mf_on", "mf_floor_attached", "emucoh_on",
               "emucoh_offdiag_only", "mf_shape_on"):
        vals = {sv: covs[sv][f_] for sv in SURVEY_ORDER}
        assert len({json.dumps(v, sort_keys=True) for v in vals.values()}) == 1, \
            f"ctx-level covariance flag {f_} differs across surveys: {vals}"
    from hcd_analysis.emulator import data_likelihood as DL
    covariance = {
        "cemu_inflate": ref["cemu_inflate"],
        "use_xclass": ref["use_xclass"],
        "mf": "ON" if ref["mf_on"] else "OFF",
        "mf_floor": "ON" if ref["mf_floor_attached"] else "OFF",
        "emucoh": ("ON (off-diagonal-only)" if ref["emucoh_on"] and
                   ref["emucoh_offdiag_only"] else ("ON" if ref["emucoh_on"] else "OFF")),
        "mf_shape": "ON" if ref["mf_shape_on"] else "OFF",
        "desi_dla_cov_reduce": bool(DL.DESI_DLA_COV_REDUCE),
    }

    # --- priors ------------------------------------------------------------------------------
    param_names = list(INF.PARAM_NAMES)          # the cache/emulator theta9 order (inference.py)
    norm_box = {n: [float(lo), float(hi)]
                for n, (lo, hi) in zip(param_names, ED.PARAM_LIMITS)}
    samp_box = {n: [float(lo), float(hi)]
                for n, (lo, hi) in zip(param_names, ED.SAMPLING_LIMITS)}
    by_survey = {sv: {
        "leg": leg_summaries[sv]["name"],
        "alpha_hcd_mu": leg_summaries[sv]["prior"]["alpha_hcd_mu"],
        "alpha_hcd_sigma": leg_summaries[sv]["prior"]["alpha_hcd_sigma"],
        "zslope_mu": leg_summaries[sv]["prior"]["zslope_mu"],
        "zslope_sigma": leg_summaries[sv]["prior"]["zslope_sigma"],
    } for sv in SURVEY_ORDER}
    pfbl = _jn(CL.PROD_FORWARD_BY_LEG, "PROD_FORWARD_BY_LEG")
    metals_by_leg = {ln: cfg["metal_prior"] for ln, cfg in pfbl.items()}
    legs_metals_on = sorted(ln for ln, cfg in pfbl.items() if cfg["metals"])
    ref_metals = leg_summaries[SURVEY_ORDER[0]]["metals"]
    priors = {
        "emulator_norm_box": norm_box,
        "theta_sampling_box": samp_box,
        "tau0": {
            "amp_range": _jn(MFP.TAU0_AMP_RANGE, "TAU0_AMP_RANGE"),
            "dtau0_range": _jn(MFP.DTAU0_RANGE, "DTAU0_RANGE"),
            "pivot_z": float(MFP.TAU0_PIVOT_Z),
            "model": "PRIYA amp x slope (uniform; Kim07 center)",
        },
        "metals": {
            "metal_prior_by_leg": metals_by_leg,
            "legs_metals_on": legs_metals_on,
            "a_siiii_max": ref_metals["a_siiii_max"],
            "metal_node_z": ref_metals["metal_node_z"],
            "metal_fnode_lo": ref_metals["metal_fnode_lo"],
            "metal_fnode_hi": ref_metals["metal_fnode_hi"],
            "metal_siII_legs": ref_metals["metal_siII_legs"],
            "note": ("flatlog2node = Gate-C Model C+ 2-node LogUniform flux-decrement nodes "
                     "on the metal legs; per-leg metal_one_minus_F_ref is recorded in "
                     "legs.<leg>.metals"),
        },
        "hcd": {
            "z_pivot": prior_constants.get("HCD_Z_PIVOT"),
            "dla_residual_frac": prior_constants.get("HCD_DLA_RESIDUAL_FRAC"),
            "lit_over_sim": prior_constants.get("HCD_LIT_OVER_SIM"),
            "lit_over_sim_slope": prior_constants.get("HCD_LIT_OVER_SIM_SLOPE"),
            "lls_survey_boost": prior_constants.get("HCD_LLS_SURVEY_BOOST"),
            "lls_survey_frac_sigma": prior_constants.get("HCD_LLS_SURVEY_FRAC_SIGMA"),
            "by_survey": by_survey,
            "marginalize_zslope": leg_summaries[SURVEY_ORDER[0]]["prior"][
                "marginalize_zslope"],
            "note": ("full HCD prior-constant payload + derivation provenance lives in the "
                     "top-level prior_constants block (hcd_prior_constants_payload)"),
        },
        "zslope": {
            # THE deployed closure/SBC z-slope prior center = the INCIDENCE slope (~2.4).
            # NEVER the retired lit/sim ratio slope [0.95, 0.15, 0.4] (hcd-dndx-zslope-bug);
            # assert_not_poisoned tripwires on it. Real-fit surveys override the LLS slot with
            # the litWLS center (see priors.hcd.by_survey zslope_mu).
            "mu": _jn(CL.HCD_INCIDENCE_SLOPE, "HCD_INCIDENCE_SLOPE"),
            "sigma": _jn(CL.ZSLOPE_PRIOR_SIGMA, "ZSLOPE_PRIOR_SIGMA"),
            "realfit_lls_mu": prior_constants.get("HCD_LLS_REALFIT_ZSLOPE"),
        },
        "resolution": {
            "f_res_amp_sigma_by_leg": {ln: cfg["f_res_amp_sigma"]
                                       for ln, cfg in pfbl.items()},
            "f_res_slope_sigma": float(CL.F_RES_SLOPE_SIGMA),
            "f_res_amp_sigma_default": float(CL.F_RES_AMP_SIGMA),
            "pivot_z": float(CL.F_RES_PIVOT_Z),
            "alpha_res_restore_arm": {"sigma_a0": float(CL.SIGMA_A0),
                                      "sigma_s": float(CL.SIGMA_S),
                                      "note": "inert under deployed NORC "
                                              "(PROD_RES_CORR_ON=False, fix_alpha_res=True)"},
        },
    }

    # --- nuts: the deployed settings exactly where run_real_fit defines them ---------------
    sig = inspect.signature(RF.run_real_fit)
    def _def(name):
        p = sig.parameters.get(name)
        if p is None or p.default is inspect.Parameter.empty:
            return f"TODO_AT_FREEZE: run_real_fit has no default for {name!r}; re-audit"
        return p.default
    src = inspect.getsource(RF.run_real_fit)
    m_ta = re.search(r"target_accept\s*=\s*([0-9.]+)", src)
    m_dm = re.search(r"dense_mass\s*=\s*(True|False)", src)
    nuts = {
        "n_chains": _def("n_chains"),
        "n_warmup": _def("n_warmup"),
        "n_samples": _def("n_samples"),
        "max_tree_depth": _def("max_tree_depth"),
        "target_accept": (float(m_ta.group(1)) if m_ta else
                          "TODO_AT_FREEZE: target_accept literal not found in run_real_fit"),
        "dense_mass": (m_dm.group(1) == "True" if m_dm else
                       "TODO_AT_FREEZE: dense_mass literal not found in run_real_fit"),
        "init": ("init_to_sample (dispersed)" if "init_strategy=init_to_sample" in src else
                 "TODO_AT_FREEZE: init strategy not found in run_real_fit"),
        "seed_default": _def("seed"),
        "seed_derivation": str(SEED.SEED_DERIVATION),
    }

    # --- decisions --------------------------------------------------------------------------
    ks_zlo_baseline = inspect.signature(DL.load_ks_leg).parameters["z_lo"].default
    ks_z28 = os.path.join(_REPO_ROOT, "scripts", "batch_real_fit_ks_z28.sh")
    ks_diag = None
    if os.path.exists(ks_z28):
        # the diagnostic script parameterizes it as KS_ZLO=${KS_ZLO:-2.8}
        m = re.search(r"KS_ZLO=\$\{KS_ZLO:-([0-9.]+)\}", open(ks_z28).read())
        ks_diag = float(m.group(1)) if m else None
    pi_records = []
    for rel in PI_DECISION_RECORDS:
        path = os.path.join(NOTES_REPO, rel)
        if os.path.exists(path):
            pi_records.append({"path": path, "sha256": _sha256_file(path)})
        else:
            pi_records.append({"path": path,
                               "sha256": "TODO_AT_FREEZE: decision record missing at "
                                         "generation base; re-audit"})
    decisions = {
        "ks_zlo": {
            "baseline": float(ks_zlo_baseline),
            "diagnostic": ks_diag,
            "note": ("baseline = the live load_ks_leg z_lo default (the PI 2026-06-08/14 "
                     "minimal closure-clean cut); diagnostic = the KS-author more-conservative "
                     "cut run via scripts/batch_real_fit_ks_z28.sh -> distinct root, same "
                     "blind.lock; does NOT change the baseline"),
        },
        "norc": {
            "prod_res_corr_on": bool(CL.PROD_RES_CORR_ON),
            "note": ("Gate-A NORC deployed forward: res_corr dropped + alpha_res pinned + KS "
                     "auto-capped (see uncovered_constants KS_NORC_KMAX_CAP); the single "
                     "reversal knob is closure_legb.PROD_RES_CORR_ON"),
        },
        "desi_dla_cov_reduce": {
            "value": bool(DL.DESI_DLA_COV_REDUCE),
            "note": ("PI disposition 2026-07-17: drop syst_e_dla_completeness from the DESI "
                     "C_data while alpha_DLA floats (cup1d 'red'); e_dla stays an out-of-model "
                     "stress test, NOT a mean template"),
        },
        "pi_decision_records": pi_records,
    }

    # --- outputs / surveys -------------------------------------------------------------------
    outputs = {
        "format": "GetDist/cobaya (.txt + .paramnames + .yaml + .health.json)",
        "public": RF.PUBLIC_DIR,
        "private": RF.PRIVATE_DIR + " (DESI; gitignored)",
    }
    surveys = {sv: {"leg": info["leg"], "private": bool(info["private"]),
                    "metals": bool(info["metals"])}
               for sv, info in RF.SURVEY.items()}

    # --- provenance ----------------------------------------------------------------------------
    self_path = os.path.abspath(__file__)
    provenance = {
        "git_commit": _git("rev-parse", "HEAD") or "UNKNOWN (not a git tree?)",
        "generator": {
            "path": "scripts/gen_analysis_lock.py",
            "sha256": _sha256_file(self_path),
        },
        "pi_decision_records": pi_records,
        "notes_repo": NOTES_REPO + " (READ-ONLY)",
        "generation_inputs": {
            "blind_lock": "blind.lock (READ-ONLY verbatim copy; NEVER written)",
            "ensemble_manifest": os.path.join("checkpoints", PE.MANIFEST_BASENAME),
            "checkpoints_dir": MAIN_CHECKPOINTS_DIR,
        },
        "contract": ("Generated ONLY by scripts/gen_analysis_lock.py. Hand-edits prohibited. "
                     "Regeneration of the committed lock happens ONLY at the freeze cut, "
                     "after the independent review panel, on the merged tree."),
    }

    lock = {
        "analysis": "hcd_priya real-data blind P1D cosmology fit",
        "blinding": blinding,
        "covariance": covariance,
        "created_utc": _utcnow(),
        "decisions": decisions,
        "emulator": emulator,
        "git_commit": (_git("rev-parse", "--short", "HEAD") or "UNKNOWN"),
        "legs": legs,
        "nuts": nuts,
        "outputs": outputs,
        "prior_constants": prior_constants,
        "priors": priors,
        "provenance": provenance,
        "signatures": signatures,
        "surveys": surveys,
        "uncovered_constants": collect_uncovered_constants(),
    }
    lock = _jn(lock, "lock")
    assert_not_poisoned(lock)
    return lock


# ---------------------------------------------------------------------------------------------
# check mode
# ---------------------------------------------------------------------------------------------
def _strip_volatile(flat):
    return {k: v for k, v in flat.items() if k not in VOLATILE_PATHS}


def check_lock(leg_summaries, allow_stale_legs=False, committed_path=None):
    """Compare the COMMITTED analysis.lock against a fresh live generation, modulo
    VOLATILE_PATHS. Returns (n_diffs, report_lines). n_diffs == 0 <=> identical."""
    committed_path = committed_path or os.path.join(_REPO_ROOT, "analysis.lock")
    if not os.path.exists(committed_path):
        return 1, [f"NO COMMITTED LOCK at {committed_path}"]
    with open(committed_path) as f:
        committed = json.load(f)
    live = generate_lock(leg_summaries, allow_stale_legs=allow_stale_legs)

    fc = _strip_volatile(_flatten(committed))
    fl = _strip_volatile(_flatten(live))
    only_committed = sorted(set(fc) - set(fl))
    only_live = sorted(set(fl) - set(fc))
    differ = sorted(k for k in set(fc) & set(fl)
                    if json.dumps(fc[k], sort_keys=True) != json.dumps(fl[k], sort_keys=True))

    lines = []
    if not (only_committed or only_live or differ):
        lines.append(f"[gen_analysis_lock --check] OK: committed {committed_path} == live "
                     f"generation (modulo volatile fields {sorted(VOLATILE_PATHS)})")
        return 0, lines

    lines.append("=" * 92)
    lines.append(f"[gen_analysis_lock --check] COMMITTED LOCK IS STALE vs LIVE CODE")
    lines.append(f"  committed: {committed_path} (git_commit="
                 f"{committed.get('git_commit', '?')}, created_utc="
                 f"{committed.get('created_utc', '?')})")
    lines.append(f"  live:      generated in-memory at git_commit={live.get('git_commit')}")
    lines.append(f"  volatile fields ignored: {sorted(VOLATILE_PATHS)}")
    lines.append("-" * 92)
    if differ:
        lines.append(f"VALUE DIFFERS ({len(differ)} fields):")
        for k in differ:
            lines.append(f"  {k}")
            lines.append(f"      committed: {json.dumps(fc[k], sort_keys=True)[:300]}")
            lines.append(f"      live:      {json.dumps(fl[k], sort_keys=True)[:300]}")
    if only_committed:
        lines.append(f"ONLY IN COMMITTED ({len(only_committed)} fields; retired/renamed in "
                     f"live):")
        for k in only_committed:
            lines.append(f"  {k} = {json.dumps(fc[k], sort_keys=True)[:200]}")
    if only_live:
        lines.append(f"ONLY IN LIVE ({len(only_live)} fields; new since the committed lock):")
        for k in only_live:
            lines.append(f"  {k} = {json.dumps(fl[k], sort_keys=True)[:200]}")
    # targeted poisoned-row callout (the known June-lock defects, named explicitly):
    poison = []
    czs = fc.get("priors.zslope.mu")
    if isinstance(czs, list) and len(czs) == 3 and \
            all(_close(a, b) for a, b in zip(czs, POISON_RATIO_ZSLOPE)):
        poison.append("committed priors.zslope.mu is the RETIRED wrong-object ratio slope "
                      f"{czs} (hcd-dndx-zslope-bug)")
    for k, v in fc.items():
        if k.endswith("alpha_hcd_mu") and isinstance(v, list) and v and \
                _close(v[0], POISON_LLS_ALLZ_MEDIAN, rtol=1e-4):
            poison.append(f"committed {k}[0] is the RETIRED all-z-median LLS centre "
                          f"{v[0]} (CENTER-construction bug, PI 2026-06-17)")
    if poison:
        lines.append("-" * 92)
        lines.append("KNOWN POISONED ROWS PRESENT IN THE COMMITTED LOCK:")
        for p in poison:
            lines.append(f"  !!! {p}")
    lines.append("-" * 92)
    lines.append(f"TOTAL: {len(differ)} differing + {len(only_committed)} committed-only + "
                 f"{len(only_live)} live-only fields.")
    lines.append("This is EXPECTED until the freeze step (the committed lock is the stale June "
                 "one; regeneration is gated on the independent review panel + the merged "
                 "tree). At the freeze cut this check must exit 0 after regeneration.")
    lines.append("=" * 92)
    return 1, lines


# ---------------------------------------------------------------------------------------------
# write guard + CLI
# ---------------------------------------------------------------------------------------------
def _refuse_reserved_basename(path, what):
    """Auxiliary writes (legs cache / leg summaries) must never target a lock file."""
    if path and os.path.basename(os.path.abspath(path)) in ("blind.lock", "analysis.lock"):
        raise SystemExit(f"REFUSED: {what}={path} targets a reserved lock filename; "
                         f"auxiliary outputs must never be named blind.lock/analysis.lock.")
    return path


def _guard_out_path(out_path, i_am_the_freeze_step):
    """REFUSE dangerous targets. blind.lock: always (no override). The tree's analysis.lock
    (this tree's, or the main tree's): only with --i-am-the-freeze-step, and then only for
    THIS generator's own tree (a worktree run must never write the main tree's lock)."""
    ap = os.path.abspath(out_path)
    if os.path.basename(ap) == "blind.lock":
        raise SystemExit("REFUSED: blind.lock is out of scope for this generator and must "
                         "never be written by ANYONE (no override exists).")
    protected = {os.path.join(_REPO_ROOT, "analysis.lock"),
                 "/home/mfho/hcd_priya/analysis.lock"}
    if ap in protected:
        if not i_am_the_freeze_step:
            raise SystemExit(
                f"REFUSED: {ap} is the committed analysis.lock. Regeneration happens ONLY at "
                f"the freeze cut (independent review panel + merged tree) via "
                f"--i-am-the-freeze-step. Normal use writes a dated review copy (default "
                f"--out under the system tmp dir).")
        if ap != os.path.join(_REPO_ROOT, "analysis.lock"):
            raise SystemExit(
                f"REFUSED: --i-am-the-freeze-step can only write THIS tree's analysis.lock "
                f"({os.path.join(_REPO_ROOT, 'analysis.lock')}); refusing to write {ap} from "
                f"a different tree. Run the generator from the tree being frozen.")
    return ap


def _default_out():
    short = _git("rev-parse", "--short", "HEAD") or "nogit"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(tempfile.gettempdir(),
                        f"analysis.lock.candidate.{stamp}.{short}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output path (default: a dated review copy under the system tmp "
                         "dir; the committed analysis.lock REQUIRES --i-am-the-freeze-step)")
    ap.add_argument("--i-am-the-freeze-step", action="store_true",
                    help="allow writing THIS tree's committed analysis.lock (freeze cut only, "
                         "after the independent review panel, on the merged tree)")
    ap.add_argument("--check", action="store_true",
                    help="compare the committed analysis.lock to a fresh live generation; "
                         "exit 0 identical-modulo-volatile, exit 1 with a per-field diff")
    ap.add_argument("--legs-cache", default=None,
                    help="reuse leg summaries from this JSON (skips the heavy ctx builds)")
    ap.add_argument("--save-legs-cache", default=None,
                    help="after building, save the leg summaries to this JSON")
    ap.add_argument("--allow-stale-legs-cache", action="store_true",
                    help="MACHINERY TESTING ONLY: accept a legs cache whose embedded "
                         "signatures differ from live (the freeze step must never pass this)")
    ap.add_argument("--in-process-legs", action="store_true",
                    help="build leg summaries in-process instead of subprocesses")
    ap.add_argument("--emit-leg-summary", default=None, metavar="SURVEY",
                    help="INTERNAL: build one survey's leg summary and write JSON to --leg-out")
    ap.add_argument("--leg-out", default=None, help="INTERNAL: --emit-leg-summary output file")
    a = ap.parse_args(argv)

    _refuse_reserved_basename(a.save_legs_cache, "--save-legs-cache")
    _refuse_reserved_basename(a.leg_out, "--leg-out")

    if a.emit_leg_summary:
        if not a.leg_out:
            raise SystemExit("--emit-leg-summary requires --leg-out")
        summary = build_leg_summary(a.emit_leg_summary)
        with open(a.leg_out, "w") as f:
            json.dump(summary, f, sort_keys=True, indent=2)
            f.write("\n")
        print(f"[gen_analysis_lock] leg summary for {a.emit_leg_summary!r} -> {a.leg_out}")
        return 0

    leg_summaries = collect_leg_summaries(legs_cache=a.legs_cache,
                                          save_legs_cache=a.save_legs_cache,
                                          in_process=a.in_process_legs)

    if a.check:
        rc, lines = check_lock(leg_summaries, allow_stale_legs=a.allow_stale_legs_cache)
        print("\n".join(lines))
        return rc

    out = _guard_out_path(a.out or _default_out(), a.i_am_the_freeze_step)
    lock = generate_lock(leg_summaries, allow_stale_legs=a.allow_stale_legs_cache)
    payload = canonical_dumps(lock)
    with open(out, "w") as f:
        f.write(payload)
    print(f"[gen_analysis_lock] wrote {out} ({len(payload)} bytes, "
          f"{len(lock)} top-level sections: {', '.join(sorted(lock))})")
    if os.path.abspath(out) == os.path.join(_REPO_ROOT, "analysis.lock"):
        print("[gen_analysis_lock] !!! FREEZE-STEP WRITE of the committed analysis.lock -- "
              "this is only legitimate after the independent review panel, on the merged "
              "tree, with PI sign-off. !!!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
