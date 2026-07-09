"""Run ONE shard of a DATA-NUISANCE injection-recovery BIAS arm on the PRODUCTION forward.

The bias gate: build_legb_ctx with the PRODUCTION baseline (N=5 ensemble, 2-param τ₀, lit-pinned
HCD, emucoh + cross-class C_emu, MF P1D+dN/dX, hierarchical_hcd=False) for ONE survey, then per
mock draw θ from the prior → Leg-A matched-C self-draw (clean bias is ZERO by construction) → ADD
the arm's data-side nuisance the forward cannot fit (or offset the truth from the per-survey LLS
pin) → NUTS → record. The recovered (A_p, n_s) shift isolates that nuisance. Per-mock RNG is
fold_in(seed,m) so a shard subset reproduces the mocks a single full run would draw.

PAIRED estimator (the power fix): each shard runs run_legb TWICE per mock at the SAME (seed, mock
index) — once CLEAN (inject_spec=None) and once INJECTED — so the truth θ AND the cosmic noise ε
are shared (both keyed off fold_in(seed,m)); only the injected contaminant differs. The pkl stores
clean_per_mock + inj_per_mock and the analyzer forms the per-mock paired Δbias_z (noise cancels).

Arms (→ inject_spec):
  metal_misspec  — {"metal_misspec":{"form":<survey>}}     : DESI→"desi_full" (the SiII–SiII ADDITIVE
                   term the multiplicative forward _metal_factor STRUCTURALLY cannot fit), eBOSS→
                   "eboss" (the McDonald SiIIIcorr at the eBOSS scale). KS is HARD-SKIPPED (metals_on
                   =False → a silent no-op = meaningless PASS). Forward sample_metals ON.
  resolution     — {"resolution":{"b_res":0.02}}           : P·exp(2 b_res k² R²); the production
                   forward has resolution_on=False so it cannot fit this distortion.
  lls_excess     — {"lls_truth_boost":<per-survey OFF-pin LLS>}: offset the TRUTH LLS OFF the
                   per-survey pin center (the irreducible HCD→n_s budget; flag, don't auto-fail).
  metal_matched  — {} but ctx sample_metals ON and the truth draws a_SiIII from the prior: the
                   marginalization-COST test (no post-hoc injection; just paying for the free a_SiIII).

PER-SURVEY lls_truth_boost (the effective LLS the mock TRUTH carries vs the forward's per-survey pin
center). The forward pin is HCD_LLS_SURVEY_BOOST×lit_over_sim (DESI 1.0×, KS 2.5×; inference.py). The
truth boost must sit OFF the pin center to be a real test:
  desi  = 1.06  (HCD_LIT_OVER_SIM[0] at z_pivot — the cosmic-average lit/sim LLS; ~0.2σ off the DESI
                 pin, σ/μ=0.30)
  ks    = 3.50  (= 2.5×1.4: the KS pin CENTER is 2.5×, so a boost of EXACTLY 2.5 would be a near-null
                 test; we put the truth ~1σ ABOVE the pin (KS σ/μ=0.40 ⇒ ×1.4) — a wrong-pin-center
                 probe within the KS selection-excess band, arXiv:2509.18271 §4.3.3 α_LLS≈2 ⇒ ~2–3×)
  eboss = 1.30  (no per-survey pin → forward keeps the cosmic average; 1.30 is a deliberate STRESS
                 offset since eBOSS is low-k/cosmic and otherwise carries no LLS excess)

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle
import time

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb
from hcd_analysis.emulator.inference import (HCD_LIT_OVER_SIM, HCD_LLS_SURVEY_BOOST,
                                             HCD_LLS_SURVEY_FRAC_SIGMA)

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"
RES_INSTR_BASIS = os.path.join(REPO, "hcd_analysis", "_emulator_data", "res_instr_injection_basis.npz")

# Per-survey TRUTH LLS boost (lit/sim effective LLS the mock truth carries vs the forward pin).
# The arm must put the TRUTH OFF the forward's per-survey pin center, else it is a near-null test.
#   desi  = 1.06 (HCD_LIT_OVER_SIM[0]): the cosmic-average lit/sim LLS dN/dX ratio; the DESI pin
#           center is 1.0× cosmic (HCD_LLS_SURVEY_BOOST["DESI"]=1.0), so 1.06 sits ~0.2σ above the
#           DESI pin (σ/μ=0.30) — a mild realistic mis-center.
#   ks    = 2.5×1.4 = 3.50: the KS pin CENTER is 2.5× cosmic. A truth boost of EXACTLY 2.5 would
#           equal the pin center → a near-NULL test (no offset to recover). We instead put the truth
#           ~1σ ABOVE the pin (KS σ/μ=0.40 ⇒ ×1.4) — a defensible wrong-pin-center probe within the
#           KS selection-excess band (arXiv:2509.18271 α_LLS≈2 ⇒ ~2–3× PRIYA; 3.5× is the high end).
#   eboss = 1.30: no eBOSS per-survey pin (forward keeps the cosmic average 1.0×), so 1.30 is a
#           deliberate STRESS offset (eBOSS is low-k/cosmic, otherwise carries no LLS excess).
LLS_TRUTH_BOOST = {
    "desi":  float(HCD_LIT_OVER_SIM[0]),                                    # 1.06 — ~0.2σ off DESI pin
    "ks":    float(HCD_LLS_SURVEY_BOOST.get("KS", 2.5))
             * (1.0 + float(HCD_LLS_SURVEY_FRAC_SIGMA.get("KS", 0.40))),    # 3.50 — ~1σ ABOVE KS pin
    "eboss": 1.30,                                                          # 1.30 — deliberate stress
}


def _resinj_oos_spec(member, strength, path=RES_INSTR_BASIS):
    """None (default: the scalar in-span arm) or the OOS basis spec dict the resolver
    (``_resolve_res_instr_inject``) loads: ``{"path": npz, "member": m, "strength": x}``."""
    if member is None:
        return None
    return {"path": path, "member": str(member), "strength": float(strength)}


def arm_inject_spec(arm, survey, *, b_res=0.02, ks_resolution_ready=False, oos_member=None, oos_strength=1.0):
    """Map (arm, survey) -> the run_legb inject_spec (PURE; no ctx build, unit-testable). ``b_res``
    sets the resolution injection strength (default 0.02 = the realistic DESI ~1-sigma level derived
    from the data's own syst_e_resolution; the option-a certification brackets it +/-1 sigma over
    {0.015, 0.02, 0.03}). ``oos_member`` (Task 2C) selects an OUT-OF-SPAN per-z basis member from
    ``RES_INSTR_BASIS`` INSTEAD of the scalar ``b_res`` (default None = unchanged scalar arm)."""
    if arm == "metal_misspec":
        if survey == "ks":
            raise SystemExit(
                "metal_misspec is a NO-OP on KS (KS legs are metals_on=False — the injection neither "
                "perturbs the mock nor is fittable, a meaningless PASS). Run metal_misspec on "
                "desi/eboss only.")
        form = "eboss" if survey == "eboss" else "desi_full"
        return {"metal_misspec": {"form": form}}
    if arm == "resolution":
        if survey == "ks" and not ks_resolution_ready:
            raise SystemExit(
                "resolution injection on KS with the DESI pixel PROXY R_z is invalid (~7-15x too large -- KS is "
                "echelle, sigma~3.2 km/s -> a ~70% distortion the forward cannot fit, the -21sigma ESS collapse). "
                "Pass ks_resolution_ready=True only when the echelle R_z + diag surgery is wired (build_arm_ctx "
                "resolution-float on KS, task #5). Then the b_res injection is on the physical echelle scale.")
        oos_spec = _resinj_oos_spec(oos_member, oos_strength)
        return {"resolution": oos_spec if oos_spec is not None else {"b_res": float(b_res)}}
    if arm == "lls_excess":
        return {"lls_truth_boost": LLS_TRUTH_BOOST[survey]}
    if arm == "metal_matched":
        return {}
    raise SystemExit(f"unknown arm {arm!r}")


# The 4-arm resolution comparison bracket: each treatment is a DISTINCT covariance/forward handling of the
# spectral-resolution systematic, run side-by-side on the SAME injected mocks. Tagged into the output name so
# the arms never collide + the analyzer compares them per leg (see 2026-07-02-coherent-cov-vs-float doc).
TREATMENTS = ("a", "b", "c", "d")


def treatment_flags(treatment, *, c_prior_sigma=0.05):
    """Map a bracket treatment label -> the run_dnuis flags:
      a = option-a  : resolution stays IN the covariance, NO float (the deployed baseline).
      b = option-b  : float f_res, TIGHT prior N(0, 0.02) (ours).
      c = option-b WIDE (cup1d-faithful / eBOSS leg-match): float f_res, prior N(0, c_prior_sigma).
      d = arm-D     : coherent cross-z covariance mode, NO float (marginalize resolution in the cov)."""
    t = str(treatment).lower()
    if t == "a":
        return dict(float_res=False, coherent_res=False, f_res_amp_sigma=None)
    if t == "b":
        return dict(float_res=True, coherent_res=False, f_res_amp_sigma=None)      # tight 0.02
    if t == "c":
        return dict(float_res=True, coherent_res=False, f_res_amp_sigma=float(c_prior_sigma))
    if t == "d":
        return dict(float_res=False, coherent_res=True, f_res_amp_sigma=None)
    raise SystemExit(f"unknown treatment {treatment!r} (choose one of {TREATMENTS})")


def build_arm_ctx(arm, survey, with_mf, with_eboss_unused=None, *, b_res=0.02, float_res=False,
                  coherent_res=False, coh_amp=1.0, f_res_amp_sigma=None, pin_hub=False,
                  oos_member=None, oos_strength=1.0):
    """Build the single-survey production ctx for an arm. metals_on/sample_metals ON for
    DESI/eBOSS (False for KS). Returns (ctx, d, inject_spec). The arm runs on ONE survey's legs:
    we build a single-survey ctx by restricting the leg list AFTER build (keep it simple).
    ``oos_member``/``oos_strength`` (Task 2C) select the OUT-OF-SPAN basis injection for the
    resolution arm instead of the scalar ``b_res``; default None -> byte-identical scalar arm."""
    members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    if not members:
        raise SystemExit(f"no production ensemble checkpoints at {PROD_PREFIX}*.eqx")

    # The leg NAME (load_*_leg) and the survey PIN key (HCD_LLS_SURVEY_BOOST/FRAC_SIGMA) per arg.
    # eBOSS's leg name is 'eBOSS' (not 'EBOSS') and it has NO per-survey LLS pin (the boost map only
    # carries DESI/KS) → survey="eBOSS" returns the cosmic-average pin (boost 1.0), which is the
    # intended eBOSS forward (the lls_excess arm then offsets the TRUTH by the 1.30 stress instead).
    LEG_NAME = {"desi": "DESI", "ks": "KS", "eboss": "eBOSS"}[survey]
    PIN_KEY = {"desi": "DESI", "ks": "KS", "eboss": "eBOSS"}[survey]
    metals = survey in ("desi", "eboss")               # KS conservative-mode subtracts metals
    # KS instrument-f_res (task #5): use the ECHELLE R_z (3.2 km/s) + diagonal cov surgery via ks_kwargs so the
    # b_res LSF injection is PHYSICAL, not the ~15x-too-large DESI pixel proxy (which made KS un-fittable, the
    # -21sigma collapse). Only on the KS resolution-float path; None elsewhere -> byte-identical for DESI/eBOSS/KS-off.
    _ks_kwargs = ({"resolution_float": True, "k_max": 0.065} if (survey == "ks" and (float_res or coherent_res)) else None)
    # build_legb_ctx assembles DESI+KS(+eBOSS); we then keep ONLY the chosen survey's leg(s) so the
    # arm fits a single-survey likelihood (the bias is per-survey).  survey= sets the LLS pin.
    ctx, d = build_legb_ctx(
        ensemble_ckpts=members, use_xclass=True,
        with_mf=with_mf, mf_with_floor=with_mf,
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=(survey == "eboss"),
        metals_on=metals, sample_metals=metals,
        sample_res=float_res,                              # option-b: float f_res + cov_b (DESI rank-1 / eBOSS rescale / KS diag)
        coherent_res=coherent_res, coh_amp=coh_amp,        # arm-D: coherent cross-z cov mode, NO forward float
        f_res_amp_sigma=f_res_amp_sigma,                   # arm-C wide / eBOSS + KS leg-match prior (None -> tight 0.02)
        ks_kwargs=_ks_kwargs,                              # KS echelle R_z + diag surgery (task #5); None => proxy (DESI/eBOSS/off)
        hierarchical_hcd=False, survey=PIN_KEY)

    # restrict to the chosen survey's legs (single-survey bias arm).
    legs = [l for l in ctx.legs if l.name == LEG_NAME]
    if not legs:
        raise SystemExit(f"no leg named {LEG_NAME!r} in ctx (legs={[l.name for l in ctx.legs]})")
    ctx = ctx._replace(legs=legs)

    # DIAGNOSTIC (mechanism ablation): PIN hub (theta9[5]) to a tight window at its box centre in BOTH the
    # truth-draw and the fit (draw_leg_a_leg_truth traces _legb_priors_only -> same theta_unit bounds). The
    # empirical mediation analysis found hub is the release valve that absorbs the eBOSS resolution offset
    # (shifts -0.78sigma) and drags n_s down (rho +0.50); pinning it tests that causally + the tighten-hub
    # mitigation. Default off -> byte-identical.
    if pin_hub:
        from hcd_analysis.emulator import closure_legb as _CL
        lo = np.asarray(_CL._THETA_UNIT_LO, float).copy(); hi = np.asarray(_CL._THETA_UNIT_HI, float).copy()
        j = 5; mid = 0.5 * (lo[j] + hi[j]); lo[j], hi[j] = mid - 0.01, mid + 0.01   # hub pinned ~box centre
        ctx = ctx._replace(theta_unit_lo=lo, theta_unit_hi=hi)

    # map arm -> inject_spec (pure helper; the resolution b_res is configurable for the +/-1 sigma
    # option-a certification). metal_misspec REALISM: desi_full (with the unfittable additive SiII-SiII
    # term) for DESI, McDonald/eBOSS SiIIIcorr for eBOSS; the KS metal no-op guard lives in the helper.
    # oos_member (Task 2C): None -> unchanged scalar b_res resolution arm (byte-identical).
    inject_spec = arm_inject_spec(arm, survey, b_res=b_res, ks_resolution_ready=(_ks_kwargs is not None),
                                  oos_member=oos_member, oos_strength=oos_strength)
    return ctx, d, inject_spec


def _out_arm(arm, b_res_oos_member):
    """Write-time output arm tag (Task 2C): an OOS member selects the "resolution_oos" tag so its
    pkls never collide with the scalar in-span "resolution" arm's; member None -> unchanged (byte-
    identical for every arm, including non-resolution ones). The OOS member is ONLY meaningful for
    arm=="resolution" (defensive: never retag a non-resolution arm, so a metals/lls pkl can't be
    mislabeled resolution_oos and given the adversarial FLAG band -- main() also raises on that combo)."""
    return "resolution_oos" if (arm == "resolution" and b_res_oos_member) else arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["metal_misspec", "resolution", "lls_excess", "metal_matched"])
    ap.add_argument("--survey", required=True, choices=["desi", "ks", "eboss"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=8)         # paired → 16 fits/cell at this default
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)     # trimmed for budget (~13 CPU-h/fit)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--b-res", type=float, default=0.02,
                    help="resolution injection strength (arm=resolution). Default 0.02 = the realistic "
                         "DESI ~1sigma level from syst_e_resolution; certification bracket +/-1sigma "
                         "{0.015,0.02,0.03}. Ignored for non-resolution arms.")
    ap.add_argument("--b-res-oos-member", choices=["bres1", "bres2", "bres_real", "bstar"], default=None,
                    help="OUT-OF-SPAN instrument-resolution arm: inject the per-z basis MEMBER from "
                         "res_instr_injection_basis.npz instead of the scalar --b-res. bstar=analytic worst-n_s "
                         "(adversarial); bres1/bres2=z-incoherent named threats (adversarial); bres_real=measured "
                         "residual (realistic). Requires --arm resolution. Default None = the scalar in-span arm.")
    ap.add_argument("--b-res-oos-strength", type=float, default=1.0,
                    help="strength scale on the OOS basis member (+/-1sigma 3-point bracket). Ignored unless "
                         "--b-res-oos-member.")
    ap.add_argument("--float-res", dest="float_res", action="store_true",
                    help="OPTION-B: float the 2-param f_res spectral-resolution nuisance + remove the "
                         "resolution mode from the covariance. BOTH legs: DESI = per-z rank-1 cov_b; eBOSS = "
                         "multiplicative sigma-rescale cov_b. Default off = option-a (resolution stays in the "
                         "cov). NOTE: the tight amp prior N(0,0.02) is DESI-derived; eBOSS's own resolution "
                         "is ~2x larger (b_res~0.044), so eBOSS option-b under-covers unless the prior is "
                         "leg-matched (open PI decision -- see 2026-07-02-resolution-findings.md).")
    ap.add_argument("--coherent-res", dest="coherent_res", action="store_true",
                    help="ARM-D: remove resolution from the covariance (per-survey) and RE-ADD it as ONE "
                         "coherent cross-z mode s^2*outer(e,e) -- NO forward f_res float. Marginalizes the "
                         "coherent resolution error in the covariance (linear-Gaussian equivalent of floating "
                         "one amplitude); s=coh_amp=1 IS the shipped 1-sigma resolution uncertainty (no prior "
                         "to tune). PREFERRED for eBOSS (low-k, n_s<->resolution degenerate -> a float is "
                         "prior-dominated + under-covers). Mutually exclusive with --float-res. See "
                         "2026-07-02-coherent-cov-vs-float-resolution.md.")
    ap.add_argument("--coh-amp", type=float, default=1.0,
                    help="ARM-D coherent-mode amplitude s (default 1.0 = the shipped 1-sigma resolution "
                         "uncertainty). cov gains s^2*outer(e,e). Ignored unless --coherent-res.")
    ap.add_argument("--treatment", choices=TREATMENTS, default=None,
                    help="4-arm resolution comparison bracket (sets the flags + the output tag; overrides "
                         "--float-res/--coherent-res): a=option-a (in cov, no float); b=option-b tight "
                         "(float, prior 0.02); c=option-b wide/cup1d-faithful (float, prior --c-prior-sigma); "
                         "d=arm-D (coherent cov, no float). The driver varies this a/b/c/d per leg+injection.")
    ap.add_argument("--c-prior-sigma", type=float, default=0.05,
                    help="arm-C (treatment c) f_res_amp prior width (default 0.05 = eBOSS-leg-matched; use a "
                         "wider value for cup1d's loose default). Ignored unless --treatment c.")
    ap.add_argument("--pin-hub", dest="pin_hub", action="store_true",
                    help="DIAGNOSTIC (mechanism ablation): pin hub (theta9[5]) to a tight window at its box "
                         "centre in BOTH the truth-draw and the fit. Tests whether hub is the release valve "
                         "that mediates the eBOSS resolution->n_s leak (empirical mediation: hub shifts "
                         "-0.78sigma, rho(n_s,hub)=+0.50). Default off = byte-identical.")
    ap.add_argument("--no-mf", dest="with_mf", action="store_false")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 mock, tiny warmup/samples — pipeline + per-mock cost probe")
    ap.set_defaults(with_mf=True)
    a = ap.parse_args()

    # OOS instrument-resolution member is a selector WITHIN the resolution arm; combining it with a
    # non-resolution --arm is a user error (the injection would be ignored yet the pkl mislabeled
    # resolution_oos + given the adversarial FLAG band). Fail loud (Task-2C review, reviewer A).
    if a.b_res_oos_member and a.arm != "resolution":
        raise SystemExit(f"--b-res-oos-member requires --arm resolution (got --arm {a.arm})")

    if a.smoke:
        a.n_mocks = 1
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 30)

    # resolve the comparison-bracket treatment (a/b/c/d) -> flags + the output TAG so the 4 arms never
    # collide + the analyzer groups them per leg. --treatment overrides the low-level flags; without it,
    # derive the tag from the flags (backward compat).
    f_res_amp_sigma = None
    if a.treatment:
        fl = treatment_flags(a.treatment, c_prior_sigma=a.c_prior_sigma)
        a.float_res, a.coherent_res, f_res_amp_sigma = fl["float_res"], fl["coherent_res"], fl["f_res_amp_sigma"]
        treatment = a.treatment
    else:
        if a.float_res and a.coherent_res:
            raise SystemExit("--float-res (option-b) and --coherent-res (arm-D) are mutually exclusive arms")
        treatment = "d" if a.coherent_res else ("b" if a.float_res else "a")
    ctx, d, inject_spec = build_arm_ctx(a.arm, a.survey, a.with_mf, b_res=a.b_res, float_res=a.float_res,
                                        coherent_res=a.coherent_res, coh_amp=a.coh_amp,
                                        f_res_amp_sigma=f_res_amp_sigma, pin_hub=a.pin_hub,
                                        oos_member=a.b_res_oos_member, oos_strength=a.b_res_oos_strength)
    n_members = len(getattr(ctx.model, "members", [None]))
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[dnuis {a.arm}/{a.survey} treat={treatment}"
          f"{'' if f_res_amp_sigma is None else f'(prior_sig={f_res_amp_sigma})'} "
          f"shard {a.shard}/{a.n_shards}] mocks={idxs} "
          f"members={n_members} legs={[l.name for l in ctx.legs]} "
          f"inject_spec={inject_spec} mf={a.with_mf} PAIRED "
          f"(warmup={a.n_warmup} samples={a.n_samples} mtd={a.max_tree_depth})")

    # PAIRED clean-vs-injected estimator (the power fix). For each mock we run run_legb TWICE at
    # the SAME (seed, mock_indices, n_warmup, n_samples): once CLEAN (inject_spec=None) and once
    # INJECTED (the arm's inject_spec). run_legb's leg_a path keys truth θ, the noise ε AND the
    # NUTS init off jax.random.fold_in(seed, m), so the clean and injected mocks share the SAME
    # truth + SAME noise — the ONLY difference is the injected contaminant. The analyzer then
    # forms the per-mock paired Δbias_z = bias(injected) − bias(clean); the shared cosmic noise
    # cancels, collapsing the unpaired SE≈0.22 to the contaminant-only SE.
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[dnuis {a.arm}/{a.survey} shard {a.shard}] CLEAN run (inject_spec=None) ...")
    clean_per_mock = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[dnuis {a.arm}/{a.survey} shard {a.shard}] INJECTED run (inject_spec={inject_spec}) ...")
    inj_per_mock = run_legb(ctx, d, inject_spec=inject_spec, **run_kw)
    wall = time.time() - t0
    n_pairs = max(min(len(clean_per_mock), len(inj_per_mock)), 1)
    per_mock_wall = wall / (2.0 * n_pairs)                  # cost of ONE fit (2 fits per paired mock)

    os.makedirs(a.out_dir, exist_ok=True)
    # tag the output with the TREATMENT so the 4 bracket arms (a/b/c/d) never overwrite each other.
    # out_arm (Task 2C): an OOS member -> "resolution_oos" so it never collides with the scalar
    # in-span "resolution" arm's pkls; member None -> unchanged (byte-identical).
    out_arm = _out_arm(a.arm, a.b_res_oos_member)
    out = os.path.join(a.out_dir, f"{out_arm}_{treatment}_{a.survey}_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a))
    meta.update(inject_spec=inject_spec, n_members=n_members, paired=True, treatment=treatment,
                f_res_amp_sigma=f_res_amp_sigma, legs=[l.name for l in ctx.legs],
                wall_s=wall, per_fit_wall_s=per_mock_wall)
    with open(out, "wb") as f:
        pickle.dump(dict(arm=out_arm, treatment=treatment, survey=a.survey, idxs=idxs,
                         clean_per_mock=clean_per_mock, inj_per_mock=inj_per_mock,
                         meta=meta), f)
    n_div = (sum(int(r.get("n_div", 0) > 0) for r in clean_per_mock)
             + sum(int(r.get("n_div", 0) > 0) for r in inj_per_mock))
    print(f"[dnuis {a.arm}/{a.survey} shard {a.shard}] wrote {len(inj_per_mock)} paired mocks "
          f"({n_div} divergent fits across both runs) -> {out}")
    if a.smoke:
        print(f"[SMOKE] {len(inj_per_mock)} paired mock(s) in {wall:.1f}s "
              f"(per-FIT {per_mock_wall:.1f}s, 2 fits/mock); n_div={n_div}")


if __name__ == "__main__":
    main()
