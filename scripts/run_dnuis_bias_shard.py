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

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb
from hcd_analysis.emulator.inference import (HCD_LIT_OVER_SIM, HCD_LLS_SURVEY_BOOST,
                                             HCD_LLS_SURVEY_FRAC_SIGMA)

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"

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


def build_arm_ctx(arm, survey, with_mf, with_eboss_unused=None):
    """Build the single-survey production ctx for an arm. metals_on/sample_metals ON for
    DESI/eBOSS (False for KS). Returns (ctx, d, inject_spec). The arm runs on ONE survey's legs:
    we build a single-survey ctx by restricting the leg list AFTER build (keep it simple)."""
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
    # build_legb_ctx assembles DESI+KS(+eBOSS); we then keep ONLY the chosen survey's leg(s) so the
    # arm fits a single-survey likelihood (the bias is per-survey).  survey= sets the LLS pin.
    ctx, d = build_legb_ctx(
        ensemble_ckpts=members, use_xclass=True,
        with_mf=with_mf, mf_with_floor=with_mf,
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=(survey == "eboss"),
        metals_on=metals, sample_metals=metals,
        hierarchical_hcd=False, survey=PIN_KEY)

    # restrict to the chosen survey's legs (single-survey bias arm).
    legs = [l for l in ctx.legs if l.name == LEG_NAME]
    if not legs:
        raise SystemExit(f"no leg named {LEG_NAME!r} in ctx (legs={[l.name for l in ctx.legs]})")
    ctx = ctx._replace(legs=legs)

    # map arm -> inject_spec
    if arm == "metal_misspec":
        # GUARD: KS legs are metals_on=False, so the metal injection is a SILENT no-op (the forward
        # cannot SEE it AND the mock is unchanged) → a meaningless PASS. Hard-fail the cell so it is
        # never run. The metal-misspec arm is only meaningful on metals_on surveys (DESI/eBOSS).
        if survey == "ks":
            raise SystemExit(
                "metal_misspec is a NO-OP on KS (KS legs are metals_on=False — the injection neither "
                "perturbs the mock nor is fittable, a meaningless PASS). Run metal_misspec on "
                "desi/eboss only.")
        # REALISM: use the metal model AT EACH SURVEY'S SCALE — the full DESI-DR1 desi_full model
        # (with the unfittable additive SiII–SiII term) for DESI, the McDonald/eBOSS SiIIIcorr form
        # for eBOSS (the metal model the eBOSS data estimator actually uses).
        form = "eboss" if survey == "eboss" else "desi_full"
        inject_spec = {"metal_misspec": {"form": form}}
    elif arm == "resolution":
        inject_spec = {"resolution": {"b_res": 0.02}}
    elif arm == "lls_excess":
        inject_spec = {"lls_truth_boost": LLS_TRUTH_BOOST[survey]}
    elif arm == "metal_matched":
        inject_spec = {}                                # no post-hoc injection; a_SiIII free in fwd
    else:
        raise SystemExit(f"unknown arm {arm!r}")
    return ctx, d, inject_spec


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
    ap.add_argument("--no-mf", dest="with_mf", action="store_false")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 mock, tiny warmup/samples — pipeline + per-mock cost probe")
    ap.set_defaults(with_mf=True)
    a = ap.parse_args()

    if a.smoke:
        a.n_mocks = 1
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 30)

    ctx, d, inject_spec = build_arm_ctx(a.arm, a.survey, a.with_mf)
    n_members = len(getattr(ctx.model, "members", [None]))
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[dnuis {a.arm}/{a.survey} shard {a.shard}/{a.n_shards}] mocks={idxs} "
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
    out = os.path.join(a.out_dir, f"{a.arm}_{a.survey}_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a))
    meta.update(inject_spec=inject_spec, n_members=n_members, paired=True,
                legs=[l.name for l in ctx.legs], wall_s=wall, per_fit_wall_s=per_mock_wall)
    with open(out, "wb") as f:
        pickle.dump(dict(arm=a.arm, survey=a.survey, idxs=idxs,
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
