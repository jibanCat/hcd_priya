"""KS selection-function mock-challenge shard runner (spec 2026-07-18 + Amendment 2, PI A2.8
sign-offs 2026-07-19; PI decision 2d).

One arm x one shard of the paired-by-construction campaign: truth drawn from the deployed
post-K1a KS prior at (seed, mock), the LLS (and, for K5, subDLA) truth column multiplied by the
arm's z-profile B(z) (registry scripts/ks_selboost_arms.py -> the closure_legb scalar-or-dict
truth-boost hook), fit with NUTS on the DEPLOYED production KS forward. K0 runs CLEAN once and
every boosted arm pairs against it CROSS-PKL (same seed + mock index => identical truth draw
and standard-normal noise vector g by run_legb's fold_in construction; the analyzer ingest
asserts the pairing certificate per pair).

Clone-and-adapt from run_dla_selfdraw_shard.py (NOT a parameterized shared runner: the assert
blocks are survey-specific and load-bearing). DELTAS vs the DESI runner (spec Sec 5):
  (a) ctx = build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True): leg KS + PIN_KEY
      "KS" (per-survey LLS pin, boost 2.5) + metals OFF. The KS branch of use_prod_forward was
      VERIFIED (risk 10): it pulls all 5 knobs from prod_forward_config("KS") (sample_res=True,
      f_res_amp_sigma=0.15, metal_prior="uniform", metals=False, ks_kwargs
      {resolution_float: True, k_max: 0.065}) + NORC via prod_norc_forward(), and the explicit
      ks_kwargs k_max OVERRIDES the 0.045 NORC auto-cap (closure_legb.build_legb_ctx) — the
      parity assert below certifies 0.045 < max(leg.k) <= 0.065 so a silent fall-back to the
      auto-cap window trips.
  (b) the KS assert block replaces the DESI one (assert_ks_forward): metals dropped, DESI
      dla_cov_reduced/metal_node_z asserts DROP OUT (KS cov = the Karacayli 2021 conservative
      182x182 file, untouched by DESI_DLA_COV_REDUCE), dla_forward_frac == 0.0 (ALSO certifies
      the K5 DLA-ride-along exact no-op claim).
  (c) PRIOR-STATE TRIPWIRE (assert_prior_state): forward_signature provably does NOT cover
      prior constants, so the required --expect-lls-boost/--expect-lls-frac-sigma args are
      asserted against the DEPLOYED inference constants, PLUS exact asserts that the deployed
      LLS center constants equal the post-correction K1a values (lit/sim 0.995, ratio-slope
      0.764, widened base sigma 0.287 / hedge 0.574, realfit z-slope 2.127) and a band assert
      on the BUILT KS prior center via inference.assert_hcd_pivot_z3. All constants + both
      freeze signatures stamped into meta; the analyzer re-asserts them. Launching before the
      K1a swap, or after any later drift, fails loud.
  (d) --arm-id (registry key) instead of --boost; profiles resolved python-side (no JSON
      through bash). Registry name + profile spec + evaluated B(z_global) + z_global +
      registry_signature stamped into meta.
  (e) mode: K0 clean-only, boosted arms boost-only (derived from the registry; an explicit
      --mode must agree — fail loud on contradiction).
  (f) emulator-span logging (risk 12): per-mock max truth alpha_hcd_z per class vs the
      training-cache w_c_cache column max, out-of-span rows flagged in meta (readout caveat,
      not a hard failure; mainly relevant to contingency C2).

Cross-campaign constants: seed 20260615, NUTS 250/300, max_tree_depth 10, dense mass
(pilot-gated per Sec 7 before any array). SMOKE writes a .smoke-suffixed pkl the analyzer can
never pool (F6a convention).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse, functools, os, pickle, time
import numpy as np
print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401 (x64)
import hcd_analysis.emulator.closure_legb as CL
import hcd_analysis.emulator.inference as INF
from hcd_analysis.emulator.closure_legb import run_legb
from scripts.run_dnuis_bias_shard import build_arm_ctx
import scripts.ks_selboost_arms as AR
from scripts.ks_selboost_arms import batch_cells  # noqa: F401 (re-export for the batch script)

# the post-correction K1a prior constants of record (landed 2026-07-18; spec Sec 8 hard gate).
# EXACT equality — any later drift of the deployed constants must abort the campaign.
K1A_EXPECT = dict(
    lit_over_sim_lls=0.995,        # inference.HCD_LIT_OVER_SIM[0] (definition-matched refit)
    lit_over_sim_slope_lls=0.764,  # inference.HCD_LIT_OVER_SIM_SLOPE[0]
    lls_base_frac_sigma=0.287,     # the WIDENED base width (kernel common-mode included)
    lls_hedge2x_frac_sigma=0.574,  # the 2x cosmic-variance hedge constant (presence certifies
                                   # the widened-width machinery landed, not just the center)
    lls_realfit_zslope=2.127,      # the constrained lit-law forward z-slope
)


def assert_prior_state(expect_lls_boost, expect_lls_frac_sigma):
    """The K1a constant-swap TRIPWIRE (spec 5c). Asserts the caller's --expect-* args equal the
    DEPLOYED per-survey KS constants AND that every K1A_EXPECT post-correction constant is the
    deployed value, then returns the prior-constants stamp dict for shard meta (the analyzer
    ingest re-asserts homogeneity across every pkl)."""
    b = float(INF.HCD_LLS_SURVEY_BOOST["KS"])
    s = float(INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"])
    assert float(expect_lls_boost) == b, (
        f"--expect-lls-boost {expect_lls_boost} != deployed HCD_LLS_SURVEY_BOOST['KS'] {b}: "
        f"the prior state this campaign was designed against has drifted (or the wrong "
        f"expectation was passed) — refusing to run")
    assert float(expect_lls_frac_sigma) == s, (
        f"--expect-lls-frac-sigma {expect_lls_frac_sigma} != deployed "
        f"HCD_LLS_SURVEY_FRAC_SIGMA['KS'] {s} — refusing to run")
    deployed = dict(
        lit_over_sim_lls=float(INF.HCD_LIT_OVER_SIM[0]),
        lit_over_sim_slope_lls=float(INF.HCD_LIT_OVER_SIM_SLOPE[0]),
        lls_base_frac_sigma=float(INF.HCD_LLS_SURVEY_FRAC_SIGMA["DESI"]),
        lls_hedge2x_frac_sigma=float(INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X["DESI"]),
        lls_realfit_zslope=float(INF.HCD_LLS_REALFIT_ZSLOPE),
    )
    for k, want in K1A_EXPECT.items():
        assert deployed[k] == want, (
            f"deployed {k} = {deployed[k]} != post-correction K1a value {want}: the campaign "
            f"is gated behind the K1a constant swap (spec Sec 8) and forward_signature does "
            f"NOT cover prior constants — refusing to run on a drifted/stale prior")
    return dict(lls_survey_boost_ks=b, lls_frac_sigma_ks=s, **deployed,
                hcd_prior_signature=INF.hcd_prior_signature())


def assert_ks_forward(ctx, leg):
    """The KS deployed-forward parity block (spec 5b) — fail LOUD before any NUTS."""
    fc = CL.prod_forward_config("KS")
    norc = CL.prod_norc_forward()
    assert bool(ctx.res_corr_on) == norc["res_corr_on"], "selboost forward != deployed NORC res_corr"
    assert bool(ctx.fix_alpha_res) == norc["fix_alpha_res"], "selboost fix_alpha_res != deployed NORC"
    assert bool(ctx.sample_res) == fc["sample_res"] is True, \
        "prod echelle f_res float not wired into the selboost arm"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"] == 0.15, \
        f"f_res prior width {ctx.f_res_amp_sigma} mismatch vs deployed KS 0.15"
    assert bool(ctx.sample_metals) is False and bool(leg.metals_on) is False \
        and fc["metals"] is False, \
        "KS conservative mode subtracts metals — the metal model must be OFF"
    assert leg.name == "KS", f"expected the KS leg, got {leg.name!r}"
    z = np.asarray(leg.z, float)
    assert z.min() >= 2.4 - 1e-6 and z.max() <= 4.6 + 1e-6, \
        f"KS leg z-range [{z.min()}, {z.max()}] outside the certified [2.4, 4.6]"
    k = np.asarray(leg.k, float)
    assert k.max() <= 0.065 + 1e-9, f"KS leg k_max {k.max()} exceeds the certified 0.065"
    assert k.max() > 0.045, (
        f"KS leg k_max {k.max()} <= 0.045: the NORC auto-cap window replaced the certified "
        f"echelle-float 0.065 window (prod ks_kwargs k_max not honoured) — deployment-"
        f"consistency violation")
    assert float(leg.dla_forward_frac) == 0.0, (
        "KS dla_forward_frac must be 0.0 (DLAs fully masked; ALSO certifies the K5 "
        "DLA-ride-along exact no-op claim)")


def arm_run_mode(arm_id):
    """'clean' for the K0 control (no inject_spec), 'boost' for every profiled arm."""
    return "clean" if AR.ARMS[arm_id]["quad"] is None else "boost"


def shard_pkl_name(arm_id, shard, *, smoke, r6_arm=None):
    """pkl naming (spec 5): ks_selboost_clean_shard_{i:03d}.pkl for K0, else
    ks_selboost_{arm_id}_shard_{i:03d}.pkl; SMOKE inserts .smoke before .pkl (F6a: the
    analyzer glob filters *.smoke.* so a smoke run can never pool as a real shard).
    R6 (2026-07-23): a DISTINCT prefix ks_r6_{legacy|mapped}_... so the campaign analyzer's
    ks_selboost_* glob never even sees an R6 pkl (defense in depth on top of the r6_override
    stamp refusal) — R6 pkls are consumed ONLY by analyze_r6_pairs.py."""
    suffix = ".smoke" if smoke else ""
    if r6_arm is not None:
        return f"ks_r6_{r6_arm}_shard_{shard:03d}{suffix}.pkl"
    tag = "clean" if arm_id == "K0_clean" else arm_id
    return f"ks_selboost_{tag}_shard_{shard:03d}{suffix}.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-id", required=True, choices=AR.arm_ids())
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=None,
                    help="default: the registry N for --arm-id (second-ask matrix)")
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--mode", choices=["clean", "boost"], default=None,
                    help="derived from --arm-id; an explicit value must AGREE (fail loud)")
    # PRIOR-STATE TRIPWIRE (spec 5c): REQUIRED — the caller must state the prior it designed
    # against; a mismatch with the deployed constants aborts before any compute.
    ap.add_argument("--expect-lls-boost", type=float, required=True)
    ap.add_argument("--expect-lls-frac-sigma", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    # R6 matched old-vs-new comparison (2026-07-23, disposition row 6): each pair = ONE mock
    # fit twice, once under the retired legacy alpha-space KS prior (--r6-arm legacy, the
    # ks_legacy_alpha_param build override) and once under the deployed mapped prior
    # (--r6-arm mapped). Pair identity comes from a SHARED truth source + shared fold_in(seed,m)
    # noise key: the mock data is a pure fn of (truth_pack, k_mock), so drawing the truth from
    # ONE stamped source in BOTH arms makes the pair's data vectors identical (run_legb truth_fn).
    ap.add_argument("--r6-arm", choices=["legacy", "mapped"], default=None,
                    help="R6 paired arm: which KS prior parameterization this run FITS under")
    ap.add_argument("--r6-truth-source", choices=["mapped-selfdraw", "mapped-fixed-draw"],
                    default=None,
                    help="R6 truth source (REQUIRED with --r6-arm, stamped + pool-keyed): "
                         "mapped-selfdraw = per-mock draws from the DEPLOYED mapped prior "
                         "(recommended default, PI checkpoint pending); mapped-fixed-draw = the "
                         "single mock-0 mapped draw shared by every pair (variance-free anchor)")
    a = ap.parse_args()
    if a.r6_arm is not None:
        assert a.arm_id == "K0_clean", \
            "--r6-arm is the CLEAN paired comparison; use --arm-id K0_clean (no injection)"
        assert a.r6_truth_source is not None, \
            "--r6-arm requires an EXPLICIT --r6-truth-source (stamped; never a silent default)"
    else:
        assert a.r6_truth_source is None, "--r6-truth-source is only meaningful with --r6-arm"
    if a.n_mocks is None:
        a.n_mocks = int(AR.ARMS[a.arm_id]["n_mocks"])
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)
    mode = arm_run_mode(a.arm_id)
    assert a.mode is None or a.mode == mode, (
        f"--mode {a.mode} contradicts arm {a.arm_id} (registry mode {mode}); K0 is clean-only "
        f"and boosted arms are boost-only (cross-pkl pairing)")

    # PRIOR-STATE TRIPWIRE first (cheap; no ctx build wasted on a stale prior).
    prior_stamp = assert_prior_state(a.expect_lls_boost, a.expect_lls_frac_sigma)

    # DEPLOYED KS forward (single source: prod_forward_config("KS") + NORC via build_arm_ctx
    # use_prod_forward; arm tag "lls_excess" only routes the UNUSED third return — the KS
    # metal_misspec tag would fail loud in arm_inject_spec, and this runner takes its
    # inject_spec from the REGISTRY, never from build_arm_ctx).
    # R6: the legacy arm builds with the stamped ks_legacy_alpha_param override (constants and
    # forward identical; ONLY the prior parameterization dispatch flips). Non-R6 = byte-identical
    # default build.
    ctx, d, _ = build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True,
                              ks_legacy_alpha_param=(a.r6_arm == "legacy"))
    if a.r6_arm is not None:
        assert bool(getattr(ctx, "ks_dndx_mapped", False)) is (a.r6_arm == "mapped"), \
            f"R6 {a.r6_arm} arm: ctx.ks_dndx_mapped={getattr(ctx, 'ks_dndx_mapped', None)}"
    assert len(ctx.legs) == 1, f"expected the single restricted KS leg, got {len(ctx.legs)}"
    L = ctx.legs[0]
    assert_ks_forward(ctx, L)
    # band assert on the BUILT prior center (belt-and-braces on top of the constants): the KS
    # LLS alpha-pivot must sit in the z=3-consistent band at boost 2.5 (assert_hcd_pivot_z3),
    # and the built width must realize the expected sigma/mu.
    alpha_mu0 = float(np.asarray(ctx.alpha_hcd_mu)[0])
    alpha_sd0 = float(np.asarray(ctx.alpha_hcd_sigma)[0])
    INF.assert_hcd_pivot_z3(alpha_mu0, z=INF.HCD_Z_PIVOT, where="run_ks_selboost_shard",
                            boost=prior_stamp["lls_survey_boost_ks"])
    assert abs(alpha_sd0 / alpha_mu0 - prior_stamp["lls_frac_sigma_ks"]) < 1e-9, \
        f"built KS LLS width sigma/mu {alpha_sd0/alpha_mu0} != {prior_stamp['lls_frac_sigma_ks']}"
    prior_stamp["alpha_lls_center_built"] = alpha_mu0
    prior_stamp["alpha_lls_sigma_built"] = alpha_sd0
    # MAPPED-ERA HONESTY (Stage-C 2026-07-22; analyzer made era-aware 2026-07-23): post-W2 the
    # KS ctx deploys the dN/dX-mapped parameterization; ctx.alpha_hcd_mu/sigma above are the
    # DORMANT LEGACY vectors (kept for the R6 override and audits). Stamp the parameterization +
    # the MAPPED pivot centre so a rerun's pkls are era-distinguishable. (The 2026-07-22 note
    # here claimed an analyzer "alpha_lls_center_built==0.4302804 provenance pin" would fail-loud
    # on these pkls — the CS design review found NO such pin exists; the analyzer now derives the
    # era from THIS value stamp and labels/normalizes accordingly.)
    prior_stamp["hcd_parameterization"] = ("dndx_mapped_v2"
                                           if getattr(ctx, "ks_dndx_mapped", False)
                                           else "alpha_pivot_powerlaw_v1")
    if getattr(ctx, "ks_dndx_mapped", False):
        from hcd_analysis.emulator.dndx_wc import w_c_corrected as _wcc
        import jax.numpy as _jnp
        prior_stamp["alpha_lls_center_built_semantics"] = "LEGACY-AUDIT-ONLY (dormant vector)"
        prior_stamp["alpha_lls_mapped_pivot_center"] = float(np.asarray(
            _wcc(_jnp.asarray(ctx.ks_dndx_ref_pivot),
                 _jnp.asarray(float(ctx.ks_xbar_pivot)),
                 _jnp.asarray(3.0)))[1])
        # MAPPED-GEOMETRY STAMPS for the era-aware analyzer (2026-07-23): the reference dN/dX
        # curves + Xbar(z) on the GLOBAL z grid and the mapped widths, so the analyzer can
        # compute the companion exact displacement D_exact and the truth-reachability check from
        # STAMPS (history-proof) without a cache load. Plain lists (the analyzer's stamp
        # homogeneity compares by ==). Module-attribute reads (rebinding trap).
        prior_stamp["ks_dndx_ref_z"] = np.asarray(ctx.ks_dndx_ref, float).tolist()
        prior_stamp["ks_xbar_z"] = np.asarray(ctx.ks_xbar_z, float).tolist()
        prior_stamp["ks_dndx_sigma_eps"] = float(INF.KS_DNDX_SIGMA_EPS)
        prior_stamp["ks_dndx_sigma_kappa"] = float(INF.KS_DNDX_SIGMA_KAPPA)

    # R6 STAMPS + SHARED TRUTH SOURCE (2026-07-23). r6_override lands on BOTH arms (Bayesian
    # design review Q4 hole 1: the mapped half would otherwise be stamp-indistinguishable from a
    # genuine ARM-P/campaign pkl and could pool). The stamps live in prior_constants, which
    # participates in every analyzer homogeneity/refusal check.
    truth_fn = None
    if a.r6_arm is not None:
        prior_stamp["r6_override"] = True
        prior_stamp["r6_arm"] = a.r6_arm
        prior_stamp["r6_truth_source"] = a.r6_truth_source
        import jax as _jax
        # the truth source is the MAPPED ctx for BOTH arms: the legacy arm needs an explicit
        # mapped build; the mapped arm reuses its own ctx (same deterministic construction).
        ctx_mapped = (ctx if a.r6_arm == "mapped"
                      else build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True)[0])
        assert bool(getattr(ctx_mapped, "ks_dndx_mapped", False)) is True
        if a.r6_truth_source == "mapped-selfdraw":
            # per-mock mapped-prior truths; run_legb passes k_truth = split(fold_in(seed,m))[0],
            # identical in both arms => identical truth => identical data vector per pair.
            truth_fn = (lambda k: CL.draw_leg_a_leg_truth(ctx_mapped, k))
        else:   # mapped-fixed-draw: ONE shared truth (mock-0's k_truth) for every pair
            _k_fix = _jax.random.split(
                _jax.random.fold_in(_jax.random.PRNGKey(int(a.seed)), 0), 3)[0]
            _fixed_truth = CL.draw_leg_a_leg_truth(ctx_mapped, _k_fix)
            truth_fn = (lambda k: _fixed_truth)
        print(f"[r6] arm={a.r6_arm} truth_source={a.r6_truth_source} "
              f"(fit parameterization {prior_stamp['hcd_parameterization']})")

    inject_spec = AR.arm_inject_spec(a.arm_id)
    if a.r6_arm is not None:
        assert inject_spec is None, "R6 arms are clean (K0_clean); inject_spec must be None"
    z_global = np.asarray(ctx.z_global, float)
    B_z = AR.arm_boost_B(a.arm_id, z_global)                  # {class: B(z_global)}
    B_pivot = {cls: AR.eval_profile(prof, AR.Z_PIVOT)
               for cls, prof in (inject_spec or {}).items()}
    print(f"[forward] DEPLOYED KS: res_corr_on={bool(ctx.res_corr_on)} "
          f"fix_alpha_res={bool(ctx.fix_alpha_res)} sample_res={bool(ctx.sample_res)} "
          f"f_res_amp_sigma={ctx.f_res_amp_sigma} metals={bool(L.metals_on)} "
          f"k_max={float(np.asarray(L.k).max()):.4f} dla_forward_frac={L.dla_forward_frac} "
          f"(== prod_forward_config('KS') + NORC) | prior: alpha_lls mu={alpha_mu0:.4f} "
          f"sd={alpha_sd0:.4f} boost={prior_stamp['lls_survey_boost_ks']} | arm={a.arm_id} "
          f"mode={mode} B(3)={ {c: round(v, 4) for c, v in B_pivot.items()} }")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    assert idxs, (f"shard {a.shard}/{a.n_shards} selects no mock at n_mocks={a.n_mocks} "
                  f"(smoke forces n_mocks=1: only shard 0 is a valid smoke cell)")
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[ks selboost {a.arm_id} shard {a.shard}] {mode.upper()} run ...")
    per_mock = run_legb(ctx, d, inject_spec=inject_spec, truth_fn=truth_fn, **run_kw)
    wall = time.time() - t0

    # EMULATOR-SPAN LOGGING (risk 12; readout caveat, not a hard failure): max boosted truth
    # alpha per class vs the training-cache structural w_c_cache column max.
    wc = np.asarray(d["w_c_cache"], float)                    # (R,4) [clean, LLS, subDLA, DLA]
    cache_max = np.nanmax(wc[:, 1:], axis=0)                  # (3,) per HCD class
    truth_max = np.nanmax(np.stack([np.asarray(r["truth_alpha_hcd_z"], float)
                                    for r in per_mock]), axis=(0, 1))   # (3,)
    oos_class = [c for j, c in enumerate(("lls", "subdla", "dla")) if truth_max[j] > cache_max[j]]
    span = dict(cache_alpha_max=cache_max.tolist(), truth_alpha_max=truth_max.tolist(),
                out_of_span_classes=oos_class)
    if oos_class:
        print(f"[span] WARNING: boosted truth alpha exceeds the training-cache grid for "
              f"{oos_class} (truth_max {truth_max} vs cache_max {cache_max}) — readout caveat")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, shard_pkl_name(a.arm_id, a.shard, smoke=a.smoke,
                                                 r6_arm=a.r6_arm))
    meta = dict(vars(a), mode=mode, paired=False, wall_s=wall,
                per_fit_wall_s=wall / max(len(per_mock), 1), run_kw=run_kw)
    # F5 convention: the closure_legb.forward_stamp SINGLE AUTHORITY (forward + prior-freeze
    # signatures + env data-selection stamps).
    meta["forward"] = CL.forward_stamp(ctx, L)
    meta["prior_constants"] = prior_stamp
    meta["arm_stamp"] = dict(
        arm_id=a.arm_id, profiles=inject_spec, B_pivot=B_pivot,
        B_z_global={cls: np.asarray(v, float).tolist() for cls, v in B_z.items()},
        z_global=z_global.tolist(), registry_signature=AR.registry_signature(),
        envelope=AR.ARMS[a.arm_id]["envelope"],
        k5_table=(dict(path=AR.K5_TABLE_PATH, sha256=AR.K5_TABLE_SHA256)
                  if AR.ARMS[a.arm_id]["quad"] == "measured" else None))
    meta["span"] = span
    pickle.dump(dict(arm=a.arm_id, survey="ks", mode=mode, idxs=idxs, per_mock=per_mock,
                     meta=meta), open(out, "wb"))
    nd = sum(int(r.get("n_div", 0) > 0) for r in per_mock)
    print(f"[ks selboost {a.arm_id} shard {a.shard}] wrote {len(per_mock)} {mode} mock(s) "
          f"({nd} div) -> {out} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
