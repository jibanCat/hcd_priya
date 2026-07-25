"""Run ONE fit of the cross-leg r6x paired prior-sensitivity campaign (PI decisions #7,
execution annex 2026-07-24; design (2) of PROPOSAL-crossleg-R6.md, panel revisions applied).

One task = one (leg, arm, mock): the DEPLOYED per-leg ctx is built exactly as ARM-P builds it
(prod_forward_config(leg) + build_legb_ctx(survey=leg) + NORC pin + per-leg slice -- the
run_prod_sbc_shard.py --deployed-prior path, cloned not parameterized: the assert blocks are
load-bearing), the truth is drawn from THAT deployed ctx's own prior sites via the run_legb
truth_fn hook with the shared fold_in(seed=20260724, mock) key (identical in both arms ->
bit-identical pair data), and the fit runs under either:

  --arm deployed   the deployed prior (ctx_fit = the deployed ctx; a genuine self-consistent
                   reference, biases ~0 = the free coherence check), or
  --arm dispprior  the DISPLACED prior: LLS centre x1.287 realized by IN-PLACE DICT ITEM
                   ASSIGNMENT on inference.HCD_LLS_SURVEY_BOOST[leg] (and, for eBOSS/DESI,
                   HCD_LLS_SURVEY_FRAC_SIGMA[leg] = 0.287/1.287 -> fixed absolute width),
                   NEVER module-attribute rebinding (crossleg_r6_common.r6x_override; the
                   from-import-trap reasoning + per-leg displacement semantics live there),
                   with the fit ctx REBUILT after the override and the realized centre/width
                   asserted on the BUILT ctx against pre-registered float expectations, the
                   signature asserted MOVED to the pre-registered displaced hex, and the
                   dicts RESTORED + re-verified in a finally block.

TILT ARMS: DROPPED (design deliverable 2; crossleg_r6_common.R6X_TILT_VERDICT + the build
memo record the verification: no freeze-safe signature-carried z-slope-centre knob exists --
the centre is an immutable from-imported float, so only the forbidden rebind mechanism could
move it, and it moves signature and ctx INCOHERENTLY). No --arm tilt exists here on purpose.

MOCK-ONLY: this driver has no real-data path (run_legb leg_a self-draw only) and refuses the
env data-selection flags at entry (assert_env_data_flags_unset, no override arg exists).

Pkl: r6x_<leg>_<arm>_shard_<mock:03d>[.smoke].pkl -- a DISTINCT prefix no other analyzer
globs; r6x_override/arm/leg/seed/truth-source stamped into prior_constants so every existing
analyzer ALSO refuses on stamps. Production NUTS: 250 warmup + 300 samples, dense mass,
max_tree_depth 10, 4 CPU. SMOKE=30/40 steps, .smoke pkl suffix.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import contextlib
import functools
import os
import pickle
import time

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp

import hcd_analysis.emulator.closure_legb as CL
import hcd_analysis.emulator.inference as INF
from hcd_analysis.emulator import data_likelihood as DLF
from hcd_analysis.emulator.closure_legb import (build_legb_ctx, run_legb, prod_forward_config,
                                                prod_norc_forward)
from hcd_analysis.emulator.prod_ensemble import production_member_paths
import scripts.crossleg_r6_common as CC

REPO = "/home/mfho/hcd_priya"


def build_deployed_style_ctx(leg):
    """The per-leg production ctx build, cloned from run_prod_sbc_shard.py --deployed-prior
    (ensemble forward + prod data-nuisance knobs + NORC + survey=leg + per-leg slice + the
    ARM-P fail-loud block). Reads the LIVE prior dicts at build time, so the SAME function
    builds the deployed ctx (no override active) and the displaced ctx (override active)."""
    members = production_member_paths(checkpoints_dir=f"{REPO}/checkpoints")
    fc = prod_forward_config(leg)
    norc = prod_norc_forward()
    ctx, d = build_legb_ctx(
        use_xclass=True, with_mf=True, mf_with_floor=True,
        res_corr_on=norc["res_corr_on"],
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=(leg == "eBOSS"), metals_on=(leg == "DESI"),
        sample_metals=(leg in ("DESI", "eBOSS")),
        sample_res=fc["sample_res"], f_res_amp_sigma=fc["f_res_amp_sigma"],
        metal_prior=fc["metal_prior"],
        ks_kwargs=(fc["ks_kwargs"] if leg == "KS" else None),
        survey=leg, hierarchical_hcd=False, ensemble_ckpts=members)
    if not norc["res_corr_on"]:
        ctx = ctx._replace(fix_alpha_res=True)
    assert ctx.res_corr_on == norc["res_corr_on"], "res_corr_on did not propagate to the ctx"
    assert ctx.fix_alpha_res == norc["fix_alpha_res"], "fix_alpha_res inconsistent with NORC"
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name.upper().startswith(leg.upper())])
    assert len(ctx.legs) == 1 and ctx.legs[0].name == leg, \
        f"per-leg slice failed: legs={[l.name for l in ctx.legs]} != {leg}"
    # ARM-P-style parity: parameterization + forward wiring (fail LOUD before any NUTS).
    want_param = INF.HCD_ALPHA_PARAMETERIZATION[leg]
    got_param = ("dndx_mapped_v2" if getattr(ctx, "ks_dndx_mapped", False)
                 else "alpha_pivot_powerlaw_v1")
    assert got_param == want_param, f"parameterization {got_param} != deployed {want_param}"
    assert ctx.metal_prior == fc["metal_prior"], "metal model not wired"
    assert bool(ctx.sample_res) == bool(fc["sample_res"]), "f_res float not wired"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch"
    for _leg in ctx.legs:
        if _leg.name == "DESI":
            assert bool(_leg.dla_cov_reduced) == bool(DLF.DESI_DLA_COV_REDUCE), \
                "DESI leg dla_cov_reduced disagrees with the DESI_DLA_COV_REDUCE authority"
    if leg == "KS":
        assert ctx.ks_dndx_mapped is True, "KS ctx not on the mapped branch"
        k = np.asarray(ctx.legs[0].k, float)
        assert 0.045 < k.max() <= 0.065 + 1e-9, \
            f"KS k_max {k.max()} outside the certified echelle-float window"
    return ctx, d, got_param


def ks_mapped_pivot_centre(ctx):
    """The mapped LLS pivot centre recomputed from the BUILT ctx (the ARM-P belt-and-braces
    recompute, run_prod_sbc_shard pattern)."""
    from hcd_analysis.emulator.dndx_wc import w_c_corrected
    return float(np.asarray(w_c_corrected(
        jnp.asarray(ctx.ks_dndx_ref_pivot), jnp.asarray(float(ctx.ks_xbar_pivot)),
        jnp.asarray(3.0)))[1])


def assert_built_arm_prior(ctx, leg, arm):
    """The r6x ctx-level tripwire (--expect pattern, panel revision 4): the BUILT ctx must
    realize the intended arm's centre/width at FLOAT EQUALITY against the pre-registered
    expectations, and the live signature must be the arm's pre-registered hex. Returns the
    realized-prior stamp dict."""
    mu_e, sd_e = CC.expected_lls_centre_width(leg, arm)
    mu_b = float(np.asarray(ctx.alpha_hcd_mu)[0])
    sd_b = float(np.asarray(ctx.alpha_hcd_sigma)[0])
    assert mu_b == mu_e, (
        f"built LLS alpha centre {mu_b!r} != pre-registered {arm} expectation {mu_e!r} "
        f"({leg}) -- the built ctx does NOT realize the declared arm; refusing")
    assert sd_b == sd_e, (
        f"built LLS alpha width {sd_b!r} != pre-registered {arm} expectation {sd_e!r} "
        f"({leg}) -- refusing")
    sig = INF.hcd_prior_signature()
    want_hex = CC.expected_hex(leg, arm)
    assert sig == want_hex, (
        f"live hcd_prior_signature {sig[:12]}... != pre-registered {arm} hex "
        f"{want_hex[:12]}... ({leg}) -- refusing")
    if arm == "dispprior":
        assert sig != CC.R6X_DEPLOYED_HEX, "displaced arm but the signature did not MOVE"
    stamp = dict(alpha_lls_center_built=mu_b, alpha_lls_sigma_built=sd_b,
                 lls_survey_boost_effective=float(INF.HCD_LLS_SURVEY_BOOST[leg]),
                 lls_frac_sigma_effective=float(INF.HCD_LLS_SURVEY_FRAC_SIGMA[leg]),
                 hcd_prior_signature=sig)
    if leg == "KS":
        stamp["alpha_lls_center_built_semantics"] = "LEGACY-AUDIT-ONLY (dormant vector)"
        piv = ks_mapped_pivot_centre(ctx)
        piv_e = CC.R6X_KS_MAPPED_PIVOT[arm]
        assert piv == piv_e, (
            f"KS mapped LLS pivot centre {piv!r} != pre-registered {arm} value {piv_e!r} -- "
            f"the mapped reference does NOT realize the declared arm; refusing")
        INF.assert_ks_mapped_pivot(piv, f"run_crossleg_r6_shard {arm}")  # frozen band guard
        # mapped widths are the PINNED literals in BOTH arms (fixed absolute width).
        assert float(INF.KS_DNDX_SIGMA_EPS) == 0.5310, "KS_DNDX_SIGMA_EPS drifted"
        stamp["alpha_lls_mapped_pivot_center"] = piv
        stamp["ks_dndx_sigma_eps"] = float(INF.KS_DNDX_SIGMA_EPS)
        stamp["ks_eps_sigma_displacement"] = (CC.R6X_KS_EPS_SIGMA_DISP
                                              if arm == "dispprior" else 0.0)
    return stamp


def regen_truth_and_data(ctx_fit, ctx_truth, d, mock, seed):
    """Regenerate the mock's truth pack + data vector OUTSIDE run_legb, replicating its key
    construction exactly (k_truth,k_mock = split(fold_in(seed,m),3)[:2]; truth from the
    DEPLOYED ctx's prior sites; data via make_leg_a_legmock on the FIT ctx). Returns
    (truth_pack, truth_sha, data_sha) -- the pair bit-identity certificate: both arms stamp
    these and the analyzer refuses a pair whose hashes differ."""
    key0 = jax.random.PRNGKey(int(seed))
    k_truth, k_mock, _ = jax.random.split(jax.random.fold_in(key0, int(mock)), 3)
    tp = CL.draw_leg_a_leg_truth(ctx_truth, k_truth)
    fid_core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
                for name, v in CL._fiducial_dla_core_per_leg(
                    d, ctx_fit.legs, ctx_fit.cache_k).items()}
    mock_legs, _info = CL.make_leg_a_legmock(ctx_fit, fid_core, tp, k_mock)
    data_vec = np.concatenate([np.asarray(l.P_data, float) for l in mock_legs])
    truth_sha = CC.sha256_of_arrays(tp["theta9"], tp["tau0_global"], tp["alpha_hcd"],
                                    tp["alpha_hcd_z"])
    data_sha = CC.sha256_of_arrays(data_vec)
    return tp, truth_sha, data_sha


def assert_rec_matches_truth(rec, tp):
    """The recorded fit's truth fields must be BIT-IDENTICAL to the regenerated truth pack
    (assert-by-regeneration; a mismatch means the truth source or keying drifted)."""
    tv = np.asarray(rec["truth_vec"])
    assert np.array_equal(tv[:9], np.asarray(tp["theta9"])), "truth theta9 != regenerated"
    n_t = len(np.asarray(tp["tau0_global"]))
    assert np.array_equal(tv[9:9 + n_t], np.asarray(tp["tau0_global"])), \
        "truth tau0 ladder != regenerated"
    assert np.array_equal(tv[9 + n_t:9 + n_t + 3], np.asarray(tp["alpha_hcd"])), \
        "truth alpha pivot != regenerated"
    assert np.array_equal(np.asarray(rec["truth_alpha_hcd_z"]),
                          np.asarray(tp["alpha_hcd_z"])), "truth alpha_hcd_z != regenerated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=list(CC.R6X_LEGS))
    ap.add_argument("--arm", required=True, choices=list(CC.R6X_ARMS),
                    help="deployed | dispprior. NO tilt arm exists: dropped per the execution "
                         "annex (no freeze-safe z-slope-centre knob; crossleg_r6_common).")
    ap.add_argument("--mock", type=int, required=True, help="mock index (0..n_pairs-1)")
    ap.add_argument("--seed", type=int, default=CC.R6X_SEED)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    # PRIOR-STATE TRIPWIRE (--expect pattern): the caller must state the DEPLOYED per-leg
    # constants the campaign was designed against; a drift aborts before any compute.
    ap.add_argument("--expect-lls-boost", type=float, required=True)
    ap.add_argument("--expect-lls-frac-sigma", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="30/40 NUTS steps + .smoke pkl suffix")
    a = ap.parse_args()
    assert int(a.seed) == CC.R6X_SEED, \
        f"--seed {a.seed} != the pre-registered fresh seed {CC.R6X_SEED} (panel revision 6)"
    if a.smoke:
        a.n_warmup, a.n_samples = min(a.n_warmup, 30), min(a.n_samples, 40)

    # MOCK-ONLY tripwire: refuse stray env data-selection flags at entry; no override exists.
    DLF.assert_env_data_flags_unset("run_crossleg_r6_shard", allow=False)
    # PRIOR-MUTATING env arms would compose incoherently with the r6x override -- refuse.
    assert float(os.environ.get("SBC_SUBDLA_AMP_SIGMA", "0")) == 0.0, \
        "run_crossleg_r6_shard refuses SBC_SUBDLA_AMP_SIGMA (prior-mutating env arm)"
    assert float(os.environ.get("SBC_TAU0_PRIOR_SIGMA", "0")) == 0.0, \
        "run_crossleg_r6_shard refuses SBC_TAU0_PRIOR_SIGMA (prior-mutating env arm)"

    # --expect args vs the pre-registered deployed state AND the live dicts.
    assert float(a.expect_lls_boost) == CC.R6X_DEPLOYED_BOOST[a.leg], (
        f"--expect-lls-boost {a.expect_lls_boost} != pre-registered deployed "
        f"{CC.R6X_DEPLOYED_BOOST[a.leg]} for {a.leg}")
    assert float(a.expect_lls_frac_sigma) == CC.R6X_DEPLOYED_FRAC[a.leg], (
        f"--expect-lls-frac-sigma {a.expect_lls_frac_sigma} != pre-registered deployed "
        f"{CC.R6X_DEPLOYED_FRAC[a.leg]} for {a.leg}")
    CC.verify_deployed_prior_state(a.leg)

    # 1) DEPLOYED ctx FIRST (build-order rule, panel revision 4): the truth source for BOTH
    #    arms, built before any override can touch the dicts.
    print(f"[r6x] leg={a.leg} arm={a.arm} mock={a.mock} seed={a.seed} "
          f"smoke={a.smoke} (truth source: {CC.R6X_TRUTH_SOURCE})")
    t_build0 = time.time()
    ctx_dep, d, param = build_deployed_style_ctx(a.leg)
    print(f"[r6x] deployed ctx built ({time.time() - t_build0:.1f}s) "
          f"parameterization={param}")
    truth_fn = (lambda k: CL.draw_leg_a_leg_truth(ctx_dep, k))

    override = (CC.r6x_override(a.leg) if a.arm == "dispprior"
                else contextlib.nullcontext())
    with override:
        # 2) FIT ctx: rebuild AFTER the override (dispprior) / reuse the deployed build.
        if a.arm == "dispprior":
            ctx_fit, _d2, param2 = build_deployed_style_ctx(a.leg)
            assert param2 == param, "parameterization changed under the override (must not)"
        else:
            ctx_fit = ctx_dep
        prior_stamp = assert_built_arm_prior(ctx_fit, a.leg, a.arm)
        prior_stamp.update(
            r6x_override=True, r6x_arm=a.arm, r6x_leg=a.leg, r6x_seed=int(a.seed),
            r6x_truth_source=CC.R6X_TRUTH_SOURCE, r6x_factor=CC.R6X_FACTOR,
            r6x_disp_spec=(CC.R6X_DISP_SPEC[a.leg] if a.arm == "dispprior" else "none"),
            r6x_expected_hex_pair=[CC.R6X_DEPLOYED_HEX, CC.R6X_DISPLACED_HEX[a.leg]],
            hcd_parameterization=param, r6x_tilt_verdict=CC.R6X_TILT_VERDICT)
        print(f"[r6x] BUILT-ctx tripwire OK: centre {prior_stamp['alpha_lls_center_built']:.6f} "
              f"width {prior_stamp['alpha_lls_sigma_built']:.6f} "
              f"sig {prior_stamp['hcd_prior_signature'][:12]}...")

        # 3) pair bit-identity certificate: regenerate truth + data, hash, and (post-fit)
        #    assert the recorded truth is bit-identical to the regeneration.
        tp, truth_sha, data_sha = regen_truth_and_data(ctx_fit, ctx_dep, d, a.mock, a.seed)
        print(f"[r6x] mock {a.mock}: truth sha {truth_sha[:12]}... data sha {data_sha[:12]}...")

        # 4) the fit (production NUTS settings; run_legb redraws the SAME truth/mock keys).
        run_kw = dict(n_mocks=max(CC.R6X_N_PAIRS, a.mock + 1), mock_indices=[int(a.mock)],
                      return_per_mock=True, leg_a=True, n_warmup=a.n_warmup,
                      n_samples=a.n_samples, seed=int(a.seed), dense_mass=True,
                      max_tree_depth=a.max_tree_depth, verbose=True)
        t0 = time.time()
        per_mock = run_legb(ctx_fit, d, truth_fn=truth_fn, inject_spec=None, **run_kw)
        wall = time.time() - t0
        assert len(per_mock) == 1, f"expected 1 record, got {len(per_mock)}"
        rec = per_mock[0]
        assert_rec_matches_truth(rec, tp)
        rec["r6x_truth_sha256"] = truth_sha
        rec["r6x_data_sha256"] = data_sha

        # 5) stamps + atomic write, WHILE the override is still active (forward_stamp embeds
        #    the live hcd_prior_signature -- it must be the ARM's hex, not the restored one).
        L = ctx_fit.legs[0]
        meta = dict(vars(a), wall_s=wall, per_fit_wall_s=wall, run_kw=run_kw)
        meta["forward"] = CL.forward_stamp(ctx_fit, L)
        assert meta["forward"]["hcd_prior_signature"] == CC.expected_hex(a.leg, a.arm), \
            "forward_stamp signature != the arm's pre-registered hex at write time"
        meta["prior_constants"] = prior_stamp
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, CC.pkl_name(a.leg, a.arm, a.mock, smoke=a.smoke))
        tmp = out + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(dict(arm=a.arm, leg=a.leg, survey=a.leg, idxs=[int(a.mock)],
                             per_mock=per_mock, meta=meta), f)
        os.replace(tmp, out)
        print(f"[r6x {a.leg} {a.arm} mock {a.mock}] wrote (n_div={rec.get('n_div', 0)}) "
              f"-> {out} wall={wall:.1f}s")
    # r6x_override's finally has now restored + re-verified; re-verify once more from here.
    CC.verify_deployed_prior_state(a.leg)
    print(f"[r6x] prior dicts restored + verified (deployed hex "
          f"{CC.R6X_DEPLOYED_HEX[:12]}...)")


if __name__ == "__main__":
    main()
