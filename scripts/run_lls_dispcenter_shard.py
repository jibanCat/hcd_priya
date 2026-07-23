"""Displaced-CENTER LLS closure arm (closing-panel decision P2, PI commission 2026-07-19).

The LLS mirror of the DLA selfdraw campaign (run_dla_selfdraw_shard.py): paired clean-vs-boosted
mocks on the DESI leg, deployed NORC forward + REDUCED covariance, measuring with NUTS the
n_s/A_p bias induced when the TRUE LLS incidence sits off the prior center -- replacing the
linear ~0.96 sigma_post-per-prior-sigma rescale with a measurement (this project has twice seen
NUTS reverse linear intuition). Paired per mock: same seed => same truth draw AND the same
standard-normal noise vector g; only the LLS truth differs between arms, so realization noise
cancels in the paired Delta.

The truth boost rides the SAME run_legb inject_spec contract as the DLA arm
({"lls_truth_boost": boost} -> apply_lls_truth_boost, scaling pivot alpha_hcd[0] AND every
z-resolved alpha_hcd_z[:,0] row identically -- verified the exact analog of the DLA hook).

BOOST = 1.287 = +1 prior-sigma in FRACTIONAL terms at the deployed DESI LLS width sigma/mu =
0.287 (HCD_LLS_SURVEY_FRAC_SIGMA["DESI"], the 1x lit-anchored primary; asserted below, so a
hedge2x (0.574) or KS (0.40) ctx aborts rather than silently rescaling the displacement).
Latent-sigma displacement (the DLA campaign's +0.408 analog): alpha_lls's latent IS alpha
(TruncatedNormal(mu, 0.287*mu, low=0), sampled directly -- no softplus), so the displacement at
the prior center is (boost-1)*mu/sigma = 0.287/0.287 = +1.0000 exactly; the low=0 truncnorm
median correction at mu/sigma ~ 3.48 is < 1e-3 (both stamped into meta["prior"]). ONE side only
(+1 sigma) per the PI commission; a MINUS arm is a config flip: --boost 0.713 (= 1 - 0.287,
displacement -1 sigma).

Both arms fit under the final production likelihood (reduced DESI covariance,
DESI_DLA_COV_REDUCE -- asserted), N = 8 paired mocks (16 fits, ~270 CPU-h on cavestru1).

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

LLS_FRAC_SIGMA_DEPLOYED = 0.287     # deployed DESI LLS sigma/mu (inference.HCD_LLS_SURVEY_FRAC_SIGMA)
DEFAULT_BOOST = 1.0 + LLS_FRAC_SIGMA_DEPLOYED   # 1.287 = +1 prior-sigma fractional displacement


def latent_sigma_displacement(mu, sigma, boost, low=None):
    """Prior-sigma displacement of the boosted truth, evaluated at the prior median:
    (boost-1)*median/sigma. alpha_lls is sampled DIRECTLY as TruncatedNormal(mu, sigma, low=0)
    (sampler_numpyro/_legb_model) -- its latent IS alpha, so no softplus^-1 transform (the DLA
    campaign's +0.408 went through softplus^-1 at latent scale 1.0). ``low=None`` uses the
    untruncated Normal median (= mu, the prior-CENTER number); ``low=0.0`` uses the exact
    truncnorm median (a <1e-3 correction at mu/sigma ~ 3.48)."""
    mu, sigma, boost = float(mu), float(sigma), float(boost)
    if low is None:
        med = mu
    else:
        from scipy.stats import truncnorm
        med = float(truncnorm.median((float(low) - mu) / sigma, np.inf, loc=mu, scale=sigma))
    return (boost - 1.0) * med / sigma


def assert_deployed_lls_width(mu, sigma, expect=LLS_FRAC_SIGMA_DEPLOYED, tol=1e-6):
    """Fail LOUD unless sigma/mu is the deployed DESI LLS fractional width (0.287). A hedge2x
    ctx (0.574) or a KS ctx (0.40) would silently rescale the latent displacement the campaign
    quotes -- abort instead."""
    ratio = float(sigma) / float(mu)
    assert abs(ratio - expect) <= tol, (
        f"LLS prior width sigma/mu = {ratio:.6f} != deployed {expect} (hedge2x/KS/stale ctx?); "
        f"the {DEFAULT_BOOST}x boost is calibrated to the deployed width -- aborting")


def lls_prior_stamp(mu, sigma, boost):
    """The meta['prior'] stamp: deployed alpha_lls prior (mu, sigma, frac) + the exact
    latent-sigma displacement at the prior center AND at the truncnorm (low=0) median."""
    return dict(alpha_lls_mu=float(mu), alpha_lls_sigma=float(sigma),
                frac_sigma=float(sigma) / float(mu), boost=float(boost),
                latent_sigma_disp_center=latent_sigma_displacement(mu, sigma, boost),
                latent_sigma_disp_median=latent_sigma_displacement(mu, sigma, boost, low=0.0))


def forward_stamp(ctx, L):
    """The meta['forward'] stamp: the deployed-forward knob values + BOTH freeze-audit signature
    hashes -- closure_legb.forward_signature() (forward decision set) AND
    inference.hcd_prior_signature() (prior-constants payload; the width-study runner precedent),
    so a reader can reject pkls from a drifted forward OR a drifted prior without re-deriving
    either."""
    return dict(res_corr_on=bool(ctx.res_corr_on), fix_alpha_res=bool(ctx.fix_alpha_res),
                sample_res=bool(ctx.sample_res), f_res_amp_sigma=float(ctx.f_res_amp_sigma),
                metal_prior=str(ctx.metal_prior),
                metal_node_z=tuple(float(z) for z in ctx.metal_node_z),
                dla_cov_reduced=bool(L.dla_cov_reduced),
                dla_forward_frac=float(L.dla_forward_frac),
                forward_signature=CL.forward_signature(),
                hcd_prior_signature=INF.hcd_prior_signature())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--boost", type=float, default=DEFAULT_BOOST,
                    help="truth alpha_LLS multiplier (pivot + z-resolved column). 1.287 = +1 "
                         "prior-sigma in fractional terms at the deployed DESI width 0.287 "
                         "(latent displacement +1.0000 at the prior center; stamped into meta). "
                         "MINUS arm = 0.713 (a config flip, not new code).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)

    # DEPLOYED forward (single source: prod_forward_config + NORC via build_arm_ctx use_prod_forward),
    # mirroring run_dla_selfdraw_shard.py's parity asserts -- fail LOUD before any NUTS.
    ctx, d, _ = build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    fc = CL.prod_forward_config("DESI"); norc = CL.prod_norc_forward(); L = ctx.legs[0]
    assert bool(ctx.res_corr_on) == norc["res_corr_on"], "dispcenter forward != deployed NORC res_corr"
    assert bool(ctx.fix_alpha_res) == norc["fix_alpha_res"], "dispcenter fix_alpha_res != deployed NORC"
    assert bool(ctx.sample_res) == fc["sample_res"], "prod f_res float not wired into the dispcenter arm"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch vs deployed DESI"
    assert ctx.metal_prior == fc["metal_prior"], "prod metal model (flatlog2node) not wired"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    assert ctx.sample_metals is True and L.metals_on is True, "DESI leg must sample+apply metals"
    # BOTH arms fit under the final REDUCED covariance (never the stale fid cov).
    assert getattr(L, "dla_cov_reduced", False) is True, (
        "DESI leg C_data is NOT the reduced covariance (syst_e_dla_completeness still inside); "
        "the dispcenter closure must run on the final production likelihood -- check "
        "data_likelihood.DESI_DLA_COV_REDUCE")
    assert float(L.dla_forward_frac) == 1.0, "DESI alpha_DLA mean model must be live (frac 1.0)"
    # The LLS-axis guard: the deployed DESI pin width (0.287) is what calibrates boost -> +1 sigma.
    mu_lls, sig_lls = float(ctx.alpha_hcd_mu[0]), float(ctx.alpha_hcd_sigma[0])
    assert_deployed_lls_width(mu_lls, sig_lls)
    prior = lls_prior_stamp(mu_lls, sig_lls, a.boost)
    print(f"[forward] DEPLOYED DESI: res_corr_on={bool(ctx.res_corr_on)} fix_alpha_res={bool(ctx.fix_alpha_res)} "
          f"sample_res={bool(ctx.sample_res)} f_res_amp_sigma={ctx.f_res_amp_sigma} "
          f"metal_prior={ctx.metal_prior!r} dla_cov_reduced={L.dla_cov_reduced} "
          f"dla_forward_frac={L.dla_forward_frac} boost={a.boost} "
          f"(== prod_forward_config('DESI') + NORC + reduced cov)")
    print(f"[prior] alpha_lls: mu={mu_lls:.6f} sigma={sig_lls:.6f} frac={prior['frac_sigma']:.6f} "
          f"-> latent displacement {prior['latent_sigma_disp_center']:+.4f} sigma at the prior "
          f"center ({prior['latent_sigma_disp_median']:+.6f} at the truncnorm low=0 median)")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[lls dispcenter shard {a.shard}] CLEAN run (boost=1, reduced cov) ...")
    clean = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[lls dispcenter shard {a.shard}] BOOSTED run (lls_truth_boost={a.boost}) ...")
    boosted = run_legb(ctx, d, inject_spec={"lls_truth_boost": float(a.boost)}, **run_kw)
    wall = time.time() - t0
    np_ = max(min(len(clean), len(boosted)), 1)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"lls_dispcenter_desi_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a), paired=True, wall_s=wall, per_fit_wall_s=wall / (2 * np_))
    meta["prior"] = prior
    meta["forward"] = forward_stamp(ctx, L)
    pickle.dump(dict(arm="lls_dispcenter", survey="desi", idxs=idxs, clean_per_mock=clean,
                     boost_per_mock=boosted, meta=meta), open(out, "wb"))
    nd = sum(int(r.get("n_div", 0) > 0) for r in clean) + sum(int(r.get("n_div", 0) > 0) for r in boosted)
    print(f"[lls dispcenter shard {a.shard}] wrote {len(boosted)} paired mocks ({nd} div) -> {out} "
          f"wall={wall:.1f}s")


if __name__ == "__main__":
    main()
