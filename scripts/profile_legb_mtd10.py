"""LEAN mtd=10 profiler for ONE Leg-B closure mock with the NEW KS config (klow=0.0055).

Purpose (the PI asked for the FIRM per-mock cost): measure, at PRODUCTION max_tree_depth=10,
  * per-leapfrog wall, leapfrogs/sample, divergences, accept_prob, max-depth hits;
  * ESS / ESS-per-sample for (A_p, n_s, τ₀);
  * the MF-SMOKE-01 postprocess efficiency fix (fast transform-only vs legacy replay) saving;
then extrapolate samples→ESS≥400, per-mock CPU-h, and N=99 / N=600 totals.

This is NOT a coverage verdict (n=1) and NOT the gated production run. It SKIPS the figures
(the profile_mf_closure_smoke.py figures step JIT-compiles the MF forward ~6× — irrelevant to
the per-mock NUTS cost). Timing is forced SYNCHRONOUS with jax.block_until_ready so the JAX
async dispatch does not under-count the sampling wall.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/profile_legb_mtd10.py \
       --n-warmup 120 --n-samples 120 --max-tree-depth 10
"""
from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import NUTS, MCMC, init_to_median
from numpyro.infer.util import constrain_fn
from numpyro.diagnostics import effective_sample_size

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator.inference import PARAM_NAMES
from hcd_analysis.emulator.data import PARAM_LIMITS

REPO = "/home/mfho/hcd_priya"
OUTDIR = Path(f"{REPO}/figures/analysis/05_likelihood")
OUTDIR.mkdir(parents=True, exist_ok=True)
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])


def _phys(col, idx):
    lo, hi = PARAM_LIMITS[idx]
    return lo + col * (hi - lo)


def _run(ctx, mock_legs, core_per_leg, seed, *, nw, ns, mtd, fast):
    """One NUTS run; returns (samples_dict, extra_dict, run_wall, post_wall)."""
    kernel = NUTS(lambda: C._legb_model(ctx, mock_legs, core_per_leg),
                  dense_mass=True, target_accept_prob=0.9, max_tree_depth=int(mtd),
                  init_strategy=init_to_median)
    pp = None
    if fast:
        def pp(z):
            return constrain_fn(lambda: C._legb_priors_only(ctx), (), {}, z,
                                return_deterministic=False)
    mcmc = MCMC(kernel, num_warmup=int(nw), num_samples=int(ns), num_chains=1,
                progress_bar=False, postprocess_fn=pp)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)),
             extra_fields=("diverging", "num_steps", "accept_prob"))
    raw = mcmc.get_samples()
    jax.block_until_ready(raw)                 # force the deferred sampling computation
    run_wall = time.perf_counter() - t0
    # postprocess timing: materialize samples + (fast) host-side deterministic reconstruction.
    tp = time.perf_counter()
    samples = {k: np.asarray(v) for k, v in raw.items()}
    if fast:
        samples = {k: np.asarray(v)
                   for k, v in C._legb_reconstruct_deterministics(ctx, raw).items()}
    post_wall = time.perf_counter() - tp
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    return samples, extra, run_wall, post_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", type=int, default=0)
    ap.add_argument("--ab-legacy", action="store_true",
                    help="also run a SHORT legacy-postprocess A/B to measure the replay waste")
    ap.add_argument("--ab-warmup", type=int, default=40)
    ap.add_argument("--ab-samples", type=int, default=40)
    args = ap.parse_args()

    print("[mtd10] building Leg-B ctx with MF forward + C_emu floor (fold 0)")
    ctx, d = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    print("[mtd10] legs: " + ", ".join(
        f"{leg.name}(n_z={leg.n_z}, N={leg.k.shape[0]}, kmin={leg.k.min():.4f}, "
        f"kmax={leg.k.max():.4f}, floor_on={leg.mf_floor_on})" for leg in ctx.legs))

    sims, _ = C.held_out_sims(d, fold=0)
    sim = sims[args.mock % len(sims)]
    truth_sim = C.make_truth_from_sim(d, sim, fold=0, mf=ctx.mf)
    key0 = jax.random.PRNGKey(args.seed)
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, args.mock), 2)
    mock_legs, truth_pack, info = C.make_legb_mock(ctx, truth_sim, k_mock)
    core_per_leg = C._mock_core_per_leg(ctx, truth_sim)
    base_seed = int(jax.random.randint(k_nuts, (), 0, 2**31 - 1))
    nZg = ctx.z_global.size
    dense_dim = 9 + nZg + 3
    n_data = int(sum(np.isfinite(np.asarray(leg.P_data)).sum() for leg in mock_legs))
    print(f"[mtd10] latent dim (dense-mass) = θ9(9)+τ₀-ladder({nZg})+α(3) = {dense_dim}; "
          f"data rows fit = {n_data}")

    # ---- optional A/B: short legacy vs fast postprocess to quantify the replay waste -------
    ab_msg = ""
    if args.ab_legacy:
        print(f"[mtd10] A/B postprocess timing (short {args.ab_warmup}+{args.ab_samples}, "
              f"mtd={args.max_tree_depth})…")
        _, _, rw_f, pw_f = _run(ctx, mock_legs, core_per_leg, base_seed,
                                nw=args.ab_warmup, ns=args.ab_samples,
                                mtd=args.max_tree_depth, fast=True)
        _, _, rw_l, pw_l = _run(ctx, mock_legs, core_per_leg, base_seed,
                                nw=args.ab_warmup, ns=args.ab_samples,
                                mtd=args.max_tree_depth, fast=False)
        per_samp_fast = pw_f / max(args.ab_samples, 1)
        per_samp_leg = pw_l / max(args.ab_samples, 1)
        ab_msg = (f"  A/B postprocess ({args.ab_samples} samples): "
                  f"FAST={pw_f:.1f}s ({per_samp_fast*1000:.0f} ms/sample) vs "
                  f"LEGACY-replay={pw_l:.1f}s ({per_samp_leg*1000:.0f} ms/sample); "
                  f"saving {(per_samp_leg-per_samp_fast)*1000:.0f} ms/sample "
                  f"= {(per_samp_leg-per_samp_fast)*400:.0f} s @400 samples")
        print("[mtd10] " + ab_msg)

    # ---- the production-config mtd=10 measurement ------------------------------------------
    print(f"[mtd10] MAIN run: warmup={args.n_warmup} samples={args.n_samples} "
          f"dense_mass=True mtd={args.max_tree_depth} FAST-postprocess — RUNNING…")
    samples, extra, run_wall, post_wall = _run(
        ctx, mock_legs, core_per_leg, base_seed,
        nw=args.n_warmup, ns=args.n_samples, mtd=args.max_tree_depth, fast=True)
    print(f"[mtd10] run+materialize wall {run_wall:.1f}s; postprocess {post_wall:.1f}s")

    diverging = extra.get("diverging", np.zeros(0, bool))
    num_steps = extra.get("num_steps", np.zeros(0))      # leapfrogs/sample (SAMPLING phase)
    accept = extra.get("accept_prob", np.zeros(0))
    tree_depth = (np.ceil(np.log2(np.maximum(num_steps, 1) + 1)).astype(int)
                  if num_steps.size else np.zeros(0, int))
    n_div = int(diverging.sum())
    n_samp = int(args.n_samples)
    total_lf = float(num_steps.sum()) if num_steps.size else np.nan
    mean_lf = float(np.mean(num_steps)) if num_steps.size else np.nan
    # per-leapfrog wall: attribute the SAMPLING-phase fraction of the wall to the sampling
    # leapfrogs (extra_fields only records the sampling phase). warmup leapfrog count is not
    # exposed; assume warmup ≈ sampling per-step cost and split the wall by sample fraction.
    samp_frac = n_samp / (args.n_warmup + n_samp)
    per_lf_ms = 1000.0 * (run_wall * samp_frac) / max(total_lf, 1)

    theta = samples["theta_unit"]
    tau0 = samples["tau0_vec"]
    ns_chain = _phys(theta[:, NS_I], NS_I)
    ap_chain = _phys(theta[:, AP_I], AP_I)
    zg = np.asarray(ctx.z_global)
    kept_global = truth_pack["kept_global_z"]
    iz_rep = int(np.argmin(np.abs(zg[kept_global] - 3.0)))
    tau0_chain = tau0[:, kept_global][:, iz_rep]
    z_rep = float(zg[kept_global][iz_rep])

    def ess1(x):
        return float(effective_sample_size(x[None, :]))
    ess_ns, ess_ap, ess_t0 = ess1(ns_chain), ess1(ap_chain), ess1(tau0_chain)
    min_ess = min(ess_ns, ess_ap, ess_t0)

    # ---- extrapolation ----------------------------------------------------------------------
    # firm per-mock cost: at mtd=10 we MEASURED leapfrogs/sample directly (no tree_factor guess).
    # samples needed for ESS≥400 on the WORST param = 400 / (min_ess/n_samp).
    ess_per_sample = min_ess / n_samp
    prod_target = 400
    samples_needed = prod_target / max(ess_per_sample, 1e-9)
    # per-sample wall (sampling phase) = run_wall*samp_frac / n_samp; total per-mock includes a
    # FIXED warmup cost (run_wall*(1-samp_frac)) paid once.
    samp_wall = run_wall * samp_frac
    warmup_wall = run_wall - samp_wall
    per_sample_wall = samp_wall / n_samp
    per_mock_wall_s = warmup_wall + per_sample_wall * samples_needed + post_wall_extrap(post_wall, n_samp, samples_needed)
    per_mock_h = per_mock_wall_s / 3600.0
    n99_h = per_mock_h * 99
    n600_h = per_mock_h * 600

    lines = []
    A = lines.append
    A("# Leg-B mtd=10 PROFILE — ONE closure mock, NEW KS config (klow=0.0055).")
    A("# n=1 PROFILING run; NOT a coverage verdict. Production = dense-mass, mtd=10, SLURM.")
    A(f"# sim: {sim}")
    A("# legs: " + ", ".join(
        f"{leg.name}(n_z={leg.n_z},N={leg.k.shape[0]},kmin={leg.k.min():.4f},"
        f"kmax={leg.k.max():.4f},floor_on={leg.mf_floor_on})" for leg in ctx.legs))
    A(f"# total data rows fit = {n_data};  MF=ON floor=ON (KS leg only)")
    A("")
    A("## NUTS config")
    A(f"  n_warmup={args.n_warmup}  n_samples={args.n_samples}  num_chains=1")
    A(f"  dense_mass=True  max_tree_depth={args.max_tree_depth}  target_accept=0.9")
    A(f"  dense-mass DIMENSION = θ9(9)+τ₀-ladder({nZg})+α(3) = {dense_dim}")
    A("")
    A("## efficiency fix (MF-SMOKE-01)")
    A("  the production postprocess is FAST transform-only (constrain via priors-only model +")
    A("  host-side deterministic reconstruction; byte-identical to numpyro's default, rtol=0).")
    A(f"  postprocess wall this run: {post_wall:.1f}s for {n_samp} samples "
      f"({1000.0*post_wall/max(n_samp,1):.0f} ms/sample).")
    if ab_msg:
        A(ab_msg.strip())
    A("")
    A("## profiling @ mtd=10")
    A(f"  run+materialize wall (warmup+sampling, 1 chain, block_until_ready): {run_wall:.1f} s")
    A(f"    (split: warmup≈{warmup_wall:.1f}s, sampling≈{samp_wall:.1f}s by sample-fraction)")
    A(f"  divergences: {n_div} / {n_samp}  (rate {100.0*n_div/max(n_samp,1):.1f}%)")
    A(f"  mean accept_prob: {np.mean(accept) if accept.size else np.nan:.3f}")
    A(f"  leapfrogs/sample: mean {mean_lf:.1f}  (total sampling-phase {total_lf:.0f})")
    A(f"  max-depth hits: {int((tree_depth >= args.max_tree_depth).sum())} / {n_samp}  "
      f"(mean depth {np.mean(tree_depth) if tree_depth.size else np.nan:.2f}, "
      f"max {int(tree_depth.max()) if tree_depth.size else -1})")
    A(f"  per-leapfrog wall (sampling-phase est): {per_lf_ms:.1f} ms")
    A("")
    A("## ESS + ESS/sample (A_p, n_s, τ₀ at z≈%.1f)" % z_rep)
    A(f"  n_s : ESS={ess_ns:.1f}  ESS/sample={ess_ns/n_samp:.3f}")
    A(f"  A_p : ESS={ess_ap:.1f}  ESS/sample={ess_ap/n_samp:.3f}")
    A(f"  τ₀  : ESS={ess_t0:.1f}  ESS/sample={ess_t0/n_samp:.3f}  (z={z_rep:.2f})")
    A(f"  min-ESS/sample over the three = {ess_per_sample:.4f}")
    A("")
    A("## FIRM extrapolation to ESS≥%d/param (assumptions stated)" % prod_target)
    A(f"  ASSUMPTIONS: (1) target ESS≥{prod_target} on the WORST of (A_p,n_s,τ₀);")
    A(f"    (2) leapfrogs/sample MEASURED at mtd=10 (no tree_factor extrapolation);")
    A(f"    (3) 1 chain, CPU single-core → wall≈CPU-h; (4) warmup={args.n_warmup} paid ONCE/mock,")
    A(f"        held FIXED (does not scale with samples); (5) ESS/sample stationary post-warmup.")
    A(f"  samples needed for ESS≥{prod_target} = {samples_needed:.0f}")
    A(f"  per-sample sampling wall = {per_sample_wall:.2f} s; warmup (fixed) = {warmup_wall:.1f} s")
    A(f"  per-mock @ production ≈ {per_mock_h:.2f} CPU-h "
      f"({per_mock_wall_s/60.0:.0f} min wall, 1 core)")
    A(f"    → N=99  mocks ≈ {n99_h:.0f} CPU-h  ({n99_h/24:.1f} core-days)")
    A(f"    → N=600 mocks ≈ {n600_h:.0f} CPU-h  ({n600_h/24:.1f} core-days)")
    A("  NOTE: SLURM array, embarrassingly parallel over mocks (fold_in seeding).")
    A(f"  cavestru0 budget ~4000 CPU-h: N=99 {'WITHIN' if n99_h < 4000 else 'EXCEEDS'} budget; "
      f"N=600 {'WITHIN' if n600_h < 4000 else 'EXCEEDS'} budget.")

    txt = "\n".join(lines)
    print("\n" + txt)
    (OUTDIR / "legb_mtd10_profile.txt").write_text(txt + "\n")
    print(f"\n[mtd10] wrote {OUTDIR / 'legb_mtd10_profile.txt'}")
    np.savez(OUTDIR / "legb_mtd10_profile.npz",
             ns_chain=ns_chain, ap_chain=ap_chain, tau0_chain=tau0_chain,
             num_steps=num_steps, accept=accept, diverging=diverging,
             run_wall=run_wall, post_wall=post_wall, z_rep=z_rep)


def post_wall_extrap(post_wall, n_samp, samples_needed):
    """The fast postprocess scales ~linearly in #samples (host-side reconstruction of the
    deterministics + a one-shot constrain). Scale the measured cost to the needed samples."""
    return post_wall * (samples_needed / max(n_samp, 1))


if __name__ == "__main__":
    main()
