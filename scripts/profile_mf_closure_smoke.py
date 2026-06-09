"""T5b SMOKE — profile ONE Leg-B closure mock THROUGH the MF forward + C_emu floor.

NOT a coverage verdict (n=1) and NOT production scale. Runs one held-out-sim mock through
the certified through-MF forward (LF P_filt × exp(g + log res_corr)) with the LF→HR + n_s-edge
C_emu floor on the small-scale (KS) leg, at SMOKE NUTS size, and reports:
  wall-clock, per-leapfrog-step time, ESS + ESS/sample for (A_p, n_s, τ₀), divergence
  count/rate, max_tree_depth hits, the dense-mass dimension, and the recovered
  (A_p, n_s, τ₀) posterior mean±sd vs the mock truth.
Then EXTRAPOLATES the N=99 / N=600 production CPU-h with stated assumptions.

Emits to figures/analysis/05_likelihood/:
  mf_closure_smoke_posterior.png, mf_closure_cemu_on_leg.png, mf_closure_whitened_resid.png,
  mf_closure_smoke.txt

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/profile_mf_closure_smoke.py \
       --n-warmup 150 --n-samples 150 --max-tree-depth 8
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

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.inference import PARAM_NAMES
from hcd_analysis.emulator.data import PARAM_LIMITS

REPO = "/home/mfho/hcd_priya"
FIGDIR = Path(f"{REPO}/figures/analysis/05_likelihood")
FIGDIR.mkdir(parents=True, exist_ok=True)
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])


def _ns_phys(theta_unit_col):
    lo, hi = PARAM_LIMITS[NS_I]
    return lo + theta_unit_col * (hi - lo)


def _ap_phys(theta_unit_col):
    lo, hi = PARAM_LIMITS[AP_I]
    return lo + theta_unit_col * (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=150)
    ap.add_argument("--n-samples", type=int, default=150)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", type=int, default=0, help="which held-out sim index")
    ap.add_argument("--dense-mass", action="store_true", default=True)
    ap.add_argument("--diag-mass", action="store_true")
    args = ap.parse_args()
    dense_mass = (not args.diag_mass)

    print("[mf-smoke] building Leg-B ctx with MF forward + C_emu floor (fold 0)")
    ctx, d = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    print(f"[mf-smoke] legs: " + ", ".join(
        f"{leg.name}(n_z={leg.n_z}, N={leg.k.shape[0]}, floor_on={leg.mf_floor_on})"
        for leg in ctx.legs))
    print(f"[mf-smoke] MF={'ON' if ctx.mf is not None else 'OFF'} "
          f"floor={'ON' if ctx.mf_floor is not None else 'OFF'}")

    sims, _ = C.held_out_sims(d, fold=0)
    sim = sims[args.mock % len(sims)]
    print(f"[mf-smoke] mock sim: {sim[:40]}…")

    # ONE mock: MF-resolution truth (gate invariant) + cosmic-only noise.
    key0 = jax.random.PRNGKey(args.seed)
    truth_sim = C.make_truth_from_sim(d, sim, fold=0, mf=ctx.mf)
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, args.mock), 2)
    mock_legs, truth_pack, info = C.make_legb_mock(ctx, truth_sim, k_mock)
    core_per_leg = C._mock_core_per_leg(ctx, truth_sim)
    kept_global = truth_pack["kept_global_z"]
    print(f"[mf-smoke] kept global τ₀ z-bins: {int(kept_global.sum())} / {kept_global.size}; "
          f"dropped: {info['dropped']}")

    # the dense-mass dimension = #latent sites the sampler adapts a dense mass over:
    #   theta_unit(9) + alpha_ladder(nZ_global) + alpha_lls(1)+alpha_subdla(1)+alpha_dla_raw(1)
    nZg = ctx.z_global.size
    dense_dim = 9 + nZg + 3
    print(f"[mf-smoke] latent dim (dense-mass): θ9 + τ₀-ladder({nZg}) + α(3) = {dense_dim}")

    base_seed = int(jax.random.randint(k_nuts, (), 0, 2**31 - 1))
    kernel = NUTS(lambda: C._legb_model(ctx, mock_legs, core_per_leg),
                  dense_mass=bool(dense_mass), target_accept_prob=0.9,
                  max_tree_depth=int(args.max_tree_depth), init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=int(args.n_warmup), num_samples=int(args.n_samples),
                num_chains=1, progress_bar=False)

    print(f"[mf-smoke] NUTS warmup={args.n_warmup} samples={args.n_samples} "
          f"dense_mass={dense_mass} max_tree_depth={args.max_tree_depth} — RUNNING…")
    t0 = time.perf_counter()
    # numpyro HMCState exposes "diverging", "num_steps" (leapfrog count/sample), "accept_prob",
    # "mean_accept_prob". tree DEPTH is NOT a field → derive it from num_steps: a NUTS tree of
    # depth t does (2^t − 1) leapfrog steps, so depth ≈ ceil(log2(num_steps + 1)); a max-depth
    # hit is num_steps ≥ 2^max_tree_depth − 1.
    mcmc.run(jax.random.PRNGKey(base_seed),
             extra_fields=("diverging", "num_steps", "accept_prob"))
    wall = time.perf_counter() - t0
    print(f"[mf-smoke] DONE in {wall:.1f}s wall")

    print("[mf-smoke] extracting samples…")
    te = time.perf_counter()
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print(f"[mf-smoke]   get_samples in {time.perf_counter()-te:.1f}s")
    te = time.perf_counter()
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    print(f"[mf-smoke]   get_extra_fields in {time.perf_counter()-te:.1f}s; computing ESS…")
    diverging = extra.get("diverging", np.zeros(0, bool))
    num_steps = extra.get("num_steps", np.zeros(0))                 # leapfrogs / sample
    accept = extra.get("accept_prob", np.zeros(0))
    # derived tree depth per sample (a NUTS depth-t tree does 2^t − 1 leapfrogs).
    tree_depth = (np.ceil(np.log2(np.maximum(num_steps, 1) + 1)).astype(int)
                  if num_steps.size else np.zeros(0, int))

    n_div = int(diverging.sum())
    n_samp = int(args.n_samples)
    # per-leapfrog-step time: total leapfrogs ≈ Σ num_steps over the SAMPLING phase only
    # (warmup leapfrogs are not in extra_fields); approximate the sampling-phase per-step time.
    total_leapfrog_sampling = float(num_steps.sum()) if num_steps.size else np.nan
    # ESS for (A_p, n_s, τ₀) — use numpyro's effective_sample_size on the chain.
    from numpyro.diagnostics import effective_sample_size
    theta = samples["theta_unit"]                     # (L, 9)  already numpy
    tau0 = samples["tau0_vec"]                         # (L, nZg) already numpy
    # physical-units chains for the reported params
    ns_chain = _ns_phys(theta[:, NS_I])
    ap_chain = _ap_phys(theta[:, AP_I])
    # representative τ₀ z-bin: the kept z closest to z=3 (the data anchor band)
    zg = np.asarray(ctx.z_global)
    iz_rep = int(np.argmin(np.abs(zg[kept_global] - 3.0)))
    tau0_kept = tau0[:, kept_global]
    tau0_chain = tau0_kept[:, iz_rep]
    z_rep = float(zg[kept_global][iz_rep])

    def ess1(x):
        return float(effective_sample_size(x[None, :]))   # (1 chain, L)

    ess_ns, ess_ap, ess_t0 = ess1(ns_chain), ess1(ap_chain), ess1(tau0_chain)

    # recovered posterior mean±sd vs truth
    ns_truth = float(_ns_phys(truth_pack["theta9"][NS_I]))
    ap_truth = float(_ap_phys(truth_pack["theta9"][AP_I]))
    tau0_truth = float(truth_pack["tau0_global"][kept_global][iz_rep])

    # ---- profiling extrapolation ---------------------------------------------
    # per-effective-sample wall at smoke; production target ESS≈400 (Stan rule-of-thumb).
    min_ess = min(ess_ns, ess_ap, ess_t0)
    wall_per_ess = wall / max(min_ess, 1e-9)
    prod_ess_target = 400
    # production NUTS is dense-mass, max_tree_depth=10 (deeper trees → more leapfrogs/sample).
    # smoke used max_tree_depth=args.max_tree_depth; production = 10 → assume the per-sample
    # leapfrog count scales ≈ 2^(10-d_smoke) in the WORST case but typically ~1.5-2× (the
    # mass matrix conditions the geometry post-warmup). We report a per-mock production
    # estimate = wall_per_ess × prod_ess_target × tree_factor, tree_factor stated below.
    tree_factor = 2.0 ** max(0, (10 - args.max_tree_depth)) ** 0.5    # conservative sub-exp
    per_mock_prod_h = (wall_per_ess * prod_ess_target * tree_factor) / 3600.0
    n99_h = per_mock_prod_h * 99
    n600_h = per_mock_prod_h * 600

    # ---- text report ----------------------------------------------------------
    lines = []
    A = lines.append
    A("# T5b SMOKE — ONE Leg-B closure mock THROUGH the MF forward + C_emu floor.")
    A("# n=1 PROFILING run; NOT a coverage verdict. Production NUTS is dense-mass, mtd=10,")
    A("# SLURM-only (~hours/mock). This smoke caps mtd to bound leapfrogs/sample.")
    A(f"# sim: {sim}")
    A(f"# legs: " + ", ".join(f"{leg.name}(n_z={leg.n_z},N={leg.k.shape[0]},floor_on={leg.mf_floor_on})"
                              for leg in ctx.legs))
    A(f"# MF=ON  floor=ON (LF→HR generalization + n_s-edge, KS leg only)")
    A("")
    A("## NUTS config")
    A(f"  n_warmup={args.n_warmup}  n_samples={args.n_samples}  num_chains=1")
    A(f"  dense_mass={dense_mass}  max_tree_depth={args.max_tree_depth}  target_accept=0.9")
    A(f"  dense-mass DIMENSION = θ9(9) + τ₀-ladder({nZg}) + α(3) = {dense_dim}")
    A("")
    A("## profiling")
    A(f"  wall-clock (sampling+warmup, 1 chain): {wall:.1f} s")
    A(f"  divergences: {n_div} / {n_samp}  (rate {100.0*n_div/max(n_samp,1):.1f}%)")
    A(f"  max_tree_depth hits: {int((tree_depth >= args.max_tree_depth).sum())} / {n_samp}  "
      f"(mean depth {np.mean(tree_depth) if tree_depth.size else np.nan:.2f}, "
      f"max {int(tree_depth.max()) if tree_depth.size else -1})")
    A(f"  mean accept_prob: {np.mean(accept) if accept.size else np.nan:.3f}")
    A(f"  total leapfrog steps (sampling phase): {total_leapfrog_sampling:.0f}  "
      f"(mean {np.mean(num_steps) if num_steps.size else np.nan:.1f} leapfrogs/sample)")
    if num_steps.size:
        # per-leapfrog-step time, attributing the SAMPLING wall fraction (warmup≈samples here)
        samp_frac = n_samp / (args.n_warmup + n_samp)
        per_step_ms = 1000.0 * (wall * samp_frac) / max(total_leapfrog_sampling, 1)
        A(f"  per-leapfrog-step time (sampling-phase est): {per_step_ms:.1f} ms")
    A("")
    A("## ESS + ESS/sample (A_p, n_s, τ₀ at z≈%.1f)" % z_rep)
    A(f"  n_s : ESS={ess_ns:.1f}   ESS/sample={ess_ns/n_samp:.3f}")
    A(f"  A_p : ESS={ess_ap:.1f}   ESS/sample={ess_ap/n_samp:.3f}")
    A(f"  τ₀  : ESS={ess_t0:.1f}   ESS/sample={ess_t0/n_samp:.3f}  (z={z_rep:.2f})")
    A("")
    A("## recovered posterior mean±sd vs mock truth (n=1; NOT a coverage verdict)")
    A(f"  n_s : {ns_chain.mean():.4f} ± {ns_chain.std():.4f}   truth {ns_truth:.4f}   "
      f"({(ns_chain.mean()-ns_truth)/max(ns_chain.std(),1e-12):+.2f}σ)")
    A(f"  A_p : {ap_chain.mean():.3e} ± {ap_chain.std():.3e}   truth {ap_truth:.3e}   "
      f"({(ap_chain.mean()-ap_truth)/max(ap_chain.std(),1e-30):+.2f}σ)")
    A(f"  τ₀  : {tau0_chain.mean():.4f} ± {tau0_chain.std():.4f}   truth {tau0_truth:.4f}   "
      f"({(tau0_chain.mean()-tau0_truth)/max(tau0_chain.std(),1e-12):+.2f}σ)  (z={z_rep:.2f})")
    A("")
    A("## EXTRAPOLATION to production coverage (assumptions stated)")
    A(f"  smoke min-ESS over (A_p,n_s,τ₀) = {min_ess:.1f}; wall/ESS = {wall_per_ess:.2f} s/eff-sample")
    A(f"  ASSUMPTIONS: (1) production targets ESS≈{prod_ess_target} per param (Stan rule);")
    A(f"    (2) production uses dense_mass + max_tree_depth=10 vs smoke mtd={args.max_tree_depth}")
    A(f"        → leapfrogs/sample inflate by tree_factor≈{tree_factor:.2f} (sub-exponential;")
    A(f"        the dense mass conditions the geometry so trees rarely saturate post-warmup);")
    A(f"    (3) 1 chain, CPU, wall≈CPU-h (single core); (4) warmup cost ≈ included in wall/ESS.")
    A(f"  per-mock @ production ≈ {per_mock_prod_h:.2f} CPU-h")
    A(f"    → N=99  mocks ≈ {n99_h:.0f} CPU-h   ({n99_h/24:.1f} core-days)")
    A(f"    → N=600 mocks ≈ {n600_h:.0f} CPU-h   ({n600_h/24:.1f} core-days)")
    A(f"  NOTE: SLURM-only; embarrassingly parallel over mocks (fold_in seeding → array job).")
    A("")
    A("## SCOPE (n=6 floor discipline)")
    A("  floor covers LF→HR GENERALIZATION only; HR→truth is the locked k<0.06 cap.")
    A("  ns>0.98 posterior rests on MF-correction EXTRAPOLATION (sigma_edge inflates C_emu;")
    A("  the (z,τ₀) interaction is untested there) — treat as GUARDED, not nominal.")

    txt = "\n".join(lines)
    print("\n" + txt)
    (FIGDIR / "mf_closure_smoke.txt").write_text(txt + "\n")
    print(f"\n[mf-smoke] wrote {FIGDIR / 'mf_closure_smoke.txt'}")

    # save a npz so the figures can be regenerated without re-running NUTS
    np.savez(FIGDIR / "mf_closure_smoke.npz",
             theta=theta, tau0=tau0, ns_chain=ns_chain, ap_chain=ap_chain,
             tau0_chain=tau0_chain, kept_global=kept_global, z_global=zg,
             ns_truth=ns_truth, ap_truth=ap_truth, tau0_truth=tau0_truth,
             z_rep=z_rep, n_div=n_div, wall=wall, dense_dim=dense_dim)

    # ---- figures --------------------------------------------------------------
    _make_figures(ctx, mock_legs, truth_pack, core_per_leg, kept_global,
                  ns_chain, ap_chain, tau0_chain, ns_truth, ap_truth, tau0_truth, z_rep)


def _make_figures(ctx, mock_legs, truth_pack, core_per_leg, kept_global,
                  ns_chain, ap_chain, tau0_chain, ns_truth, ap_truth, tau0_truth, z_rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ----- FIG 1: posteriors (1D for A_p,n_s,τ₀ + 2D A_p-n_s), truth marked -----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (chain, truth, label) in zip(
            axes.ravel()[:3],
            [(ns_chain, ns_truth, "n_s"), (ap_chain, ap_truth, "A_p"),
             (tau0_chain, tau0_truth, f"τ₀(z≈{z_rep:.1f})")]):
        ax.hist(chain, bins=25, color="steelblue", alpha=0.8, density=True)
        ax.axvline(truth, color="crimson", lw=2, ls="--", label=f"truth {truth:.3g}")
        ax.axvline(chain.mean(), color="k", lw=1.5, label=f"mean {chain.mean():.3g}")
        ax.set_xlabel(label); ax.set_ylabel("posterior density"); ax.legend(fontsize=8)
    ax = axes.ravel()[3]
    ax.scatter(ns_chain, ap_chain, s=6, alpha=0.4, color="steelblue")
    ax.axvline(ns_truth, color="crimson", ls="--", lw=1.5)
    ax.axhline(ap_truth, color="crimson", ls="--", lw=1.5)
    ax.plot(ns_truth, ap_truth, "*", color="crimson", ms=16, label="truth")
    ax.set_xlabel("n_s"); ax.set_ylabel("A_p"); ax.legend(fontsize=8)
    fig.suptitle("MF closure SMOKE (n=1) — posterior vs mock truth  [MF fwd + C_emu floor]\n"
                 "NOT a coverage verdict; LF→HR floor only (HR→truth = k<0.06 cap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mf_closure_smoke_posterior.png", dpi=150)
    plt.close(fig)
    print(f"[mf-smoke] wrote {FIGDIR / 'mf_closure_smoke_posterior.png'}")

    # ----- FIG 2: C_emu/C_data per leg, MF-with-floor vs LF ----------------------
    theta9_t = jnp.asarray(truth_pack["theta9"])
    tau0_t = jnp.asarray(truth_pack["tau0_global"])
    alpha_t = jnp.asarray(truth_pack["alpha_hcd"])
    zg = np.asarray(ctx.z_global)
    fig, axes = plt.subplots(1, len(ctx.legs), figsize=(6 * len(ctx.legs), 4.5), squeeze=False)
    for ax, leg in zip(axes[0], mock_legs):
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        tau0_vec = tau0_t[jnp.asarray(sel)]
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
        core = core_per_leg[leg.name]
        keep = np.isfinite(np.asarray(leg.P_data))
        # MF + floor
        _, C_mf = DL.predict_P_obs_on_leg(
            ctx.model, theta9_t, tau0_vec, alpha_t, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor)
        # LF (no MF, no floor)
        _, C_lf = DL.predict_P_obs_on_leg(
            ctx.model, theta9_t, tau0_vec, alpha_t, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cd = np.diag(np.asarray(leg.C_data))
        cemu_mf = np.diag(np.asarray(C_mf)) - Cd
        cemu_lf = np.diag(np.asarray(C_lf)) - Cd
        kk = np.asarray(leg.k)
        ax.scatter(kk[keep], (cemu_lf / Cd)[keep], s=14, color="gray", label="LF C_emu/C_data")
        ax.scatter(kk[keep], (cemu_mf / Cd)[keep], s=14, color="crimson",
                   label="MF+floor C_emu/C_data")
        ax.set_xlabel("angular k [s/km]"); ax.set_ylabel("C_emu / C_data (diag)")
        ax.set_title(f"{leg.name}  (floor_on={leg.mf_floor_on})"); ax.legend(fontsize=8)
        ax.set_yscale("log")
    fig.suptitle("MF C_emu floor effect on each leg (diagonal), at the mock truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mf_closure_cemu_on_leg.png", dpi=150)
    plt.close(fig)
    print(f"[mf-smoke] wrote {FIGDIR / 'mf_closure_cemu_on_leg.png'}")

    # ----- FIG 3: whitened residual (P_data − P_model)/σ at the mock truth -------
    fig, axes = plt.subplots(1, len(ctx.legs), figsize=(6 * len(ctx.legs), 4.5), squeeze=False)
    for ax, leg in zip(axes[0], mock_legs):
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        tau0_vec = tau0_t[jnp.asarray(sel)]
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
        core = core_per_leg[leg.name]
        P_model, C_tot = DL.predict_P_obs_on_leg(
            ctx.model, theta9_t, tau0_vec, alpha_t, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor)
        keep = np.isfinite(np.asarray(leg.P_data))
        kr = np.where(keep)[0]
        r = np.asarray(leg.P_data)[kr] - np.asarray(P_model)[kr]
        Csub = np.asarray(C_tot)[np.ix_(kr, kr)]
        L = np.linalg.cholesky(Csub + 1e-12 * np.mean(np.diag(Csub)) * np.eye(len(kr)))
        wres = np.linalg.solve(L, r)        # whitened residual
        kk = np.asarray(leg.k)[kr]
        ax.scatter(kk, wres, s=14, color="darkgreen")
        ax.axhline(0, color="k", lw=0.8)
        for lvl in (-2, 2):
            ax.axhline(lvl, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("angular k [s/km]"); ax.set_ylabel("(P_data − P_model)/σ (whitened)")
        ax.set_title(f"{leg.name}  χ²/dof={float(wres @ wres)/len(kr):.2f}")
    fig.suptitle("Whitened residual at the mock truth (MF fwd + C_emu floor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mf_closure_whitened_resid.png", dpi=150)
    plt.close(fig)
    print(f"[mf-smoke] wrote {FIGDIR / 'mf_closure_whitened_resid.png'}")


if __name__ == "__main__":
    main()
