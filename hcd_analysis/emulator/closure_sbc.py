"""Phase-C T4a — Leg-A true-SBC orchestration (the heavy path; NUTS lives here).

Leg A certifies the SAMPLER (plan §0): mocks from the emulator's own generative model →
rank-uniformity is the correct, attributable null. Per mock:
  1. draw a prior truth + a matched-C mock (``closure_mocks``);
  2. run NUTS (``sampler_numpyro.run_nuts``), track divergences;
  3. thin to near-independence (``closure_diagnostics.thin_to_ess``);
  4. SBC-rank the packed truth per quantity (``sbc_ranks_multiparam``) + the joint
     log-likelihood rank (``loglik_rank`` — Modrak+2023, the primary 1-D statistic);
then ``ecdf_pit_bands`` per quantity → pass/fail (Säilynoja+2022 simultaneous bands).

``--smoke`` runs the FULL path at tiny size (N≈20, n_warmup/n_samples≈60, n_z=1) end to
end and prints per-quantity pass/fail + #divergences. Heavy orchestration is kept out of
import time (everything is under functions / ``__main__``).
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

# unbuffered prints so progress shows under a pipe / SLURM redirect
print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.closure_ctx import (
    Ctx, ctx_kim, log_lik_from_ctx, param_names, unpack,
)
from hcd_analysis.emulator.meanflux_prior import meanflux_tau0_prior
from hcd_analysis.emulator.inference import hcd_incidence_prior
from hcd_analysis.emulator.predict import predict_P_obs
from hcd_analysis.emulator.closure_mocks import draw_leg_a_truth, make_leg_a_mock
from hcd_analysis.emulator.sampler_numpyro import run_nuts
from hcd_analysis.emulator.closure_diagnostics import (
    sbc_ranks_multiparam, loglik_rank, thin_to_ess, ecdf_pit_bands,
)

REPO = "/home/mfho/hcd_priya"
# The ONE production model: the C_emu LOSO error vector was built from the final_fold*
# recipe, so final_fold0 is the matched pair (plan §5).
CKPT = f"{REPO}/checkpoints/final_fold0"
ERROR_VECTOR = f"{REPO}/checkpoints/error_vector.npz"

# The ECDF simultaneous-band gate is only correctly sized for L≳99 thinned draws: under a
# TRUE discrete-uniform rank null the band (calibrated on CONTINUOUS uniforms) over-rejects
# at small L — measured type-I = 0.20 at L=49, 0.05 at L=99, 0.043 at L≥199 (verify script
# 2026-06-05). So the gate is only VALID at L_eff ≥ L_FLOOR; below it the pass/fail is
# PATH-only (the --smoke regime). Production thins to L≳99 where the gate is well-calibrated.
L_FLOOR = 99
# Re-run a divergent mock at higher target_accept before excluding it: HMC divergences
# cluster on hard geometry (not random wrt truth), so silent exclusion biases the kept set
# toward easy regions (CS review). Escalate target_accept; exclude only if still divergent.
DIVERGENCE_RETRY_TARGET_ACCEPT = (0.95, 0.99)


# ----------------------------------------------------------------------------
# Build a production ctx from final_fold0 + error_vector.npz.
# ----------------------------------------------------------------------------
def build_ctx(n_z=3, seed=0, ckpt=CKPT, error_vector=ERROR_VECTOR,
              shot_inflate=10.0, cemu_inflate=1.0):
    """Build a real Ctx matched to the production (final_fold0, error_vector.npz) pair.

    Picks ``n_z`` in-range redshifts, maps each to its z-band slice of the (4,K,Zb,Tb)
    error vector, sets the τ₀ prior from the Kim MVP (``meanflux_tau0_prior``) and the HCD
    incidence prior from ``hcd_incidence_prior``. cosmic_cov is a 5% fractional floor on
    the forward-modelled truth (an MVP placeholder — see LYA-CONSULT in the report). The
    returned P_data is a placeholder (the SBC overwrites it per mock); the truth here is
    only a sane fiducial for the fixture, NOT a Leg-A truth.
    """
    model, meta, norm = T.load_checkpoint(ckpt)
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    ev = np.load(error_vector, allow_pickle=True)
    sigma = ev["sigma"]                                # (4,K,Zb,Tb)
    C4, K, Zb, Tb = sigma.shape
    alpha_centres = jnp.asarray(ev["tau0_band_centres"])
    dla_shot_flag_k = jnp.asarray(ev["dla_shot_flag"])
    z_band_edges = ev["z_band_edges"]                  # (Zb+1,) = [-inf, ...interior..., inf]

    rng = np.random.default_rng(seed)
    # n_z in-range redshifts (data range z∈[2.2,4.6]); map each to a z-band slice of sigma.
    z = np.linspace(2.4, 4.2, n_z)
    zb_of_z = np.clip(np.digitize(z, z_band_edges[1:-1]), 0, Zb - 1)
    sigma_zb = jnp.asarray(np.stack([sigma[:, :, zb_of_z[i], :] for i in range(n_z)]))  # (n_z,4,K,Tb)

    # z_unit per the cache Z_LIMITS (2.0..5.4)
    z_unit = jnp.asarray((z - 2.0) / (5.4 - 2.0))
    z = jnp.asarray(z)

    # τ₀ mean-flux prior (Kim MVP, matched to the cache ladder).
    tau0_mu, tau0_sigma = meanflux_tau0_prior(z)

    # HCD incidence prior (per-class μ,σ) from a toy structural w_c at the z pivot.
    w_c_fid = jnp.asarray([1.0, 0.05, 0.02, 0.01])     # toy structural weights (clean,LLS,sub,DLA)
    alpha_mu_, alpha_sd_ = hcd_incidence_prior(w_c_fid[1:], z=3.0)  # (3,) on (LLS,sub,DLA)

    # a fiducial truth (unit-cube interior) to forward-model the fixture cosmic_cov.
    theta_fid = jnp.asarray(rng.uniform(0.35, 0.65, size=9))
    tau0_fid = jnp.asarray(np.asarray(tau0_mu))
    alpha_fid = jnp.asarray(np.maximum(np.asarray(alpha_mu_), 1e-4))
    dla_core = jnp.asarray(rng.uniform(0, 1e-3, size=(n_z, K)))     # toy positive FIXED core

    # valid_k = finite-σ bins (the real data-range mask: above-Nyquist σ rows are all-NaN).
    finite_sig = np.isfinite(np.asarray(sigma_zb)).all(axis=(1, 3))  # (n_z,K)
    valid_k = jnp.asarray(finite_sig)

    # forward-model the fiducial → cosmic_cov as a 5% fractional floor (MVP placeholder).
    P_fid = jnp.stack([predict_P_obs(model, theta_fid, z_unit[i], tau0_fid[i], alpha_fid,
                                     pf, dla_core[i]) for i in range(n_z)])  # (n_z,K)
    cosmic_cov = jnp.asarray((0.05 * np.nan_to_num(np.asarray(P_fid))) ** 2 + 1e-30)
    P_data = jnp.where(valid_k, P_fid, jnp.nan)        # placeholder (SBC overwrites per mock)

    dla_shot_flag = jnp.asarray(np.broadcast_to(np.asarray(dla_shot_flag_k), (n_z, K)))

    ctx = Ctx(
        model=model, pf_stats=pf, z=z, z_unit=z_unit, sigma_zb=sigma_zb,
        alpha_centres=alpha_centres, cosmic_cov=cosmic_cov, P_data=P_data,
        dla_core=dla_core, dla_shot_flag=dla_shot_flag, valid_k=valid_k, w_c_fid=w_c_fid,
        tau0_mu=tau0_mu, tau0_sigma=tau0_sigma, alpha_hcd_mu=jnp.asarray(alpha_mu_),
        alpha_hcd_sigma=jnp.asarray(alpha_sd_), n_z=n_z, K=int(K), Tb=int(Tb),
        shot_inflate=shot_inflate, cemu_inflate=cemu_inflate, include_logdet=True)
    return ctx


# ----------------------------------------------------------------------------
# Per-mock: NUTS → thinned draws matrix in the packed order.
# ----------------------------------------------------------------------------
def _draws_matrix(samples, n_z):
    """Stack NUTS sample sites into a (L, P) matrix in the packed param order
    [θ9, τ₀(n_z), α_lls, α_subdla, α_dla]. ``tau0_vec`` and ``alpha_dla`` are read from
    the deterministic sites (the τ₀ and DLA the forward model actually used)."""
    theta = np.asarray(samples["theta_unit"])          # (L,9)
    tau0 = np.asarray(samples["tau0_vec"])             # (L,n_z)
    a_lls = np.asarray(samples["alpha_lls"])[:, None]  # (L,1)
    a_sub = np.asarray(samples["alpha_subdla"])[:, None]
    a_dla = np.asarray(samples["alpha_dla"])[:, None]  # deterministic softplus
    return np.concatenate([theta, tau0, a_lls, a_sub, a_dla], axis=1)  # (L, 9+n_z+3)


def _loglik_of_draws(ctx_mock, draws):
    """log_lik(draw, mock_data) for each thinned draw — SAME ctx (mock data) as the truth.
    Vectorized over draws via vmap of ``log_lik_from_ctx``."""
    n_z = ctx_mock.n_z

    def one(vec):
        th, t0, al = unpack(jnp.asarray(vec), n_z)
        return log_lik_from_ctx(th, t0, al, ctx_mock)
    return np.asarray(jax.vmap(one)(jnp.asarray(draws)))


def run_leg_a_sbc(ctx0: Ctx, *, n_mocks, n_warmup, n_samples, seed, thin=True,
                  prob=0.95, verbose=True):
    """Full Leg-A true-SBC over ``n_mocks`` mocks. Returns a dict of per-quantity ranks +
    pass/fail + divergence bookkeeping.

    Per mock: draw truth+mock from the priors (matched C), NUTS, thin to ESS, rank the
    packed truth per quantity + the joint log-lik rank. Mocks with >0 divergences are
    FLAGGED (and excluded from the rank arrays) — divergences make the geometry suspect,
    so their ranks are not trustworthy (plan §3: exclude+re-run or flag).

    Returns dict:
      names         — the P quantity names + 'loglik';
      ranks         — (n_kept, P+1) int ranks (last col = loglik rank);
      L             — the (min) #thinned draws ranked against (ranks ∈ {0..L});
      n_divergent   — #mocks with >0 divergences (flagged/excluded);
      passed        — {name: bool} from ecdf_pit_bands;
      ecdf_bands    — {name: (lower,upper,ecdf,grid)} for plotting.
    """
    names = param_names(ctx0.n_z)
    key = jax.random.PRNGKey(int(seed))

    kept = []               # per kept mock: truth_vec, thinned draws, ll_true, ll_draws
    L_list = []
    n_div_total = 0
    n_divergent_mocks = 0
    n_excluded = 0

    for m in range(int(n_mocks)):
        key, k_truth, k_mock, k_nuts = jax.random.split(key, 4)
        truth = draw_leg_a_truth(ctx0, k_truth)
        ctx_mock, truth_vec, _info = make_leg_a_mock(ctx0, truth, k_mock)
        truth_vec = np.asarray(truth_vec)

        # run NUTS; on divergence, ESCALATE target_accept and re-run the SAME mock before
        # excluding (divergences are not random wrt truth → silent exclusion biases the set).
        base_seed = int(jax.random.randint(k_nuts, (), 0, 2**31 - 1))
        ta_schedule = (0.9,) + tuple(DIVERGENCE_RETRY_TARGET_ACCEPT)
        samples = n_div = None
        for attempt, ta in enumerate(ta_schedule):
            samples, n_div, _extra = run_nuts(
                ctx_mock, n_warmup=n_warmup, n_samples=n_samples,
                seed=base_seed + attempt, target_accept=ta)
            if n_div == 0:
                if attempt and verbose:
                    print(f"  [mock {m}] cleared divergences at target_accept={ta}")
                break
            if verbose:
                print(f"  [mock {m}] {n_div} divergence(s) at target_accept={ta}"
                      + (" -> retry" if attempt < len(ta_schedule) - 1 else ""))
        n_div_total += n_div
        if n_div > 0:                               # still divergent after the schedule
            n_divergent_mocks += 1
            n_excluded += 1
            if verbose:
                print(f"  [mock {m}] still divergent after retries -> FLAGGED + excluded")
            continue

        draws = _draws_matrix(samples, ctx0.n_z)        # (Lraw, P)
        if thin:
            draws_t, step, ess_min = thin_to_ess(draws)
        else:
            draws_t, step, ess_min = draws, 1, float(draws.shape[0])
        L = draws_t.shape[0]
        if L < 2:                                        # degenerate chain: cannot rank
            n_excluded += 1
            if verbose:
                print(f"  [mock {m}] thinned to L={L} (<2) -> excluded")
            continue

        # log-lik of the truth + thinned draws on the SAME mock data (Modrak+2023);
        # stash with the draws and defer ranking until L_common is known (below) so EVERY
        # mock is ranked against the IDENTICAL number of draws (valid-on-one-grid, Talts+2018).
        ll_true = float(log_lik_from_ctx(
            jnp.asarray(truth["theta9"]), jnp.asarray(truth["tau0_vec"]),
            jnp.asarray(truth["alpha_hcd"]), ctx_mock))
        ll_draws = _loglik_of_draws(ctx_mock, draws_t)          # (L,)
        kept.append(dict(truth_vec=truth_vec, draws=draws_t, ll_true=ll_true,
                         ll_draws=ll_draws))
        L_list.append(L)
        if verbose:
            print(f"  [mock {m}] L={L} (step {step}, ess_min {ess_min:.1f})")

    all_names = list(names) + ["loglik"]
    passed, ecdf_bands = {}, {}
    # ONE common grid: subsample EVERY mock's thinned draws to L_common = min over mocks
    # (evenly-spaced indices — NOT clipping the ranks, which would corrupt uniformity).
    # SBC ranks are only Uniform{0..L_common} if all mocks rank against the same #draws.
    L_eff = int(min(L_list)) if L_list else 0
    rank_rows = []
    if kept and L_eff >= 2:
        for rec in kept:
            sub = np.linspace(0, rec["draws"].shape[0] - 1, L_eff).round().astype(int)
            d = rec["draws"][sub]                               # (L_eff, P)
            lld = rec["ll_draws"][sub]                          # (L_eff,)
            pranks = sbc_ranks_multiparam(rec["truth_vec"], d)  # (P,)
            llrank = loglik_rank(rec["ll_true"], lld)
            rank_rows.append(np.concatenate([pranks, [llrank]]))
    ranks = np.array(rank_rows, dtype=int)                      # (n_kept, P+1)
    if ranks.size and L_eff >= 2:
        for j, nm in enumerate(all_names):
            lower, upper, ecdf, ok, grid = ecdf_pit_bands(ranks[:, j], L_eff, prob=prob)
            passed[nm] = bool(ok)
            ecdf_bands[nm] = (lower, upper, ecdf, grid)

    # the ECDF gate is only correctly sized at L_eff ≥ L_FLOOR (see the constant); below it
    # the bands over-reject and pass/fail is PATH-only, not a calibration verdict.
    gate_valid = bool(L_eff >= L_FLOOR)
    return dict(names=all_names, ranks=ranks, L=L_eff, gate_valid=gate_valid,
                n_div_total=n_div_total, n_divergent=n_divergent_mocks,
                n_excluded=n_excluded,
                n_kept=int(ranks.shape[0]) if ranks.size else 0,
                passed=passed, ecdf_bands=ecdf_bands)


# ----------------------------------------------------------------------------
# CLI — --smoke exercises the FULL path end to end at tiny size.
# ----------------------------------------------------------------------------
def _smoke(args):
    print(f"[smoke] building ctx (n_z={args.n_z}) from {CKPT}")
    ctx0 = build_ctx(n_z=args.n_z, seed=args.seed)
    print(f"[smoke] ctx: n_z={ctx0.n_z} K={ctx0.K} Tb={ctx0.Tb} "
          f"valid_k/z={[int(np.asarray(ctx0.valid_k)[i].sum()) for i in range(ctx0.n_z)]}")
    print(f"[smoke] running Leg-A SBC: N={args.n_mocks} mocks, "
          f"warmup/samples={args.n_warmup}/{args.n_samples}")
    res = run_leg_a_sbc(ctx0, n_mocks=args.n_mocks, n_warmup=args.n_warmup,
                        n_samples=args.n_samples, seed=args.seed, prob=args.prob)
    print("\n========== Leg-A SBC smoke summary ==========")
    print(f"mocks kept={res['n_kept']}/{args.n_mocks}  "
          f"divergent(excluded)={res['n_divergent']}  total_div={res['n_div_total']}  "
          f"L(thinned ranked-against)={res['L']}  gate_valid(L≥{L_FLOOR})={res['gate_valid']}")
    if res["passed"]:
        print(f"per-quantity ECDF-band pass/fail (prob={args.prob}):")
        for nm in res["names"]:
            print(f"  {nm:14s} : {'PASS' if res['passed'][nm] else 'FAIL'}")
        n_pass = sum(res["passed"].values())
        verdict = ("a CALIBRATION verdict" if res["gate_valid"]
                   else f"PATH-only — L={res['L']}<{L_FLOOR}, the band over-rejects at small L")
        print(f"\n{n_pass}/{len(res['names'])} quantities PASS ({verdict})")
    else:
        print("no kept mocks -> no band test (increase N or reduce divergences)")
    return res


def main():
    ap = argparse.ArgumentParser(description="Phase-C T4a Leg-A true-SBC orchestration")
    ap.add_argument("--smoke", action="store_true",
                    help="run the FULL path at tiny size (N≈20, n_z=1) and print pass/fail")
    ap.add_argument("--n-mocks", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--n-z", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prob", type=float, default=0.95)
    args = ap.parse_args()
    if args.smoke:
        _smoke(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
