"""Research round R1, T3 benchmark kit (hcd_analysis/emulator/r1_bench.py): the factorial likelihood, the mode-aligned T3
factor bound at the query theta, the bounded MAP with Laplace widths, and the benchmark metrics. Cache-free."""
import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hcd_analysis.emulator import cemu_t3 as T3
from hcd_analysis.emulator import r1_bench as RB
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def _psd(rng, n):
    X = rng.normal(0, 1, (n, n))
    return X @ X.T + n * np.eye(n)


def test_factorial_loglik_is_the_gaussian_when_both_matrices_agree():
    rng = np.random.default_rng(0)
    C = _psd(rng, 7); r = rng.normal(0, 1, 7)
    np.testing.assert_allclose(float(RB.factorial_loglik(jnp.asarray(r), jnp.asarray(C), jnp.asarray(C))),
                               multivariate_normal(np.zeros(7), C).logpdf(r), rtol=1e-12)


def test_factorial_loglik_splits_quadratic_and_log_determinant():
    rng = np.random.default_rng(1)
    Cq, Cd = _psd(rng, 5), _psd(rng, 5); r = rng.normal(0, 1, 5)
    expect = -0.5 * r @ np.linalg.solve(Cq, r) - 0.5 * np.linalg.slogdet(Cd)[1] - 2.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(RB.factorial_loglik(jnp.asarray(r), jnp.asarray(Cq), jnp.asarray(Cd))), expect,
                               rtol=1e-12)


def test_mode_aligned_factor_is_the_numpy_binding_times_the_mode_basis():
    rng = np.random.default_rng(2)
    zs = np.round(np.arange(2.0, 4.41, 0.2), 1)
    kb = np.array([0.012, 0.02, 0.031, 0.015, 0.045]); iz = np.array([3, 3, 3, 6, 6])
    th = rng.uniform(0.1, 0.9, 9)
    Mi = T3.mode_set(KCOM, zs, iz, kb, np.zeros(9), np.ones(9))
    U = rng.normal(0, 0.01, (Mi.size, 3))
    rows = np.array([1, 2, 4, 6, 7]); N = 9
    got = np.asarray(RB.mode_aligned_factor(jnp.asarray(th), KCOM, zs, iz, kb, Mi, jnp.asarray(U), rows, N))
    W = T3.weights(KCOM, th, zs, iz, kb)[:, Mi]
    expect = np.zeros((N, 3)); expect[rows] = W @ U
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-18)


def test_map_and_laplace_on_a_linear_gaussian_model_are_exact():
    rng = np.random.default_rng(3)
    n, N = 4, 30
    A = rng.normal(0, 1, (N, n)); C = _psd(rng, N) * 0.1
    p_true = np.array([0.3, -0.2, 0.5, 0.1])
    d = A @ p_true + rng.multivariate_normal(np.zeros(N), C)
    nlp = lambda p: 0.5 * (d - A @ p) @ jnp.linalg.solve(jnp.asarray(C), d - A @ p)
    out = RB.map_laplace(nlp, p0=np.zeros(n), bounds=[(-10, 10)] * n)
    F = A.T @ np.linalg.solve(C, A)
    np.testing.assert_allclose(out["map"], np.linalg.solve(F, A.T @ np.linalg.solve(C, d)), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(out["sigma"], np.sqrt(np.diag(np.linalg.inv(F))), rtol=1e-6)
    assert out["converged"] and not np.any(out["at_bound"])


def test_pull_coverage_and_bootstrap_by_truth():
    pulls = np.array([[0.2, 3.0], [-0.5, 0.1], [1.5, -0.2], [0.0, 2.5]])          # (mocks, params)
    groups = np.array([0, 0, 1, 1])                                                 # two noise draws per truth
    m = RB.pull_summary(pulls, groups, n_boot=200, seed=0)
    np.testing.assert_allclose(m["mean"], pulls.mean(0)); np.testing.assert_allclose(m["rms"], np.sqrt((pulls ** 2).mean(0)))
    np.testing.assert_allclose(m["cov68"], [0.75, 0.5]); np.testing.assert_allclose(m["cov95"], [1.0, 0.5])
    assert m["mean_ci"].shape == (2, 2)


def test_paired_difference_resamples_whole_truths():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 96); b = a + 0.3
    groups = np.repeat(np.arange(48), 2)
    lo, hi = RB.paired_ci(a, b, groups, n_boot=500, seed=0)
    np.testing.assert_allclose([lo, hi], [0.3, 0.3], atol=1e-12)            # paired b - a: a constant shift has no spread



# --------------------------------------------------------------------------------------------- #
#  truth (mock) builder: the forward at the truth's parameters with the truth's own spectra in place of the emulator
#  reproduces the mock to round-off (coordinate, MF, HCD and DLA-core algebra identical). Real cache (gate mode).
# --------------------------------------------------------------------------------------------- #
GATEC = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC"
MF = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/mf/gateD/mf_modes_all6.npz"
DLA_CORE = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateE/products/dla_core_gateE.npz"


def test_truth_equals_the_forward_with_an_oracle_emulator(monkeypatch):
    import os
    from tests.gate_helpers import real_cache_path
    if not all(os.path.exists(x) for x in (MF, DLA_CORE, f"{GATEC}/prod_repaired_seed0.eqx")):
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail("inputs absent")
        pytest.skip("inputs absent")
    from hcd_analysis.emulator import closure_legb as CL, fisher_kit as FK, forward as FW
    from hcd_analysis.emulator.data import load_cache, tau0_ladder_factor
    ctx, _ = CL.build_legb_ctx(ensemble_ckpts=[f"{GATEC}/prod_repaired_seed{i}" for i in range(5)], mf_product=MF,
                               dla_core_product=DLA_CORE, ks_kwargs=dict(k_max=0.065), survey="DESI")
    leg = next(l for l in ctx.legs if l.name == "DESI")
    d = load_cache(real_cache_path("lf"))
    names = np.asarray(d["sim_name"]).astype(str)
    sim = sorted(set(names))[7]
    rows = RB.truth_rows(d, sim, 1.0112, np.asarray(leg.z))
    th = np.asarray(d["params_unit"], float)[rows[0]]
    alpha_exact = float(np.mean(tau0_ladder_factor(np.asarray(d["tau0"], float)[rows], np.asarray(leg.z))))
    p = FK.p_centre(ctx, th); p[9] = alpha_exact                   # the rung's exact ladder factor
    _, tau0_g, alpha_g = FK.forward_inputs(ctx, jnp.asarray(p))
    zg = np.asarray(ctx.z_global)
    sel = np.array([int(np.argmin(np.abs(zg - z))) for z in leg.z])
    truth = RB.truth_on_leg(d, rows, leg, ctx.mf, np.asarray(alpha_g)[sel], np.asarray(ctx.dla_core_leg["DESI"]))
    zrow = {float(np.round(z, 6)): r for z, r in zip(leg.z, rows)}

    def oracle(model, theta9, z_unit, tau0, pf_stats):
        z = float(np.round(2.0 + 3.4 * float(z_unit), 6))
        return jnp.asarray(np.asarray(d["P_filt"], float)[zrow[z]])
    monkeypatch.setattr(FW, "predict_P_filt", oracle)
    out = FW.predict_leg(ctx.model, jnp.asarray(th), tau0_g[sel], alpha_g[sel], leg=leg, k_com=ctx.k_com_hmpc,
                         pf_stats=ctx.pf_stats, dla_core=ctx.dla_core_leg["DESI"], mf=ctx.mf)
    keep = np.isfinite(np.asarray(leg.P_data))
    np.testing.assert_allclose(np.asarray(out.P_model)[keep], truth[keep], rtol=1e-9)
    assert np.allclose(np.asarray(tau0_g)[sel], np.asarray(d["tau0"], float)[rows], rtol=1e-10)
    assert np.allclose(tau0_ladder_factor(np.asarray(d["tau0"], float)[rows], np.asarray(leg.z)), 1.0112, atol=1e-4)



# --------------------------------------------------------------------------------------------- #
#  per-variant negative log posterior (synthetic context): P/M/T1 are Gaussian with their own covariance; the factorial
#  diagnostics take the quadratic from one and the log-det from the other; the force decomposition sums to the gradient
# --------------------------------------------------------------------------------------------- #
def _bench_setup():
    from tests.test_fisher_kit import _synthetic
    from tests.test_forward_v2 import _t1
    ctx = _synthetic()
    leg = ctx.legs[0]
    rho, ac = _t1(leg.n_z, Tb=1)
    ctx = ctx._replace(alpha_centres=jnp.asarray([1.0]))
    rng = np.random.default_rng(9)
    N = leg.k.size
    zs = np.round(np.arange(2.0, 4.41, 0.2), 1)
    iz = np.array([int(np.argmin(np.abs(zs - z))) for z in np.asarray(leg.z_row)])
    Mi = T3.mode_set(KCOM, zs, iz, np.asarray(leg.k), np.zeros(9), np.ones(9))
    spec = dict(z_cells=zs, iz=iz, k_bins=np.asarray(leg.k), Mi=Mi, rows=np.arange(N), n_leg=N)
    args = dict(model=ctx.model, pf=ctx.pf_stats, d=jnp.asarray(np.asarray(leg.P_data) * 1.01), rho=rho,
                UP=jnp.asarray(rng.normal(0, 0.02, (N, 3))), wP=jnp.asarray([1.0, 0.5, 0.2]),
                UM=jnp.asarray(rng.normal(0, 0.02, (Mi.size, 3))), wM=jnp.asarray([1.0, 0.5, 0.2]),
                Pamp=jnp.asarray(np.asarray(leg.P_data)))
    return ctx, leg, spec, args


def test_variant_losses_are_gaussians_with_their_own_covariance_and_factorial_mixes():
    from hcd_analysis.emulator import fisher_kit as FK
    ctx, leg, spec, args = _bench_setup()
    p = jnp.asarray(FK.p_centre(ctx, np.full(9, 0.45)))
    parts = RB.leg_parts(ctx, leg, spec)
    mu, C = {}, {}
    for v in ("P", "M", "T1"):
        mu[v], C[v] = parts(p, args, v)
    r = args["d"][jnp.asarray(FK.kept(leg))] - mu["P"]
    lp = RB.log_prior(ctx)(p)
    for v, (cq, cd) in dict(P=("P", "P"), M=("M", "M"), T1=("T1", "T1"), MQ=("M", "P"), MD=("P", "M")).items():
        f = RB.make_nlp(ctx, leg, spec, v)
        expect = -RB.factorial_loglik(r, C[cq], C[cd]) - lp
        np.testing.assert_allclose(float(f(p, args)), float(expect), rtol=1e-12)
    assert not np.allclose(np.asarray(C["P"]), np.asarray(C["M"]))


def test_force_decomposition_sums_to_the_log_likelihood_gradient():
    from hcd_analysis.emulator import fisher_kit as FK
    ctx, leg, spec, args = _bench_setup()
    p = jnp.asarray(FK.p_centre(ctx, np.full(9, 0.45)))
    g = RB.force_terms(ctx, leg, spec, "M", p, args)
    import jax
    total = jax.grad(lambda q: -RB.make_nlp(ctx, leg, spec, "M")(q, args) - RB.log_prior(ctx)(q))(p)
    np.testing.assert_allclose(np.asarray(g["mean"] + g["cov_quad"] + g["logdet"]), np.asarray(total), rtol=1e-8,
                               atol=1e-10 * float(np.max(np.abs(np.asarray(total)))))
