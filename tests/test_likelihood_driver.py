"""Phase-C T3 — τ₀-smooth C_emu + logdet likelihood driver (+ post-review hardening).

Pins: (a) sigma_at_tau0 interpolates the τ₀-banded error smoothly + reduces to the band
value at a centre; (b) gaussian_loglik == −½rᵀC⁻¹r − ½logdet C; (c) the driver is finite
+ differentiable, ∂logL/∂τ₀ matches central FD INCLUDING the logdet — and the logdet term
materially changes the τ₀ gradient (negative control); (d) the smooth unit-box prior is 0
inside, finite-gradient outside.

POST-REVIEW REGRESSIONS (the production NaN path the original fixture missed):
  C1  ∂logL/∂τ₀ stays FINITE when σ has all-NaN-over-Tb rows + P_data has NaN bins
      (the high-k Nyquist / out-of-range cells of the real error vector).
  I1  a degenerate C (zero diagonal) gives finite logL + grad (SPD jitter).
  inflate  cemu_inflate scales the emulator variance.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_likelihood_driver.py -v
"""
import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.predict import predict_P_obs
from hcd_analysis.emulator.likelihood import (
    sigma_at_tau0, gaussian_loglik, assemble_covariance, _KIM_AMP, _KIM_SLOPE,
)
from hcd_analysis.emulator import inference as I


def _ctx(n_k=12, n_basis=6, n_tb=4, seed=0, nan=False):
    """Synthetic single-z context. ``nan=True`` injects production-realistic all-NaN-
    over-Tb σ rows + NaN P_data bins at the top-2 k (the high-k Nyquist cells), with a
    valid_k mask — the path the original fixture missed (JAX review C1)."""
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    w_c = jnp.asarray(rng.uniform(0.5, 1.0, 4))
    delta = jnp.asarray(rng.uniform(-0.3, 0.3, (3, n_k)))
    alpha_hcd = jnp.asarray(rng.uniform(0.5, 1.5, 3))
    sigma_np = rng.uniform(0.01, 0.05, (4, n_k, n_tb))
    cosmic_np = rng.uniform(1.0, 4.0, n_k)
    z, z_unit, theta9 = 3.0, 0.5, jnp.full(9, 0.5)
    P_data = np.asarray(predict_P_obs(model, theta9, z_unit, 0.40, alpha_hcd,
                                      w_c, pf, delta)) + rng.normal(0, 0.5, n_k)
    valid_k = np.ones(n_k, bool)
    if nan:
        sigma_np[:, -2:, :] = np.nan          # all-NaN-over-Tb rows (Nyquist)
        sigma_np[:, -3, 0] = np.nan           # partial-NaN row (one band empty)
        P_data[-2:] = np.nan                  # no data at out-of-range bins
        valid_k[-2:] = False
    return dict(model=model, theta9=theta9, z=z, z_unit=z_unit, w_c=w_c, delta=delta,
                alpha_hcd=alpha_hcd, pf=pf, sigma_zb=jnp.asarray(sigma_np),
                alpha_centres=jnp.asarray([0.66, 0.83, 1.15, 1.33]),
                cosmic_cov=jnp.asarray(cosmic_np), dla_shot_flag=jnp.zeros(n_k, bool),
                P_data=jnp.asarray(P_data), valid_k=jnp.asarray(valid_k))


def _f(c, include_logdet=True, cemu_inflate=1.0):
    return lambda th, t, a: I.log_lik_single_z(
        c["model"], th, c["z_unit"], c["z"], t, a, w_c=c["w_c"], delta_hcd=c["delta"],
        pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
        cosmic_cov=c["cosmic_cov"], P_data=c["P_data"], dla_shot_flag=c["dla_shot_flag"],
        valid_k=c["valid_k"], cemu_inflate=cemu_inflate, include_logdet=include_logdet)


def test_sigma_at_tau0_recovers_band_value_at_centre():
    c = _ctx(); z = c["z"]; ac = c["alpha_centres"]
    for tb, a in enumerate(np.asarray(ac)):
        tau0 = float(a * _KIM_AMP * (1 + z) ** _KIM_SLOPE)
        s = np.asarray(sigma_at_tau0(c["sigma_zb"], ac, z, tau0))
        assert np.allclose(s, np.asarray(c["sigma_zb"])[:, :, tb], rtol=1e-6)


def test_sigma_at_tau0_interpolates_and_is_differentiable():
    c = _ctx(); z = c["z"]; ac = c["alpha_centres"]
    a_mid = 0.5 * (float(ac[0]) + float(ac[1]))
    tau0 = a_mid * _KIM_AMP * (1 + z) ** _KIM_SLOPE
    s = np.asarray(sigma_at_tau0(c["sigma_zb"], ac, z, tau0))
    lo = np.asarray(c["sigma_zb"])[:, :, 0]; hi = np.asarray(c["sigma_zb"])[:, :, 1]
    assert np.all((s >= np.minimum(lo, hi) - 1e-9) & (s <= np.maximum(lo, hi) + 1e-9))
    g = jax.grad(lambda t: jnp.sum(sigma_at_tau0(c["sigma_zb"], ac, z, t)))(tau0)
    assert np.isfinite(float(g))


def test_gaussian_loglik_matches_reference():
    rng = np.random.default_rng(1); K = 8
    A = rng.normal(size=(K, K)); C = A @ A.T + K * np.eye(K)
    r = rng.normal(size=K)
    got = float(gaussian_loglik(jnp.asarray(r), jnp.asarray(C)))
    _, logdet = np.linalg.slogdet(C)
    ref = -0.5 * (r @ np.linalg.solve(C, r)) - 0.5 * logdet
    assert np.isclose(got, ref, rtol=1e-6)   # jitter is ~1e-10·mean diag -> negligible


def test_driver_finite_and_differentiable():
    c = _ctx(); f = _f(c)
    assert np.isfinite(float(f(c["theta9"], 0.4, c["alpha_hcd"])))
    gth, gt, ga = jax.grad(f, argnums=(0, 1, 2))(c["theta9"], 0.4, c["alpha_hcd"])
    assert np.isfinite(np.asarray(gth)).all()
    assert np.isfinite(float(gt)) and np.isfinite(np.asarray(ga)).all()


def test_dlogL_dtau0_matches_finite_diff_and_logdet_is_load_bearing():
    c = _ctx(); f_full = _f(c, include_logdet=True)
    t0 = 0.40
    g_full = float(jax.grad(lambda t: f_full(c["theta9"], t, c["alpha_hcd"]))(t0))
    ff = lambda t: float(f_full(c["theta9"], t, c["alpha_hcd"]))
    fd = (ff(t0 + 1e-4) - ff(t0 - 1e-4)) / 2e-4
    assert np.isclose(g_full, fd, rtol=1e-4), f"grad {g_full} vs FD {fd}"
    g_nolog = float(jax.grad(lambda t: _f(c, include_logdet=False)(c["theta9"], t, c["alpha_hcd"]))(t0))
    assert abs(g_full - g_nolog) > 1e-6 * (abs(g_full) + 1.0), "logdet must be load-bearing"


def test_C1_nan_sigma_and_data_give_finite_gradient():
    """JAX review C1 regression: real error vector has all-NaN-over-Tb σ rows + NaN data
    bins; ∂logL/∂τ₀ and ∂logL/∂θ must stay FINITE (NUTS-killer otherwise)."""
    c = _ctx(nan=True); f = _f(c)
    for t0 in (0.30, 0.40, 0.55):
        val = float(f(c["theta9"], t0, c["alpha_hcd"]))
        gth, gt, ga = jax.grad(f, argnums=(0, 1, 2))(c["theta9"], t0, c["alpha_hcd"])
        assert np.isfinite(val), f"logL non-finite at τ₀={t0}"
        assert np.isfinite(float(gt)), f"∂logL/∂τ₀ NaN at τ₀={t0} (C1)"
        assert np.isfinite(np.asarray(gth)).all() and np.isfinite(np.asarray(ga)).all()


def test_I1_degenerate_covariance_is_jittered_finite():
    """JAX review I1 regression: a zero-diagonal C must not NaN-poison via Cholesky."""
    K = 6
    C = jnp.asarray(np.diag([2.0, 0.0, 3.0, 1.0, 0.0, 4.0]))   # two zero diagonals
    r = jnp.asarray(np.linspace(-1, 1, K))
    val = float(gaussian_loglik(r, C))
    g = jax.grad(lambda rr: gaussian_loglik(rr, C))(r)
    assert np.isfinite(val) and np.isfinite(np.asarray(g)).all()


def test_cemu_inflate_scales_emulator_variance():
    c = _ctx()
    sig_ck = sigma_at_tau0(c["sigma_zb"], c["alpha_centres"], c["z"], 0.4)
    from hcd_analysis.emulator.predict import predict_P_filt
    P_filt = predict_P_filt(c["model"], c["theta9"], c["z_unit"], 0.4, c["pf"])
    dscale = jnp.abs(c["delta"])
    base = assemble_covariance(jnp.zeros(P_filt.shape[1]), sig_ck, c["w_c"],
                               jnp.zeros((3, P_filt.shape[1])), c["alpha_hcd"], P_filt,
                               dscale, c["dla_shot_flag"], cemu_inflate=1.0)
    infl = assemble_covariance(jnp.zeros(P_filt.shape[1]), sig_ck, c["w_c"],
                               jnp.zeros((3, P_filt.shape[1])), c["alpha_hcd"], P_filt,
                               dscale, c["dla_shot_flag"], cemu_inflate=4.0)
    assert np.allclose(np.diag(np.asarray(infl)), 4.0 * np.diag(np.asarray(base)), rtol=1e-6)


def test_unit_box_logprior_zero_inside_finite_grad_outside():
    assert float(I.unit_box_logprior(jnp.full(9, 0.5))) == 0.0
    out = jnp.array([0.5] * 8 + [1.3])
    assert float(I.unit_box_logprior(out)) < 0.0
    g = jax.grad(I.unit_box_logprior)(out)
    assert np.isfinite(np.asarray(g)).all() and float(g[8]) < 0.0
