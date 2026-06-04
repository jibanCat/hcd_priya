"""Phase-C T3 — τ₀-smooth C_emu + logdet likelihood driver.

Pins: (a) sigma_at_tau0 interpolates the τ₀-banded error smoothly + reduces to the band
value at a centre; (b) gaussian_loglik == −½rᵀC⁻¹r − ½logdet C; (c) the driver is
finite + differentiable, ∂logL/∂τ₀ (jax.grad) matches central FD INCLUDING the logdet —
and the logdet term materially changes the τ₀ gradient (negative control: dropping it
is detectable); (d) the smooth unit-box prior is 0 inside, finite-gradient outside.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_likelihood_driver.py -v
"""
import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.likelihood import sigma_at_tau0, gaussian_loglik, _KIM_AMP, _KIM_SLOPE
from hcd_analysis.emulator import inference as I


def _ctx(n_k=12, n_basis=6, n_tb=4, seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    w_c = jnp.asarray(rng.uniform(0.5, 1.0, 4))
    delta = jnp.asarray(rng.uniform(-0.3, 0.3, (3, n_k)))
    alpha_hcd = jnp.asarray(rng.uniform(0.5, 1.5, 3))
    sigma_zb = jnp.asarray(rng.uniform(0.01, 0.05, (4, n_k, n_tb)))
    alpha_centres = jnp.asarray(np.array([0.66, 0.83, 1.15, 1.33]))
    cosmic_cov = jnp.asarray(rng.uniform(1.0, 4.0, n_k))         # diag variance
    dla_shot_flag = jnp.zeros(n_k, bool)
    z, z_unit = 3.0, 0.5
    theta9 = jnp.full(9, 0.5)
    # a plausible "data": the model's own P_obs at a nearby τ₀ + noise so r != 0
    tau0_data = 0.40
    P_data = np.asarray(I.predict_P_obs(model, theta9, z_unit, tau0_data, alpha_hcd,
                                        w_c, pf, delta)) + rng.normal(0, 0.5, n_k)
    return dict(model=model, theta9=theta9, z=z, z_unit=z_unit, w_c=w_c, delta=delta,
                alpha_hcd=alpha_hcd, pf=pf, sigma_zb=sigma_zb, alpha_centres=alpha_centres,
                cosmic_cov=cosmic_cov, dla_shot_flag=dla_shot_flag, P_data=jnp.asarray(P_data))


def test_sigma_at_tau0_recovers_band_value_at_centre():
    c = _ctx()
    z = c["z"]; ac = c["alpha_centres"]
    for tb, a in enumerate(np.asarray(ac)):
        tau0 = float(a * _KIM_AMP * (1 + z) ** _KIM_SLOPE)      # α == centre tb
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
    rng = np.random.default_rng(1)
    K = 8
    A = rng.normal(size=(K, K)); C = A @ A.T + K * np.eye(K)   # SPD
    r = rng.normal(size=K)
    got = float(gaussian_loglik(jnp.asarray(r), jnp.asarray(C)))
    sign, logdet = np.linalg.slogdet(C)
    ref = -0.5 * (r @ np.linalg.solve(C, r)) - 0.5 * logdet
    assert np.isclose(got, ref, rtol=1e-8)


def test_driver_finite_and_differentiable():
    c = _ctx()
    f = lambda th, t, a: I.log_lik_single_z(
        c["model"], th, c["z_unit"], c["z"], t, a, w_c=c["w_c"], delta_hcd=c["delta"],
        pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
        cosmic_cov=c["cosmic_cov"], P_data=c["P_data"], dla_shot_flag=c["dla_shot_flag"])
    val = float(f(c["theta9"], 0.4, c["alpha_hcd"]))
    assert np.isfinite(val)
    gth, gt, ga = jax.grad(f, argnums=(0, 1, 2))(c["theta9"], 0.4, c["alpha_hcd"])
    assert np.isfinite(np.asarray(gth)).all()
    assert np.isfinite(float(gt)) and np.isfinite(np.asarray(ga)).all()


def test_dlogL_dtau0_matches_finite_diff_and_logdet_is_load_bearing():
    c = _ctx()

    def make(include_logdet):
        return lambda t: I.log_lik_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], t, c["alpha_hcd"],
            w_c=c["w_c"], delta_hcd=c["delta"], pf_stats=c["pf"],
            sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic_cov"], P_data=c["P_data"],
            dla_shot_flag=c["dla_shot_flag"], include_logdet=include_logdet)

    f_full = make(True)
    t0 = 0.40
    g_full = float(jax.grad(f_full)(t0))
    fd = (float(f_full(t0 + 1e-4)) - float(f_full(t0 - 1e-4))) / 2e-4
    # (c) autodiff ∂logL/∂τ₀ matches central FD of the FULL (logdet-bearing) log-lik
    assert np.isclose(g_full, fd, rtol=1e-4), f"grad {g_full} vs FD {fd}"

    # negative control: the logdet term materially changes the τ₀ gradient
    g_nolog = float(jax.grad(make(False))(t0))
    assert abs(g_full - g_nolog) > 1e-6 * (abs(g_full) + 1.0), \
        "logdet contributes ~0 to ∂logL/∂τ₀ — it must be load-bearing once C depends on τ₀"


def test_unit_box_logprior_zero_inside_finite_grad_outside():
    inside = I.unit_box_logprior(jnp.full(9, 0.5))
    assert float(inside) == 0.0
    out = jnp.array([0.5] * 8 + [1.3])                 # one param above the box
    assert float(I.unit_box_logprior(out)) < 0.0
    g = jax.grad(I.unit_box_logprior)(out)
    assert np.isfinite(np.asarray(g)).all() and float(g[8]) < 0.0   # pushes back in
