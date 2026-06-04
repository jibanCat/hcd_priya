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
    sigma_at_tau0, gaussian_loglik, _KIM_AMP, _KIM_SLOPE,
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
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))     # DLA-core add-back (K,)
    alpha_hcd = jnp.asarray(rng.uniform(0.0, 0.5, 3))      # effective per-class incidence
    sigma_np = rng.uniform(0.01, 0.05, (4, n_k, n_tb))
    cosmic_np = rng.uniform(1.0, 4.0, n_k)
    z, z_unit, theta9 = 3.0, 0.5, jnp.full(9, 0.5)
    P_data = np.asarray(predict_P_obs(model, theta9, z_unit, 0.40, alpha_hcd,
                                      pf, dla_core)) + rng.normal(0, 0.5, n_k)
    valid_k = np.ones(n_k, bool)
    if nan:
        sigma_np[:, -2:, :] = np.nan          # all-NaN-over-Tb rows (Nyquist)
        sigma_np[:, -3, 0] = np.nan           # partial-NaN row (one band empty)
        P_data[-2:] = np.nan                  # no data at out-of-range bins
        valid_k[-2:] = False
    return dict(model=model, theta9=theta9, z=z, z_unit=z_unit, dla_core=dla_core,
                alpha_hcd=alpha_hcd, pf=pf, sigma_zb=jnp.asarray(sigma_np),
                alpha_centres=jnp.asarray([0.66, 0.83, 1.15, 1.33]),
                cosmic_cov=jnp.asarray(cosmic_np), dla_shot_flag=jnp.zeros(n_k, bool),
                P_data=jnp.asarray(P_data), valid_k=jnp.asarray(valid_k))


def _f(c, include_logdet=True, cemu_inflate=1.0):
    return lambda th, t, a: I.log_lik_single_z(
        c["model"], th, c["z_unit"], c["z"], t, a,
        pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
        cosmic_cov=c["cosmic_cov"], P_data=c["P_data"], dla_core=c["dla_core"],
        dla_shot_flag=c["dla_shot_flag"], valid_k=c["valid_k"],
        cemu_inflate=cemu_inflate, include_logdet=include_logdet)


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


def test_dlogL_dtau0_matches_finite_diff():
    """Autodiff ∂logL/∂τ₀ (full, logdet-bearing) matches central FD at non-zero residual."""
    c = _ctx(); f_full = _f(c, include_logdet=True)
    t0 = 0.40
    g_full = float(jax.grad(lambda t: f_full(c["theta9"], t, c["alpha_hcd"]))(t0))
    ff = lambda t: float(f_full(c["theta9"], t, c["alpha_hcd"]))
    fd = (ff(t0 + 1e-4) - ff(t0 - 1e-4)) / 2e-4
    assert np.isclose(g_full, fd, rtol=1e-4), f"grad {g_full} vs FD {fd}"


def test_logdet_is_load_bearing_at_zero_residual():
    """Scale-independent negative control: at r=0 the chi2 gradient vanishes, so any
    ∂logL/∂τ₀ is PURELY the logdet term. The full driver must have a non-zero τ₀
    gradient there; the chi2-only (no-logdet) driver must be ≈0."""
    c = _ctx()
    t0 = 0.40
    # zero-residual data: P_data = P_obs(t0) so r(t0)=0 exactly
    P_obs = np.asarray(predict_P_obs(c["model"], c["theta9"], c["z_unit"], t0,
                                     c["alpha_hcd"], c["pf"], c["dla_core"]))
    c0 = dict(c, P_data=jnp.asarray(P_obs), valid_k=jnp.ones(len(P_obs), bool))
    g_full = float(jax.grad(lambda t: _f(c0, include_logdet=True)(c0["theta9"], t, c0["alpha_hcd"]))(t0))
    g_nolog = float(jax.grad(lambda t: _f(c0, include_logdet=False)(c0["theta9"], t, c0["alpha_hcd"]))(t0))
    assert abs(g_nolog) < 1e-8, f"chi2 gradient at r=0 must vanish, got {g_nolog}"
    # the logdet drives a non-zero τ₀ gradient that DOMINATES the (vanishing) chi2 one
    assert abs(g_full) > 1e-7 and abs(g_full) > 50 * abs(g_nolog), \
        f"logdet must drive ∂logL/∂τ₀ at r=0: g_full={g_full}, g_nolog={g_nolog}"


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


def test_cemu_inflate_raises_emulator_variance_in_driver():
    """cemu_inflate scales the driver's per-class emu_var -> a more negative −½logdet C
    (the conservative-inflation knob, review I1). Compare the driver's logL with vs without
    inflation at zero residual (r=0 so only the logdet term differs)."""
    c = _ctx()
    # zero-residual data so logL = −½logdet C; inflation grows C -> logL decreases.
    P_obs = np.asarray(predict_P_obs(c["model"], c["theta9"], c["z_unit"], 0.4,
                                     c["alpha_hcd"], c["pf"], c["dla_core"]))
    c2 = dict(c, P_data=jnp.asarray(P_obs), valid_k=jnp.ones(len(P_obs), bool))
    ll1 = float(_f(c2, cemu_inflate=1.0)(c2["theta9"], 0.4, c2["alpha_hcd"]))
    ll4 = float(_f(c2, cemu_inflate=4.0)(c2["theta9"], 0.4, c2["alpha_hcd"]))
    assert ll4 < ll1, "cemu_inflate must enlarge C_emu (more negative −½logdet)"


def test_hcd_incidence_prior_centers_on_observed_not_sim():
    """The incidence prior centers on the OBSERVED incidence = (lit/sim)·w_c (not the sim
    weight), since PRIYA mis-predicts subDLA/DLA by ~30%; DLA on the 0.30× residual."""
    w = jnp.array([0.06, 0.02, 0.01])      # fiducial (LLS, subDLA, DLA) sim weights
    # with lit==sim the center is the bare sim weight (LLS/subDLA) / 0.30× (DLA)
    mu0, _ = I.hcd_incidence_prior(w, lit_over_sim=(1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(mu0), [0.06, 0.02, 0.30 * 0.01])
    # the DEFAULT offset (PRIYA over-predicts subDLA, under-predicts DLA) shifts the centers
    r = I.HCD_LIT_OVER_SIM
    mu, sig = I.hcd_incidence_prior(w)   # z = pivot = 3.0 (≤3.5 → no DLA σ inflation)
    assert np.allclose(np.asarray(mu), [r[0] * 0.06, r[1] * 0.02, 0.30 * r[2] * 0.01])
    # widths: LLS tight 0.15, subDLA BROAD 0.40 (poor measurement), DLA 0.10 at z≤3.5
    assert np.allclose(np.asarray(sig), [0.15 * mu[0], 0.40 * mu[1], 0.10 * mu[2]])
    assert float(mu[1]) < 0.02, "subDLA center below sim weight (PRIYA over-predicts)"
    assert float(mu[2]) > 0.30 * 0.01, "DLA residual center above naive 0.30× (PRIYA under-predicts)"
    g = jax.grad(lambda a: jnp.sum(I.gaussian_logprior(a, mu, sig)))(
        jnp.asarray([0.05, 0.015, 0.004]))
    assert np.isfinite(np.asarray(g)).all()
    # z-SLOPE (the τ₀-analog): centers grow toward high z; pivot recovers the base ratio.
    assert np.allclose(np.asarray(I.lit_over_sim_at_z(I.HCD_Z_PIVOT)), I.HCD_LIT_OVER_SIM)
    mu_lo, _ = I.hcd_incidence_prior(w, z=2.4)
    mu_hi, sig_hi = I.hcd_incidence_prior(w, z=4.4)
    assert float(mu_hi[0]) > float(mu_lo[0]), "LLS center must grow with z (slope>0)"
    # DLA z-slope is WEAK now (+0.40 not +1.08) but still positive
    assert float(mu_hi[2]) > float(mu_lo[2]), "DLA center grows weakly with z"
    # DLA σ WIDENS beyond z=3.5 (data, not prior, sets high-z DLA)
    assert float(sig_hi[2]) > 0.10 * float(mu_hi[2]), "DLA σ must widen above z=3.5"


def test_log_lik_multiz_equals_loop_sum():
    """CS review M2/M3: the vmap-over-z likelihood-only sum == an explicit per-z loop, and is
    differentiable in θ and the per-z τ₀ vector."""
    rng = np.random.default_rng(5)
    n_z, n_k, n_tb = 3, 12, 4
    c = _ctx(n_k=n_k, n_tb=n_tb)
    model, pf = c["model"], c["pf"]
    z = np.array([2.6, 3.0, 3.6]); z_unit = np.array([0.3, 0.5, 0.7])
    tau0_vec = jnp.asarray([0.30, 0.40, 0.55]); theta9 = c["theta9"]; alpha = c["alpha_hcd"]
    ac = c["alpha_centres"]
    sigma_zb = jnp.asarray(rng.uniform(0.01, 0.05, (n_z, 4, n_k, n_tb)))
    cosmic = jnp.asarray(rng.uniform(1.0, 4.0, (n_z, n_k)))
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, (n_z, n_k)))
    flag = jnp.zeros((n_z, n_k), bool); valid_k = jnp.ones((n_z, n_k), bool)
    P_data = jnp.asarray(np.stack([
        np.asarray(predict_P_obs(model, theta9, float(z_unit[i]), float(tau0_vec[i]), alpha,
                                 pf, dla_core[i])) + rng.normal(0, 0.3, n_k) for i in range(n_z)]))
    kw = dict(pf_stats=pf, z=z, z_unit=z_unit, sigma_zb=sigma_zb, alpha_centres=ac,
              cosmic_cov=cosmic, P_data=P_data, dla_core=dla_core, dla_shot_flag=flag, valid_k=valid_k)
    total = float(I.log_lik_multiz(model, theta9, tau0_vec, alpha, **kw))
    loop = sum(float(I.log_lik_single_z(
        model, theta9, float(z_unit[i]), float(z[i]), float(tau0_vec[i]), alpha, pf_stats=pf,
        sigma_zb=sigma_zb[i], alpha_centres=ac, cosmic_cov=cosmic[i], P_data=P_data[i],
        dla_core=dla_core[i], dla_shot_flag=flag[i], valid_k=valid_k[i])) for i in range(n_z))
    assert np.isclose(total, loop, rtol=1e-10), f"vmap {total} vs loop {loop}"
    g = jax.grad(lambda th: I.log_lik_multiz(model, th, tau0_vec, alpha, **kw))(theta9)
    gt = jax.grad(lambda t: I.log_lik_multiz(model, theta9, t, alpha, **kw))(tau0_vec)
    assert np.isfinite(np.asarray(g)).all() and np.isfinite(np.asarray(gt)).all()


def test_unit_box_logprior_zero_inside_finite_grad_outside():
    assert float(I.unit_box_logprior(jnp.full(9, 0.5))) == 0.0
    out = jnp.array([0.5] * 8 + [1.3])
    assert float(I.unit_box_logprior(out)) < 0.0
    g = jax.grad(I.unit_box_logprior)(out)
    assert np.isfinite(np.asarray(g)).all() and float(g[8]) < 0.0
