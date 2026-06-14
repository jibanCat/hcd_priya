"""Phase-1 (physical-tau0 plan): the smooth 2-param tau0(z) power-law builder + fitter.

The closure replaces the 13 independent per-z tau0 rungs with a physical
tau0(z) = exp(logA)*((1+z)/(1+zref))^beta + c, so tau0 cannot absorb emulator
residual into free per-z wiggle (which leaks into A_p). These tests lock the builder
+ the closure-truth fitter.
"""
import numpy as np
import hcd_analysis.emulator  # noqa: F401  (enable x64 before jax)
import jax.numpy as jnp
from hcd_analysis.emulator.meanflux_prior import (
    tau0_powerlaw, fit_tau0_powerlaw, becker13_tau0,
    BECKER13_TAU0, BECKER13_BETA, BECKER13_ZREF, BECKER13_C,
)


def test_powerlaw_matches_becker_at_its_params():
    z = jnp.linspace(2.0, 4.6, 14)
    out = tau0_powerlaw(z, jnp.log(BECKER13_TAU0), BECKER13_BETA,
                        zref=BECKER13_ZREF, c=BECKER13_C)
    np.testing.assert_allclose(np.asarray(out), np.asarray(becker13_tau0(z)), rtol=1e-12)


def test_powerlaw_fits_a_smooth_tau0_curve_to_subpercent():
    z = jnp.linspace(2.0, 4.6, 14)
    truth = 0.62 * ((1 + z) / (1 + BECKER13_ZREF)) ** 3.1 - 0.10
    logA, beta, c = fit_tau0_powerlaw(np.asarray(z), np.asarray(truth))
    rec = np.asarray(tau0_powerlaw(z, logA, beta, c=c))
    assert np.max(np.abs(rec / np.asarray(truth) - 1)) < 0.01
    assert abs(beta - 3.1) < 0.05


def test_powerlaw_differentiable_in_logA_beta():
    import jax
    z = jnp.linspace(2.2, 4.2, 8)
    g = jax.grad(lambda la, b: jnp.sum(tau0_powerlaw(z, la, b)), argnums=(0, 1))
    dlogA, dbeta = g(jnp.log(0.7), 2.9)
    assert np.isfinite(float(dlogA)) and np.isfinite(float(dbeta))
    assert float(dlogA) > 0  # tau0 increases with amplitude


def test_priya_alpha_central_rung_is_unity():
    # tau0_amp=1, dtau0=0 -> alpha(z)=1 (the Kim central ladder rung) at all z
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya
    z = jnp.linspace(2.0, 4.6, 14)
    np.testing.assert_allclose(np.asarray(tau0_alpha_priya(z, 1.0, 0.0)), 1.0, rtol=1e-12)


def test_priya_alpha_fit_roundtrip():
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya, fit_tau0_alpha_priya
    z = jnp.linspace(2.0, 4.6, 14)
    amp_t, d_t = 1.07, 0.22
    alpha = np.asarray(tau0_alpha_priya(z, amp_t, d_t))
    amp, d = fit_tau0_alpha_priya(np.asarray(z), alpha)
    assert abs(amp - amp_t) < 1e-6 and abs(d - d_t) < 1e-6


def test_priya_alpha_differentiable():
    import jax
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya
    z = jnp.linspace(2.2, 4.2, 8)
    g = jax.grad(lambda amp, d: jnp.sum(tau0_alpha_priya(z, amp, d)), argnums=(0, 1))
    damp, dd = g(1.0, 0.0)
    assert np.isfinite(float(damp)) and np.isfinite(float(dd)) and float(damp) > 0
