"""Phase-1: the Leg-B model now samples the PRIYA 2-param τ₀ (tau0_amp, dtau0) instead of
the 13 independent per-z alpha_ladder rungs. Lock the prior sites + the tau0_vec reconstruction."""
import numpy as np
import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from numpyro import handlers
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.meanflux_prior import (
    kim_tau0, TAU0_AMP_RANGE, DTAU0_RANGE, TAU0_PIVOT_Z)


def _fake_ctx():
    return SimpleNamespace(
        z_global=np.linspace(2.2, 4.6, 13),
        tau0_amp_range=TAU0_AMP_RANGE, dtau0_range=DTAU0_RANGE, tau0_pivot_z=TAU0_PIVOT_Z,
        alpha_hcd_mu=jnp.array([0.29, 0.069, 0.003]),
        alpha_hcd_sigma=jnp.array([0.044, 0.028, 0.0015]),
        marginalize_zslope=False, zslope_mu=None, zslope_sigma=None,
    )


def test_priors_only_has_priya_tau0_sites_not_alpha_ladder():
    tr = handlers.trace(handlers.seed(
        lambda: CL._legb_priors_only(_fake_ctx()), jax.random.PRNGKey(0))).get_trace()
    assert "tau0_amp" in tr and "dtau0" in tr
    assert "alpha_ladder" not in tr            # the 13-rung site is gone
    assert float(tr["tau0_amp"]["fn"].low) == TAU0_AMP_RANGE[0]
    assert float(tr["tau0_amp"]["fn"].high) == TAU0_AMP_RANGE[1]
    assert float(tr["dtau0"]["fn"].low) == DTAU0_RANGE[0]
    assert float(tr["dtau0"]["fn"].high) == DTAU0_RANGE[1]


def test_reconstruct_tau0_vec_from_amp_dtau0():
    ctx = _fake_ctx()
    zg = jnp.asarray(ctx.z_global); L = 5
    amp = jnp.linspace(0.8, 1.2, L); dt = jnp.linspace(-0.3, 0.2, L)
    samples = dict(tau0_amp=amp, dtau0=dt, alpha_lls=jnp.zeros(L),
                   alpha_subdla=jnp.zeros(L), alpha_dla_raw=jnp.zeros(L))
    out = CL._legb_reconstruct_deterministics(ctx, samples)
    exp = np.array([
        float(amp[i]) * ((1 + np.asarray(zg)) / (1 + TAU0_PIVOT_Z)) ** float(dt[i])
        * np.asarray(kim_tau0(zg)) for i in range(L)])
    np.testing.assert_allclose(np.asarray(out["tau0_vec"]), exp, rtol=1e-10)
    assert out["tau0_vec"].shape == (L, len(zg))
