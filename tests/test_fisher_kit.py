"""Gate E amendment A1 rev 1 section 0: the linearized (Fisher) projection kit. The parameter map equals the production
numpyro model's (tau0_vec, alpha_hcd_z) on the default and the KS dN/dX-mapped branches; J equals finite differences of
the leg mean; the projected shift is the linearized MAP; prior precisions are the registered ones. Cache-free."""
import hcd_analysis.emulator  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import fisher_kit as FK
from tests.test_forward_ctx_v2 import _ctx
from tests.test_forward_v2 import _emu, _leg


def _synthetic(ks=False):
    model, pf, _ = _emu(30)
    zg = np.array([2.4, 3.2, 4.0])
    legs = [_leg([2.4, 3.2], name="DESI"), _leg([3.2, 4.0], name="KS", dff=0.0)]
    ctx = _ctx(model, pf, legs, zg)._replace(fix_alpha_res=True, metal_prior="uniform",
                                             dla_core_leg={l.name: jnp.zeros(l.k.size) for l in legs})
    if ks:
        rng = np.random.default_rng(5)
        ctx = ctx._replace(ks_dndx_mapped=True, ks_dndx_ref=jnp.asarray(rng.uniform(0.05, 0.3, (3, 3))),
                           ks_xbar_z=jnp.asarray(rng.uniform(3.0, 4.0, 3)),
                           ks_dndx_ref_pivot=jnp.asarray(rng.uniform(0.05, 0.3, 3)), ks_xbar_pivot=3.5)
    return ctx


@pytest.mark.parametrize("ks", [False, True], ids=["default", "ks_mapped"])
def test_parameter_map_equals_the_production_model(ks):
    ctx = _synthetic(ks)
    p = FK.p_centre(ctx, np.full(9, 0.4)) + np.r_[np.zeros(9), 0.05, -0.2, 0.01, -0.01, 0.1]
    names = FK.param_names(ctx)
    sub = {"theta_unit": jnp.asarray(p[:9]), "tau0_amp": p[9], "dtau0": p[10]}
    sub.update({n: p[11 + i] for i, n in enumerate(names[11:])})
    if ks:
        sub.update(kappa_lls=0.0, t_sub=0.0, t_dla=0.0)
    else:                                    # marginalize_zslope (production default): slopes at the centre
        s = np.asarray(CL.HCD_INCIDENCE_SLOPE if ctx.zslope_mu is None else ctx.zslope_mu)
        sub.update(s_lls=s[0], s_subdla=s[1], s_dla=s[2])
    model = lambda: CL._legb_model(ctx, ctx.legs, ctx.dla_core_leg)
    tr = numpyro.handlers.trace(numpyro.handlers.substitute(numpyro.handlers.seed(model, 0), data=sub)).get_trace()
    th, tau0, alpha = FK.forward_inputs(ctx, jnp.asarray(p))
    np.testing.assert_allclose(np.asarray(tau0), np.asarray(tr["tau0_vec"]["value"]), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(alpha), np.asarray(tr["alpha_hcd_z"]["value"]), rtol=1e-13)


def test_jacobian_equals_finite_differences_of_the_leg_mean():
    ctx = _synthetic()
    leg = ctx.legs[0]
    p0 = FK.p_centre(ctx, np.full(9, 0.45))
    f = FK.mean_fn(ctx, leg)
    J = np.asarray(FK.jacobian(ctx, leg, p0))
    for i in (0, 1, 5, 9, 10, 11, 13):
        h = 1e-5
        e = np.zeros_like(p0); e[i] = h
        fd = (np.asarray(f(jnp.asarray(p0 + e))) - np.asarray(f(jnp.asarray(p0 - e)))) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-4, atol=1e-5 * np.abs(fd).max())   # FD round-off level


def test_prior_precision_is_the_registered_one():
    ctx = _synthetic()
    P = np.asarray(FK.prior_precision(ctx))
    lo, hi = CL._THETA_UNIT_LO, CL._THETA_UNIT_HI
    np.testing.assert_allclose(np.diag(P)[:9], 12.0 / (np.asarray(hi) - np.asarray(lo)) ** 2, rtol=1e-14)
    assert np.diag(P)[9] == 0 and np.diag(P)[10] == 0
    np.testing.assert_allclose(np.diag(P)[11:13], 1.0 / np.asarray(ctx.alpha_hcd_sigma)[:2] ** 2, rtol=1e-14)
    assert np.diag(P)[13] == 1.0


def test_projected_shift_is_the_linearized_map():
    rng = np.random.default_rng(1)
    N, n = 40, 6
    J = rng.normal(0, 1, (N, n)); A = rng.normal(0, 0.1, (N, N)); C = A @ A.T + np.eye(N)
    P = np.diag(rng.uniform(0, 1, n)); r = rng.normal(0, 1, N)
    d = FK.map_shift(J, C, P, r)
    Li = np.linalg.cholesky(np.linalg.inv(C))
    Ls = np.linalg.cholesky(P + 1e-300 * np.eye(n)) if np.all(np.diag(P) > 0) else np.sqrt(P)
    stack = np.vstack([Li.T @ J, Ls.T]); rhs = np.concatenate([Li.T @ r, np.zeros(n)])
    np.testing.assert_allclose(d, np.linalg.lstsq(stack, rhs, rcond=None)[0], rtol=1e-10)
    F = FK.fisher(J, C, P)
    np.testing.assert_allclose(FK.marginal_sigma(F), np.sqrt(np.diag(np.linalg.inv(F))), rtol=1e-12)
