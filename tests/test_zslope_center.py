"""Regression guard for the FORWARD HCD z-slope center (dN/dX wrong-slope fix).

The forward incidence weight is α_c(z) = α_pivot · ((1+z)/(1+z_p))^s_c, where s_c is the
per-class z-slope returned by ``closure_legb._zslope_sites`` on the NON-2D path (fixed OR
marginalize_zslope; the 2D-tilt path is a different site and is covered by test_legb_2d_tilt).

s_c MUST be centered on the SIM incidence-weight slope ``HCD_INCIDENCE_SLOPE`` (≈2.4 — the
slope the held-out-sim mock TRUTH's native w_c(z) carries), NOT on the lit/sim RATIO slope
``HCD_LIT_OVER_SIM_SLOPE`` (≈0.95, a DIFFERENT object: d ln(lit/sim)/d ln(1+z), evaluated at
the z=3 pivot only). Centering on 0.95 makes the predicted dN/dX(z) FALL with z (truth rises)
and puts the mock-truth slope 2.9–6σ off-center — the bug.

These tests are lightweight (SimpleNamespace ctx; no cache/ckpt/data) — they trace ONLY the
HCD z-slope sample sites of ``_zslope_sites``.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_zslope_center.py -q
"""
import numpy as np
import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.inference import HCD_LIT_OVER_SIM_SLOPE


def _fake_ctx(marginalize_zslope=False, zslope_mu=None, zslope_sigma=None):
    return SimpleNamespace(
        marginalize_zslope=marginalize_zslope,
        zslope_mu=zslope_mu,
        zslope_sigma=zslope_sigma,
    )


def test_fixed_branch_returns_incidence_slope_not_ratio_slope():
    """marginalize_zslope=False → _zslope_sites returns the SIM incidence slope
    HCD_INCIDENCE_SLOPE (the mock-truth w_c(z) slope), NOT the lit/sim ratio HCD_LIT_OVER_SIM_SLOPE."""
    ctx = _fake_ctx(marginalize_zslope=False)
    s_c = np.asarray(CL._zslope_sites(ctx))
    incidence = np.asarray(CL.HCD_INCIDENCE_SLOPE)
    ratio = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    np.testing.assert_allclose(s_c, incidence, rtol=0, atol=0,
                               err_msg=f"fixed forward s_c {s_c} must be the incidence slope "
                                       f"{incidence}, not the lit/sim ratio {ratio}")
    # and it must NOT be the ratio slope (guard against the bug recurring).
    assert not np.allclose(s_c, ratio), \
        f"fixed forward s_c is the lit/sim RATIO slope {ratio} — the wrong-object bug"


def test_marginalized_default_center_is_incidence_slope():
    """marginalize_zslope=True, zslope_mu=None → the Normal prior CENTER (loc) of the s_lls/
    s_subdla/s_dla sites is HCD_INCIDENCE_SLOPE, NOT HCD_LIT_OVER_SIM_SLOPE."""
    ctx = _fake_ctx(marginalize_zslope=True, zslope_mu=None)
    tr = handlers.trace(handlers.seed(
        lambda: CL._zslope_sites(ctx), jax.random.PRNGKey(0))).get_trace()
    incidence = np.asarray(CL.HCD_INCIDENCE_SLOPE)
    ratio = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    locs = np.array([float(tr[nm]["fn"].loc) for nm in ("s_lls", "s_subdla", "s_dla")])
    np.testing.assert_allclose(locs, incidence, rtol=0, atol=0,
                               err_msg=f"marginalized None-default center {locs} must be the "
                                       f"incidence slope {incidence}, not the ratio {ratio}")
    assert not np.allclose(locs, ratio), \
        f"marginalized None-default center is the lit/sim RATIO slope {ratio} — the bug"


def test_marginalized_explicit_mu_is_respected():
    """An explicit ctx.zslope_mu still overrides (the fix only changes the None-DEFAULT center)."""
    custom = jnp.asarray([1.1, 1.2, 1.3])
    ctx = _fake_ctx(marginalize_zslope=True, zslope_mu=custom)
    tr = handlers.trace(handlers.seed(
        lambda: CL._zslope_sites(ctx), jax.random.PRNGKey(0))).get_trace()
    locs = np.array([float(tr[nm]["fn"].loc) for nm in ("s_lls", "s_subdla", "s_dla")])
    np.testing.assert_allclose(locs, np.asarray(custom), rtol=0, atol=0)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
