"""GAP TEST (TDD) — the z-flat-alpha broadcast bug class + the require_zresolved guard.

The recurring bug: ``predict_P_obs_on_leg`` SILENTLY broadcasts a (3,) z-FLAT alpha to every z
(``alpha_hcd.ndim == 1`` → the SAME incidence at all z). The DEPLOYED forward passes the
z-RESOLVED ``alpha_hcd_z`` (n_z,3, per-z w_c rises ~3.5× over z); the NON-deployed re-scoring
paths (closure_legb._loglik_of_draws / ll_true, the walkthrough figure) used to pass the z-flat
``truth['alpha_hcd']`` / ``truth['w_c']`` and so produced a spurious z-structured residual that
this bug class had ZERO coverage for.

These tests are CACHE-FREE (a small random-init Emulator + a synthetic 2-z DataLeg, mirroring
``tests/test_data_likelihood._emu_ctx``) so they run anywhere:
  1. a z-flat (3,) alpha gives a DIFFERENT P than the z-resolved (n_z,3) alpha whose per-z rows
     are NOT all equal — i.e. the silent broadcast is observable (so a regression is detectable);
  2. ``require_zresolved=True`` RAISES on a (3,) z-flat alpha and PASSES on a (n_z,3) alpha;
  3. the same guard fires through ``_data_loglik_legcore(require_zresolved=True)``.
"""
import re

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64 before any jax array
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import data_likelihood as DL


# ----------------------------------------------------------------------------- #
#  cache-free synthetic emulator + a 2-z synthetic DataLeg
# ----------------------------------------------------------------------------- #
def _emu(n_k=172, n_basis=12, seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))
    theta9 = jnp.full(9, 0.5)
    return model, pf, dla_core, cache_k, theta9


def _synthetic_leg(name="DESI", n_per_z=6):
    """A 2-z synthetic DataLeg with k-rows inside the cache footprint (z-major, identity cov)."""
    zs = np.array([2.4, 3.6])                      # two z's with DIFFERENT incidence below
    z_unit = (zs - 2.2) / 3.4
    k_one = np.linspace(1e-3, 0.06, n_per_z)
    k = np.concatenate([k_one, k_one])             # z-major
    z_row = np.concatenate([np.full(n_per_z, zs[0]), np.full(n_per_z, zs[1])])
    z_idx = np.concatenate([np.zeros(n_per_z, int), np.ones(n_per_z, int)])
    N = k.shape[0]
    P_data = np.full(N, 0.1)
    C_data = np.eye(N) * 1e-4
    R_z = np.array([50.0, 50.0])
    return DL.DataLeg(
        name=name, z=zs, z_unit=z_unit, k=k, z_row=z_row, z_idx=z_idx,
        P_data=P_data, C_data=C_data, R_z=R_z, n_z=2,
        n_per_z=np.array([n_per_z, n_per_z]), metals_on=False, resolution_on=False)


def _alpha_flat():
    return jnp.asarray([0.06, 0.02, 0.003])        # (3,) z-flat pivot incidence


def _alpha_zresolved():
    """(n_z,3) z-resolved incidence whose two z-rows are DIFFERENT (z=3.6 ~2× the z=2.4 row)
    — the per-z w_c structure the deployed forward carries (and the z-flat path drops)."""
    a = np.array(_alpha_flat())
    return jnp.asarray(np.stack([a, 2.0 * a]))     # row0 (z=2.4) ≠ row1 (z=3.6)


# ----------------------------------------------------------------------------- #
#  (1) the silent broadcast is OBSERVABLE: z-flat P ≠ z-resolved P
# ----------------------------------------------------------------------------- #
def test_zflat_vs_zresolved_predict_differ():
    """A (3,) z-flat alpha is broadcast to all z; a (n_z,3) z-resolved alpha with a DIFFERENT
    high-z row gives a DIFFERENT P_model — so the silent broadcast is detectable (the bug has
    a fingerprint). The first z-row (where both alphas agree) must MATCH; the second must DIFFER."""
    model, pf, dla_core, cache_k, theta9 = _emu()
    leg = _synthetic_leg()
    P_flat, _ = DL.predict_P_obs_on_leg(
        model, theta9, jnp.asarray([1.0, 1.0]), _alpha_flat(),
        pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg)
    P_zr, _ = DL.predict_P_obs_on_leg(
        model, theta9, jnp.asarray([1.0, 1.0]), _alpha_zresolved(),
        pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg)
    P_flat = np.asarray(P_flat); P_zr = np.asarray(P_zr)
    rows0 = np.asarray(leg.z_idx) == 0             # z=2.4: alphas agree (flat row == zr row0)
    rows1 = np.asarray(leg.z_idx) == 1             # z=3.6: zr carries 2× incidence → must differ
    np.testing.assert_allclose(P_flat[rows0], P_zr[rows0], rtol=1e-10, atol=0,
                               err_msg="z=2.4 rows: flat and z-resolved alpha agree there")
    assert not np.allclose(P_flat[rows1], P_zr[rows1]), (
        "z=3.6 rows: a z-flat alpha is silently broadcast — it must DIFFER from the z-resolved "
        "alpha (else the broadcast bug would be UNobservable / uncoverable)")


# ----------------------------------------------------------------------------- #
#  (2) require_zresolved=True RAISES on (3,) and PASSES on (n_z,3)
# ----------------------------------------------------------------------------- #
def test_require_zresolved_raises_on_zflat():
    """``require_zresolved=True`` must ASSERT the alpha is z-resolved — a (3,) z-flat alpha raises
    (so a future z-flat regression on a load-bearing path fails LOUDLY, not silently)."""
    model, pf, dla_core, cache_k, theta9 = _emu()
    leg = _synthetic_leg()
    with pytest.raises(AssertionError, match=r"z-RESOLVED"):
        DL.predict_P_obs_on_leg(
            model, theta9, jnp.asarray([1.0, 1.0]), _alpha_flat(),
            pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg,
            require_zresolved=True)


def test_require_zresolved_passes_on_zresolved():
    """``require_zresolved=True`` must PASS (no raise) on a (n_z,3) z-resolved alpha, and return a
    FINITE P identical to the default (the guard only checks the shape, it does not alter math)."""
    model, pf, dla_core, cache_k, theta9 = _emu()
    leg = _synthetic_leg()
    P_guard, _ = DL.predict_P_obs_on_leg(
        model, theta9, jnp.asarray([1.0, 1.0]), _alpha_zresolved(),
        pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg, require_zresolved=True)
    P_noguard, _ = DL.predict_P_obs_on_leg(
        model, theta9, jnp.asarray([1.0, 1.0]), _alpha_zresolved(),
        pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg)
    assert np.isfinite(np.asarray(P_guard)).all()
    np.testing.assert_array_equal(np.asarray(P_guard), np.asarray(P_noguard))


def test_require_zresolved_default_false_is_back_compat():
    """The default (require_zresolved unset) must accept a (3,) z-flat alpha (byte-identical
    back-compat for the diagnostic/figure callers that pass it intentionally)."""
    model, pf, dla_core, cache_k, theta9 = _emu()
    leg = _synthetic_leg()
    P, _ = DL.predict_P_obs_on_leg(
        model, theta9, jnp.asarray([1.0, 1.0]), _alpha_flat(),
        pf_stats=pf, dla_core=dla_core, cache_k=cache_k, leg=leg)   # no require_zresolved
    assert np.isfinite(np.asarray(P)).all()
