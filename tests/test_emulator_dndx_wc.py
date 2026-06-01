import jax
jax.config.update("jax_enable_x64", True)  # exact sum-to-1 (1e-12); emulator path is float64

import numpy as np, jax.numpy as jnp
from hcd_analysis.emulator.dndx_wc import w_c_from_mu


def test_wc_sums_to_one_and_matches_paper_point():
    mu = jnp.array([0.231, 0.064, 0.032])           # (LLS, subDLA, DLA)
    w = w_c_from_mu(mu)                              # (clean, LLS, subDLA, DLA)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert np.allclose(np.array(w), [0.72121, 0.18715, 0.06027, 0.03136], atol=5e-4)


def test_wc_batched_shape_and_sum():
    mu = jnp.array([[0.231, 0.064, 0.032],
                    [0.10, 0.05, 0.01],
                    [0.0, 0.0, 0.0]])               # (3, 3)
    w = w_c_from_mu(mu)                              # (3, 4)
    assert w.shape == (3, 4)
    assert np.allclose(np.array(w.sum(axis=-1)), 1.0, atol=1e-12)
    # all-zero mu -> fully clean
    assert np.allclose(np.array(w[2]), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


from hcd_analysis.emulator.dndx_wc import w_c_corrected, delta_c


def test_delta_c_correction_keeps_sum_to_one_and_is_differentiable():
    dndx = jnp.array([0.36, 0.10, 0.05]); Xbar = jnp.array(0.642); z = jnp.array(3.0)
    w = w_c_corrected(dndx, Xbar, z)
    assert w.shape == (4,)
    assert abs(float(w.sum()) - 1.0) < 1e-10            # renormalised after correction
    g = jax.grad(lambda d: w_c_corrected(d, Xbar, z)[3])(dndx)   # d w_DLA / d dndx
    assert jnp.all(jnp.isfinite(g))


def test_delta_c_polynomial_values():
    # delta_c(z) is deg-2 per class (clean,LLS,subDLA,DLA), np.polyval order coeffs.
    import numpy as np
    z = 3.0
    out = np.array(delta_c(jnp.array(z)))
    assert out.shape == (4,)
    # clean coeff [-0.00446, 0.02941, -0.04148] at z=3 -> small (~ -0.0% to +1%)
    assert np.all(np.abs(out) < 0.05)
