"""PRIYA's mean-flux model as a conversion, with no prior: the Kim07 curve the training cache's tau0 ladder is built on,
and the two-parameter PRIYA model (Bird et al. 2023, section 2.7.1) in the ladder coordinate. The public predictor
imports these; ``meanflux_prior`` (the historical analyses' mean-flux priors) re-exports them unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

from .data import KIM_AMP, KIM_SLOPE

TAU0_PIVOT_Z = 3.0  # the (1+z)/(1+z_p) pivot (PRIYA (1+z)/4, so z_p = 3)


def kim_tau0(z):
    """Kim07 central mean-flux tau0(z) = KIM_AMP (1+z)^KIM_SLOPE (the cache ladder anchor)."""
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


def tau0_alpha_priya(z, tau0_amp, dtau0, *, z_pivot=TAU0_PIVOT_Z):
    """PRIYA mean-flux model in the alpha = tau0 / Kim ladder coordinate (Bird et al. 2023 section 2.7.1, eqs 2.14-15:
    tau_eff = tau0 ((1+z)/4)^dtau0 tau_Kim): ``alpha(z) = tau0_amp ((1+z)/(1+z_pivot))^dtau0``. ``tau0_amp=1,
    dtau0=0`` is the Kim central rung (the cache ladder anchor). Multiply by ``kim_tau0(z)`` for tau0(z).
    Differentiable in (tau0_amp, dtau0)."""
    z = jnp.asarray(z)
    return tau0_amp * ((1.0 + z) / (1.0 + z_pivot)) ** dtau0
