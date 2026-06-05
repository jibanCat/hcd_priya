"""Phase-C T4a — the per-z mean-flux τ₀ prior (μ_z, σ_z) for the closure sampler.

The τ_eff–cosmology degeneracy biases cosmology unless τ₀(z) is anchored on the
observed mean flux (design §4); this module is the PRODUCER of that anchor. For the MVP
the centre μ_z is the **Kim2013 curve** (``KIM_AMP·(1+z)^KIM_SLOPE``), the SAME curve the
cache's τ₀ ladder is built on — so the prior is matched to the cache by construction
(α = τ₀/Kim(z) = 1 at the centre, the ladder's interior). The width is a fractional
fraction of μ_z (the measurement width); ``flat=True`` gives a genuinely-wide-but-finite
prior for the ∞-prior scan (NOT σ=1e30 — that would make the Normal ill-conditioned and
the ladder coord σ/Kim numerically huge; we use a wide MULTIPLE of the nominal width).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .data import KIM_AMP, KIM_SLOPE

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"

# Nominal fractional width of the mean-flux prior, δτ₀/τ₀. The Kim2013 / Becker19 ⟨F⟩(z)
# is measured to a few-percent; 0.05 is a conservative measurement-width MVP value (the
# production value should track the actual ⟨F⟩(z) error budget — see LYA-CONSULT below).
DEFAULT_FRAC_SIGMA = 0.05
# ``flat=True`` multiplier on the nominal width: a wide-but-FINITE prior for the
# ∞-prior robustness scan. 20× the nominal fractional width (→ ~σ/μ = 1.0, i.e. a τ₀
# prior that spans the full physical range without going singular). Documented choice;
# NOT σ=1e30 (which makes σ_ladder = σ_τ₀/Kim(z) ~ 1e30·few, an ill-conditioned Normal
# whose log_prob underflows and whose NUTS mass matrix degenerates).
FLAT_SIGMA_MULT = 20.0


def kim_tau0(z):
    """Kim2013 central mean-flux τ₀(z) = KIM_AMP·(1+z)^KIM_SLOPE (the cache ladder anchor)."""
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


def meanflux_tau0_prior(z, *, frac_sigma=DEFAULT_FRAC_SIGMA, flat=False):
    """Per-z mean-flux prior ``(tau0_mu, tau0_sigma)`` in τ₀ units.

    μ_z = Kim2013 ⟨τ_eff⟩(z) (the MVP centre — matched to the cache τ₀ ladder);
    σ_z = frac_sigma·μ_z (the measurement width). ``flat=True`` widens σ_z by
    ``FLAT_SIGMA_MULT`` to a wide-but-finite prior for the ∞-prior scan.

    Args:
      z:          (n_z,) or scalar physical redshift.
      frac_sigma: δτ₀/τ₀ fractional prior width (default 0.05).
      flat:       if True, return the wide-but-finite ∞-prior-scan width.
    Returns: ``(tau0_mu (n_z,), tau0_sigma (n_z,))`` in τ₀ units.

    LYA-CONSULT: the PRODUCTION μ_z must be the OBSERVED mean-flux optical depth
    ⟨τ_eff⟩(z) = −ln⟨F⟩(z) **in the data's DLA-masked / metal-corrected state** (the
    Becker+2019 / Turner+2024 / Kim+2013 choice the data pipeline actually uses, incl.
    which metal/DLA treatment ⟨F⟩(z) was measured under). Using the Kim2013 curve here is
    a deliberate MVP so the prior is matched to the cache ladder (α=1 at centre); it is
    NOT a claim that Kim2013 is the right observational anchor. The Lyα agent must pick
    the production ⟨F⟩(z) (and its z-dependent error budget for ``frac_sigma``); do not
    ship the cosmology result on the Kim MVP without that sign-off.
    """
    z = jnp.asarray(z)
    mu = kim_tau0(z)
    sigma = frac_sigma * mu
    if flat:
        sigma = sigma * FLAT_SIGMA_MULT
    return mu, sigma
