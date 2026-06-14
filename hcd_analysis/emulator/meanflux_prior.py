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


# Becker+2013 (arXiv:1208.2584) τ_eff(z) — the DLA-MASKED + metal-corrected effective-
# optical-depth curve, the field-standard ⟨F⟩(z) anchor the DESI/KS pipelines use:
#   τ_eff(z) = BECKER13_TAU0·((1+z)/(1+BECKER13_ZREF))^BECKER13_BETA + BECKER13_C
# (the +C "metal/continuum offset" term of Becker+2013 Eq. 6; C = −0.132).
BECKER13_TAU0 = 0.751
BECKER13_BETA = 2.90
BECKER13_C = -0.132
BECKER13_ZREF = 3.5     # the pivot (1+z)/(1+3.5) = (1+z)/4.5 in Becker+2013 Eq. 6


def kim_tau0(z):
    """Kim2013 central mean-flux τ₀(z) = KIM_AMP·(1+z)^KIM_SLOPE (the cache ladder anchor)."""
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


def becker13_tau0(z):
    """Becker+2013 (1208.2584 Eq. 6) τ_eff(z) = 0.751·((1+z)/4.5)^2.90 − 0.132.

    The DLA-masked + metal-corrected effective-optical-depth curve — the observational
    ⟨F⟩(z) anchor (production ``center="becker13"`` choice). Differentiable in z."""
    z = jnp.asarray(z)
    return BECKER13_TAU0 * ((1.0 + z) / (1.0 + BECKER13_ZREF)) ** BECKER13_BETA + BECKER13_C


# PRIYA-NATIVE mean-flux model (Bird+2023 §2.7.1 Eq.2.14-15; the SAME form + priors in the
# eBOSS fit Fernandez+2024 arXiv:2309.03943 AND the KODIAQ-SQUAD fit arXiv:2509.18271):
#   τ_eff(z) = τ₀·((1+z)/4)^dτ₀·τ_Kim(z),  τ_Kim = 0.0023(1+z)^3.65 (Kim07), pivot z=3, NO +C.
# 2 GLOBAL params (amplitude τ₀, slope dτ₀), UNIFORM priors (NOT a tight Gaussian — arXiv:2509.18271
# §5: "τ₀ absorbs residual HCD contamination at high k … should not be placed with a Gaussian prior
# unless co-varied with a HCD a_HCD"). The degeneracy-break is the SMOOTH 2-param structure (no
# per-z wiggle) + co-varying HCD, not a tight prior. These REPLACE the 13 independent per-z rungs.
TAU0_AMP_RANGE = (0.75, 1.25)    # PRIYA uniform prior on amplitude τ₀ (×, center 1.0 = Kim07)
DTAU0_RANGE = (-0.40, 0.25)      # PRIYA uniform prior on slope dτ₀ (center 0 = Kim07 slope)
TAU0_PIVOT_Z = 3.0               # the (1+z)/(1+z_p) pivot (PRIYA (1+z)/4 → z_p=3)
# (legacy τ₀-space power-law-with-C widths, kept for the Becker-anchored utility path only:)
TAU0_LOGA_SIGMA = 0.05
TAU0_BETA_SIGMA = 0.15


def tau0_alpha_priya(z, tau0_amp, dtau0, *, z_pivot=TAU0_PIVOT_Z):
    """PRIYA mean-flux model in the α = τ₀/Kim LADDER coordinate (Bird+2023 §2.7.1, Eq.2.14–15:
    τ_eff = τ₀·((1+z)/4)^dτ₀·τ_Kim): ``α(z) = tau0_amp · ((1+z)/(1+z_pivot))^dtau0``.

    Replaces the 13 INDEPENDENT per-z α rungs with this 2-parameter smooth curve, so τ₀ evolves
    on a physically-informed amplitude+slope (PRIYA-native) instead of a free per-z wiggle that
    biases A_p. ``tau0_amp=1, dtau0=0`` = the Kim central rung (the cache ladder anchor).
    Multiply by ``kim_tau0(z)`` for τ₀(z). Differentiable in (tau0_amp, dtau0) for NUTS."""
    z = jnp.asarray(z)
    return tau0_amp * ((1.0 + z) / (1.0 + z_pivot)) ** dtau0


def fit_tau0_alpha_priya(z, alpha_z, *, z_pivot=TAU0_PIVOT_Z):
    """LSQ (tau0_amp, dtau0) for a sim's per-z α(z)=τ₀/Kim — the CLOSURE truth params (and the
    real-fit prior center when fed Becker/Kim). Linear fit of log α vs log((1+z)/(1+z_pivot))."""
    import numpy as _np
    z = _np.asarray(z, float); a = _np.asarray(alpha_z, float)
    x = _np.log((1.0 + z) / (1.0 + z_pivot))
    b, c = _np.polyfit(x, _np.log(a), 1)       # slope b = dtau0, intercept c = log(tau0_amp)
    return float(_np.exp(c)), float(b)


def tau0_powerlaw(z, logA, beta, *, zref=BECKER13_ZREF, c=0.0):
    """Smooth physical τ₀(z) = exp(logA)·((1+z)/(1+zref))^beta + c (Becker-form power-law).

    The 2-parameter (amplitude logA + slope beta) mean-flux model that REPLACES the 13
    independent per-z τ₀ rungs in the closure — so τ₀ evolves on a physically-informed
    amplitude+slope, not a free per-z wiggle that A_p trades against. Differentiable in
    (logA, beta) for NUTS; ``c`` is the fixed Becker continuum/metal offset (or 0)."""
    z = jnp.asarray(z)
    return jnp.exp(logA) * ((1.0 + z) / (1.0 + zref)) ** beta + c


def fit_tau0_powerlaw(z, tau0, *, zref=BECKER13_ZREF):
    """Least-squares (logA, beta, c) for a sim's per-z τ₀(z) — the CLOSURE truth params.

    Fit log(τ₀−c) linear in log((1+z)/(1+zref)); grid the small offset ``c`` then linfit
    (c is the Becker continuum term, small). Returns floats (logA, beta, c). Pure-numpy
    (host-side; used to set the closure mock's true τ₀ amplitude+slope, and the
    closure prior center)."""
    import numpy as _np
    z = _np.asarray(z, dtype=float); tau0 = _np.asarray(tau0, dtype=float)
    best = None
    for c in _np.linspace(-0.20, 0.05, 26):
        y = tau0 - c
        if _np.any(y <= 0):
            continue
        x = _np.log((1.0 + z) / (1.0 + zref))
        b, a = _np.polyfit(x, _np.log(y), 1)          # slope b, intercept a = logA
        resid = _np.sum((a + b * x - _np.log(y)) ** 2)
        if best is None or resid < best[0]:
            best = (resid, a, b, c)
    if best is None:
        raise ValueError("fit_tau0_powerlaw: no positive (τ₀−c) branch found")
    _, logA, beta, c = best
    return float(logA), float(beta), float(c)


def meanflux_tau0_prior(z, *, frac_sigma=DEFAULT_FRAC_SIGMA, flat=False, center="kim"):
    """Per-z mean-flux prior ``(tau0_mu, tau0_sigma)`` in τ₀ units.

    ``center="kim"`` (default, MVP): μ_z = Kim2013 ⟨τ_eff⟩(z) — matched to the cache τ₀
    ladder (α=τ₀/Kim(z)=1 at centre).  ``center="becker13"``: μ_z = Becker+2013 Eq. 6
    (the DLA-masked + metal-corrected curve the DESI/KS pipelines use) — the production
    anchor.  σ_z = frac_sigma·μ_z (the measurement width; Becker+2013 reports ~3–8%
    z-dependent — pass a per-z ``frac_sigma`` array for the production budget).
    ``flat=True`` widens σ_z by ``FLAT_SIGMA_MULT`` to a wide-but-finite prior for the
    ∞-prior scan.

    Args:
      z:          (n_z,) or scalar physical redshift.
      frac_sigma: δτ₀/τ₀ fractional prior width (default 0.05; scalar OR (n_z,) array).
      flat:       if True, return the wide-but-finite ∞-prior-scan width.
      center:     "kim" (cache-matched MVP) or "becker13" (production observed curve).
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
    if center == "becker13":
        mu = becker13_tau0(z)
    elif center == "kim":
        mu = kim_tau0(z)
    else:
        raise ValueError(f"center must be 'kim' or 'becker13', got {center!r}")
    sigma = jnp.asarray(frac_sigma) * mu
    if flat:
        sigma = sigma * FLAT_SIGMA_MULT
    return mu, sigma
