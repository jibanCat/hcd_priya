"""dN/dX -> w_c diagonal telescoping-Poisson map (M0) + JAX-pure delta_c(z).

Internal record: the w_c / dN/dX Poisson-coupling note of 2026-05-29. JAX-pure: no python
branching on traced values, so the whole dN/dX -> w_c -> P_obs path is one jit.
Class order throughout: (clean, LLS, subDLA, DLA); mu/dN-dX inputs are the 3 HCD
classes in order (LLS, subDLA, DLA).

The delta_c(z) correction is a map argument (``delta_map``, a ``DeltaCMap``): the release path passes the map refitted
on the release training cache and shipped with a model bundle (``wc_map.WcMap.delta_c_map()``). ``delta_map=None``
selects ``HISTORICAL_DELTA_C``, the coefficients fitted in 2026-06 to counts that include the pre-fix box-edge
duplicates; they remain only so the historical analyses reproduce, and the public predictor never uses them.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import numpy as np
import jax.numpy as jnp
from numpy.typing import ArrayLike


def w_c_from_mu(mu: jax.Array | np.ndarray) -> jax.Array:
    """Class fractions of sightlines from the mean number of absorbers per sightline.

    Each sightline is put in ONE class: that of its strongest absorber. If the absorbers of class c
    fall on sightlines as a Poisson process with mean ``mu_c`` per sightline, independently for
    each class, the fractions of sightlines are::

        w_DLA    = 1 - exp(-mu_DLA)                                   at least one DLA
        w_subDLA = (1 - exp(-mu_subDLA)) * exp(-mu_DLA)               a subDLA and no DLA
        w_LLS    = (1 - exp(-mu_LLS)) * exp(-mu_subDLA - mu_DLA)      an LLS and nothing stronger
        w_clean  = exp(-mu_LLS - mu_subDLA - mu_DLA)                  no absorber

    The four sum to one. ``w_c_corrected`` adds a small fitted correction to this model.

    Parameters
    ----------
    mu : numpy or JAX array
        Shape (..., 3): the mean number of absorbers per sightline for LLS, subDLA, DLA. It is the
        incidence times the mean path of a sightline, ``dndx * Xbar`` (``mu_from_dndx``).

    Returns
    -------
    jax.Array
        Shape (..., 4): the fractions in the order clean, LLS, subDLA, DLA. Differentiable.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hcd_analysis.emulator.dndx_wc import w_c_from_mu
    >>> w = w_c_from_mu(jnp.array([0.1, 0.02, 0.01]))      # synthetic means: LLS, subDLA, DLA
    >>> w.shape, round(float(w.sum()), 12)
    ((4,), 1.0)
    >>> round(float(w[3]), 6) == round(1 - float(jnp.exp(-0.01)), 6)      # the DLA fraction
    True
    """
    mu_LLS, mu_sub, mu_DLA = mu[..., 0], mu[..., 1], mu[..., 2]
    w_DLA = 1.0 - jnp.exp(-mu_DLA)
    w_sub = (1.0 - jnp.exp(-mu_sub)) * jnp.exp(-mu_DLA)
    w_LLS = (1.0 - jnp.exp(-mu_LLS)) * jnp.exp(-(mu_sub + mu_DLA))
    w_clean = jnp.exp(-(mu_LLS + mu_sub + mu_DLA))
    return jnp.stack([w_clean, w_LLS, w_sub, w_DLA], axis=-1)


def mu_from_dndx(dndx: jax.Array | np.ndarray, Xbar: jax.Array | np.ndarray) -> jax.Array:
    """Mean absorbers per sightline: ``mu_c = dndx_c * Xbar``.

    ``dndx``: shape (..., 3), the incidence dN/dX of LLS, subDLA, DLA (absorbers per unit absorption
    distance X). ``Xbar``: shape (...), the mean absorption distance of one sightline. Both are numpy or JAX
    arrays (not lists or floats). Returns (..., 3).
    """
    return dndx * Xbar[..., None]


class DeltaCMap(NamedTuple):
    """A delta_c(z) correction: ``coeffs`` (4, 3) np.polyval-order coefficients of the degree-2 polynomial per class
    (clean, LLS, subDLA, DLA), applied with z clamped to [``z_lo``, ``z_hi``] (the fit's redshift range: the quadratic
    is never extrapolated). Host values; pass it to traced code as a closed-over constant, not as a traced argument.
    """

    coeffs: np.ndarray
    z_lo: float
    z_hi: float
    label: str


# HISTORICAL delta_c(z) deg-2 coeffs from scripts/calibrate_delta_c.py (89 LF shards, 1060 snap-blocks, 60 sims),
# fitted to counts that include the pre-fix box-edge duplicates. np.polyval order [a2,a1,a0]; class order
# (clean, LLS, subDLA, DLA). See the internal note 2026-06-01-delta_c-coeffs. Used only when
# ``delta_map`` is None (the historical analyses); a release model bundle carries its own refitted map (wc_map.py).
HISTORICAL_DELTA_C = DeltaCMap(
    coeffs=np.array(
        [
            [-0.0044601756018632, 0.02941370276821661, -0.04147804040644924],  # clean
            [-7.391610723256435e-06, 0.008526578464137916, -0.03592195195833433],  # LLS
            [-0.0006304124952415818, 0.009001921148434061, -0.03672596701410354],  # subDLA
            [-0.0002204519439413697, 0.0033098470327118175, -0.0245078526302848],  # DLA
        ],
        dtype=np.float64,
    ),
    z_lo=2.0,
    z_hi=5.4,
    label="historical: fitted to the pre-fix (box-edge duplicate) counts; historical analyses only",
)
_DELTA_C_COEFFS = jnp.asarray(HISTORICAL_DELTA_C.coeffs)  # historical alias (same values)


# Calibration z-range of the HISTORICAL delta_c fit. The deg-2 polynomial extrapolates as an
# unbounded quadratic outside this; clamp z to the range so HMC / tau0-edge queries
# get the boundary value, not a divergent correction (CS finding I3).
Z_FIT_LO, Z_FIT_HI = HISTORICAL_DELTA_C.z_lo, HISTORICAL_DELTA_C.z_hi

# HISTORICAL per-class fit-residual std of delta_c (clean,LLS,subDLA,DLA), from
# the internal delta_c-coefficients note of 2026-06-01. These are the per-class w_c/alpha_c
# PRIOR WIDTHS of the historical analyses: the irreducible cosmology-dependent delta_c scatter at fixed z that the
# deg-2 z-trend fit cannot remove. Pre-fix counts; not used by the public predictor.
DELTA_C_RESID_STD = jnp.array([0.0093, 0.0059, 0.0032, 0.0016])


def _delta_map(delta_map):
    m = HISTORICAL_DELTA_C if delta_map is None else delta_map
    if np.shape(m.coeffs) != (4, 3):
        raise ValueError(
            f"delta_c map coefficients must have shape (4, 3), got {np.shape(m.coeffs)}"
        )
    return m


def delta_c(z, delta_map=None):
    """Per-class diagonal-Poisson correction delta_c(z), JAX-pure deg-2 polynomial.
    z: scalar or (...,). Returns (...,4) in class order (clean,LLS,subDLA,DLA).
    z is clamped to the map's fit range so the fit is never extrapolated. ``delta_map``: a ``DeltaCMap``; None is the
    HISTORICAL map (pre-fix counts; historical analyses only)."""
    m = _delta_map(delta_map)
    z = jnp.clip(jnp.asarray(z), m.z_lo, m.z_hi)
    zz = jnp.stack([z**2, z, jnp.ones_like(z)], axis=-1)  # (...,3)
    return zz @ jnp.asarray(m.coeffs).T  # (...,4)


def w_c_corrected(
    dndx: jax.Array | np.ndarray,
    Xbar: jax.Array | np.ndarray,
    z: ArrayLike,
    delta_map: DeltaCMap | None = None,
) -> jax.Array:
    """Class fractions of sightlines from the incidence dN/dX, with the fitted correction.

    Three steps::

        mu  = dndx * Xbar                        mean absorbers per sightline (mu_from_dndx)
        w0  = w_c_from_mu(mu)                    the Poisson model, 4 fractions
        w_c = w0 * (1 + delta_c(z)) / sum_c[ w0 * (1 + delta_c(z)) ]

    ``delta_c(z)`` is a small correction per class, a quadratic in redshift fitted to the simulations'
    own counts; the division makes the four fractions sum to one again.

    For a released model do not call this directly: use ``emu.wc_map.w_c(dndx, z)``, which passes the
    model's own ``Xbar(z)`` and ``delta_c``. Note that ``delta_map=None`` selects the HISTORICAL map,
    fitted to counts made before a catalogue fix; it is kept only so that old analyses reproduce.

    Parameters
    ----------
    dndx : numpy or JAX array
        Shape (..., 3): the incidence dN/dX of LLS, subDLA, DLA.
    Xbar : numpy or JAX array
        Shape (...): the mean absorption distance of one sightline.
    z : array_like
        Shape (...): the redshift.
    delta_map : DeltaCMap, optional
        The correction. A released model's is ``emu.wc_map.delta_c_map()``. None: the historical map.

    Returns
    -------
    jax.Array
        Shape (..., 4): the fractions in the order clean, LLS, subDLA, DLA. Differentiable.

    Examples
    --------
    With a zero correction the result is the Poisson model (synthetic numbers):

    >>> import numpy as np, jax.numpy as jnp
    >>> from hcd_analysis.emulator.dndx_wc import DeltaCMap, w_c_corrected, w_c_from_mu
    >>> no_correction = DeltaCMap(np.zeros((4, 3)), 2.0, 5.4, "no correction")
    >>> dndx, Xbar = jnp.array([0.1, 0.02, 0.01]), jnp.array(1.0)
    >>> w = w_c_corrected(dndx, Xbar, jnp.array(3.0), no_correction)
    >>> bool(np.allclose(w, w_c_from_mu(dndx * Xbar)))
    True
    """
    w0 = w_c_from_mu(mu_from_dndx(dndx, Xbar))  # (...,4)
    w = w0 * (1.0 + delta_c(z, delta_map))
    return w / jnp.sum(w, axis=-1, keepdims=True)


# --- Task 3: alpha_c(z) incidence parametrization + analytic M0-inverse ----------
_LOG_FLOOR = 1e-12  # positive floor on (1 - w...) so log stays finite & differentiable


def dndx_powerlaw(z: ArrayLike, A: ArrayLike, gamma: ArrayLike) -> jax.Array:
    """A power law in redshift for the incidence of each HCD class.

    ``dN/dX_c(z) = A_c * (1 + z)**gamma_c``. It is the incidence model of the research likelihood (named
    PW14 in that code); the public predictor does not use it.

    Parameters
    ----------
    z : array_like
        The redshift: a scalar or shape (...).
    A, gamma : array_like
        The amplitude and the slope: a scalar, or shape (3,) for LLS, subDLA, DLA.

    Returns
    -------
    jax.Array
        Shape (..., 3): dN/dX of LLS, subDLA, DLA at ``z`` (shape (..., 1) when ``A`` and ``gamma`` are
        both scalars). Differentiable.

    Examples
    --------
    >>> from hcd_analysis.emulator.dndx_wc import dndx_powerlaw
    >>> [round(float(v), 12) for v in dndx_powerlaw(3.0, [0.2, 0.1, 0.05], [1.0, 1.0, 1.0])]   # synthetic
    [0.8, 0.4, 0.2]
    """
    z = jnp.asarray(z)
    A = jnp.asarray(A)
    gamma = jnp.asarray(gamma)
    return A * (1.0 + z[..., None]) ** gamma


def alpha_from_dndx_law(A, gamma, Xbar, z, delta_map=None):
    """Forward alpha_c(z) parametrized by (A_c, gamma_c): the 3 HCD entries of
    w_c_corrected applied to the PW14 incidence law.
    A, gamma: scalar or (3,); Xbar: (...); z: scalar or (...). Returns (...,3)
    for (LLS, subDLA, DLA). ``delta_map`` as in ``delta_c`` (None: the historical map)."""
    dndx = dndx_powerlaw(z, A, gamma)  # (...,3)
    return w_c_corrected(dndx, Xbar, z, delta_map)[..., 1:]  # drop clean -> HCD (LLS,subDLA,DLA)


def alpha_to_dndx(alpha_hcd, Xbar, z, apply_delta=True, delta_map=None):
    """DEPRECATED approximate M0-inverse. RETAINED SOLELY TO REPRODUCE PRE-2026-07-22
    FROZEN ARTIFACTS (paper exports built with this approximate inverse). ALL NEW
    CODE MUST USE ``alpha_to_dndx_exact`` instead (readout defect B, 2026-07-22).

    Two measured defects (2026-07-22, on in-range deployed-prior draws):
      1. RENORM ERROR. With apply_delta=True this divides out the per-class
         (1+delta_c) factor but NOT the global 4-class renormalisation of
         ``w_c_corrected``, so it is NOT the inverse of the deployed forward. The
         median relative dN/dX error is ~2e-3 (the "typically sub-percent" claim
         formerly here was FALSE), with catastrophic tails near the simplex edge.
      2. SILENT SATURATION. Outside the valid domain (sum(alpha) >= 1, or infeasible
         telescoping args) the _LOG_FLOOR clip silently SATURATES the affected class
         at mu = -log(1e-12) = 27.631021, i.e. dN/dX = 27.631021/Xbar, instead of
         failing. 0.07-0.5% of in-range deployed-prior draws hit this
         (ensemble-dependent). This corrupted shipped paper artifacts.

    alpha_hcd: (...,3) per-class effective incidence weights for (LLS,subDLA,DLA).
    Xbar: (...); z: scalar or (...). Returns (...,3) dN/dX_c for (LLS,subDLA,DLA).
    ``delta_map`` as in ``delta_c`` (None: the historical map).

    Steps:
      1. If apply_delta, divide out the delta_c correction on the HCD slice:
             w = alpha_hcd / (1 + delta_c(z)[...,1:]).
         (Defect 1 above: the 4-class renorm of w_c_corrected is NOT undone here.)
         With apply_delta=False this step is skipped and the telescoping inverse
         below is the EXACT inverse of w_c_from_mu on the HCD classes -- except for
         the clip floor (defect 2).
      2. Invert the telescoping-Poisson (top-down DLA->subDLA->LLS), with the
         _LOG_FLOOR clip (defect 2).
      3. dN/dX_c = mu_c / Xbar.
    """
    w = jnp.asarray(alpha_hcd)
    if apply_delta:
        w = w / (1.0 + delta_c(z, delta_map)[..., 1:])
    w_LLS, w_sub, w_DLA = w[..., 0], w[..., 1], w[..., 2]

    # Exact top-down inverse of w_c_from_mu. Clip the (1 - ...) args to a tiny positive
    # floor so log stays finite for valid inputs (keeps it differentiable).
    mu_DLA = -jnp.log(jnp.clip(1.0 - w_DLA, _LOG_FLOOR, None))
    mu_sub = -jnp.log(
        jnp.clip(1.0 - w_sub / jnp.clip(1.0 - w_DLA, _LOG_FLOOR, None), _LOG_FLOOR, None)
    )
    mu_LLS = -jnp.log(
        jnp.clip(1.0 - w_LLS / jnp.clip((1.0 - w_DLA) - w_sub, _LOG_FLOOR, None), _LOG_FLOOR, None)
    )

    mu = jnp.stack([mu_LLS, mu_sub, mu_DLA], axis=-1)  # (...,3)
    return mu / jnp.asarray(Xbar)[..., None]


def alpha_to_dndx_exact(alpha_hcd, Xbar, z, *, mode="raise", delta_map=None):
    """EXACT inverse of the ``w_c_corrected`` forward on the HCD classes (the readout).

    SUPERSEDES ``alpha_to_dndx`` for ALL new readouts/exports/plots (readout defect B,
    2026-07-22): that map ignores the 4-class renormalisation (median relative error
    ~2e-3 on in-range deployed-prior draws, catastrophic tails) and silently saturates
    at mu = -log(1e-12) = 27.631021 outside its domain. This function inverts the full
    forward exactly, with NO clip floor, and fails loud (or masks) outside the domain.
    Host-side numpy readout: NOT jit/vmap/grad-safe; do not call it from traced code.

    Derivation (one line): the forward is alpha_j = w0_j*(1+d_j)/S with
    S = sum_j w0_j*(1+d_j) and sum_j w0_j = 1 (j over clean + 3 HCD classes,
    d = delta_c(z)); imposing sum_j w0_j = 1 on w0_j = alpha_j*S/(1+d_j) gives
        Z = S = 1 / [ (1 - sum_c a_c)/(1 + d_clean) + sum_c a_c/(1 + d_c) ]
    (sums over the 3 HCD classes), then w0_c = a_c * Z / (1 + d_c), followed by the
    exact top-down telescoping inverse (log1p form, no clip):
        mu_DLA = -log1p(-w0_DLA)
        mu_sub = -log1p(-w0_sub / (1 - w0_DLA))
        mu_LLS = -log1p(-w0_LLS / ((1 - w0_DLA) - w0_sub))
    and dN/dX_c = mu_c / Xbar.

    Domain: all(alpha_hcd >= 0) AND (1 - sum(alpha_hcd)) > 0 strictly (the occupancy
    simplex with a strictly positive clean fraction). Inside it every log1p argument
    is strictly inside (-1, 0], so the result is finite with no floor needed.

    Conditioning: the round-trip error grows like 1/w0_clean near the simplex edge.
    Measured: at 1 - sum(alpha) ~ 1e-12 a 1-ulp input perturbation moves mu_LLS by
    ~3e-6 relative (~2.7e-4 at 1e-14; unbounded at 1e-16). Inputs with
    1 - sum(alpha) < ~1e-12 are therefore domain-VALID but numerically UNRELIABLE.

    alpha_hcd: (...,3) per-class effective incidence weights for (LLS,subDLA,DLA);
    Xbar: (...); z: scalar or (...). numpy or jax arrays accepted (materialised via
    np.asarray; the computation is plain numpy).

    mode="raise" (default): host-side domain check; raises ValueError naming the
      violation class(es) if any element is outside the domain. NOTE: draws from the
      pre-2026-07-22 UNBOUNDED alpha prior can legitimately violate the domain --
      use mode="mask" to read such historical draws.
    mode="mask": returns (dndx, valid) with valid = all(alpha >= 0, axis=-1)
      & (1 - sum(alpha) > 0); invalid entries are NaN in dndx, no exception.

    ``delta_map`` as in ``delta_c`` (None: the historical map).

    Returns: (...,3) dN/dX_c for (LLS,subDLA,DLA) [mode="raise"], or the tuple
    (dndx, valid) [mode="mask"].
    """
    if mode not in ("raise", "mask"):
        raise ValueError(f"alpha_to_dndx_exact: unknown mode {mode!r} (use 'raise' or 'mask')")
    a = np.asarray(alpha_hcd, dtype=np.float64)
    Xb = np.asarray(Xbar, dtype=np.float64)
    s = a.sum(axis=-1)
    a_clean = 1.0 - s
    nonneg = np.all(a >= 0.0, axis=-1)
    valid = nonneg & (a_clean > 0.0)

    if mode == "raise" and not np.all(valid):
        n_bad = int(np.size(valid) - np.count_nonzero(valid))
        worst_sum = float(np.max(np.atleast_1d(s)[~np.atleast_1d(valid)]))
        kinds = []
        n_neg = int(np.count_nonzero(~np.atleast_1d(nonneg)))
        if n_neg:
            kinds.append(f"negative alpha in {n_neg} row(s)")
        n_over = int(np.count_nonzero(np.atleast_1d(nonneg & ~(a_clean > 0.0))))
        if n_over:
            kinds.append(f"sum(alpha) >= 1 (outside the occupancy simplex) in {n_over} row(s)")
        raise ValueError(
            f"alpha_to_dndx_exact: {n_bad} input row(s) outside the valid domain "
            f"[all(alpha) >= 0 and 1 - sum(alpha) > 0]: "
            + "; ".join(kinds)
            + f". Worst sum(alpha) = {worst_sum!r}. Draws from the pre-2026-07-22 "
            f"unbounded alpha prior can legitimately violate this domain; use "
            f"mode='mask' to read such historical draws (invalid entries -> NaN)."
        )

    # Mask invalid rows to NaN BEFORE the logs (NaN propagates silently -- no
    # log-of-negative warnings). In raise mode everything is valid at this point.
    a = np.where(valid[..., None], a, np.nan)
    a_clean = np.where(valid, a_clean, np.nan)

    d = np.asarray(delta_c(z, delta_map), dtype=np.float64)  # (...,4)
    Z = 1.0 / (a_clean / (1.0 + d[..., 0]) + np.sum(a / (1.0 + d[..., 1:]), axis=-1))
    w0 = a * Z[..., None] / (1.0 + d[..., 1:])  # (...,3)
    w_LLS, w_sub, w_DLA = w0[..., 0], w0[..., 1], w0[..., 2]

    mu_DLA = -np.log1p(-w_DLA)
    mu_sub = -np.log1p(-(w_sub / (1.0 - w_DLA)))
    mu_LLS = -np.log1p(-(w_LLS / ((1.0 - w_DLA) - w_sub)))
    dndx = np.stack([mu_LLS, mu_sub, mu_DLA], axis=-1) / Xb[..., None]
    if mode == "mask":
        return dndx, np.broadcast_to(valid, dndx.shape[:-1]).copy()
    return dndx
