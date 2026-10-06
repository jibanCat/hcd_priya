"""THE canonical physical k coordinate of the emulator: k_skm(z, theta). Nothing downstream may invent another.
Boundary of the 2026 k-grid representation regression; see hcd_priya_notes
docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md."""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .data import PARAM_LIMITS
from .schema import CHECKPOINT_SCHEMA_VERSION, K_CONVENTION, L_BOX_HMPC

H0_PER_H_KMS_MPC = 100.0
_I_HUB, _I_OMH2 = 5, 6          # positions of hub and omegamh2 in the 9-vector (data.PARAM_LIMITS order)
CONVENTION = K_CONVENTION


def E_of_z(z, hub, omegamh2):
    """Flat LCDM, no radiation: E(z) = sqrt(Om (1+z)^3 + 1 - Om), Om = omegamh2 / hub^2."""
    om = omegamh2 / hub ** 2
    return jnp.sqrt(om * (1.0 + z) ** 3 + 1.0 - om)


def vbox_kms(z, hub, omegamh2):
    """Velocity width of the L_box = 120 Mpc/h box at (z, theta): L 100 E(z) / (1+z) km/s (h cancels)."""
    return L_BOX_HMPC * H0_PER_H_KMS_MPC * E_of_z(z, hub, omegamh2) / (1.0 + z)


def k_skm_from_kcom(k_com_hmpc, z, hub, omegamh2):
    """k [s/km, angular] of comoving modes k_com [h/Mpc] at (z, hub, omegamh2): k_com (1+z) / (100 E(z))."""
    return jnp.asarray(k_com_hmpc) * (1.0 + z) / (H0_PER_H_KMS_MPC * E_of_z(z, hub, omegamh2))


def hub_omegamh2_from_theta9(theta9_unit):
    """Physical (hub, omegamh2) from the unit-cube 9-vector via data.PARAM_LIMITS (the emulator's own map)."""
    if tuple(jnp.shape(theta9_unit)) != (9,):                  # static shape: checked under tracing too
        raise ValueError(f"theta9 must have shape (9,), got {tuple(jnp.shape(theta9_unit))}; vmap over batches")
    if not isinstance(theta9_unit, jax.core.Tracer):          # concrete input: refuse physical values loudly
        arr = np.asarray(theta9_unit, float)
        if np.any(arr < -1e-9) or np.any(arr > 1 + 1e-9):
            raise ValueError("theta9 must be in the unit cube (sampling coordinates), not physical values")
    lim = np.asarray(PARAM_LIMITS, float)
    lo = jnp.asarray(lim[:, 0])
    hi = jnp.asarray(lim[:, 1])
    phys = lo + jnp.asarray(theta9_unit) * (hi - lo)
    return phys[_I_HUB], phys[_I_OMH2]


def k_skm_from_theta9(k_com_hmpc, z, theta9_unit):
    hub, omh2 = hub_omegamh2_from_theta9(theta9_unit)
    return k_skm_from_kcom(k_com_hmpc, z, hub, omh2)


class KGrid(eqx.Module):
    """The canonical coordinate object: every physical array of a prediction is defined on ``k_skm``. ``z`` and
    ``schema_version`` are static, so the object passes through jit and vmap."""
    k_skm: jnp.ndarray
    z: float = eqx.field(static=True)
    hub: jnp.ndarray = None
    omegamh2: jnp.ndarray = None
    k_com_hmpc: jnp.ndarray = None
    schema_version: str = eqx.field(static=True, default=CHECKPOINT_SCHEMA_VERSION)


def kgrid(k_com_hmpc, z, theta9_unit):
    hub, omh2 = hub_omegamh2_from_theta9(theta9_unit)
    # hub/omegamh2 stay JAX scalars (differentiable in theta); z is a fixed physical redshift per leg.
    return KGrid(k_skm_from_kcom(k_com_hmpc, z, hub, omh2), float(z), hub, omh2,
                 jnp.asarray(k_com_hmpc), CHECKPOINT_SCHEMA_VERSION)


class ModeBinding(NamedTuple):
    """Where data bins sit on the mode axis of one (leg, z) for one theta: fractional mode index u = k / k_skm,1,
    bracketing mode j (1-based; modes j and j + 1), linear and log-k weights, and the number of bins outside the
    simulated modes. The ONLY object that places mode-axis quantities at physical k (GATE_E_SPEC v1 section 2)."""
    kg: KGrid
    u: jnp.ndarray
    j: jnp.ndarray
    t_lin: jnp.ndarray
    t_log: jnp.ndarray
    n_out: jnp.ndarray


def _require_box_modes(k_com_hmpc):
    if isinstance(k_com_hmpc, jax.core.Tracer):
        return
    k = np.asarray(k_com_hmpc, float)
    if not np.allclose(k, 2 * np.pi * np.arange(1, k.size + 1) / L_BOX_HMPC, rtol=1e-12, atol=0):
        raise ValueError("binding needs the box modes k_com = 2 pi n / L, n = 1..K (k_skm,n = n k_skm,1)")


def bind(kg, k_data):
    """Bind data wavenumbers k_data [s/km] to the mode axis of ``kg``. Linear interpolation at u on the fixed mode
    axis equals linear interpolation in k (k_skm,n = n k_skm,1), and outside the modes it clamps like jnp.interp;
    ``n_out`` counts the bins outside [k_skm,1, k_skm,K] (the caller refuses them)."""
    _require_box_modes(kg.k_com_hmpc)
    K = kg.k_skm.shape[-1]
    u = jnp.asarray(k_data) / kg.k_skm[0]
    j = jnp.clip(jnp.floor(u), 1.0, K - 1.0)
    t_lin = jnp.clip(u - j, 0.0, 1.0)
    t_log = jnp.clip((jnp.log(jnp.maximum(u, 1e-300)) - jnp.log(j)) / (jnp.log(j + 1.0) - jnp.log(j)), 0.0, 1.0)
    n_out = jnp.sum((u < 1.0) | (u > K))
    return ModeBinding(kg, u, j.astype(jnp.int32), t_lin, t_log, n_out)


def at_data(b, f_mode):
    """A mode-axis quantity f (..., K) (index 0 = mode 1) at the bound data bins, linear in k."""
    f = jnp.asarray(f_mode)
    return (1.0 - b.t_lin) * jnp.take(f, b.j - 1, axis=-1) + b.t_lin * jnp.take(f, b.j, axis=-1)


def kbounds_over_box(k_com_hmpc, z, lo_unit, hi_unit):
    """(max over the box of k_skm,1, min over the box of k_skm,K) at z for the unit-cube sampling box [lo, hi]. k_skm
    falls with E(z), which rises with Omega_m = omegamh2 / hub^2 (z > 0), so the extremes sit at the box's
    (hub_max, omegamh2_min) and (hub_min, omegamh2_max) corners."""
    lim = np.asarray(PARAM_LIMITS, float)
    lo, hi = np.asarray(lo_unit, float), np.asarray(hi_unit, float)
    hub_lo, hub_hi = lim[_I_HUB, 0] + lo[_I_HUB] * np.ptp(lim[_I_HUB]), lim[_I_HUB, 0] + hi[_I_HUB] * np.ptp(lim[_I_HUB])
    om_lo, om_hi = lim[_I_OMH2, 0] + lo[_I_OMH2] * np.ptp(lim[_I_OMH2]), lim[_I_OMH2, 0] + hi[_I_OMH2] * np.ptp(lim[_I_OMH2])
    k = np.asarray(k_com_hmpc, float)
    k1_max = float(k_skm_from_kcom(k[0], z, hub_hi, om_lo))
    kK_min = float(k_skm_from_kcom(k[-1], z, hub_lo, om_hi))
    return k1_max, kK_min
