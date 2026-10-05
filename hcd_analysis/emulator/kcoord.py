"""THE canonical physical k coordinate of the emulator: k_skm(z, theta). Nothing downstream may invent another.
Boundary of the 2026 k-grid representation regression; see hcd_priya_notes
docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md."""
from __future__ import annotations

from typing import NamedTuple

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


class KGrid(NamedTuple):
    """The canonical coordinate object: every physical array of a prediction is defined on ``k_skm``."""
    k_skm: jnp.ndarray
    z: float
    hub: jnp.ndarray
    omegamh2: jnp.ndarray
    k_com_hmpc: jnp.ndarray
    schema_version: str


def kgrid(k_com_hmpc, z, theta9_unit):
    hub, omh2 = hub_omegamh2_from_theta9(theta9_unit)
    # hub/omegamh2 stay JAX scalars (differentiable in theta); z is a fixed physical redshift per leg.
    return KGrid(k_skm_from_kcom(k_com_hmpc, z, hub, omh2), float(z), hub, omh2,
                 jnp.asarray(k_com_hmpc), CHECKPOINT_SCHEMA_VERSION)
