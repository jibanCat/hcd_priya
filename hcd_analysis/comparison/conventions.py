"""Named parameters and explicit conversions shared by every adapter (numpy only).

Conventions:
  * comoving box modes k_com = 2 pi n / L_box [h/Mpc], n = 1..n_k;
  * angular velocity wavenumber k_skm = k_com (1+z) / (100 E(z)) [s/km], E = sqrt(Om (1+z)^3 + 1 - Om),
    Om = omegamh2 / hub^2 (flat LCDM, no radiation; h cancels);
  * P1D in km/s is P in comoving Mpc/h times 100 E(z) / (1+z);
  * PRIYA mean flux: alpha(z) = tau0_amp ((1+z)/(1+z_pivot))^dtau0 times Kim07 0.0023 (1+z)^3.65.
    PRIYA's ``tau0`` is the dimensionless amplitude ``tau0_amp``; hcd_priya's emulator input ``tau0`` is the optical
    depth alpha(z) tau_Kim07(z). They are different quantities and are only ever converted through this module."""
from __future__ import annotations

import numpy as np

PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
MEAN_FLUX_KEYS = ("tau0_amp", "dtau0", "z_pivot")
H0_PER_H = 100.0
KIM07_AMP, KIM07_SLOPE = 0.0023, 3.65


class ConventionError(ValueError):
    """A state or vector that cannot be converted without guessing."""


def E_of_z(z, hub, omegamh2):
    om = np.asarray(omegamh2, float) / np.asarray(hub, float) ** 2
    zz = np.asarray(z, float)
    return np.sqrt(om * (1.0 + zz) ** 3 + 1.0 - om)


def k_com_modes(n, L_box_hmpc):
    """Comoving wavenumber [h/Mpc] of integer box modes ``n``."""
    n = np.asarray(n)
    if n.dtype.kind not in "iu" or np.any(n < 1):
        raise ConventionError("box modes must be positive integers")
    return 2.0 * np.pi * n / float(L_box_hmpc)


def k_skm_from_kcom(k_com_hmpc, z, hub, omegamh2):
    return np.asarray(k_com_hmpc, float) * (1.0 + z) / (H0_PER_H * E_of_z(z, hub, omegamh2))


def kcom_from_kskm(k_skm, z, hub, omegamh2):
    return np.asarray(k_skm, float) * H0_PER_H * E_of_z(z, hub, omegamh2) / (1.0 + z)


def P_kms_from_P_hmpc(P_hmpc, z, hub, omegamh2):
    """P1D [km/s] from P1D [comoving Mpc/h]: the length unit of the 1D power converts like 1/k."""
    return np.asarray(P_hmpc, float) * H0_PER_H * E_of_z(z, hub, omegamh2) / (1.0 + z)


def P_hmpc_from_P_kms(P_kms, z, hub, omegamh2):
    return np.asarray(P_kms, float) * (1.0 + z) / (H0_PER_H * E_of_z(z, hub, omegamh2))


def kim07_tau(z):
    return KIM07_AMP * (1.0 + np.asarray(z, float)) ** KIM07_SLOPE


def check_mean_flux(mf):
    if not isinstance(mf, dict) or set(mf) != set(MEAN_FLUX_KEYS):
        raise ConventionError(f"mean-flux state must have exactly {MEAN_FLUX_KEYS}, got {mf!r}")
    for k in MEAN_FLUX_KEYS:
        if not (isinstance(mf[k], (int, float)) and np.isfinite(mf[k])) or isinstance(mf[k], bool):
            raise ConventionError(f"mean-flux {k} = {mf[k]!r} is not a finite number")


def alpha_priya(z, mean_flux):
    """PRIYA ladder coordinate alpha(z) = tau_eff / tau_Kim07 of a (tau0_amp, dtau0, z_pivot) state."""
    check_mean_flux(mean_flux)
    return mean_flux["tau0_amp"] * ((1.0 + np.asarray(z, float)) / (1.0 + mean_flux["z_pivot"])) ** mean_flux["dtau0"]


def hcd_priya_tau0_input(z, mean_flux):
    """hcd_priya's emulator input tau0(z): the optical depth alpha(z) tau_Kim07(z) (NOT PRIYA's tau0)."""
    return alpha_priya(z, mean_flux) * kim07_tau(z)


def check_params(params):
    """A physical state must name exactly the nine PRIYA parameters, with finite values."""
    if not isinstance(params, dict) or set(params) != set(PARAM_NAMES):
        got = sorted(params) if isinstance(params, dict) else params
        raise ConventionError(f"parameters must be exactly {PARAM_NAMES}, got {got}")
    for k in PARAM_NAMES:
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ConventionError(f"parameter {k} = {v!r} is not a finite number")


def vector_from_names(params, index):
    """Physical vector ordered by a reference's own ``{name: position}`` map (never by assumed position)."""
    check_params(params)
    if set(index) != set(PARAM_NAMES) or sorted(index.values()) != list(range(len(PARAM_NAMES))):
        raise ConventionError(f"reference parameter index {index!r} is not a permutation of {PARAM_NAMES}")
    v = np.empty(len(PARAM_NAMES))
    for name, i in index.items():
        v[i] = params[name]
    return v


def params_from_vector(vector, names=PARAM_NAMES):
    v = np.asarray(vector, float)
    if v.shape != (len(names),):
        raise ConventionError(f"vector shape {v.shape} does not match {len(names)} names")
    return {n: float(x) for n, x in zip(names, v)}


def in_limits(params, limits):
    """``limits`` = {name: (lo, hi)}; True when every parameter lies inside (inclusive)."""
    check_params(params)
    return all(limits[n][0] <= params[n] <= limits[n][1] for n in PARAM_NAMES)
