"""The production forward on the canonical coordinate (gate E, emulator-debug campaign 2026-10; spec GATE_E_SPEC.md v1
in the notes repository; incident 2026-10-05-INCIDENT-kgrid-representation-regression).

Everything the emulator produces stays on the mode axis n = 1..K: the ensemble's per-class P_filt, the MF correction
(``mf_modes.ModeMF``, theta-independent) and the mode-indexed DLA core. For each (leg, z) ONE binding from the query
theta (``kcoord.kgrid`` + ``kcoord.bind``) places them at the leg's data k; the class combination, metals and
resolution then act on the data bins exactly as before. Bins outside the simulated modes are counted (``n_out``); the
likelihood refuses them. ``kcoord`` is called through the module so tests can intercept the coordinate."""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import data_likelihood as DL
from . import kcoord as KC
from .data import Z_LIMITS
from .predict import predict_P_filt

_METAL_KEYS = ("a_SiIII", "a_SiII", "k_SiIII", "k_SiII", "f_SiIII_nodes", "f_SiII_nodes", "metal_node_z",
               "k_SiIII_nodes", "k_SiII_nodes")
_NUIS_KEYS = _METAL_KEYS + ("b_res", "b_res_vec")


class LegPrediction(NamedTuple):
    P_model: jnp.ndarray          # (N,) flat, z-major (the leg's row order)
    n_out: jnp.ndarray            # number of the leg's bins outside the simulated modes (all z)
    C_total: jnp.ndarray          # (N, N) data covariance + the emulator-error terms supplied


def predict_leg(model, theta9, tau0_vec, alpha_hcd, *, leg, k_com, pf_stats, dla_core, mf=None, nuis=None,
                t1=None, cemu_inflate=1.0, require_zresolved=True):
    """The model P1D on every bin of ``leg`` and its total covariance. ``alpha_hcd`` is (n_z, 3) per-z incidence (LLS,
    subDLA, DLA); a z-flat (3,) alpha is refused unless ``require_zresolved=False`` (the z-flat broadcast bug class).
    ``dla_core`` is the mode-indexed core, (K,) or (n_z, K). ``nuis``: metal and resolution parameters (keys
    ``_NUIS_KEYS``). ``t1`` = (rho per leg z (n_z, 4, 4, K, Tb), tau0-band centres (Tb,)): the cross-class emulator
    variance on the modes from the LF P_filt, bound at the data bins and scaled by the squared nuisance factors."""
    nuis = dict(nuis or {})
    unknown = set(nuis) - set(_NUIS_KEYS)
    if unknown:
        raise ValueError(f"unknown nuisance keys {sorted(unknown)}")
    metal_kw = {k: nuis[k] for k in _METAL_KEYS if k in nuis}
    b_res, b_res_vec = nuis.get("b_res", 0.0), nuis.get("b_res_vec")
    alpha_hcd = jnp.asarray(alpha_hcd)
    if require_zresolved and alpha_hcd.ndim != 2:
        raise ValueError(f"predict_leg: alpha_hcd must be z-RESOLVED (n_z, 3), got shape {tuple(alpha_hcd.shape)}; "
                         "a z-flat alpha would be broadcast to every z (the z-flat bug class)")
    dff = float(getattr(leg, "dla_forward_frac", 1.0))
    if dff != 1.0:
        scale = jnp.array([1.0, 1.0, dff])
        alpha_hcd = alpha_hcd * (scale if alpha_hcd.ndim == 1 else scale[None, :])
    tau0_vec = jnp.asarray(tau0_vec)
    dla_core = jnp.asarray(dla_core)
    k_leg = np.asarray(leg.k)
    z_idx = np.asarray(leg.z_idx)
    R_z = jnp.asarray(leg.R_z)
    P_model = jnp.zeros(k_leg.shape[0])
    emu_var = jnp.zeros(k_leg.shape[0])
    n_out = jnp.asarray(0)
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        k_sub = jnp.asarray(k_leg[rows])
        kg = KC.kgrid(k_com, float(leg.z[iz]), theta9)
        b = KC.bind(kg, k_sub)
        z_unit = (kg.z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        tau0 = tau0_vec[iz]
        core_modes = dla_core if dla_core.ndim == 1 else dla_core[iz]
        a = alpha_hcd if alpha_hcd.ndim == 1 else alpha_hcd[iz]
        P_lf = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)                     # (4, K) on the modes
        P_filt = P_lf
        if mf is not None:
            x = jnp.concatenate([jnp.asarray(theta9), jnp.asarray([z_unit])])
            P_filt = P_lf * jnp.exp(mf(x, tau0))
        Pc = KC.at_data(b, P_filt)                                                         # (4, n) at data k
        core = KC.at_data(b, core_modes)
        P_z = Pc[0] + a[0] * (Pc[1] - Pc[0]) + a[1] * (Pc[2] - Pc[0]) + a[2] * (Pc[3] + core - Pc[0])
        fac = jnp.ones(rows.size)
        if leg.metals_on:
            fac = fac * DL.metal_factor_at_z(k_sub, kg.z, tau0, **metal_kw)
        if leg.resolution_on:
            fac = fac * DL._resolution_factor(k_sub, R_z[iz], b_res=b_res if b_res_vec is None else b_res_vec[iz])
        P_model = P_model.at[jnp.asarray(rows)].set(P_z * fac)
        if t1 is not None:
            rho_leg, alpha_centres = t1
            ev = DL.emu_var_modes(P_lf, kg.z, tau0, a, dla_core=core_modes, alpha_centres=alpha_centres,
                                  rho_zb=rho_leg[iz], cemu_inflate=cemu_inflate)              # (K,) on the modes
            emu_var = emu_var.at[jnp.asarray(rows)].set(KC.at_data(b, ev) * fac ** 2)
        n_out = n_out + b.n_out
    C_total = jnp.asarray(leg.C_data) + jnp.diag(emu_var) if t1 is not None else jnp.asarray(leg.C_data)
    return LegPrediction(P_model, n_out, C_total)
