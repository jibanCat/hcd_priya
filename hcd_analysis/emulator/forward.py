"""The production forward on the canonical coordinate (gate E, emulator-debug campaign 2026-10; spec GATE_E_SPEC.md v1
in the notes repository; incident 2026-10-05-INCIDENT-kgrid-representation-regression).

Everything the emulator produces stays on the mode axis n = 1..K: the ensemble's per-class P_filt, the MF correction
(``mf_modes.ModeMF``, theta-independent) and the T1 cross-class block. For each (leg, z) ONE binding from the query
theta (``kcoord.kgrid`` + ``kcoord.bind``) places them at the leg's data k; the DLA core is a fixed function of physical
k given at the data bins (gate E amendment A1 rev 1 section 4); the class combination, metals and resolution then act
on the data bins exactly as before. Bins outside the simulated modes are counted (``n_out``); the
likelihood refuses them. ``kcoord`` is called through the module so tests can intercept the coordinate."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import data_likelihood as DL
from . import kcoord as KC
from .data import Z_LIMITS
from .likelihood import rho_at_tau0
from .predict import predict_P_filt

_METAL_KEYS = ("a_SiIII", "a_SiII", "k_SiIII", "k_SiII", "f_SiIII_nodes", "f_SiII_nodes", "metal_node_z",
               "k_SiIII_nodes", "k_SiII_nodes")
_NUIS_KEYS = _METAL_KEYS + ("b_res", "b_res_vec")
NS_BOX = (0.86, 0.98)          # the HR n_s cluster box of the MF n_s-edge term (floor spec section 4)
EDGE_SLOPE_MULT = 2.0          # 2 x the 6-simulation edge slope (floor spec section 4.1)


class LegPrediction(NamedTuple):
    P_model: jnp.ndarray          # (N,) flat, z-major (the leg's row order)
    n_out: jnp.ndarray            # number of the leg's bins outside the simulated modes (all z)
    C_total: jnp.ndarray          # (N, N) data covariance + the emulator-error terms supplied
    outside: jnp.ndarray          # (N,) bool: the bin lies outside the simulated modes (the caller counts kept rows)


def predict_leg(model, theta9, tau0_vec, alpha_hcd, *, leg, k_com, pf_stats, dla_core, mf=None, nuis=None,
                t1=None, t2=None, t3=None, cemu_inflate=1.0, require_zresolved=True):
    """The model P1D on every bin of ``leg`` and its total covariance. ``alpha_hcd`` is (n_z, 3) per-z incidence (LLS,
    subDLA, DLA); a z-flat (3,) alpha is refused unless ``require_zresolved=False`` (the z-flat broadcast bug class).
    ``dla_core`` is the leg's data-bin DLA core (N,): a fixed function of physical k at the leg's bins, no binding
    (amendment A1 rev 1 section 4; a mode-indexed core is refused). ``nuis``: metal and resolution parameters (keys
    ``_NUIS_KEYS``). ``t1`` = (rho per leg z (n_z, 4, 4, K, Tb), tau0-band centres (Tb,)): the cross-class emulator
    variance at the data bins, fac^2 sum coef coef bind(rho(tau0)) A_c A_c' with A_c the LF P_filt bound to the data
    bins and the DLA core added to A_DLA (A1 rev 1 section 1).
    ``t2`` = (sigma_floor per leg z (n_z,), slope per leg z (n_z,)): the MF floor and n_s-edge term on the
    post-nuisance model (``t2_var``). ``t3`` = (U (N, m), w (m,)): the fractional k-coherence factor on the leg's bins,
    amplitude from P_data, off-diagonal only (``assemble_cov``)."""
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
    if dla_core.shape != k_leg.shape:
        raise ValueError(f"predict_leg: dla_core must be the leg's data-bin core of shape {k_leg.shape}, got "
                         f"{tuple(dla_core.shape)} (the mode-indexed core is retired, amendment A1 rev 1 section 4)")
    z_idx = np.asarray(leg.z_idx)
    R_z = jnp.asarray(leg.R_z)
    P_model = jnp.zeros(k_leg.shape[0])
    emu_var = jnp.zeros(k_leg.shape[0])
    floor_var = jnp.zeros(k_leg.shape[0])
    ns_sg = jax.lax.stop_gradient(DL._ns_phys_from_theta9(theta9)) if t2 is not None else None
    n_out = jnp.asarray(0)
    outside = jnp.zeros(k_leg.shape[0], dtype=bool)
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        k_sub = jnp.asarray(k_leg[rows])
        kg = KC.kgrid(k_com, float(leg.z[iz]), theta9)
        b = KC.bind(kg, k_sub)
        z_unit = (kg.z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        tau0 = tau0_vec[iz]
        core = dla_core[jnp.asarray(rows)]
        a = alpha_hcd if alpha_hcd.ndim == 1 else alpha_hcd[iz]
        P_lf = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)                     # (4, K) on the modes
        P_filt = P_lf
        if mf is not None:
            x = jnp.concatenate([jnp.asarray(theta9), jnp.asarray([z_unit])])
            P_filt = P_lf * jnp.exp(mf(x, tau0))
        Pc = KC.at_data(b, P_filt)                                                         # (4, n) at data k
        P_z = Pc[0] + a[0] * (Pc[1] - Pc[0]) + a[1] * (Pc[2] - Pc[0]) + a[2] * (Pc[3] + core - Pc[0])
        fac = jnp.ones(rows.size)
        if leg.metals_on:
            fac = fac * DL.metal_factor_at_z(k_sub, kg.z, tau0, **metal_kw)
        if leg.resolution_on:
            fac = fac * DL._resolution_factor(k_sub, R_z[iz], b_res=b_res if b_res_vec is None else b_res_vec[iz])
        P_model = P_model.at[jnp.asarray(rows)].set(P_z * fac)
        if t1 is not None:
            rho_leg, alpha_centres = t1
            rho_d = KC.at_data(b, rho_at_tau0(rho_leg[iz], alpha_centres, kg.z, tau0))      # (4, 4, n) at data k
            ev = DL.emu_var_at_data(KC.at_data(b, P_lf), core, rho_d, a, cemu_inflate=cemu_inflate)
            emu_var = emu_var.at[jnp.asarray(rows)].set(ev * fac ** 2)
        if t2 is not None:
            floor_var = floor_var.at[jnp.asarray(rows)].set(t2_var(t2[0][iz], t2[1][iz], P_z * fac, ns_sg))
        n_out = n_out + b.n_out
        outside = outside.at[jnp.asarray(rows)].set((b.u < 1.0) | (b.u > kg.k_skm.shape[-1]))
    C_total = assemble_cov(jnp.asarray(leg.C_data), emu_var, floor_var, t3=t3,
                           P_fid=jnp.nan_to_num(jnp.asarray(leg.P_data)))
    return LegPrediction(P_model, n_out, C_total, outside)


def t2_var(sigma_floor, slope, P, ns, ns_box=NS_BOX, edge_mult=EDGE_SLOPE_MULT):
    """The MF floor variance at one z: (sigma_floor P)^2 + (sigma_edge P)^2, sigma_edge = max(edge_mult |slope| d_ns,
    0.5 sigma_floor d_ns / 0.03), d_ns = the distance of the physical n_s outside ``ns_box`` (floor spec sections
    2-4). ``ns`` is passed stop-gradient by ``predict_leg``: that removes the edge term's n_s force from jax.grad ONLY;
    its value stays in the density, so it can move the posterior and any MAP (the sampler's target is unchanged, a
    jax.grad-based Fisher or MAP omits the force). Audited by finite differences of the value (gate E amendment A1
    section 5 (vi))."""
    d_ns = jnp.maximum(jnp.maximum(ns - ns_box[1], ns_box[0] - ns), 0.0)
    sig_edge = jnp.maximum(edge_mult * jnp.abs(slope) * d_ns, 0.5 * sigma_floor * (d_ns / 0.03))
    P = jnp.asarray(P)
    return (sigma_floor * P) ** 2 + (sig_edge * P) ** 2


def assemble_cov(C_data, t1_var, t2_var, t3=None, P_fid=None):
    """C_total with the production algebra (``data_likelihood.py`` 1286-1327 at 80bbc5d): without T3, C_data +
    diag(T1 + T2); with T3 = (U, w), the fractional k-coherence term B B^T, B = P_fid U sqrt(w), enters off-diagonal
    only, its diagonal absorbed into T1 by max (never under-counted), T2 topped up on the remaining diagonal.
    PD: C_data PD + diag(>= 0) + (B B^T - diag) with the subtracted diagonal restored inside max(T1, diag)."""
    if t3 is None:
        return C_data + jnp.diag(t1_var + t2_var)
    U, w = t3
    B = jnp.asarray(P_fid)[:, None] * jnp.asarray(U) * jnp.sqrt(jnp.asarray(w))[None, :]
    td = jnp.sum(B ** 2, axis=1)
    C_shape = B @ B.T - jnp.diag(td)
    topup = jnp.maximum(0.0, t2_var - jnp.diag(C_shape))
    return C_data + jnp.diag(jnp.maximum(t1_var, td) + topup) + C_shape
