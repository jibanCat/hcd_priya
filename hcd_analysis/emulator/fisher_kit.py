"""Linearized (Fisher) projections for the gate E audits, the T3 comparison and the E7 projected calibration (gate E
amendment A1 rev 1 section 0, registered PU-0068).

Parameters (the production sample sites): theta9 (unit cube), tau0_amp, dtau0 and the 3 HCD amplitude sites
(alpha_lls, alpha_subdla, alpha_dla_raw on the default branch; eps_lls, m_sub, dla_raw on the KS dN/dX-mapped
branch). Held at their prior centres, as registered: metals and resolution (Model C+ nodes at the log-midpoints of
their log-uniform priors; the uniform a_SiIII at the middle of its range when sampled, else 0; b_res 0), and the HCD
z-slopes / KS exponent deviations (the production centre). Prior precision: theta9 the variance of the uniform
sampling box (width^2 / 12), none for tau0_amp and dtau0 (uniform priors), the production widths for the HCD sites.
The map from parameters to the forward's (tau0 per z, alpha_hcd per z) is tested against the production numpyro
model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import closure_legb as CL
from . import forward as FW
from .dndx_wc import w_c_corrected
from .meanflux_prior import tau0_alpha_priya

TAU0_NAMES = ("tau0_amp", "dtau0")
HCD_NAMES_DEFAULT = ("alpha_lls", "alpha_subdla", "alpha_dla_raw")
HCD_NAMES_KS = ("eps_lls", "m_sub", "dla_raw")


def _ks(ctx):
    return bool(getattr(ctx, "ks_dndx_mapped", False))


def _refuse_unsupported(ctx):
    for f in ("hierarchical_hcd", "hcd_2d_tilt", "per_leg_alpha", "per_leg_zslope"):
        if getattr(ctx, f, False):
            raise NotImplementedError(f"fisher_kit: the {f} branch is not a production configuration")
    if getattr(ctx, "tau0_curv_sigma", None) is not None:
        raise NotImplementedError("fisher_kit: the tau0 curvature diagnostic is not a production configuration")


def param_names(ctx):
    return tuple(f"theta{i}" for i in range(9)) + TAU0_NAMES + (HCD_NAMES_KS if _ks(ctx) else HCD_NAMES_DEFAULT)


def p_centre(ctx, theta9_unit):
    """The parameter vector at ``theta9_unit`` with tau0 at the Kim curve (amp 1, slope 0) and the HCD sites at their
    prior centres."""
    from . import inference as INF
    if _ks(ctx):
        hcd = [0.0, 0.0, float(INF.KS_DNDX_DLA_RAW_MU0)]
    else:
        mu = np.asarray(ctx.alpha_hcd_mu, float)
        hcd = [mu[0], mu[1], float(CL._dla_raw_mu(mu[2]))]
    return np.concatenate([np.asarray(theta9_unit, float), [1.0, 0.0], hcd])


def _zslopes(ctx):
    if getattr(ctx, "marginalize_zslope", False) and getattr(ctx, "zslope_mu", None) is not None:
        return jnp.asarray(ctx.zslope_mu)
    return jnp.asarray(CL.HCD_INCIDENCE_SLOPE)


def forward_inputs(ctx, p):
    """(theta9, tau0 on ctx.z_global, alpha_hcd (n_zg, 3)) at the parameter vector ``p``, as the production model."""
    _refuse_unsupported(ctx)
    zg = jnp.asarray(ctx.z_global)
    p = jnp.asarray(p)
    tau0 = tau0_alpha_priya(zg, p[9], p[10], z_pivot=ctx.tau0_pivot_z) * CL._kim(zg)
    if _ks(ctx):
        from . import inference as INF
        dla_amp = jax.nn.softplus(p[13]) / jax.nn.softplus(INF.KS_DNDX_DLA_RAW_MU0)
        ones = jnp.ones_like(zg)
        fac = jnp.stack([jnp.exp(p[11]) * ones, jnp.exp(p[12]) * ones, dla_amp * ones], axis=-1)
        alpha = CL._simplex_tie_norm(
            w_c_corrected(jnp.asarray(ctx.ks_dndx_ref) * fac, jnp.asarray(ctx.ks_xbar_z), zg)[..., 1:])
    else:
        piv = jnp.stack([p[11], p[12], jax.nn.softplus(p[13])])
        alpha = piv[None, :] * ((1.0 + zg)[:, None] / (1.0 + CL.HCD_Z_PIVOT)) ** _zslopes(ctx)
    return p[:9], tau0, alpha


def nuis_centre(ctx, leg):
    """Metal and resolution nuisances at their prior centres for ``leg`` (forward.predict_leg ``nuis``)."""
    if not leg.metals_on:
        return {}
    if getattr(ctx, "metal_prior", "uniform") == "flatlog2node":
        f = float(np.sqrt(ctx.metal_fnode_lo * ctx.metal_fnode_hi))
        k = float(np.sqrt(getattr(ctx, "metal_knode_lo", 1e-3) * getattr(ctx, "metal_knode_hi", 0.1)))
        nu = {"f_SiIII_nodes": jnp.asarray([f, f]), "k_SiIII_nodes": jnp.asarray([k, k]),
              "metal_node_z": tuple(getattr(ctx, "metal_node_z", (2.2, 4.2)))}
        if leg.name in tuple(getattr(ctx, "metal_siII_legs", ())):
            nu.update(f_SiII_nodes=jnp.asarray([f, f]), k_SiII_nodes=jnp.asarray([k, k]))
        return nu
    a3 = 0.5 * float(ctx.a_siiii_max) if getattr(ctx, "sample_metals", False) else 0.0
    a2 = 0.5 * float(ctx.a_siiii_max) if getattr(ctx, "sample_a_siii", False) else 0.0
    return {"a_SiIII": a3, "a_SiII": a2, "metal_node_z": tuple(getattr(ctx, "metal_node_z", (2.2, 4.2)))}


def kept(leg):
    return np.where(np.isfinite(np.asarray(leg.P_data)))[0]


def mean_fn(ctx, leg, nuis=None):
    """p -> the production model P on the leg's kept bins (no emulator-error terms: the mean only)."""
    zg = np.asarray(ctx.z_global)
    sel = jnp.asarray(np.array([int(np.argmin(np.abs(zg - z))) for z in np.asarray(leg.z)]))
    kr = jnp.asarray(kept(leg))
    nu = nuis_centre(ctx, leg) if nuis is None else nuis

    def f(p):
        th, tau0, alpha = forward_inputs(ctx, p)
        out = FW.predict_leg(ctx.model, th, tau0[sel], alpha[sel], leg=leg, k_com=ctx.k_com_hmpc,
                             pf_stats=ctx.pf_stats, dla_core=ctx.dla_core_leg[leg.name], mf=ctx.mf, nuis=nu)
        return out.P_model[kr]
    return f


def jacobian(ctx, leg, p, nuis=None):
    """dP/dp (N_kept, n_params) by forward-mode autodiff of the mean."""
    return jax.jacfwd(mean_fn(ctx, leg, nuis))(jnp.asarray(p, float))


def prior_precision(ctx):
    """Diagonal prior precision over ``param_names(ctx)`` (registered: box variance for theta9, none for tau0_amp and
    dtau0, production widths for the HCD sites)."""
    from . import inference as INF
    lo = np.asarray(CL._THETA_UNIT_LO if getattr(ctx, "theta_unit_lo", None) is None else ctx.theta_unit_lo, float)
    hi = np.asarray(CL._THETA_UNIT_HI if getattr(ctx, "theta_unit_hi", None) is None else ctx.theta_unit_hi, float)
    if _ks(ctx):
        hcd = [1.0 / INF.KS_DNDX_SIGMA_EPS ** 2, 1.0 / INF.KS_DNDX_SIGMA_MSUB ** 2, 1.0]
    else:
        sd = np.asarray(ctx.alpha_hcd_sigma, float)
        hcd = [1.0 / sd[0] ** 2, 1.0 / sd[1] ** 2, 1.0]
    return np.diag(np.concatenate([12.0 / (hi - lo) ** 2, [0.0, 0.0], hcd]))


def fisher(J, C, P):
    """F = J^T C^-1 J + P."""
    J = np.asarray(J, float)
    return J.T @ cho_solve(cho_factor(np.asarray(C, float)), J) + np.asarray(P, float)


def map_shift(J, C, P, r):
    """The linearized MAP shift of the parameters for a data-space residual r: F^-1 J^T C^-1 r."""
    J = np.asarray(J, float)
    return np.linalg.solve(fisher(J, C, P), J.T @ cho_solve(cho_factor(np.asarray(C, float)), np.asarray(r, float)))


def marginal_sigma(F):
    return np.sqrt(np.diag(np.linalg.inv(np.asarray(F, float))))


def projected(F, g):
    """The shift F^-1 g in units of the marginal sigma of every parameter."""
    return np.linalg.solve(np.asarray(F, float), np.asarray(g, float)) / marginal_sigma(F)


def contraction(F, P):
    """Fisher contraction per parameter, 1 - sigma_post^2 / sigma_prior^2 (project convention; NaN where the prior
    is flat)."""
    post = np.diag(np.linalg.inv(np.asarray(F, float)))
    pr = np.diag(np.asarray(P, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pr > 0, 1.0 - post * pr, np.nan)
