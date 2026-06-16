"""Multi-fidelity (LF -> HF) resolution-correction layer for the Phase-2b emulator.

Forward model (spec):

    P_MF(theta, z, k) = ( rho(k) * f_LF(theta, z, k) + delta(theta, z, k) ) * res_corr(z, k)

where

  * ``f_LF``      -- the FINALIZED LF backbone (per-(z,tau0) Equinox emulator,
                     checkpoints ``checkpoints/final_fold{0..7}.eqx``); its linear
                     per-class P_filt is decoded via ``data.reconstruct_P_filt`` on
                     the LF 172-bin grid, then LOG-LOG interpolated/extrapolated up
                     to the HF k_max so it can be combined with the HF-resolution
                     correction. f_LF is FROZEN (not retrained here).
  * ``delta``     -- a LEARNED LF->HF resolution correction (the ``DeltaHead``),
                     trained on the 6 HF sims. The 6 HR cosmologies are EXACT LF
                     design points, so delta is measured at matched (theta, z, tau0)
                     with target = logP_HF - logP_hat_LF (a log-RATIO; the
                     additive-space delta = rho*f_LF*(exp(g)-1) is recovered exactly).
                     delta MUST be a learned function of (theta, z, k) (the measured
                     ratio is strongly theta- and z-dependent), not a fixed kernel.
  * ``rho(k)``    -- per-k AR1 amplitude (PRIYA-style). RECOMMENDED DEFAULT (validated):
                     rho = the per-k MEAN log-ratio over the HF sims (carried as
                     ``log_rho``), so the learned delta only models the (small,
                     well-constrained) theta/z DEPARTURE from the coherent mean tilt.
                     This is critical in the few-6-HF-sim regime: with rho==1 the
                     delta head must learn the whole coherent high-k tilt AND its
                     theta-dependence from 5 LOSO sims and over-attributes it to n_s
                     (n_s Fisher bias ~14 sigma); with rho = mean log-ratio the n_s
                     bias drops to ~2 sigma (see scripts/build_mf_delta.py). rho==1
                     (delta carries everything) is available via ``--no-rho``.
  * ``res_corr``  -- a FIXED, parameter-independent particle-convergence factor
                     (L15n512 / L15n384), 15z x 59k, applied MULTIPLICATIVELY on the
                     linear P1D, bilinearly interpolated in (z, log10 k) to the eval
                     grid. See ``load_res_corr`` / ``interp_res_corr``.

DESIGN CHOICE (reported to controller): we work in LOG-RATIO space.  Define

    g(theta, z, k) = log P_HF(theta, z, k) - log P_hat_LF(theta, z, k)

The DeltaHead predicts ``g`` (a smooth, low-rank function of k whose amplitude is a
learned function of theta and z). The forward model is then

    log P_MF = log P_hat_LF + g + log res_corr
    P_MF     = ( f_LF * exp(g) ) * res_corr

which is the spec's additive form with ``rho == 1`` and the equivalent additive
``delta = f_LF * (exp(g) - 1)``.  Log-ratio is the natural space for a TILT (the
measured correction is a coherent rising ratio), keeps P_MF strictly positive, and
is what the few-HF-sim regularization is expressed in (an L2 prior pulling g toward
its per-k mean, i.e. toward rho).

Everything in the forward path (LF eval + extrapolation, delta, res_corr interp) is
PURE and DIFFERENTIABLE so the whole P_MF(theta) chain is HMC-usable.

x64 is enabled by importing the emulator package (``hcd_analysis.emulator`` sets it).
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

# The LF backbone norm dict (``lf_norm``) is carried as an Equinox STATIC field of
# ``MultiFidelity`` -- a read-only host-side constant, NOT a parameter.  equinox emits
# a UserWarning "A JAX array is being set as static" for any static field whose leaves
# pass ``eqx.is_array``, and ``is_array`` flags NUMPY arrays too, so a static field of
# numpy norm arrays trips it even though that is exactly the intended use here.  Filter
# that ONE benign message (everything else still surfaces).
_warnings.filterwarnings(
    "ignore", message="A JAX array is being set as static", category=UserWarning)

from .data import (
    load_cache, normalize_params, reconstruct_P_filt, safe_log,
    COARSE_NAMES, DATA_RANGE, Z_LIMITS,
)
from .model import Emulator
from . import train as T

# ---------------------------------------------------------------------------- #
# Paths / constants
# ---------------------------------------------------------------------------- #
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
RES_CORR_DIR = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/res_corr"
N_CLASSES = 4

# KODIAQ reach (cut to validated range; HF reaches ~0.29 s/km, but we evaluate /
# validate to KODIAQ's ~0.1-0.2 s/km). The eval grid below caps here.
KODIAQ_KMAX = 0.2            # s/km -- upper edge of the validated MF eval band
KODIAQ_BAND = (0.07, 0.2)    # the high-k band the LF cannot reach on its own


# ---------------------------------------------------------------------------- #
# res_corr (fixed particle-convergence factor)
# ---------------------------------------------------------------------------- #
# Flat-LCDM constants for the L15 box fundamental (kbox).  These pin the res_corr
# anchor scale (5x kbox) and are the PRIYA/res_corr table's reference cosmology; they
# are FIXED (the anchor is a fixed-cosmology resolution artifact, not a fit param).
_RC_H0 = 70.0
_RC_h = 0.7
_RC_OM = 0.30
_RC_OL = 0.70


def kbox_skm(z):
    """The L15 (15 Mpc/h) box fundamental wavenumber in s/km at redshift ``z``.

    ``kbox(z) = (2*pi / 15) * h * (1+z) / H(z)`` with the flat-LCDM Hubble rate
    ``H(z) = H0 * sqrt(Om*(1+z)^3 + OL)`` (h=0.7, Om=0.30, OL=0.70, H0=70 km/s/Mpc),
    so the (2*pi/L_box) comoving fundamental [h/Mpc] is converted to velocity units
    [s/km] via the Hubble flow ``a*H(z)/h = H(z)/((1+z)*h)``.  Works for scalar or
    array ``z`` (pure jnp -- jittable / differentiable).  At z=3 this is ~0.00376 s/km;
    the res_corr table is anchored to 1 below ``anchor_mult * kbox(z)`` (default 5x).
    """
    z = jnp.asarray(z)
    Hz = _RC_H0 * (_RC_OM * (1.0 + z) ** 3 + _RC_OL) ** 0.5
    return (2.0 * jnp.pi / 15.0) * _RC_h * (1.0 + z) / Hz


def load_res_corr(res_dir=RES_CORR_DIR):
    """Load the fixed L15n512/L15n384 resolution-correction table.

    Returns ``(z_rc (Nz,), logk_rc (Nz, Nk), rc (Nz, Nk))`` with ``z_rc`` SORTED
    ASCENDING.  The k-grid is PER-Z (k in s/km depends on the Hubble flow at each z,
    so the table's rows have DISTINCT k-grids -- it is a true irregular 2D table),
    so ``logk_rc = log10(k)`` is itself ``(Nz, Nk)`` (one ascending log-k row per z).
    ``rc`` is the multiplicative factor (0.82-1.06).  All float64 numpy (host side;
    the differentiable interpolation lives in ``interp_res_corr``).
    """
    rc = np.loadtxt(f"{res_dir}/resolution_correction.txt").astype(np.float64)   # (Nz, Nk)
    kf = np.loadtxt(f"{res_dir}/kfkms.txt").astype(np.float64)                    # (Nz, Nk)
    zo = np.loadtxt(f"{res_dir}/zout.txt").astype(np.float64)                     # (Nz,)
    if rc.shape != kf.shape or rc.shape[0] != zo.shape[0]:
        raise ValueError(f"res_corr shape mismatch: rc{rc.shape} kf{kf.shape} z{zo.shape}")
    if not np.all(np.diff(kf, axis=1) > 0):
        raise ValueError("res_corr per-z k-grid is not strictly increasing")
    # sort z ascending so the searchsorted interpolation is monotone.
    order = np.argsort(zo)
    return zo[order], np.log10(kf[order]), rc[order]


def interp_res_corr(z_rc, logk_rc, rc_vals, z_eval, k_eval, anchor_mult=5.0):
    """Differentiable res_corr at (z_eval, k_eval): bilinear over a PER-Z k-grid.

    ``z_rc (Nz,)`` ascending; ``logk_rc (Nz, Nk)`` per-z ascending log10-k grids;
    ``rc_vals (Nz, Nk)``.  ``z_eval`` scalar, ``k_eval (K,)``.  Returns ``(K,)``.
    Because the k-grid differs per z-row, we interpolate in log10-k WITHIN the two
    bracketing z-rows (each on its own k-grid) and then linearly blend in z.
    CLAMPED-edge (constant extrapolation) beyond the table support in BOTH z and k
    -- the table covers z in [2.2, 5.0] and k in [0.003, 0.242] s/km, which spans
    the KODIAQ band; the clamp guards the few cache rows at z<2.2 / k below the table
    k_min.  Fully jittable / differentiable (linear interp via searchsorted + lerp).

    LOW-k ANCHOR (Task 1.1):  the raw L15n512/L15n384 table has a spurious +6.3%
    bump at its k_min (z=3: k~0.0037 s/km, rc~1.063) -- a few-mode resolution artifact
    AT the L15 box fundamental -- that the old clamped edge injected into the data's
    lowest bins.  We force res_corr -> 1 BELOW ``anchor_mult * kbox_skm(z_eval)``
    (default 5x the box fundamental) via a smooth tanh blend (width 0.12 dex in
    log10 k), leaving res_corr UNCHANGED well above the anchor:

        w = 0.5*(1 + tanh((log10 k - log10(anchor_mult*kbox(z))) / 0.12))
        rc <- 1 + (rc - 1)*w

    so w -> 0 (rc -> 1) far below the anchor and w -> 1 (rc unchanged) far above.
    ``anchor_mult=0.0`` DISABLES the anchor and reproduces the raw clamped table
    exactly (the regression reference; the ``anchor_mult>0`` guard is a STATIC python
    comparison on the kwarg, so the branch is traced-value-free / jittable)."""
    z_rc = jnp.asarray(z_rc); logk_rc = jnp.asarray(logk_rc); rc_vals = jnp.asarray(rc_vals)
    logk = jnp.log10(k_eval)

    # z bracket (clamped so [iz, iz+1] is valid), constant-edge in z.
    iz = jnp.clip(jnp.searchsorted(z_rc, z_eval, side="right") - 1, 0, z_rc.shape[0] - 2)
    wz = jnp.clip((z_eval - z_rc[iz]) / (z_rc[iz + 1] - z_rc[iz]), 0.0, 1.0)

    def row_interp(zi):
        # 1-D log-log interp on z-row zi's OWN k-grid; jnp.interp clamps at edges.
        return jnp.interp(logk, logk_rc[zi], rc_vals[zi])

    c0 = row_interp(iz)              # (K,)
    c1 = row_interp(iz + 1)         # (K,)
    rc = c0 * (1 - wz) + c1 * wz

    if anchor_mult > 0:
        # smooth tanh blend to rc==1 below anchor_mult*kbox(z), width 0.12 dex.
        logk_anchor = jnp.log10(anchor_mult * kbox_skm(z_eval))
        w = 0.5 * (1.0 + jnp.tanh((logk - logk_anchor) / 0.12))
        rc = 1.0 + (rc - 1.0) * w
    return rc


# ---------------------------------------------------------------------------- #
# LF backbone evaluation + log-log extrapolation to the HF eval grid
# ---------------------------------------------------------------------------- #
def _freeze(model):
    """Return a copy of ``model`` whose ARRAY leaves are stop_gradient'd.

    The LF backbone is FROZEN: HMC needs grads of logP wrt the INPUT theta (which
    flow through the LF forward), but NEVER wrt the LF WEIGHTS.  A plain
    ``eqx.field(static=...)`` does NOT freeze a sub-Module (equinox recurses into it
    and its arrays stay dynamic leaves -- verified), so we explicitly
    ``stop_gradient`` the model's parameters here.  ``jax.lax.stop_gradient`` is a
    no-op on the forward value and zeroes the backward pass to the weights, so the
    LF weights can never receive a gradient regardless of how the MF is differentiated.
    """
    params, static = eqx.partition(model, eqx.is_array)
    params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
    return eqx.combine(params, static)


def lf_predict_logP(lf_model, lf_norm, x, tau0):
    """LF backbone -> log P_filt (4, K_lf) on the LF native grid, differentiable.

    ``x`` (10,) = [params_unit(9), z_unit(1)]; ``tau0`` scalar.  Returns log P_filt
    (natural log) for the 4 coarse classes on the LF 172-bin grid.  This is the
    backbone's prediction in LOG space (the space the LF<->HF ratio lives in).  The
    LF WEIGHTS are stop_gradient'd (frozen): grads flow to the inputs (theta, tau0)
    but never to the LF parameters."""
    pf = lf_norm["P_filt"]
    pred = _freeze(lf_model)(x, tau0)
    base = pred["P_filt_base"] * jnp.asarray(pf["sig_marg"]) + jnp.asarray(pf["mu_marg"])
    logP = base + jnp.asarray(pf["sig_cosmo"]) * pred["P_filt_resid"]     # (4, K_lf)
    return logP


def loglog_interp_extrap(logk_src, logP_src, logk_dst, n_tail=6):
    """Differentiable log-log interpolation + linear-in-log-k tail EXTRAPOLATION.

    ``logk_src (Ks,)`` ascending; ``logP_src (Ks,)`` the source log P1D; ``logk_dst
    (Kd,)`` the target log-k.  In the source support we linearly interpolate logP in
    log-k; BEYOND the source k_max we EXTRAPOLATE with the slope fit (least squares)
    over the last ``n_tail`` source bins (a smooth power-law tail in log-log).  Below
    the source k_min we clamp to the first value (the cache k_min is below the data
    k_min, so this edge is never used in-range).

    NOTE: the LF backbone's k_max (~0.069 s/km) is BELOW the KODIAQ band, so for the
    high-k band the LF prediction is a TAIL EXTRAPOLATION -- which the LF<->HF ratio
    measurements show is unreliable above the LF Nyquist (the LF P1D rolls off
    steeply there).  The DeltaHead is what corrects this; this function just gives a
    smooth, differentiable LF "backbone" value to correct FROM.  Fully jittable.
    """
    logk_src = jnp.asarray(logk_src); logP_src = jnp.asarray(logP_src)
    logk_dst = jnp.asarray(logk_dst)
    Ks = logk_src.shape[0]
    # least-squares slope/intercept over the last n_tail bins (smooth tail).
    kt = logk_src[Ks - n_tail:]
    pt = logP_src[Ks - n_tail:]
    kbar = jnp.mean(kt); pbar = jnp.mean(pt)
    slope = jnp.sum((kt - kbar) * (pt - pbar)) / jnp.maximum(jnp.sum((kt - kbar) ** 2), 1e-30)

    kmax = logk_src[-1]
    Pmax = logP_src[-1]
    # interpolate within support; jnp.interp clamps at edges (left/right = endpoints).
    interp = jnp.interp(logk_dst, logk_src, logP_src)
    extrap = Pmax + slope * (logk_dst - kmax)
    return jnp.where(logk_dst > kmax, extrap, interp)


def lf_logP_on_eval_grid(lf_model, lf_norm, lf_logk, x, tau0, eval_logk, n_tail=6):
    """LF backbone log P_filt (4, K_eval) on the HF eval grid, per class.

    Evaluates ``lf_predict_logP`` on the LF native grid then log-log interp/extrap
    each class to ``eval_logk``.  ``lf_logk (K_lf,)`` is log10 of the LF k-grid;
    ``eval_logk (K_eval,)`` is log10 of the eval k-grid.  Differentiable in (x, tau0).
    """
    logP_lf = lf_predict_logP(lf_model, lf_norm, x, tau0)          # (4, K_lf) natural log
    # interp/extrap works in log10-k; logP itself stays natural-log (just the
    # y-values being interpolated, monotone-agnostic).
    def per_class(lp):
        return loglog_interp_extrap(lf_logk, lp, eval_logk, n_tail=n_tail)
    return jax.vmap(per_class)(logP_lf)                            # (4, K_eval)


# ---------------------------------------------------------------------------- #
# Smooth k-basis for the delta head (low-complexity, heavily regularized)
# ---------------------------------------------------------------------------- #
def smooth_k_basis(eval_logk, n_basis=4):
    """A small, FIXED smooth basis over log10-k for the log-ratio g(k): (n_basis, K).

    Row 0 is the constant (the coherent amplitude / mean log-ratio).  Rows 1.. are
    shifted, rescaled Chebyshev-like polynomials in the normalized log-k coordinate
    u in [-1, 1] (so the basis is smooth and low-order -- a TILT + mild curvature,
    not a wiggly per-k kernel).  With ``n_basis=4`` g(k) = c0 + c1*T1(u) + c2*T2(u)
    + c3*T3(u): mean, slope, curvature, S-shape -- enough for the measured rising
    coherent tilt, few enough to fit from 6 HF sims.  Returns float64 (n_basis, K).
    """
    u = jnp.asarray(eval_logk)
    u = 2.0 * (u - u.min()) / jnp.maximum(u.max() - u.min(), 1e-12) - 1.0   # -> [-1, 1]
    cols = [jnp.ones_like(u)]
    if n_basis >= 2:
        cols.append(u)                                  # T1
    if n_basis >= 3:
        cols.append(2.0 * u ** 2 - 1.0)                 # T2
    if n_basis >= 4:
        cols.append(4.0 * u ** 3 - 3.0 * u)             # T3
    for d in range(5, n_basis + 1):
        # higher Chebyshev via recurrence T_n = 2u T_{n-1} - T_{n-2}
        cols.append(2.0 * u * cols[-1] - cols[-2])
    return jnp.stack(cols[:n_basis], axis=0)            # (n_basis, K)


class DeltaHead(eqx.Module):
    """LEARNED LF->HF log-ratio correction g(theta, z, k) -- low-complexity MLP.

    Maps the conditioning vector ``[params_unit(9), z_unit(1), tau0(1)]`` (12-dim)
    through a SMALL MLP to ``n_basis`` smooth-basis coefficients PER CLASS, decoding
    g_c(k) = coeffs_c @ basis (basis from ``smooth_k_basis``).  g is the log-RATIO
    log P_HF - log P_hat_LF.

    HEAVY REGULARIZATION (only 6 HF sims):
      * tiny width (default 1 hidden layer of 16 units) -- a near-linear map of theta;
      * a low-order smooth k-basis (default n_basis=4: mean/slope/curve/S) so g(k) is
        a TILT, never a per-k wiggle;
      * the coefficient L2 + a "mean-ratio prior" handled in the loss (pull the
        per-k MEAN of g toward rho's log; see ``mf_delta_loss``).

    The head is differentiable in (theta, tau0); ``n_basis``/``n_classes``/widths are
    STATIC (set shapes).  Output: ``g`` (n_classes, K_eval) = coeffs @ basis.
    """
    layers: list
    n_basis: int = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)

    def __init__(self, n_basis=4, n_classes=N_CLASSES, in_dim=11,
                 width=16, n_layers=1, key=None):
        # in_dim = 9 params + z_unit + tau0 = 11 by default.
        if n_layers < 0:
            raise ValueError("n_layers must be >= 0")
        out_dim = n_classes * n_basis
        ks = jax.random.split(key, n_layers + 1)
        dims = [in_dim] + [width] * n_layers
        hidden = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(n_layers)]
        out = eqx.nn.Linear(dims[-1], out_dim, key=ks[n_layers])
        self.layers = hidden + [out]
        self.n_basis = n_basis
        self.n_classes = n_classes

    def coeffs(self, cond):
        """Conditioning (in_dim,) -> per-class basis coeffs (n_classes, n_basis)."""
        h = cond
        for lin in self.layers[:-1]:
            h = jax.nn.gelu(lin(h))
        y = self.layers[-1](h)
        return y.reshape(self.n_classes, self.n_basis)

    def __call__(self, cond, basis):
        """g (n_classes, K) = coeffs(cond) @ basis. ``basis`` is (n_basis, K)."""
        return self.coeffs(cond) @ basis                # (n_classes, K)


class FixedMeanHead(eqx.Module):
    """The DEFAULT (delta_mode='none') head: a FIXED, THETA-INDEPENDENT z-(and,
    optionally, tau0-)resolved mean correction on top of the per-k ``log_rho``.

    The validated MF default (commit d93fadc / scripts/diag_mf_complexity.py): the
    LF->HF correction is ~94-95% the fixed resolution tilt and only ~5% theta-
    departure, and the learned MLP delta-head OVER-FITS that 5% on 6 HF sims (n_s
    Fisher bias 4 sigma).  The rho(k,z)-ONLY model (this head + log_rho) cures it
    (worst |n_s|~2 sigma, |A_p|~1.4 sigma -- best for A_p, the priority param).

    ``MultiFidelity.g`` returns ``log_rho[None,:] + head(cond, basis)``.  This head
    returns ``g(z,tau0,k) - log_rho`` (the fixed mean MINUS the per-k global mean),
    so the combined g is exactly the THETA-INDEPENDENT fixed-mean LF->HF log-ratio.
    ALL theta-dependence stays in f_LF; there is NO learned theta-correction.

    TWO RESOLUTIONS (selected by ``resolved``, a STATIC bool):

    * ``resolved=False`` (POOLED, the historical default).  ``g == gbar(z,k)`` only.
      ``gbar_tab`` (Nz, n_classes, K) is the per-z fixed-mean table MINUS log_rho;
      ``z_tab`` (Nz,) the sorted physical z grid.  dg/dtau0 == 0 IDENTICALLY -- this
      pools over tau0 and ignores ``cond[10]``.

    * ``resolved=True`` (T1, the tau0+z-resolved correction; review-LOCKED form).
      ``g(z,tau0,k) = gbar_z(z,k) + gbar_tau(tau0,k) + a(k)*u_z(z)*u_tau(tau0)`` --
      SEPARABLE in (z) and (tau0) PLUS ONE rank-1 (z x tau0) interaction.  This
      captures the documented (z,tau0) sign-flip (cosmology lens) with too few DOF to
      alias the n_s tilt into the tau0 axis (the failure mode of a free 2-D table over
      only 6 clustered HR sims, which got delta_mode='mlp' retired at 4 sigma).  The
      ``gbar_z_tab`` (Nz,C,K) term is the table MINUS log_rho (so log_rho cancels and
      the combined g recovers the fixed mean); the tau0-marginal ``gtau_tab``
      (Ntau,C,K) and the rank-1 term ``a_k``(C,K)*``u_z``(Nz)*``u_tau``(Ntau) carry NO
      net log_rho (they are zero-mean departures).  At eval the head reads ``cond[9]``
      (z_unit) and ``cond[10]`` (tau0) and CLAMPED-linear interpolates each marginal
      (constant-edge beyond the table; mirrors ``interp_res_corr``), so g is
      C0-continuous and NaN-free over/beyond the (z,tau0) grid -- NUTS-safe.

    Differentiable & jittable in both modes; ``coeffs`` returns zeros (no basis is
    used, so the coeff-L2 term is a no-op for this head).
    """
    gbar_tab: jax.Array        # (Nz, n_classes, K)  pooled z-trend (gbar - log_rho)
    z_tab: jax.Array           # (Nz,) physical z, ascending
    # resolved-mode extra components (zero-sized arrays in pooled mode):
    gtau_tab: jax.Array        # (Nrung, n_classes, K)  zero-mean rung-marginal
    tau_tab: jax.Array         # (Nrung,) rung-INDEX grid 0..Nrung-1
    tau_by_z: jax.Array        # (Nz, Nrung)  physical tau0 of each rung at each z
    a_k: jax.Array             # (n_classes, K) rank-1 k-shape
    u_z: jax.Array             # (Nz,) rank-1 z-profile
    u_tau: jax.Array           # (Nrung,) rank-1 rung-profile
    n_classes: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)

    def __init__(self, gbar_tab, z_tab, n_basis=4, *, resolved=False,
                 gtau_tab=None, tau_tab=None, tau_by_z=None,
                 a_k=None, u_z=None, u_tau=None):
        self.gbar_tab = jnp.asarray(gbar_tab)
        self.z_tab = jnp.asarray(z_tab)
        self.n_classes = int(jnp.asarray(gbar_tab).shape[1])
        self.n_basis = int(n_basis)
        self.resolved = bool(resolved)
        K = self.gbar_tab.shape[-1]
        C = self.n_classes
        nz = self.z_tab.shape[0]
        if resolved:
            if any(v is None for v in
                   (gtau_tab, tau_tab, tau_by_z, a_k, u_z, u_tau)):
                raise ValueError(
                    "resolved=True requires gtau_tab, tau_tab, tau_by_z, a_k, "
                    "u_z, u_tau")
            self.gtau_tab = jnp.asarray(gtau_tab)
            self.tau_tab = jnp.asarray(tau_tab)
            self.tau_by_z = jnp.asarray(tau_by_z)
            self.a_k = jnp.asarray(a_k)
            self.u_z = jnp.asarray(u_z)
            self.u_tau = jnp.asarray(u_tau)
        else:
            # pooled mode: 1-row placeholder marginals (never read in __call__).
            self.gtau_tab = jnp.zeros((1, C, K))
            self.tau_tab = jnp.zeros((1,))
            self.tau_by_z = jnp.zeros((nz, 1))
            self.a_k = jnp.zeros((C, K))
            self.u_z = jnp.zeros((nz,))
            self.u_tau = jnp.zeros((1,))

    def coeffs(self, cond):
        return jnp.zeros((self.n_classes, self.n_basis))

    @staticmethod
    def _interp_zfirst(grid, tab, q):
        """Clamped-linear interp of ``tab`` (N, C, K) along axis 0 at scalar ``q``
        over the ascending 1-D ``grid`` (N,).  jnp.interp clamps at both edges
        (constant-edge extrapolation).  Returns (C, K)."""
        C, K = tab.shape[1], tab.shape[2]
        flat = tab.reshape(grid.shape[0], -1)              # (N, C*K)
        vals = jax.vmap(lambda col: jnp.interp(q, grid, col), in_axes=1)(flat)
        return vals.reshape(C, K)

    def _rung_coord(self, z_phys, tau0):
        """Map a continuous (z, tau0) to a fractional RUNG coordinate, clamped.

        tau0 is a deterministic monotone-in-rung function of (rung, z): tau_by_z
        (Nz, Nrung).  We blend the two bracketing z-rows' tau0 ladders, then invert
        (interp tau0 -> rung index on the blended, ascending ladder).  jnp.interp
        clamps at both ends, so tau0 outside the measured ladder pins to rung 0 /
        rung Nrung-1 (constant-edge).  Differentiable in (z, tau0)."""
        zt = self.z_tab
        iz = jnp.clip(jnp.searchsorted(zt, z_phys, side="right") - 1, 0, zt.shape[0] - 2)
        wz = jnp.clip((z_phys - zt[iz]) / (zt[iz + 1] - zt[iz]), 0.0, 1.0)
        ladder = self.tau_by_z[iz] * (1 - wz) + self.tau_by_z[iz + 1] * wz  # (Nrung,)
        return jnp.interp(tau0, ladder, self.tau_tab)      # fractional rung, clamped

    def __call__(self, cond, basis):
        z_unit = cond[9]
        z_phys = z_unit * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
        gz = self._interp_zfirst(self.z_tab, self.gbar_tab, z_phys)    # (C,K)
        if not self.resolved:
            return gz
        tau0 = cond[10]
        rung = self._rung_coord(z_phys, tau0)                          # fractional rung
        gt = self._interp_zfirst(self.tau_tab, self.gtau_tab, rung)    # (C,K)
        # rank-1 (z x rung) interaction: a_k(C,K) * u_z(z) * u_tau(rung), CLAMPED.
        uz = jnp.interp(z_phys, self.z_tab, self.u_z)                  # scalar
        ut = jnp.interp(rung, self.tau_tab, self.u_tau)               # scalar
        return gz + gt + self.a_k * (uz * ut)


class GlobalLinearHead(eqx.Module):
    """delta_mode='linear': rho + a SINGLE GLOBAL linear-in-theta amplitude * a
    fixed smooth per-class k-shape (the sweep's variant (b)).

    The MINIMAL smooth theta-correction, for users who want a WEAK theta term WITH
    an explicit A_p-direction prior (the global linear amplitude can be priored
    toward the A_p direction).  g_delta(theta,z,k) = a(theta) * shape_c(k), where
    a(theta) = w . (cond - 0.5) + b is ONE scalar linear functional of the
    conditioning (9 cosmo params + z_unit + tau0), and shape_c(k) = S[c] @ basis is
    a per-class smooth k-shape.  Params: (in_dim + 1) + n_classes*n_basis -- a tiny,
    smooth, near-fixed correction.  Differentiable & jittable; coeffs() returns
    a(theta)*S (per-class basis coeffs, so the coeff-L2 term controls it).
    """
    w: jax.Array               # (in_dim,) linear weights (cosmo + z_unit + tau0)
    b: jax.Array               # () scalar bias
    S: jax.Array               # (n_classes, n_basis) per-class smooth k-shape coeffs
    n_classes: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, in_dim=11, n_classes=N_CLASSES, n_basis=4, key=None):
        k1, k2 = jax.random.split(key)
        self.w = 1e-3 * jax.random.normal(k1, (in_dim,))
        self.b = jnp.zeros(())
        self.S = 1e-3 * jax.random.normal(k2, (n_classes, n_basis))
        self.n_classes = int(n_classes)
        self.n_basis = int(n_basis)

    def amp(self, cond):
        # centre the unit-cube inputs at 0.5 so a==b at the fiducial centre.
        return jnp.dot(self.w, cond - 0.5) + self.b

    def coeffs(self, cond):
        return self.amp(cond) * self.S                 # (n_classes, n_basis)

    def __call__(self, cond, basis):
        return self.coeffs(cond) @ basis               # (n_classes, K)


def make_cond(x, tau0):
    """Conditioning vector for the DeltaHead: [params_unit(9), z_unit(1), tau0(1)].

    ``x`` (10,) = [params_unit(9), z_unit(1)] (the LF encoder input), ``tau0`` scalar.
    Returns (11,).  Differentiable / jittable."""
    return jnp.concatenate([jnp.asarray(x), jnp.atleast_1d(tau0)])


# ---------------------------------------------------------------------------- #
# The full multi-fidelity forward model
# ---------------------------------------------------------------------------- #
class MultiFidelity(eqx.Module):
    """P_MF = ( rho(k,z) * f_LF ) * res_corr  -- the full MF forward model.

    DEFAULT forward model (``delta_mode='none'``, validated -- commit d93fadc /
    scripts/diag_mf_complexity.py):

        log P_MF = log P_hat_LF(theta,z,k) + rho(k,z) + log res_corr(z,k)

    where ``rho(k,z) = log_rho(k) + gbar(z,k)-correction`` is the FIXED per-(k,z)
    mean LF->HF log-ratio (THETA-INDEPENDENT: ALL theta-dependence lives in f_LF).
    This is the rho(k,z)-ONLY model -- the learned MLP delta-head is REMOVED from the
    default path because it OVER-FITS the theta-gradient on 6 HF sims (n_s Fisher
    bias 4 sigma; the LF->HF correction is 94-95% fixed resolution tilt, only ~5%
    theta-departure).  rho(k,z)-only cures it (worst |n_s|~2 sigma, |A_p|~1.4 sigma --
    best for A_p, the priority param).

    The trainable submodule is held in ``delta_head`` and selected by
    ``delta_mode`` (carried as a STATIC field, informational):
      * ``'none'``   -- ``FixedMeanHead``: g == gbar(z,k), no theta term (DEFAULT).
      * ``'linear'`` -- ``GlobalLinearHead``: rho + ONE global linear-in-theta
                        amplitude * a fixed per-class k-shape (the minimal smooth
                        theta-correction, for users wanting a weak theta term WITH an
                        A_p-direction prior; sweep variant (b)).
      * ``'mlp'``    -- ``DeltaHead``: the over-fit learned MLP (kept for diagnostics
                        / ablation ONLY -- NOT recommended; n_s Fisher bias 4 sigma).
    All three share the ``(cond, basis) -> (n_classes, K)`` head interface so the
    forward (``g``) is identical; only the head's functional form differs.

    Also holds the FROZEN LF backbone (``lf_model`` + ``lf_norm``), the FIXED smooth
    k-basis, the per-k ``log_rho``, and the FIXED res_corr table.  All forward
    methods are pure & differentiable in (theta, tau0).

    Grids (all log10-k, jnp float64, STATIC arrays):
      * ``lf_logk``   -- the LF backbone's native 172-bin grid (for the LF eval);
      * ``eval_logk`` -- the HF eval grid the MF prediction lives on (KODIAQ-capped).

    The LF backbone is FROZEN via ``jax.lax.stop_gradient`` on its weights inside the
    forward (see ``_freeze``/``lf_predict_logP``): HMC grads of logP flow to the
    INPUT theta through the LF forward, but NEVER to the LF parameters, so only the
    delta_head (when it has trainable leaves) is ever trained.  (NB: ``lf_model`` is a
    NORMAL dynamic field -- NOT a ``static`` field: a static field holding a
    sub-Module does NOT freeze it because equinox recurses into it AND the
    static-captured copy can desync from the differentiated leaves, so the
    stop_gradient would not bind; keeping it dynamic gives a single, unambiguous leaf
    set that ``_freeze``'s stop_gradient pins to zero gradient.)  ``lf_norm`` is a
    dict of plain numpy arrays (host-side constants read via ``jnp.asarray``) and
    stays static.  NB the default ``FixedMeanHead`` has NO trainable leaves (its
    ``gbar_tab``/``z_tab`` are fixed host-side tables, dynamic leaves but not
    optimized), so the default MF forward is a PURE fixed function of (theta,tau0)
    that is still fully HMC-differentiable in theta through f_LF.
    """
    delta_head: eqx.Module
    lf_model: Emulator
    basis: jax.Array
    log_rho: jax.Array
    eval_logk: jax.Array
    lf_logk: jax.Array
    z_rc: jax.Array
    logk_rc: jax.Array
    rc_vals: jax.Array
    n_tail: int = eqx.field(static=True)
    delta_mode: str = eqx.field(static=True)
    lf_norm: object = eqx.field(static=True)

    def __init__(self, lf_model, lf_norm, eval_logk, lf_logk, delta_head, basis,
                 log_rho, z_rc, logk_rc, rc_vals, n_tail=6, delta_mode="none"):
        self.lf_model = lf_model
        # lf_norm is a STATIC field: a fixed host-side constant (the LF backbone's
        # P_filt/channel norm dict), read via jnp.asarray in the forward and never
        # differentiated.  Coerce any JAX arrays in it to numpy (production lf_norm
        # from load_checkpoint is already numpy).  NB equinox warns "A JAX array is
        # being set as static" for static fields whose leaves pass `is_array` -- and
        # `is_array` flags numpy arrays too -- so that (benign, expected) warning is
        # filtered at module import; see the filterwarnings call at module top.
        self.lf_norm = jax.tree_util.tree_map(
            lambda v: np.asarray(v) if isinstance(v, jax.Array) else v, lf_norm)
        self.eval_logk = jnp.asarray(eval_logk)
        self.lf_logk = jnp.asarray(lf_logk)
        self.delta_head = delta_head
        self.basis = jnp.asarray(basis)
        self.log_rho = jnp.asarray(log_rho)
        self.z_rc = jnp.asarray(z_rc)
        self.logk_rc = jnp.asarray(logk_rc)
        self.rc_vals = jnp.asarray(rc_vals)
        self.n_tail = n_tail
        if delta_mode not in ("none", "linear", "mlp"):
            raise ValueError(
                f"delta_mode must be 'none'|'linear'|'mlp', got {delta_mode!r}")
        self.delta_mode = delta_mode

    # -- pieces -------------------------------------------------------------- #
    def lf_logP(self, x, tau0):
        """LF backbone log P_filt (4, K_eval) on the eval grid (interp/extrap)."""
        return lf_logP_on_eval_grid(self.lf_model, self.lf_norm, self.lf_logk,
                                    x, tau0, self.eval_logk, n_tail=self.n_tail)

    def g(self, x, tau0):
        """Log-ratio correction g (4, K_eval) = log_rho + delta_head(cond, basis).

        For the DEFAULT ``delta_mode='none'`` the head is a ``FixedMeanHead`` whose
        output is ``gbar(z,k)-log_rho``, so ``g == gbar(z,k)`` -- the FIXED,
        THETA-INDEPENDENT z-resolved mean LF->HF log-ratio (all theta-dependence is
        in f_LF).  For ``'linear'``/``'mlp'`` the head adds a (small) learned
        theta-correction on top of log_rho."""
        cond = make_cond(x, tau0)
        return self.log_rho[None, :] + self.delta_head(cond, self.basis)

    def res_corr(self, z):
        """res_corr (K_eval,) at scalar redshift z (broadcasts over classes)."""
        k_eval = jnp.power(10.0, self.eval_logk)
        return interp_res_corr(self.z_rc, self.logk_rc, self.rc_vals, z, k_eval)

    # -- forward ------------------------------------------------------------- #
    def logP_mf(self, x, tau0, apply_res_corr=True):
        """log P_MF (4, K_eval): logP_LF + g + (log res_corr).  Differentiable."""
        z = x[..., 9]                                    # z_unit is x[9]
        # NOTE: res_corr table is in PHYSICAL z, but the LF input x[9] is z_unit.
        # convert: z = z_unit*(z_hi-z_lo)+z_lo.
        z_phys = z * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
        lp = self.lf_logP(x, tau0) + self.g(x, tau0)     # (4, K_eval)
        if apply_res_corr:
            lp = lp + jnp.log(self.res_corr(z_phys))[None, :]
        return lp

    def P_mf(self, x, tau0, apply_res_corr=True):
        """LINEAR P_MF (4, K_eval).  Strictly positive; differentiable in theta.

        JIT NOTE (HMC): jit the MODULE with ``eqx.filter_jit`` (or close it over a
        lambda: ``jax.jit(lambda x, t: mf.P_mf(x, t))``).  Do NOT do
        ``jax.jit(mf.P_mf)`` -- jitting a bound method makes the module a traced
        positional arg and JAX tries to hash it as static, which fails on the
        DeltaHead's ``layers`` python list (``TypeError: unhashable type: 'list'``).
        ``vmap`` over (x, tau0) is fine: ``jax.vmap(lambda x, t: mf.P_mf(x, t))``.
        """
        return jnp.exp(self.logP_mf(x, tau0, apply_res_corr=apply_res_corr))

    def __call__(self, x, tau0):
        return self.P_mf(x, tau0)


# ---------------------------------------------------------------------------- #
# HR<->LF matching + delta target measurement
# ---------------------------------------------------------------------------- #
def _rowkey(d):
    """(params_unit rounded, z rounded, alpha_idx) key for exact HR<->LF matching."""
    pu = np.round(normalize_params(d["params"]), 6)
    z = np.round(d["z_grid"], 4)
    a = d["alpha_idx"].astype(int)
    return [(tuple(pu[i]), z[i], a[i]) for i in range(len(z))]


def match_hr_to_lf(lf, hr):
    """Return ``[(hr_row, lf_row), ...]`` exact matches on (params_unit, z, alpha)."""
    idx = {k: i for i, k in enumerate(_rowkey(lf))}
    return [(h, idx[k]) for h, k in enumerate(_rowkey(hr)) if k in idx]


def build_eval_grid(hr, k_max=KODIAQ_KMAX, n_k=48):
    """A log-uniform eval k-grid up to ``k_max`` s/km, spanning LF mid-band -> KODIAQ.

    Lower edge = the LF data k_min (DATA_RANGE['k_min']); upper edge = ``k_max``
    (KODIAQ-capped, < the HF k_max so delta is interpolated, not extrapolated, in HF).
    Returns ``(k_eval (n_k,), eval_logk (n_k,))`` float64.
    """
    k_lo = DATA_RANGE["k_min"]
    eval_logk = np.linspace(np.log10(k_lo), np.log10(k_max), n_k)
    return np.power(10.0, eval_logk), eval_logk


def hr_logP_on_eval_grid(hr, hr_row, eval_logk):
    """HR truth log P_filt (4, K_eval) interpolated (log-log) onto the eval grid.

    Per class: log-log interp of the HR cache's finite, positive P_filt bins onto
    ``eval_logk``.  Returns (4, K_eval) with NaN where a class has < 2 finite bins
    (e.g. an empty HCD class) so the loss can mask it.  Host-side numpy."""
    Phr = hr["P_filt"][hr_row]                           # (4, K_hr)
    khr = hr["kfkms"][hr_row]                            # (K_hr,)
    out = np.full((N_CLASSES, len(eval_logk)), np.nan)
    for ci in range(N_CLASSES):
        m = np.isfinite(Phr[ci]) & (Phr[ci] > 0) & np.isfinite(khr)
        if m.sum() < 2:
            continue
        out[ci] = np.interp(eval_logk, np.log10(khr[m]), np.log(Phr[ci][m]),
                            left=np.nan, right=np.nan)
    return out


def measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk,
                          pairs, n_tail=6):
    """Measure g = log P_HF - log P_hat_LF on every matched HR row, on the eval grid.

    Returns a dict of host-side numpy arrays over the matched rows (length M):
      ``x``      (M, 10)   LF encoder inputs (the matched LF rows' x);
      ``tau0``   (M,)      tau0 of the matched rows;
      ``hr_row`` (M,)      HR cache row index;
      ``lf_row`` (M,)      LF cache row index;
      ``g``      (M, 4, K) the measured log-ratio target (NaN where HR class empty
                           or HR interp out of HR support);
      ``logP_lf``(M, 4, K) the LF backbone log P on the eval grid (for diagnostics);
      ``logP_hr``(M, 4, K) the HR truth log P on the eval grid.
    g is finite ONLY where both the LF prediction and the HR truth are finite.
    """
    Mx, Mt, Mh, Ml, Ma = [], [], [], [], []
    G, LPlf, LPhr = [], [], []
    lf_logk_j = jnp.asarray(lf_logk)
    eval_logk_j = jnp.asarray(eval_logk)
    # alpha_idx (the tau0 RUNG) of each matched LF row, if the cache carries it -- the
    # tau0-resolved fixed mean (T1) bins by RUNG (a clean per-(z,rung) 6-sim mean),
    # because the continuous tau0 = -log(target_F) is a deterministic, bit-identical
    # function of (rung, z), so raw-tau0 cells would each hold a SINGLE sim.
    aidx = lf.get("alpha_idx")
    for h, l in pairs:
        x = jnp.asarray(lf["x"][l]); tau0 = jnp.asarray(lf["tau0"][l])
        lp_lf = np.asarray(lf_logP_on_eval_grid(
            lf_model, lf_norm, lf_logk_j, x, tau0, eval_logk_j, n_tail=n_tail))  # (4,K)
        lp_hr = hr_logP_on_eval_grid(hr, h, eval_logk)                          # (4,K)
        g = lp_hr - lp_lf
        Mx.append(np.asarray(lf["x"][l])); Mt.append(float(lf["tau0"][l]))
        Mh.append(h); Ml.append(l)
        Ma.append(int(aidx[l]) if aidx is not None else -1)
        G.append(g); LPlf.append(lp_lf); LPhr.append(lp_hr)
    return dict(
        x=np.asarray(Mx), tau0=np.asarray(Mt), alpha_idx=np.asarray(Ma),
        hr_row=np.asarray(Mh), lf_row=np.asarray(Ml),
        g=np.asarray(G), logP_lf=np.asarray(LPlf), logP_hr=np.asarray(LPhr),
    )


# ---------------------------------------------------------------------------- #
# Delta-head training (heavily regularized; 6 HF sims, HF-LOSO)
# ---------------------------------------------------------------------------- #
def mean_log_ratio_rho(targets, eval_logk):
    """Per-k MEAN log-ratio over all measured rows/classes -> a PRIYA-style rho(k).

    ``log_rho(k) = < g(:, :, k) >`` (nan-mean over rows and classes).  This is the
    optional per-k AR1 amplitude (in log space).  If used as ``log_rho`` in the
    ``MultiFidelity``, the DeltaHead then only learns the theta/z DEPARTURE from this
    mean tilt.  Returns (K,) float64.
    """
    g = targets["g"]                                     # (M,4,K)
    with np.errstate(invalid="ignore"):
        return np.nanmean(g.reshape(-1, g.shape[-1]), axis=0)


def fixed_mean_table(targets, log_rho, *, train_mask_rows=None):
    """Per-z fixed-mean table ``gbar(z,k) - log_rho`` for the FixedMeanHead (default).

    ``gbar(z,k)`` is the THETA-INDEPENDENT mean LF->HF log-ratio at each redshift,
    averaged over ALL sims & alpha at that z (the population fixed mean).  We
    subtract the per-k global ``log_rho`` so the table is what ``FixedMeanHead``
    adds ON TOP of ``log_rho`` (then ``MultiFidelity.g == log_rho + (gbar-log_rho)
    == gbar(z,k)``).  For HF-LOSO build it from TRAIN rows only (``train_mask_rows``)
    so the held-out sim never leaks into the fixed mean.

    Returns ``(tab (Nz, n_classes, K), z_tab (Nz,))`` -- host-side numpy, z ascending.
    """
    rows = (np.arange(targets["x"].shape[0]) if train_mask_rows is None
            else np.asarray(train_mask_rows))
    g = targets["g"][rows]                               # (B,4,K)
    z = np.round(targets["x"][rows, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
    zvals = np.array(sorted(set(z)))
    K = g.shape[-1]
    log_rho = np.asarray(log_rho)
    tab = np.zeros((len(zvals), N_CLASSES, K))
    with np.errstate(invalid="ignore"):
        for i, zz in enumerate(zvals):
            m = (z == zz)
            gm = np.nanmean(g[m], axis=0)                # (4,K) mean over sims&alpha at z
            gm = np.where(np.isfinite(gm), gm, 0.0)
            tab[i] = gm - log_rho[None, :]               # subtract per-k global mean
    return tab, zvals


def fixed_mean_table_resolved(targets, log_rho, *, train_mask_rows=None):
    """SEPARABLE + rank-1 (z, tau0, k) decomposition of the LF->HR fixed mean (T1).

    Bins the measured log-ratio ``g`` by BOTH (z, tau0-RUNG) -- not pooled over tau0
    as ``fixed_mean_table`` -- and decomposes the per-(z,rung,k) population mean into

        gbar(z,tau0,k) = gbar_z(z,k) + gbar_tau(rung,k) + a(k)*u_z(z)*u_tau(rung)

    the review-LOCKED form: a SINGLE rank-1 interaction captures the documented
    (z,tau0) sign-flip with too few DOF to alias the n_s tilt into the tau0 axis (a
    free 2-D table over 6 clustered HR sims does the opposite -- the reason the learned
    mlp-head was retired at a 4 sigma n_s Fisher bias).

    WHY RUNG, NOT RAW tau0:  tau0 = -log(target_F) is a DETERMINISTIC, bit-identical
    function of (rung, z) -- at fixed (rung,z) all 6 sims share the same tau0, and tau0
    GROWS with z within a rung -- so binning by raw tau0 would put a SINGLE sim in each
    cell (over-fit).  We bin by the 20-rung mean-flux ladder (clean per-(z,rung) 6-sim
    means) and carry the per-z tau0->rung ladder ``tau_by_z`` (Nz, N_rung) so the
    eval-time continuous-tau0 lookup maps back to the right ladder position
    (``FixedMeanHead`` does this, mirroring ``interp_res_corr``'s per-z grid).

    Construction:
      1. ``M(z,rung,c,k)`` = the 6-sim population mean of g at each (z,rung,k).
      2. ``gbar_z(z,k)`` = mean over rung of M;  ``gbar_tau(rung,k)`` = mean over z of
         (M - gbar_z), ZERO-MEAN in z (the additive separable fit).
      3. residual R = M - gbar_z - gbar_tau; the shared rank-1 ``a(k)u_z(z)u_tau(rung)``.
         STEP-0 (Bayesian rec): ``u_tau`` is FIXED to the PHYSICAL mean-flux axis (a
         monotone, zero-mean, unit vector in the rung index -- tau0 grows with rung), NOT
         a free SVD direction that can rotate into the n_s-aliasing axis over 6 noisy
         sims.  Given the fixed u_tau, ``u_z`` is the LS-optimal z-profile (projection of
         R onto u_tau) and a(k) the per-class LS amplitude, so the interaction is still
         exactly ONE (z x tau0) DOF, now with a physically-anchored tau0 axis.
      4. log_rho subtracted from gbar_z ONLY (the other terms are zero-mean), so
         ``MultiFidelity.g = log_rho + head == gbar``.

    Returns a dict: ``gbar_z_tab`` (Nz,C,K), ``gtau_tab`` (Nrung,C,K), ``a_k`` (C,K),
    ``u_z`` (Nz,), ``u_tau`` (Nrung,), ``z_tab`` (Nz,), ``tau_tab`` (Nrung,) = the
    rung INDEX axis (0..Nrung-1), and ``tau_by_z`` (Nz, Nrung) = the physical tau0 of
    each rung at each z (ascending in rung).  ``train_mask_rows`` selects TRAIN rows
    (HF-LOSO).
    """
    rows = (np.arange(targets["x"].shape[0]) if train_mask_rows is None
            else np.asarray(train_mask_rows))
    g = targets["g"][rows]                                # (B,C,K)
    z = np.round(targets["x"][rows, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
    tau0 = np.asarray(targets["tau0"])[rows]
    aidx = np.asarray(targets.get("alpha_idx", np.full(len(rows), -1)))[rows]
    if np.all(aidx < 0):
        raise ValueError(
            "fixed_mean_table_resolved needs alpha_idx in targets (re-run "
            "measure_delta_targets on a cache that carries alpha_idx)")
    zvals = np.array(sorted(set(z)))
    rungs = np.array(sorted(set(int(a) for a in aidx)))
    C, K = g.shape[1], g.shape[2]
    nz, nr = len(zvals), len(rungs)
    log_rho = np.asarray(log_rho)

    # 1. population mean M(z,rung,c,k) over the 6 sims at each cell + per-z tau0 ladder.
    M = np.full((nz, nr, C, K), np.nan)
    tau_by_z = np.full((nz, nr), np.nan)
    with np.errstate(invalid="ignore"):
        for zi, zz in enumerate(zvals):
            for ri, rr in enumerate(rungs):
                m = (z == zz) & (aidx == rr)
                if m.any():
                    M[zi, ri] = np.nanmean(g[m], axis=0)
                    tau_by_z[zi, ri] = np.nanmean(tau0[m])   # bit-identical across sims
    Mf = np.where(np.isfinite(M), M, np.nan)
    # fill any empty tau ladder cell from the nearest populated rung at that z so the
    # eval-time tau0->rung inversion is monotone & gap-free.
    for zi in range(nz):
        row = tau_by_z[zi]
        good = np.where(np.isfinite(row))[0]
        if good.size:
            tau_by_z[zi] = np.interp(np.arange(nr), good, row[good])
        else:
            tau_by_z[zi] = np.linspace(0.0, 1.0, nr)

    # 2. separable marginals (nan-aware).
    with np.errstate(invalid="ignore"):
        gbar_z = np.nanmean(Mf, axis=1)                   # (nz,C,K)
        dep = Mf - gbar_z[:, None, :, :]                  # (nz,nr,C,K)
        gbar_tau = np.nanmean(dep, axis=0)                # (nr,C,K)
    gbar_z = np.where(np.isfinite(gbar_z), gbar_z, 0.0)
    gbar_tau = np.where(np.isfinite(gbar_tau), gbar_tau, 0.0)

    # 3. residual after the separable fit; ONE shared rank-1 (z x rung) interaction.
    #
    # STEP-0 REFINEMENT (Bayesian rec, 2026-06-08, the T3 adoption gate): the rung
    # profile ``u_tau`` is FIXED to the PHYSICAL mean-flux axis -- a fixed, monotone,
    # zero-mean unit vector in the rung index -- and is NOT a free SVD direction.  The
    # free leading-SVD ``u_tau`` (the prior code) can ROTATE into the n_s-aliasing
    # direction over only 6 noisy HR sims (n_s and tau0 are ~-0.66 correlated through
    # the LF two-stage head, so a noise-fit rung profile is the most dangerous aliasing
    # vector -- spec section 4 / 3.4).  Pinning u_tau to the deterministic mean-flux
    # ladder removes that DOF: the interaction can only express a (monotone-in-mean-flux)
    # x (z-profile) sign-flip, which is the physically-motivated (z,tau0) interaction,
    # not an arbitrary rung pattern.  Given the FIXED u_tau, u_z is then the LS-optimal
    # z-profile (the projection of the residual matrix onto u_tau) and a(k) the per-class
    # LS amplitude -- so the rank-1 term is still a single (z x tau0) DOF, now with a
    # physically-anchored tau0 axis.  (u_z and a(k) are fit as before; only u_tau is fixed.)
    R = Mf - gbar_z[:, None, :, :] - gbar_tau[None, :, :, :]   # (nz,nr,C,K)
    R = np.where(np.isfinite(R), R, 0.0)
    Rzt = R.mean(axis=(2, 3))                              # (nz,nr) shared shape
    # FIXED physical mean-flux axis: monotone increasing in rung index (tau0 grows with
    # rung), centred to zero-mean (so it is a pure departure that adds NO net log_rho)
    # and unit-normalized.  rung index 0..nr-1 is the deterministic mean-flux ladder.
    if nr >= 2:
        u_tau = (rungs - rungs.mean()).astype(float)      # monotone, zero-mean
    else:
        u_tau = np.zeros(nr)
    nt_norm = np.linalg.norm(u_tau)
    if nt_norm > 0:
        u_tau = u_tau / nt_norm
    # u_z LS-optimal given the FIXED u_tau: project Rzt onto u_tau (over the rung axis),
    # normalize -> a unit z-profile; the rank-1 amplitude is folded into a(k).
    if nz >= 2 and nr >= 2 and np.any(Rzt != 0) and nt_norm > 0:
        u_z = Rzt @ u_tau                                 # (nz,) = Σ_rung Rzt·u_tau
    else:
        u_z = np.zeros(nz)
    nz_norm = np.linalg.norm(u_z)
    if nz_norm > 0:
        u_z = u_z / nz_norm
    outer = u_z[:, None] * u_tau[None, :]                  # (nz,nr)
    denom = float(np.sum(outer * outer))
    a_k = np.zeros((C, K))
    if denom > 0:
        a_k = np.einsum("ztck,zt->ck", R, outer) / denom   # (C,K) LS amplitude

    gbar_z_tab = gbar_z - log_rho[None, None, :]
    return dict(
        gbar_z_tab=gbar_z_tab, gtau_tab=gbar_tau, a_k=a_k,
        u_z=u_z, u_tau=u_tau, z_tab=zvals,
        tau_tab=np.arange(nr, dtype=float), tau_by_z=tau_by_z,
    )


@eqx.filter_value_and_grad
def _loss_and_grad(delta_head, basis, log_rho, cond, g_target, mask, w_k,
                   coeff_l2, mean_prior_w):
    """Loss for the delta head ON ONE BATCH (vmapped over rows). Differentiable.

    Terms:
      * masked weighted MSE of (log_rho + delta_head) vs g_target (the measured
        log-ratio) -- the data term;
      * coeff L2 (ridge on the per-class basis coefficients) -- complexity control;
      * mean-ratio prior: pull the per-(class) MEAN over k of the delta-head output
        toward 0 (i.e. g's per-k mean toward log_rho) -- keeps the coherent amplitude
        anchored to rho so the few-sim fit cannot run away.
    """
    # vmap the head over the batch rows.
    g_pred = jax.vmap(lambda c: delta_head(c, basis))(cond)        # (B,4,K)
    g_full = log_rho[None, None, :] + g_pred                       # add rho
    diff = jnp.where(mask, g_full - g_target, 0.0)                 # 0 on masked bins
    wk = w_k[None, None, :]
    # weighted masked MEAN: numerator over masked diffs, denominator = Σ(w_k·mask).
    data = jnp.sum((diff ** 2) * wk) / jnp.maximum(jnp.sum(wk * mask), 1.0)

    coeffs = jax.vmap(delta_head.coeffs)(cond)                     # (B,4,n_basis)
    l2 = coeff_l2 * jnp.mean(coeffs ** 2)

    # mean-ratio prior on the DELTA part (per row & class, mean over k).
    mean_g = jnp.mean(g_pred, axis=2)                              # (B,4)
    prior = mean_prior_w * jnp.mean(mean_g ** 2)
    return data + l2 + prior


def train_delta_head(targets, basis, log_rho, eval_logk, *, train_mask_rows=None,
                     n_basis=4, width=16, n_layers=1, lr=3e-3, epochs=400,
                     coeff_l2=1e-2, mean_prior_w=1e-2, w_k=None, seed=0,
                     verbose=False):
    """Fit the DeltaHead on the measured g targets (heavily regularized).

    ``targets`` from ``measure_delta_targets``; ``train_mask_rows`` selects the rows
    to TRAIN on (e.g. all rows whose HF sim is NOT the held-out one, for HF-LOSO);
    None = all rows.  ``w_k`` (K,) optional per-k loss weight (default uniform).
    Returns ``(delta_head, history)``.  Full-batch Adam (the data set is small).
    """
    import optax
    K = basis.shape[1]
    rows = (np.arange(targets["x"].shape[0]) if train_mask_rows is None
            else np.asarray(train_mask_rows))
    cond = jnp.asarray(np.concatenate(
        [targets["x"][rows], targets["tau0"][rows][:, None]], axis=1))   # (B,11)
    g_target = jnp.asarray(np.nan_to_num(targets["g"][rows], nan=0.0))   # (B,4,K)
    mask = jnp.asarray(np.isfinite(targets["g"][rows]))                  # (B,4,K)
    w_k = jnp.ones(K) if w_k is None else jnp.asarray(w_k)

    head = DeltaHead(n_basis=n_basis, n_classes=N_CLASSES, in_dim=cond.shape[1],
                     width=width, n_layers=n_layers, key=jax.random.PRNGKey(seed))
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(head, eqx.is_array))
    basis_j = jnp.asarray(basis); log_rho_j = jnp.asarray(log_rho)

    @eqx.filter_jit
    def step(head, opt_state):
        loss, grad = _loss_and_grad(head, basis_j, log_rho_j, cond, g_target, mask,
                                    w_k, coeff_l2, mean_prior_w)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(head, eqx.is_array))
        head = eqx.apply_updates(head, updates)
        return head, opt_state, loss

    history = []
    for ep in range(epochs):
        head, opt_state, loss = step(head, opt_state)
        history.append(float(loss))
        if verbose and (ep % max(epochs // 10, 1) == 0):
            print(f"  delta-head ep {ep}: loss={float(loss):.5g}")
    return head, {"loss": history}


def train_global_linear_head(targets, basis, log_rho, eval_logk, *,
                             train_mask_rows=None, n_basis=4, lr=3e-3, epochs=400,
                             coeff_l2=1e-1, mean_prior_w=1e-1, w_k=None, seed=0,
                             ap_dir=None, ap_prior_w=0.0):
    """Fit the ``GlobalLinearHead`` (delta_mode='linear') on the measured g targets.

    The sweep's variant (b): ONE global linear-in-theta amplitude * a fixed per-class
    k-shape.  Same masked weighted-MSE + coeff-L2 + mean-ratio prior as
    ``train_delta_head`` (so it's scored identically), PLUS an OPTIONAL A_p-direction
    prior on the linear weight ``w``: if ``ap_dir`` (in_dim,) is given, add
    ``ap_prior_w * || w - (w.u_hat) u_hat ||^2`` where ``u_hat = ap_dir/|ap_dir|`` --
    i.e. softly pull the global amplitude to vary ONLY along the A_p direction (so the
    weak theta term is interpretable as an A_p amplitude tweak, not a free n_s tilt).
    ``ap_prior_w=0`` (default) leaves it a free global linear amplitude.
    Returns ``(head, history)``.  Full-batch Adam.
    """
    import optax
    K = basis.shape[1]
    rows = (np.arange(targets["x"].shape[0]) if train_mask_rows is None
            else np.asarray(train_mask_rows))
    cond = jnp.asarray(np.concatenate(
        [targets["x"][rows], targets["tau0"][rows][:, None]], axis=1))   # (B,in)
    g_target = jnp.asarray(np.nan_to_num(targets["g"][rows], nan=0.0))
    mask = jnp.asarray(np.isfinite(targets["g"][rows]))
    w_k = jnp.ones(K) if w_k is None else jnp.asarray(w_k)
    basis_j = jnp.asarray(basis); log_rho_j = jnp.asarray(log_rho)

    head = GlobalLinearHead(in_dim=cond.shape[1], n_classes=N_CLASSES,
                            n_basis=n_basis, key=jax.random.PRNGKey(seed))
    if ap_dir is not None and ap_prior_w > 0.0:
        u = jnp.asarray(ap_dir)
        u = u / jnp.maximum(jnp.linalg.norm(u), 1e-30)
    else:
        u = None

    @eqx.filter_value_and_grad
    def loss_fn(head):
        g_pred = jax.vmap(lambda c: head(c, basis_j))(cond)            # (B,4,K)
        g_full = log_rho_j[None, None, :] + g_pred
        diff = jnp.where(mask, g_full - g_target, 0.0)
        wk = w_k[None, None, :]
        data = jnp.sum((diff ** 2) * wk) / jnp.maximum(jnp.sum(wk * mask), 1.0)
        coeffs = jax.vmap(head.coeffs)(cond)
        l2 = coeff_l2 * jnp.mean(coeffs ** 2)
        mean_g = jnp.mean(g_pred, axis=2)
        prior = mean_prior_w * jnp.mean(mean_g ** 2)
        ap_term = 0.0
        if u is not None:
            # penalize the component of w ORTHOGONAL to the A_p direction.
            w_perp = head.w - jnp.dot(head.w, u) * u
            ap_term = ap_prior_w * jnp.sum(w_perp ** 2)
        return data + l2 + prior + ap_term

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(head, eqx.is_array))

    @eqx.filter_jit
    def step(head, opt_state):
        loss, grad = loss_fn(head)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(head, eqx.is_array))
        head = eqx.apply_updates(head, updates)
        return head, opt_state, loss

    history = []
    for _ in range(epochs):
        head, opt_state, loss = step(head, opt_state)
        history.append(float(loss))
    return head, {"loss": history}


def build_default_head(targets, log_rho, *, train_mask_rows=None, n_basis=4,
                       resolved=False):
    """Build the DEFAULT (delta_mode='none') ``FixedMeanHead`` from measured targets.

    Convenience wrapper.  NO fitting -- the fixed mean is a pure average over the
    (train) rows.  Use with ``build_multifidelity(..., log_rho=..., delta_mode='none')``.

    ``resolved=False`` (default): the historical POOLED z-only fixed-mean table
    (``fixed_mean_table``), tau0-independent.

    ``resolved=True`` (T1): the tau0+z-RESOLVED separable + rank-1 correction
    (``fixed_mean_table_resolved``) -- ``g(z,tau0,k) = gbar_z + gbar_tau +
    a(k)u_z u_tau``.  dg/dtau0 != 0; reproduces the per-(z,tau0) rho to the 6-sim
    noise; clamped & NaN-free over the (z,tau0) edges."""
    if not resolved:
        tab, ztab = fixed_mean_table(targets, log_rho, train_mask_rows=train_mask_rows)
        return FixedMeanHead(tab, ztab, n_basis=n_basis)
    c = fixed_mean_table_resolved(targets, log_rho, train_mask_rows=train_mask_rows)
    return FixedMeanHead(
        c["gbar_z_tab"], c["z_tab"], n_basis=n_basis, resolved=True,
        gtau_tab=c["gtau_tab"], tau_tab=c["tau_tab"], tau_by_z=c["tau_by_z"],
        a_k=c["a_k"], u_z=c["u_z"], u_tau=c["u_tau"])


# ---------------------------------------------------------------------------- #
# T6: dN/dX + CDDF FIXED (theta-independent) resolution correction (Head A)
#
# MF is otherwise P1D-only; Head A (per-class incidence dN/dX, CDDF f_nhi) has no
# resolution correction, yet HR != LF dN/dX (HR/LF ~ 1.0-1.3, z-resolved).  PI
# directive: MF the dN/dX/CDDF via a FIXED (theta-INDEPENDENT) resolution correction,
# and KEEP the existing HCD incidence-prior widths (inference.py) UNCHANGED.
#
# dN/dX is tau0-INVARIANT (no alpha ladder): the snap-level rows are sims x z, indexed
# by ``snap_*`` datasets / ``snap_group_idx`` (NOT the P1D row map).  HR snaps are EXACT
# LF design points, so we match on (params_unit, z) and measure HR/LF per class.  A
# z-resolved fixed ratio captures ~97%/75%/52% of the (LLS/subDLA/DLA) ratio variance;
# the per-(z,class) residual is the irreducible ~1.7-2.2% 6-sim noise floor.
# ---------------------------------------------------------------------------- #
def load_snap_cache(path):
    """Load the snap-level (dN/dX, CDDF) cache rows + their (params_unit, z).

    Returns a dict of host-side numpy:
      ``dndx``   (S,3)  per-snap dN/dX for (LLS, subDLA, DLA);
      ``f_nhi``  (S,30) CDDF;  ``log_nhi`` (30,) the log10 N_HI bin centres;
      ``z``      (S,)   physical z per snap;  ``pu`` (S,9) params_unit;
      ``params`` (S,9), ``sim`` (S,) sim name.
    The snap rows index sims x z (tau0-invariant); each P1D row's ``snap_group_idx``
    points at its snap row, so we read (params, z) of the FIRST P1D row in each group.
    """
    import h5py
    with h5py.File(path, "r") as f:
        sg = f["snap_group_idx"][:]
        z_row = np.round(f["z_grid"][:], 4)
        params_row = f["params"][:]
        dndx = np.stack([f["snap_dNdX_LLS"][:], f["snap_dNdX_subDLA"][:],
                         f["snap_dNdX_DLA"][:]], axis=1)          # (S,3)
        f_nhi = f["snap_f_nhi"][:]                                # (S,30)
        log_nhi = f["log_nhi_centres"][:]
        ssim = f["snap_sim_name"][:]
    S = dndx.shape[0]
    z = np.full(S, np.nan); params = np.full((S, 9), np.nan)
    for gi in range(S):
        rows = np.where(sg == gi)[0]
        z[gi] = z_row[rows[0]]
        params[gi] = params_row[rows[0]]
    sim = np.array([s.decode() if isinstance(s, bytes) else s for s in ssim])
    return dict(dndx=dndx, f_nhi=f_nhi, log_nhi=log_nhi, z=z,
                params=params, pu=np.round(normalize_params(params), 6), sim=sim)


def match_snap_hr_to_lf(lf_snap, hr_snap):
    """Match each HR snap row to its LF snap row on (params_unit, z).  Returns
    ``[(hr_row, lf_row), ...]``.  dN/dX is tau0-invariant so there is NO alpha key
    (unlike the P1D ``match_hr_to_lf``)."""
    def key(d):
        return [(tuple(d["pu"][i]), round(float(d["z"][i]), 4))
                for i in range(len(d["z"]))]
    idx = {k: i for i, k in enumerate(key(lf_snap))}
    return [(h, idx[k]) for h, k in enumerate(key(hr_snap)) if k in idx]


def build_dndx_res_corr(lf_dndx, lf_z, hr_dndx, hr_z, *, n_min=1):
    """Measure the FIXED, z-resolved HR/LF dN/dX ratio per HCD class (T6, step 1).

    ``lf_dndx``/``hr_dndx`` (M,3) the per-snap MATCHED dN/dX for (LLS,subDLA,DLA) (one
    row per matched HR<->LF snap pair); ``lf_z``/``hr_z`` (M,) their z (equal by
    construction).  At each distinct z we take the 6-sim mean of the per-row ratio
    HR/LF (theta-pooled -> theta-INDEPENDENT, per the PI directive; the residual after
    this z-mean is the irreducible ~2% 6-sim noise).  Cells with <n_min sims fall back
    to the nearest populated z so the table has no gaps (clamped lookup handles edges).

    Returns ``{'z': (Nz,), 'ratio': (Nz,3)}`` host-side numpy, z ascending.  The ratio
    is a pure function of (z, class) -- NO cosmology dependence.
    """
    lf_dndx = np.asarray(lf_dndx, float); hr_dndx = np.asarray(hr_dndx, float)
    lf_z = np.round(np.asarray(lf_z, float), 4); hr_z = np.round(np.asarray(hr_z, float), 4)
    if not np.allclose(lf_z, hr_z):
        raise ValueError("lf_z and hr_z must match row-for-row (matched pairs)")
    valid = (lf_dndx > 0) & (hr_dndx > 0)
    ratio_row = np.where(valid, hr_dndx / np.where(lf_dndx > 0, lf_dndx, 1.0), np.nan)
    zvals = np.array(sorted(set(hr_z)))
    ratio = np.full((len(zvals), 3), np.nan)
    counts = np.zeros((len(zvals), 3), int)
    with np.errstate(invalid="ignore"):
        for zi, zz in enumerate(zvals):
            m = (hr_z == zz)
            counts[zi] = np.sum(valid[m], axis=0)
            ratio[zi] = np.nanmean(ratio_row[m], axis=0)
    # fill under-sampled / NaN cells from the nearest populated z (per class) so the
    # table is gap-free and the differentiable lookup never hits a NaN.
    for c in range(3):
        good = np.where(np.isfinite(ratio[:, c]) & (counts[:, c] >= n_min))[0]
        if good.size == 0:
            ratio[:, c] = 1.0
            continue
        for zi in range(len(zvals)):
            if not (np.isfinite(ratio[zi, c]) and counts[zi, c] >= n_min):
                ratio[zi, c] = ratio[good[np.argmin(np.abs(good - zi))], c]
    return dict(z=zvals, ratio=ratio)


def dndx_res_factor(tab, z):
    """Differentiable, theta-INDEPENDENT dN/dX resolution factor at scalar z: (3,).

    ``tab`` from ``build_dndx_res_corr``.  CLAMPED-linear interp of the per-class HR/LF
    ratio in z (jnp.interp constant-edge beyond the table z-range, so HMC / z-edge
    queries get the boundary value -- never NaN, never a divergent extrapolation).
    A pure function of (z, class): no cosmology input, so d(factor)/d(theta) == 0."""
    zt = jnp.asarray(tab["z"]); r = jnp.asarray(tab["ratio"])      # (Nz,), (Nz,3)
    z = jnp.asarray(z)
    return jax.vmap(lambda col: jnp.interp(z, zt, col), in_axes=1)(r)   # (3,)


def apply_dndx_res_corr(tab, dndx, z):
    """Apply the FIXED resolution correction to Head-A dN/dX (T6, step 3).

    ``dndx`` (...,3) the emulator Head-A per-class incidence (LLS,subDLA,DLA); ``z``
    scalar or (...).  Returns ``dndx * R_c(z)`` with R_c the z-resolved HR/LF ratio --
    i.e. LF-resolution dN/dX scaled up to HR resolution.  Differentiable in (dndx, z);
    the factor is theta-free so the incidence-prior widths (inference.py) are UNCHANGED
    (this multiplies the predicted incidence, it does not touch the prior).
    """
    dndx = jnp.asarray(dndx)
    z = jnp.asarray(z)
    if z.ndim == 0:
        fac = dndx_res_factor(tab, z)                              # (3,)
    else:
        fac = jax.vmap(lambda zz: dndx_res_factor(tab, zz))(z)     # (...,3)
    return dndx * fac


def build_cddf_res_corr(lf_fnhi, lf_z, hr_fnhi, hr_z, log_nhi, *, n_min=1):
    """Measure the FIXED, z-resolved HR/LF CDDF (f_nhi) ratio vs (z, log N_HI) (T6).

    ``lf_fnhi``/``hr_fnhi`` (M,Nbin) the matched per-snap CDDF; ``log_nhi`` (Nbin,) the
    log10 N_HI bin centres.  At each distinct z, the 6-sim mean of the per-row,
    per-bin ratio (theta-pooled).  Cells where the CDDF is a structural zero (the bin
    is unpopulated in LF or HR -- e.g. bin 0, or the highest-N_HI bins at high z) get
    ratio=1 (no correction) so the shift is well-defined everywhere.

    Returns ``{'z':(Nz,), 'log_nhi':(Nbin,), 'ratio':(Nz,Nbin)}`` -- the multiplicative
    CDDF resolution factor (equivalently an additive shift in log f_nhi).
    """
    lf_fnhi = np.asarray(lf_fnhi, float); hr_fnhi = np.asarray(hr_fnhi, float)
    lf_z = np.round(np.asarray(lf_z, float), 4); hr_z = np.round(np.asarray(hr_z, float), 4)
    if not np.allclose(lf_z, hr_z):
        raise ValueError("lf_z and hr_z must match row-for-row (matched pairs)")
    valid = (lf_fnhi > 0) & (hr_fnhi > 0) & np.isfinite(lf_fnhi) & np.isfinite(hr_fnhi)
    ratio_row = np.where(valid, hr_fnhi / np.where(lf_fnhi > 0, lf_fnhi, 1.0), np.nan)
    zvals = np.array(sorted(set(hr_z)))
    Nbin = lf_fnhi.shape[1]
    ratio = np.ones((len(zvals), Nbin))
    with np.errstate(invalid="ignore"):
        for zi, zz in enumerate(zvals):
            m = (hr_z == zz)
            cnt = np.sum(valid[m], axis=0)
            rm = np.nanmean(ratio_row[m], axis=0)
            ratio[zi] = np.where((cnt >= n_min) & np.isfinite(rm), rm, 1.0)
    return dict(z=zvals, log_nhi=np.asarray(log_nhi), ratio=ratio)


def cddf_res_factor(tab, z):
    """Differentiable, theta-INDEPENDENT CDDF resolution factor at scalar z: (Nbin,).

    Clamped-linear interp in z of the per-bin HR/LF f_nhi ratio (constant-edge).  Pure
    function of (z, log N_HI bin) -- no cosmology input."""
    zt = jnp.asarray(tab["z"]); r = jnp.asarray(tab["ratio"])
    z = jnp.asarray(z)
    return jax.vmap(lambda col: jnp.interp(z, zt, col), in_axes=1)(r)   # (Nbin,)


# ---------------------------------------------------------------------------- #
# Convenience: load the frozen LF backbone + build a MultiFidelity
# ---------------------------------------------------------------------------- #
def load_lf_backbone(fold=0, ckpt_dir="checkpoints"):
    """Load a finalized LF fold checkpoint -> (model, meta, norm, lf_logk).

    ``lf_logk`` = log10 of the LF native k-grid (from meta).  The model + norm are
    the frozen backbone for the MF layer."""
    model, meta, norm = T.load_checkpoint(f"{ckpt_dir}/final_fold{fold}")
    lf_logk = np.log10(np.asarray(meta["kfkms"]))
    return model, meta, norm, lf_logk


_HEAD_MODE = {"FixedMeanHead": "none", "GlobalLinearHead": "linear",
              "DeltaHead": "mlp"}


def build_multifidelity(lf_model, lf_norm, lf_logk, delta_head, *, eval_logk,
                        log_rho=None, n_basis=4, res_dir=RES_CORR_DIR, n_tail=6,
                        delta_mode=None):
    """Assemble a ``MultiFidelity`` from a frozen LF backbone + a head.

    ``delta_head`` is one of ``FixedMeanHead`` (default 'none'), ``GlobalLinearHead``
    ('linear'), or ``DeltaHead`` ('mlp').  ``delta_mode`` is inferred from the head
    TYPE when None (override only for a custom head).  ``log_rho`` (K,) optional
    (default zeros == rho==1).  Builds the smooth k-basis and loads res_corr
    internally.  Returns the ``MultiFidelity`` module."""
    basis = smooth_k_basis(jnp.asarray(eval_logk), n_basis=n_basis)
    if log_rho is None:
        log_rho = np.zeros(len(eval_logk))
    if delta_mode is None:
        delta_mode = _HEAD_MODE.get(type(delta_head).__name__, "mlp")
    z_rc, logk_rc, rc_vals = load_res_corr(res_dir)
    return MultiFidelity(
        lf_model=lf_model, lf_norm=lf_norm, eval_logk=eval_logk, lf_logk=lf_logk,
        delta_head=delta_head, basis=basis, log_rho=log_rho,
        z_rc=z_rc, logk_rc=logk_rc, rc_vals=rc_vals, n_tail=n_tail,
        delta_mode=delta_mode)
