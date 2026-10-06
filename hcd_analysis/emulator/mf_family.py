"""The certified LF -> HF multi-fidelity model family (moved verbatim from multifidelity.py at gate E, emulator-debug
campaign 2026-10; spec GATE_E_SPEC.md v1 in the notes repository): the theta-independent resolved FixedMeanHead
(log_rho + gbar_z + gbar_tau + rank-1), its conditioning vector, its table construction and the exact HR/LF row
matching. The gate E forward reaches the MF correction only through this module and mf_modes; the pre-2026-10
single-grid machinery stays in multifidelity.py (historical), which re-imports these names."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .data import Z_LIMITS, normalize_params


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


def make_cond(x, tau0):
    """Conditioning vector for the DeltaHead: [params_unit(9), z_unit(1), tau0(1)].

    ``x`` (10,) = [params_unit(9), z_unit(1)] (the LF encoder input), ``tau0`` scalar.
    Returns (11,).  Differentiable / jittable."""
    return jnp.concatenate([jnp.asarray(x), jnp.atleast_1d(tau0)])


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
