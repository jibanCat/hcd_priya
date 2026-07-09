"""BUILD (Task 2A): the z-incoherent OUT-OF-SPAN instrument-resolution injection
basis for the Gate-B spectral-resolution (option-b / f_res) injection-recovery gate.

WHY
---
The forward marginalizes the spectral resolution with a 2-param f_res nuisance

    b_res(z) = f_res_amp * ((1+z)/(1+F_RES_PIVOT_Z))^f_res_slope

threaded FORWARD-ONLY as  P -> P * exp(2*b_res(z)*k^2*R_z(z)^2)  per z-block
(closure_legb ``_bres_of_z`` / data_likelihood ``_resolution_factor``). To first
order about (amp=0, slope=0) the perturbation the nuisance can absorb is spanned,
in per-z ``b_res``-space, by

    u1 = ones(n_z)                          (the amplitude direction, d/damp)
    u2 = log((1+z)/(1+F_RES_PIVOT_Z))       (the z-slope direction, d/dslope).

The injection gate must inject a per-z resolution MISSPECIFICATION the 2-param
f_res CANNOT represent (out-of-span) AND concentrated in the He-II window z>=2.8
(else the gate passes vacuously). The natural inner product on ``b_res``-space is
the DATA-metric PULLBACK

    M = B^T C_data^-1 B ,   B[rows_iz, iz] = 2 * k[rows_iz]^2 * R_z(z_iz)^2

(the map ``p = B @ b_res`` sends a per-z ``b_res`` to its flat-P log-perturbation).
We build the top-2 out-of-span directions in the He-II sub-block metric ``M_sub``,
scale to the leg's own 1-sigma resolution envelope, and store the per-z ``(n_z,)``
vectors ``{leg}_bres1``/``bres2`` the Task-2B resolver consumes.

WHAT (per leg DESI/eBOSS/KS, in the M_sub = B_sub^T C_sub^-1 B_sub metric)
-------------------------------------------------------------------------
1. SPAN  {u1=ones, u2=logfac}  restricted to the He-II nodes -> M_sub-orthonormal.
2. z-shape dictionary on the He-II nodes (z-incoherent / z-curvature / He-II ramp),
   each projected out of span{u1,u2} in M_sub, stacked + SVD in M_sub -> top-2
   directions bres1_sub, bres2_sub (provably M_sub-orthogonal to span).
3. SCALE each so max_z34 |exp(2*bres*k^2*R_z^2) - 1| == the leg's 1-sigma envelope.
4. ASSERT cos_M(bres_j, span{u1,u2}) < 0.8 (the OUT-OF-SPAN certificate).
5. n_s response pulled into b_res-coords via the M-projection
   r_ns_bres = M_sub^-1 B_sub^T C_sub^-1 r_ns[hiZ]; pick the worst-n_s member.
6. Tier-R (realistic) member from the MEASURED resolution error (flagged FLAG-not-
   gate when the error is a symmetric 1-sigma envelope, not a signed residual).
7. SAVE per-z bres1/bres2/bres_real + metadata to
   _emulator_data/res_instr_injection_basis.npz ; FIGURE -> notes 08_resolution/.

This is the instrument analogue of scripts/build_res_corr_injection_basis.py: the
ONLY changes are (1) work in b_res-space (n_z) with metric M instead of P-space (N)
with C_data^-1; (2) the span is {ones, logfac}; (3) the perturbation is B@b_res;
(4) store per-z (n_z,) vectors.

Env (HEAVY -- run via SLURM on cavestru0, NOT the interactive node):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/build_res_instr_injection_basis.py
"""
from __future__ import annotations
import functools
print = functools.partial(print, flush=True)

import numpy as np  # module top-level is numpy-ONLY so the pure helper imports cheap

OUT_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/res_instr_injection_basis.npz"
FIG = "/home/mfho/hcd_priya_notes/figures/analysis/08_resolution/res_instr_oos_basis.png"

# DESI/eBOSS raw-data sources for the (un-stored) 1-sigma resolution error re-read.
DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
KS_BASE = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"

Z_HEII = 2.8                 # He-II window lower edge (localization)
F_RES_PIVOT_Z = 3.0          # the f_res(z) pivot (1+z)/(1+3) -- closure_legb.F_RES_PIVOT_Z
ENVELOPE_FALLBACK = 0.05     # +-5% fallback envelope (res_corr constant) if resolution_e absent
COS_GATE = 0.8               # assert cos_M(bres_j, span{u1,u2}) < this (OUT-OF-SPAN certificate)


# ---------------------------------------------------------------------------- #
#  Whitened-metric closures (copied from scripts/build_res_corr_injection_basis)
#  -- here the "Cinv" argument is a generic SPD metric (M_sub or C_sub^-1).
# ---------------------------------------------------------------------------- #
def _whitened_inner(Cinv):
    """Return <a,b> = a^T Cinv b and the induced norm/cosine closures (Cinv SPD)."""
    def inner(a, b):
        return float(a @ Cinv @ b)
    def norm(a):
        return float(np.sqrt(max(a @ Cinv @ a, 0.0)))
    def cos(a, b):
        den = norm(a) * norm(b)
        return float(inner(a, b) / den) if den > 0 else float("nan")
    return inner, norm, cos


def _gram_schmidt_whitened(vecs, Cinv, tol=1e-12):
    """Cinv-orthonormal basis of span{vecs} (drops near-degenerate columns)."""
    inner, norm, _ = _whitened_inner(Cinv)
    basis = []
    for v in vecs:
        u = np.array(v, float).copy()
        for q in basis:
            u = u - inner(u, q) * q
        nu = norm(u)
        if nu > tol:
            basis.append(u / nu)
    return basis


def _project_out(R, ortho_basis, Cinv):
    """Remove the components of R along the Cinv-orthonormal ortho_basis."""
    inner, _, _ = _whitened_inner(Cinv)
    out = np.array(R, float).copy()
    for q in ortho_basis:
        out = out - inner(out, q) * q
    return out


def analytic_worst_ns(r_ns_bres, ortho_span, M_sub, *, degenerate_tol=1e-8):
    """THE ANALYTIC worst-n_s out-of-span direction b* (Task 2A-2).

    The z-shape DICTIONARY members (bres1/bres2) only approximate the worst-n_s
    out-of-span direction; the TRUE argmax has a closed form. Over the set of
    M_sub-unit vectors b with cos_M(b, span{u1,u2}) == 0 (out-of-span), the
    maximizer of cos_M(b, r_ns_bres) is the (M_sub-)normalized residual of the
    M_sub-projection of r_ns_bres onto span:

        b* = P_perp_span(r_ns_bres) / ||P_perp_span(r_ns_bres)||_M ,
        P_perp_span(r) = r - sum_q <r, q>_M q   (q in ortho_span, M-orthonormal)

    and the achieved cosine is exactly

        f_perp = ||P_perp_span(r_ns_bres)||_M / ||r_ns_bres||_M

    (the fraction of the n_s pullback response living OUT of the 2-param f_res
    span). This is a Cauchy-Schwarz argmax: any out-of-span unit b decomposes
    r_ns_bres . the b-component of r_ns_bres in the b* direction is bounded by
    f_perp with equality iff b == +-b*.

    Parameters
    ----------
    r_ns_bres  : (n_heII,)      M_sub-pullback n_s response (build_oos_bres output).
    ortho_span : list of (n_heII,)  M_sub-orthonormal basis of span{u1,u2}.
    M_sub      : (n_heII, n_heII)   SPD pullback metric.
    degenerate_tol : guard -- if ||P_perp_span(r_ns_bres)||_M <= degenerate_tol *
                     ||r_ns_bres||_M, the n_s response is (numerically) fully
                     IN-SPAN: no out-of-span n_s direction exists to inject.

    Returns
    -------
    (bstar_sub, f_perp, degenerate) :
      bstar_sub  : (n_heII,) or None   M_sub-unit, M_sub-orthogonal to span
                   (None iff degenerate -- caller substitutes a dict fallback).
      f_perp     : float   achieved cos_M(bstar_sub, r_ns_bres); 0.0 if degenerate.
      degenerate : bool
    """
    inner_M, norm_M, _ = _whitened_inner(M_sub)
    r_ns_bres = np.asarray(r_ns_bres, float)
    r_perp = _project_out(r_ns_bres, ortho_span, M_sub)
    norm_rns = norm_M(r_ns_bres)
    norm_perp = norm_M(r_perp)
    if norm_rns <= 0.0 or norm_perp <= degenerate_tol * norm_rns:
        return None, 0.0, True
    bstar_sub = r_perp / norm_perp
    f_perp = float(inner_M(bstar_sub, r_ns_bres) / norm_rns)   # == norm_perp/norm_rns, > 0
    return bstar_sub, f_perp, False


def _scale_to_envelope(bres, B, band, env_target):
    """Scale a per-z b_res vector (n_z,) so max_z(band) |exp(B@bres)-1| ==
    env_target, with the SIGN convention: the largest-|amp| cell in ``band`` is a
    DEFICIT (negative log-perturbation), matching the bres1/bres2 convention.
    Returns (bres_scaled, env_realized)."""
    bres = np.asarray(bres, float)
    band = np.asarray(band, bool)
    p = B @ bres                                     # flat-P log-perturbation
    peak = np.max(np.abs(p[band])) if band.any() else np.max(np.abs(p))
    if peak <= 0:
        return bres, 0.0
    s = np.log1p(env_target) / peak                  # max|exp(p_scaled)-1| == env_target
    bres_s = bres * s
    p_s = p * s
    imax = np.argmax(np.abs(np.where(band, p_s, 0.0)))
    if p_s[imax] > 0:                                # largest |amp| cell -> a DEFICIT (negative)
        bres_s = -bres_s
        p_s = -p_s
    env_realized = float(np.max(np.abs(np.expm1(p_s[band]))))
    return bres_s, env_realized


# ---------------------------------------------------------------------------- #
#  THE LOAD-BEARING PURE HELPER (numpy-only; unit-tested independently of ctx)
# ---------------------------------------------------------------------------- #
def build_oos_bres(B, C_data, u1, u2, hiZ_cells, hiZ_nodes, r_ns_P,
                   *, cos_gate=COS_GATE, svd_rank_tol=1e-9):
    """Build the top-2 out-of-span per-z resolution directions in the He-II
    sub-block PULLBACK metric ``M_sub = B_sub^T C_sub^-1 B_sub``.

    Parameters
    ----------
    B          : (N, n_z)   B[i, iz] = 2*k_i^2*R_z(z_iz)^2 on z-block iz (0 elsewhere).
    C_data     : (N, N)     SPD data covariance (flat z-major, matches B rows).
    u1, u2     : (n_z,)      f_res span directions in b_res-space (ones, logfac).
    hiZ_cells  : (N,) bool   He-II flat cells (z_row >= Z_HEII).
    hiZ_nodes  : (n_z,) bool He-II nodes (z >= Z_HEII).
    r_ns_P     : (N,)        P-space n_s AD Jacobian dlogP/d(n_s_unit).

    Returns
    -------
    dict with (all He-II-restricted quantities carry ``_sub``):
      sub_nodes, cell_idx, C_sub, Cinv_sub, B_sub, M_sub,
      u1_sub, u2_sub, ortho_span,
      bres1_sub, bres2_sub          (n_heII_nodes,)  M-unit, M-orthogonal to span
      bres1, bres2                  (n_z,)           embedded (zero off He-II nodes)
      r_ns_bres                     (n_heII_nodes,)  = M_sub^-1 B_sub^T C_sub^-1 r_ns[hiZ]
      r_ns_P_sub                    (n_heII_cells,)  r_ns restricted to He-II cells
      cos_span                      (2,)  cos_M(bres_j, span{u1,u2})   (< cos_gate)
      cos_ns                        (2,)  cos_M(bres_j, r_ns_bres)
      worst_ns_member               int 1/2  (argmax_j |cos_ns_j|)
      S, rank, cos_gate

    The construction guarantees (exactly, to numerical precision) that each
    ``bres_j_sub`` is M_sub-orthogonal to span{u1,u2}, and that the pullback
    identity  <a, r_ns_bres>_M == <B_sub a, r_ns[hiZ]>_{C_sub^-1}  holds for all a
    (the reason a naive ``B_sub^T r_ns`` is WRONG).
    """
    B = np.asarray(B, float)
    C_data = np.asarray(C_data, float)
    u1 = np.asarray(u1, float)
    u2 = np.asarray(u2, float)
    hiZ_cells = np.asarray(hiZ_cells, bool)
    hiZ_nodes = np.asarray(hiZ_nodes, bool)
    r_ns_P = np.asarray(r_ns_P, float)
    N, n_z = B.shape

    cell_idx = np.where(hiZ_cells)[0]
    sub_nodes = np.where(hiZ_nodes)[0]
    n_heII = int(sub_nodes.size)
    if n_heII < 3:
        raise ValueError(f"build_oos_bres: only {n_heII} He-II nodes (<3) -- cannot "
                         "form a rank-2 out-of-span basis beyond the 2-D span")

    # ---- He-II sub-block pullback metric M_sub ----
    C_sub = C_data[np.ix_(cell_idx, cell_idx)]
    Cinv_sub = np.linalg.inv(C_sub)
    B_sub = B[np.ix_(cell_idx, sub_nodes)]              # (n_heII_cells, n_heII_nodes)
    M_sub = B_sub.T @ Cinv_sub @ B_sub
    M_sub = 0.5 * (M_sub + M_sub.T)                     # symmetrize (SPD by construction)

    u1_sub = u1[sub_nodes]
    u2_sub = u2[sub_nodes]

    inner_M, norm_M, cos_M = _whitened_inner(M_sub)
    ortho_span = _gram_schmidt_whitened([u1_sub, u2_sub], M_sub)   # M-orthonormal span

    # ---- candidate OUT-OF-SPAN z-shape dictionary on the He-II nodes ----
    ii = np.arange(n_heII)
    s_incoh = (-1.0) ** ii                               # z-incoherent (sign-alternating in z)
    s_curv = u2_sub ** 2 - np.mean(u2_sub ** 2)          # z-curvature (quadratic-in-logfac)
    s_ramp = np.linspace(0.0, 1.0, n_heII)               # monotone within-He-II ramp
    dict_shapes = [s_incoh, s_curv, s_ramp]

    # ---- project each out of span, whiten in M_sub, stack + SVD, keep top-2 ----
    L_M = np.linalg.cholesky(M_sub)                      # M_sub = L_M L_M^T
    cols = []
    for s in dict_shapes:
        sp = _project_out(s, ortho_span, M_sub)          # M-orthogonal to span
        cols.append(L_M.T @ sp)                          # whitened column
    Mw = np.column_stack(cols)                           # (n_heII, n_dict)
    U, S, Vt = np.linalg.svd(Mw, full_matrices=False)
    smax = S[0] if S.size else 0.0
    rank = int(np.sum(S > svd_rank_tol * max(smax, 1e-30)))
    Linv_T = np.linalg.inv(L_M.T)
    bres1_sub = Linv_T @ U[:, 0]                         # M-unit, M-orthogonal to span
    bres2_sub = Linv_T @ U[:, 1]

    bres1 = np.zeros(n_z); bres1[sub_nodes] = bres1_sub
    bres2 = np.zeros(n_z); bres2[sub_nodes] = bres2_sub

    # ---- n_s response pulled into b_res-coords (the M-projection identity) ----
    rhs = B_sub.T @ Cinv_sub @ r_ns_P[cell_idx]
    r_ns_bres = np.linalg.solve(M_sub, rhs)

    # ---- cosines in M_sub ----
    def _cos_to_span(bs):
        proj = np.zeros_like(bs)
        for q in ortho_span:
            proj = proj + inner_M(bs, q) * q
        nb = norm_M(bs)
        return float(norm_M(proj) / nb) if nb > 0 else float("nan")

    cos_span = np.array([_cos_to_span(bres1_sub), _cos_to_span(bres2_sub)])
    cos_ns = np.array([cos_M(bres1_sub, r_ns_bres), cos_M(bres2_sub, r_ns_bres)])
    worst = 1 if abs(cos_ns[0]) >= abs(cos_ns[1]) else 2

    # ---- ANALYTIC worst-n_s out-of-span member b* (Task 2A-2): the TRUE argmax,
    # vs. the z-shape dictionary bres1/bres2 which only approximate it ----
    bstar_raw, f_perp, bstar_degenerate = analytic_worst_ns(r_ns_bres, ortho_span, M_sub)
    if bstar_degenerate:
        # n_s response fully in-span -> no out-of-span direction to inject; fall
        # back to the existing worst-n_s dict member (already out-of-span).
        bstar_sub = bres1_sub if worst == 1 else bres2_sub
        cos_bstar_ns = float(cos_ns[worst - 1])
    else:
        bstar_sub = bstar_raw
        cos_bstar_ns = float(f_perp)
    bstar_cos_span = _cos_to_span(bstar_sub)
    assert bstar_cos_span < 1e-6, (
        f"bstar not out-of-span: cos_M(bstar,span)={bstar_cos_span:.3e} !< 1e-6")
    bstar = np.zeros(n_z); bstar[sub_nodes] = bstar_sub

    return dict(
        sub_nodes=sub_nodes, cell_idx=cell_idx, C_sub=C_sub, Cinv_sub=Cinv_sub,
        B_sub=B_sub, M_sub=M_sub, u1_sub=u1_sub, u2_sub=u2_sub, ortho_span=ortho_span,
        bres1_sub=bres1_sub, bres2_sub=bres2_sub, bres1=bres1, bres2=bres2,
        r_ns_bres=r_ns_bres, r_ns_P_sub=r_ns_P[cell_idx],
        cos_span=cos_span, cos_ns=cos_ns, worst_ns_member=int(worst),
        bstar_sub=bstar_sub, bstar=bstar, cos_bstar_ns=cos_bstar_ns,
        bstar_degenerate=bool(bstar_degenerate),
        S=S, rank=rank, cos_gate=float(cos_gate),
    )


# ---------------------------------------------------------------------------- #
#  Measured 1-sigma resolution error re-read (not stored on the DataLeg)
# ---------------------------------------------------------------------------- #
def reread_resolution_e(leg):
    """Best-effort re-read of the per-cell 1-sigma resolution error aligned to
    THIS leg's kept flat rows (leg.z_row, leg.k). Returns (res_e, is_signed) or
    (None, None) on any failure. The DataLeg does not store resolution_e (it is
    consumed into C_data at load), so we re-map from the raw source by (z,k).

    is_signed=False for all current legs: the sources (syst_e_resolution,
    syst_resolution, esyst_res_ks) are all-positive 1-sigma ERROR ENVELOPES, not
    signed residuals (the Phase-0 semantics gate -> Tier-R is FLAG not hard-gate).
    """
    import hcd_analysis.emulator.data_likelihood as DL
    name = leg.name
    z_row = np.asarray(leg.z_row, float)
    k_row = np.asarray(leg.k, float)
    try:
        if name == "DESI":
            d = np.load(DESI_NPZ, allow_pickle=True)
            zr, kr, er = np.asarray(d["z"], float), np.asarray(d["k"], float), np.asarray(d["syst_e_resolution"], float)
            res_e = _map_by_zk(z_row, k_row, zr, kr, er)
        elif name == "eBOSS":
            d = np.load(EBOSS_NPZ, allow_pickle=True)
            zr, kr, er = np.asarray(d["z"], float), np.asarray(d["k"], float), np.asarray(d["syst_resolution"], float)
            res_e = _map_by_zk(z_row, k_row, zr, kr, er)
        elif name == "KS":
            detail = KS_BASE.rstrip("/") + "/detailed-p1d-results-karacayli_etal2021.txt"
            res_e = DL._read_ks_resolution_e(detail, z_row, k_row)  # aligned to leg rows directly
        else:
            return None, None
        res_e = np.asarray(res_e, float)
        if res_e.shape != k_row.shape or not np.all(np.isfinite(res_e)):
            return None, None
        is_signed = bool(np.any(res_e < 0.0))     # all sources are all-positive -> False
        return res_e, is_signed
    except Exception as exc:               # pragma: no cover - robustness fallback
        print(f"   [WARN] {name}: resolution_e re-read failed ({exc!r}) -> fallback envelope")
        return None, None


def _map_by_zk(z_row, k_row, z_raw, k_raw, e_raw):
    """Nearest-(z,k) map of e_raw (on the raw grid) onto the leg's (z_row, k_row).
    Leg rows are a value-subset of the raw grid, so the nearest match is exact."""
    z_raw = np.asarray(z_raw, float); k_raw = np.asarray(k_raw, float); e_raw = np.asarray(e_raw, float)
    sz = max(np.ptp(z_raw), 1e-9)
    sk = max(np.ptp(k_raw), 1e-9)
    out = np.empty(z_row.shape, float)
    for i, (zz, kk) in enumerate(zip(z_row, k_row)):
        d2 = ((z_raw - zz) / sz) ** 2 + ((k_raw - kk) / sk) ** 2
        out[i] = e_raw[int(np.argmin(d2))]
    return out


# ---------------------------------------------------------------------------- #
#  The heavy builder (ctx + jacfwd; run under SLURM)
# ---------------------------------------------------------------------------- #
def main():
    import os
    import hcd_analysis.emulator  # noqa: F401  (enable x64 BEFORE jax)
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hcd_analysis.emulator.closure_legb import build_legb_ctx, F_RES_PIVOT_Z as CFG_PIVOT
    from hcd_analysis.emulator.data_likelihood import _predict_P_obs_mf
    from hcd_analysis.emulator.data import SAMPLING_LIMITS, normalize_params

    pivot = float(CFG_PIVOT)
    assert abs(pivot - F_RES_PIVOT_Z) < 1e-9, f"pivot drift {pivot} vs {F_RES_PIVOT_Z}"

    print("=== building production Leg-B ctx (MF forward, DESI+KS+eBOSS; f_res ON) ===")
    ctx, _ = build_legb_ctx(with_mf=True, with_eboss=True, sample_res=True,
                            f_res_amp_sigma=0.15,
                            ks_kwargs={"resolution_float": True, "k_max": 0.065})
    model, pf_stats, mf = ctx.model, ctx.pf_stats, ctx.mf
    cache_k = np.asarray(ctx.cache_k)
    z_global = np.asarray(ctx.z_global)
    tau0_mu = np.asarray(ctx.tau0_mu)
    alpha_hcd_fid = jnp.asarray(ctx.alpha_hcd_mu)

    lim = SAMPLING_LIMITS
    theta_phys_fid = 0.5 * (lim[:, 0] + lim[:, 1])
    theta_unit_fid = jnp.asarray(normalize_params(theta_phys_fid))
    ns_phys_fid = float(theta_phys_fid[0])
    print(f"fiducial n_s={ns_phys_fid:.4f}; F_RES_PIVOT_Z={pivot}; Z_HEII={Z_HEII}; "
          f"fallback envelope=+-{ENVELOPE_FALLBACK*100:.0f}%")

    out, rows = {}, []
    n_leg = len(ctx.legs)
    fig, axes = plt.subplots(2, n_leg, figsize=(6.0 * n_leg, 9.2), squeeze=False)

    for col, leg in enumerate(ctx.legs):
        name = leg.name
        if not getattr(leg, "resolution_ready", True):
            print(f"[{name}] WARN resolution_ready=False -> SKIP (untrustworthy R_z proxy)")
            continue

        k_leg = np.asarray(leg.k)
        z_idx = np.asarray(leg.z_idx)
        z_leg = np.asarray(leg.z)
        R_z = np.asarray(leg.R_z)
        C_data = np.asarray(leg.C_data)
        P_data = np.asarray(leg.P_data)
        N, n_z = k_leg.shape[0], int(leg.n_z)
        zrow = z_leg[z_idx]

        # ---- B (N, n_z): B[i, iz] = 2 k_i^2 R_z(z_iz)^2 on z-block iz ----
        B = np.zeros((N, n_z))
        for iz in range(n_z):
            rsel = np.where(z_idx == iz)[0]
            if rsel.size:
                B[rsel, iz] = 2.0 * k_leg[rsel] ** 2 * R_z[iz] ** 2

        # ---- span directions in b_res-space ----
        u1 = np.ones(n_z)
        u2 = np.log((1.0 + z_leg) / (1.0 + pivot))       # logfac per node

        hiZ_cells = zrow >= Z_HEII
        hiZ_nodes = z_leg >= Z_HEII

        # ---- P-space n_s response (AD), exactly as res_corr ----
        tau0_fid_np = np.array([tau0_mu[int(np.argmin(np.abs(z_global - zz)))] for zz in z_leg])
        rest0 = theta_unit_fid[1:]
        dla_core = ctx.dla_core_leg[name]

        def _logP_ns(ns_unit):
            theta = jnp.concatenate([ns_unit[None], rest0])
            P = jnp.zeros(N)
            for iz in range(n_z):
                rsel = np.where(z_idx == iz)[0]
                if rsel.size == 0:
                    continue
                z_unit = float(leg.z_unit[iz])
                P_cache = _predict_P_obs_mf(mf, model, theta, z_unit,
                                            jnp.asarray(tau0_fid_np[iz]),
                                            alpha_hcd_fid, pf_stats, dla_core[iz])
                P_z = jnp.interp(jnp.asarray(k_leg[rsel]), jnp.asarray(cache_k), P_cache)
                P = P.at[jnp.asarray(rsel)].set(P_z)
            return jnp.log(P)

        r_ns = np.asarray(jax.jacfwd(_logP_ns)(theta_unit_fid[0]))

        # ---- CORE: the out-of-span basis (pure helper) ----
        res = build_oos_bres(B, C_data, u1, u2, hiZ_cells, hiZ_nodes, r_ns)
        bres1, bres2 = res["bres1"], res["bres2"]
        cos_span, cos_ns = res["cos_span"], res["cos_ns"]
        worst, S = res["worst_ns_member"], res["S"]
        bstar = res["bstar"]
        cos_bstar_ns, bstar_degenerate = res["cos_bstar_ns"], res["bstar_degenerate"]
        if res["rank"] < 2:
            print(f"   [WARN] {name}: OOS residual rank={res['rank']} (<2); S={np.array2string(S, precision=2)}")

        # ---- envelope target (measured 1-sigma if re-readable, else +-5%) ----
        z34 = (zrow >= 3.0) & (zrow <= 4.0)
        band = z34 if z34.any() else hiZ_cells
        res_e, is_signed = reread_resolution_e(leg)
        if res_e is not None and P_data is not None:
            frac = np.abs(res_e[band]) / np.maximum(np.abs(P_data[band]), 1e-30)
            env_target = float(np.median(frac))
            env_used = "resolution_e"
        else:
            env_target = ENVELOPE_FALLBACK
            env_used = "fallback_5pct"
        env_target = float(np.clip(env_target, 1e-4, 0.5))

        # ---- scale each bres_j to that envelope (in the exp-multiplier sense) ----
        scaled, env_realized = [], []
        for bres in (bres1, bres2):
            bres_s, er = _scale_to_envelope(bres, B, band, env_target)
            scaled.append(bres_s); env_realized.append(er)
        bres1_s, bres2_s = scaled
        env_realized = np.array(env_realized)

        # ---- scale b* (Task 2A-2) to the SAME envelope, same helper ----
        bstar_s, env_realized_bstar = _scale_to_envelope(bstar, B, band, env_target)
        gate_member = "bstar" if not bstar_degenerate else f"bres{worst}"

        # ---- k^2 R_z^2 at k_max (leg-triviality is empirical, blocker B3) ----
        imk = int(np.argmax(k_leg))
        k2Rz2_kmax = float(k_leg[imk] ** 2 * R_z[z_idx[imk]] ** 2)
        k2Rz2_cellmax = float(np.max(k_leg ** 2 * R_z[z_idx] ** 2))

        # ---- Tier-R (realistic) member from the measured resolution error ----
        bres_real, bres_real_isflag, bres_real_absent, kshape_uncaptured = _tier_r_member(
            B, C_data, u1, u2, P_data, res_e, is_signed, n_z)

        # ---- report + npz ----
        assert cos_span[0] < COS_GATE, f"[{name}] cos_M(bres1,span)={cos_span[0]:.4f} !< {COS_GATE}"
        assert cos_span[1] < COS_GATE, f"[{name}] cos_M(bres2,span)={cos_span[1]:.4f} !< {COS_GATE}"

        print(f"\n[{name}] N={N} n_z={n_z} He-II nodes={int(hiZ_nodes.sum())} cells={int(hiZ_cells.sum())}"
              f"  k[{k_leg.min():.4f},{k_leg.max():.4f}]")
        print(f"   cos_M(bres1,span)={cos_span[0]:+.2e}  cos_M(bres2,span)={cos_span[1]:+.2e}  "
              f"(gate < {COS_GATE}) -> OUT-OF-SPAN OK")
        print(f"   cos_M(bres1,n_s)={cos_ns[0]:+.4f}  cos_M(bres2,n_s)={cos_ns[1]:+.4f}  "
              f"-> worst-n_s member = bres{worst}")
        print(f"   env_used={env_used} env_target={env_target*100:.3f}%  "
              f"realized bres1={env_realized[0]*100:.3f}% bres2={env_realized[1]*100:.3f}%")
        print(f"   k^2 R_z^2 @kmax={k2Rz2_kmax:.4g} (max cell {k2Rz2_cellmax:.4g})  "
              f"Tier-R: isflag={bres_real_isflag} absent={bres_real_absent} "
              f"kshape_uncaptured={kshape_uncaptured:.3f}")
        print(f"   [Task 2A-2] cos_M(bstar,n_s)={cos_bstar_ns:+.4f} (analytic ARGMAX, vs dict-"
              f"captured max={max(abs(cos_ns[0]), abs(cos_ns[1])):.4f})  degenerate={bstar_degenerate}  "
              f"realized bstar={env_realized_bstar*100:.3f}%  -> PRIMARY gate_member={gate_member}")

        rows.append(dict(name=name, cos_span=cos_span, cos_ns=cos_ns, worst=worst,
                         env_used=env_used, env_realized=env_realized, k2Rz2_kmax=k2Rz2_kmax,
                         isflag=bres_real_isflag, absent=bres_real_absent,
                         kshape=kshape_uncaptured, n_heII=int(hiZ_nodes.sum()),
                         cos_bstar_ns=cos_bstar_ns, bstar_degenerate=bstar_degenerate,
                         gate_member=gate_member))

        out[f"{name}_bres1"] = bres1_s.astype(float)
        out[f"{name}_bres2"] = bres2_s.astype(float)
        out[f"{name}_bres_real"] = bres_real.astype(float)
        out[f"{name}_bres_real_isflag"] = bool(bres_real_isflag)
        out[f"{name}_bres_real_absent"] = bool(bres_real_absent)
        out[f"{name}_z"] = z_leg.astype(float)
        out[f"{name}_R_z"] = R_z.astype(float)
        out[f"{name}_cos_span"] = cos_span.astype(float)
        out[f"{name}_cos_ns"] = cos_ns.astype(float)
        out[f"{name}_worst_ns_member"] = int(worst)
        out[f"{name}_env_realized"] = env_realized.astype(float)
        out[f"{name}_k2Rz2_kmax"] = float(k2Rz2_kmax)
        out[f"{name}_env_used"] = env_used
        out[f"{name}_kshape_uncaptured_frac"] = float(kshape_uncaptured)
        # Task 2A-2: the analytic worst-n_s out-of-span member b* + the PRIMARY
        # gate-member selector the Phase-2 adversarial injection should consume.
        out[f"{name}_bstar"] = bstar_s.astype(float)
        out[f"{name}_cos_bstar_ns"] = float(cos_bstar_ns)
        out[f"{name}_bstar_degenerate"] = bool(bstar_degenerate)
        out[f"{name}_env_realized_bstar"] = float(env_realized_bstar)
        out[f"{name}_gate_member"] = gate_member
        # extras (debug/figure)
        out[f"{name}_r_ns_bres"] = res["r_ns_bres"].astype(float)
        out[f"{name}_sub_nodes"] = res["sub_nodes"].astype(int)

        # ---- figure ----
        ax = axes[0][col]
        ax.plot(z_leg, bres1_s, "-o", color="C3", lw=2, ms=4, label="bres1 (per-z)")
        ax.plot(z_leg, bres2_s, "-o", color="C2", lw=2, ms=4, label="bres2 (per-z)")
        rb = np.zeros(n_z); rb[res["sub_nodes"]] = res["r_ns_bres"]
        rb_s = rb * (np.max(np.abs(bres1_s)) / max(np.max(np.abs(rb)), 1e-30))
        ax.plot(z_leg, rb_s, "--", color="C0", lw=1.2, label="r_ns_bres (rescaled)")
        ax.axvline(Z_HEII, color="k", ls=":", lw=1, label=f"z={Z_HEII}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("z"); ax.set_ylabel("b_res(z) [log-mult /2k^2R^2]" if col == 0 else "")
        ax.set_title(f"{name}  cos(bres,span)=[{cos_span[0]:.1e},{cos_span[1]:.1e}]\n"
                     f"cos(bres,n_s)=[{cos_ns[0]:+.2f},{cos_ns[1]:+.2f}] worst=bres{worst}", fontsize=8.5)
        ax.legend(fontsize=6.5, loc="best")

        ax2 = axes[1][col]
        p1 = B @ bres1_s
        sc = ax2.scatter(k_leg, zrow, c=p1, cmap="RdBu_r",
                         vmin=-env_realized[0], vmax=env_realized[0], s=14)
        ax2.axhline(Z_HEII, color="k", ls=":", lw=1)
        ax2.set_xscale("log"); ax2.set_xlabel("k [s/km]")
        ax2.set_ylabel("z" if col == 0 else "")
        ax2.set_title(f"{name}  P-perturbation exp(B@bres1)-1  (env {env_used})", fontsize=8.5)
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.02, label="log-P pert")

    fig.suptitle(f"OUT-OF-SPAN z-incoherent instrument-resolution basis (M=B^T C^-1 B pullback; "
                 f"span{{1,logfac}}; He-II z>={Z_HEII}; fiducial n_s={ns_phys_fid:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nsaved figure -> {FIG}")

    out["_meta_z_heii"] = Z_HEII
    out["_meta_f_res_pivot_z"] = pivot
    out["_meta_cos_gate"] = COS_GATE
    out["_meta_envelope_fallback"] = ENVELOPE_FALLBACK
    out["_meta_legs"] = np.array([r["name"] for r in rows])
    out["_meta_doc"] = np.array(
        "Per-leg z-incoherent OUT-OF-SPAN instrument-resolution injection basis. "
        "{leg}_bres1,{leg}_bres2 are PER-Z (n_z,) resolution perturbations applied as "
        "P_truth *= exp(2*b_res(z)*k^2*R_z(z)^2). Built M_sub-orthogonal to span{u1=1, "
        "u2=log((1+z)/(1+3))} (the 2-param f_res span) in the pullback metric "
        "M_sub=B_sub^T C_sub^-1 B_sub, localized to z>=2.8, scaled to the leg's 1-sigma "
        "resolution envelope. worst_ns_member (1/2) = the max-|cos_M(bres,n_s)| member. "
        "bres_real = Tier-R representative from the measured resolution error (isflag=True "
        "when the error is a symmetric 1-sigma envelope, FLAG not hard-gate). "
        "(Task 2A-2) {leg}_bstar = the ANALYTIC worst-n_s out-of-span member (the TRUE "
        "argmax of cos_M(b,n_s) over out-of-span b, closed form b*=normalize_M(P_perp_"
        "span(r_ns_bres))), scaled to the same envelope; {leg}_cos_bstar_ns = the achieved "
        "cosine f_perp = ||P_perp_span(r_ns_bres)||_M/||r_ns_bres||_M (>= either dict "
        "member's |cos_ns|); {leg}_bstar_degenerate=True iff the n_s response is fully "
        "in-span (no out-of-span n_s direction exists -> bstar falls back to the dict "
        "worst member); {leg}_gate_member = the PRIMARY member (\"bstar\" unless "
        "degenerate) the Phase-2 adversarial injection should use."
    )
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"saved basis -> {OUT_NPZ}")

    print("\n================ PER-LEG INSTRUMENT-RES BASIS REPORT ================")
    print(f"{'leg':<7}{'cos(bres1,span)':>16}{'cos(bres2,span)':>16}"
          f"{'cos(b1,ns)':>12}{'cos(b2,ns)':>12}{'worst':>7}{'env_used':>14}"
          f"{'k2Rz2@kmax':>12}{'Tier-R':>18}")
    for r in rows:
        tr = ("absent" if r["absent"] else ("FLAG" if r["isflag"] else "residual")) + f"/{r['kshape']:.2f}"
        print(f"{r['name']:<7}{r['cos_span'][0]:>16.2e}{r['cos_span'][1]:>16.2e}"
              f"{r['cos_ns'][0]:>+12.3f}{r['cos_ns'][1]:>+12.3f}{'bres'+str(r['worst']):>7}"
              f"{r['env_used']:>14}{r['k2Rz2_kmax']:>12.4g}{tr:>18}")
    allok = all(r["cos_span"][0] < COS_GATE and r["cos_span"][1] < COS_GATE for r in rows)
    print(f"\nASSERT cos_M(bres, span f_res) < {COS_GATE} for ALL members, ALL legs: "
          f"{'PASS' if allok else 'FAIL'}")
    print("Dictionary worst-n_s member (bres1/bres2) per leg:")
    for r in rows:
        wc = r["cos_ns"][r["worst"] - 1]
        print(f"   {r['name']:<7} -> bres{r['worst']}  (cos_M(bres{r['worst']},n_s)={wc:+.4f})")
    print("\n(Task 2A-2) PRIMARY gate member (analytic worst-n_s b*, TRUE argmax) per leg:")
    for r in rows:
        print(f"   {r['name']:<7} -> {r['gate_member']}  "
              f"(cos_M(bstar,n_s)={r['cos_bstar_ns']:+.4f}, degenerate={r['bstar_degenerate']})")


def _tier_r_member(B, C_data, u1, u2, P_data, res_e, is_signed, n_z):
    """Tier-R: fit the best-fit per-z b_res to the measured resolution error in the
    C_data^-1-weighted sense (b_res_meas = M^-1 B^T C^-1 logfrac), M-project out
    span{u1,u2} -> the part the 2-param z-model cannot represent. Returns
    (bres_real (n_z,), isflag, absent, kshape_uncaptured_frac).

    All current legs' resolution_e is an all-positive 1-sigma ENVELOPE (not a signed
    residual) so isflag=True (FLAG not hard-gate); absent=True when it is fully
    in-span (the z-model captures it) or when resolution_e is unavailable.
    """
    if res_e is None or P_data is None:
        return np.zeros(n_z), False, True, float("nan")
    B = np.asarray(B, float); C_data = np.asarray(C_data, float)
    Cinv = np.linalg.inv(C_data)
    logfrac = np.asarray(res_e, float) / np.maximum(np.abs(np.asarray(P_data, float)), 1e-30)

    M_full = B.T @ Cinv @ B
    M_full = 0.5 * (M_full + M_full.T)
    b_meas = np.linalg.solve(M_full, B.T @ Cinv @ logfrac)

    # fraction of the resolution_e norm NOT captured by the k^2 R_z^2 projection
    p_fit = B @ b_meas
    resid = logfrac - p_fit
    _, norm_C, _ = _whitened_inner(Cinv)
    denom = norm_C(logfrac)
    kshape_uncaptured = float(norm_C(resid) / denom) if denom > 0 else float("nan")

    # M-project out span{u1,u2}
    _, norm_M, cos_M = _whitened_inner(M_full)
    ortho_span = _gram_schmidt_whitened([np.asarray(u1, float), np.asarray(u2, float)], M_full)
    bres_real = _project_out(b_meas, ortho_span, M_full)

    isflag = not bool(is_signed)                   # env (not signed residual) -> FLAG not hard-gate
    n_meas = norm_M(b_meas)
    n_real = norm_M(bres_real)
    if n_meas <= 0 or n_real < 1e-8 * n_meas:      # fully in-span -> the z-model captures it
        return np.zeros(n_z), isflag, True, kshape_uncaptured
    return bres_real.astype(float), isflag, False, kshape_uncaptured


if __name__ == "__main__":
    main()
