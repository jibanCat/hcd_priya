#!/usr/bin/env python3
"""Phase-C T4 §2 — Validate C_emu as an error model (NO-NUTS certification).

The gate BEFORE Leg-B coverage: is the emulator-error covariance C_emu a valid error
model for the held-out-sim residual? Four cheap, figure-producing checks on the
production (``final_fold0``, ``error_vector.npz``) pair (plan §2):

  (2) WHITENING (the key check): pool fold-0 HELD-OUT P_obs residuals
      ``r = P_obs_emu − P_obs_truth`` at each row's OWN contamination α = w_c_cache;
      whiten by C_emu ALONE (``cosmic_cov=0`` — the sim residual IS the emulator error,
      no observational noise) via ``closure_diagnostics.whitening_test``. mean≈0?
      var≈1? χ²/dof≈1? KS-uniform? var≫1 ⇒ error vector UNDER-sized (→ the cemu_inflate
      the calibration needs); var≪1 ⇒ conservative. Q-Q + per-k χ²/dof strip.
  (3) diag(C_emu)/diag(C_cosmic) heatmap over (k,z) per τ₀-band, at 2% & 5% cosmic
      floors (production DESI/KODIAQ cov NOT wired — illustrative). Marks where C_emu
      DOMINATES (ratio>1) — the field-standard "is the emulator error sub-dominant".
  (4) BIAS-vs-VARIANCE: per (class,k,z) cell, mean(r_frac) next to RMS(r_frac) (the
      error_vector σ IS the RMS). |mean|≳RMS ⇒ BIAS-dominated ⇒ the symmetric-Gaussian
      C_emu needs a caveat. mean/RMS heatmap per class.
  (5) LF-Nyquist k_max: the error vector has k_min but NO k_max. k_Nyq = π/dv (kfkms
      ANGULAR). Report #in-range bins above it + the proposed k_max for DATA_RANGE.

Conventions (matched to inference.py / run_loso_sweep.py):
  - FRACTIONAL per-class residual r_frac_c(k) = (P_filt_emu_c − P_filt_truth_c)/P_filt_truth_c
    (run_loso_sweep's EXACT sign: emu − truth; the σ error vector is RMS of THIS). The
    whitening/bias use the SAME emu−truth sign so the mean's sign is the emulator's bias.
  - ASSEMBLED residual r(k) = P_obs_emu − P_obs_truth, both = Σ_c coef_c·P_cls_c with
    coef = [1−Σα, α], α = w_c_cache[row,1:] (the row's OWN contamination). P_cls TRUTH =
    cache P_filt (+ dla_core on the DLA class); P_cls EMU = emulated P_filt (+ SAME core).
    The dla_core CANCELS in the difference (both sides carry it), so the assembled residual
    is purely the emulator P_filt error propagated through the row's class coefficients —
    exactly what C_emu = Σ coef²σ²P² is built to size. (Toy build_ctx dla_core is irrelevant
    here for the SAME reason; we use the cache's REAL per-row DLA delta for completeness.)
  - C_emu assembled by ``inference.predict_P_obs_and_cov_single_z`` with cosmic_cov=0 (the
    EXACT per-class emu_var the likelihood uses), at the row's (θ9_unit, z, τ₀, α).

Does NOT modify any committed library module. Proposes (does not edit) the DATA_RANGE k_max.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/diag_cemu_validation.py [--max-rows N]
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax
import jax.numpy as jnp
import h5py

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import (
    load_cache, make_splits, datarange_mask, DATA_RANGE, COARSE_NAMES,
)
from hcd_analysis.emulator.predict import predict_P_filt
from hcd_analysis.emulator.inference import predict_P_obs_and_cov_single_z

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/final_fold0"
ERROR_VECTOR = f"{REPO}/checkpoints/error_vector.npz"
FIGDIR = f"{REPO}/figures/analysis/05_likelihood"

# Kim2013 ladder (== data.KIM_AMP/SLOPE) for the z-band edge selection commentary only.
_KIM_AMP, _KIM_SLOPE = 2.3e-3, 3.65


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# 1. Held-out residual extraction (per-class P_filt + assembled P_obs).
# ----------------------------------------------------------------------------
def extract_holdout_residuals(d, model, pf, val_idx, sigma, z_band_edges,
                              alpha_centres, dla_shot_flag_k, max_rows=None, rho=None):
    """Per held-out row: (r_frac (4,K), r_obs (K,), C_emu (K,K), valid (K,), z, zb, cls coef).

    r_frac = (P_filt_emu − P_filt_truth)/P_filt_truth  (run_loso_sweep sign: emu−truth).
    r_obs  = P_obs_emu − P_obs_truth at α = w_c_cache[row,1:] (the row's OWN contamination);
             P_cls truth uses the cache P_filt (+ per-row DLA core); P_cls emu uses the
             emulated P_filt (+ the SAME core → core cancels). C_emu = the likelihood's
             per-class Σ coef²σ²P² at this row, cosmic_cov=0 (whiten by C_emu ALONE).

    ``rho`` (opt-in, (4,4,K,Zb,Tb)): assemble the CROSS-CLASS C_emu instead of the diagonal
    one (the re-validation of the cross-class upgrade); the row's z-band slice of ``rho`` is
    passed as ``rho_zb`` to ``predict_P_obs_and_cov_single_z``. ``rho=None`` → the diagonal
    σ path (the original whitening).

    valid = in-data-range (z∈[2.2,4.6], k≥k_min) AND finite cache P_filt across all classes
    (above-native-Nyquist bins are NaN; here all 172 LF bins are finite). Returns lists.
    """
    val_idx = np.asarray(val_idx)
    if max_rows is not None and len(val_idx) > max_rows:
        # even stride so all z / τ₀ bands are represented (rows are sim-major within a fold).
        sel = np.linspace(0, len(val_idx) - 1, max_rows).round().astype(int)
        val_idx = val_idx[np.unique(sel)]

    Zb = sigma.shape[2]
    z_grid = d["z_grid"]
    x_all = d["x"]                                # (R,10) [θ9_unit, z_unit]
    tau0_all = d["tau0"]
    P_filt_true_all = d["P_filt"]                 # (R,4,K) LINEAR truth
    delta_all = d["delta"]                        # (R,3,K) = P_unf − P_filt (LLS,sub,DLA)
    w_c_all = d["w_c_cache"]                       # (R,4)
    inrange_all = datarange_mask(d)               # (R,K) z∈[2.2,4.6] & k≥k_min

    K = sigma.shape[1]
    n = len(val_idx)
    # --- stack per-row inputs so the EXACT likelihood assembly vmaps (compile ONCE) ---
    x_sel = jnp.asarray(x_all[val_idx])                              # (n,10)
    tau0_sel = jnp.asarray(tau0_all[val_idx])                       # (n,)
    z_sel = jnp.asarray(z_grid[val_idx])                            # (n,)
    a_sel = jnp.asarray(w_c_all[val_idx, 1:])                       # (n,3) [LLS,sub,DLA]
    dla_core_sel = jnp.asarray(delta_all[val_idx, 2])              # (n,K) cache DLA core
    zb_of_row = np.clip(np.digitize(z_grid[val_idx], z_band_edges[1:-1]), 0, Zb - 1)
    # sigma[:,:,zb_of_row,:] -> (4,K,n,Tb); want (n,4,K,Tb) for the vmap over rows.
    sigma_zb_sel = jnp.asarray(sigma[:, :, zb_of_row, :].transpose(2, 0, 1, 3))  # (n,4,K,Tb)
    flag = jnp.asarray(dla_shot_flag_k)
    use_xclass = rho is not None
    # rho[:,:,:,zb_of_row,:] -> (4,4,K,n,Tb); want (n,4,4,K,Tb) for the per-row vmap.
    rho_zb_sel = (jnp.asarray(rho[:, :, :, zb_of_row, :].transpose(3, 0, 1, 2, 4))
                  if use_xclass else None)                                       # (n,4,4,K,Tb)

    def assemble_one(x_row, z, t0, a, sz, dc, rz):
        # the EXACT likelihood (P_obs, C_emu) — cosmic_cov=0 (whiten by C_emu ALONE).
        return predict_P_obs_and_cov_single_z(
            model, x_row[:9], x_row[9], z, t0, a, pf_stats=pf, sigma_zb=sz,
            alpha_centres=alpha_centres, cosmic_cov=jnp.zeros(K),
            dla_core=dc, dla_shot_flag=flag, rho_zb=rz)
    in_axes = (0, 0, 0, 0, 0, 0, 0 if use_xclass else None)
    P_obs_emu_all, C_emu_all = jax.vmap(assemble_one, in_axes=in_axes)(
        x_sel, z_sel, tau0_sel, a_sel, sigma_zb_sel, dla_core_sel, rho_zb_sel)   # (n,K),(n,K,K)
    P_obs_emu_all = np.asarray(P_obs_emu_all)
    C_emu_all = np.asarray(C_emu_all)
    # emulated per-class P_filt (same forward; vmap once).
    P_filt_emu_all = np.asarray(jax.vmap(
        lambda xr, t0: predict_P_filt(model, xr[:9], xr[9], t0, pf))(x_sel, tau0_sel))  # (n,4,K)

    out = []
    for i, row in enumerate(val_idx):
        P_emu = P_filt_emu_all[i]                                     # (4,K)
        P_true = P_filt_true_all[row]                                 # (4,K)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_frac = (P_emu - P_true) / P_true                       # (4,K) emu−truth
        a = w_c_all[row, 1:]
        coef = np.concatenate([[1.0 - a.sum()], a])                  # (4,)
        dla_core_row = delta_all[row, 2]                             # (K,)
        # truth P_obs: SAME assembly with the CACHE per-class P_filt (core cancels in emu−truth).
        P_cls_true = np.stack([P_true[0], P_true[1], P_true[2],
                               P_true[3] + dla_core_row])             # (4,K) [clean,LLS,sub,DLA_unf]
        P_obs_true = np.einsum("c,ck->k", coef, P_cls_true)          # (K,)
        r_obs = P_obs_emu_all[i] - P_obs_true                        # (K,) emu − truth
        valid = (np.asarray(inrange_all[row])
                 & np.isfinite(P_true).all(0) & np.isfinite(P_emu).all(0)
                 & (P_true != 0.0).all(0))                            # (K,)
        out.append(dict(row=int(row), z=float(z_grid[row]), zb=int(zb_of_row[i]),
                        coef=coef, r_frac=r_frac, r_obs=r_obs,
                        C_emu=C_emu_all[i], valid=valid))
    return out


# ----------------------------------------------------------------------------
# 2. Whitening test on the pooled assembled P_obs residuals.
# ----------------------------------------------------------------------------
def run_whitening(recs):
    """Whiten each row's r_obs by its C_emu (in-range bins only) and pool.

    Per row we restrict r and C to that row's valid (in-range, finite) k-bins, so the
    pooled whitened set never includes a bin C_emu does not cover (out-of-range σ=NaN →
    emu_var=0 → C singular there). ``whitening_test`` jitters C before cholesky exactly
    like ``gaussian_loglik``. Variable K per row → pool the whitened components, plus a
    per-k-bin χ²/dof on the COMMON valid grid for the strip figure.
    """
    from scipy.linalg import solve_triangular
    whit_pool = []           # pooled whitened components across rows
    K = recs[0]["r_obs"].shape[0]
    # per-(row,k) whitened value on the full K grid (NaN where invalid) for the k-strip.
    Wk = np.full((len(recs), K), np.nan)
    JITTER = 1e-10           # matches closure_diagnostics.whiten_residual / gaussian_loglik
    for m, rec in enumerate(recs):
        vk = rec["valid"]
        if vk.sum() < 2:
            continue
        r = rec["r_obs"][vk]
        C = rec["C_emu"][np.ix_(vk, vk)].copy()
        # SAME SPD jitter + lower-Cholesky + forward-solve as whiten_residual, but lean
        # (solve_triangular, no per-row scipy stats) so 600 rows are ~instant.
        C[np.diag_indices_from(C)] += JITTER * np.mean(np.diag(C))
        L = np.linalg.cholesky(C)
        w = solve_triangular(L, r, lower=True)                       # r_white = L⁻¹ r
        whit_pool.append(w)
        Wk[m, vk] = w
    whit_pool = np.concatenate(whit_pool)
    # pooled moments of the already-whitened pool (each row whitened by its OWN C above).
    # Computed directly (NOT via whitening_test+identity, which would Cholesky an N×N matrix).
    from scipy import stats
    n = whit_pool.size
    mean = float(whit_pool.mean())
    var = float(whit_pool.var(ddof=1))
    mean_z = mean * np.sqrt(n)
    chi2_var = (n - 1) * var
    pooled = dict(
        n=n, mean=mean, var=var, mean_z=mean_z,
        mean_p=2.0 * stats.norm.sf(abs(mean_z)),
        var_chi2_p=2.0 * min(stats.chi2.cdf(chi2_var, n - 1), stats.chi2.sf(chi2_var, n - 1)),
        chi2_over_dof=float(np.sum(whit_pool ** 2) / n))
    ks_stat, ks_p = stats.kstest(whit_pool, "norm")
    pooled["ks_stat"], pooled["ks_p"] = float(ks_stat), float(ks_p)
    # per-k-bin χ²/dof (mean of whitened² over rows with that bin valid).
    with np.errstate(invalid="ignore"):
        chi2_k = np.nanmean(Wk ** 2, axis=0)                         # (K,)
        n_k = np.isfinite(Wk).sum(axis=0)                            # rows per bin
    return pooled, whit_pool, chi2_k, n_k, Wk


def fig_whitening(pooled, whit_pool, chi2_k, n_k, kgrid, k_max, figdir):
    """Q-Q (pooled r_white vs N(0,1)) + per-k-bin χ²/dof strip."""
    from scipy import stats
    plt = _plt()
    fig, (axq, axs) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- Q-Q ---
    w = np.sort(whit_pool)
    n = w.size
    q_theory = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    axq.plot(q_theory, w, ".", ms=2, color="C0", alpha=0.4, rasterized=True)
    lim = [min(q_theory.min(), w.min()), max(q_theory.max(), w.max())]
    axq.plot(lim, lim, "k--", lw=1.2, label="N(0,1) ideal")
    axq.set_xlabel("theoretical N(0,1) quantile")
    axq.set_ylabel(r"pooled whitened residual  $r_{\rm white}=L^{-1}r$")
    axq.set_title(f"Q-Q  (var={pooled['var']:.3f}, "
                  f"$\\chi^2$/dof={pooled['chi2_over_dof']:.3f}, KS p={pooled['ks_p']:.2g})")
    axq.legend(fontsize=9); axq.grid(alpha=0.3); axq.set_aspect("equal", "box")

    # --- per-k χ²/dof strip ---
    k = np.asarray(kgrid)
    ok = np.isfinite(chi2_k) & (n_k > 0)
    axs.semilogx(k[ok], chi2_k[ok], "-o", ms=3, color="C3")
    axs.axhline(1.0, color="k", ls="--", lw=1.0, label=r"$\chi^2$/dof = 1 (ideal)")
    axs.axhspan(0.5, 2.0, color="green", alpha=0.08, label="0.5–2 band")
    if k_max is not None:
        axs.axvline(k_max, color="purple", ls=":", lw=1.4,
                    label=f"LF k_max={k_max:.3g}")
    axs.set_xlabel("k [s/km] (angular)")
    axs.set_ylabel(r"per-k-bin $\chi^2$/dof  $=\langle r_{\rm white}^2\rangle_{\rm rows}$")
    axs.set_title("where C_emu mis-sizes the residual (>1 under-sized, <1 conservative)")
    axs.legend(fontsize=8); axs.grid(alpha=0.3, which="both")

    fig.suptitle("C_emu whitening test — pooled held-out P_obs residuals "
                 f"(N={n} components, {int((n_k>0).sum())} k-bins)")
    p = Path(figdir) / "cemu_whitening_qq.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


# ----------------------------------------------------------------------------
# 2b. CROSS-CLASS re-validation: pool held-out residuals over ALL 8 folds, whiten by
#     BOTH the diagonal and the cross-class C_emu, and produce the before/after Q-Q.
# ----------------------------------------------------------------------------
def revalidate_xclass(d, pf_by_fold, models, sigma, rho, z_band_edges, alpha_centres,
                      dla_shot_flag_k, n_folds=8, max_rows_per_fold=400):
    """Pool the held-out residuals from ALL ``n_folds`` folds and whiten by the DIAGONAL
    C_emu (``rho=None``) AND the CROSS-CLASS C_emu (``rho``). Each fold uses ITS OWN trained
    model + norm (LOSO: a sim is held out in exactly one fold), so the pool is every sim's
    held-out residual exactly once — the honest all-folds whitening the cross-class build was
    fit to. Returns ``(pooled_diag, whit_diag, chi2k_diag, ..., pooled_x, whit_x, ...)``."""
    recs_diag, recs_x = [], []
    for f in range(n_folds):
        model = models[f]
        pf = pf_by_fold[f]
        _tr, va, _ho = make_splits(d, f, n_folds=n_folds)
        rd = extract_holdout_residuals(d, model, pf, va, sigma, z_band_edges,
                                       alpha_centres, dla_shot_flag_k,
                                       max_rows=max_rows_per_fold, rho=None)
        rx = extract_holdout_residuals(d, model, pf, va, sigma, z_band_edges,
                                       alpha_centres, dla_shot_flag_k,
                                       max_rows=max_rows_per_fold, rho=rho)
        recs_diag.extend(rd)
        recs_x.extend(rx)
        print(f"  fold {f}: pooled {len(rd)} held-out rows "
              f"({int(np.sum([r['valid'].sum() for r in rd]))} valid k-bins)")
    pooled_d, whit_d, chi2k_d, nk_d, Wk_d = run_whitening(recs_diag)
    pooled_x, whit_x, chi2k_x, nk_x, Wk_x = run_whitening(recs_x)
    return (pooled_d, whit_d, chi2k_d, nk_d), (pooled_x, whit_x, chi2k_x, nk_x)


def fig_whitening_before_after(diag, xcl, kgrid, k_max, figdir):
    """Before (diagonal) vs After (cross-class) Q-Q + per-k χ²/dof ramp, side by side."""
    from scipy import stats
    plt = _plt()
    (pd_, wd, ck_d, nk_d) = diag
    (px, wx, ck_x, nk_x) = xcl
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    for col, (pooled, w, ck, nk, tag, color) in enumerate([
            (pd_, wd, ck_d, nk_d, "BEFORE: diagonal C_emu", "C0"),
            (px, wx, ck_x, nk_x, "AFTER: cross-class C_emu", "C2")]):
        axq = axes[0][col]
        ws = np.sort(w); n = ws.size
        q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        axq.plot(q, ws, ".", ms=2, color=color, alpha=0.4, rasterized=True)
        lim = [min(q.min(), ws.min()), max(q.max(), ws.max())]
        axq.plot(lim, lim, "k--", lw=1.2, label="N(0,1) ideal")
        axq.set_xlabel("theoretical N(0,1) quantile")
        axq.set_ylabel(r"pooled whitened residual $L^{-1}r$")
        axq.set_title(f"{tag}\nvar={pooled['var']:.3f}  "
                      f"$\\chi^2$/dof={pooled['chi2_over_dof']:.3f}  KS p={pooled['ks_p']:.2g}")
        axq.legend(fontsize=9); axq.grid(alpha=0.3); axq.set_aspect("equal", "box")

        axs = axes[1][col]
        k = np.asarray(kgrid); ok = np.isfinite(ck) & (nk > 0)
        axs.semilogx(k[ok], ck[ok], "-o", ms=3, color=color)
        axs.axhline(1.0, color="k", ls="--", lw=1.0, label=r"$\chi^2$/dof = 1")
        axs.axhspan(0.5, 2.0, color="green", alpha=0.08, label="0.5–2 band")
        if k_max is not None:
            axs.axvline(k_max, color="purple", ls=":", lw=1.4, label=f"k_max={k_max:.3g}")
        axs.set_xlabel("k [s/km]"); axs.set_ylabel(r"per-k $\chi^2$/dof")
        axs.set_title("per-k χ²/dof ramp"); axs.legend(fontsize=8)
        axs.grid(alpha=0.3, which="both"); axs.set_ylim(0, max(3.0, np.nanmax(ck_d[np.isfinite(ck_d)]) * 1.1))

    fig.suptitle("C_emu whitening — diagonal → cross-class (pooled ALL 8 folds' held-out "
                 f"residuals, N_diag={wd.size}, N_xcl={wx.size})", fontsize=13)
    p = Path(figdir) / "cemu_whitening_qq_xclass.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


# ----------------------------------------------------------------------------
# 3. diag(C_emu)/diag(C_cosmic) heatmap (k,z) per τ₀-band, at 2% & 5%.
# ----------------------------------------------------------------------------
def cemu_diag_grid(d, model, pf, sigma, z_band_edges, alpha_centres, dla_shot_flag_k,
                   z_repr, alpha_bands, theta_unit_repr=0.5):
    """diag(C_emu)(k) on a (z, τ₀-band) grid at a representative interior (θ, α_HCD).

    At each (z, τ₀-band-centre) we forward the emulator at a representative interior θ
    (unit-cube centre) + the HCD contamination α_HCD ≈ a small interior incidence, set τ₀
    = α_band·Kim(z), and read diag(C_emu) = emu_var = Σ_c coef²σ²P² (cosmic_cov=0). Returns
    ``cemu_diag (n_z, Tb, K)`` and ``P_obs (n_z, Tb, K)`` (the latter supplies the X% cosmic
    floor diag(C_cosmic) = (frac·P_obs)²).
    """
    Zb = sigma.shape[2]
    n_z = len(z_repr)
    Tb = len(alpha_bands)
    K = sigma.shape[1]
    theta9 = jnp.full(9, theta_unit_repr)
    # a representative interior HCD incidence (small, field-typical post-masking residual).
    alpha_hcd = jnp.asarray([0.05, 0.02, 0.01])               # (LLS,subDLA,DLA)

    # flatten the (z × band) grid and vmap the EXACT assembly over it (compile ONCE).
    z_arr = np.asarray(z_repr)
    zb_of = np.clip(np.digitize(z_arr, z_band_edges[1:-1]), 0, Zb - 1)
    zz, bb = np.meshgrid(np.arange(n_z), np.arange(Tb), indexing="ij")
    zz, bb = zz.ravel(), bb.ravel()                          # (n_z*Tb,)
    z_flat = jnp.asarray(z_arr[zz])
    zu_flat = jnp.asarray((z_arr[zz] - 2.0) / (5.4 - 2.0))
    a_flat = jnp.asarray(np.asarray(alpha_bands)[bb])        # τ₀-band α centre
    tau0_flat = a_flat * (_KIM_AMP * (1.0 + z_flat) ** _KIM_SLOPE)
    sigma_zb_flat = jnp.asarray(sigma[:, :, zb_of[zz], :].transpose(2, 0, 1, 3))  # (G,4,K,Tb)
    dla_core = jnp.zeros(K)
    flag = jnp.asarray(dla_shot_flag_k)

    def one(z, zu, t0, sz):
        return predict_P_obs_and_cov_single_z(
            model, theta9, zu, z, t0, alpha_hcd, pf_stats=pf, sigma_zb=sz,
            alpha_centres=alpha_centres, cosmic_cov=jnp.zeros(K), dla_core=dla_core,
            dla_shot_flag=flag)
    P_obs_g, C_g = jax.vmap(one)(z_flat, zu_flat, tau0_flat, sigma_zb_flat)
    P_obs_g = np.asarray(P_obs_g)                            # (G,K)
    C_diag_g = np.diagonal(np.asarray(C_g), axis1=1, axis2=2)  # (G,K)
    cemu = np.full((n_z, Tb, K), np.nan)
    Pobs = np.full((n_z, Tb, K), np.nan)
    for g in range(zz.size):
        cemu[zz[g], bb[g]] = C_diag_g[g]
        Pobs[zz[g], bb[g]] = P_obs_g[g]
    return cemu, Pobs


def fig_cemu_over_cdata(cemu, Pobs, kgrid, z_repr, alpha_bands, k_max, figdir,
                        fracs=(0.02, 0.05)):
    """Heatmap of diag(C_emu)/diag(C_cosmic) over (k,z) per τ₀-band, at 2% & 5% floors.

    C_cosmic = (frac·P_obs)². ratio>1 ⇒ C_emu DOMINATES (bad). Rows = τ₀-bands; cols =
    cosmic-floor level. Overlays the LF k_max. Log-color, symmetric about ratio=1.
    """
    from matplotlib.colors import LogNorm
    plt = _plt()
    n_z, Tb, K = cemu.shape
    k = np.asarray(kgrid)
    nF = len(fracs)
    fig, axes = plt.subplots(Tb, nF, figsize=(6.2 * nF, 3.1 * Tb),
                             squeeze=False)
    vmin, vmax = 1e-3, 1e1
    ki = np.arange(K)
    for ib in range(Tb):
        for jf, frac in enumerate(fracs):
            ax = axes[ib][jf]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = cemu[:, ib, :] / (frac * Pobs[:, ib, :]) ** 2   # (n_z,K)
            ratio = np.where(np.isfinite(ratio) & (Pobs[:, ib, :] > 0), ratio, np.nan)
            im = ax.pcolormesh(k, np.asarray(z_repr), ratio, shading="nearest",
                               norm=LogNorm(vmin=vmin, vmax=vmax), cmap="RdBu_r")
            # mark where C_emu dominates (ratio>1)
            dom = ratio > 1.0
            zz, kk = np.where(dom)
            if zz.size:
                ax.plot(k[kk], np.asarray(z_repr)[zz], "k.", ms=2, alpha=0.5)
            if k_max is not None:
                ax.axvline(k_max, color="purple", ls=":", lw=1.4)
            ax.set_xscale("log")
            ax.set_xlabel("k [s/km]")
            if jf == 0:
                ax.set_ylabel(f"z\n(τ₀-band {ib}, α≈{alpha_bands[ib]:.2f})")
            frac_dom = float(np.nanmean(dom)) if np.isfinite(ratio).any() else np.nan
            ax.set_title(f"cosmic={int(frac*100)}%  (C_emu dominates {100*frac_dom:.0f}% of k,z)")
            fig.colorbar(im, ax=ax, label="diag(C_emu)/diag(C_cosmic)")
    fig.suptitle("diag(C_emu)/diag(C_cosmic) over (k,z) per τ₀-band  "
                 "[ILLUSTRATIVE — production DESI DR1 + KODIAQ cov NOT wired; see LYA-CONSULT]")
    p = Path(figdir) / "cemu_over_cdata_heatmap.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


# ----------------------------------------------------------------------------
# 4. Bias-vs-variance in the error vector (mean/RMS per cell).
# ----------------------------------------------------------------------------
def bias_vs_variance(recs, z_band_edges, n_cls, K, Zb):
    """Per (class,k,z-band): mean(r_frac) and RMS(r_frac) over held-out rows in the cell.

    mean/RMS = bias-to-noise. |mean/RMS|→1 ⇒ the residual is BIAS-dominated (a coherent
    emulator offset), so the symmetric-Gaussian C_emu (zero-mean by construction) needs a
    caveat there. Returns mean (4,K,Zb), rms (4,K,Zb), ratio (4,K,Zb), n (Zb,).
    """
    # accumulate per (zb) the stacked r_frac (rows in that band, in-range bins → NaN else).
    by_zb = {zb: [] for zb in range(Zb)}
    for rec in recs:
        rf = np.where(rec["valid"][None, :], rec["r_frac"], np.nan)   # (4,K) masked
        by_zb[rec["zb"]].append(rf)
    mean = np.full((n_cls, K, Zb), np.nan)
    rms = np.full((n_cls, K, Zb), np.nan)
    nrows = np.zeros(Zb, int)
    for zb in range(Zb):
        if not by_zb[zb]:
            continue
        stack = np.stack(by_zb[zb])                                   # (nb,4,K)
        nrows[zb] = stack.shape[0]
        with np.errstate(invalid="ignore"):
            mean[:, :, zb] = np.nanmean(stack, axis=0)
            rms[:, :, zb] = np.sqrt(np.nanmean(stack ** 2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(mean) / rms
    return mean, rms, ratio, nrows


def fig_bias_vs_variance(ratio, kgrid, z_band_edges, k_max, figdir):
    """|mean|/RMS heatmap over (k,z-band) per class. >1 ⇒ bias-dominated cell."""
    from matplotlib.colors import LogNorm
    plt = _plt()
    n_cls, K, Zb = ratio.shape
    k = np.asarray(kgrid)
    fig, axes = plt.subplots(1, n_cls, figsize=(4.6 * n_cls, 4.4), squeeze=False)
    axes = axes[0]
    for ci in range(n_cls):
        ax = axes[ci]
        R = ratio[ci].T                                              # (Zb,K)
        zc = np.arange(Zb)
        im = ax.pcolormesh(k, zc, R, shading="nearest",
                           norm=LogNorm(vmin=1e-2, vmax=1e1), cmap="RdBu_r")
        dom = R > 1.0
        zz, kk = np.where(dom)
        if zz.size:
            ax.plot(k[kk], zc[zz], "k.", ms=3, alpha=0.6)
        if k_max is not None:
            ax.axvline(k_max, color="purple", ls=":", lw=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km]")
        ax.set_yticks(zc)
        ax.set_yticklabels([f"zb{zb}\n[{z_band_edges[zb] if np.isfinite(z_band_edges[zb]) else '−∞'},"
                            f"{z_band_edges[zb+1] if np.isfinite(z_band_edges[zb+1]) else '∞'})"
                            for zb in range(Zb)], fontsize=7)
        fdom = float(np.nanmean(dom))
        ax.set_title(f"{COARSE_NAMES[ci]}  (bias-dom {100*fdom:.0f}% of k,z)")
        fig.colorbar(im, ax=ax, label="|mean|/RMS")
    fig.suptitle("Bias-vs-variance of the error vector: |mean(r_frac)| / RMS(r_frac)  "
                 "(>1 ⇒ bias-dominated ⇒ symmetric-Gaussian C_emu caveat)")
    p = Path(figdir) / "cemu_bias_vs_variance.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


# ----------------------------------------------------------------------------
# 5. LF-Nyquist k_max.
# ----------------------------------------------------------------------------
def lf_nyquist_kmax(cache_path, kgrid):
    """k_Nyq = π/dv (kfkms ANGULAR). Report #in-range stored bins above it + proposed k_max.

    The stored LF grid is the FIRST n_k FFT bins (PRIYA convention), so it may sit well
    below Nyquist; we report both the Nyquist and the actual stored k_max, and propose the
    k_max to add to DATA_RANGE = min(k_Nyq, stored max) — the largest k the error vector
    can legitimately cover.
    """
    with h5py.File(cache_path, "r") as h:
        dv = h["dv_kms"][:]
        nb = h["nbins_native"][:]
    k_nyq = np.pi / dv                                # per-row angular Nyquist
    k_nyq_min = float(k_nyq.min())                    # the TIGHTEST (conservative) cut
    k = np.asarray(kgrid)
    in_range = np.isfinite(k) & (k >= DATA_RANGE["k_min"])
    n_above = int((in_range & (k > k_nyq_min)).sum())
    stored_max = float(np.nanmax(k))
    # propose the tighter of the Nyquist and the stored top bin (the error vector cannot
    # cover beyond the grid it was built on); round DOWN to the nearest stored bin ≤ cut.
    k_cut = min(k_nyq_min, stored_max)
    below = k[in_range & (k <= k_cut)]
    k_max_proposed = float(below.max()) if below.size else stored_max
    n_removed = int((in_range & (k > k_max_proposed)).sum())
    return dict(k_nyq_min=k_nyq_min, k_nyq_med=float(np.median(k_nyq)),
                stored_max=stored_max, n_in_range=int(in_range.sum()),
                n_above_nyq=n_above, k_max_proposed=k_max_proposed,
                n_removed=n_removed, nbins_native_min=int(nb.min()),
                nbins_native_med=int(np.median(nb)))


# ----------------------------------------------------------------------------
# --xclass: the CROSS-CLASS re-validation (the proof the upgrade worked).
# ----------------------------------------------------------------------------
def _xclass_revalidate(args):
    """Pool ALL 8 folds' held-out residuals; whiten by the DIAGONAL and the CROSS-CLASS
    C_emu; report var / χ²/dof / the per-k ramp for both + the before/after Q-Q.

    The diagonal whitening here is the HONEST all-folds baseline (the original
    diag_cemu_validation pooled fold-0 ONLY); the cross-class is the upgrade. Target: the
    cross-class var → ~1 (the +1.18 cross term + the fold-amplitude variance, both captured
    by the ρ pooled over all folds)."""
    print("\n=== CROSS-CLASS C_emu RE-VALIDATION (pool ALL 8 folds' held-out residuals) ===")
    evx = np.load(args.xclass, allow_pickle=True)
    rho = evx["rho"]                                          # (4,4,K,Zb,Tb)
    sigma = evx["sigma"]                                     # (4,K,Zb,Tb) carried diagonal
    n_cls, n_cls2, K, Zb, Tb = rho.shape
    alpha_centres = jnp.asarray(evx["tau0_band_centres"])
    dla_shot_flag_k = np.asarray(evx["dla_shot_flag"])
    z_band_edges = evx["z_band_edges"]
    kgrid = np.asarray(evx["kfkms"])
    print(f"  loaded {args.xclass}: rho {rho.shape}, diag-match median "
          f"{float(evx['diag_match_median']):.3f}, n_pool {int(evx['n_pool'])}")

    # load each fold's model + norm (LOSO matched: fold f's held-out sims use fold f's model).
    models, pf_by_fold = [], []
    for f in range(args.n_folds):
        model, meta, norm = T.load_checkpoint(f"{args.ckpt_prefix}{f}")
        models.append(model)
        pf_by_fold.append({k: jnp.asarray(norm["P_filt"][k])
                           for k in ("mu_marg", "sig_marg", "sig_cosmo")})
    cache_path = meta["cache_path"]
    d = load_cache(cache_path)
    nyq = lf_nyquist_kmax(cache_path, kgrid)
    k_max = nyq["k_max_proposed"]

    diag, xcl = revalidate_xclass(
        d, pf_by_fold, models, sigma, rho, z_band_edges, alpha_centres,
        dla_shot_flag_k, n_folds=args.n_folds, max_rows_per_fold=args.max_rows_per_fold)
    (pd_, wd, ck_d, nk_d) = diag
    (px, wx, ck_x, nk_x) = xcl

    def _report(tag, pooled):
        print(f"\n  [{tag}]  N={pooled['n']}")
        print(f"    mean    = {pooled['mean']:+.4f}  (target 0; z-p={pooled['mean_p']:.2g})")
        print(f"    var     = {pooled['var']:.4f}   (target 1; var-χ² p={pooled['var_chi2_p']:.2g})")
        print(f"    χ²/dof  = {pooled['chi2_over_dof']:.4f}   (target 1)")
        print(f"    KS      = {pooled['ks_stat']:.4f}  (p={pooled['ks_p']:.2g})")
    _report("BEFORE: diagonal C_emu (all-folds)", pd_)
    _report("AFTER : cross-class C_emu (all-folds)", px)

    vd, vx = pd_["var"], px["var"]
    print(f"\n  var: {vd:.3f} (diagonal) -> {vx:.3f} (cross-class)   "
          f"[reduction ×{vd / vx:.2f}]")
    if vx <= 1.15:
        print(f"  → cross-class C_emu RECOVERS var≈1 (residual cemu_inflate ≈ {vx:.3f}, ~unity).")
    else:
        print(f"  → cross-class fixes the bulk; residual var={vx:.3f} ⇒ a small "
              f"cemu_inflate ≈ {vx:.3f} (σ-scale √ = {np.sqrt(vx):.3f}) still recommended "
              f"(the fold-amplitude term not fully absorbed).")

    # per-k ramp extremes for both
    for tag, ck, nk in [("diagonal", ck_d, nk_d), ("cross-class", ck_x, nk_x)]:
        ok = np.isfinite(ck) & (nk > 0)
        worst = np.argsort(np.where(ok, np.abs(np.log(np.maximum(ck, 1e-9))), -np.inf))[::-1][:4]
        print(f"  per-k χ²/dof extremes [{tag}]: " +
              "  ".join(f"k={kgrid[j]:.3f}:{ck[j]:.2f}" for j in worst))

    fp = fig_whitening_before_after(diag, xcl, kgrid, k_max, args.figdir)
    print(f"\n  figure -> {fp}")
    print("\nDONE (--xclass).")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--error-vector", default=ERROR_VECTOR)
    ap.add_argument("--figdir", default=FIGDIR)
    ap.add_argument("--max-rows", type=int, default=600,
                    help="cap on held-out rows for the whitening pool (even-strided)")
    ap.add_argument("--theta-repr", type=float, default=0.5,
                    help="unit-cube θ for the representative-interior C_emu/C_data grid")
    ap.add_argument("--xclass", default=None,
                    help="path to error_vector_xclass.npz → run the CROSS-CLASS re-validation "
                         "(pool ALL 8 folds' held-out residuals, whiten by both diagonal and "
                         "cross-class C_emu, emit the before/after Q-Q) and exit")
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--ckpt-prefix", default=f"{REPO}/checkpoints/final_fold",
                    help="per-fold checkpoint prefix (+{f}) for the --xclass all-folds pool")
    ap.add_argument("--max-rows-per-fold", type=int, default=400)
    args = ap.parse_args()
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())

    if args.xclass is not None:
        _xclass_revalidate(args)
        return

    # --- load production pair ---
    model, meta, norm = T.load_checkpoint(args.ckpt)
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    ev = np.load(args.error_vector, allow_pickle=True)
    sigma = ev["sigma"]                              # (4,K,Zb,Tb)
    n_cls, K, Zb, Tb = sigma.shape
    alpha_centres = jnp.asarray(ev["tau0_band_centres"])
    dla_shot_flag_k = np.asarray(ev["dla_shot_flag"])
    z_band_edges = ev["z_band_edges"]
    kgrid = np.asarray(ev["kfkms"])
    cache_path = meta["cache_path"]
    print(f"loaded {args.ckpt}; error_vector sigma {sigma.shape}; "
          f"τ₀-centres {np.array2string(np.asarray(alpha_centres), precision=3)}; "
          f"z-edges {z_band_edges}; n_dla_shot={int(dla_shot_flag_k.sum())}")

    d = load_cache(cache_path)
    train_idx, val_idx, holdout_idx = make_splits(d, 0)
    print(f"fold-0 split: train={len(train_idx)} val(held-out)={len(val_idx)} "
          f"holdout(τ₀-edge)={len(holdout_idx)}")

    # --- (5) LF-Nyquist k_max FIRST (cheap; needed by all figures) ---
    nyq = lf_nyquist_kmax(cache_path, kgrid)
    k_max = nyq["k_max_proposed"]
    print("\n=== (5) LF-Nyquist k_max ===")
    print(f"  k_Nyq = π/dv  (kfkms ANGULAR): min={nyq['k_nyq_min']:.4f} "
          f"med={nyq['k_nyq_med']:.4f} s/km   (nbins_native min={nyq['nbins_native_min']} "
          f"med={nyq['nbins_native_med']}, stored n_k={K})")
    print(f"  stored k grid max = {nyq['stored_max']:.4f} s/km; "
          f"in-range k-bins (k≥{DATA_RANGE['k_min']:.0e}) = {nyq['n_in_range']}")
    print(f"  in-range bins ABOVE k_Nyq: {nyq['n_above_nyq']}  → "
          f"PROPOSED DATA_RANGE['k_max'] = {k_max:.5f} s/km "
          f"(removes {nyq['n_removed']} in-range bins)")

    # --- (1)+(2) held-out residual extraction + whitening ---
    print("\n=== extracting held-out residuals (P_filt + assembled P_obs) ===")
    recs = extract_holdout_residuals(
        d, model, pf, val_idx, sigma, z_band_edges, alpha_centres,
        dla_shot_flag_k, max_rows=args.max_rows)
    n_valid_bins = int(np.sum([rec["valid"].sum() for rec in recs]))
    print(f"  {len(recs)} held-out rows; {n_valid_bins} valid (in-range) k-bins pooled")

    print("\n=== (2) WHITENING TEST (the key check) — C_emu ALONE (cosmic_cov=0) ===")
    pooled, whit_pool, chi2_k, n_k, Wk = run_whitening(recs)
    print(f"  pooled N = {pooled['n']} whitened components")
    print(f"  mean          = {pooled['mean']:+.4f}   (target 0; z-p={pooled['mean_p']:.2g})")
    print(f"  var           = {pooled['var']:.4f}    (target 1; var-χ² p={pooled['var_chi2_p']:.2g})")
    print(f"  χ²/dof        = {pooled['chi2_over_dof']:.4f}    (target 1)")
    print(f"  KS vs N(0,1)  = {pooled['ks_stat']:.4f}  (p={pooled['ks_p']:.2g})")
    var = pooled["var"]
    if var > 1.0:
        print(f"  → C_emu UNDER-sizes the residual: suggested cemu_inflate ≈ var = {var:.3f} "
              f"(σ-scale √var = {np.sqrt(var):.3f})")
    else:
        print(f"  → C_emu is CONSERVATIVE (var<1): the budget already covers the residual.")
    fp_qq = fig_whitening(pooled, whit_pool, chi2_k, n_k, kgrid, k_max, args.figdir)

    # per-k χ²/dof localization summary
    ok = np.isfinite(chi2_k) & (n_k > 0)
    worst = np.argsort(np.where(ok, np.abs(np.log(np.maximum(chi2_k, 1e-9))), -np.inf))[::-1][:5]
    print("  per-k χ²/dof extremes (k, χ²/dof):")
    for j in worst:
        print(f"    k={kgrid[j]:.4f}  χ²/dof={chi2_k[j]:.3f}  (n_rows={int(n_k[j])})")

    # --- (3) diag(C_emu)/diag(C_cosmic) heatmap ---
    print("\n=== (3) diag(C_emu)/diag(C_cosmic) heatmap (k,z) per τ₀-band ===")
    # representative z grid spanning the data range; τ₀-bands = the error-vector centres.
    z_repr = np.linspace(2.4, 4.4, 9)
    alpha_bands = np.asarray(alpha_centres)
    cemu, Pobs = cemu_diag_grid(d, model, pf, sigma, z_band_edges, alpha_centres,
                                dla_shot_flag_k, z_repr, alpha_bands,
                                theta_unit_repr=args.theta_repr)
    for frac in (0.02, 0.05):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cemu / (frac * Pobs) ** 2
        ratio = np.where(np.isfinite(ratio) & (Pobs > 0), ratio, np.nan)
        med = float(np.nanmedian(ratio))
        frac_dom = float(np.nanmean(ratio > 1.0))
        print(f"  cosmic={int(frac*100)}%: median ratio={med:.3g}; "
              f"C_emu dominates {100*frac_dom:.1f}% of (k,z,τ₀) cells")
    fp_heat = fig_cemu_over_cdata(cemu, Pobs, kgrid, z_repr, alpha_bands, k_max, args.figdir)

    # --- (4) bias-vs-variance ---
    print("\n=== (4) BIAS-vs-VARIANCE (|mean|/RMS per class,k,z-band) ===")
    mean, rms, ratio_bv, nrows = bias_vs_variance(recs, z_band_edges, n_cls, K, Zb)
    print(f"  rows per z-band: {dict(enumerate(nrows.tolist()))}")
    for ci in range(n_cls):
        with np.errstate(invalid="ignore"):
            r = ratio_bv[ci]
            fdom = float(np.nanmean(r > 1.0))
            med = float(np.nanmedian(r))
        print(f"  {COARSE_NAMES[ci]}: median |mean|/RMS = {med:.3f}; "
              f"bias-dominated (|mean|>RMS) in {100*fdom:.1f}% of (k,z) cells")
    # list the bias-dominated cells (class,k,z-band) compactly
    bias_cells = []
    for ci in range(n_cls):
        for ki in range(K):
            for zb in range(Zb):
                rr = ratio_bv[ci, ki, zb]
                if np.isfinite(rr) and rr > 1.0:
                    bias_cells.append((COARSE_NAMES[ci], float(kgrid[ki]), zb, float(rr)))
    print(f"  TOTAL bias-dominated cells: {len(bias_cells)}")
    if bias_cells:
        bias_cells.sort(key=lambda t: -t[3])
        print("  top-10 by |mean|/RMS:")
        for c, kk, zb, rr in bias_cells[:10]:
            print(f"    {c:7s} k={kk:.4f} zb{zb}  |mean|/RMS={rr:.2f}")
    fp_bv = fig_bias_vs_variance(ratio_bv, kgrid, z_band_edges, k_max, args.figdir)

    print("\n=== FIGURES ===")
    for p in (fp_qq, fp_heat, fp_bv):
        print("  ", p)
    print("\nDONE.")


if __name__ == "__main__":
    main()
