#!/usr/bin/env python3
"""Build + validate the MULTI-FIDELITY (LF->HF) correction layer.

Forward model (see hcd_analysis/emulator/multifidelity.py):

    P_MF(theta, z, k) = ( rho(k) * f_LF(theta, z, k) + delta(theta, z, k) ) * res_corr(z, k)
                      = ( f_LF * exp(g(theta, z, k)) ) * res_corr           [rho == 1]

with f_LF the FROZEN finalized LF backbone (log-log extrapolated up to the KODIAQ
k_max), g the learned log-ratio DeltaHead trained on the 6 HF sims, and res_corr the
fixed L15n512/L15n384 particle-convergence factor.

This script:
  1. Loads the LF + HR caches and a frozen LF backbone fold; measures the delta
     targets g = logP_HF - logP_hat_LF on the matched (theta, z, tau0) HR rows.
  2. HF-LOSO: holds out one of the 6 HF sims at a time, fits the DeltaHead on the
     other 5, and evaluates the HELD-OUT HF sim, comparing the MF prediction error
     vs k (per class, esp. the KODIAQ high-k band 0.07-0.2) to two baselines:
       (a) LF-extrapolated-alone (g == 0, no delta);
       (b) HF-standalone (a 6-sim emulator surrogate: the SAME small head fit to
           ABSOLUTE logP_HF from the 5 training HF sims, no LF backbone) -- the
           hopeless few-sim baseline MF is meant to beat.
  3. A_p & n_s Fisher-bias of the MF prediction in the data range INCLUDING the
     KODIAQ high-k band, per held-out HF sim.
  4. res_corr sanity: P_MF with vs without res_corr (the ~12% high-k suppression).
  5. Figures: MF vs LF-alone vs HF-standalone error vs k (held-out HF); res_corr
     effect.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/build_mf_delta.py [args]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import hcd_analysis.emulator  # enables x64
from hcd_analysis.emulator.data import load_cache, Z_LIMITS
from hcd_analysis.emulator import multifidelity as MF

CLS = MF.COARSE_NAMES                              # ("clean","LLS","subDLA","DLA")
# DESI-DR1-like diagonal per-mode fractional covariance (matches diag_ap_fisher_bias).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
KODIAQ_BAND = MF.KODIAQ_BAND                       # (0.07, 0.2)
# the LF backbone's per-row Nyquist sits ~0.06-0.098; above this f_LF is a tail
# extrapolation, so this is the "LF cannot reach" band.
LF_NYQUIST = 0.069


# ---------------------------------------------------------------------------- #
# HF-standalone baseline: absolute logP_HF head (the hopeless 6-sim emulator)
# ---------------------------------------------------------------------------- #
def measure_hf_abs_targets(hr, lf, eval_logk, pairs):
    """Absolute logP_HF (M,4,K) on the eval grid + the matched conditioning.

    Mirrors ``measure_delta_targets`` but the target is the ABSOLUTE HR log P1D
    (no LF backbone), for the HF-standalone baseline."""
    x = np.asarray([lf["x"][l] for _, l in pairs])
    tau0 = np.asarray([lf["tau0"][l] for _, l in pairs])
    hr_row = np.asarray([h for h, _ in pairs])
    logP = np.asarray([MF.hr_logP_on_eval_grid(hr, h, eval_logk) for h, _ in pairs])
    # standardize the absolute target per-(class,k) so the same small head/regular-
    # ization scale applies as for the log-ratio (mean/std over all rows, nan-aware).
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(logP.reshape(-1, 4, logP.shape[-1]), axis=0)   # (4,K)
        sd = np.nanstd(logP.reshape(-1, 4, logP.shape[-1]), axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return dict(x=x, tau0=tau0, hr_row=hr_row, g=(logP - mu[None]) / sd[None],
                logP=logP, mu=mu, sd=sd)


def hf_standalone_predict(hr_targets, basis, eval_logk, train_rows, eval_rows,
                          *, n_basis, width, n_layers, lr, epochs, coeff_l2, seed):
    """Fit the small head to ABSOLUTE (standardized) logP_HF on ``train_rows``;
    return predicted ABSOLUTE logP_HF (len(eval_rows),4,K).  The few-sim baseline."""
    head, _ = MF.train_delta_head(
        hr_targets, basis, np.zeros(basis.shape[1]), eval_logk,
        train_mask_rows=train_rows, n_basis=n_basis, width=width, n_layers=n_layers,
        lr=lr, epochs=epochs, coeff_l2=coeff_l2, mean_prior_w=0.0, seed=seed)
    cond = jnp.asarray(np.concatenate(
        [hr_targets["x"][eval_rows], hr_targets["tau0"][eval_rows][:, None]], axis=1))
    gstd = np.asarray(jax.vmap(lambda c: head(c, jnp.asarray(basis)))(cond))  # (B,4,K) std
    return gstd * hr_targets["sd"][None] + hr_targets["mu"][None]             # de-standardize


# ---------------------------------------------------------------------------- #
# Error metrics
# ---------------------------------------------------------------------------- #
def frac_err_vs_k(logP_pred, logP_true):
    """RMS fractional P1D error per (class,k) over rows: sqrt(<(P_pred/P_true-1)^2>).

    ``logP_pred``/``logP_true`` (R,4,K) natural-log.  NaN-safe over rows."""
    with np.errstate(over="ignore", invalid="ignore"):
        frac = np.exp(logP_pred - logP_true) - 1.0          # (R,4,K)
    frac = np.where(np.isfinite(logP_true) & np.isfinite(logP_pred), frac, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(frac ** 2, axis=0))       # (4,K)


def band_rms(err_ck, k_eval, lo, hi):
    """RMS of per-(class,k) error over the k-band [lo,hi], per class -> (4,)."""
    sel = (k_eval >= lo) & (k_eval <= hi)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(err_ck[:, sel] ** 2, axis=1))


# ---------------------------------------------------------------------------- #
# Fisher A_p / n_s bias of the MF prediction (data range incl. KODIAQ high-k)
# ---------------------------------------------------------------------------- #
def mf_fisher_bias(mf, hr_targets_full, eval_rows, k_eval, *, z_fid=3.0):
    """delta_theta = (J^T C^-1 J)^-1 J^T C^-1 d in sigma_Fisher units over 9 params.

    J = d logP_MF / d theta (9 params) at the fiducial unit-cube point + z_fid (via
    jacfwd on the differentiable MF forward), evaluated on the eval grid INCLUDING
    the KODIAQ high-k band.  d = the MF coherent log-error (mean over the held-out HF
    rows at z_fid of logP_MF - logP_HF).  C = the DESI-DR1-like diagonal per-mode
    fractional covariance.  Returns the per-param bias/sigma + the A_p, n_s entries.
    """
    # fiducial: unit-cube centre, z_fid, median tau0 at z_fid over the eval rows.
    z_unit_fid = float((z_fid - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0]))
    zsel = np.isclose(hr_targets_full["x"][eval_rows, 9], z_unit_fid, atol=0.02)
    if not zsel.any():
        zsel = np.ones(len(eval_rows), bool)
    tau0_fid = float(np.median(hr_targets_full["tau0"][eval_rows][zsel]))
    fiducial = np.full(9, 0.5)

    z_j = jnp.asarray(z_unit_fid)

    def logP_of_theta(theta9):
        x = jnp.concatenate([theta9, z_j[None]])
        return mf.logP_mf(x, jnp.asarray(tau0_fid))         # (4,K)

    p0 = jnp.asarray(fiducial, dtype=jnp.float64)
    J = np.asarray(jax.jacfwd(logP_of_theta)(p0))           # (4,K,9)

    # coherent MF log-error at the fiducial z-slice (mean over held-out HF rows).
    rows = np.asarray(eval_rows)[zsel]
    logP_hr = hr_targets_full["logP"][rows]                 # (n,4,K)
    xs = jnp.asarray(hr_targets_full["x"][rows]); ts = jnp.asarray(hr_targets_full["tau0"][rows])
    logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, ts))      # (n,4,K)
    m = np.isfinite(logP_hr) & np.isfinite(logP_mf)
    with np.errstate(invalid="ignore"):
        d = np.array([[np.nanmean(np.where(m[:, ci, j], (logP_mf - logP_hr)[:, ci, j], np.nan))
                       for j in range(J.shape[1])] for ci in range(4)])    # (4,K)

    # data range: z in [2.2,4.6] (z_fid=3 ok), k >= data k_min, k <= KODIAQ k_max.
    kmask = (k_eval >= MF.DATA_RANGE["k_min"]) & (k_eval <= MF.KODIAQ_KMAX)
    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(J.shape[1]):
            if not kmask[j] or not np.all(np.isfinite(J[ci, j])) or not np.isfinite(d[ci, j]):
                continue
            rows_J.append(J[ci, j]); rows_d.append(d[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    Jm = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    F = (Jm.T * Cinv) @ Jm
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((Jm.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    return dict(
        n_modes=int(len(dv)), z_fid=z_fid, tau0_fid=tau0_fid,
        fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
        bias_in_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)},
    )


# ---------------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_error_vs_k(per_sim, k_eval, figdir):
    """MF vs LF-alone vs HF-standalone RMS frac. error vs k, per class, per held-out
    HF sim (one row of panels per sim, 4 class columns)."""
    plt = _plt()
    sims = list(per_sim)
    fig, axes = plt.subplots(len(sims), 4, figsize=(18, 3.0 * len(sims)),
                             sharex=True, squeeze=False)
    for si, s in enumerate(sims):
        r = per_sim[s]
        for ci, c in enumerate(CLS):
            ax = axes[si][ci]
            ax.loglog(k_eval, r["mf"][ci], "-", color="C0", lw=1.8, label="MF")
            ax.loglog(k_eval, r["lf"][ci], "--", color="C3", lw=1.4, label="LF-alone")
            ax.loglog(k_eval, r["hf"][ci], ":", color="C2", lw=1.4, label="HF-standalone")
            ax.axvspan(KODIAQ_BAND[0], KODIAQ_BAND[1], color="grey", alpha=0.12)
            ax.axvline(LF_NYQUIST, color="k", ls="-.", lw=0.7, alpha=0.6)
            ax.grid(alpha=0.3, which="both")
            if si == 0:
                ax.set_title(c)
            if ci == 0:
                ax.set_ylabel(f"{s[:10]}..\nRMS frac err")
            if si == len(sims) - 1:
                ax.set_xlabel("k [s/km]")
            if si == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.suptitle("HF-LOSO: MF vs LF-extrapolated-alone vs HF-standalone — RMS "
                 "fractional P1D error vs k (held-out HF sim)\n"
                 "shaded = KODIAQ band 0.07–0.2 s/km; dash-dot = LF Nyquist ~0.069")
    p = Path(figdir) / "mf_error_vs_k_hf_loso.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


def fig_res_corr_effect(mf, hr_targets_full, eval_rows, k_eval, figdir, z_fid=3.0):
    """P_MF with vs without res_corr at z_fid, per class — the ~12% high-k drop."""
    plt = _plt()
    z_unit_fid = float((z_fid - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0]))
    zsel = np.isclose(hr_targets_full["x"][eval_rows, 9], z_unit_fid, atol=0.02)
    rows = np.asarray(eval_rows)[zsel] if zsel.any() else np.asarray(eval_rows)[:1]
    xs = jnp.asarray(hr_targets_full["x"][rows]); ts = jnp.asarray(hr_targets_full["tau0"][rows])
    P_with = np.asarray(jax.vmap(lambda x, t: mf.P_mf(x, t, apply_res_corr=True))(xs, ts))
    P_no = np.asarray(jax.vmap(lambda x, t: mf.P_mf(x, t, apply_res_corr=False))(xs, ts))
    ratio = np.nanmean(P_with / P_no, axis=0)               # (4,K) == res_corr (class-indep)
    rc = np.asarray(mf.res_corr(jnp.asarray(z_fid)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for ci, c in enumerate(CLS):
        ax[0].semilogx(k_eval, ratio[ci], "-", label=c)
    ax[0].semilogx(k_eval, rc, "k--", lw=1.5, label="res_corr table")
    ax[0].axvspan(KODIAQ_BAND[0], KODIAQ_BAND[1], color="grey", alpha=0.12)
    ax[0].axhline(1.0, color="grey", lw=0.6)
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("P_MF(with) / P_MF(without)")
    ax[0].set_title(f"res_corr multiplicative effect @ z={z_fid}")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    # absolute spectra (clean) with/without
    ax[1].loglog(k_eval, np.nanmean(P_no[:, 0], 0), "C3--", label="P_MF no res_corr")
    ax[1].loglog(k_eval, np.nanmean(P_with[:, 0], 0), "C0-", label="P_MF with res_corr")
    ax[1].axvspan(KODIAQ_BAND[0], KODIAQ_BAND[1], color="grey", alpha=0.12)
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("P_clean [s/km]")
    ax[1].set_title("clean-class P_MF, res_corr on/off")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("res_corr sanity: the fixed L15n512/L15n384 high-k suppression")
    p = Path(figdir) / "mf_res_corr_effect.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    rc_band = rc[(k_eval >= KODIAQ_BAND[0]) & (k_eval <= KODIAQ_BAND[1])]
    return str(p), float(np.nanmin(rc_band)), float(np.nanmean(rc_band))


# ---------------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lf-fold", type=int, default=0,
                    help="which finalized LF backbone fold to freeze as f_LF")
    ap.add_argument("--n-k", type=int, default=48, help="eval-grid k-bins")
    ap.add_argument("--k-max", type=float, default=MF.KODIAQ_KMAX)
    ap.add_argument("--n-basis", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--coeff-l2", type=float, default=1e-1)
    ap.add_argument("--mean-prior-w", type=float, default=1e-1)
    ap.add_argument("--no-rho", dest="use_rho", action="store_false",
                    help="set rho==1 (delta carries the FULL correction). DEFAULT is "
                         "rho = per-k mean log-ratio (delta learns only the theta/z "
                         "departure) — this cuts the n_s Fisher bias from ~14sigma to "
                         "~2sigma in the few-6-HF-sim regime (see report).")
    ap.set_defaults(use_rho=True)
    ap.add_argument("--figdir", default="figures/analysis/05_multifidelity")
    ap.add_argument("--out", default="figures/analysis/05_multifidelity/mf_delta_results.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 held-out sims, fewer epochs — quick end-to-end check")
    args = ap.parse_args()

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    t0 = time.time()

    lf = load_cache(MF.LF_CACHE)
    hr = load_cache(MF.HR_CACHE)
    lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=args.lf_fold)
    pairs = MF.match_hr_to_lf(lf, hr)
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=args.k_max, n_k=args.n_k)
    print(f"matched {len(pairs)} HR<->LF rows; eval grid {k_eval[0]:.4f}..{k_eval[-1]:.4f} "
          f"s/km ({args.n_k} bins); LF fold {args.lf_fold} ({time.time()-t0:.1f}s)")

    # measure delta (log-ratio) + absolute-HF targets on the matched rows.
    tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk, pairs,
                                  n_tail=6)
    hf_abs = measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    if args.smoke:
        sims = sims[:2]
        args.epochs = min(args.epochs, 200)
    print(f"6 HF sims: {[s[:12] for s in sorted(set(sim_of_row))]}")

    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=args.n_basis))
    # rho: optional per-k mean log-ratio over ALL rows (PRIYA-style AR1 amplitude).
    log_rho_global = (MF.mean_log_ratio_rho(tg, eval_logk) if args.use_rho
                      else np.zeros(len(eval_logk)))
    rho_note = ("per-k mean log-ratio (delta learns theta/z departure)" if args.use_rho
                else "rho == 1 (delta carries the full correction)")
    print(f"rho: {rho_note}")

    per_sim = {}
    fisher_per_sim = {}
    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]
        eval_rows = np.where(held)[0]
        ts = time.time()

        # --- fit the delta head on the 5 training HF sims (HF-LOSO) ----------
        head, hist = MF.train_delta_head(
            tg, basis, log_rho_global, eval_logk, train_mask_rows=train_rows,
            n_basis=args.n_basis, width=args.width, n_layers=args.n_layers,
            lr=args.lr, epochs=args.epochs, coeff_l2=args.coeff_l2,
            mean_prior_w=args.mean_prior_w, seed=0)
        mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, head,
                                    eval_logk=eval_logk, log_rho=log_rho_global,
                                    n_basis=args.n_basis)

        # --- predictions on the held-out HF sim ------------------------------
        xs = jnp.asarray(tg["x"][eval_rows]); tts = jnp.asarray(tg["tau0"][eval_rows])
        logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, tts))            # (n,4,K)
        # LF-alone (no delta, WITH res_corr — the fair "extrapolated LF" baseline):
        logP_lf = np.asarray(jax.vmap(lambda x, t: mf.lf_logP(x, t)
                                      + jnp.log(mf.res_corr(
                                          x[..., 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]))[None, :]
                                      )(xs, tts))
        # HF-standalone (6-sim emulator surrogate fit to ABSOLUTE logP_HF):
        logP_hf = hf_standalone_predict(
            hf_abs, basis, eval_logk, train_rows, eval_rows,
            n_basis=args.n_basis, width=args.width, n_layers=args.n_layers,
            lr=args.lr, epochs=args.epochs, coeff_l2=args.coeff_l2, seed=0)
        logP_true = hf_abs["logP"][eval_rows]                          # (n,4,K)

        err_mf = frac_err_vs_k(logP_mf, logP_true)
        err_lf = frac_err_vs_k(logP_lf, logP_true)
        err_hf = frac_err_vs_k(logP_hf, logP_true)
        per_sim[s] = {"mf": err_mf, "lf": err_lf, "hf": err_hf}

        # --- Fisher bias (data range incl. KODIAQ high-k) -------------------
        FB = mf_fisher_bias(mf, hf_abs, eval_rows, k_eval, z_fid=3.0)
        fisher_per_sim[s] = FB

        # band summaries
        mf_kod = band_rms(err_mf, k_eval, *KODIAQ_BAND)
        lf_kod = band_rms(err_lf, k_eval, *KODIAQ_BAND)
        hf_kod = band_rms(err_hf, k_eval, *KODIAQ_BAND)
        print(f"\n[{s[:14]}..] trained {time.time()-ts:.1f}s; KODIAQ-band RMS frac err "
              f"(clean): MF={mf_kod[0]:.3f} LF={lf_kod[0]:.3f} HF={hf_kod[0]:.3f}")
        print(f"   A_p bias={FB['bias_in_sigma']['Ap']:+.2f}sigma  "
              f"ns bias={FB['bias_in_sigma']['ns']:+.2f}sigma  "
              f"({FB['n_modes']} modes, cond={FB['fisher_cond']:.1e})")

    # --- figures --------------------------------------------------------------
    fp_err = fig_error_vs_k(per_sim, k_eval, args.figdir)
    mf_final = MF.build_multifidelity(
        lf_model, lf_norm, lf_logk,
        MF.train_delta_head(tg, basis, log_rho_global, eval_logk,
                            n_basis=args.n_basis, width=args.width,
                            n_layers=args.n_layers, lr=args.lr, epochs=args.epochs,
                            coeff_l2=args.coeff_l2, mean_prior_w=args.mean_prior_w)[0],
        eval_logk=eval_logk, log_rho=log_rho_global, n_basis=args.n_basis)
    all_rows = np.arange(len(tg["hr_row"]))
    fp_rc, rc_min, rc_mean = fig_res_corr_effect(mf_final, hf_abs, all_rows, k_eval, args.figdir)

    # --- aggregate + report ---------------------------------------------------
    def agg(key):
        return np.sqrt(np.nanmean(np.stack([band_rms(per_sim[s][key], k_eval, *KODIAQ_BAND)
                                            for s in sims]) ** 2, axis=0))
    mf_kod_all, lf_kod_all, hf_kod_all = agg("mf"), agg("lf"), agg("hf")
    full = (MF.DATA_RANGE["k_min"], LF_NYQUIST)
    def agg_full(key):
        return np.sqrt(np.nanmean(np.stack([band_rms(per_sim[s][key], k_eval, *full)
                                            for s in sims]) ** 2, axis=0))
    mf_lo, lf_lo, hf_lo = agg_full("mf"), agg_full("lf"), agg_full("hf")

    print("\n========== MF HF-LOSO SUMMARY ==========")
    print(f"eval grid: {k_eval[0]:.4f}..{k_eval[-1]:.4f} s/km; KODIAQ band {KODIAQ_BAND}; "
          f"LF Nyquist ~{LF_NYQUIST}")
    print(f"rho: {rho_note}; delta head: n_basis={args.n_basis} width={args.width} "
          f"n_layers={args.n_layers} coeff_l2={args.coeff_l2} mean_prior_w={args.mean_prior_w}")
    print("RMS frac err over HF-LOSO folds, per class:")
    print(f"  {'band':18} {'class':8} {'MF':>8} {'LF-alone':>10} {'HF-stand':>10}")
    for ci, c in enumerate(CLS):
        print(f"  {'LF-band(<Nyq)':18} {c:8} {mf_lo[ci]:8.3f} {lf_lo[ci]:10.3f} {hf_lo[ci]:10.3f}")
    for ci, c in enumerate(CLS):
        print(f"  {'KODIAQ(0.07-0.2)':18} {c:8} {mf_kod_all[ci]:8.3f} "
              f"{lf_kod_all[ci]:10.3f} {hf_kod_all[ci]:10.3f}")
    print(f"\nres_corr KODIAQ-band suppression: min={rc_min:.3f} mean={rc_mean:.3f} "
          f"(~{(1-rc_mean)*100:.0f}% suppression)")
    print("A_p / n_s Fisher bias (sigma) per held-out HF sim (data range incl. KODIAQ):")
    for s in sims:
        FB = fisher_per_sim[s]
        print(f"  {s[:14]}..  A_p={FB['bias_in_sigma']['Ap']:+.2f}  "
              f"ns={FB['bias_in_sigma']['ns']:+.2f}")

    out = {
        "config": vars(args), "rho_note": rho_note,
        "k_eval": k_eval.tolist(), "kodiaq_band": list(KODIAQ_BAND),
        "lf_nyquist": LF_NYQUIST,
        "per_sim_err": {s: {k: per_sim[s][k].tolist() for k in ("mf", "lf", "hf")}
                        for s in sims},
        "agg_kodiaq": {"mf": mf_kod_all.tolist(), "lf": lf_kod_all.tolist(),
                       "hf": hf_kod_all.tolist()},
        "agg_lfband": {"mf": mf_lo.tolist(), "lf": lf_lo.tolist(), "hf": hf_lo.tolist()},
        "res_corr_kodiaq": {"min": rc_min, "mean": rc_mean},
        "fisher_per_sim": fisher_per_sim,
        "figures": {"error_vs_k": fp_err, "res_corr": fp_rc},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out}\nfigures:\n  {fp_err}\n  {fp_rc}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
