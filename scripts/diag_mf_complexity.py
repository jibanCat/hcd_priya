#!/usr/bin/env python3
"""Diagnose the OPTIMAL COMPLEXITY of the multi-fidelity (LF->HF) delta correction.

Tests the PI's hypothesis: the SIMPLEST smooth MF correction (weakly cosmology-
dependent, ~fixed resolution correction, PRIYA-style) GENERALIZES BETTER than the
learned 4-term-Chebyshev MLP delta-head of commit e092df1 -- and likely cures the
n_s Fisher-bias tail (worst held-out sim ns0.859: ns bias +3.84 sigma).

Two analyses (READ-ONLY on production; reuses the e092df1 MF module + the
build_mf_delta HF-LOSO/Fisher harness):

  PART 1 -- g DECOMPOSITION (no model fitting; pure ANOVA on the measured log-ratio).
    g(theta,z,k) = log P_HF - log P_hat_LF, measured on the matched HR<->LF rows.
    The 6 HR cosmologies are EXACT LF design points; at each (sim, z) there are 20
    realizations that differ ONLY in alpha (tau0 / mean-flux rescale), same cosmology.
    So at fixed (z,k):
      * gbar(z,k)   = mean over ALL sims & alpha     -- the FIXED (theta-indep) mean.
      * dg(theta)   = (per-sim mean over alpha) - gbar -- the THETA-DEPARTURE.
      * noise       = within-(sim,z) scatter over the 20 alpha -- the NOISE FLOOR
                      (tau0/mean-flux + interp/measurement scatter that g should NOT
                      attribute to cosmology).
    One-way ANOVA per (z,k): Var_total = Var_between-sim (theta) + Var_within-sim
    (noise).  We report, per k-band (esp. KODIAQ 0.07-0.2), the variance fraction
    that is the FIXED mean vs the THETA-departure vs the noise floor, AND whether the
    theta-departure RMS EXCEEDS the noise floor (significant) or sits WITHIN it
    (-> the correction is effectively theta-independent: the PI is right).

  PART 2 -- MF COMPLEXITY SWEEP (HF-LOSO, hold out 1 of 6, score the held-out KODIAQ
    P1D RMS + A_p/n_s Fisher bias).  All variants share rho = per-k mean log-ratio
    (the production default) and the FIXED res_corr; they differ ONLY in the delta head:
      (a) rho(k,z) ONLY      -- delta == 0; the simplest, NO theta-dependence.
                                (z-resolved fixed mean: gbar(z,k) - rho added back).
      (b) rho + GLOBAL LINEAR -- one scalar linear-in-theta amplitude * a fixed smooth
                                k-shape (per class): ~ (4 + 9) params; the minimal
                                smooth theta-correction.
      (c) rho + LINEAR map    -- DeltaHead with n_layers=0 (pure affine theta->coeffs,
                                4 Chebyshev coeffs/class): a low-order theta-correction.
      (d) rho + MLP           -- the e092df1 baseline (n_basis=4, width=16, n_layers=1).
    For each: held-out KODIAQ RMS frac P1D error (per class + clean) and the A_p/n_s
    Fisher bias (mean + WORST sim).  Which is the SIMPLEST that minimizes the HELD-OUT
    error/bias?  Does simpler generalize better (over-fitting hypothesis)?

Figures -> figures/analysis/05_multifidelity/:
  * mf_g_decomposition.png         -- gbar / theta-departure / noise floor vs k.
  * mf_complexity_generalization.png -- held-out KODIAQ error & |ns|/|Ap| Fisher bias
                                        vs model complexity (a)->(d).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_complexity.py [args]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import hcd_analysis.emulator  # enables x64
from hcd_analysis.emulator.data import load_cache, Z_LIMITS
from hcd_analysis.emulator import multifidelity as MF

# reuse the build_mf_delta HF-LOSO / Fisher harness verbatim (same metrics, same
# Fisher bias, same HF-standalone target measurement) so the sweep is scored
# IDENTICALLY to the e092df1 baseline.
_spec = importlib.util.spec_from_file_location(
    "build_mf_delta", str(Path(__file__).with_name("build_mf_delta.py")))
BMF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BMF)

CLS = MF.COARSE_NAMES
KODIAQ_BAND = MF.KODIAQ_BAND               # (0.07, 0.2)
LF_NYQUIST = BMF.LF_NYQUIST                # 0.069
N_CLASSES = MF.N_CLASSES


# ============================================================================ #
# PART 1 -- g decomposition (ANOVA: fixed-mean vs theta-departure vs noise)
# ============================================================================ #
def decompose_g(tg, sim_of_row, k_eval):
    """One-way ANOVA of g(theta,z,k) into fixed-mean / theta-departure / noise.

    ``tg`` from ``MF.measure_delta_targets`` (``g`` is (M,4,K), NaN where empty).
    ``sim_of_row`` (M,) the HF sim name of each row.  At each (sim, z) the 20 rows
    differ ONLY in alpha (same cosmology) -> within-(sim,z) scatter = NOISE FLOOR;
    between-sim scatter of the per-sim (per-z) means = THETA signal.

    Returns a dict with, per (class, k):
      gbar(z,k) collapsed over z to a per-k fixed mean ``gbar_k`` (4,K);
      ``rms_fixed`` (4,K)  = RMS over (z,k-fixed) of the fixed mean (the coherent tilt);
      ``rms_theta`` (4,K)  = RMS over sims&z of the per-sim-mean theta-departure;
      ``rms_noise`` (4,K)  = RMS within-(sim,z) over the 20 alpha (the floor);
      variance fractions (fixed / theta / noise) per k, and per-k-band summaries.
    """
    g = tg["g"]                                       # (M,4,K)
    z = np.round(tg["x"][:, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
    sims = sorted(set(sim_of_row))
    zvals = sorted(set(z))
    K = g.shape[-1]

    # per-(sim,z) MEAN over the 20 alpha and per-(sim,z) WITHIN-cell variance.
    cell_mean = {}   # (sim,z) -> (4,K) mean over alpha
    cell_var = {}    # (sim,z) -> (4,K) variance over alpha (the noise floor)
    cell_n = {}
    for s in sims:
        for zz in zvals:
            m = (sim_of_row == s) & (z == zz)
            if m.sum() == 0:
                continue
            gc = g[m]                                  # (n,4,K)
            with np.errstate(invalid="ignore"):
                cell_mean[(s, zz)] = np.nanmean(gc, axis=0)        # (4,K)
                # ddof=1 within-cell variance (the irreducible alpha/tau0 scatter).
                cell_var[(s, zz)] = np.nanvar(gc, axis=0, ddof=1)
            cell_n[(s, zz)] = np.isfinite(gc).sum(axis=0)          # (4,K)

    # FIXED MEAN gbar(z,k): mean of the per-sim means over sims, per z.
    gbar_z = {}     # z -> (4,K) fixed mean across the 6 sims at that z
    for zz in zvals:
        stacks = [cell_mean[(s, zz)] for s in sims if (s, zz) in cell_mean]
        if not stacks:
            continue
        with np.errstate(invalid="ignore"):
            gbar_z[zz] = np.nanmean(np.stack(stacks), axis=0)      # (4,K)

    # THETA-DEPARTURE dg(sim,z,k) = per-sim-mean - gbar(z,k).
    # collapse to per-(class,k) RMS over (sim,z): the size of the theta-departure.
    # FIXED-mean RMS over (class,k): collapse gbar over z (the coherent tilt size).
    rms_theta = np.zeros((N_CLASSES, K))
    rms_fixed = np.zeros((N_CLASSES, K))
    rms_noise = np.zeros((N_CLASSES, K))
    # accumulate sums of squares with finite counts, per (class,k).
    ss_theta = np.zeros((N_CLASSES, K)); n_theta = np.zeros((N_CLASSES, K))
    ss_fixed = np.zeros((N_CLASSES, K)); n_fixed = np.zeros((N_CLASSES, K))
    sw_noise = np.zeros((N_CLASSES, K)); n_noise = np.zeros((N_CLASSES, K))
    for zz in zvals:
        if zz not in gbar_z:
            continue
        gb = gbar_z[zz]                                # (4,K)
        fin_b = np.isfinite(gb)
        ss_fixed += np.where(fin_b, gb ** 2, 0.0); n_fixed += fin_b
        for s in sims:
            if (s, zz) not in cell_mean:
                continue
            dg = cell_mean[(s, zz)] - gb               # (4,K) theta departure
            fin_d = np.isfinite(dg)
            ss_theta += np.where(fin_d, dg ** 2, 0.0); n_theta += fin_d
            cv = cell_var[(s, zz)]                      # (4,K) within-cell variance
            fin_v = np.isfinite(cv)
            sw_noise += np.where(fin_v, cv, 0.0); n_noise += fin_v
    rms_fixed = np.sqrt(ss_fixed / np.maximum(n_fixed, 1))
    rms_theta = np.sqrt(ss_theta / np.maximum(n_theta, 1))
    rms_noise = np.sqrt(sw_noise / np.maximum(n_noise, 1))   # sqrt(mean within-var)

    # per-k variance fractions (fixed / theta / noise of the total g variance budget).
    v_fixed = rms_fixed ** 2
    v_theta = rms_theta ** 2
    v_noise = rms_noise ** 2
    v_tot = v_fixed + v_theta + v_noise
    frac_fixed = v_fixed / np.maximum(v_tot, 1e-30)
    frac_theta = v_theta / np.maximum(v_tot, 1e-30)
    frac_noise = v_noise / np.maximum(v_tot, 1e-30)

    # per-k SIGNIFICANCE of the theta-departure vs the noise floor.
    theta_over_noise = rms_theta / np.maximum(rms_noise, 1e-30)

    def band_summary(lo, hi):
        sel = (k_eval >= lo) & (k_eval <= hi)
        out = {}
        for ci, c in enumerate(CLS):
            out[c] = dict(
                rms_fixed=float(np.sqrt(np.mean(rms_fixed[ci, sel] ** 2))),
                rms_theta=float(np.sqrt(np.mean(rms_theta[ci, sel] ** 2))),
                rms_noise=float(np.sqrt(np.mean(rms_noise[ci, sel] ** 2))),
                frac_fixed=float(np.mean(frac_fixed[ci, sel])),
                frac_theta=float(np.mean(frac_theta[ci, sel])),
                frac_noise=float(np.mean(frac_noise[ci, sel])),
                theta_over_noise=float(np.mean(theta_over_noise[ci, sel])),
            )
        return out

    bands = {
        "LF_band(<Nyq)": band_summary(MF.DATA_RANGE["k_min"], LF_NYQUIST),
        "KODIAQ(0.07-0.2)": band_summary(*KODIAQ_BAND),
        "full(kmin-0.2)": band_summary(MF.DATA_RANGE["k_min"], MF.KODIAQ_KMAX),
    }
    return dict(
        rms_fixed=rms_fixed, rms_theta=rms_theta, rms_noise=rms_noise,
        frac_fixed=frac_fixed, frac_theta=frac_theta, frac_noise=frac_noise,
        theta_over_noise=theta_over_noise, gbar_z=gbar_z, bands=bands,
        sims=sims, zvals=zvals,
    )


# ============================================================================ #
# PART 2 -- delta-head variants (all conform to the DeltaHead interface so the
#           existing MultiFidelity.g / Fisher harness work unchanged)
# ============================================================================ #
class ZeroFixedMeanHead(eqx.Module):
    """delta == 0 beyond a FIXED (theta-independent) z-resolved mean gbar(z,k).

    Variant (a): rho(k,z) ONLY.  ``MultiFidelity.g`` returns ``log_rho[None,:] +
    head(cond, basis)``; we set log_rho = per-k GLOBAL mean (production default) and
    have this head return ``gbar(z,k) - log_rho`` (a fixed, theta-INDEPENDENT z-trend
    on top of the per-k mean).  NO theta-dependence at all (the simplest correction).

    ``gbar_tab`` (Nz, n_classes, K) is the per-z fixed mean MINUS log_rho; ``z_tab``
    (Nz,) the sorted physical z grid.  At eval the head linearly interpolates the
    table in z (clamped edges).  Reads z from the conditioning vector cond[9]
    (z_unit) -> physical z.  Differentiable; coeffs() returns zeros (no basis use).
    """
    gbar_tab: jax.Array        # (Nz, n_classes, K)  fixed z-trend (gbar - log_rho)
    z_tab: jax.Array           # (Nz,) physical z, ascending
    n_classes: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, gbar_tab, z_tab, n_basis):
        self.gbar_tab = jnp.asarray(gbar_tab)
        self.z_tab = jnp.asarray(z_tab)
        self.n_classes = gbar_tab.shape[1]
        self.n_basis = int(n_basis)

    def coeffs(self, cond):
        return jnp.zeros((self.n_classes, self.n_basis))

    def __call__(self, cond, basis):
        z_unit = cond[9]
        z_phys = z_unit * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
        # linear interp in z per (class,k); jnp.interp clamps at edges.
        def per_ck(col):                                   # col over z: (Nz,)
            return jnp.interp(z_phys, self.z_tab, col)
        # vmap over classes & k.
        flat = self.gbar_tab.reshape(self.z_tab.shape[0], -1)   # (Nz, C*K)
        vals = jax.vmap(per_ck, in_axes=1)(flat)                 # (C*K,)
        return vals.reshape(self.n_classes, -1)


class GlobalLinearHead(eqx.Module):
    """rho + a SINGLE GLOBAL linear-in-theta amplitude * a fixed smooth k-shape/class.

    Variant (b): the MINIMAL smooth theta-correction.  g_delta(theta,z,k) =
    a(theta) * shape_c(k), where a(theta) = w . (theta_unit - 0.5) + w_z*(z_unit-0.5)
    + b is ONE scalar linear functional of (the 9 cosmo params + z), and shape_c(k) =
    sum_j S[c,j] * basis_j(k) is a per-class smooth k-shape (so different classes can
    scale the same theta-amplitude differently).  Params: 9+1+1 (w, w_z, b) + 4*4 (S)
    ~ 27 -- a tiny, smooth, near-fixed correction.  Differentiable; reads cond =
    [theta_unit(9), z_unit, tau0].  coeffs() returns a(theta)*S (per-class basis coeffs).
    """
    w: jax.Array               # (n_in,) linear theta weights (incl z + tau0)
    b: jax.Array               # () scalar bias
    S: jax.Array               # (n_classes, n_basis) per-class smooth k-shape coeffs
    n_classes: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, n_in, n_classes, n_basis, key):
        k1, k2 = jax.random.split(key)
        self.w = 1e-3 * jax.random.normal(k1, (n_in,))
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


# -- training of the two custom heads (mirrors MF.train_delta_head's loss) ----- #
@eqx.filter_value_and_grad
def _custom_loss(head, basis, log_rho, cond, g_target, mask, w_k, coeff_l2, mean_prior_w):
    g_pred = jax.vmap(lambda c: head(c, basis))(cond)              # (B,4,K)
    g_full = log_rho[None, None, :] + g_pred
    diff = jnp.where(mask, g_full - g_target, 0.0)
    wk = w_k[None, None, :]
    data = jnp.sum((diff ** 2) * wk) / jnp.maximum(jnp.sum(wk * mask), 1.0)
    coeffs = jax.vmap(head.coeffs)(cond)                           # (B,4,n_basis)
    l2 = coeff_l2 * jnp.mean(coeffs ** 2)
    mean_g = jnp.mean(g_pred, axis=2)                              # (B,4)
    prior = mean_prior_w * jnp.mean(mean_g ** 2)
    return data + l2 + prior


def train_custom_head(head, tg, basis, log_rho, *, train_rows, lr, epochs,
                      coeff_l2, mean_prior_w, w_k=None):
    """Full-batch Adam on a custom head (GlobalLinearHead). Same loss as the MLP."""
    import optax
    K = basis.shape[1]
    rows = np.asarray(train_rows)
    cond = jnp.asarray(np.concatenate(
        [tg["x"][rows], tg["tau0"][rows][:, None]], axis=1))       # (B,11)
    g_target = jnp.asarray(np.nan_to_num(tg["g"][rows], nan=0.0))
    mask = jnp.asarray(np.isfinite(tg["g"][rows]))
    w_k = jnp.ones(K) if w_k is None else jnp.asarray(w_k)
    basis_j = jnp.asarray(basis); log_rho_j = jnp.asarray(log_rho)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(head, eqx.is_array))

    @eqx.filter_jit
    def step(head, opt_state):
        loss, grad = _custom_loss(head, basis_j, log_rho_j, cond, g_target, mask,
                                  w_k, coeff_l2, mean_prior_w)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(head, eqx.is_array))
        head = eqx.apply_updates(head, updates)
        return head, opt_state, loss

    hist = []
    for _ in range(epochs):
        head, opt_state, loss = step(head, opt_state)
        hist.append(float(loss))
    return head, {"loss": hist}


def build_fixed_mean_table(tg, sim_of_row, eval_logk, log_rho):
    """Per-z fixed-mean table (gbar(z,k) - log_rho) for variant (a)'s ZeroFixedMeanHead.

    Built from the FULL set of rows (the fixed mean is theta-independent -> using all
    sims is the population mean; for HF-LOSO we rebuild it from the TRAIN rows only so
    no held-out leakage)."""
    g = tg["g"]
    z = np.round(tg["x"][:, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
    zvals = np.array(sorted(set(z)))
    tab = np.zeros((len(zvals), N_CLASSES, g.shape[-1]))
    for i, zz in enumerate(zvals):
        m = (z == zz)
        with np.errstate(invalid="ignore"):
            gm = np.nanmean(g[m], axis=0)              # (4,K) mean over sims&alpha at z
        gm = np.where(np.isfinite(gm), gm, 0.0)
        tab[i] = gm - log_rho[None, :]                 # subtract per-k global mean
    return tab, zvals


# ============================================================================ #
# Scoring one variant on one HF-LOSO fold (reuses BMF metrics + Fisher)
# ============================================================================ #
def score_variant(mf, tg, hf_abs, eval_rows, k_eval):
    """Held-out KODIAQ RMS frac error (4,) + A_p/n_s Fisher bias for one fold."""
    xs = jnp.asarray(tg["x"][eval_rows]); tts = jnp.asarray(tg["tau0"][eval_rows])
    logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, tts))           # (n,4,K)
    logP_true = hf_abs["logP"][eval_rows]                         # (n,4,K)
    err_mf = BMF.frac_err_vs_k(logP_mf, logP_true)                # (4,K)
    kod = BMF.band_rms(err_mf, k_eval, *KODIAQ_BAND)              # (4,)
    FB = BMF.mf_fisher_bias(mf, hf_abs, eval_rows, k_eval, z_fid=3.0)
    return err_mf, kod, FB


# ============================================================================ #
# Figures
# ============================================================================ #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_g_decomposition(dec, k_eval, figdir):
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
    for ci, c in enumerate(CLS):
        ax = axes[0][ci]
        ax.loglog(k_eval, dec["rms_fixed"][ci], "-", color="C0", lw=2.0,
                  label="fixed mean |gbar|")
        ax.loglog(k_eval, dec["rms_theta"][ci], "-", color="C3", lw=2.0,
                  label="theta-departure RMS")
        ax.loglog(k_eval, dec["rms_noise"][ci], "-", color="0.5", lw=1.6,
                  label="noise floor (alpha)")
        ax.axvspan(KODIAQ_BAND[0], KODIAQ_BAND[1], color="grey", alpha=0.12)
        ax.axvline(LF_NYQUIST, color="k", ls="-.", lw=0.7, alpha=0.6)
        ax.set_title(c); ax.grid(alpha=0.3, which="both")
        if ci == 0:
            ax.set_ylabel("RMS of log-ratio g")
            ax.legend(fontsize=8)
        # bottom: theta / noise ratio (significance) + variance fractions.
        bx = axes[1][ci]
        bx.semilogx(k_eval, dec["theta_over_noise"][ci], "-", color="C3", lw=2.0)
        bx.axhline(1.0, color="k", lw=0.8, ls="--")
        bx.axvspan(KODIAQ_BAND[0], KODIAQ_BAND[1], color="grey", alpha=0.12)
        bx.axvline(LF_NYQUIST, color="k", ls="-.", lw=0.7, alpha=0.6)
        bx.set_xlabel("k [s/km]"); bx.grid(alpha=0.3, which="both")
        bx.set_ylim(0, max(2.0, float(np.nanmax(dec["theta_over_noise"][ci]) * 1.1)))
        if ci == 0:
            bx.set_ylabel("theta-departure / noise floor\n(>1 = significant)")
    fig.suptitle("g = logP_HF - logP_hat_LF decomposition: FIXED mean vs THETA-"
                 "departure vs NOISE floor (within-sim alpha scatter)\n"
                 "TOP: RMS of each component vs k.  BOTTOM: theta-departure / noise "
                 "(<1 => theta-departure within noise => correction ~theta-independent)."
                 "  shaded = KODIAQ 0.07-0.2; dash-dot = LF Nyquist.")
    p = Path(figdir) / "mf_g_decomposition.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


def fig_complexity_generalization(sweep, figdir):
    plt = _plt()
    order = ["a_rho_only", "b_global_linear", "c_linear_map", "d_mlp"]
    labels = ["(a) rho(k,z)\nonly", "(b) +global\nlinear", "(c) +linear\nmap",
              "(d) MLP\n(e092df1)"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    # panel 1: held-out KODIAQ RMS frac err (clean + mean over classes).
    clean = [sweep[v]["kodiaq_rms_mean"][0] for v in order]
    allc = [float(np.sqrt(np.mean(np.array(sweep[v]["kodiaq_rms_mean"]) ** 2)))
            for v in order]
    axes[0].plot(x, clean, "o-", color="C0", lw=2, ms=9, label="clean class")
    axes[0].plot(x, allc, "s--", color="C1", lw=2, ms=8, label="RMS over 4 classes")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("held-out KODIAQ RMS frac P1D error")
    axes[0].set_title("Generalization error vs complexity")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # panel 2: |ns| Fisher bias mean + worst.
    ns_mean = [sweep[v]["ns_bias_mean_abs"] for v in order]
    ns_worst = [sweep[v]["ns_bias_worst_abs"] for v in order]
    axes[1].plot(x, ns_mean, "o-", color="C3", lw=2, ms=9, label="mean |bias|")
    axes[1].plot(x, ns_worst, "s--", color="C3", lw=2, ms=8, alpha=0.6,
                 label="worst sim |bias|")
    axes[1].axhline(2.0, color="grey", lw=0.9, ls=":")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("|n_s Fisher bias| [sigma]")
    axes[1].set_title("n_s bias vs complexity (lower = better)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    # panel 3: |Ap| Fisher bias mean + worst.
    ap_mean = [sweep[v]["ap_bias_mean_abs"] for v in order]
    ap_worst = [sweep[v]["ap_bias_worst_abs"] for v in order]
    axes[2].plot(x, ap_mean, "o-", color="C2", lw=2, ms=9, label="mean |bias|")
    axes[2].plot(x, ap_worst, "s--", color="C2", lw=2, ms=8, alpha=0.6,
                 label="worst sim |bias|")
    axes[2].axhline(2.0, color="grey", lw=0.9, ls=":")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("|A_p Fisher bias| [sigma]")
    axes[2].set_title("A_p bias vs complexity (lower = better)")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("MF complexity sweep (HF-LOSO): does the SIMPLEST smooth correction "
                 "generalize BETTER than the learned MLP?")
    p = Path(figdir) / "mf_complexity_generalization.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


# ============================================================================ #
# main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lf-fold", type=int, default=0)
    ap.add_argument("--n-k", type=int, default=48)
    ap.add_argument("--k-max", type=float, default=MF.KODIAQ_KMAX)
    ap.add_argument("--n-basis", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--coeff-l2", type=float, default=1e-1)
    ap.add_argument("--mean-prior-w", type=float, default=1e-1)
    ap.add_argument("--figdir", default="figures/analysis/05_multifidelity")
    ap.add_argument("--out",
                    default="figures/analysis/05_multifidelity/mf_complexity_results.json")
    ap.add_argument("--smoke", action="store_true", help="2 held-out sims, fewer epochs")
    args = ap.parse_args()

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    t0 = time.time()

    lf = load_cache(MF.LF_CACHE)
    hr = load_cache(MF.HR_CACHE)
    lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=args.lf_fold)
    pairs = MF.match_hr_to_lf(lf, hr)
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=args.k_max, n_k=args.n_k)
    print(f"matched {len(pairs)} HR<->LF rows; eval {k_eval[0]:.4f}..{k_eval[-1]:.4f} "
          f"s/km ({args.n_k} bins); LF fold {args.lf_fold} ({time.time()-t0:.1f}s)")

    tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk, pairs)
    hf_abs = BMF.measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=args.n_basis))

    # -------- PART 1: decomposition (uses ALL rows) -------------------------- #
    print("\n========== PART 1: g DECOMPOSITION (ANOVA) ==========")
    dec = decompose_g(tg, sim_of_row, k_eval)
    fp_dec = fig_g_decomposition(dec, k_eval, args.figdir)
    print(f"{'band':18} {'class':8} {'|gbar|':>8} {'dtheta':>8} {'noise':>8} "
          f"{'%fix':>6} {'%th':>6} {'%noi':>6} {'th/noi':>7}")
    for band, bd in dec["bands"].items():
        for c in CLS:
            r = bd[c]
            print(f"{band:18} {c:8} {r['rms_fixed']:8.4f} {r['rms_theta']:8.4f} "
                  f"{r['rms_noise']:8.4f} {100*r['frac_fixed']:6.1f} "
                  f"{100*r['frac_theta']:6.1f} {100*r['frac_noise']:6.1f} "
                  f"{r['theta_over_noise']:7.2f}")

    # -------- PART 2: complexity sweep (HF-LOSO) ----------------------------- #
    if args.smoke:
        sims = sims[:2]; args.epochs = min(args.epochs, 200)
    print(f"\n========== PART 2: COMPLEXITY SWEEP (HF-LOSO over {len(sims)} sims) ==========")

    variants = ["a_rho_only", "b_global_linear", "c_linear_map", "d_mlp"]
    # per-variant accumulation across folds.
    acc = {v: {"kod": [], "ns": [], "ap": [], "err": []} for v in variants}

    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]
        eval_rows = np.where(held)[0]
        ts = time.time()
        # rho rebuilt from TRAIN rows ONLY (no held-out leakage).
        log_rho = MF.mean_log_ratio_rho(
            {"g": tg["g"][train_rows]}, eval_logk)
        log_rho = np.where(np.isfinite(log_rho), log_rho, 0.0)

        heads = {}
        # (a) rho(k,z) only: ZeroFixedMeanHead with per-z fixed mean (TRAIN rows).
        tab, ztab = build_fixed_mean_table(
            {"g": tg["g"][train_rows], "x": tg["x"][train_rows]},
            sim_of_row[train_rows], eval_logk, log_rho)
        heads["a_rho_only"] = ZeroFixedMeanHead(tab, ztab, args.n_basis)
        # (b) rho + global linear-in-theta amplitude * per-class smooth k-shape.
        h_b = GlobalLinearHead(n_in=11, n_classes=N_CLASSES, n_basis=args.n_basis,
                               key=jax.random.PRNGKey(0))
        h_b, _ = train_custom_head(h_b, tg, basis, log_rho, train_rows=train_rows,
                                   lr=args.lr, epochs=args.epochs,
                                   coeff_l2=args.coeff_l2, mean_prior_w=args.mean_prior_w)
        heads["b_global_linear"] = h_b
        # (c) rho + linear map (DeltaHead n_layers=0: affine theta->coeffs).
        h_c, _ = MF.train_delta_head(
            tg, basis, log_rho, eval_logk, train_mask_rows=train_rows,
            n_basis=args.n_basis, width=args.width, n_layers=0, lr=args.lr,
            epochs=args.epochs, coeff_l2=args.coeff_l2,
            mean_prior_w=args.mean_prior_w, seed=0)
        heads["c_linear_map"] = h_c
        # (d) rho + MLP (the e092df1 baseline: width=16, n_layers=1).
        h_d, _ = MF.train_delta_head(
            tg, basis, log_rho, eval_logk, train_mask_rows=train_rows,
            n_basis=args.n_basis, width=args.width, n_layers=1, lr=args.lr,
            epochs=args.epochs, coeff_l2=args.coeff_l2,
            mean_prior_w=args.mean_prior_w, seed=0)
        heads["d_mlp"] = h_d

        line = f"[{s[:14]}..] {time.time()-ts:.1f}s "
        for v in variants:
            mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, heads[v],
                                        eval_logk=eval_logk, log_rho=log_rho,
                                        n_basis=args.n_basis)
            err_mf, kod, FB = score_variant(mf, tg, hf_abs, eval_rows, k_eval)
            acc[v]["kod"].append(kod)
            acc[v]["err"].append(err_mf)
            acc[v]["ns"].append(FB["bias_in_sigma"]["ns"])
            acc[v]["ap"].append(FB["bias_in_sigma"]["Ap"])
            line += (f"| {v.split('_')[0]}: kod={kod[0]:.3f} "
                     f"ns={FB['bias_in_sigma']['ns']:+.2f} ap={FB['bias_in_sigma']['Ap']:+.2f} ")
        print(line)

    # -------- aggregate the sweep -------------------------------------------- #
    sweep = {}
    for v in variants:
        kod = np.stack(acc[v]["kod"])                  # (nsim,4)
        kod_mean = np.sqrt(np.mean(kod ** 2, axis=0))  # (4,) RMS over folds per class
        ns = np.array(acc[v]["ns"]); apv = np.array(acc[v]["ap"])
        sweep[v] = dict(
            kodiaq_rms_mean=kod_mean.tolist(),
            kodiaq_rms_per_sim=kod.tolist(),
            ns_bias=ns.tolist(), ap_bias=apv.tolist(),
            ns_bias_mean_abs=float(np.mean(np.abs(ns))),
            ns_bias_worst_abs=float(np.max(np.abs(ns))),
            ap_bias_mean_abs=float(np.mean(np.abs(apv))),
            ap_bias_worst_abs=float(np.max(np.abs(apv))),
            ns_bias_rms=float(np.sqrt(np.mean(ns ** 2))),
            ap_bias_rms=float(np.sqrt(np.mean(apv ** 2))),
        )

    fp_cx = fig_complexity_generalization(sweep, args.figdir)

    # -------- report --------------------------------------------------------- #
    print("\n========== COMPLEXITY-SWEEP SUMMARY (HF-LOSO) ==========")
    print(f"{'variant':18} {'kod(clean)':>11} {'kod(4cls)':>10} "
          f"{'|ns|mean':>9} {'|ns|worst':>10} {'|Ap|mean':>9} {'|Ap|worst':>10}")
    nice = {"a_rho_only": "(a) rho(k,z) only", "b_global_linear": "(b) +glob linear",
            "c_linear_map": "(c) +linear map", "d_mlp": "(d) MLP e092df1"}
    for v in variants:
        sw = sweep[v]
        kod4 = float(np.sqrt(np.mean(np.array(sw["kodiaq_rms_mean"]) ** 2)))
        print(f"{nice[v]:18} {sw['kodiaq_rms_mean'][0]:11.3f} {kod4:10.3f} "
              f"{sw['ns_bias_mean_abs']:9.2f} {sw['ns_bias_worst_abs']:10.2f} "
              f"{sw['ap_bias_mean_abs']:9.2f} {sw['ap_bias_worst_abs']:10.2f}")
    print("\nPer-sim n_s bias (sigma):")
    for i, s in enumerate(sims):
        row = "  ".join(f"{v.split('_')[0]}={sweep[v]['ns_bias'][i]:+.2f}" for v in variants)
        print(f"  {s[:14]}..  {row}")

    out = {
        "config": vars(args),
        "k_eval": k_eval.tolist(), "kodiaq_band": list(KODIAQ_BAND),
        "lf_nyquist": LF_NYQUIST, "sims": sims,
        "decomposition_bands": dec["bands"],
        "decomposition_per_k": {
            "rms_fixed": dec["rms_fixed"].tolist(),
            "rms_theta": dec["rms_theta"].tolist(),
            "rms_noise": dec["rms_noise"].tolist(),
            "theta_over_noise": dec["theta_over_noise"].tolist(),
        },
        "sweep": sweep,
        "figures": {"g_decomposition": fp_dec, "complexity": fp_cx},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out}\nfigures:\n  {fp_dec}\n  {fp_cx}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
