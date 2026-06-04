"""DATA-RANGE-RESTRICTED re-evaluation of the deployed LF emulator's error metrics.

The headline diagnostics (scripts/diag_ap_fisher_bias.py, diag_residual_decomposition.py)
sum the A_p Fisher-bias and the coherent low-k tilt over the FULL cache footprint:
  z ∈ {2.0 .. 5.4}  (18 bins)   and   k ∈ [4.4e-4 .. 0.076] s/km (172 bins).
But the DESI DR1 P1D data live ONLY in
  z ∈ [2.2, 4.6]   and   k ≳ 1e-3 s/km
so z=2.0 and z∈{4.8,5.0,5.2,5.4} are OUT of range, and the lowest two k bins
(4.4e-4, 8.8e-4 s/km) are below DESI's k_min. The worst "catastrophic" low-k
coherent residual lives in exactly those k<1e-3 bins, and the A_p Fisher-bias is
LOW-k-weighted (the Δ²_* pivot k_* ≈ 0.009 s/km), so the headline A_p bias /
per-fold scatter MAY be inflated by modes the data never constrain.

This script re-runs the A_p (+9-param) Fisher-bias and the coherent tilt over the
8 LOSO folds, comparing the FULL footprint to the DATA-RANGE MASK
  M_data = { z ∈ [2.2, 4.6] }  AND  { k ≥ 1e-3 s/km },
and DECOMPOSES the (full) A_p risk by region:
  (a) k < 1e-3   (below DESI k_min),
  (b) z > 4.6    (above DESI z_max),
  (c) z < 2.2    (below DESI z_min, i.e. z=2.0),
  (d) the in-range CORE (z ∈ [2.2,4.6] AND k ≥ 1e-3).
by recomputing the Fisher J=∂P/∂θ + bias projection with each region removed.

KEY DESIGN DIFFERENCE vs the gate script: the gate (diag_ap_fisher_bias.fisher_bias)
sums the Fisher modes over a SINGLE fiducial z-slice (z=3.0). To decompose the bias
by z-region we MUST sum over ALL held-out z-slices, so here the Fisher rows span
(class, z-slice, k): J is evaluated per (z,τ₀)-cell at the cube-centre θ (the
baseline is θ-blind, so ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ depends on z via the z_unit input
and on τ₀), and δ is the deployed coherent log-error per (class,z,k) (mean over the
held-out val sims of that cell). A per-z-slice fiducial-only restriction reproduces
the gate's z=3.0 number as a cross-check.

READ-ONLY on production code. Reuses the deployed per-fold checkpoints
checkpoints/decomp_nb24_fold{0..7} (n_basis=24, deployed recipe) when present;
trains them otherwise (deployed config, matching diag_residual_decomposition).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_datarange_mask.py [--folds N]
Writes figures + a JSON summary to figures/analysis/04_emulator/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log,
)
from hcd_analysis.emulator import train as T

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
CV_JSON = f"{OUT}/diag_lfhf_tilt_and_cv.json"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
IAP, INS = 1, 0   # A_p and n_s column indices

# DESI-DR1-like diagonal per-mode covariance (same assumption as the gate script).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
KPIVOT = 0.009    # Lyα amplitude pivot Δ²_* (A_p maps onto low-k)

# DESI DR1 P1D data range.
Z_DATA = (2.2, 4.6)
K_DATA_MIN = 1e-3

# Coherent-tilt bands (match diag_residual_decomposition).
LOWK = (0.0, 0.01)
MIDK = (0.01, 0.03)
HIGHK = (0.03, 0.07)

# Deployed production recipe (matches diag_residual_decomposition.DEPLOYED).
DEPLOYED = dict(n_basis=24, epochs=180, lr=1e-3, batch=512, patience=25,
                seed=0, early_stop_metric="auto", weight_decay=1e-4)


# --------------------------------------------------------------------------- #
# Train / load one fold at the deployed config (reuse decomp_nb24_fold{f}).
# --------------------------------------------------------------------------- #
def train_or_load_fold(d, fold, n_k, *, reuse=True):
    ckpt = f"/home/mfho/hcd_priya/checkpoints/decomp_nb24_fold{fold}"
    tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
    if reuse and Path(ckpt + ".eqx").exists():
        model, meta, norm = T.load_checkpoint(ckpt)
        return model, norm, (tr, va, ho)
    model, norm, hist = T.train_fold(
        d, tr, va, n_basis=DEPLOYED["n_basis"], lr=DEPLOYED["lr"],
        epochs=DEPLOYED["epochs"], batch_size=DEPLOYED["batch"],
        seed=DEPLOYED["seed"], key=jax.random.PRNGKey(DEPLOYED["seed"]),
        patience=DEPLOYED["patience"], n_k=n_k,
        early_stop_metric=DEPLOYED["early_stop_metric"],
        weight_decay=DEPLOYED["weight_decay"])
    T.save_checkpoint(ckpt, model,
                      {"in_dim": 10, "n_k": n_k, "n_basis": DEPLOYED["n_basis"]},
                      norm, seed=DEPLOYED["seed"], kfkms=d["kfkms"], cache_path=CACHE)
    print(f"  [train] fold {fold}: {len(hist['train_loss'])} epochs -> {ckpt}")
    return model, norm, (tr, va, ho)


# --------------------------------------------------------------------------- #
# Build the per-mode Fisher ingredients over ALL held-out z-slices.
#   rows = (class, z-slice, k);  J[m] = ∂logP̂/∂θ (9+τ₀),  δ[m] = coherent log-err.
# Each row carries (class, z, k) so we can apply arbitrary region masks AFTER.
# --------------------------------------------------------------------------- #
def build_fisher_rows(model, d, va, norm):
    """Assemble (J, delta, Cinv, class_idx, z, k) over all (class, held-out z, k)
    modes. J is per (z,τ₀)-cell at θ=cube-centre; δ is the deployed coherent
    log-error (mean over held-out val sims sharing that z-slice)."""
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]                       # (4,K)
    n_k = sc.shape[1]
    zg_va = d["z_grid"][va]
    z_unique = np.unique(np.round(zg_va, 4))   # held-out z-slices

    # deployed prediction on all val rows (for δ).
    x_va = jnp.asarray(d["x"][va]); tau_va = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])      # (n,4,K)
    logP_true = safe_log(d["P_filt"][va])                        # (n,4,K)
    finite = np.isfinite(logP_true)

    # k-grid (median over rows, as the other scripts do).
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"][va]),
                               d["kfkms"][va], np.nan), 0)        # (K,)

    rows_J, rows_d, rows_C = [], [], []
    rows_ci, rows_z, rows_k = [], [], []
    for z_fid in z_unique:
        vz = np.isclose(np.round(zg_va, 4), z_fid, atol=1e-6)
        if not vz.any():
            continue
        # representative τ₀ / z_unit for this z-slice (median over its val rows).
        tau0_fid = float(np.median(d["tau0"][va][vz]))
        z_unit_fid = float(np.median(d["x"][va][vz, 9]))
        z_j = jnp.asarray(z_unit_fid)

        def rhat(theta9_tau):
            x = jnp.concatenate([theta9_tau[:9], z_j[None]])
            return model(x, theta9_tau[9])["P_filt_resid"]       # (4,n_k)

        p0 = jnp.asarray(np.concatenate([np.full(9, 0.5), [tau0_fid]]),
                         dtype=jnp.float64)
        Jr = np.asarray(jax.jacfwd(rhat)(p0))                    # (4,n_k,10)
        J_logP = Jr * sc[:, :, None]                             # ∂logP̂/∂(θ,τ₀)

        # coherent deployed log-error at this z-slice (mean over val sims).
        m_slice = finite[vz]
        err = (logP_hat - logP_true)[vz]
        kvalid = (m_slice.mean(0) > 0.5)                         # (4,K)
        with np.errstate(invalid="ignore"):
            delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j],
                                                   err[:, ci, j], np.nan))
                               for j in range(n_k)] for ci in range(4)])  # (4,K)
        for ci, nm in enumerate(CLS):
            for j in range(n_k):
                if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
                   or not np.isfinite(delta[ci, j]):
                    continue
                rows_J.append(J_logP[ci, j])
                rows_d.append(delta[ci, j])
                rows_C.append(SIGMA_FRAC[nm] ** 2)
                rows_ci.append(ci); rows_z.append(float(z_fid)); rows_k.append(kf[j])
    return dict(
        J=np.array(rows_J),                  # (M,10)
        delta=np.array(rows_d),              # (M,)
        Cinv=1.0 / np.array(rows_C),         # (M,)
        ci=np.array(rows_ci, int),           # (M,)
        z=np.array(rows_z),                  # (M,)
        k=np.array(rows_k),                  # (M,)
    )


def fisher_project(rows, sel):
    """9-param Fisher-bias (in σ units) + abs unit-cube shift over selected modes.

    Returns the bias_sigma & dtheta_unit for A_p and n_s plus diagnostics. ``sel``
    is a boolean mask over the rows. Mirrors diag_ap_fisher_bias.fisher_bias's
    9-param block exactly (same ridge, same C^{-1}-weighting)."""
    J = rows["J"][sel][:, :9]                 # (m,9)
    dv = rows["delta"][sel]                    # (m,)
    Cinv = rows["Cinv"][sel]                   # (m,)
    if len(dv) < 9:
        nanp = {p: float("nan") for p in PARAMS}
        return dict(n_modes=int(len(dv)), bias_sigma=nanp, dtheta_unit=nanp,
                    sigma_fisher=nanp, fisher_cond=float("nan"),
                    ap_bias_sigma=float("nan"), ap_dtheta_unit=float("nan"),
                    ns_bias_sigma=float("nan"), ns_dtheta_unit=float("nan"))
    F = (J.T * Cinv) @ J                       # (9,9)
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    return dict(
        n_modes=int(len(dv)),
        bias_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        dtheta_unit={PARAMS[i]: float(dtheta[i]) for i in range(9)},
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)},
        fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
        ap_bias_sigma=float(bias_sigma[IAP]), ap_dtheta_unit=float(dtheta[IAP]),
        ns_bias_sigma=float(bias_sigma[INS]), ns_dtheta_unit=float(dtheta[INS]),
    )


# --------------------------------------------------------------------------- #
# Fisher projections.
#
# IMPORTANT (apples-to-apples with the headline +0.029σ ± 0.160σ): the deployed
# gate (diag_ap_fisher_bias / diag_residual_decomposition) builds ONE Fisher
# matrix from a SINGLE z-slice (z=3.0): 4 classes × 172 k = 688 modes. Summing
# the Fisher over all 18 z-slices (≈12k modes) over-stiffens F (σ_Fisher shrinks
# ∝1/√N_modes) and inflates bias/σ by ~√18, which is NOT comparable to the
# headline. So here:
#   * the headline FULL-vs-MASKED comparison restricts to the fiducial z=3.0
#     slice and only changes the k-cut (drop k<1e-3) — apples-to-apples;
#   * the z-region decomposition runs the SAME 688-mode-style projection at EACH
#     z-slice independently (so each slice has its own well-conditioned Fisher),
#     then reports A_p bias per z-slice. This directly answers "is the A_p risk
#     out-of-range (z>4.6, z=2.0) or in-range (z≈3.4–3.8)?".
# --------------------------------------------------------------------------- #
def slice_masks(rows, z_fid, *, mask_k):
    """Boolean row-mask for ONE z-slice; optionally also drop k<1e-3."""
    sel = np.isclose(rows["z"], z_fid, atol=1e-6)
    if mask_k:
        sel = sel & (rows["k"] >= K_DATA_MIN)
    return sel


# --------------------------------------------------------------------------- #
# Coherent tilt (deployed |P̂/P−1| coherent bias) per class, FULL vs MASKED.
#   Mirrors diag_residual_decomposition's coherent-fraction, but split by the
#   data-range z/k mask and reported against the CV floor.
# --------------------------------------------------------------------------- #
def coherent_tilt(model, d, va, norm, kf):
    """Per-(class,k) coherent fractional bias = mean over held-out rows of
    exp(logP̂−logP_true)−1, split by z-region (in-range vs full). Returns the
    per-class coherent RMS in named bands for FULL z and DATA-RANGE z."""
    pf = norm["P_filt"]; sc = pf["sig_cosmo"]
    x_va = jnp.asarray(d["x"][va]); tau_va = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])
    logP_true = safe_log(d["P_filt"][va])
    fin = np.isfinite(logP_true)
    frac = np.where(fin, np.exp(logP_hat - logP_true) - 1.0, np.nan)   # (n,4,K)
    zg = np.round(d["z_grid"][va], 4)
    in_z = (zg >= Z_DATA[0] - 1e-6) & (zg <= Z_DATA[1] + 1e-6)

    def coh(rowsel):
        with np.errstate(invalid="ignore"):
            return np.nanmean(frac[rowsel], axis=0)        # (4,K)

    coh_full = coh(np.ones(len(va), bool))
    coh_inz = coh(in_z)
    return dict(coh_full=coh_full, coh_inz=coh_inz, kf=kf)


def load_per_class_floor():
    """Per-class irreducible residual floor (fractional) from the CV diagnostic's
    floor_comparison.per_class_floor_pct (the deployable-target floor per class).
    NOTE this is much larger for DLA (~4.8%) than clean (~1.3%); the per-k-band CV
    proxy below is the CLEAN P1D cosmic-variance floor (~0.5% at low-k)."""
    if not Path(CV_JSON).exists():
        return {}
    try:
        pcf = json.load(open(CV_JSON))["measurement_2_cosmic_variance"][
            "floor_comparison"]["per_class_floor_pct"]
        return {k: float(v) / 100.0 for k, v in pcf.items()}
    except (KeyError, TypeError):
        return {}


def load_cv_band():
    """Per-band fractional CV floor (median_pct) keyed by (lo,hi)."""
    if not Path(CV_JSON).exists():
        return None
    cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
    out = {}
    for band, v in cvj.items():
        lo, hi = (float(x) for x in band.split("-"))
        out[(lo, hi)] = v["median_pct"] / 100.0
    return out


def cv_band_value(cv, band):
    if cv is None:
        return float("nan")
    sel = [v for (lo, hi), v in cv.items()
           if band[0] <= (lo + hi) / 2 < band[1]]
    if sel:
        return float(np.median(sel))
    best = min(cv, key=lambda b: abs((b[0] + b[1]) / 2 - (band[0] + band[1]) / 2))
    return float(cv[best])


def band_rms(coh_ck, kf, ci, band, k_floor=0.0):
    """Coherent RMS + signed mean over a k-band for class ci, optionally with a
    lower k floor (to exclude k<1e-3 from the 'low-k' band when masking)."""
    kb = (np.isfinite(kf) & (kf >= max(band[0], k_floor)) & (kf < band[1]))
    v = coh_ck[ci, kb]
    if not np.isfinite(v).any():
        return float("nan"), float("nan"), 0
    return (float(np.sqrt(np.nanmean(v ** 2))), float(np.nanmean(v)),
            int(np.isfinite(v).sum()))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_perfold_ap(perfold, path):
    """Per-fold A_p & n_s at z=3.0: FULL (all k) vs k≥1e-3 (drop the 2 low-k bins).
    This is the apples-to-apples comparison to the +0.029σ±0.160σ headline."""
    folds = sorted(perfold)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for key, lab, c, mk in (("ap_full", "A_p FULL (all k, z=3)", "C3", "o"),
                            ("ap_masked", "A_p k≥1e-3 (z=3)", "C0", "s")):
        v = np.array([perfold[f][key] for f in folds])
        m, s = np.nanmean(v), np.nanstd(v)
        axes[0].plot(folds, v, "-" + mk, color=c,
                     label=f"{lab}: {m:+.3f}±{s:.3f}σ")
        axes[0].axhline(m, color=c, ls=":", lw=0.8)
    axes[0].axhline(0.2, color="grey", ls="--", lw=1, label="±0.2σ gate")
    axes[0].axhline(-0.2, color="grey", ls="--", lw=1)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("LOSO fold"); axes[0].set_ylabel("A_p Fisher-bias [σ]")
    axes[0].set_title("A_p Fisher-bias per fold (z=3.0): FULL k vs k≥1e-3")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for key, lab, c, mk in (("ns_full", "n_s FULL (all k, z=3)", "C3", "o"),
                            ("ns_masked", "n_s k≥1e-3 (z=3)", "C0", "s")):
        v = np.array([perfold[f][key] for f in folds])
        m, s = np.nanmean(v), np.nanstd(v)
        axes[1].plot(folds, v, "-" + mk, color=c,
                     label=f"{lab}: {m:+.3f}±{s:.3f}σ")
        axes[1].axhline(m, color=c, ls=":", lw=0.8)
    axes[1].axhline(0.2, color="grey", ls="--", lw=1, label="±0.2σ gate")
    axes[1].axhline(-0.2, color="grey", ls="--", lw=1)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("LOSO fold"); axes[1].set_ylabel("n_s Fisher-bias [σ]")
    axes[1].set_title("n_s Fisher-bias per fold (z=3.0): FULL k vs k≥1e-3")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def fig_perz_ap(perfold, path):
    """A_p Fisher-bias vs z-slice (each slice its own 688-mode Fisher), full-k.
    Shows per-fold curves + the mean±scatter band; shades the OUT-of-range z so
    the reader can see whether the A_p risk lives out-of-range or in-range z≈3.5."""
    folds = sorted(perfold)
    z_slices = sorted(set(np.round(np.array(
        [float(z) for z in perfold[0]["per_z"]]), 4)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, key, ttl in ((axes[0], "ap_full", "A_p bias per z-slice (FULL k)"),
                         (axes[1], "ap_kmasked", "A_p bias per z-slice (k≥1e-3)")):
        M = np.array([[perfold[f]["per_z"][f"{z:.1f}"][key] for z in z_slices]
                      for f in folds])              # (n_fold, n_z)
        for fi, f in enumerate(folds):
            ax.plot(z_slices, M[fi], "-", color="grey", lw=0.7, alpha=0.5)
        mean = np.nanmean(M, axis=0); std = np.nanstd(M, axis=0)
        ax.plot(z_slices, mean, "-o", color="C1", lw=2.0, label="mean over folds")
        ax.fill_between(z_slices, mean - std, mean + std, color="C1", alpha=0.2,
                        label="±1σ scatter over folds")
        # shade out-of-range z
        ax.axvspan(min(z_slices) - 0.1, Z_DATA[0], color="red", alpha=0.07)
        ax.axvspan(Z_DATA[1], max(z_slices) + 0.1, color="red", alpha=0.07,
                   label="out-of-range z")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(0.2, color="grey", ls="--", lw=0.9)
        ax.axhline(-0.2, color="grey", ls="--", lw=0.9)
        ax.axvspan(3.3, 3.9, color="C0", alpha=0.10, label="named worst z≈3.4–3.8")
        ax.set_xlabel("z-slice"); ax.set_title(ttl); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("A_p Fisher-bias [σ] (per-slice 688-mode Fisher)")
    fig.suptitle("A_p Fisher-bias by z-slice (each slice its own Fisher): "
                 "is the A_p risk out-of-range (z<2.2 or z>4.6) or in-range (z≈3.4–3.8)?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def fig_coherent_tilt(coh, cv, path):
    kf = coh["kf"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    g = np.isfinite(kf)
    for ci, nm in enumerate(CLS):
        ax = axes[ci]
        ax.axhline(0, color="k", lw=0.6)
        ax.semilogx(kf[g], 100 * coh["coh_full"][ci, g], "C3-", lw=1.8,
                    label="FULL (all z)")
        ax.semilogx(kf[g], 100 * coh["coh_inz"][ci, g], "C0--", lw=1.8,
                    label="z∈[2.2,4.6] only")
        if cv is not None:
            for (lo, hi), v in cv.items():
                ax.hlines([100 * v, -100 * v], max(lo, kf[g].min()),
                          min(hi, kf[g].max()), color="grey", lw=1.0, alpha=0.6)
        ax.axvspan(kf[g].min() * 0.9, K_DATA_MIN, color="red", alpha=0.08)  # k<1e-3
        ax.axvline(K_DATA_MIN, color="red", ls=":", lw=1.0)
        ax.axvline(KPIVOT, color="purple", ls=":", lw=0.9)
        ax.set_title(nm); ax.set_xlabel("k [s/km]"); ax.grid(alpha=0.3, which="both")
        if ci == 0:
            ax.set_ylabel("coherent bias ⟨P̂/P−1⟩ [%]")
            ax.legend(fontsize=8)
    # zoom y to the in-range structure
    lim = max(2.0, float(np.nanpercentile(
        np.abs(100 * coh["coh_inz"][:, (kf >= K_DATA_MIN)]), 99)) * 1.3)
    for ax in axes:
        ax.set_ylim(-lim, lim)
    fig.suptitle("Coherent deployed bias ⟨P̂/P−1⟩ per class: FULL z vs DATA-RANGE z; "
                 "red band/line = k<1e-3 (below DESI k_min); grey = ±CV floor; "
                 "purple = A_p pivot k*≈0.009")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--no-reuse", action="store_true",
                    help="retrain folds instead of reusing decomp_nb24 checkpoints")
    args = ap.parse_args()

    d = load_cache(CACHE)
    n_k = d["P_tier_p"].shape[1]
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"]), d["kfkms"], np.nan), 0)
    cv = load_cv_band()
    n_klow = int((kf < K_DATA_MIN).sum())
    zall = sorted(set(np.round(d["z_grid"], 4)))
    z_oob = [z for z in zall if z < Z_DATA[0] - 1e-6 or z > Z_DATA[1] + 1e-6]
    print(f"cache: n_k={n_k} ({n_klow} bins k<{K_DATA_MIN:.0e} OUT), "
          f"z bins={zall}\n  OUT-of-range z: {z_oob}  (DESI z∈{Z_DATA}, k≥{K_DATA_MIN:.0e})")

    Z_FID = 3.0   # the headline fiducial z-slice (gate uses z=3.0)
    perfold = {}
    coh0 = None
    for fold in range(args.folds):
        model, norm, (tr, va, ho) = train_or_load_fold(
            d, fold, n_k, reuse=not args.no_reuse)
        rows = build_fisher_rows(model, d, va, norm)
        z_slices = sorted(set(np.round(rows["z"], 4)))

        # ---- KEY RESULT: FULL vs DATA-RANGE-MASKED at the fiducial z=3.0 slice
        # (apples-to-apples with the +0.029σ ± 0.160σ headline; only the k<1e-3
        # cut changes). z=3.0 is in-range so the z-mask is a no-op here; the
        # z-region effect is captured by the per-z-slice decomposition below.
        full = fisher_project(rows, slice_masks(rows, Z_FID, mask_k=False))
        masked = fisher_project(rows, slice_masks(rows, Z_FID, mask_k=True))

        # ---- z-region decomposition: A_p bias per z-slice (each its own Fisher),
        # FULL k vs k≥1e-3. Lets us read off whether the A_p risk is out-of-range
        # (z>4.6, z=2.0) or in-range (z≈3.4–3.8).
        per_z = {}
        for zf in z_slices:
            fz = fisher_project(rows, slice_masks(rows, zf, mask_k=False))
            mz = fisher_project(rows, slice_masks(rows, zf, mask_k=True))
            per_z[f"{zf:.1f}"] = dict(
                ap_full=fz["ap_bias_sigma"], ap_kmasked=mz["ap_bias_sigma"],
                ns_full=fz["ns_bias_sigma"], ns_kmasked=mz["ns_bias_sigma"],
                in_range=bool(Z_DATA[0] - 1e-6 <= zf <= Z_DATA[1] + 1e-6),
                n_modes_full=fz["n_modes"], n_modes_kmasked=mz["n_modes"])

        perfold[fold] = dict(
            ap_full=full["ap_bias_sigma"], ap_masked=masked["ap_bias_sigma"],
            ap_dtheta_full=full["ap_dtheta_unit"], ap_dtheta_masked=masked["ap_dtheta_unit"],
            ns_full=full["ns_bias_sigma"], ns_masked=masked["ns_bias_sigma"],
            ns_dtheta_full=full["ns_dtheta_unit"], ns_dtheta_masked=masked["ns_dtheta_unit"],
            n_modes_full=full["n_modes"], n_modes_masked=masked["n_modes"],
            sigma_ap_full=full["sigma_fisher"]["Ap"],
            sigma_ap_masked=masked["sigma_fisher"]["Ap"],
            cond_full=full["fisher_cond"], cond_masked=masked["fisher_cond"],
            bias_sigma_full=full["bias_sigma"], bias_sigma_masked=masked["bias_sigma"],
            per_z=per_z,
        )
        print(f"  fold {fold}: A_p@z3 FULL={full['ap_bias_sigma']:+.3f}σ  "
              f"k≥1e-3={masked['ap_bias_sigma']:+.3f}σ   "
              f"| n_s FULL={full['ns_bias_sigma']:+.3f}σ k≥1e-3={masked['ns_bias_sigma']:+.3f}σ "
              f"| modes {full['n_modes']}->{masked['n_modes']}")
        if fold == 0:
            coh0 = coherent_tilt(model, d, va, norm, kf)

    # ---- summary stats over folds ----
    def stat(getter):
        v = np.array([getter(perfold[f]) for f in perfold], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return dict(mean=float("nan"), std=float("nan"), min=float("nan"),
                        max=float("nan"), vals=[])
        return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                    min=float(np.min(v)), max=float(np.max(v)),
                    vals=[float(x) for x in v])

    Z_FID = 3.0
    # z-slice groups for the region decomposition.
    z_slices = sorted(set(np.round(np.array(
        [float(z) for z in perfold[0]["per_z"]]), 4)))
    in_range_z = [z for z in z_slices if Z_DATA[0] - 1e-6 <= z <= Z_DATA[1] + 1e-6]
    zhigh = [z for z in z_slices if z > Z_DATA[1] + 1e-6]
    zlow = [z for z in z_slices if z < Z_DATA[0] - 1e-6]
    core_z = [z for z in in_range_z if abs(z - 3.4) < 1e-6 or abs(z - 3.6) < 1e-6
              or abs(z - 3.8) < 1e-6]   # the named worst in-range band z≈3.4–3.8

    def ap_at(z, key="ap_full"):
        return lambda r: r["per_z"][f"{z:.1f}"][key]

    summ = {
        # headline FULL vs DATA-RANGE-MASKED at z=3.0 (only k<1e-3 cut changes)
        "ap_full_z3": stat(lambda r: r["ap_full"]),
        "ap_masked_z3": stat(lambda r: r["ap_masked"]),
        "ns_full_z3": stat(lambda r: r["ns_full"]),
        "ns_masked_z3": stat(lambda r: r["ns_masked"]),
        # per-z-slice A_p bias (each its own Fisher), full-k and k-masked
        "ap_perz_full": {f"{z:.1f}": stat(ap_at(z, "ap_full")) for z in z_slices},
        "ap_perz_kmasked": {f"{z:.1f}": stat(ap_at(z, "ap_kmasked")) for z in z_slices},
        "ns_perz_full": {f"{z:.1f}": stat(ap_at(z, "ns_full")) for z in z_slices},
        # region aggregates: RMS-over-z of the per-z A_p bias mean (a region's typical
        # |A_p| leverage) + the scatter over folds of the region-RMS.
        "z_groups": {"in_range": in_range_z, "z_gt_4.6": zhigh,
                     "z_lt_2.2": zlow, "core_3.4_3.8": core_z},
    }

    # region A_p risk: per fold, the RMS over the region's z-slices of |A_p bias|,
    # then mean±std over folds. Big RMS => that region drives A_p bias/scatter.
    def region_rms(zgroup, key="ap_full"):
        def getter(r):
            vals = [r["per_z"][f"{z:.1f}"][key] for z in zgroup]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
        return stat(getter)

    summ["region_ap_rms"] = {
        "in_range_fullk": region_rms(in_range_z, "ap_full"),
        "core_3.4_3.8_fullk": region_rms(core_z, "ap_full"),
        "z_gt_4.6_fullk": region_rms(zhigh, "ap_full"),
        "z_lt_2.2_fullk": region_rms(zlow, "ap_full"),
        "in_range_kmasked": region_rms(in_range_z, "ap_kmasked"),
        "core_3.4_3.8_kmasked": region_rms(core_z, "ap_kmasked"),
    }
    # k<1e-3 effect on A_p at z=3.0: |Δ(A_p bias)| from dropping the 2 low-k bins.
    summ["ap_klow_shift_z3"] = stat(
        lambda r: r["ap_full"] - r["ap_masked"])

    # ---- coherent tilt FULL vs MASKED, per class, vs CV floor (fold 0) ----
    per_class_floor = load_per_class_floor()
    tilt = {}
    for ci, nm in enumerate(CLS):
        rec = {}
        for bname, band in (("lowk", LOWK), ("mid", MIDK), ("highk", HIGHK)):
            # FULL: full z, full k (incl k<1e-3 in the low-k band)
            rf, mf, nf = band_rms(coh0["coh_full"], kf, ci, band, k_floor=0.0)
            # MASKED: in-range z, k>=1e-3 (drops k<1e-3 from the low-k band)
            rm, mm, nm_ = band_rms(coh0["coh_inz"], kf, ci, band, k_floor=K_DATA_MIN)
            rec[bname] = dict(full_rms=rf, full_mean=mf, full_nk=nf,
                              masked_rms=rm, masked_mean=mm, masked_nk=nm_,
                              cv_floor_frac=cv_band_value(cv, band),
                              per_class_floor_frac=per_class_floor.get(nm, float("nan")))
        tilt[nm] = rec

    # ---- figures ----
    fig_perfold_ap(perfold, f"{OUT}/datarange_perfold_ap.png")
    fig_perz_ap(perfold, f"{OUT}/datarange_perz_ap.png")
    fig_coherent_tilt(coh0, cv, f"{OUT}/datarange_coherent_tilt.png")

    # ---- assemble JSON ----
    out = {
        "definition": {
            "data_range": {"z": list(Z_DATA), "k_min_skm": K_DATA_MIN,
                           "out_of_range_z": [float(z) for z in z_oob],
                           "n_k_below_kmin": n_klow,
                           "kf_below_kmin": [float(x) for x in kf[kf < K_DATA_MIN]]},
            "fisher_modes": "Per z-slice: rows = (class, k) at that z (4×172=688 modes, "
                            "matching the deployed gate's single-slice Fisher). "
                            "J=σ_cosmo·∂r̂/∂θ at θ=cube-centre (baseline θ-blind); "
                            "δ = deployed coherent log-error (mean over held-out val sims "
                            "of the slice); C = diag SIGMA_FRAC². Each z-slice gets its "
                            "OWN Fisher (NOT summed across z — that would over-stiffen F "
                            "and inflate bias/σ by ~√N_z, breaking comparability to the "
                            "+0.029σ±0.160σ headline).",
            "headline_comparison": "ap_full_z3 vs ap_masked_z3 = z=3.0 slice, all-k vs "
                                   "k≥1e-3. ap_full_z3 reproduces the gate's z=3.0 number.",
            "z_region_decomposition": "ap_perz_full / ap_perz_kmasked = A_p bias at each "
                                      "z-slice; region_ap_rms = RMS-over-z of |A_p bias| "
                                      "per region (in-range core, z>4.6, z<2.2, z≈3.4–3.8).",
            "sigma_frac": SIGMA_FRAC,
        },
        "perfold": perfold,
        "summary": summ,
        "coherent_tilt_fold0": tilt,
        "cv_floor_per_band": {f"{lo}-{hi}": v for (lo, hi), v in (cv or {}).items()},
        "config": {"deployed": DEPLOYED, "n_folds": args.folds, "z_fid": Z_FID},
    }
    jpath = f"{OUT}/datarange_mask.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jpath}")

    # ---- console summary ----
    print("\n===== KEY RESULT: A_p Fisher-bias at z=3.0, FULL k vs k≥1e-3 (mean±scatter over folds) =====")
    print("  (apples-to-apples with the +0.029σ±0.160σ headline; only the 2 k<1e-3 bins dropped)")
    print(f"  A_p  FULL k : {summ['ap_full_z3']['mean']:+.3f}σ ± {summ['ap_full_z3']['std']:.3f}σ  "
          f"[{summ['ap_full_z3']['min']:+.3f}, {summ['ap_full_z3']['max']:+.3f}]")
    print(f"  A_p  k≥1e-3 : {summ['ap_masked_z3']['mean']:+.3f}σ ± {summ['ap_masked_z3']['std']:.3f}σ  "
          f"[{summ['ap_masked_z3']['min']:+.3f}, {summ['ap_masked_z3']['max']:+.3f}]")
    print(f"  |Δ A_p| from dropping k<1e-3: {summ['ap_klow_shift_z3']['mean']:+.3f}σ ± "
          f"{summ['ap_klow_shift_z3']['std']:.3f}σ")
    print(f"  n_s  FULL k : {summ['ns_full_z3']['mean']:+.3f}σ ± {summ['ns_full_z3']['std']:.3f}σ")
    print(f"  n_s  k≥1e-3 : {summ['ns_masked_z3']['mean']:+.3f}σ ± {summ['ns_masked_z3']['std']:.3f}σ")

    print("\n===== A_p bias by z-slice (per-slice Fisher, FULL k; mean±scatter over folds) =====")
    for z in z_slices:
        s = summ["ap_perz_full"][f"{z:.1f}"]
        flag = "  in-range" if z in in_range_z else "  OUT-of-range"
        star = " <== named worst" if z in core_z else ""
        print(f"  z={z:.1f}: {s['mean']:+.3f}σ ± {s['std']:.3f}σ"
              f"  [{s['min']:+.3f},{s['max']:+.3f}]{flag}{star}")

    print("\n===== A_p RISK by region (RMS-over-z of |A_p bias|; mean±std over folds) =====")
    for key, lab in (("in_range_fullk", "in-range core (z 2.2–4.6, full k)"),
                     ("core_3.4_3.8_fullk", "  of which z≈3.4–3.8 (full k)  "),
                     ("z_gt_4.6_fullk", "out-of-range z>4.6 (full k)      "),
                     ("z_lt_2.2_fullk", "out-of-range z<2.2 / z=2.0       "),
                     ("in_range_kmasked", "in-range core (k≥1e-3)          "),
                     ("core_3.4_3.8_kmasked", "  of which z≈3.4–3.8 (k≥1e-3)  ")):
        s = summ["region_ap_rms"][key]
        print(f"  {lab}: {s['mean']:.3f}σ ± {s['std']:.3f}σ")

    print("\n===== COHERENT TILT FULL vs MASKED vs CV floor (fold 0; low-k band) =====")
    print("  (CV floor = clean-P1D cosmic-variance proxy ~0.51% at low-k; per-class")
    print("   irreducible floor in brackets — DLA's ~4.8% dwarfs its coherent tilt)")
    for nm in CLS:
        r = tilt[nm]["lowk"]
        cvf = r["cv_floor_frac"]; pcf = r["per_class_floor_frac"]
        vfull = "ABOVE" if r["full_rms"] > cvf else "within"
        vmask = "ABOVE" if r["masked_rms"] > cvf else "within"
        print(f"  {nm:7} low-k: FULL rms={r['full_rms']*100:5.2f}% ({vfull} CV) "
              f"-> MASKED rms={r['masked_rms']*100:5.2f}% ({vmask} CV)   "
              f"CV floor={cvf*100:.2f}% [per-class {pcf*100:.2f}%]  "
              f"(nk {r['full_nk']}->{r['masked_nk']})")


if __name__ == "__main__":
    main()
