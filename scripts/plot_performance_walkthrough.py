#!/usr/bin/env python3
"""Comprehensive INTERMEDIATE-PLOT WALKTHROUGH of the finalized Phase-2b emulator.

Covers BOTH heads of the deployed emulator, reusing either the finalized LOSO
checkpoints (``checkpoints/final_fold{0..7}.eqx``, the FINAL_RECIPE in
``scripts/run_loso_sweep.py``, evaluated on each fold's HELD-OUT sims) or the DEPLOYED
PRODUCTION ENSEMBLE (``checkpoints/final_prod_seed{0..4}.eqx``, trained on ALL sims —
the object the real fit actually uses), plus the multi-fidelity layer
(``hcd_analysis/emulator/multifidelity.py``, the validated rho(k,z)-only default head).
READ-ONLY on the production caches/checkpoints.

THE LOAD-BEARING FRAMING (``--emulator {loso,ensemble}``):
  The production ensemble SAW EVERY SIM, so its pred-vs-true is IN-SAMPLE (a fit-quality /
  convergence check), NOT generalization. We never relabel in-sample ensemble accuracy as
  held-out generalization. Each figure is classified and treated accordingly:

  * DEPLOYED-OBJECT PROPERTY figures (B3 cosmology_response, B4 theta_tracking,
    B6 mf_high_k, B7 delta_c_templates, A5 alpha_prior_sanity) — these are emulator
    responses/templates the real fit LITERALLY uses, so the ensemble IS the correct
    object. In ``--emulator ensemble`` they are regenerated FROM the production ensemble
    and labelled "deployed production ensemble (all-sims)".
  * ACCURACY / pred-vs-true figures (B1 pred_vs_true_p1d, B2 deployed_frac_err_vs_k,
    A1 dndx, A2 cddf, A3 wc_ptierp) — in ``--emulator ensemble`` we OVERLAY both: the
    LOSO held-out curve (solid, the real accuracy claim) AND the in-sample ensemble curve
    (dashed, the deployed-fit reference), clearly labelled. The held-out LOSO is never
    dropped.
  * INHERENTLY-LOSO generalization figures (B5 fisher_bias_perfold, A4 head_a_error_heatmap)
    — KEPT as LOSO held-out; the ensemble has no held-out by construction, so we do NOT
    fake an ensemble version. Relabelled "LOSO held-out generalization (companion)".

  ``--emulator loso`` (DEFAULT) is byte-identical to the original LOSO walkthrough.

Deployed combination (mirrored EXACTLY from the real fit, see ``predict.predict_P_filt`` +
``ensemble.EnsembleEmulator`` + ``run_prod_sbc_shard.py``/``build_legb_ctx`` ensemble_ckpts):
the ensemble prediction is the MEAN over the 5 members of the reconstructed (post-exp)
LINEAR P_filt — the mean is taken AFTER exp because P_filt is exp-nonlinear, and all
members share one norm. For Head-A (f_nhi/dN/dX) we mean the per-member PHYSICAL outputs.

  HEAD B (P1D / cosmology), all 8 finalized folds on their held-out sims:
    B1  pred_vs_true_p1d        per-class P1D(k), log-log + frac-resid subpanel (data-range marked)
    B2  deployed_frac_err_vs_k  per-class deployed |frac err| vs k (in-range) + CV floor overlay
    B3  cosmology_response      sigma_cosmo*d r_hat / d theta vs k for the 9 params (jax.jacfwd)
    B4  theta_tracking          within-(z,tau0)-cell theta-tracking + honest (a)/(b) decomposition
    B5  fisher_bias_perfold     per-fold A_p & n_s Fisher-bias (in-range), the 8-fold spread
    B6  mf_high_k               MF vs LF-extrapolated vs HF-standalone error vs k (KODIAQ band)
    B7  delta_c_templates       the Delta_c HCD templates per class vs k / z

  HEAD A (dN/dX + CDDF), all 8 folds held-out:
    A1  dndx_pred_vs_true       per-class dN/dX vs z (held-out)
    A2  cddf_pred_vs_true       f_NHI / CDDF vs logNHI (class boundaries 17.2/19.0/20.3, shot tail >=21.5)
    A3  wc_ptierp_coupling      emulated-dN/dX w_c vs true + P_tier_p faithfulness (<=2.5% target)
    A4  head_a_error_heatmap    per-class / z / fold Head-A error heatmap
    A5  alpha_prior_sanity      dN/dX -> alpha_c prior center sanity (emulated vs cache CDDF integral)

Each figure is >=150 dpi, titled, axis-labeled. A JSON of headline numbers is written
alongside. The MF figure (B6) trains the small rho(k,z)-only fixed-mean head per HF-LOSO
fold (fast); everything else is pure evaluation of the frozen finalized models.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_performance_walkthrough.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # CPU-only; silence CUDA probe
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
# Publication styling (applied to every figure; numbers/classification untouched)
# ---------------------------------------------------------------------------- #
DPI = 200
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.7",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
})

# Encoding glyphs reused in every accuracy-figure caption so the
# true / held-out / in-sample mapping stays identical across the set.
_ENC_OVERLAY = ("solid = true   ·   dashed = LOSO held-out (out-of-sample)   ·   "
                "dotted = in-sample ensemble (saw these sims)")


def _caption(fig, text, y=0.005):
    """One terse encoding/units caption pinned to the figure foot (not the title)."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color="0.30")

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log,
    reconstruct_P_filt, untransform_prediction, datarange_mask,
    COARSE_NAMES, DATA_RANGE,
)
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.dndx_wc import w_c_corrected, alpha_to_dndx
from hcd_analysis.emulator.model import structural_tier_p
from hcd_analysis.emulator.ensemble import load_ensemble, EnsembleEmulator
from hcd_analysis.emulator.predict import predict_P_filt as _predict_P_filt_jax

# ---------------------------------------------------------------------------- #
# Paths / constants
# ---------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]   # repo root (portable; NB02 reuses W.ROOT)
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
CKPT = str(ROOT / "checkpoints/final_fold{f}")
PROD_CKPT = str(ROOT / "checkpoints/final_prod_seed{k}")   # deployed N-seed ensemble members
N_PROD = 5                                                 # final_prod_seed{0..4}
OUTDIR = ROOT / "figures/analysis/06_performance_walkthrough"
CV_JSON = ROOT / "figures/analysis/04_emulator/diag_lfhf_tilt_and_cv.json"

N_FOLDS = 8
CLS = COARSE_NAMES                       # ("clean","LLS","subDLA","DLA")
HCD = ("LLS", "subDLA", "DLA")
PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback"]
# CDDF class boundaries (log10 N_HI) — LLS/subDLA at 19.0, subDLA/DLA at 20.3,
# the optically-thick onset (LLS lower edge) ~17.2; shot-noise tail starts ~21.5.
NHI_BOUNDS = {"LLS onset 17.2": 17.2, "LLS|subDLA 19.0": 19.0, "subDLA|DLA 20.3": 20.3}
NHI_SHOT = 21.5
N_SIGHTLINES = 691200                    # constant per row (diag_head_a)
# DESI-DR1-like diagonal per-mode fractional covariance (matches diag_ap_fisher_bias).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
COLORS = {c: f"C{i}" for i, c in enumerate(CLS)}


# ---------------------------------------------------------------------------- #
# Shared loaders
# ---------------------------------------------------------------------------- #
def load_all_folds():
    """Load the 8 finalized LOSO checkpoints + their held-out splits.

    Returns ``(models, norms, splits)`` keyed by fold, where splits[f] = (tr,va,ho)."""
    models, norms, splits = {}, {}, {}
    for f in range(N_FOLDS):
        m, meta, nrm = T.load_checkpoint(CKPT.format(f=f))
        models[f], norms[f] = m, nrm
        splits[f] = make_splits(load_all_folds._d, f, n_folds=N_FOLDS, holdout_frac=0.15)
    return models, norms, splits


def load_production_ensemble():
    """Load the DEPLOYED production ensemble (the 5 ``final_prod_seed{0..4}`` members).

    Returns ``(ensemble, norm)`` where ``ensemble`` is an ``EnsembleEmulator`` whose
    deployed prediction is the MEAN over members of the reconstructed (post-exp) linear
    P_filt — EXACTLY the combination ``predict.predict_P_filt`` applies when the real fit
    passes ``ensemble_ckpts=`` to ``build_legb_ctx`` (see ``run_prod_sbc_shard.py``). All
    members share one norm (``load_ensemble`` asserts it); ``norm`` is member-0's.

    The production ensemble saw EVERY sim, so anything evaluated through it is IN-SAMPLE
    (a fit-quality check), NOT out-of-sample generalization — callers must label it so.
    """
    # PINNED members (freeze decision 6): the manifest loader verifies sha256 + exact pairing +
    # count + the stray-member tripwire before loading (checkpoints/production_ensemble_manifest.json).
    from hcd_analysis.emulator.prod_ensemble import load_production_ensemble as _load_pinned
    ens, meta, norm, _manifest = _load_pinned(
        checkpoints_dir=os.path.dirname(PROD_CKPT.format(k=0)))
    assert len(ens.members) == N_PROD
    return ens, norm


def _pf_jnp(norm):
    """The (mu_marg, sig_marg, sig_cosmo) P_filt norm as a jnp dict for predict_P_filt."""
    return {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}


def predict_P(model, d, idx, norm):
    """Reconstructed LINEAR P_filt (n,4,K) on rows ``idx``, for a SINGLE ``Emulator``.

    Single-model path (LOSO fold); for the ensemble use ``predict_P_ens``. Also returns
    the raw two-head dict (base/resid/f_nhi/dndx/delta) for downstream diagnostics."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pred = jax.vmap(model)(x, tau0)
    P = reconstruct_P_filt(np.asarray(pred["P_filt_base"]),
                           np.asarray(pred["P_filt_resid"]), norm["P_filt"])
    return P, {k: np.asarray(v) for k, v in pred.items()}


def predict_P_ens(ens, d, idx, norm):
    """Reconstructed LINEAR P_filt (n,4,K) on rows ``idx`` for the production ENSEMBLE.

    Mirrors the deployed combination: mean over members of the POST-EXP reconstructed
    P_filt (``predict.predict_P_filt`` duck-typing on ``.members``). The mean is taken
    after exp because P_filt is exp-nonlinear; this is byte-for-byte the object the real
    fit's likelihood consumes. Returns ``(P, None)`` — the per-member raw two-head dict has
    no single deployed value, so callers needing f_nhi/dndx use ``predict_headA_ens``."""
    pf = _pf_jnp(norm)
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])

    def one(xi, ti):                                  # ensemble-aware mean P_filt (4,K)
        return _predict_P_filt_jax(ens, xi[:9], xi[9], ti, pf)

    P = np.asarray(jax.vmap(one)(x, tau0))            # (n,4,K)
    return P, None


def predict_headA_ens(ens, d, idx, norm):
    """Mean-over-members PHYSICAL Head-A outputs (f_nhi, dndx) for the production ensemble.

    Head-A (CDDF / dN/dX) has no deployed-likelihood ensemble convention (the likelihood
    only consumes P_filt through the ensemble), so for the IN-SAMPLE accuracy figures we
    take the natural mean of the per-member PHYSICAL predictions (the same averaging spirit
    as the deployed P_filt mean). Returns the physical dict {"f_nhi","dndx"}."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    fns, dns = [], []
    for m in ens.members:
        pred = jax.vmap(m)(x, tau0)
        phys = untransform_prediction(
            {"f_nhi": np.asarray(pred["f_nhi"]), "dndx": np.asarray(pred["dndx"])}, norm)
        fns.append(phys["f_nhi"]); dns.append(phys["dndx"])
    return {"f_nhi": np.mean(fns, axis=0), "dndx": np.mean(dns, axis=0)}


# Deployed-object labels — kept in one place so every title/caption is consistent.
LBL_ENS = "deployed production ensemble (all-sims, IN-SAMPLE)"
LBL_LOSO = "LOSO held-out generalization"


def load_cv_floor():
    """Per-k-band cosmic-variance floor (fractional) from the CV diagnostic JSON."""
    if not CV_JSON.exists():
        return None
    cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
    out = {}
    for band, v in cvj.items():
        lo, hi = (float(x) for x in band.split("-"))
        out[(lo, hi)] = float(v["median_pct"]) / 100.0
    return out


# ============================================================================ #
# HEAD B
# ============================================================================ #
def _pick_class_rows(d, idx_pool, ci, n=3):
    cnt = d["coarse_counts"][idx_pool, ci]
    return idx_pool[np.argsort(-cnt)][:n]


def figB1_pred_vs_true_p1d(models, norms, splits, d, kf, ens=None, ens_norm=None):
    """B1: pred-vs-true per-class P1D, log-log + frac-resid subpanel.

    Uses fold-0's held-out sims (3 best-sampled rows per class). The DESI data range
    (k>=1e-3) is shaded out below k_min; the per-row Nyquist edge is the right turnover.

    ACCURACY figure (classification: pred-vs-true). When ``ens`` is given the SAME rows are
    ALSO predicted by the deployed production ensemble and OVERLAID: solid=true,
    dashed=LOSO held-out pred (the real accuracy claim), dotted=in-sample ensemble pred (the
    deployed-fit reference, evaluated on rows it SAW in training). The held-out residual
    RMS is the headline number; the ensemble's is reported separately as in-sample."""
    model, norm = models[0], norms[0]
    _, va, _ = splits[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08,
                                          "wspace": 0.06})
    fr_rms = {}; fr_rms_ens = {}
    for ci, nm in enumerate(CLS):
        rows = _pick_class_rows(d, va, ci, n=3)
        P_pred, _ = predict_P(model, d, rows, norm)
        P_true = d["P_filt"][rows]
        P_ens = predict_P_ens(ens, d, rows, ens_norm)[0] if ens is not None else None
        ax = axes[0, ci]; axr = axes[1, ci]
        accum = []; accum_e = []
        for ri in range(len(rows)):
            pt = P_true[ri, ci]; pp = P_pred[ri, ci]
            ok = np.isfinite(pt) & (pt > 0) & np.isfinite(pp) & (pp > 0)
            ax.loglog(kf[ok], pt[ok], "-", color=f"C{ri}", alpha=0.85,
                      label="true" if (ci == 0 and ri == 0) else None)
            ax.loglog(kf[ok], pp[ok], "--", color=f"C{ri}", alpha=0.85,
                      label="LOSO held-out pred" if (ci == 0 and ri == 0) else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = pp[ok] / pt[ok] - 1.0
            axr.semilogx(kf[ok], fr, "-", color=f"C{ri}", alpha=0.85)
            accum.append(fr[np.isfinite(fr)])
            if P_ens is not None:
                pe = P_ens[ri, ci]
                oke = np.isfinite(pt) & (pt > 0) & np.isfinite(pe) & (pe > 0)
                ax.loglog(kf[oke], pe[oke], ":", color=f"C{ri}", alpha=0.85, lw=1.4,
                          label="in-sample ensemble pred" if (ci == 0 and ri == 0) else None)
                with np.errstate(invalid="ignore", divide="ignore"):
                    fre = pe[oke] / pt[oke] - 1.0
                axr.semilogx(kf[oke], fre, ":", color=f"C{ri}", alpha=0.85, lw=1.4)
                accum_e.append(fre[np.isfinite(fre)])
        fr_rms[nm] = float(np.sqrt(np.mean(np.concatenate(accum) ** 2))) if accum else np.nan
        if accum_e:
            fr_rms_ens[nm] = float(np.sqrt(np.mean(np.concatenate(accum_e) ** 2)))
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        axr.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ax.set_title(nm, fontweight="bold"); ax.grid(alpha=0.3, which="both")
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_ylim(-0.3, 0.3); axr.set_xlabel(r"$k$  [s km$^{-1}$]")
        if ci == 0:
            ax.set_ylabel(r"$P_{\rm 1D}^{\rm filt}$  [km s$^{-1}$]")
            axr.set_ylabel("pred/true − 1")
            ax.legend(loc="lower left", fontsize=8.5)
    cap = (_ENC_OVERLAY if ens is not None else
           "solid = true   ·   dashed = LOSO held-out pred   ·   subpanel = pred/true − 1")
    cap += r";   grey band = below data $k_{\min}=10^{-3}$ s km$^{-1}$"
    fig.suptitle("B1 — Predicted vs true per-class P1D (fold-0 held-out sims)",
                 fontsize=13, fontweight="bold")
    _caption(fig, cap)
    p = OUTDIR / "B1_pred_vs_true_p1d.png"
    fig.tight_layout(rect=(0, 0.035, 1, 0.97)); fig.savefig(p); plt.close(fig)
    return p, fr_rms, fr_rms_ens


def _deployed_frac_in_range(models, norms, splits, d, ens=None, ens_norm=None):
    """Stack deployed |P_pred/P_true-1| over ALL folds' held-out rows, restricted to
    the DESI data range (z in [2.2,4.6], k>=k_min). Returns (frac (N,4,K), kf).

    When ``ens`` is given, predict the SAME row population with the production ENSEMBLE
    (using its OWN ``ens_norm``) instead of the per-fold LOSO net — i.e. IN-SAMPLE accuracy
    on the identical rows (the ensemble saw them in training). Same masking, so the only
    difference is the emulator."""
    fr, kfs = [], None
    for f in range(N_FOLDS):
        _, va, _ = splits[f]
        if ens is None:
            P_pred, _ = predict_P(models[f], d, va, norms[f])
        else:
            P_pred, _ = predict_P_ens(ens, d, va, ens_norm)
        P_true = d["P_filt"][va]
        mask_k = d["mask"][va]
        in_range = datarange_mask(d)[va]
        keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true > 0)
                & mask_k[:, None, :] & in_range[:, None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(P_pred / P_true - 1.0)
        fr.append(np.where(keep, r, np.nan))
        if kfs is None:
            kfs = d["kfkms"][va][0]
    return np.concatenate(fr, 0), kfs


def figB2_deployed_frac_err_vs_k(frac, kf, cv_floor, frac_ens=None):
    """B2: per-class deployed |frac err| vs k (in-range, all folds) + CV floor.

    ACCURACY figure. ``frac`` = the LOSO HELD-OUT |frac err| (the real accuracy claim,
    solid median + IQR). When ``frac_ens`` (the production ensemble's IN-SAMPLE |frac err|
    over the SAME rows) is given it is OVERLAID as a dashed median line per class — the
    deployed-fit reference, NOT a generalization claim. Returns (path, rms, rms_ens)."""
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    rms = {}; rms_ens = {}
    for ci, nm in enumerate(CLS):
        med = np.nanmedian(frac[:, ci, :], axis=0)
        q1 = np.nanpercentile(frac[:, ci, :], 25, axis=0)
        q3 = np.nanpercentile(frac[:, ci, :], 75, axis=0)
        ok = np.isfinite(med)
        ax.fill_between(kf[ok], q1[ok], q3[ok], color=COLORS[nm], alpha=0.13)
        r = float(np.sqrt(np.nanmean(frac[:, ci, :] ** 2)))
        rms[nm] = r
        lab = (f"{nm} — held-out (RMS {r*100:.1f}%)" if frac_ens is not None
               else f"{nm} (RMS {r*100:.1f}%)")
        ax.loglog(kf[ok], med[ok], color=COLORS[nm], lw=2.0, label=lab)
        if frac_ens is not None:
            mede = np.nanmedian(frac_ens[:, ci, :], axis=0)
            oke = np.isfinite(mede)
            re = float(np.sqrt(np.nanmean(frac_ens[:, ci, :] ** 2)))
            rms_ens[nm] = re
            ax.loglog(kf[oke], mede[oke], color=COLORS[nm], lw=1.4, ls="--",
                      label=f"{nm} — in-sample ens. (RMS {re*100:.1f}%)")
    if cv_floor is not None:
        first = True
        for (lo, hi), v in cv_floor.items():
            ax.hlines(v, max(lo, kf.min()), hi, color="k", ls=":", lw=1.4,
                      label="cosmic-variance floor (clean)" if first else None)
            first = False
    ax.axvline(DATA_RANGE["k_min"], color="grey", ls=":", lw=1)
    ax.set_xlabel(r"$k$  [s km$^{-1}$]")
    ax.set_ylabel(r"$|{\rm pred/true}-1|$   (median; IQR shaded = held-out)")
    if frac_ens is not None:
        ttl = "B2 — Per-class fractional P1D error vs k (in-range, 8 folds)"
        cap = ("solid = LOSO held-out (out-of-sample)   ·   dashed = in-sample ensemble "
               "(same rows)   ·   dotted = CV floor")
    else:
        ttl = "B2 — Deployed per-class fractional P1D error vs k (in-range, 8 folds)"
        cap = "dotted black = cosmic-variance floor (irreducible sampling scatter)"
    ax.set_title(ttl, fontsize=13, fontweight="bold")
    # legend outside-right so it never sits on the curves
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              borderaxespad=0.0)
    ax.grid(alpha=0.3, which="both")
    _caption(fig, cap)
    p = OUTDIR / "B2_deployed_frac_err_vs_k.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(p); plt.close(fig)
    return p, rms, rms_ens


def figB3_cosmology_response(models, norms, splits, d, kf, ens=None, ens_norm=None):
    """B3: ∂lnP/∂θ vs k for the 9 params (jax.jacfwd) — the DEPLOYED cosmology response.

    DEPLOYED-OBJECT PROPERTY figure: this Jacobian is the differentiable readout the real
    fit's HMC consumes, so in ``--emulator ensemble`` it is computed FROM the production
    ensemble (the object the real fit literally uses). For the single-model (LOSO) path it
    is σ_cosmo·∂r̂/∂θ on fold-0; for the ensemble it is ∂ln(mean_m P_filt)/∂θ, the response
    of the deployed mean prediction (the σ_cosmo·∂r̂ identity holds per-member, but the
    ensemble combines post-exp, so we differentiate the deployed log mean P_filt directly)."""
    deployed = ens is not None
    norm = ens_norm if deployed else norms[0]
    pf = _pf_jnp(norm)
    sig_cosmo = jnp.asarray(norm["P_filt"]["sig_cosmo"])     # (4,K)
    _, va, _ = splits[0]
    r0 = va[len(va) // 2]
    x0 = jnp.asarray(d["x"][r0]); tau0 = float(d["tau0"][r0])
    z_r0 = float(d["z_grid"][r0])

    if deployed:
        def logP_of_x(x):                                    # d ln(mean_m P_filt) / d x
            return jnp.log(_predict_P_filt_jax(ens, x[:9], x[9], tau0, pf))
    else:
        model = models[0]

        def logP_of_x(x):
            return sig_cosmo * model(x, tau0)["P_filt_resid"]    # d logP_hat / d x

    J = np.asarray(jax.jacfwd(logP_of_x)(x0))                # (4,K,10)

    # display labels for the 9 cosmology/astro params (kept 1:1 with PARAM_NAMES)
    PAR_TEX = {"ns": r"$n_s$", "Ap": r"$A_p$", "herei": "HeII reion. (early)",
               "heref": "HeII reion. (late)", "alphaq": r"$\alpha_q$ (quasar)",
               "hub": r"$h$", "omegamh2": r"$\Omega_m h^2$",
               "hireionz": r"$z_{\rm HI\,reion}$", "bhfeedback": "BH feedback"}
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharex=True)
    axes = axes.ravel()
    constr = {}
    for j, pp in enumerate(PARAM_NAMES):
        ax = axes[j]
        # response amplitude (RMS over class & k, in-range) — "how much P1D moves"
        with np.errstate(invalid="ignore"):
            amp = np.sqrt(np.nanmean(J[:, kf >= DATA_RANGE["k_min"], j] ** 2))
        constr[pp] = float(amp)
        for ci, nm in enumerate(CLS):
            ax.semilogx(kf, J[ci, :, j], color=COLORS[nm], label=nm if j == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ax.set_title(rf"{PAR_TEX[pp]}    $|\partial\ln P/\partial\theta|_{{\rm rms}}={amp:.3f}$",
                     fontsize=11)
        ax.grid(alpha=0.3, which="both")
        if j % 3 == 0:
            ax.set_ylabel(r"$\partial \ln P / \partial \theta_{\rm unit}$")
        if j >= 6:
            ax.set_xlabel(r"$k$  [s km$^{-1}$]")
        if j == 0:
            ax.legend(loc="upper left", fontsize=9, ncol=2)
    src = LBL_ENS if deployed else "fold-0 (LOSO single net)"
    fig.suptitle(r"B3 — Deployed cosmology response $\partial\ln P/\partial\theta$ vs $k$"
                 f"  ({src}; z={z_r0:.2f}, τ₀={tau0:.2f})", fontsize=13, fontweight="bold")
    _caption(fig, "jax.jacfwd;  larger |response| = tighter constraint;  "
                  "sign-flips across k = shape (tilt) sensitivity")
    p = OUTDIR / "B3_cosmology_response.png"
    fig.tight_layout(rect=(0, 0.025, 1, 0.97)); fig.savefig(p); plt.close(fig)
    return p, constr


def _honest_decomp(model, norm, d, va, ens=None):
    """Honest within-cell theta-tracking + (a) residual-fit / (b) baseline-misfit
    decomposition (port of diag_theta_tracking_honest, on ONE fold). Returns a dict
    of arrays for plotting + per-class metrics.

    When ``ens`` is given (the deployed production ensemble), the deployed prediction is
    the MEAN over members of the POST-EXP P_filt, so we set ``lp = log(mean_m P_filt_m)``
    (the actual deployed log prediction) and the baseline ``m_hat`` = the MEAN over members
    of the per-member baseline ``base·σ_marg+μ_marg``; the effective residual is then
    ``(lp − m_hat)/σ_cosmo``. This keeps the (a)/(b) algebra exact for the deployed object."""
    pf = norm["P_filt"]
    sig_marg, mu_marg, sig_cosmo = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    b = make_batch(d, va, norm)
    if ens is None:
        pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
        base = np.asarray(pred["P_filt_base"])
        resid = np.asarray(pred["P_filt_resid"])
        m_hat = base * sig_marg + mu_marg                    # deployed baseline logP
        lp = m_hat + sig_cosmo * resid                       # deployed pred logP
    else:
        # ensemble: baseline = mean of per-member baselines; pred = log(mean post-exp P_filt)
        bb = []
        for m in ens.members:
            pm = jax.vmap(m)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
            bb.append(np.asarray(pm["P_filt_base"]))
        base = np.mean(bb, axis=0)
        m_hat = base * sig_marg + mu_marg
        pf_j = _pf_jnp(norm)
        x = jnp.asarray(d["x"][va]); tau0 = jnp.asarray(d["tau0"][va])
        lp = np.log(np.asarray(jax.vmap(
            lambda xi, ti: _predict_P_filt_jax(ens, xi[:9], xi[9], ti, pf_j))(x, tau0)))
        resid = (lp - m_hat) / sig_cosmo                     # effective deployed residual
    lt = safe_log(d["P_filt"][va])                           # true logP
    cv = cell_id(d, va)
    m_cell = np.stack([pf["cell_mean"].get(int(c), mu_marg) for c in cv])

    r_pred = resid
    r_true = (lt - m_hat) / sig_cosmo
    t_resid = (lt - m_cell) / sig_cosmo
    err_resid = r_pred - t_resid                             # (a)
    err_base = (m_hat - m_cell) / sig_cosmo                  # (b)
    err_tot = r_pred - r_true                                # (a)+(b)

    dev_t_hat = lt - m_hat                                   # honest (m_hat ref)
    dev_p_hat = lp - m_hat
    keep = np.isin(cv, [cl for cl in np.unique(cv) if (cv == cl).sum() >= 2])

    def _wrms(x):
        g = np.isfinite(x)
        return float(np.sqrt(np.nanmean(x[g] ** 2)))

    rows = {}
    for ci, nm in enumerate(CLS):
        dt = dev_t_hat[keep, ci, :].ravel(); dp = dev_p_hat[keep, ci, :].ravel()
        g = np.isfinite(dt) & np.isfinite(dp)
        corr = float(np.corrcoef(dt[g], dp[g])[0, 1]) if g.sum() > 3 else np.nan
        st = np.nanstd(r_true[:, ci, :][np.isfinite(r_true[:, ci, :])])
        rows[nm] = dict(honest_corr=corr,
                        a_resid=_wrms(err_resid[:, ci, :]) / st,
                        b_base=_wrms(err_base[:, ci, :]) / st,
                        tot=_wrms(err_tot[:, ci, :]) / st,
                        fracP=_wrms(np.exp(lp[:, ci, :] - lt[:, ci, :]) - 1.0))
    return dict(dev_t=dev_t_hat[keep], dev_p=dev_p_hat[keep], rows=rows,
                err_resid=err_resid, err_base=err_base, err_tot=err_tot, sig_cosmo=sig_cosmo)


def figB4_theta_tracking(models, norms, splits, d, ens=None, ens_norm=None):
    """B4: within-(z,tau0)-cell theta-tracking (honest, deployed m_hat ref) + the
    (a) residual-fit vs (b) baseline-misfit whitened-error decomposition.

    DEPLOYED-OBJECT PROPERTY figure: the θ-tracking of the object the real fit uses. In
    ``--emulator ensemble`` the decomposition runs on the production ENSEMBLE (in-sample on
    fold-0 sims, since the ensemble has no held-out); in ``--emulator loso`` it is fold-0's
    held-out single net."""
    deployed = ens is not None
    norm = ens_norm if deployed else norms[0]
    model = None if deployed else models[0]
    _, va, _ = splits[0]
    dec = _honest_decomp(model, norm, d, va, ens=ens)

    fig = plt.figure(figsize=(15, 5.8))
    # left: honest hexbin (all classes pooled)
    ax0 = fig.add_subplot(1, 2, 1)
    dt = dec["dev_t"].ravel(); dp = dec["dev_p"].ravel()
    g = np.isfinite(dt) & np.isfinite(dp)
    hb = ax0.hexbin(dt[g], dp[g], gridsize=70, mincnt=1, cmap="viridis")
    lim = np.nanpercentile(np.abs(dt[g]), 99)
    corr = float(np.corrcoef(dt[g], dp[g])[0, 1])
    ratio = float(np.nanstd(dp[g]) / np.nanstd(dt[g]))
    ax0.plot([-lim, lim], [-lim, lim], "r--", lw=1.2, label="$y=x$")
    ax0.set_xlim(-lim, lim); ax0.set_ylim(-lim, lim)
    ax0.set_xlabel(r"TRUE log-P1D deviation from $\hat{m}(z,\tau_0)$  [cosmology signal]")
    ax0.set_ylabel(r"PRED deviation from $\hat{m}$")
    ax0.set_title(rf"Honest $\theta$-tracking (deployed $\hat{{m}}$ ref):  "
                  rf"corr = {corr:.3f},  spread-ratio = {ratio:.3f}")
    ax0.legend(loc="upper left"); fig.colorbar(hb, ax=ax0, label="count")

    # right: stacked (a)/(b) decomposition bars per class
    ax1 = fig.add_subplot(1, 2, 2)
    rows = dec["rows"]
    xpos = np.arange(len(CLS))
    a = np.array([rows[c]["a_resid"] for c in CLS])
    bb = np.array([rows[c]["b_base"] for c in CLS])
    tot = np.array([rows[c]["tot"] for c in CLS])
    ax1.bar(xpos, a, color="C2", label="(a) residual-head fit error")
    ax1.bar(xpos, bb, bottom=a, color="C7",
            label=r"(b) baseline mis-fit ($\sigma_{\rm cosmo}$-amplified)")
    ax1.plot(xpos, tot, "kD", ms=7, label=r"deployed total / $\sigma_{\rm signal}$")
    for i, c in enumerate(CLS):
        ax1.text(i, tot[i] + 0.02, rf"{rows[c]['fracP']*100:.1f}%" + "\n" + r"$|\hat P/P-1|$",
                 ha="center", fontsize=7.5, color="navy")
    ax1.set_xticks(xpos); ax1.set_xticklabels(CLS)
    ax1.set_ylabel(r"whitened RMS error  /  $\sigma_{\rm signal}$")
    ax1.set_title("Error decomposition: (a) residual fit + (b) baseline mis-fit")
    ax1.legend(loc="upper left", fontsize=8.5); ax1.grid(alpha=0.3, axis="y")
    src = LBL_ENS + ", fold-0 sims" if deployed else "fold-0 LOSO held-out"
    fig.suptitle(r"B4 — Within-($z,\tau_0$)-cell $\theta$-tracking + honest (a)/(b) error "
                 f"decomposition  ({src})", fontsize=13, fontweight="bold")
    _caption(fig, "whitened deployed cosmology-signal error (NOT the inflated val-mean metric)")
    p = OUTDIR / "B4_theta_tracking.png"
    fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(p); plt.close(fig)
    return p, {"honest_corr": corr, "spread_ratio": ratio,
               "per_class": {c: rows[c] for c in CLS}}


def _fisher_bias_fold(model, norm, d, va, z_fid=3.0):
    """9-param Fisher-bias of the deployed model (port of diag_ap_fisher_bias.fisher_bias,
    in-range modes, single fold). Returns the per-param bias in sigma units."""
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]; n_k = sc.shape[1]
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - z_fid))], atol=1e-6)
    tau0_fid = float(np.median(d["tau0"][zsel]))
    z_unit_fid = float(np.median(d["x"][zsel, 9]))
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    mask_va = np.isfinite(safe_log(d["P_filt"][va]))
    in_range = datarange_mask(d)[va]                          # (n,K)
    kvalid = (mask_va[vz].mean(0) > 0.5) & (in_range[vz].mean(0) > 0.5)[None, :]

    z_j = jnp.asarray(z_unit_fid)

    def rhat(theta9):
        x = jnp.concatenate([theta9, z_j[None]])
        return model(x, jnp.asarray(tau0_fid))["P_filt_resid"]
    p0 = jnp.asarray(np.full(9, 0.5), dtype=jnp.float64)
    Jr = np.asarray(jax.jacfwd(rhat)(p0))                     # (4,n_k,9)
    J_logP = Jr * sc[:, :, None]

    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])
    logP_true = safe_log(d["P_filt"][va][vz])
    m_slice = np.isfinite(logP_true)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j],
                                               (logP_hat - logP_true)[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])

    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
               or not np.isfinite(delta[ci, j]):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    J9 = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    F = (J9.T * Cinv) @ J9
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J9.T * Cinv) @ dv)
    bias = dtheta / sigma
    return {PARAM_NAMES[i]: float(bias[i]) for i in range(9)}, int(len(dv))


def figB5_fisher_bias_perfold(models, norms, splits, d):
    """B5: per-fold A_p & n_s Fisher-bias (in-range) — the 8-fold spread.

    INHERENTLY-LOSO generalization figure (companion validation): this is the per-fold
    HELD-OUT bias spread, which only exists for the LOSO nets — the production ensemble has
    NO held-out by construction (it saw every sim), so we do NOT fake an ensemble version.
    It is ALWAYS the 8 LOSO folds regardless of ``--emulator``."""
    biases = {}
    for f in range(N_FOLDS):
        _, va, _ = splits[f]
        b, nm = _fisher_bias_fold(models[f], norms[f], d, va)
        biases[f] = b
    folds = sorted(biases)
    ap = np.array([biases[f]["Ap"] for f in folds])
    ns = np.array([biases[f]["ns"] for f in folds])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, lab, col in ((axes[0], ap, r"$A_p$", "C3"), (axes[1], ns, r"$n_s$", "C0")):
        ax.bar(folds, vals, color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-0.2, 0.2, color="green", alpha=0.12, label=r"$\pm0.2\sigma$ gate")
        rms = float(np.sqrt(np.mean(vals ** 2)))
        ax.axhline(rms, color="k", ls=":", lw=1)
        ax.axhline(-rms, color="k", ls=":", lw=1, label=rf"RMS = {rms:.3f}$\sigma$")
        ax.set_xlabel("LOSO fold"); ax.set_ylabel(rf"{lab} Fisher-bias  [$\sigma$]")
        ax.set_title(rf"{lab}: per-fold deployed Fisher-bias (in-range, $z=3$)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
        lim = max(0.3, 1.15 * np.abs(vals).max())
        ax.set_ylim(-lim, lim)
    fig.suptitle(r"B5 — Per-fold $A_p$ & $n_s$ Fisher-bias spread "
                 "(8 LOSO held-out folds)", fontsize=13, fontweight="bold")
    _caption(fig, r"DESI-DR1-like diagonal covariance;  inference gate $|{\rm bias}|<0.2\sigma$;  "
                  "the ensemble has no held-out by construction")
    p = OUTDIR / "B5_fisher_bias_perfold.png"
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(p); plt.close(fig)
    out = {"per_fold": {str(f): biases[f] for f in folds},
           "Ap_rms_sigma": float(np.sqrt(np.mean(ap ** 2))),
           "ns_rms_sigma": float(np.sqrt(np.mean(ns ** 2))),
           "Ap_max_abs": float(np.abs(ap).max()), "ns_max_abs": float(np.abs(ns).max())}
    return p, out


def figB6_mf_high_k(d_lf):
    """B6: MF (rho(k,z)-only default) vs LF-extrapolated-alone vs HF-standalone RMS
    frac error vs k, aggregated over HF-LOSO folds, with the KODIAQ band marked.

    Uses the validated rho(k,z)-only FixedMeanHead (the default MF; the learned-MLP
    delta-head is the ablation). Returns (path, summary) or (None, reason) if the HR
    cache / res_corr table is unavailable."""
    try:
        from hcd_analysis.emulator import multifidelity as MF
    except Exception as e:                                    # pragma: no cover
        return None, f"MF import failed: {e!r}"
    try:
        lf = d_lf
        hr = load_cache(MF.HR_CACHE)
        lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=0)
        pairs = MF.match_hr_to_lf(lf, hr)
        k_eval, eval_logk = MF.build_eval_grid(hr, k_max=MF.KODIAQ_KMAX, n_k=48)
        tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk,
                                      pairs, n_tail=6)
        from scripts.build_mf_delta import (measure_hf_abs_targets,
                                            hf_standalone_predict, frac_err_vs_k)
        hf_abs = measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    except Exception as e:                                    # pragma: no cover
        return None, f"MF setup failed: {e!r}"

    from hcd_analysis.emulator.data import Z_LIMITS
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    log_rho = MF.mean_log_ratio_rho(tg, eval_logk)
    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=4))

    per_sim = {}
    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]; eval_rows = np.where(held)[0]
        # DEFAULT MF: rho(k,z)-only FixedMeanHead (NO learned theta head) built from TRAIN rows.
        head = MF.build_default_head(tg, log_rho, train_mask_rows=train_rows, n_basis=4)
        mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, head,
                                    eval_logk=eval_logk, log_rho=log_rho, n_basis=4)
        xs = jnp.asarray(tg["x"][eval_rows]); tts = jnp.asarray(tg["tau0"][eval_rows])
        logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, tts))
        logP_lf = np.asarray(jax.vmap(lambda x, t: mf.lf_logP(x, t)
                                      + jnp.log(mf.res_corr(
                                          x[..., 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]))[None, :]
                                      )(xs, tts))
        logP_hf = hf_standalone_predict(hf_abs, basis, eval_logk, train_rows, eval_rows,
                                        n_basis=4, width=16, n_layers=1, lr=3e-3,
                                        epochs=400, coeff_l2=1e-1, seed=0)
        logP_true = hf_abs["logP"][eval_rows]
        per_sim[s] = {"mf": frac_err_vs_k(logP_mf, logP_true),
                      "lf": frac_err_vs_k(logP_lf, logP_true),
                      "hf": frac_err_vs_k(logP_hf, logP_true)}

    def agg(key):                                            # RMS over HF-LOSO folds (4,K)
        return np.sqrt(np.nanmean(np.stack([per_sim[s][key] for s in sims]) ** 2, axis=0))
    mf_a, lf_a, hf_a = agg("mf"), agg("lf"), agg("hf")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharex=True, sharey=True)
    for ci, nm in enumerate(CLS):
        ax = axes[ci]
        ax.loglog(k_eval, mf_a[ci], "-", color="C0", lw=2.2, label=r"MF ($\rho(k,z)$-only)")
        ax.loglog(k_eval, lf_a[ci], "--", color="C3", lw=1.6, label="LF-extrapolated")
        ax.loglog(k_eval, hf_a[ci], ":", color="C2", lw=1.6, label="HF-standalone (6 sims)")
        ax.axvspan(MF.KODIAQ_BAND[0], MF.KODIAQ_BAND[1], color="grey", alpha=0.13)
        ax.axvline(0.069, color="k", ls="-.", lw=0.8, alpha=0.6)
        ax.set_title(nm, fontweight="bold"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$k$  [s km$^{-1}$]")
        if ci == 0:
            ax.set_ylabel("RMS fractional P1D error"); ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("B6 — Multi-fidelity high-k: MF vs LF-extrapolated vs HF-standalone "
                 "(HF-LOSO, RMS over held-out HF sims)", fontsize=13, fontweight="bold")
    _caption(fig, "shaded = KODIAQ band 0.07–0.2 s km$^{-1}$;  "
                  "dash-dot = LF Nyquist ~0.069 s km$^{-1}$")
    p = OUTDIR / "B6_mf_high_k.png"
    fig.tight_layout(rect=(0, 0.05, 1, 0.94)); fig.savefig(p); plt.close(fig)

    band = (k_eval >= MF.KODIAQ_BAND[0]) & (k_eval <= MF.KODIAQ_BAND[1])
    summ = {nm: {"mf_kodiaq": float(np.sqrt(np.nanmean(mf_a[ci, band] ** 2))),
                 "lf_kodiaq": float(np.sqrt(np.nanmean(lf_a[ci, band] ** 2))),
                 "hf_kodiaq": float(np.sqrt(np.nanmean(hf_a[ci, band] ** 2)))}
            for ci, nm in enumerate(CLS)}
    return p, summ


def figB7_delta_c_templates(models, norms, splits, d, kf, ens=None, ens_norm=None):
    """B7: the CORRECTED per-class HCD excess templates R_c = P_c − P_clean, across z.

    DEPLOYED-OBJECT PROPERTY figure: these ARE the templates the forward model re-weights,
    P_obs = P_clean + Σ_c α_c·R_c, with R_c live-emulated each step (predict_excess). In
    ``--emulator ensemble`` they are emulated from the production ENSEMBLE (predict_excess /
    predict_P_filt are ensemble-aware — they duck-type on ``.members`` and mean post-exp),
    so they ARE the deployed templates. This REPLACES the deprecated HeadB `delta` head
    (P_c^unf − P_c^filt), which gave Δ_LLS≡0. For DLA we overlay the FILTERED excess and the
    UNFILTERED excess actually used (P_filt[DLA]+dla_core−P_clean)."""
    from hcd_analysis.emulator.predict import predict_excess, predict_P_filt
    deployed = ens is not None
    model = ens if deployed else models[0]
    norm = ens_norm if deployed else norms[0]
    pf = norm["P_filt"]
    _, va, _ = splits[0]
    z_all = np.round(d["z_grid"], 3)
    zs = np.unique(z_all[va])
    z_pick = zs[np.linspace(0, len(zs) - 1, 4).astype(int)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharex=True)
    for ci, nm in enumerate(HCD):       # the 3 HCD classes
        ax = axes[ci]
        for zi, zval in enumerate(z_pick):
            rows = va[np.isclose(z_all[va], zval)]
            r0 = rows[len(rows) // 2]
            theta9 = jnp.asarray(d["x"][r0, :9]); z_unit = float(d["x"][r0, 9])
            tau0 = float(d["tau0"][r0]); dla_core = jnp.asarray(d["delta"][r0, 2])
            R = np.asarray(predict_excess(model, theta9, z_unit, tau0, pf, dla_core))  # (3,K)
            ax.plot(kf, R[ci], color=f"C{zi}", lw=1.6, label=f"z = {zval:.2f}")
            if nm == "DLA":             # overlay the filtered (pre-unmask) DLA excess
                P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf)
                R_filt = np.asarray(P_filt[3] - P_filt[0])
                ax.plot(kf, R_filt, color=f"C{zi}", lw=1.0, ls=":", alpha=0.7)
        ax.set_xscale("log")
        # symlog y: the very-low-k spike (out-of-range) AND the in-range tail are both
        # legible; the linthresh keeps the small in-range structure resolved.
        ax.set_yscale("symlog", linthresh=0.05)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(kf.min(), DATA_RANGE["k_min"], color="grey", alpha=0.12)
        ttl = f"{nm}" + ("   (solid = unfiltered/used, dotted = filtered)" if nm == "DLA" else "")
        ax.set_title(ttl, fontweight="bold"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$k$  [s km$^{-1}$]")
        if ci == 0:
            ax.set_ylabel(r"$R_c = P_c - P_{\rm clean}$  (physical; symlog)")
            ax.legend(loc="upper right", fontsize=9)
    src = LBL_ENS if deployed else "fold-0 LOSO single net"
    fig.suptitle(r"B7 — Per-class HCD excess $R_c = P_c - P_{\rm clean}$ across $z$"
                 f"  ({src})", fontsize=13, fontweight="bold")
    _caption(fig, r"forward-model reweight basis: $P_{\rm obs}=P_{\rm clean}+\sum_c \alpha_c R_c$;  "
                  r"LLS now non-zero (old $P_c^{\rm unf}-P_c^{\rm filt}$ gave $\Delta_{\rm LLS}\equiv0$);  "
                  r"grey = below data $k_{\min}$")
    p = OUTDIR / "B7_delta_c_templates.png"
    fig.tight_layout(rect=(0, 0.035, 1, 0.95)); fig.savefig(p); plt.close(fig)
    return p


# ============================================================================ #
# HEAD A
# ============================================================================ #
def _head_a_stacks(models, norms, splits, d, ens=None, ens_norm=None):
    """Stack Head-A predictions/truth over all 8 folds' val rows.

    By default (``ens=None``) each fold's val rows are predicted by that fold's HELD-OUT
    LOSO net — the honest out-of-sample accuracy. When ``ens`` is given the SAME row
    population is predicted by the production ENSEMBLE (mean per-member PHYSICAL f_nhi/dndx,
    via ``predict_headA_ens``) — IN-SAMPLE accuracy on rows the ensemble saw in training.
    Everything else (truth, w_c, P_tier_p) is computed identically, so the only difference
    between the two stacks is the emulator.

    Returns dndx (signed frac err (N,3) + z (N,)), fnhi (signed frac err (N,30) + valid
    + centres), w_c (emu/true/cache (N,4)), P_tier_p frac err, z, and per-(fold,z,class)
    dndx error for the heatmap."""
    grp_all = d["snap_group_idx"]
    z_all = d["z_grid"]
    Xbar_all = d["snap_total_path_dX"][grp_all] / N_SIGHTLINES
    import h5py
    with h5py.File(CACHE, "r") as h:
        centres = h["log_nhi_centres"][:]

    dndx_fe, dndx_z, fnhi_fe, fnhi_valid = [], [], [], []
    wc_emu, wc_true, wc_cache, ptp_fe_all, zrow = [], [], [], [], []
    fold_of_row = []
    for f in range(N_FOLDS):
        model, norm = models[f], norms[f]
        _, va, _ = splits[f]
        if ens is None:
            x = jnp.asarray(d["x"][va]); tau0 = jnp.asarray(d["tau0"][va])
            pred = jax.vmap(model)(x, tau0)
            phys = untransform_prediction(
                {"f_nhi": np.asarray(pred["f_nhi"]), "dndx": np.asarray(pred["dndx"])}, norm)
        else:                          # in-sample: mean per-member physical Head-A
            phys = predict_headA_ens(ens, d, va, ens_norm)
        fnhi_p, dndx_p = phys["f_nhi"], phys["dndx"]
        grp = grp_all[va]
        fnhi_t = d["snap_f_nhi"][grp]; dndx_t = d["snap_dNdX"][grp]
        zv = z_all[va]
        fv = np.isfinite(fnhi_t) & (fnhi_t > 0)
        dv = np.isfinite(dndx_t) & (dndx_t > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ff = np.where(fv & np.isfinite(fnhi_p), (fnhi_p - fnhi_t) / fnhi_t, np.nan)
            df = np.where(dv & np.isfinite(dndx_p), (dndx_p - dndx_t) / dndx_t, np.nan)
        dndx_fe.append(df); dndx_z.append(zv); fnhi_fe.append(ff); fnhi_valid.append(fv)
        fold_of_row.append(np.full(len(va), f))

        Xb = Xbar_all[va]
        we = np.asarray(w_c_corrected(jnp.asarray(dndx_p), jnp.asarray(Xb), jnp.asarray(zv)))
        wt = np.asarray(w_c_corrected(jnp.asarray(dndx_t), jnp.asarray(Xb), jnp.asarray(zv)))
        Pf = d["P_filt"][va]
        ptp_e = np.asarray(structural_tier_p(jnp.asarray(we), jnp.asarray(Pf)))
        ptp_t = np.asarray(structural_tier_p(jnp.asarray(wt), jnp.asarray(Pf)))
        ptp_mask = np.isfinite(ptp_t) & (ptp_t > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ptp_fe = np.where(ptp_mask, (ptp_e - ptp_t) / ptp_t, np.nan)
        wc_emu.append(we); wc_true.append(wt); wc_cache.append(d["w_c_cache"][va])
        ptp_fe_all.append(ptp_fe); zrow.append(zv)

    return dict(
        dndx_fe=np.concatenate(dndx_fe), dndx_z=np.concatenate(dndx_z),
        fnhi_fe=np.concatenate(fnhi_fe), fnhi_valid=np.concatenate(fnhi_valid),
        centres=centres, wc_emu=np.concatenate(wc_emu), wc_true=np.concatenate(wc_true),
        wc_cache=np.concatenate(wc_cache), ptp_fe=np.concatenate(ptp_fe_all),
        zrow=np.concatenate(zrow), fold=np.concatenate(fold_of_row),
        Xbar_all=Xbar_all, grp_all=grp_all)


def _dndx_med_curve(fe, zv, z_levels, c):
    med = []
    for z in z_levels:
        sel = np.isclose(zv, z)
        col = np.abs(fe[sel, c]); col = col[np.isfinite(col)]
        med.append(np.median(col) * 100 if col.size else np.nan)
    return np.array(med)


def figA1_dndx_pred_vs_true(S, z_levels, S_ens=None):
    """A1: per-class dN/dX fractional accuracy vs z (LOSO held-out, 8 folds).

    ACCURACY figure. ``S`` = the LOSO HELD-OUT stack (the real accuracy claim, solid
    median + dashed p95). When ``S_ens`` (the production ensemble's IN-SAMPLE stack over the
    SAME rows) is given, its median is OVERLAID (dotted) — the deployed-fit reference, NOT a
    generalization claim. Returns (path, out) with both medians keyed in ``out``."""
    fe, zv = S["dndx_fe"], S["dndx_z"]
    fe_e = S_ens["dndx_fe"] if S_ens is not None else None
    zv_e = S_ens["dndx_z"] if S_ens is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    out = {}
    for c in range(3):
        ax = axes[c]; p95 = []
        med = _dndx_med_curve(fe, zv, z_levels, c)
        for z in z_levels:
            sel = np.isclose(zv, z)
            col = np.abs(fe[sel, c]); col = col[np.isfinite(col)]
            p95.append(np.percentile(col, 95) * 100 if col.size else np.nan)
        p95 = np.array(p95)
        ax.plot(z_levels, med, "o-", color="C0", ms=4,
                label="LOSO held-out median" if S_ens is not None else "median |frac err|")
        ax.plot(z_levels, p95, "s--", color="C3", ms=4, label="LOSO held-out p95")
        if fe_e is not None:
            med_e = _dndx_med_curve(fe_e, zv_e, z_levels, c)
            ax.plot(z_levels, med_e, "^:", color="C2", lw=1.5, ms=4,
                    label="in-sample ensemble median")
            out.setdefault("_ens", {})[HCD[c]] = {
                "median_pct": float(np.nanmedian(np.abs(fe_e[:, c]) * 100))}
        ax.axhline(2.5, color="r", ls=":", lw=1.2, label="2.5% target")
        ax.set_title(rf"d$N$/d$X$   {HCD[c]}", fontweight="bold")
        ax.set_xlabel(r"$z$"); ax.grid(alpha=0.3)
        ax.set_ylim(0, max(8, np.nanmax(p95) * 1.1))
        if c == 0:
            ax.set_ylabel("|fractional error|  [%]"); ax.legend(loc="upper center", fontsize=8.5)
        out[HCD[c]] = {"median_pct": float(np.nanmedian(np.abs(fe[:, c]) * 100))}
    fig.suptitle(r"A1 — Head-A d$N$/d$X$ fractional accuracy vs $z$ (LOSO held-out, 8 folds)",
                 fontsize=13, fontweight="bold")
    if S_ens is not None:
        _caption(fig, "solid/dashed = LOSO held-out (out-of-sample)   ·   "
                      "dotted = in-sample ensemble (saw these sims)")
        rect = (0, 0.05, 1, 0.94)
    else:
        rect = (0, 0, 1, 0.95)
    p = OUTDIR / "A1_dndx_pred_vs_true.png"
    fig.tight_layout(rect=rect); fig.savefig(p); plt.close(fig)
    return p, out


def figA2_cddf_pred_vs_true(S, S_ens=None):
    """A2: f_NHI / CDDF fractional accuracy vs logNHI, with class boundaries +
    shot-noise tail flagged.

    ACCURACY figure. ``S`` = LOSO HELD-OUT (solid median + dashed p95, the real claim).
    When ``S_ens`` is given the production ensemble's IN-SAMPLE median is OVERLAID (dotted)."""
    fe, valid, centres = S["fnhi_fe"], S["fnhi_valid"], S["centres"]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    med, p95, frac = [], [], []
    for b in range(fe.shape[1]):
        col = np.abs(fe[:, b][valid[:, b]]); col = col[np.isfinite(col)]
        med.append(np.median(col) * 100 if col.size else np.nan)
        p95.append(np.percentile(col, 95) * 100 if col.size else np.nan)
        frac.append(valid[:, b].mean())
    ax.plot(centres, med, "o-", color="C0", ms=4,
            label="LOSO held-out median" if S_ens is not None else "median |frac err|")
    ax.plot(centres, p95, "s--", color="C3", ms=4, label="LOSO held-out p95")
    out = {"median_pct": float(np.median(np.abs(fe[valid][np.isfinite(fe[valid])])) * 100),
           "p95_pct": float(np.percentile(
               np.abs(fe[valid][np.isfinite(fe[valid])]), 95) * 100)}
    if S_ens is not None:
        fe_e, valid_e = S_ens["fnhi_fe"], S_ens["fnhi_valid"]
        med_e = []
        for b in range(fe_e.shape[1]):
            col = np.abs(fe_e[:, b][valid_e[:, b]]); col = col[np.isfinite(col)]
            med_e.append(np.median(col) * 100 if col.size else np.nan)
        ax.plot(centres, med_e, "^:", color="C2", lw=1.5, ms=4, label="in-sample ensemble median")
        oe = np.abs(fe_e[valid_e]); oe = oe[np.isfinite(oe)]
        out["_ens"] = {"median_pct": float(np.median(oe) * 100),
                       "p95_pct": float(np.percentile(oe, 95) * 100)}
    ax.axhline(2.5, color="r", ls=":", lw=1.2, label="2.5% target")
    for lab, v in NHI_BOUNDS.items():
        ax.axvline(v, color="grey", ls="-", lw=1.1, alpha=0.7)
        ax.text(v, ax.get_ylim()[1], lab.split()[0], rotation=90, va="top",
                ha="right", fontsize=7.5, color="grey")
    ax.axvspan(NHI_SHOT, centres.max(), color="orange", alpha=0.12)
    ax.text(NHI_SHOT + 0.1, 3.0, "shot-noise tail\n" + r"($\log N_{\rm HI}\geq21.5$)",
            fontsize=8.5, color="darkorange")
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax.set_ylabel("|fractional error|  [%]")
    ax.set_yscale("log")
    ax.set_title(r"A2 — Head-A $f_{N_{\rm HI}}$ (CDDF) fractional accuracy per $N_{\rm HI}$ bin "
                 "(LOSO held-out, 8 folds)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(centres, frac, color="gray", alpha=0.45, lw=1.2)
    ax2.set_ylabel("valid-row fraction (gray)", color="gray"); ax2.set_ylim(0, 1.05)
    cap = ("solid/dashed = LOSO held-out   ·   dotted = in-sample ensemble"
           if S_ens is not None
           else "vertical lines = class boundaries;  shaded = shot-noise tail")
    _caption(fig, cap)
    p = OUTDIR / "A2_cddf_pred_vs_true.png"
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(p); plt.close(fig)
    return p, out


def figA3_wc_ptierp_coupling(S, S_ens=None):
    """A3: w_c round-trip (emulated vs true dN/dX) + structural P_tier_p faithfulness.

    ACCURACY figure. ``S`` = LOSO HELD-OUT. The left scatter shows the held-out round-trip;
    the right histogram shows the held-out P_tier_p faithfulness (filled) and, when ``S_ens``
    is given, OVERLAYS the production ensemble's IN-SAMPLE faithfulness (dotted outline)."""
    we, wt, ptp_fe = S["wc_emu"], S["wc_true"], S["ptp_fe"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for c in range(4):
        ax.scatter(wt[:, c], we[:, c], s=3, alpha=0.25, color=COLORS[CLS[c]], label=CLS[c])
    lim = [0, max(wt.max(), we.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(r"$w_c$ (from TRUE d$N$/d$X$)")
    ax.set_ylabel(r"$w_c$ (from EMULATED d$N$/d$X$)")
    ax.set_title(r"$w_c$ round-trip (LOSO held-out)", fontweight="bold")
    ax.legend(fontsize=9, markerscale=4); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in range(4):
        err = np.abs(we[:, c] - wt[:, c])
        ax.hist(err, bins=60, histtype="step", color=COLORS[CLS[c]], label=CLS[c], density=True)
    ax.axvline(0.025, color="r", ls=":", lw=1.2, label="2.5% abs")
    ax.set_xlabel(r"$|w_c^{\rm emu} - w_c^{\rm true}|$  (LOSO held-out)"); ax.set_ylabel("density")
    ax.set_title(r"$w_c$ coupling error", fontweight="bold"); ax.set_xlim(0, 0.04)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    fe = ptp_fe[np.isfinite(ptp_fe)]
    ax.hist(fe * 100, bins=80, histtype="stepfilled", color="C0", alpha=0.6,
            label="LOSO held-out", density=True)
    out = {"ptierp_median_pct": float(np.median(np.abs(fe)) * 100),
           "ptierp_p95_pct": float(np.percentile(np.abs(fe), 95) * 100)}
    if S_ens is not None:
        fe_e = S_ens["ptp_fe"][np.isfinite(S_ens["ptp_fe"])]
        ax.hist(fe_e * 100, bins=80, histtype="step", color="C2", lw=1.5, ls=":",
                label="in-sample ensemble", density=True)
        out["_ens"] = {"ptierp_median_pct": float(np.median(np.abs(fe_e)) * 100),
                       "ptierp_p95_pct": float(np.percentile(np.abs(fe_e), 95) * 100)}
    ax.axvline(0, color="k", lw=1)
    for q, ls in [(50, "-"), (95, "--")]:
        v = np.percentile(np.abs(fe) * 100, q)
        ax.axvline(v, color="r", ls=ls, lw=1, label=f"held-out p{q} |err| = {v:.3f}%")
        ax.axvline(-v, color="r", ls=ls, lw=1)
    ax.axvspan(-2.5, 2.5, color="green", alpha=0.08)
    ax.set_xlabel(r"$P_{\rm tier,p}$ frac err (emu vs true $w_c$)  [%]"); ax.set_ylabel("density")
    ax.set_title(r"Structural $P_{\rm tier,p}$ faithfulness (≤2.5% target)", fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.set_xlim(-3, 3)
    fig.suptitle(r"A3 — $w_c \rightarrow P_{\rm tier,p}$ coupling: emulated-d$N$/d$X$ $w_c$ vs "
                 "true + faithfulness (8 folds)", fontsize=13, fontweight="bold")
    if S_ens is not None:
        _caption(fig, "right panel:  filled = LOSO held-out   ·   dotted = in-sample ensemble")
        rect = (0, 0.04, 1, 0.94)
    else:
        rect = (0, 0, 1, 0.95)
    p = OUTDIR / "A3_wc_ptierp_coupling.png"
    fig.tight_layout(rect=rect); fig.savefig(p); plt.close(fig)
    return p, out


def figA4_head_a_error_heatmap(S, z_levels):
    """A4: per-class / z / fold Head-A dN/dX error heatmap (median |frac err| %).

    INHERENTLY-LOSO generalization figure (companion validation): the per-fold HELD-OUT
    spread only exists for the LOSO nets — the ensemble saw every sim, so there is no
    per-fold held-out version. ALWAYS the 8 LOSO folds regardless of ``--emulator``."""
    fe, zv, fold = S["dndx_fe"], S["dndx_z"], S["fold"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    vmax = 0.0
    grids = []
    for c in range(3):
        G = np.full((N_FOLDS, len(z_levels)), np.nan)
        for fi in range(N_FOLDS):
            for zi, z in enumerate(z_levels):
                sel = (fold == fi) & np.isclose(zv, z)
                col = np.abs(fe[sel, c]); col = col[np.isfinite(col)]
                if col.size:
                    G[fi, zi] = np.median(col) * 100
        grids.append(G); vmax = max(vmax, np.nanpercentile(G, 95))
    for c in range(3):
        ax = axes[c]
        im = ax.imshow(grids[c], aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax,
                       extent=[z_levels.min(), z_levels.max(), -0.5, N_FOLDS - 0.5])
        ax.set_title(rf"d$N$/d$X$   {HCD[c]}", fontweight="bold"); ax.set_xlabel(r"$z$")
        if c == 0:
            ax.set_ylabel("LOSO fold")
        ax.set_yticks(range(N_FOLDS))
        fig.colorbar(im, ax=ax, label="median |frac err| [%]" if c == 2 else None)
    fig.suptitle(r"A4 — Head-A d$N$/d$X$ error heatmap: per class × $z$ × fold "
                 "(median |frac err|, %, LOSO held-out)", fontsize=13, fontweight="bold")
    _caption(fig, "the ensemble has no per-fold held-out;  "
                  "a hot row/column flags a weak fold or $z$")
    p = OUTDIR / "A4_head_a_error_heatmap.png"
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(p); plt.close(fig)
    return p


def figA5_alpha_prior_sanity(S, deployed=False):
    """A5: dN/dX -> alpha_c prior-center sanity — emulated w_c vs cache (CDDF-derived) w_c.

    DEPLOYED-OBJECT PROPERTY figure: the emulated dN/dX feeds the α_c prior center via
    w_c_corrected; in ``--emulator ensemble`` the ``S`` stack's ``wc_emu`` comes FROM the
    production ensemble, so this IS the deployed prior center. We compare it to the cache's
    empirical CDDF-derived w_c (a fixed, emulator-independent reference)."""
    we, wcache = S["wc_emu"], S["wc_cache"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for c in range(4):
        ax.scatter(wcache[:, c], we[:, c], s=3, alpha=0.25, color=COLORS[CLS[c]], label=CLS[c])
    lim = [0, max(wcache.max(), we.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(r"$w_c$ (cache CDDF integral)")
    ax.set_ylabel(r"$w_c$ (emulated d$N$/d$X$ → prior center)")
    ax.set_title(r"Emulated $w_c$ ($\alpha_c$ prior center) vs cache CDDF $w_c$",
                 fontweight="bold")
    ax.legend(fontsize=9, markerscale=4); ax.grid(alpha=0.3)

    ax = axes[1]
    out = {}
    for c in range(4):
        err = np.abs(we[:, c] - wcache[:, c])
        ax.hist(err, bins=60, histtype="step", color=COLORS[CLS[c]], label=CLS[c], density=True)
        out[CLS[c]] = {"median": float(np.median(err)), "p95": float(np.percentile(err, 95))}
    ax.axvline(0.025, color="r", ls=":", lw=1.2, label="2.5% abs")
    ax.set_xlabel(r"$|w_c^{\rm emu} - w_c^{\rm cache}|$"); ax.set_ylabel("density")
    ax.set_title(r"$\alpha_c$ prior-center deviation from cache CDDF integral", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 0.06)
    src = LBL_ENS if deployed else "8 LOSO folds (held-out)"
    fig.suptitle(r"A5 — d$N$/d$X$ → $\alpha_c$ prior-center sanity: emulated $w_c$ vs cache CDDF "
                 f"integral  ({src})", fontsize=12.5, fontweight="bold")
    p = OUTDIR / "A5_alpha_prior_sanity.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(p); plt.close(fig)
    return p, out


# ============================================================================ #
# Main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-mf", action="store_true",
                    help="skip the multi-fidelity figure B6 (needs the HR cache + res_corr)")
    ap.add_argument("--emulator", choices=["loso", "ensemble"], default="loso",
                    help="loso (DEFAULT, byte-identical to the original LOSO walkthrough: every "
                         "figure on each fold's HELD-OUT sims) OR ensemble (the DEPLOYED production "
                         "ensemble — property figures regenerated FROM it, accuracy figures OVERLAY "
                         "in-sample ensemble on the retained LOSO held-out, inherently-LOSO figures "
                         "kept as LOSO companions). See the module docstring for the per-figure "
                         "classification.")
    args = ap.parse_args()
    ENS_MODE = (args.emulator == "ensemble")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    _mode_desc = ("DEPLOYED production ensemble (in-sample) + retained LOSO held-out overlays"
                  if ENS_MODE else "LOSO held-out (out-of-sample)")
    print(f"--emulator {args.emulator}  ({_mode_desc})")
    d = load_cache(CACHE)
    kf = d["kfkms"][0]
    z_levels = np.unique(np.round(d["z_grid"], 1))
    print(f"cache: {d['P_tier_p'].shape[0]} rows, n_k={len(kf)}, z-levels={len(z_levels)}")

    load_all_folds._d = d
    models, norms, splits = load_all_folds()
    print(f"loaded {len(models)} finalized LOSO folds (MF = ρ(k,z)-only FixedMeanHead default)")

    ens, ens_norm = (None, None)
    if ENS_MODE:
        ens, ens_norm = load_production_ensemble()
        print(f"loaded production ENSEMBLE: {len(ens.members)} members "
              f"(final_prod_seed0..{len(ens.members)-1}); deployed pred = mean post-exp P_filt")

    # head = dual-keyed: loso_held_out always present; ensemble_in_sample added in ensemble mode.
    produced, skipped = {}, {}
    head = {"_meta": {"emulator_mode": args.emulator,
                      "loso_ckpt": CKPT, "prod_ensemble": PROD_CKPT if ENS_MODE else None,
                      "framing": "loso_held_out = out-of-sample generalization (the accuracy "
                                 "claim); ensemble_in_sample = the deployed all-sims ensemble's "
                                 "in-sample fit (NOT a generalization claim)"},
            "loso_held_out": {}, "ensemble_in_sample": {} if ENS_MODE else None}
    L = head["loso_held_out"]; E = head["ensemble_in_sample"]
    cv_floor = load_cv_floor()

    # -------- HEAD B --------
    # B1 (ACCURACY): overlay in-sample ensemble on the retained LOSO held-out.
    p, fr, fr_e = figB1_pred_vs_true_p1d(models, norms, splits, d, kf, ens=ens, ens_norm=ens_norm)
    produced["B1"] = p; L["B1_pred_frac_rms"] = fr
    if ENS_MODE:
        E["B1_pred_frac_rms"] = fr_e

    # B2 (ACCURACY): LOSO held-out frac + overlaid in-sample ensemble frac (same rows).
    frac, kf2 = _deployed_frac_in_range(models, norms, splits, d)
    frac_e = (_deployed_frac_in_range(models, norms, splits, d, ens=ens, ens_norm=ens_norm)[0]
              if ENS_MODE else None)
    p, rms, rms_e = figB2_deployed_frac_err_vs_k(frac, kf2, cv_floor, frac_ens=frac_e)
    produced["B2"] = p; L["B2_deployed_frac_rms_inrange"] = rms
    if ENS_MODE:
        E["B2_deployed_frac_rms_inrange"] = rms_e

    # B3 (DEPLOYED-OBJECT PROPERTY): the ensemble IS the deployed response in ensemble mode.
    p, constr = figB3_cosmology_response(models, norms, splits, d, kf, ens=ens, ens_norm=ens_norm)
    produced["B3"] = p
    (E if ENS_MODE else L)["B3_response_amplitude"] = constr

    # B4 (DEPLOYED-OBJECT PROPERTY).
    p, tt = figB4_theta_tracking(models, norms, splits, d, ens=ens, ens_norm=ens_norm)
    produced["B4"] = p
    (E if ENS_MODE else L)["B4_theta_tracking"] = tt

    # B5 (INHERENTLY-LOSO generalization companion): always the LOSO folds.
    p, fb = figB5_fisher_bias_perfold(models, norms, splits, d); produced["B5"] = p
    L["B5_fisher_bias"] = fb

    # B6 (MF deployed-object property; independent of the main-emulator checkpoint).
    if args.no_mf:
        skipped["B6"] = "skipped via --no-mf"
    else:
        p, mfsum = figB6_mf_high_k(d)
        if p is None:
            skipped["B6"] = mfsum
        else:
            produced["B6"] = p; L["B6_mf_kodiaq"] = mfsum

    # B7 (DEPLOYED-OBJECT PROPERTY): the templates the forward model uses.
    p = figB7_delta_c_templates(models, norms, splits, d, kf, ens=ens, ens_norm=ens_norm)
    produced["B7"] = p

    # -------- HEAD A --------
    print("building Head-A LOSO held-out stacks (8 folds)...")
    S = _head_a_stacks(models, norms, splits, d)
    S_ens = None
    if ENS_MODE:
        print("building Head-A IN-SAMPLE ensemble stack (same rows)...")
        S_ens = _head_a_stacks(models, norms, splits, d, ens=ens, ens_norm=ens_norm)

    # A1/A2/A3 (ACCURACY): LOSO held-out + overlaid in-sample ensemble.
    p, a1 = figA1_dndx_pred_vs_true(S, z_levels, S_ens=S_ens); produced["A1"] = p
    L["A1_dndx"] = {k: v for k, v in a1.items() if k != "_ens"}
    if ENS_MODE and "_ens" in a1:
        E["A1_dndx"] = a1["_ens"]
    p, a2 = figA2_cddf_pred_vs_true(S, S_ens=S_ens); produced["A2"] = p
    L["A2_cddf"] = {k: v for k, v in a2.items() if k != "_ens"}
    if ENS_MODE and "_ens" in a2:
        E["A2_cddf"] = a2["_ens"]
    p, a3 = figA3_wc_ptierp_coupling(S, S_ens=S_ens); produced["A3"] = p
    L["A3_wc_ptierp"] = {k: v for k, v in a3.items() if k != "_ens"}
    if ENS_MODE and "_ens" in a3:
        E["A3_wc_ptierp"] = a3["_ens"]

    # A4 (INHERENTLY-LOSO generalization companion): always the LOSO folds.
    p = figA4_head_a_error_heatmap(S, z_levels); produced["A4"] = p

    # A5 (DEPLOYED-OBJECT PROPERTY): emulated w_c prior center from the deployed object.
    p, a5 = figA5_alpha_prior_sanity(S_ens if ENS_MODE else S, deployed=ENS_MODE)
    produced["A5"] = p
    (E if ENS_MODE else L)["A5_alpha_prior"] = a5

    if not ENS_MODE:
        head.pop("ensemble_in_sample")

    with open(OUTDIR / "headline_numbers.json", "w") as f:
        json.dump(head, f, indent=2, default=float)

    print("\n==== HEADLINE NUMBERS (dual-keyed) ====")
    print(json.dumps(head, indent=2, default=float))
    print("\n==== FIGURES PRODUCED ====")
    for k, v in produced.items():
        print(f"  {k}: {v}")
    if skipped:
        print("==== SKIPPED ====")
        for k, v in skipped.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
