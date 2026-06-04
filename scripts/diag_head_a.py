#!/usr/bin/env python3
"""Head-A (tau0-invariant CDDF f_NHI + dN/dX) VALIDATION on the finalized LOSO models.

Head A predicts the column-density distribution function f_NHI (30 log-NHI bins) and
the per-class HCD incidence dN/dX (3 classes: LLS/subDLA/DLA). These feed the
structural class weights w_c (-> P_tier_p) and the alpha_c prior (the HCD deliverable).
All prior emulator validation was on the P1D / P_filt cosmology channel (Head B); this
script is the FIRST validation of Head A.

For each of the 8 finalized LOSO folds (checkpoints/final_fold{f}.eqx, the FINAL_RECIPE
models) we run the model on its HELD-OUT sims and compare Head-A predictions to truth in
PHYSICAL space (untransform_prediction). We report:

  Task 1  -- Head-A fit quality: fractional accuracy of dN/dX and f_NHI vs truth, per
             class (LLS/subDLA/DLA), per z, per fold; median/p95; where good/bad.
  Task 2  -- w_c coupling: propagate emulated dN/dX -> w_c (via dndx_wc) and check the
             structural P_tier_p = sum_c w_c P_c^filt stays faithful vs w_c from TRUE
             dN/dX (the round-trip; spec sec.8 <=2.5% target).
  Task 3  -- dN/dX -> alpha_c prior: is the emulated dN/dX a sensible prior CENTER for
             alpha_c? Sanity of emulated w_c vs the cache (CDDF-derived) w_c.
  Task 4  -- fold-3 offset: investigate the flagged Head-A/baseline offset on fold-3.
  Task 5  -- convergence: is Head A trained well (its f_nhi/dndx loss terms), or
             under-trained relative to p_resid (term_w[p_resid]=8)?

READ-ONLY on production. Writes figures to figures/analysis/04_emulator/ and a JSON
summary. Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya emu-jax python3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, fit_target_norm, safe_log, signed_log,
    untransform_prediction, COARSE_NAMES,
)
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.dndx_wc import (
    w_c_from_mu, mu_from_dndx, w_c_corrected, alpha_to_dndx,
)
from hcd_analysis.emulator.model import structural_tier_p

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CKPT = "/home/mfho/hcd_priya/checkpoints/final_fold{f}"
FIGDIR = Path("/home/mfho/hcd_priya/figures/analysis/04_emulator")
N_FOLDS = 8
HCD = ("LLS", "subDLA", "DLA")
N_SIGHTLINES = 691200  # constant per row (verified); Xbar = total_path / N_sightlines


def predict_head_a(model, d, idx, norm_stats):
    """Physical-space Head-A predictions (f_nhi, dndx) for rows ``idx``.

    Returns dicts of (n,30) f_nhi and (n,3) dndx in PHYSICAL space (untransformed)."""
    x = jnp.asarray(d["x"][idx])
    tau0 = jnp.asarray(d["tau0"][idx])
    pred = jax.vmap(model)(x, tau0)
    phys = untransform_prediction(
        {"f_nhi": np.asarray(pred["f_nhi"]), "dndx": np.asarray(pred["dndx"])},
        norm_stats)
    return phys["f_nhi"], phys["dndx"]


def frac_err(pred, true, valid):
    """Signed fractional error (pred-true)/true where valid; NaN elsewhere."""
    with np.errstate(invalid="ignore", divide="ignore"):
        fe = (pred - true) / true
    return np.where(valid & (true != 0.0) & np.isfinite(true) & np.isfinite(pred),
                    fe, np.nan)


def summarize(arr):
    a = np.abs(arr[np.isfinite(arr)])
    if a.size == 0:
        return {"median": np.nan, "p95": np.nan, "n": 0}
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = load_cache(CACHE)
    grp_all = d["snap_group_idx"]
    z_all = d["z_grid"]
    z_levels = np.unique(np.round(z_all, 3))
    Xbar_all = d["snap_total_path_dX"][grp_all] / N_SIGHTLINES  # (R,)

    # accumulate per-fold held-out predictions/truth
    per_fold = {}
    # global stacks (over all val rows of all folds) for f_nhi & dndx fractional error
    dndx_fe_all = []          # list of (n,3)
    dndx_z_all = []           # list of (n,)
    fnhi_fe_all = []          # list of (n,30)
    fnhi_valid_all = []       # list of (n,30) bool
    # w_c coupling accumulators
    wc_emu_all, wc_true_all, wc_cache_all = [], [], []
    ptierp_emu_all, ptierp_true_all = [], []
    z_row_all = []

    log_nhi_centres = None
    import h5py
    with h5py.File(CACHE, "r") as h:
        log_nhi_centres = h["log_nhi_centres"][:]

    for f in range(N_FOLDS):
        model, meta, norm = T.load_checkpoint(CKPT.format(f=f))
        tr, va, ho = make_splits(d, f, n_folds=N_FOLDS)
        # sanity: re-fit norm on train and confirm it matches the checkpoint's norm
        # (the checkpoint's norm is the source of truth; we use the SAVED one).
        fnhi_pred, dndx_pred = predict_head_a(model, d, va, norm)
        grp = grp_all[va]
        fnhi_true = d["snap_f_nhi"][grp]      # (n,30)
        dndx_true = d["snap_dNdX"][grp]       # (n,3)
        zv = z_all[va]
        # validity: cache value finite & > 0 (structural zeros excluded -- matches
        # _valid_target_mask used in training)
        fnhi_valid = np.isfinite(fnhi_true) & (fnhi_true > 0)
        dndx_valid = np.isfinite(dndx_true) & (dndx_true > 0)

        fnhi_fe = frac_err(fnhi_pred, fnhi_true, fnhi_valid)   # (n,30)
        dndx_fe = frac_err(dndx_pred, dndx_true, dndx_valid)   # (n,3)

        dndx_fe_all.append(dndx_fe); dndx_z_all.append(zv)
        fnhi_fe_all.append(fnhi_fe); fnhi_valid_all.append(fnhi_valid)

        # --- w_c coupling (Task 2/3) -------------------------------------------
        Xb = Xbar_all[va]
        # emulated w_c (with delta_c correction, as deployed in w_c_corrected)
        wc_emu = np.asarray(w_c_corrected(jnp.asarray(dndx_pred), jnp.asarray(Xb),
                                          jnp.asarray(zv)))      # (n,4)
        wc_true = np.asarray(w_c_corrected(jnp.asarray(dndx_true), jnp.asarray(Xb),
                                           jnp.asarray(zv)))     # (n,4)
        wc_cache = d["w_c_cache"][va]                            # (n,4) empirical
        # structural P_tier_p with emulated vs true w_c, on the SAME (true) P_filt
        Pf = d["P_filt"][va]                                    # (n,4,K) linear
        ptp_emu = np.asarray(structural_tier_p(jnp.asarray(wc_emu), jnp.asarray(Pf)))
        ptp_true = np.asarray(structural_tier_p(jnp.asarray(wc_true), jnp.asarray(Pf)))

        wc_emu_all.append(wc_emu); wc_true_all.append(wc_true); wc_cache_all.append(wc_cache)
        ptierp_emu_all.append(ptp_emu); ptierp_true_all.append(ptp_true)
        z_row_all.append(zv)

        # per-fold dndx + f_nhi summary
        per_fold[f] = {
            "held_sims": sorted(set(d["sim_name"][va].tolist())),
            "n_val_rows": int(len(va)),
            "dndx": {HCD[c]: summarize(dndx_fe[:, c]) for c in range(3)},
            "fnhi": summarize(fnhi_fe),
        }

    dndx_fe = np.concatenate(dndx_fe_all, 0)        # (Nval,3)
    dndx_z = np.concatenate(dndx_z_all, 0)          # (Nval,)
    fnhi_fe = np.concatenate(fnhi_fe_all, 0)        # (Nval,30)
    fnhi_valid = np.concatenate(fnhi_valid_all, 0)  # (Nval,30)
    wc_emu = np.concatenate(wc_emu_all, 0)
    wc_true = np.concatenate(wc_true_all, 0)
    wc_cache = np.concatenate(wc_cache_all, 0)
    ptp_emu = np.concatenate(ptierp_emu_all, 0)
    ptp_true = np.concatenate(ptierp_true_all, 0)
    z_row = np.concatenate(z_row_all, 0)
    mask_row = d["mask"]  # not used; per-row recompute below

    summary = {"cache": CACHE, "n_folds": N_FOLDS}

    # ===================== Task 1: Head-A fit quality ========================
    # dN/dX per class (global, over all held-out rows)
    summary["dndx_per_class"] = {HCD[c]: summarize(dndx_fe[:, c]) for c in range(3)}
    # dN/dX per class per z
    summary["dndx_per_class_per_z"] = {}
    for c in range(3):
        d_c = {}
        for z in z_levels:
            sel = np.isclose(dndx_z, z)
            d_c[f"{z:.1f}"] = summarize(dndx_fe[sel, c])
        summary["dndx_per_class_per_z"][HCD[c]] = d_c
    # f_NHI overall + per-NHI-bin (low vs high N_HI)
    summary["fnhi_overall"] = summarize(fnhi_fe[fnhi_valid])
    fnhi_per_bin = []
    for b in range(fnhi_fe.shape[1]):
        col = fnhi_fe[:, b][fnhi_valid[:, b]]
        fnhi_per_bin.append(summarize(col))
    summary["fnhi_per_nhi_bin"] = {
        f"{log_nhi_centres[b]:.2f}": fnhi_per_bin[b] for b in range(len(fnhi_per_bin))}

    # ===================== Task 2/3: w_c coupling ============================
    # round-trip w_c: emulated vs true dN/dX
    wc_rt_err = np.abs(wc_emu - wc_true)              # (Nval,4)
    summary["wc_roundtrip_emu_vs_true"] = {
        COARSE_NAMES[c]: {"median": float(np.median(wc_rt_err[:, c])),
                          "p95": float(np.percentile(wc_rt_err[:, c], 95)),
                          # relative to the w_c value itself
                          "median_rel": float(np.median(
                              wc_rt_err[:, c] / np.maximum(wc_true[:, c], 1e-6))),
                          "p95_rel": float(np.percentile(
                              wc_rt_err[:, c] / np.maximum(wc_true[:, c], 1e-6), 95))}
        for c in range(4)}
    # w_c emulated vs cache (empirical) -- sanity of the prior CENTER (Task 3)
    wc_cache_err = np.abs(wc_emu - wc_cache)
    summary["wc_emu_vs_cache"] = {
        COARSE_NAMES[c]: {"median": float(np.median(wc_cache_err[:, c])),
                          "p95": float(np.percentile(wc_cache_err[:, c], 95))}
        for c in range(4)}
    # P_tier_p faithfulness (structural sum with emu vs true w_c), in-mask k only
    # recompute per-row mask from finiteness of P_tier_p truth
    ptp_mask = np.isfinite(ptp_true) & (ptp_true > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ptp_fe = (ptp_emu - ptp_true) / ptp_true
    ptp_fe = np.where(ptp_mask, ptp_fe, np.nan)
    summary["ptierp_wc_coupling"] = summarize(ptp_fe)

    # ===================== Task 4: fold-3 offset =============================
    # Re-derive the fold-3 train-norm and locate the blow-up term.
    f = 3
    tr3, va3, _ = make_splits(d, f, n_folds=N_FOLDS)
    ns3 = fit_target_norm(d, tr3)
    b3 = make_batch(d, va3, ns3)
    fold3 = {}
    # per-channel max masked-in standardized target (the loss sees these)
    m = np.asarray(b3["mask"])
    for key, mk in [("t_f_nhi", b3["t_f_nhi_mask"]), ("t_dndx", b3["t_dndx_mask"])]:
        arr = np.asarray(b3[key]); mask = np.asarray(mk)
        vals = np.abs(arr[mask & np.isfinite(arr)])
        fold3[key] = {"max_masked_in": float(vals.max()) if vals.size else np.nan,
                      "p99": float(np.percentile(vals, 99)) if vals.size else np.nan}
    for key in ("t_p_base", "t_p_resid", "t_delta"):
        arr = np.asarray(b3[key])
        nc = arr.shape[1]
        mm = np.broadcast_to(m[:, None, :], arr.shape) & np.isfinite(arr)
        vals = np.abs(arr[mm])
        fold3[key] = {"max_masked_in": float(vals.max()) if vals.size else np.nan,
                      "p99": float(np.percentile(vals, 99)) if vals.size else np.nan}
    # delta norm std-collapse diagnostic
    dstd = ns3["delta"]["std"]
    fold3["delta_norm_std_min"] = float(dstd.min())
    fold3["delta_norm_std_n_below_1e-3"] = int((dstd < 1e-3).sum())
    fold3["delta_norm_std_total"] = int(dstd.size)
    # Head-A val_loss contribution: f_nhi/dndx masked-in mean-sq error vs the model
    # (does the giant 136128 come from a Head-A term? -> compare term sizes)
    model3, _, norm3 = T.load_checkpoint(CKPT.format(f=3))
    pr = jax.vmap(model3)(jnp.asarray(b3["x"]), jnp.asarray(b3["tau0"]))

    def masked_msq(pred, tgt, mask, weight=None):
        pred = np.asarray(pred); tgt = np.nan_to_num(np.asarray(tgt), nan=0.0)
        mask = np.asarray(mask)
        diff = np.where(mask, pred - tgt, 0.0) ** 2
        if weight is not None:
            w = weight * mask
            return float(np.sum(diff * weight) / max(np.sum(w), 1.0))
        return float(np.sum(diff) / max(np.sum(mask), 1.0))

    inv_na = np.asarray(b3["inv_nalpha"])[:, None]
    inv_nc = np.asarray(b3["inv_nc"])
    m3 = m[:, None, :]
    term_la_cddf = masked_msq(pr["f_nhi"], b3["t_f_nhi"], b3["t_f_nhi_mask"], inv_na)
    term_la_dndx = masked_msq(pr["dndx"], b3["t_dndx"], b3["t_dndx_mask"], inv_na)
    term_delta = masked_msq(pr["delta"],
                            b3["t_delta"],
                            np.broadcast_to(m3, b3["t_delta"].shape),
                            inv_nc[:, 1:, None])
    fold3["loss_terms_masked_msq"] = {
        "f_nhi(HeadA)": term_la_cddf, "dndx(HeadA)": term_la_dndx,
        "delta(HeadB)": term_delta}
    summary["fold3_offset"] = fold3

    # ===================== Task 5: convergence ===============================
    # read per-fold histories: final f_nhi/dndx vs p_resid val behavior. The joint
    # history doesn't split terms, but we recompute the per-term val MSE on each
    # fold's val set with the SAVED model to compare Head-A vs p_resid magnitudes.
    conv = {}
    for f in range(N_FOLDS):
        model, _, norm = T.load_checkpoint(CKPT.format(f=f))
        tr, va, _ = make_splits(d, f, n_folds=N_FOLDS)
        bb = make_batch(d, va, norm)
        prr = jax.vmap(model)(jnp.asarray(bb["x"]), jnp.asarray(bb["tau0"]))
        mfa = np.asarray(bb["inv_nalpha"])[:, None]
        mm = np.asarray(bb["mask"])[:, None, :]
        nc = np.asarray(bb["inv_nc"])
        t_la_cddf = masked_msq(prr["f_nhi"], bb["t_f_nhi"], bb["t_f_nhi_mask"], mfa)
        t_la_dndx = masked_msq(prr["dndx"], bb["t_dndx"], bb["t_dndx_mask"], mfa)
        t_presid = masked_msq(prr["P_filt_resid"], bb["t_p_resid"],
                              np.broadcast_to(mm, bb["t_p_resid"].shape),
                              nc[:, :, None])
        conv[f] = {"f_nhi_val_msq": t_la_cddf, "dndx_val_msq": t_la_dndx,
                   "p_resid_val_msq": t_presid}
    summary["convergence_val_msq_per_fold"] = conv

    summary["per_fold"] = per_fold

    # ===================== FIGURES ===========================================
    _fig_dndx(dndx_fe, dndx_z, z_levels)
    _fig_fnhi(fnhi_fe, fnhi_valid, log_nhi_centres)
    _fig_perfold(per_fold, conv)
    _fig_wc_coupling(wc_emu, wc_true, wc_cache, ptp_fe)

    outp = FIGDIR / "head_a_validation.json"
    with open(outp, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {outp}")
    _print_report(summary)


# ----------------------------- figures ------------------------------------

def _fig_dndx(dndx_fe, dndx_z, z_levels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for c in range(3):
        ax = axes[c]
        med, p95 = [], []
        for z in z_levels:
            sel = np.isclose(dndx_z, z)
            col = np.abs(dndx_fe[sel, c]); col = col[np.isfinite(col)]
            med.append(np.median(col) if col.size else np.nan)
            p95.append(np.percentile(col, 95) if col.size else np.nan)
        ax.plot(z_levels, np.array(med) * 100, "o-", label="median |frac err|")
        ax.plot(z_levels, np.array(p95) * 100, "s--", label="p95 |frac err|")
        ax.axhline(2.5, color="r", ls=":", lw=1, label="2.5% target")
        ax.set_title(f"dN/dX  {HCD[c]}")
        ax.set_xlabel("z")
        if c == 0:
            ax.set_ylabel("|fractional error| [%]")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(8, np.nanmax(p95) * 110))
    fig.suptitle("Head-A dN/dX fractional accuracy vs z (held-out, 8 LOSO folds)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "head_a_dndx_pred_vs_true.png", dpi=120)
    plt.close(fig)


def _fig_fnhi(fnhi_fe, fnhi_valid, centres):
    fig, ax = plt.subplots(figsize=(9, 5))
    med, p95, frac = [], [], []
    for b in range(fnhi_fe.shape[1]):
        col = np.abs(fnhi_fe[:, b][fnhi_valid[:, b]])
        col = col[np.isfinite(col)]
        med.append(np.median(col) if col.size else np.nan)
        p95.append(np.percentile(col, 95) if col.size else np.nan)
        frac.append(fnhi_valid[:, b].mean())
    ax.plot(centres, np.array(med) * 100, "o-", label="median |frac err|")
    ax.plot(centres, np.array(p95) * 100, "s--", label="p95 |frac err|")
    ax.axhline(2.5, color="r", ls=":", lw=1, label="2.5%")
    ax.set_xlabel(r"$\log_{10} N_{HI}$")
    ax.set_ylabel("|fractional error| [%]")
    ax.set_yscale("log")
    ax.set_title("Head-A f_NHI (CDDF) fractional accuracy per N_HI bin (held-out)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(centres, frac, color="gray", alpha=0.4, lw=1)
    ax2.set_ylabel("valid-row fraction (gray)", color="gray")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "head_a_fnhi_pred_vs_true.png", dpi=120)
    plt.close(fig)


def _fig_perfold(per_fold, conv):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    folds = sorted(per_fold.keys())
    ax = axes[0]
    for c in range(3):
        med = [per_fold[f]["dndx"][HCD[c]]["median"] * 100 for f in folds]
        p95 = [per_fold[f]["dndx"][HCD[c]]["p95"] * 100 for f in folds]
        ax.plot(folds, med, "o-", label=f"{HCD[c]} median")
        ax.plot(folds, p95, "s--", alpha=0.5)
    fn = [per_fold[f]["fnhi"]["median"] * 100 for f in folds]
    ax.plot(folds, fn, "k^-", label="f_NHI median")
    ax.axhline(2.5, color="r", ls=":", lw=1)
    ax.set_xlabel("fold"); ax.set_ylabel("|frac err| [%]")
    ax.set_title("Per-fold Head-A error (dN/dX per class + f_NHI)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    fnv = [conv[f]["f_nhi_val_msq"] for f in folds]
    ddv = [conv[f]["dndx_val_msq"] for f in folds]
    prv = [conv[f]["p_resid_val_msq"] for f in folds]
    ax.plot(folds, fnv, "o-", label="f_nhi val MSE")
    ax.plot(folds, ddv, "s-", label="dndx val MSE")
    ax.plot(folds, prv, "^-", label="p_resid val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("fold"); ax.set_ylabel("masked-in val MSE (std space)")
    ax.set_title("Head-A vs p_resid val MSE (convergence balance)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "head_a_perfold_error.png", dpi=120)
    plt.close(fig)


def _fig_wc_coupling(wc_emu, wc_true, wc_cache, ptp_fe):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    for c in range(4):
        ax.scatter(wc_true[:, c], wc_emu[:, c], s=2, alpha=0.2,
                   label=COARSE_NAMES[c])
    lim = [0, max(wc_true.max(), wc_emu.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("w_c (from TRUE dN/dX)"); ax.set_ylabel("w_c (from EMULATED dN/dX)")
    ax.set_title("w_c round-trip: emulated vs true dN/dX")
    ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in range(4):
        err = np.abs(wc_emu[:, c] - wc_true[:, c])
        ax.hist(err, bins=60, histtype="step", label=COARSE_NAMES[c], density=True)
    ax.axvline(0.025, color="r", ls=":", lw=1, label="2.5% abs")
    ax.set_xlabel("|w_c(emu) - w_c(true)|"); ax.set_ylabel("density")
    ax.set_title("w_c coupling error (round-trip)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 0.04)

    ax = axes[2]
    fe = ptp_fe[np.isfinite(ptp_fe)]
    ax.hist(fe * 100, bins=80, histtype="stepfilled", alpha=0.6)
    ax.axvline(0, color="k", lw=1)
    for q, ls in [(50, "-"), (95, "--")]:
        v = np.percentile(np.abs(fe) * 100, q)
        ax.axvline(v, color="r", ls=ls, lw=1, label=f"p{q} |err|={v:.3f}%")
        ax.axvline(-v, color="r", ls=ls, lw=1)
    ax.set_xlabel("P_tier_p frac err (emu w_c vs true w_c) [%]")
    ax.set_ylabel("count")
    ax.set_title("Structural P_tier_p faithfulness")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(-3, 3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "head_a_wc_coupling.png", dpi=120)
    plt.close(fig)


def _print_report(s):
    p = lambda *a: print(*a)
    p("\n" + "=" * 70)
    p("HEAD-A VALIDATION REPORT")
    p("=" * 70)
    p("\n[Task 1] dN/dX fractional accuracy per class (held-out, all folds):")
    for c in HCD:
        d_ = s["dndx_per_class"][c]
        p(f"  {c:7s}  median={d_['median']*100:6.3f}%  p95={d_['p95']*100:6.3f}%  (n={d_['n']})")
    p(f"\n  f_NHI overall: median={s['fnhi_overall']['median']*100:.3f}%  "
      f"p95={s['fnhi_overall']['p95']*100:.3f}%")
    p("\n[Task 2/3] w_c round-trip (emulated vs true dN/dX):")
    for c, v in s["wc_roundtrip_emu_vs_true"].items():
        p(f"  {c:7s}  |Δw_c| median={v['median']:.5f} p95={v['p95']:.5f}  "
          f"(rel median={v['median_rel']*100:.3f}% p95={v['p95_rel']*100:.3f}%)")
    pt = s["ptierp_wc_coupling"]
    p(f"\n  P_tier_p faithfulness (emu vs true w_c): median={pt['median']*100:.4f}%  "
      f"p95={pt['p95']*100:.4f}%")
    p("\n[Task 4] fold-3 offset -- max masked-in standardized target per term:")
    for k, v in s["fold3_offset"].items():
        if isinstance(v, dict) and "max_masked_in" in v:
            p(f"  {k:12s} max|t|(masked-in)={v['max_masked_in']:.4g}  p99={v['p99']:.4g}")
    p(f"  delta-norm std min={s['fold3_offset']['delta_norm_std_min']:.3g}  "
      f"(#bins<1e-3: {s['fold3_offset']['delta_norm_std_n_below_1e-3']}/"
      f"{s['fold3_offset']['delta_norm_std_total']})")
    lt = s["fold3_offset"]["loss_terms_masked_msq"]
    p("  fold-3 masked-in MSE per term: " +
      "  ".join(f"{k}={v:.4g}" for k, v in lt.items()))
    p("\n[Task 5] Head-A vs p_resid val MSE per fold (std space):")
    for f, v in s["convergence_val_msq_per_fold"].items():
        p(f"  fold {f}: f_nhi={v['f_nhi_val_msq']:.4g}  dndx={v['dndx_val_msq']:.4g}  "
          f"p_resid={v['p_resid_val_msq']:.4g}")
    p("=" * 70)


if __name__ == "__main__":
    main()
