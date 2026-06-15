"""Honest example-prediction figure for the emulator README §5 (+ the 20% diagnosis).

The OLD README §5 figure (pred_vs_true_p1d_fold0.png) plotted the per-class pred/cache
residual over the FULL native k-grid, including bins ABOVE the LF Nyquist (k≈0.069 s/km,
where the LF emulator is neither used nor validated) and the lowest, cosmic-variance-sparse
modes. Those k-extremes reach ±20%, which alarms a reader even though the analysis band
(DESI k≤0.041, KS k≤0.06) is ~1-2%.

This script:
  1. DIAGNOSIS — for fold-0's held-out val rows, recomputes the per-class pred/cache
     residual and reports |resid| split by k-region: DESI band (k≤0.041), KS band
     (0.041<k≤0.06), Nyquist approach (0.06<k≤0.069), ABOVE Nyquist (k>0.069), and the
     low-k cosmic-variance edge (k<0.005). Prints in-range vs out-of-range numbers.
  2. HONEST FIGURE — re-renders the same example prediction but (a) restricts the residual
     panel to the in-range band, (b) shades+labels the above-Nyquist tail "not used", and
     (c) draws the DESI/KS k_max lines. Saved to the CODE repo figures dir.

The rigorous all-folds LOSO error (per-fold RMS, the A_p/n_s Fisher gate, the per-k U-shape)
lives in the notes validation doc:
  hcd_priya_notes/docs/superpowers/2026-06-14-validation-loso-emulator-lf-mf.md  (§2)
and its figure figures/analysis/06_validation_summary/loso_perk_pred_error.png.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_pred_vs_true_honest.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 hard-assert BEFORE jax
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import (
    load_cache, make_splits, untransform_prediction, COARSE_NAMES,
)

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CKPT = "/home/mfho/hcd_priya/checkpoints/final_fold0"
FIGDIR = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
FIG = f"{FIGDIR}/pred_vs_true_p1d_fold0_inrange.png"

# analysis-band / Nyquist k landmarks (angular s/km)
K_DESI = 0.041     # DESI DR1 P1D k_max used in the analysis
K_KS = 0.06        # KODIAQ-SQUAD high-res k_max
K_NYQ = 0.069      # LF native Nyquist of the n_k=172 angular grid
K_LOWK_CV = 0.005  # below this = the lowest, cosmic-variance-sparse modes


def main():
    d = load_cache(CACHE)
    n_k = d["P_tier_p"].shape[1]
    # same split machinery the deployed run / training figure used (fold 0, 8 folds)
    tr, va, holdout = make_splits(d, fold=0, n_folds=8, holdout_frac=0.15)
    norm = T.load_checkpoint(CKPT)[2]
    model = T.load_checkpoint(CKPT)[0]

    # ---- predict ALL held-out val rows (the honest, fold-0 generalization set) ----
    va = np.asarray(va)
    x = jnp.asarray(d["x"][va]); tau0 = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(x, tau0)
    P_pred = untransform_prediction(
        {"P_filt_base": np.asarray(pred["P_filt_base"]),
         "P_filt_resid": np.asarray(pred["P_filt_resid"])}, norm)["P_filt"]  # (n,4,K)
    P_true = d["P_filt"][va]          # (n,4,K) linear cache
    kf = d["kfkms"][va]              # (n,K) angular k per row

    # ---- DIAGNOSIS: |resid| split by k-region, per class ----
    regions = {
        "low-k CV edge (k<0.005)":  lambda k: k < K_LOWK_CV,
        "DESI band (0.005-0.041)":  lambda k: (k >= K_LOWK_CV) & (k <= K_DESI),
        "KS band (0.041-0.06)":     lambda k: (k > K_DESI) & (k <= K_KS),
        "Nyq approach (0.06-0.069)": lambda k: (k > K_KS) & (k <= K_NYQ),
        "ABOVE Nyquist (k>0.069)":  lambda k: k > K_NYQ,
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = P_pred / np.where(P_true > 0, P_true, np.nan) - 1.0   # (n,4,K)
    finite = np.isfinite(resid) & np.isfinite(P_true) & (P_true > 0)

    print(f"# fold-0 held-out val rows: {len(va)}  (n_k={n_k})")
    print(f"# k landmarks: DESI={K_DESI}, KS={K_KS}, Nyquist={K_NYQ} s/km (angular)\n")
    print(f"{'region':28s} {'k-bins':>7s} | " + " ".join(f"{c:>8s}" for c in COARSE_NAMES))
    print(f"{'(median |pred/cache-1|, %)':28s} {'':>7s} | " + " ".join("--------" for _ in COARSE_NAMES))
    # broadcast k mask to (n,K), then per class
    for rname, rfn in regions.items():
        kmask = rfn(kf)                                   # (n,K)
        nbins_typ = int(np.round(kmask.sum() / len(va))) if len(va) else 0
        row = []
        for c in range(4):
            sel = kmask & finite[:, c, :]
            vals = np.abs(resid[:, c, :][sel])
            row.append(np.nan if vals.size == 0 else 100 * np.median(vals))
        print(f"{rname:28s} {nbins_typ:>7d} | " + " ".join(f"{v:8.2f}" for v in row))

    # in-range (DESI+KS, k<=0.06) vs out-of-range (k>0.069) pooled over all classes
    kin = (kf > K_LOWK_CV) & (kf <= K_KS)
    kout = kf > K_NYQ
    def pooled(mask):
        vals = []
        for c in range(4):
            sel = mask & finite[:, c, :]
            vals.append(np.abs(resid[:, c, :][sel]))
        a = np.concatenate(vals)
        return 100 * np.median(a), 100 * np.percentile(a, 95), 100 * np.max(a)
    mi, hi95i, mxi = pooled(kin)
    mo, hi95o, mxo = pooled(kout)
    print("\n# ===== in-range vs out-of-range (all 4 classes pooled) =====")
    print(f"  IN-RANGE  (0.005<k<=0.06):  median {mi:.2f}%  95pct {hi95i:.2f}%  max {mxi:.2f}%")
    print(f"  ABOVE-NYQ (k>0.069):        median {mo:.2f}%  95pct {hi95o:.2f}%  max {mxo:.2f}%")
    klowk = kf < K_LOWK_CV
    ml, _, mxl = pooled(klowk)
    print(f"  LOW-k CV  (k<0.005):        median {ml:.2f}%                  max {mxl:.2f}%")

    # ---- HONEST FIGURE (first 3 val rows, to match the old figure's look) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = np.arange(min(3, len(va)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for ci, cname in enumerate(COARSE_NAMES):
        ax = axes[0, ci]; axr = axes[1, ci]
        for ri in rows:
            k = kf[ri]; pt = P_true[ri, ci]; pp = P_pred[ri, ci]
            ok = np.isfinite(pt) & (pt > 0) & np.isfinite(pp) & (pp > 0)
            ax.loglog(k[ok], pt[ok], "-", color=f"C{ri}", alpha=0.8,
                      label=f"row{va[ri]} cache" if ci == 0 else None)
            ax.loglog(k[ok], pp[ok], "--", color=f"C{ri}", alpha=0.8,
                      label=f"row{va[ri]} pred" if ci == 0 else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = pp[ok] / pt[ok] - 1.0
            axr.semilogx(k[ok], 100 * r, "-", color=f"C{ri}", alpha=0.85)
        # shade above-Nyquist (not used) + draw analysis-band k_max lines
        for a in (ax, axr):
            a.axvspan(K_NYQ, k[ok].max() if ok.any() else K_NYQ * 1.4,
                      color="0.85", alpha=0.6, zorder=0)
            a.axvline(K_DESI, color="C3", ls=":", lw=1.0, alpha=0.8)
            a.axvline(K_KS, color="C2", ls="--", lw=1.0, alpha=0.8)
            a.axvline(K_NYQ, color="0.4", ls="-", lw=0.8, alpha=0.8)
        ax.set_title(cname); ax.grid(alpha=0.3)
        axr.axhline(0, color="k", lw=0.6); axr.grid(alpha=0.3)
        axr.set_ylim(-5, 5)         # the analysis band lives here; keeps focus honest
        axr.set_xlabel("k [s/km]  (angular)")
        if ci == 0:
            ax.set_ylabel("P1D (filtered)")
            axr.set_ylabel("pred/cache - 1  [%]")
            ax.text(K_DESI, ax.get_ylim()[0] * 1.5, " DESI", color="C3",
                    fontsize=7, rotation=90, va="bottom")
            ax.text(K_KS, ax.get_ylim()[0] * 1.5, " KS", color="C2",
                    fontsize=7, rotation=90, va="bottom")
            ax.text(K_NYQ * 1.02, ax.get_ylim()[1] * 0.4, "above Nyquist\n(not used)",
                    color="0.4", fontsize=7, va="top")
            ax.legend(fontsize=7)
    fig.suptitle("Fold-0 example prediction vs cache, per HCD class — residual shown on the USED band "
                 "(±5%); grey = above LF Nyquist (not used / not validated)")
    fig.tight_layout()
    Path(FIGDIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {FIG}")
    print("[done]")


if __name__ == "__main__":
    main()
