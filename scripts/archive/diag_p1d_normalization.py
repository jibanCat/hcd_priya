"""Diagnose the Phase-2b P1D normalization + what the trained emulator actually learns.

Produces the figures for docs/superpowers/2026-06-02-training-walkthrough.md:
 1. variance decomposition: cosmology vs (z,tau0) fraction of per-k log-variance
 2. spectrum spread: per-k mean P1D +/- total std (tau0/z) vs +/- within-cell std (cosmology)
 3. fold-0 cosmology tracking: pred-dev vs true-dev from the (z,tau0)-cell mean

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_p1d_normalization.py
"""
from __future__ import annotations
import numpy as np, jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hcd_analysis.emulator.data import (load_cache, make_batch, make_splits,
                                        untransform_prediction, safe_log)
from hcd_analysis.emulator.train import load_checkpoint

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
FIGDIR = "figures/analysis/04_emulator"
CLS = ("clean", "LLS", "subDLA", "DLA")


def main():
    d = load_cache(CACHE)
    R, C, K = d["P_filt"].shape
    logP = safe_log(d["P_filt"])
    kf = d["kfkms"][0]
    zc = np.round(d["z_grid"], 4); ai = d["alpha_idx"].astype(int)
    cell = (zc * 1000).astype(int) * 100 + ai          # (z, alpha) cell id
    cells = np.unique(cell)

    # --- variance decomposition (cosmology = within (z,alpha) cell) ---
    vt = np.nanvar(logP, axis=0)                        # (C,K)
    vw = np.zeros((C, K))
    for cl in cells:
        m = cell == cl
        if m.sum() < 2: continue
        vw += np.nanvar(logP[m], axis=0) * m.sum()
    vw /= R
    frac = vw / np.where(vt > 0, vt, 1)                 # cosmology fraction of log-var

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, nm in enumerate(CLS):
        ax.semilogx(kf, 100 * frac[ci], label=nm)
    ax.set_xlabel("k  [s/km]"); ax.set_ylabel("cosmology fraction of log-var  [%]")
    ax.set_title("Variance decomposition: cosmology is <1% of per-k log-variance\n"
                 "(the other ~99.5% is the tau0/z spread the NN gets as inputs)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/norm_variance_decomp.png", dpi=160); plt.close(fig)

    # --- spectrum spread: total vs within-cell (cosmology) ---
    mu = np.nanmean(logP, axis=0); sig_tot = np.nanstd(logP, axis=0)
    sig_cos = np.sqrt(vw)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ci, (nm, axx) in enumerate(zip(CLS, axes.ravel())):
        axx.fill_between(kf, mu[ci] - sig_tot[ci], mu[ci] + sig_tot[ci],
                         alpha=0.25, label="+/- total std (tau0/z)")
        axx.fill_between(kf, mu[ci] - sig_cos[ci], mu[ci] + sig_cos[ci],
                         alpha=0.55, label="+/- cosmology std")
        axx.semilogx(kf, mu[ci], "k", lw=1)
        axx.set_title(nm); axx.grid(alpha=0.3)
        if ci == 0: axx.legend(fontsize=8)
    fig.supxlabel("k  [s/km]"); fig.supylabel("log P1D")
    fig.suptitle("Per-k spectrum: the cosmology signal (dark) is buried inside the tau0/z spread (light)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/norm_spectrum_spread.png", dpi=160); plt.close(fig)

    # --- fold-0 cosmology tracking ---
    try:
        model, meta, norm = load_checkpoint("checkpoints/loso_fold0")
        tr, va, ho = make_splits(d, 0, 8)
        b = make_batch(d, va, norm); pred = jax.vmap(model)(b["x"], b["tau0"])
        phys = untransform_prediction({k: np.asarray(pred[k]) for k in pred}, norm)
        lp = safe_log(np.asarray(phys["P_filt"])); lt = safe_log(d["P_filt"][va])
        cv = cell[va]
        dev_t, dev_p = [], []
        for cl in np.unique(cv):
            m = cv == cl
            if m.sum() < 2: continue
            dev_t.append((lt[m] - lt[m].mean(0)).ravel())
            dev_p.append((lp[m] - lp[m].mean(0)).ravel())
        dev_t = np.concatenate(dev_t); dev_p = np.concatenate(dev_p)
        g = np.isfinite(dev_t) & np.isfinite(dev_p)
        corr = np.corrcoef(dev_t[g], dev_p[g])[0, 1]
        slope = np.nanstd(dev_p[g]) / np.nanstd(dev_t[g])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.hexbin(dev_t[g], dev_p[g], gridsize=60, mincnt=1, cmap="viridis")
        lim = np.nanpercentile(np.abs(dev_t[g]), 99)
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("TRUE log-P1D deviation from (z,tau0)-cell mean  [cosmology signal]")
        ax.set_ylabel("PRED deviation from cell mean")
        ax.set_title(f"Fold-0 held-out cosmology tracking\ncorr={corr:.2f}, spread-ratio={slope:.2f} "
                     f"(1=perfect, 0=ignores cosmology)")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{FIGDIR}/fold0_cosmology_tracking.png", dpi=160); plt.close(fig)
        print(f"fold-0 corr={corr:.3f} spread-ratio={slope:.3f}")
    except Exception as e:
        print("fold-0 tracking skipped:", repr(e))

    print("cosmology log-var fraction (median/k):",
          {nm: float(np.nanmedian(frac[ci])) for ci, nm in enumerate(CLS)})
    print("figures ->", FIGDIR)


if __name__ == "__main__":
    main()
