#!/usr/bin/env python3
"""Gate B data-nuisance — per-cell CLEAN-vs-INJECTED posterior CORNER plots (READ-ONLY).

For each data-nuisance cell of the Gate B campaign we overlay the CLEAN-arm (blue) and the
INJECTED-arm (red) posteriors on a corner of (n_s, A_p, tau0_amp, dtau0[, a_SiIII]) and mark the
truth at the origin. The shift of the RED cloud off the origin is the cosmology bias the
contaminant induces — the same quantity the paired analyzer certifies as

    Δbias_z = bias_z(injected) − bias_z(clean),   bias_z = (post_mean − truth)/post_sd.

PAIRING / TRUTH HANDLING. The shards (scripts/run_dnuis_bias_shard.py) store a PAIRED estimator:
`clean_per_mock` + `inj_per_mock` (8 mocks/cell, 4 shards) sharing the SAME truth θ and the SAME
cosmic noise ε per mock — only the injected contaminant differs. Each mock's truth θ is its own
prior draw, so RAW stacking across mocks would mix 8 different truths and wash out the bias. We
therefore plot each draw as (param − truth_of_that_mock) and stack across all 8 mocks: truth sits
at the ORIGIN for every param, the clean cloud is centered there (Leg-A self-draw → ~0 bias by
construction) and the injected cloud's offset IS the bias. This pools ~500–690 draws/cell so the
KDE contours are well sampled, and is faithful to the analyzer (post − truth, per mock).

tau0. The sampler's 2-param τ₀ model is stored only as its 13 deterministic per-z outputs
tau0_z0..tau0_z12 (z0..z12 are ~monotone, corr(z0,z12)≈0.78). We surface it as two derived axes:
    tau0_amp  = tau0_z0                (the low-z amplitude — the τ₀ pivot)
    dtau0     = tau0_z12 − tau0_z0     (the high-minus-low z evolution — the slope proxy)
These are honest reductions of the stored deterministic block, labeled as derived.

a_SiIII. Present (col 26) for the metals-on surveys (DESI/eBOSS); ABSENT for KS (metals_on=False,
25 cols). The metal cells add it as a 5th corner axis; KS cells use the 4 cosmo/τ₀ axes.

METHOD. KDE 68/95% contours (full draws exist), reusing the prior-sensitivity helpers' style
(scripts/diag_desi_hcd_prior_sensitivity_figs.py): widened bandwidth + light grid smoothing so the
few-hundred-draw clouds read cleanly without inventing structure; HPD levels on the raw KDE grid
(honest mass fractions). Cells whose injected arm is draw-starved (resolution_ks: ~155 inj draws,
mock-0 = 3) are flagged ILL-CONDITIONED on the figure.

Per-cell Δbias_z (from scripts/analyze_dnuis_bias.py on the campaign shards) is printed in each
title so the visual shift can be read against the certified number.

READ-ONLY on the pkls. Writes ONLY new PNGs into the notes figure dir. Edits NO existing module.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 /home/mfho/.conda/envs/emu-jax/bin/python3
"""
import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# Slide-friendly fonts.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

CLEAN_C = "#1f5fa6"   # clean arm (blue)
INJ_C = "#c0392b"     # injected arm (red)
TRU_C = "#111111"     # truth markers (neutral)

# KDE on a few-hundred draws is blobby — widen the bandwidth and lightly smooth the grid so the
# contours read cleanly without inventing structure (HPD levels stay on the raw grid).
KDE_BW = 1.35
KDE_BW_2D = 1.45
GRID_SMOOTH = 1.1

BASE = "/scratch/cavestru_root/cavestru1/mfho/dnuis_bias"
FIG_DIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood/gate_b_corners"

# Per-cell certified PAIRED Δbias_z (ns, Ap) from scripts/analyze_dnuis_bias.py on the campaign.
# (ns, Ap) reproduced here only for the titles — the figure data are read fresh from the pkls.
DBIAS = {
    "metal_misspec_desi":  dict(ns=-0.963, Ap=-0.497, note=None),
    "metal_misspec_eboss": dict(ns=-0.544, Ap=-0.389, note=None),
    "resolution_desi":     dict(ns=-1.752, Ap=-0.825, note=None),
    "resolution_ks":       dict(ns=+0.299, Ap=-21.795,
                                note="ILL-CONDITIONED (b_res too large on KS high-k)"),
    "lls_excess_desi":     dict(ns=-0.186, Ap=+0.068, note="FLAG (<=0.50 sigma HCD->n_s budget)"),
    "lls_excess_ks":       dict(ns=-1.246, Ap=+1.007, note=None),
    "metal_matched_desi":  dict(ns=+0.000, Ap=+0.000, note="NULL CONTROL (clean == injected)"),
    "metal_matched_eboss": dict(ns=+0.000, Ap=+0.000, note="NULL CONTROL (clean == injected)"),
}

CELL_TITLE = {
    "metal_misspec_desi":  "metal_misspec : DESI  (unfittable additive SiII-SiII term)",
    "metal_misspec_eboss": "metal_misspec : eBOSS  (McDonald SiIIIcorr at the eBOSS scale)",
    "resolution_desi":     "resolution : DESI  (P*exp(2 b_res k^2 R^2), forward resolution_on=False)",
    "resolution_ks":       "resolution : KS  (b_res distortion on KS high-k)",
    "lls_excess_desi":     "lls_excess : DESI  (truth LLS off the per-survey pin, 1.06x)",
    "lls_excess_ks":       "lls_excess : KS  (truth LLS ~1 sigma above the 2.5x KS pin, 3.5x)",
    "metal_matched_desi":  "metal_matched : DESI  (marginalization-cost null; a_SiIII free)",
    "metal_matched_eboss": "metal_matched : eBOSS  (marginalization-cost null; a_SiIII free)",
}

LABS = {
    "ns": r"$\Delta n_s$",
    "Ap": r"$\Delta A_p$",
    "tau0_amp": r"$\Delta\tau_{0}$ (amp, $z_0$)",
    "dtau0": r"$\Delta(d\tau_0)$  [$z_{12}{-}z_0$]",
    "a_SiIII": r"$\Delta a_{\rm SiIII}$",
}


# --------------------------------------------------------------------- loaders / derived columns
def load_cell(cell):
    """Load + concatenate all shards of a cell. Returns (clean_recs, inj_recs, meta)."""
    fs = sorted(glob.glob(os.path.join(BASE, f"{cell}_shard_*.pkl")))
    if not fs:
        raise SystemExit(f"no shard pkls for cell {cell!r} in {BASE}")
    clean, inj, meta = [], [], None
    for p in fs:
        with open(p, "rb") as f:
            d = pickle.load(f)
        clean.extend(d["clean_per_mock"])
        inj.extend(d["inj_per_mock"])
        meta = meta or d.get("meta", {})
    return clean, inj, meta


def _derived(rec, key):
    """Return (draws_col, truth_scalar) for a corner axis from one per-mock record.
    Cosmo/τ₀ axes are read by name from the deterministic block; tau0_amp/dtau0 are derived."""
    names = list(rec["names"])
    draws = np.asarray(rec["draws"], float)
    tv = np.asarray(rec["truth_vec"], float)
    if key == "tau0_amp":
        j = names.index("tau0_z0")
        return draws[:, j], float(tv[j])
    if key == "dtau0":
        j0, j1 = names.index("tau0_z0"), names.index("tau0_z12")
        return draws[:, j1] - draws[:, j0], float(tv[j1] - tv[j0])
    if key not in names:
        return None, None
    j = names.index(key)
    return draws[:, j], float(tv[j])


def stacked_residuals(recs, cols):
    """Stack (draw − truth_of_that_mock) over all mocks for each requested column.
    Truth sits at 0 by construction; the cloud's offset from 0 is the bias. Returns
    {col: 1-D array of residuals pooled over mocks}."""
    out = {c: [] for c in cols}
    for rec in recs:
        for c in cols:
            d, t = _derived(rec, c)
            if d is None:
                continue
            out[c].append(np.asarray(d) - t)
    return {c: (np.concatenate(v) if v else np.array([])) for c, v in out.items()}


# --------------------------------------------------------------------- KDE helpers (reused style)
def kde_1d(ax, x, color, lw=2.2, fill_alpha=0.18):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.ptp(x) == 0:
        if x.size:
            ax.axvline(float(np.median(x)), color=color, lw=lw)
        return
    pad = np.ptp(x) * 0.10 + 1e-9
    g = np.linspace(x.min() - pad, x.max() + pad, 256)
    try:
        d = gaussian_kde(x, bw_method=KDE_BW)(g)
    except np.linalg.LinAlgError:
        ax.axvline(float(np.median(x)), color=color, lw=lw)
        return
    ax.fill_between(g, d, color=color, alpha=fill_alpha, zorder=1)
    ax.plot(g, d, color=color, lw=lw, zorder=3, solid_capstyle="round")


def kde_contour(ax, x, y, color, fill_alpha=0.20, lw=2.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 6 or np.ptp(x) == 0 or np.ptp(y) == 0:
        if x.size:
            ax.scatter(x, y, s=14, color=color, alpha=0.7, zorder=3, edgecolors="none")
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=KDE_BW_2D)
    except np.linalg.LinAlgError:
        ax.scatter(x, y, s=14, color=color, alpha=0.7, zorder=3, edgecolors="none")
        return
    px = np.ptp(x) * 0.30 + 1e-9
    py = np.ptp(y) * 0.30 + 1e-9
    xx, yy = np.mgrid[x.min() - px:x.max() + px:140j, y.min() - py:y.max() + py:140j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = gaussian_filter(zz, GRID_SMOOTH)
    zs = np.sort(zz.ravel())[::-1]
    cum = np.cumsum(zs) / zs.sum()
    lv = sorted(zs[np.searchsorted(cum, L)] for L in (0.95, 0.68))
    if lv[0] == lv[1]:
        lv[1] = lv[1] * 1.0001 + 1e-12
    ax.contourf(xx, yy, zz, levels=lv + [zz.max()], colors=[color, color],
                alpha=fill_alpha, zorder=1)
    ax.contour(xx, yy, zz, levels=lv, colors=color, linestyles="-",
               linewidths=[lw * 0.8, lw], zorder=3)


# --------------------------------------------------------------------- the corner
def make_corner(cell, path):
    clean, inj, meta = load_cell(cell)
    has_metal = any("a_SiIII" in list(r["names"]) for r in clean)
    cols = ["ns", "Ap", "tau0_amp", "dtau0"] + (["a_SiIII"] if has_metal else [])

    rc = stacked_residuals(clean, cols)
    ri = stacked_residuals(inj, cols)
    n_clean = max((v.size for v in rc.values()), default=0)
    n_inj = max((v.size for v in ri.values()), default=0)

    db = DBIAS[cell]
    P = len(cols)
    fig, ax = plt.subplots(P, P, figsize=(3.1 * P, 3.1 * P))
    for r in range(P):
        for c in range(P):
            a = ax[r, c]
            if c > r:
                a.axis("off")
                continue
            ci, cj = cols[r], cols[c]
            if r == c:
                kde_1d(a, rc[ci], CLEAN_C, lw=2.0, fill_alpha=0.14)
                kde_1d(a, ri[ci], INJ_C, lw=2.4, fill_alpha=0.20)
                a.axvline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=1.6, zorder=4)
                a.set_yticks([])
                a.set_ylim(bottom=0)
            else:
                kde_contour(a, rc[cj], rc[ci], CLEAN_C, fill_alpha=0.16, lw=1.9)
                kde_contour(a, ri[cj], ri[ci], INJ_C, fill_alpha=0.24, lw=2.2)
                a.axvline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=1.0, zorder=4)
                a.axhline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=1.0, zorder=4)
                a.plot(0.0, 0.0, marker="*", ms=13, color=TRU_C, mec="white",
                       mew=0.8, zorder=5)
            a.tick_params(length=3)
            a.grid(True, color="0.9", lw=0.5, zorder=0)
            if r == P - 1:
                a.set_xlabel(LABS[cols[c]])
                a.tick_params(axis="x", labelrotation=30)
                for lbl in a.get_xticklabels():
                    lbl.set_ha("right")
            else:
                a.set_xticklabels([])
            if c == 0 and r > 0:
                a.set_ylabel(LABS[cols[r]])
            elif r != c:
                a.set_yticklabels([])

    handles = [
        plt.Line2D([], [], color=CLEAN_C, lw=4, label=f"CLEAN arm (no injection)  N={n_clean} draws"),
        plt.Line2D([], [], color=INJ_C, lw=4, label=f"INJECTED arm (contaminant)  N={n_inj} draws"),
        plt.Line2D([], [], color=TRU_C, ls=(0, (4, 2)), lw=1.6, marker="*", ms=13,
                   mec="white", mew=0.8, label="truth (origin; per-mock truth subtracted)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985),
               fontsize=14, frameon=True, framealpha=0.95, edgecolor="0.8",
               borderpad=0.8, labelspacing=0.7)

    sub = (f"Gate B data-nuisance  —  {CELL_TITLE[cell]}\n"
           rf"paired $\Delta$bias$_z$:  $n_s$ = {db['ns']:+.2f}$\sigma$,   "
           rf"$A_p$ = {db['Ap']:+.2f}$\sigma$   "
           "(contours 68/95%; axes are draw $-$ per-mock truth, pooled over 8 mocks)")
    if db["note"]:
        sub += f"\n[{db['note']}]"
    note_color = "#b30000" if (db["note"] and "ILL" in db["note"]) else "#222222"
    fig.suptitle(sub, fontsize=16, y=0.995, color=note_color)

    # For the ILL-CONDITIONED cell the pooled (draw - truth) clouds look deceptively tame: the
    # -21.8 sigma is NOT a large pooled mean shift but a per-mock posterior-WIDTH COLLAPSE (the KS
    # high-k b_res distortion makes the likelihood near-singular, so NUTS stalls — most injected
    # mocks keep only 3-7 draws with A_p sd ~0.01-0.02, giving bias_z of -30..-63 sigma each).
    # Annotate so the figure does not understate the pathology.
    if db["note"] and "ILL" in db["note"]:
        fig.text(0.5, 0.965,
                 "pooled clouds look tame, but the gate blows up from per-mock POSTERIOR-WIDTH "
                 "COLLAPSE:\ninjected A_p sd ~0.01-0.02 with only 3-7 kept draws/mock "
                 "(near-singular likelihood) -> bias_z -30..-63 sigma/mock",
                 ha="center", va="top", fontsize=11.5, color="#b30000",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff3f3", ec="#b30000", lw=1.0,
                           alpha=0.95))

    fig.tight_layout(rect=(0, 0, 1, 0.905 if (db["note"] and "ILL" in db["note"])
                           else (0.93 if db["note"] else 0.945)))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return n_clean, n_inj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--cells", nargs="*", default=[
        "metal_misspec_desi", "metal_misspec_eboss", "resolution_desi", "resolution_ks",
        "lls_excess_desi", "lls_excess_ks", "metal_matched_desi"])
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    for cell in a.cells:
        path = os.path.join(a.fig_dir, f"dnuis_corner_{cell}.png")
        nc, ni = make_corner(cell, path)
        print(f"[corner] {cell:22s} clean={nc:4d} inj={ni:4d} draws -> {path}")


if __name__ == "__main__":
    main()
