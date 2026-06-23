#!/usr/bin/env python3
"""DESI leg — HCD incidence-prior sensitivity figures (PI, 2026-06-21).

WITH (deployed widths 0.15/0.40/0.50) vs WITHOUT (~flat 5/5/5) the per-class HCD
incidence prior on the DESI leg. Three deliverables in a KDE-contour style (contours only,
NO scatter points; KDE 1-D marginals; 68/95% HPD). The ~200-draw KDE is blobby, so the
bandwidth is widened (KDE_BW/KDE_BW_2D) and the 2-D grid lightly smoothed (GRID_SMOOTH) so
the contours read cleanly while the HPD mass fractions stay honest. Truth = neutral star/dashes.

  FIG 1 (headline) corner of (n_s, A_p, alpha_LLS, alpha_subDLA, alpha_DLA) on the MATCHED
        mock 0 (identical data, only the prior differs): WITH-prior (blue, pinned, round,
        decoupled) over WITHOUT-prior (red, floats; the cosmology<->HCD contours open / tilt).
  FIG 2 A_p, n_s 1-D marginals (KDE) overlaid + the posterior shift in sigma + the err change.
  FIG 3 recovered per-class dN/dX(z) (LLS/subDLA/DLA), with vs without, vs the prior center
        (observed dN/dX pin) and the sim truth.

READ-ONLY on the pkls. Writes only into the figure dir. The OFF arm is PARTIAL -> the
cross-mock aggregate (FIG 2) is PRELIMINARY and pinned to the doc's N=2 snapshot via
--off-limit; the matched mock-0 corner (FIG 1) and the WITH-prior aggregate are solid.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3
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

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

ON_C = "#1f5fa6"   # with prior (pinned, solid blue)
OFF_C = "#c0392b"  # without prior (free, red)
TRU_C = "#111111"  # truth markers (neutral, readable on both fills)
COLS = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"]
LABS = {"ns": r"$n_s$", "Ap": r"$A_p$", "alpha_lls": r"$\alpha_{\rm LLS}$",
        "alpha_subdla": r"$\alpha_{\rm subDLA}$", "alpha_dla": r"$\alpha_{\rm DLA}$"}
CLASS_C = {"LLS": "#1f5fa6", "subDLA": "#2c8c4a", "DLA": "#e08214"}

# KDE on ~200 draws is blobby; widen the bandwidth and smooth the grid so contours
# read cleanly without inventing structure. Bandwidth factor applied on top of Scott.
KDE_BW = 1.35    # 1-D marginals
KDE_BW_2D = 1.45  # 2-D contours (a touch wider — 200 pts in 2-D is sparser)
GRID_SMOOTH = 1.1  # Gaussian smoothing (grid cells) of the density before contouring


# --------------------------------------------------------------------- loaders
def load_arm(out_dir, limit=None):
    """Load mock_*.pkl in name order. `limit` caps the count — the OFF arm is still
    running, so we pin the aggregate to the N the companion doc is written against
    (keeps figures and the doc's PRELIMINARY numbers in lock-step; raise --off-limit
    once the doc is updated)."""
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "mock_*.pkl"))):
        with open(p, "rb") as f:
            recs.append(pickle.load(f))
    if not recs:
        raise SystemExit(f"no mock_*.pkl in {out_dir}")
    if limit is not None:
        recs = recs[:limit]
    return recs


def _col(rec, key):
    nm = list(rec["names"])
    return np.asarray(rec["draws"])[:, nm.index(key)]


def _truth(rec, key):
    nm = list(rec["names"])
    return float(np.asarray(rec["truth_vec"])[nm.index(key)])


def _get_mock(recs, m):
    for r in recs:
        if int(r.get("mock", -1)) == int(m):
            return r
    raise SystemExit(f"mock {m} not found")


# --------------------------------------------------------------------- KDE helpers
def kde_1d(ax, x, color, lw=2.0, ls="-", fill=True, fill_alpha=0.16):
    """Smoothed 1-D KDE marginal. Widened bandwidth tames ~200-draw blobbiness."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    pad = (x.max() - x.min()) * 0.06 + 1e-9
    g = np.linspace(x.min() - pad, x.max() + pad, 256)
    d = gaussian_kde(x, bw_method=KDE_BW)(g)
    if fill:
        ax.fill_between(g, d, color=color, alpha=fill_alpha, zorder=1)
    ax.plot(g, d, color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="round")


def kde_contour(ax, x, y, color, ls="-", fill=True, fill_alpha=0.20, lw=2.0):
    """Smoothed 2-D KDE 68/95% contours (no scatter points). Widened bandwidth +
    light grid smoothing keep the contours legible on ~200 draws without inventing
    structure; HPD levels are computed on the raw KDE grid (honest mass fractions)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=KDE_BW_2D)
    px = (x.max() - x.min()) * 0.30 + 1e-9
    py = (y.max() - y.min()) * 0.30 + 1e-9
    xx, yy = np.mgrid[x.min() - px:x.max() + px:140j, y.min() - py:y.max() + py:140j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = gaussian_filter(zz, GRID_SMOOTH)
    zs = np.sort(zz.ravel())[::-1]
    cum = np.cumsum(zs) / zs.sum()
    lv = sorted(zs[np.searchsorted(cum, L)] for L in (0.95, 0.68))
    if fill:
        ax.contourf(xx, yy, zz, levels=lv + [zz.max()],
                    colors=[color, color], alpha=fill_alpha, zorder=1)
    ax.contour(xx, yy, zz, levels=lv, colors=color, linestyles=ls,
               linewidths=[lw * 0.8, lw], zorder=3)


# --------------------------------------------------------------------- FIG 1: headline corner
def fig_corner(on, off, mock, path):
    ro = _get_mock(on, mock); rf = _get_mock(off, mock)
    data = {"on": {c: _col(ro, c) for c in COLS}, "off": {c: _col(rf, c) for c in COLS}}
    truth = {c: _truth(ro, c) for c in COLS}
    P = len(COLS)
    fig, ax = plt.subplots(P, P, figsize=(2.85 * P, 2.85 * P))
    for r in range(P):
        for c in range(P):
            a = ax[r, c]
            if c > r:
                a.axis("off"); continue
            ci, cj = COLS[r], COLS[c]
            if r == c:
                # WITHOUT first (wide, light), WITH on top (pinned, bolder fill).
                kde_1d(a, data["off"][ci], OFF_C, lw=1.9, fill_alpha=0.12)
                kde_1d(a, data["on"][ci], ON_C, lw=2.3, fill_alpha=0.22)
                a.axvline(truth[ci], color=TRU_C, ls=(0, (4, 2)), lw=1.5, zorder=4)
                a.set_yticks([]); a.set_ylim(bottom=0)
            else:
                kde_contour(a, data["off"][cj], data["off"][ci], OFF_C,
                            fill_alpha=0.14, lw=1.8)
                kde_contour(a, data["on"][cj], data["on"][ci], ON_C,
                            fill_alpha=0.26, lw=2.2)
                a.axvline(truth[cj], color=TRU_C, ls=(0, (4, 2)), lw=1.0, zorder=4)
                a.axhline(truth[ci], color=TRU_C, ls=(0, (4, 2)), lw=1.0, zorder=4)
                a.plot(truth[cj], truth[ci], marker="*", ms=11, color=TRU_C,
                       mec="white", mew=0.7, zorder=5)
            a.tick_params(length=3)
            a.grid(True, color="0.9", lw=0.5, zorder=0)
            if r == P - 1:
                a.set_xlabel(LABS[COLS[c]], fontsize=13)
                a.tick_params(axis="x", labelrotation=30)
                for lbl in a.get_xticklabels():
                    lbl.set_ha("right")
            else:
                a.set_xticklabels([])
            if c == 0 and r > 0:
                a.set_ylabel(LABS[COLS[r]], fontsize=13)
            elif r != c:
                a.set_yticklabels([])
    handles = [
        plt.Line2D([], [], color=ON_C, lw=3,
                   label=r"WITH HCD prior  ($\sigma$=0.15/0.40/0.50) — pinned"),
        plt.Line2D([], [], color=OFF_C, lw=3,
                   label=r"WITHOUT HCD prior  ($\sigma{\approx}$5 flat) — floats"),
        plt.Line2D([], [], color=TRU_C, ls=(0, (4, 2)), lw=1.5, marker="*",
                   ms=11, mec="white", mew=0.7, label="truth"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.97),
               fontsize=14, frameon=True, framealpha=0.95, edgecolor="0.8",
               borderpad=0.8, labelspacing=0.7)
    fig.suptitle(
        f"DESI leg — matched mock {mock} (identical data, only the HCD prior differs)\n"
        r"removing the prior lets the HCD amplitudes float; the cosmology$\leftrightarrow$HCD "
        r"contours open and tilt  (contours 68/95%)",
        fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- FIG 2: cosmology shift
def fig_cosmo_shift(on, off, path, off_n):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    out = {}
    for ax, par in zip(axes, ("ns", "Ap")):
        s_on = np.concatenate([_col(r, par) for r in on])
        s_off = np.concatenate([_col(r, par) for r in off])
        kde_1d(ax, s_off, OFF_C, lw=2.0, fill_alpha=0.14)
        kde_1d(ax, s_on, ON_C, lw=2.5, fill_alpha=0.22)
        t = _truth(on[0], par)
        ax.axvline(t, color=TRU_C, ls=(0, (4, 2)), lw=1.6, zorder=4)
        mo, so = s_on.mean(), s_on.std(ddof=1)
        mf, sf = s_off.mean(), s_off.std(ddof=1)
        out[par] = dict(on=(mo, so), off=(mf, sf), shift=(mf - mo) / so, ratio=sf / so)
        ax.set_xlabel(LABS[par]); ax.set_ylabel("posterior density (KDE)")
        ax.set_ylim(bottom=0)
        ax.grid(True, color="0.92", lw=0.5, zorder=0)
        ax.set_title(f"{LABS[par]}:  aggregate shift {(mf-mo)/so:+.2f}$\\sigma$,  "
                     f"$\\sigma_{{\\rm off}}/\\sigma_{{\\rm on}}$ = {sf/so:.2f}")
    handles = [
        plt.Line2D([], [], color=ON_C, lw=3, label="with HCD prior  (N=%d)" % len(on)),
        plt.Line2D([], [], color=OFF_C, lw=3,
                   label="without HCD prior  (N=%d, PRELIM)" % off_n),
        plt.Line2D([], [], color=TRU_C, ls=(0, (4, 2)), lw=1.6, label="truth"),
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=True,
                   framealpha=0.95, edgecolor="0.8")
    fig.suptitle(r"DESI leg — ($n_s$, $A_p$) aggregate marginals, with vs without the HCD "
                 "incidence prior   [OFF arm PRELIMINARY]", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------- FIG 3: dN/dX(z)
def fig_dndx(on, off, path, mock, off_n):
    ro = _get_mock(on, mock); rf = _get_mock(off, mock)
    z = np.asarray(ro["dndx_z"], float)
    order = list(ro["dndx_class_order"])
    # prior-center dN/dX(z) ~ sim-truth scaled by mu/alpha_pivot_truth (amplitude pin; the
    # observed-dN/dX center the incidence prior pulls toward).
    mu = np.asarray(ro["alpha_hcd_mu"], float)
    apt = np.asarray(ro["alpha_pivot_truth"], float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    for ci, (cls, ax) in enumerate(zip(order, axes)):
        for arm, rec, col, lab in (("off", rf, OFF_C, "without prior (free)"),
                                   ("on", ro, ON_C, "with prior (pinned)")):
            dd = np.asarray(rec["dndx_draws"])[:, :, ci]
            med = np.median(dd, axis=0)
            lo, hi = np.percentile(dd, [16, 84], axis=0)
            ax.fill_between(z, lo, hi, color=col, alpha=0.18, zorder=1)
            ax.plot(z, med, color=col, lw=2.4, label=lab, zorder=3,
                    solid_capstyle="round")
        tr = np.asarray(ro["dndx_truth"])[:, ci]
        ax.plot(z, tr, color="k", ls=(0, (5, 2)), lw=1.9, label="sim truth", zorder=4)
        center = tr * (mu[ci] / apt[ci])
        ax.plot(z, center, color=TRU_C, ls=(0, (1, 1.5)), lw=1.8,
                label="prior center (obs. dN/dX)", zorder=4)
        ax.set_title(f"{cls}", color=CLASS_C[cls], fontsize=14, fontweight="bold")
        ax.set_xlabel("redshift  $z$")
        ax.grid(True, color="0.92", lw=0.5, zorder=0)
        if ci == 0:
            ax.set_ylabel("dN/dX  (recovered / truth-normalised)")
        # cap y at a readable range so the WITHOUT-prior high-z z-slope runaway (DESI does not
        # constrain dN/dX there) does not flatten the comparison; annotate where the band clips.
        ymax = float(np.percentile(np.asarray(rf["dndx_draws"])[:, :, ci], 84, axis=0).max())
        ytop = min(ymax * 1.25, 4.0 * float(tr.max()))
        ax.set_ylim(0, ytop)
        if ymax > ytop:
            ax.annotate("WITHOUT-prior band runs\noff-scale (z-slope\nunconstrained at high z)",
                        xy=(0.97, 0.62), xycoords="axes fraction", ha="right", va="top",
                        fontsize=9, color=OFF_C,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OFF_C, lw=0.8,
                                  alpha=0.9))
        ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="0.8")
    fig.suptitle(f"DESI leg — recovered per-class dN/dX(z), matched mock {mock}   "
                 r"(WITH $\sigma$=0.15/0.40/0.50  vs  WITHOUT $\sigma{\approx}$5 flat);  16-84% band",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-dir", default="/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_on")
    ap.add_argument("--off-dir", default="/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_off")
    ap.add_argument("--mock", type=int, default=0)
    # OFF arm is still landing; pin the aggregate to the doc's N=2 PRELIMINARY snapshot
    # (mock_0000 + mock_0001) so figures match the written numbers. Raise to refresh.
    ap.add_argument("--off-limit", type=int, default=2,
                    help="cap OFF-arm mocks in the aggregate (doc snapshot = 2)")
    ap.add_argument("--fig-dir",
                    default="/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation/desi_hcd_prior")
    ap.add_argument("--tag", default="",
                    help="filename suffix for the corner + dN/dX (e.g. _mock3 for a per-mock gallery)")
    ap.add_argument("--no-aggregate", action="store_true", help="skip the (mock-independent) cosmo-shift fig")
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    on = load_arm(a.on_dir); off = load_arm(a.off_dir, limit=a.off_limit)
    off_n = len(off)
    print(f"[plot] ON N={len(on)}  OFF N={off_n}  mock={a.mock} tag='{a.tag}'")
    pre = "desi_hcd_prior"
    p1 = os.path.join(a.fig_dir, f"{pre}_corner{a.tag}.png")
    p2 = os.path.join(a.fig_dir, f"{pre}_cosmo_shift.png")
    p3 = os.path.join(a.fig_dir, f"{pre}_dndx{a.tag}.png")
    fig_corner(on, off, a.mock, p1)
    stats = fig_cosmo_shift(on, off, p2, off_n) if not a.no_aggregate else {}
    fig_dndx(on, off, p3, a.mock, off_n)
    for par, sd in stats.items():
        print(f"  {par}: ON {sd['on'][0]:.5f}+-{sd['on'][1]:.5f} | OFF {sd['off'][0]:.5f}+-{sd['off'][1]:.5f} "
              f"| shift={sd['shift']:+.2f}sig ratio={sd['ratio']:.2f}")
    print("[plot] wrote:\n ", p1, "\n ", p2, "\n ", p3)


if __name__ == "__main__":
    main()
