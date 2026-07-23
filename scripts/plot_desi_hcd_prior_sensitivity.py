"""Build the 3 deliverable figures for the DESI HCD-prior sensitivity study (PI, 2026-06-21).

WITH vs WITHOUT the HCD incidence prior on the DESI leg:
  FIG 1  A_p, n_s 1-D marginals overlaid (KDE), + the posterior-mean shift in sigma units and the
         error-bar change -- does dropping the prior shift or merely inflate cosmology?
  FIG 2  Corner / contour of (A_p, n_s, alpha_LLS, alpha_subDLA, alpha_DLA), with vs without prior
         overlaid, 68/95% smoothed CONTOURS, truth markers.
  FIG 3  Recovered per-class dN/dX(z) (LLS, subDLA, DLA): with vs without prior, vs the prior center
         / observed dN/dX and the sim truth.

FIGURE STYLE (mandated, this doc ONLY): smoothed KDE contours (NOT 2-D histograms); 1-D marginals
KDE-smoothed (NOT histograms). Backend: getdist if installed, else scipy.stats.gaussian_kde +
matplotlib. THIS ENV HAS NO getdist -> the scipy/KDE backend is wired below (auto-detected).

DO NOT RUN until the on/off pkls have landed in both OUTDIRs. This is the STUB the analysis agent
calls; the KDE/contour helpers are complete.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3
Usage:
  python scripts/plot_desi_hcd_prior_sensitivity.py \
      --on-dir  /scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_on \
      --off-dir /scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_off \
      --corner-mock 0 \
      --fig-dir /home/mfho/hcd_priya_notes/figures/analysis
"""
import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    import getdist  # noqa: F401
    HAVE_GETDIST = True
except Exception:
    HAVE_GETDIST = False

# Pretty labels (class order [LLS, subDLA, DLA]).
LABELS = {"Ap": r"$A_p$", "ns": r"$n_s$",
          "alpha_lls": r"$\alpha_{\rm LLS}$",
          "alpha_subdla": r"$\alpha_{\rm subDLA}$",
          "alpha_dla": r"$\alpha_{\rm DLA}$"}
ARM_COLOR = {"on": "C0", "off": "C3"}
ARM_LABEL = {"on": "with HCD prior (0.15/0.40/0.50)", "off": "without HCD prior (~flat 5/5/5)"}
CLASS_ORDER = ("LLS", "subDLA", "DLA")
CLASS_COLOR = {"LLS": "C0", "subDLA": "C2", "DLA": "C1"}


# ------------------------------------------------------------------- loaders
def load_arm(out_dir):
    """Load all mock pkls in an arm dir -> list of records (sorted by mock index)."""
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "mock_*.pkl"))):
        with open(p, "rb") as f:
            recs.append(pickle.load(f))
    if not recs:
        raise SystemExit(f"no mock_*.pkl in {out_dir}")
    return recs


def _col(rec, key):
    names = list(rec["names"])
    return np.asarray(rec["draws"])[:, names.index(key)]


def _stack_param(recs, key):
    """Concatenate one param's draws across all mocks in an arm (the aggregate marginal)."""
    return np.concatenate([_col(r, key) for r in recs])


# ------------------------------------------------------------------- KDE helpers (scipy backend)
def kde_1d(samples, grid=None, n=400, bw=None):
    """KDE-smoothed 1-D density on a grid (NOT a histogram). Returns (x, density)."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    k = gaussian_kde(s, bw_method=bw)
    if grid is None:
        lo, hi = s.min(), s.max()
        pad = 0.15 * (hi - lo + 1e-12)
        grid = np.linspace(lo - pad, hi + pad, n)
    return grid, k(grid)


def kde_contour_levels(density_xy, levels=(0.68, 0.95)):
    """Density thresholds enclosing the requested HPD probability mass (for a gridded KDE)."""
    flat = np.sort(density_xy.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [float(flat[np.searchsorted(csum, lv)]) for lv in levels][::-1]   # ascending for contour


def kde_2d(x, y, gridsize=120, pad=0.15):
    """2-D KDE on a regular grid. Returns (XX, YY, ZZ)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    k = gaussian_kde(np.vstack([x, y]))
    xlo, xhi = x.min(), x.max(); ylo, yhi = y.min(), y.max()
    px = pad * (xhi - xlo + 1e-12); py = pad * (yhi - ylo + 1e-12)
    xs = np.linspace(xlo - px, xhi + px, gridsize)
    ys = np.linspace(ylo - py, yhi + py, gridsize)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return XX, YY, ZZ


def _contour_2d(ax, x, y, color, levels=(0.68, 0.95), fill=True, label=None):
    """Smoothed KDE 68/95% contours (NOT a 2-D histogram) for one arm onto ax."""
    XX, YY, ZZ = kde_2d(x, y)
    clev = kde_contour_levels(ZZ, levels=levels)
    if fill:
        ax.contourf(XX, YY, ZZ, levels=clev + [ZZ.max()], colors=[color, color],
                    alpha=0.18)
    ax.contour(XX, YY, ZZ, levels=clev, colors=color, linewidths=1.6)
    if label is not None:
        ax.plot([], [], color=color, lw=1.6, label=label)


# ------------------------------------------------------------------- FIG 1: A_p, n_s 1-D + shift
def fig_cosmo_shift(arms, fig_path):
    """arms: {'on': recs, 'off': recs}. KDE 1-D marginals of n_s, A_p overlaid + the shift stat."""
    params = ["ns", "Ap"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stats = {}
    for ax, par in zip(axes, params):
        for arm in ("on", "off"):
            s = _stack_param(arms[arm], par)
            x, dens = kde_1d(s)
            ax.plot(x, dens, color=ARM_COLOR[arm], lw=2, label=ARM_LABEL[arm])
            ax.fill_between(x, dens, color=ARM_COLOR[arm], alpha=0.12)
            stats.setdefault(par, {})[arm] = (float(np.mean(s)), float(np.std(s)))
        # truth marker (shared across arms; take from the 'on' arm's first record)
        tv = arms["on"][0]["truth_vec"]; nm = list(arms["on"][0]["names"])
        ax.axvline(float(tv[nm.index(par)]), color="k", ls="--", lw=1.3, label="truth")
        ax.set_xlabel(LABELS[par]); ax.set_ylabel("posterior density (KDE)")
        m_on, s_on = stats[par]["on"]; m_off, s_off = stats[par]["off"]
        # shift in sigma-units (use the WITH-prior width as the reference scale)
        shift_sig = (m_off - m_on) / (s_on + 1e-30)
        err_ratio = s_off / (s_on + 1e-30)
        ax.set_title(f"{LABELS[par]}: mean shift {shift_sig:+.2f}$\\sigma$, "
                     f"$\\sigma_{{off}}/\\sigma_{{on}}$={err_ratio:.2f}")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("DESI leg -- (A_p, n_s) with vs without the HCD incidence prior", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return stats


# ------------------------------------------------------------------- FIG 2: corner / contours
def fig_corner(arms, corner_mock, fig_path):
    """Corner of (Ap, ns, alpha_lls, alpha_subdla, alpha_dla) for ONE representative mock fit BOTH
    ways (identical data, two priors). KDE 1-D on the diagonal, KDE 68/95% contours off-diagonal."""
    cols = ["Ap", "ns", "alpha_lls", "alpha_subdla", "alpha_dla"]
    # pick the matched mock record from each arm
    def _get(arm):
        for r in arms[arm]:
            if int(r.get("mock", -1)) == int(corner_mock):
                return r
        raise SystemExit(f"corner mock {corner_mock} not found in arm '{arm}'")
    recs = {arm: _get(arm) for arm in ("on", "off")}
    data = {arm: {c: _col(recs[arm], c) for c in cols} for arm in ("on", "off")}
    tv = recs["on"]["truth_vec"]; nm = list(recs["on"]["names"])
    truth = {c: float(tv[nm.index(c)]) for c in cols}

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            ci, cj = cols[i], cols[j]
            if i == j:
                for arm in ("on", "off"):
                    x, dens = kde_1d(data[arm][ci])
                    ax.plot(x, dens, color=ARM_COLOR[arm], lw=1.8,
                            label=ARM_LABEL[arm] if (i == 0) else None)
                    ax.fill_between(x, dens, color=ARM_COLOR[arm], alpha=0.12)
                ax.axvline(truth[ci], color="k", ls="--", lw=1.1)
                ax.set_yticks([])
            else:
                for arm in ("on", "off"):
                    _contour_2d(ax, data[arm][cj], data[arm][ci], ARM_COLOR[arm])
                ax.plot(truth[cj], truth[ci], "k*", ms=11, mec="w")
            if i == n - 1:
                ax.set_xlabel(LABELS[cols[j]])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[cols[i]])
            elif j != 0:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=ARM_COLOR[a], lw=2, label=ARM_LABEL[a]) for a in ("on", "off")]
    handles.append(plt.Line2D([], [], color="k", marker="*", ls="none", ms=11, label="truth"))
    fig.legend(handles=handles, loc="upper right", fontsize=11)
    fig.suptitle(f"DESI leg corner -- mock {corner_mock} (identical data), with vs without HCD prior",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- FIG 3: dN/dX(z) per class
def fig_dndx(arms, fig_path):
    """Recovered per-class dN/dX(z): with vs without prior (aggregate posterior band over mocks),
    vs the sim truth + the prior-center pin. 3 panels (LLS, subDLA, DLA)."""
    z = np.asarray(arms["on"][0]["dndx_z"], float)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ci, (cls, ax) in enumerate(zip(CLASS_ORDER, axes)):
        for arm in ("on", "off"):
            # aggregate draws across mocks: (sum L, nZ)
            dd = np.concatenate([np.asarray(r["dndx_draws"])[:, :, ci] for r in arms[arm]], axis=0)
            med = np.median(dd, axis=0)
            lo, hi = np.percentile(dd, [16, 84], axis=0)
            ax.plot(z, med, color=ARM_COLOR[arm], lw=2, label=ARM_LABEL[arm])
            ax.fill_between(z, lo, hi, color=ARM_COLOR[arm], alpha=0.18)
        # sim truth (mean across mocks -- each mock is its own sim, but the class shape is stable)
        tr = np.mean([np.asarray(r["dndx_truth"])[:, ci] for r in arms["on"]], axis=0)
        ax.plot(z, tr, color="k", ls="--", lw=1.6, label="sim truth (mean)")
        ax.set_title(f"dN/dX -- {cls}"); ax.set_xlabel("z")
        if ci == 0:
            ax.set_ylabel("dN/dX")
        ax.legend(fontsize=9)
    fig.suptitle("DESI leg -- recovered per-class dN/dX(z), with vs without the HCD prior", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_figures(on_dir, off_dir, fig_dir, corner_mock=0, prefix="2026-06-21-desi-hcd-prior"):
    """The single entry point the analysis agent calls once the pkls land. Returns the cosmo-shift
    stats dict + the 3 figure paths."""
    os.makedirs(fig_dir, exist_ok=True)
    arms = {"on": load_arm(on_dir), "off": load_arm(off_dir)}
    backend = "getdist" if HAVE_GETDIST else "scipy.gaussian_kde"
    print(f"[plot] backend={backend} | on={len(arms['on'])} mocks, off={len(arms['off'])} mocks")
    p1 = os.path.join(fig_dir, f"{prefix}_cosmo_shift.png")
    p2 = os.path.join(fig_dir, f"{prefix}_corner.png")
    p3 = os.path.join(fig_dir, f"{prefix}_dndx.png")
    stats = fig_cosmo_shift(arms, p1)
    fig_corner(arms, corner_mock, p2)
    fig_dndx(arms, p3)
    for par, sd in stats.items():
        m_on, s_on = sd["on"]; m_off, s_off = sd["off"]
        print(f"  {par}: on mean={m_on:.4f} sd={s_on:.4f} | off mean={m_off:.4f} sd={s_off:.4f} "
              f"| shift={(m_off-m_on)/(s_on+1e-30):+.2f}sig sd_ratio={s_off/(s_on+1e-30):.2f}")
    print(f"[plot] wrote:\n  {p1}\n  {p2}\n  {p3}")
    return stats, (p1, p2, p3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-dir", required=True)
    ap.add_argument("--off-dir", required=True)
    ap.add_argument("--corner-mock", type=int, default=0)
    ap.add_argument("--fig-dir", default="/home/mfho/hcd_priya_notes/figures/analysis")
    a = ap.parse_args()
    build_figures(a.on_dir, a.off_dir, a.fig_dir, corner_mock=a.corner_mock)


if __name__ == "__main__":
    main()
