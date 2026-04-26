"""
Compare the Rogers+2018 prior α (z-independent) to the PRIYA-fitted α(z).

Two figures:

1. `rogers_alpha_vs_priya_fit.png`
   - 4 panels (one per Rogers component: LLS / Sub-DLA / Small-DLA / Large-DLA).
   - Blue horizontal line at α_Rogers = 0.1 (the uniform prior used in
     `tests/validate_rogers_subset.py`; Rogers+2018 best-fit values from
     BOSS DR12 are not hardcoded in this repo, so this is a stand-in).
   - Red line + shaded band: PRIYA LF mean ± 1σ of the per-(sim, z) fit.
   - Green symbols: HR mean with 1σ error bar.

2. `template_measured_vs_rogers_per_z.png`
   - Rows = class (LLS, subDLA, DLA).
   - Cols = z bins (2.2, 2.8, 3.4, 4.0, 4.6).
   - Each panel: measured P_class_only / P_clean averaged across LF sims
     (with 1σ shaded band) overlaid with the Rogers+2018 prediction whose
     α is fitted per panel via weighted linear least squares
     (r_obs(k) ≈ 1 + α · K(k);  weight = 1/σ² from sim-to-sim spread).
   - For the DLA row, Rogers "Small-DLA + Large-DLA" contribution is summed
     under a shared α.  The fitted α is annotated in the top-left of each
     panel and also shown in the legend on the top-right corner.

Run:
    python3 scripts/plot_rogers_alpha_vs_priya.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir
from hcd_analysis.hcd_template import template_contributions

SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT = ROOT / "figures" / "analysis" / "03_templates_and_p1d"
OUT.mkdir(parents=True, exist_ok=True)

# Rogers stand-in value (uniform prior used in validate_rogers_subset.py).
# Rogers+2018 best-fit values from BOSS DR12 are not hardcoded in this repo.
ROGERS_ALPHA_PRIOR = 0.1

ALPHA_LABELS   = [r"$\alpha_{\rm LLS}$", r"$\alpha_{\rm Sub}$",
                  r"$\alpha_{\rm Small-DLA}$", r"$\alpha_{\rm Large-DLA}$"]
ROGERS_COMPONENTS = ["LLS", "Sub-DLA", "Small-DLA", "Large-DLA"]

# PRIYA-classification k range (angular)
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20

# Per-class → Rogers-component mapping for the DLA row
# (PRIYA "DLA" = log N ≥ 20.3 ≈ Small-DLA + Large-DLA).
PRIYA_TO_ROGERS = {
    "LLS":    ["LLS"],
    "subDLA": ["Sub-DLA"],
    "DLA":    ["Small-DLA", "Large-DLA"],
}

# Representative z-bins for the (measured vs Rogers) overlay
Z_BINS = [2.2, 2.8, 3.4, 4.0, 4.6]
Z_TOL = 0.1


# ---------------------------------------------------------------------------
# Figure 1 — α(z) comparison
# ---------------------------------------------------------------------------

def _load_alpha_summary():
    p = data_dir() / "rogers_alpha_summary.h5"
    with h5py.File(p, "r") as f:
        suite = np.array([s.decode() for s in f["suite"][:]])
        z = f["z"][:]
        alpha = f["alpha"][:]
    return suite, z, alpha


def _bin_by_z(z, y, z_bins, mask):
    zc, m, s = [], [], []
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        sel = mask & (z >= lo) & (z < hi) & np.isfinite(y)
        if not sel.any():
            continue
        zc.append(0.5 * (lo + hi))
        m.append(float(np.mean(y[sel])))
        s.append(float(np.std(y[sel])))
    return np.array(zc), np.array(m), np.array(s)


def plot_alpha_comparison(outpath: Path):
    suite, z, alpha = _load_alpha_summary()
    is_lf = suite == "lf"
    is_hr = suite == "hr"
    z_bins = np.arange(1.9, 5.7, 0.2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
    z_grid = np.linspace(2.0, 5.5, 100)
    for i, (name, label, col) in enumerate(
        zip(ROGERS_COMPONENTS, ALPHA_LABELS, ["C2", "C1", "C0", "C3"])
    ):
        ax = axes[i]
        # Rogers prior (z-independent)
        ax.axhline(
            ROGERS_ALPHA_PRIOR, color="steelblue", lw=2.0, ls="--",
            label=fr"Rogers prior  $\alpha = {ROGERS_ALPHA_PRIOR}$ (z-independent)",
        )

        # PRIYA LF fit α(z)
        zc_lf, m_lf, s_lf = _bin_by_z(z, alpha[:, i], z_bins, is_lf)
        ax.plot(zc_lf, m_lf, "o-", color=col, lw=1.8, ms=6,
                label="PRIYA LF fit (60 sims, mean ± 1σ)")
        ax.fill_between(zc_lf, m_lf - s_lf, m_lf + s_lf,
                        color=col, alpha=0.2)

        # PRIYA HR fit α(z)
        zc_hr, m_hr, s_hr = _bin_by_z(z, alpha[:, i], z_bins, is_hr)
        ax.errorbar(zc_hr, m_hr, yerr=s_hr, fmt="s--", color=col, mec="k",
                    ms=7, capsize=3, label="PRIYA HR fit (4 sims)")

        ax.set_xlabel("z")
        ax.set_title(label, fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        if i == 0:
            ax.set_ylabel(r"$\alpha$  (Rogers+2018 HCD amplitude)")

    fig.suptitle(
        "Rogers+2018 prior α (z-independent) vs PRIYA-fitted α(z) per class\n"
        "PRIYA fit α is z-dependent — it compensates for HCD contamination growing "
        "faster with z than the Rogers $(1+z)^{-3.55}$ amplitude scaling built into the template",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — measured P_class_only/P_clean vs Rogers prior α model
# ---------------------------------------------------------------------------

def _find_priya_template_at_z(z_target: float, tol: float) -> list[Path]:
    """Return all LF sim `p1d_per_class.h5` paths whose snap z is within tol."""
    paths = []
    for sim_dir in sorted(SCRATCH.iterdir()):
        if not sim_dir.is_dir() or not sim_dir.name.startswith("ns"):
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            if not snap_dir.name.startswith("snap_"):
                continue
            pc = snap_dir / "p1d_per_class.h5"
            meta = snap_dir / "meta.json"
            if not (pc.exists() and meta.exists() and (snap_dir / "done").exists()):
                continue
            try:
                z = float(json.load(open(meta))["z"])
            except Exception:
                continue
            if abs(z - z_target) < tol:
                paths.append(pc)
                break  # only one snap per sim at this z
    return paths


def _avg_template_across_sims(paths: list[Path], cls: str):
    """Return (k_ang, ratio_mean, ratio_std) for PRIYA P_class_only / P_clean
    averaged across sims at a single z."""
    per_sim_ratios = []
    k_ref = None
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                k_cyc = f["k"][:]
                P_clean = f["P_clean"][:]
                P_cls = f[f"P_{cls}_only"][:]
                n_cls = int(f[f"n_sightlines_{cls}"][()])
        except Exception:
            continue
        if n_cls == 0 or (P_clean > 0).sum() < 10:
            continue
        k_ang = 2.0 * np.pi * k_cyc
        sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX) & (P_clean > 0) & (P_cls > 0)
        if not sel.any():
            continue
        if k_ref is None:
            k_ref = k_ang[sel]
        if len(k_ang[sel]) != len(k_ref):
            continue  # skip mismatched k-grid
        per_sim_ratios.append((P_cls[sel] / P_clean[sel]))
    if not per_sim_ratios:
        return None, None, None
    arr = np.stack(per_sim_ratios)
    return k_ref, arr.mean(axis=0), arr.std(axis=0)


def _rogers_contribution_combined(k_ang, z, rogers_names, alpha_prior):
    """Sum (contribution - 1) over the requested Rogers components + 1, using
    a uniform α_prior scalar across all four components."""
    alpha_vec = np.full(4, alpha_prior, dtype=np.float64)
    contribs = template_contributions(k_ang, z, alpha_vec)
    total = np.ones_like(k_ang)
    for name in rogers_names:
        total += (contribs[name] - 1.0)
    return total


def _rogers_kernel(k_ang, z, rogers_names):
    """Per-class α-kernel K(k): r_model(k) = 1 + α · K(k), evaluated by
    probing template_contributions with α = 1 on the requested components."""
    alpha_vec = np.ones(4, dtype=np.float64)
    contribs = template_contributions(k_ang, z, alpha_vec)
    kernel = np.zeros_like(k_ang)
    for name in rogers_names:
        kernel = kernel + (contribs[name] - 1.0)
    return kernel


def _fit_single_alpha(k_ang, r_obs, r_std, z, rogers_names):
    """Unweighted linear least-squares fit of a single Rogers α against the
    measured ratio r_obs(k) ≈ 1 + α · K(k), bounded to α ≥ 0.

    We deliberately use uniform weights (not 1/σ²): sim-to-sim scatter is
    biggest at low k where the HCD signal is strongest, so 1/σ² weighting
    would suppress exactly the bins that carry the enhancement.

    σ(α) is estimated from the fit residuals: σ̂² = RSS/(n-1), so the
    quoted error reflects how well the Rogers shape reproduces the
    measured curve rather than the sim-to-sim scatter in r_obs.
    """
    K = _rogers_kernel(k_ang, z, rogers_names)
    denom = float(np.sum(K * K))
    if denom <= 0:
        return np.nan, np.nan, np.nan
    num = float(np.sum(K * (r_obs - 1.0)))
    alpha_hat = max(num / denom, 0.0)
    resid = r_obs - (1.0 + alpha_hat * K)
    n = len(K)
    sigma2 = float(np.sum(resid ** 2)) / max(1, n - 1)
    alpha_err = float(np.sqrt(sigma2 / denom))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return alpha_hat, alpha_err, rmse


def plot_measured_vs_rogers_per_z(outpath: Path):
    classes = ["LLS", "subDLA", "DLA"]
    cols = ["C2", "C1", "C3"]
    n_z = len(Z_BINS)

    fig, axes = plt.subplots(3, n_z, figsize=(4 * n_z, 11),
                             sharex=True, sharey="row")
    for i, (cls, col) in enumerate(zip(classes, cols)):
        for j, z_target in enumerate(Z_BINS):
            ax = axes[i, j]
            paths = _find_priya_template_at_z(z_target, Z_TOL)
            k_ang, m, s = _avg_template_across_sims(paths, cls)
            if k_ang is None:
                ax.set_title(f"{cls}  z≈{z_target:.1f}  (no data)")
                continue
            # Measured curve + band
            ax.plot(k_ang, m, "-", color=col, lw=1.6,
                    label=f"measured ⟨P_{cls} / P_clean⟩  (n={len(paths)} sims)")
            ax.fill_between(k_ang, m - s, m + s, color=col, alpha=0.2)

            # Rogers with α fitted per panel (single α, shared across the
            # Rogers components that map to this PRIYA class)
            alpha_hat, alpha_err, rmse = _fit_single_alpha(
                k_ang, m, s, z_target, PRIYA_TO_ROGERS[cls]
            )
            rogers_fit = _rogers_contribution_combined(
                k_ang, z_target, PRIYA_TO_ROGERS[cls], alpha_hat
            )
            ax.plot(k_ang, rogers_fit, "--", color="steelblue", lw=1.4,
                    label=(fr"Rogers fit  $\alpha = {alpha_hat:.3g}"
                           fr" \pm {alpha_err:.2g}$"
                           f"  [{' + '.join(PRIYA_TO_ROGERS[cls])}]"))

            # Annotate fitted α directly on the panel (with shape-mismatch RMSE)
            ax.text(0.03, 0.97,
                    fr"$\alpha_{{\rm fit}} = {alpha_hat:.3g}$" "\n"
                    fr"RMSE $= {rmse:.2g}$",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="steelblue", alpha=0.85))

            ax.axhline(1.0, color="gray", lw=0.5, ls=":")
            ax.set_xscale("log")
            if i == 2:
                ax.set_xlabel(r"$k$ [rad·s/km]")
            if j == 0:
                ax.set_ylabel(f"P_{cls} / P_clean")
            if i == 0:
                ax.set_title(f"z ≈ {z_target:.1f}", fontsize=11)
            ax.grid(alpha=0.3, which="both")
            if i == 0 and j == n_z - 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Measured P_class / P_clean vs Rogers+2018 template with α fitted per panel\n"
        "Rows = class (LLS / subDLA / DLA); columns = z-bins; "
        "single α per panel (shared across Rogers sub-components for that class)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    print("Plotting α comparison (Rogers prior vs PRIYA fit)…")
    plot_alpha_comparison(OUT / "rogers_alpha_vs_priya_fit.png")
    print(f"  wrote {OUT / 'rogers_alpha_vs_priya_fit.png'}")

    print("Plotting measured template vs Rogers prior α per z…")
    plot_measured_vs_rogers_per_z(OUT / "template_measured_vs_rogers_per_z.png")
    print(f"  wrote {OUT / 'template_measured_vs_rogers_per_z.png'}")


if __name__ == "__main__":
    main()
