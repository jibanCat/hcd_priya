#!/usr/bin/env python
"""Panel-designed cache diagnostic (PI decision 2b, 2026-07-18):

Measure, across the 60 LF PRIYA sims in the v3.3 emulator cache, the correlation
between the LLS incidence weight w_LLS (= snapshot-level dN/dX_LLS, tau0-invariant)
and the forest amplitude / IGM parameters. Deliverables:

  * partial slopes d ln w_LLS / d theta_j (controlling for the other theta),
    with d ln A_p (physical log-amplitude) as the amplitude axis,
  * the FIXED-THETA scatter (dof-corrected residual sd of ln w_LLS after
    regressing out all 9 theta),
  * the finite-sample error (~scatter/sqrt(60)),
  * the decision-rule evaluation: |slope| x sigma_post(ln A_p) vs sim scatter,
    with sigma_post ~ 0.03 (typical DESI-leg posterior sd, interpreted as the
    sd of ln A_p, i.e. ~3% fractional amplitude; stated explicitly),
  * z-dependence: same regression at z = 2.4, 3.0, 4.2.

Read-only on the cache. numpy + h5py + matplotlib only (no jax stack import).
Login-node light (reads only the small snapshot-level + parameter datasets).

NOTE (stated, not a finding): the cache tau0 axis is a POST-HOC uniform optical-
depth rescale; the NHI-classified incidence is invariant under it BY CONSTRUCTION,
so d ln w_LLS / d tau0 == 0 identically and is not reported as a measurement.

CAVEAT (regression dilution / range mismatch): the design hypercube spans far
wider than any posterior (ln A_p range ~0.75 vs sigma_post ~0.03). The partial
slopes are broad-box averages; any local curvature near the best fit is averaged
over, and the decision-rule product is a box-scale extrapolation down to
posterior scale.
"""
from __future__ import annotations

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
FIG = ("/home/mfho/hcd_priya/figures/analysis/01_catalog_obs/"
       "diag_wlls_forest_correlation.png")

Z_TARGETS = (3.0, 2.4, 4.2)
Z_MAIN = 3.0
SIGMA_POST_LNAP = 0.03  # assumed sd of ln A_p (typical DESI-leg posterior; ~3% fractional)

# Physical design-box ranges (data.py PARAM_LIMITS), used only to quote a
# "per full box range" effect size for ranking; slopes themselves are per
# physical unit (per unit ln A_p for the amplitude).
PARAM_LIMITS = {
    "ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.5),
    "heref": (2.2, 3.2), "alphaq": (1.3, 3.0), "hub": (0.65, 0.75),
    "omegamh2": (0.14, 0.146), "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07),
}


def load_per_sim():
    """Return (sims, P(60,9), param_names, z_snap(18), lnw(60,18))."""
    with h5py.File(CACHE, "r") as f:
        pnames = [n.decode() if isinstance(n, bytes) else n
                  for n in f["param_names"][:]]
        ssn = np.array([s.decode() if isinstance(s, bytes) else s
                        for s in f["snap_sim_name"][:]])
        dndx_lls = f["snap_dNdX_LLS"][:]
        sgi = f["snap_group_idx"][:]
        zg = f["z_grid"][:]
        sn = np.array([s.decode() if isinstance(s, bytes) else s
                       for s in f["sim_name"][:]])
        params = f["params"][:]

    n_snap = dndx_lls.shape[0]
    # z of each snapshot group = z_grid of its first row (constant within group).
    first_row = np.full(n_snap, -1, dtype=int)
    for r, g in enumerate(sgi):
        if first_row[g] < 0:
            first_row[g] = r
    assert (first_row >= 0).all()
    z_snap_all = zg[first_row]

    sims = np.unique(ssn)
    assert len(sims) == 60, f"expected 60 LF sims, got {len(sims)}"

    # per-sim params (constant across the sim's rows; verified)
    P = np.empty((len(sims), len(pnames)))
    for i, s in enumerate(sims):
        rows = np.where(sn == s)[0]
        ps = params[rows]
        assert np.allclose(ps, ps[0]), f"params vary within sim {s}"
        P[i] = ps[0]

    # per-sim (z, ln dN/dX_LLS) tracks. Snapshot coverage is NOT identical
    # across sims (7 sims miss z=2.0; one misses z=3.0 and gets genuine
    # interpolation between 2.8 and 3.2), so keep per-sim z arrays.
    tracks = []
    for s in sims:
        m = ssn == s
        order = np.argsort(z_snap_all[m])
        zs = z_snap_all[m][order]
        w = dndx_lls[m][order]
        assert (w > 0).all(), f"non-positive dN/dX_LLS in sim {s}"
        assert zs.min() <= min(Z_TARGETS) and zs.max() >= max(Z_TARGETS), \
            f"sim {s} z coverage [{zs.min()},{zs.max()}] misses a target"
        tracks.append((zs, np.log(w)))
    return sims, P, pnames, tracks


def build_design(P, pnames):
    """Design matrix: intercept + ln(Ap) + the 8 other params (physical units).
    Returns (X(60,10), labels(10), full-box ranges for the 9 predictors)."""
    cols, labels, box = [np.ones(P.shape[0])], ["const"], []
    for j, name in enumerate(pnames):
        lo, hi = PARAM_LIMITS[name]
        if name == "Ap":
            cols.append(np.log(P[:, j]))
            labels.append("ln A_p")
            box.append(np.log(hi) - np.log(lo))
        else:
            cols.append(P[:, j])
            labels.append(name)
            box.append(hi - lo)
    return np.column_stack(cols), labels, np.array(box)


def ols(X, y):
    """Classical OLS: beta, se, residual sd (dof-corrected), residuals, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, np.sqrt(s2), resid, r2


def added_variable(X, y, j):
    """Partial-regression (added-variable) coordinates for predictor j:
    residualize y and X[:,j] on the other columns."""
    keep = [c for c in range(X.shape[1]) if c != j]
    Xo = X[:, keep]
    H = Xo @ np.linalg.lstsq(Xo, np.eye(X.shape[0]), rcond=None)[0]
    ey = y - H @ y
    ex = X[:, j] - H @ X[:, j]
    return ex, ey


def main():
    sims, P, pnames, tracks = load_per_sim()
    X, labels, box = build_design(P, pnames)
    n = len(sims)

    results = {}
    for zt in Z_TARGETS:
        y = np.array([np.interp(zt, zs, lw) for zs, lw in tracks])
        results[zt] = (y,) + ols(X, y)

    # ---------------- printed report ----------------
    print("=" * 78)
    print("diag_wlls_forest_correlation: ln w_LLS (= ln dN/dX_LLS, tau0-invariant)")
    print(f"60 LF PRIYA sims, cache {CACHE}")
    print("Partial OLS slopes d ln w_LLS / d theta_j (controlling the other 8),")
    print("per PHYSICAL unit; amplitude axis = ln A_p (per unit ln A_p).")
    print("tau0: post-hoc rescale axis, incidence invariant BY CONSTRUCTION ->")
    print("      slope == 0 identically; excluded from the table (not a finding).")
    print("=" * 78)

    for zt in Z_TARGETS:
        y, beta, se, s_fix, resid, r2 = results[zt]
        print(f"\n--- z = {zt} ---")
        print(f"{'param':>10s} {'slope':>10s} {'se':>9s} {'t':>7s} "
              f"{'|slope|xbox':>11s}")
        for j in range(1, X.shape[1]):
            eff = abs(beta[j]) * box[j - 1]
            print(f"{labels[j]:>10s} {beta[j]:>10.4f} {se[j]:>9.4f} "
                  f"{beta[j]/se[j]:>7.2f} {eff:>11.4f}")
        print(f"  R^2 = {r2:.3f}")
        print(f"  fixed-theta scatter (residual sd, dof={n - X.shape[1]}): "
              f"{s_fix:.4f}  (in ln w_LLS)")
        print(f"  finite-sample error on the sim-mean level: "
              f"scatter/sqrt(60) = {s_fix/np.sqrt(n):.4f}")

    # decision rule at z_main
    y, beta, se, s_fix, resid, r2 = results[Z_MAIN]
    j_ap = labels.index("ln A_p")
    slope_ap = beta[j_ap]
    induced = abs(slope_ap) * SIGMA_POST_LNAP
    print("\n" + "=" * 78)
    print("DECISION RULE (panel Bayesian lens), z = 3:")
    print(f"  assumed sigma_post(A_p) = {SIGMA_POST_LNAP} interpreted as the sd of "
          f"ln A_p\n  (typical DESI-leg posterior sd in these units, ~3% fractional).")
    print(f"  |d ln w_LLS/d ln A_p| x sigma_post = {abs(slope_ap):.4f} x "
          f"{SIGMA_POST_LNAP} = {induced:.5f}")
    print(f"  fixed-theta sim scatter               = {s_fix:.4f}")
    print(f"  ratio (induced / scatter)             = {induced/s_fix:.4f}")
    verdict = ("correlated alpha_LLS prior is formally WORTHLESS "
               "(induced shift << sim scatter)"
               if induced < 0.1 * s_fix else
               "correlated prior NOT negligible by this rule")
    print(f"  => {verdict}")
    print("\nCAVEAT: design hypercube >> posterior (regression dilution / "
          "box-average slope);\nln A_p spans "
          f"{np.log(PARAM_LIMITS['Ap'][1]/PARAM_LIMITS['Ap'][0]):.2f} in the box "
          f"vs sigma_post {SIGMA_POST_LNAP}; slopes are broad-box averages.")

    # ---------------- figure ----------------
    # top-2 predictors by |t| at z=3
    tvals = np.abs(beta[1:] / se[1:])
    top2 = 1 + np.argsort(tvals)[::-1][:2]

    ink, muted = "#222222", "#777777"
    c_pts, c_fit = "#3b6fb6", "#c2571a"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})
    for ax, j in zip(axes[:2], top2):
        ex, ey = added_variable(X, y, j)
        ax.scatter(ex, ey, s=22, color=c_pts, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.4)
        xs = np.linspace(ex.min(), ex.max(), 2)
        ax.plot(xs, beta[j] * xs, color=c_fit, lw=2, zorder=2)
        ax.text(0.03, 0.94,
                f"partial slope = {beta[j]:.3f} $\\pm$ {se[j]:.3f}"
                f"  (t = {beta[j]/se[j]:.1f})",
                transform=ax.transAxes, va="top", fontsize=9, color=ink)
        ax.set_xlabel(f"{labels[j]} (residualized on other $\\theta$)",
                      color=ink)
        ax.set_ylabel(r"ln $w_{\rm LLS}$ (residualized)", color=ink)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)

    # table panel
    axt = axes[2]
    axt.axis("off")
    rows = []
    for j in range(1, X.shape[1]):
        rows.append([labels[j], f"{beta[j]:.3f}", f"{se[j]:.3f}",
                     f"{beta[j]/se[j]:.1f}"])
    tbl = axt.table(cellText=rows,
                    colLabels=["param", "slope", "se", "t"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold", color=ink)
    axt.set_title("partial slopes at z = 3\n"
                  f"fixed-$\\theta$ scatter = {s_fix:.3f}, "
                  f"$R^2$ = {r2:.2f}", fontsize=9, color=ink)

    fig.suptitle(r"ln $w_{\rm LLS}$ (dN/dX$_{\rm LLS}$) vs forest/IGM params, "
                 "60 LF PRIYA sims, z = 3   "
                 r"[$\tau_0$ axis: invariant by construction, excluded]",
                 fontsize=10.5, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=160)
    print(f"\nfigure written: {FIG}")


if __name__ == "__main__":
    main()
