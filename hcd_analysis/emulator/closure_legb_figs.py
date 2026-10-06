"""Phase-C T4b — the three Leg-B diagnostic figures (plan §3 surfacing).

  legb_cemu_on_leg.png   : the cross-class C_emu sqrt-diag vs the data error (sqrt diag
                           C_data) on the DESI + KS grids — C_emu sub-dominance on the REAL
                           grid. (Also the diagonal-vs-cross-class C_emu comparison.)
  legb_mock_example.png  : ONE mock per leg — the sim-truth-on-leg, the noisy mock, and the
                           emulator prediction at the truth θ (sanity of the mock + binding).
  legb_smoke_coverage.png: the smoke's per-param coverage (with the small-N caveat) + the
                           loglik-rank ECDF.

Matplotlib is imported lazily (Agg backend); pure plotting (no NUTS). Figures land in
``figures/analysis/05_likelihood/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from . import data_likelihood as DL
from .closure_legb import (
    make_truth_from_sim, make_legb_mock, held_out_sims, _mock_core_per_leg, _kim,
)

from ..paths import REPO_ROOT_STR as _REPO

FIGDIR = f"{_REPO}/figures/analysis/05_likelihood"


def _truth_alpha_zresolved_on_leg(truth, leg):
    """THE SANCTIONED comparison-alpha builder (use THIS, or ``alpha_pivot[None,:]*shape_zg``, in ANY
    forward-vs-truth comparison: C_emu sizing, residual, loglik. NEVER hand-build alpha from a z-median /
    z-flat w_c). The Z-RESOLVED truth incidence α(z) (n_z,3) on a leg's z-bins, the per-z sim w_c (rises
    ~3.5× over z), NOT the z-flat z-median ``truth['w_c']`` the figure used to pass (which the
    binding silently broadcasts → a SPURIOUS z-ramp in P_emu vs the z-resolved P_truth). EXACTLY
    the "EXACT per-z w_c(z)" arm of scripts/diag_legb_zresolved_alpha_check.py (the reference that
    does it right): ``truth['w_c_z']`` (per-z [LLS,sub,DLA] = ``d['w_c_cache'][truth['rows'],1:]``)
    nearest-z mapped onto the leg z. The figure's P_truth is the DLA-MASKED baseline P_obs_true, so
    — matching the diag reference — the DLA column is the raw per-z w_c (no §0c 10% scaling; the
    masked baseline holds the DLA class at the clean level either way)."""
    import numpy as _np
    z_sim = _np.asarray(truth["z"])
    w_c_z = _np.asarray(truth["w_c_z"], float)              # (nZs,3) per-z structural w_c
    sel = _np.array([int(_np.argmin(_np.abs(z_sim - zz))) for zz in _np.asarray(leg.z)])
    return jnp.asarray(w_c_z[sel])                          # (n_z,3) z-resolved (the fix)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# C_emu vs C_data on the leg grids (the sub-dominance figure).
# ----------------------------------------------------------------------------
def _emu_var_for_leg(ctx, leg, theta9, tau0_vec, alpha_hcd, core, *, use_xclass):
    """The pre-2026-10 closure figure helper (bound on the single grid ``ctx.cache_k``). Retired at gate E;
    rebuilt at gate F with the closure machinery."""
    raise NotImplementedError("_emu_var_for_leg: retired with the single-grid forward; rebuilt at gate F")


def fig_cemu_on_leg(ctx, d, figdir=FIGDIR):
    """sqrt-diag C_emu (cross-class + diagonal) vs sqrt-diag C_data per leg, vs k, colored by
    z. Reports the median diag(C_emu)/diag(C_data) per leg (diagonal + cross-class)."""
    plt = _plt()
    sims, _ = held_out_sims(d, 0)
    truth = make_truth_from_sim(d, sims[0], 0)
    core = _mock_core_per_leg(ctx, truth)
    zg = np.asarray(ctx.z_global)

    fig, axes = plt.subplots(1, len(ctx.legs), figsize=(7.0 * len(ctx.legs), 5.4),
                             squeeze=False)
    ratios = {}
    for li, leg in enumerate(ctx.legs):
        ax = axes[0][li]
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        tau0_vec = jnp.asarray(truth_tau0_on_leg(truth, leg))
        a_zr = _truth_alpha_zresolved_on_leg(truth, leg)     # Z-RESOLVED truth α(z) for C_emu sizing
        ev_x = _emu_var_for_leg(ctx, leg, jnp.asarray(truth["params_unit"]), tau0_vec,
                                a_zr, core[leg.name], use_xclass=True)
        ev_d = _emu_var_for_leg(ctx, leg, jnp.asarray(truth["params_unit"]), tau0_vec,
                                a_zr, core[leg.name], use_xclass=False)
        cdata = np.diag(np.asarray(leg.C_data))
        k = np.asarray(leg.k)
        zr = np.asarray(leg.z_row)

        sc = ax.scatter(k, np.sqrt(np.maximum(ev_x, 0)), c=zr, s=14, cmap="viridis",
                        label=r"$\sqrt{\mathrm{diag}\,C_{\rm emu}}$ (cross-class)")
        ax.scatter(k, np.sqrt(np.maximum(ev_d, 0)), c=zr, s=8, marker="x", cmap="viridis",
                   alpha=0.6, label=r"$\sqrt{\mathrm{diag}\,C_{\rm emu}}$ (diagonal)")
        ax.scatter(k, np.sqrt(cdata), c="0.4", s=10, marker="s",
                   label=r"$\sqrt{\mathrm{diag}\,C_{\rm data}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("k [s/km] (angular)")
        ax.set_ylabel(r"$\sqrt{\mathrm{diag}}$  [P1D units]")
        rx = float(np.median(ev_x / cdata)); rd = float(np.median(ev_d / cdata))
        ratios[leg.name] = dict(xclass=rx, diag=rd,
                                max_xclass=float(np.max(ev_x / cdata)))
        ax.set_title(f"{leg.name}: median C_emu/C_data = {rx:.3f} (x-class), {rd:.3f} (diag); "
                     f"max {float(np.max(ev_x / cdata)):.2f} (low-k)")
        ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3, which="both")
        fig.colorbar(sc, ax=ax, label="z")
    fig.suptitle("Leg-B: C_emu sub-dominance on the REAL DESI + KS grids "
                 "(at a held-out sim's truth θ,τ₀,α)")
    p = Path(figdir) / "legb_cemu_on_leg.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p), ratios


def truth_tau0_on_leg(truth, leg):
    """Nearest-z map the sim-truth τ₀ onto a leg's z (the binding's per-z τ₀)."""
    z_sim = np.asarray(truth["z"]); tau0 = np.asarray(truth["tau0"])
    return np.array([tau0[int(np.argmin(np.abs(z_sim - zz)))] for zz in leg.z])


# ----------------------------------------------------------------------------
# One-mock example (truth-on-leg, noisy mock, emulator prediction at truth θ).
# ----------------------------------------------------------------------------
def fig_mock_example(ctx, d, figdir=FIGDIR, seed=0):
    """The pre-2026-10 closure figure helper (bound on the single grid ``ctx.cache_k``). Retired at gate E;
    rebuilt at gate F with the closure machinery."""
    raise NotImplementedError("fig_mock_example: retired with the single-grid forward; rebuilt at gate F")


# ----------------------------------------------------------------------------
# Smoke coverage + loglik-rank ECDF.
# ----------------------------------------------------------------------------
def fig_smoke_coverage(res, figdir=FIGDIR, suptitle=None):
    """The per-param coverage bars (68 & 95%, with Wilson CIs) and the (DIAGNOSTIC-ONLY)
    loglik-rank ECDF panel. ``suptitle`` overrides the default smoke caption (the merge passes
    a pilot-appropriate one)."""
    plt = _plt()
    names = res["names"]
    q_levels = res["q_levels"]
    fig, (axc, axe) = plt.subplots(1, 2, figsize=(15, 5.6))

    x = np.arange(len(names))
    width = 0.8 / max(len(q_levels), 1)
    for qi, q in enumerate(q_levels):
        cov = [res["coverage"][q][nm]["coverage"] for nm in names]
        lo = [res["coverage"][q][nm]["ci_low"] for nm in names]
        hi = [res["coverage"][q][nm]["ci_high"] for nm in names]
        yerr = np.vstack([np.array(cov) - np.array(lo), np.array(hi) - np.array(cov)])
        yerr = np.clip(yerr, 0, None)
        axc.bar(x + qi * width, cov, width, yerr=yerr, capsize=3,
                label=f"{int(q*100)}% CR", alpha=0.85)
        axc.axhline(q, ls="--", lw=1.0, color=f"C{qi}", alpha=0.7)
    axc.set_xticks(x + 0.4 - width / 2)
    axc.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axc.set_ylabel("empirical coverage")
    axc.set_ylim(0, 1.05)
    axc.set_title(f"Leg-B per-param coverage (N={res['n_mocks']} mocks — SMALL-N, PATH check)")
    axc.legend(fontsize=9); axc.grid(alpha=0.3, axis="y")

    if res.get("ecdf_ll") is not None:
        lower, upper, ecdf, grid = res["ecdf_ll"]
        axe.fill_between(grid, lower, upper, color="0.8",
                         label="95% simultaneous band (uniform null)")
        axe.plot(grid, ecdf, "-o", ms=3, color="C2", label="loglik-rank ECDF")
        axe.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
        band = "in-band" if res.get("ll_ecdf_in_band") else "out-of-band"
        axe.set_title(f"loglik-rank ECDF ({band}) — DIAGNOSTIC ONLY, not a Leg-B gate")
    else:
        axe.text(0.5, 0.5, "no loglik-rank ECDF (too few mocks)", ha="center",
                 transform=axe.transAxes)
    axe.set_xlabel("PIT"); axe.set_ylabel("ECDF")
    axe.legend(fontsize=9); axe.grid(alpha=0.3)
    fig.suptitle(suptitle or
                 "Leg-B smoke diagnostics (the FULL path; tiny N — PATH proof, not a verdict)")
    p = Path(figdir) / "legb_smoke_coverage.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


def make_all(ctx, d, res, figdir=FIGDIR):
    """Emit the three Leg-B figures; print their paths + the C_emu/C_data ratios."""
    p1, ratios = fig_cemu_on_leg(ctx, d, figdir=figdir)
    p2 = fig_mock_example(ctx, d, figdir=figdir)
    p3 = fig_smoke_coverage(res, figdir=figdir)
    print("\n[figures]")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print("\n[C_emu/C_data on-leg median ratios]")
    for nm, r in ratios.items():
        print(f"  {nm}: x-class {r['xclass']:.4f}  diag {r['diag']:.4f}  "
              f"(x-class max {r['max_xclass']:.3f})")
    return dict(cemu=p1, mock=p2, coverage=p3, ratios=ratios)
