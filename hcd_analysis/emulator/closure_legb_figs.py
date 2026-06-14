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

FIGDIR = "/home/mfho/hcd_priya/figures/analysis/05_likelihood"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# C_emu vs C_data on the leg grids (the sub-dominance figure).
# ----------------------------------------------------------------------------
def _emu_var_for_leg(ctx, leg, theta9, tau0_vec, alpha_hcd, core, *, use_xclass):
    """diag(C_emu) on a leg = diag(C_total − C_data) at (θ,τ₀,α) (cross-class OR diagonal)."""
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if (use_xclass and ctx.rho_zb_per_leg) else None
    _, C_total = DL.predict_P_obs_on_leg(
        ctx.model, theta9, tau0_vec, alpha_hcd, pf_stats=ctx.pf_stats, dla_core=core,
        cache_k=ctx.cache_k, leg=leg, sigma_zb=(None if rzb is not None else szb),
        alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
    return np.diag(np.asarray(C_total)) - np.diag(np.asarray(leg.C_data))


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
        ev_x = _emu_var_for_leg(ctx, leg, jnp.asarray(truth["params_unit"]), tau0_vec,
                                jnp.asarray(truth["w_c"]), core[leg.name], use_xclass=True)
        ev_d = _emu_var_for_leg(ctx, leg, jnp.asarray(truth["params_unit"]), tau0_vec,
                                jnp.asarray(truth["w_c"]), core[leg.name], use_xclass=False)
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
    """One Leg-B mock per leg, COLORED BY z (the prior render overplotted every z-bin in one
    colour → unreadable spaghetti). Two rows per leg:
      top   — k·P1D/π: sim-truth (line) + emulator-at-truth-θ (dashed) + noisy mock (points,
              thin ±σ_data bars), all coloured by z. The emu dashed should sit on the truth.
      bottom— the BINDING residual P_emu/P_truth−1 (%) vs k, coloured by z, with the grey
              ±σ_data/P_truth envelope: shows the emulator tracks truth WELL INSIDE the data
              error (the real sanity — emu error is sub-dominant to the noise it lives under).
    """
    plt = _plt()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sims, _ = held_out_sims(d, 0)
    truth = make_truth_from_sim(d, sims[0], 0)
    mock_legs, tp, info = make_legb_mock(ctx, truth, jax.random.PRNGKey(seed))
    core = _mock_core_per_leg(ctx, truth)
    cache_k = np.asarray(ctx.cache_k)
    z_sim = np.asarray(truth["z"]); P_sim = np.asarray(truth["P_obs_true"])

    nL = len(ctx.legs)
    fig, axes = plt.subplots(2, nL, figsize=(7.0 * nL, 8.4), squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.0, 1.0]))
    cmap = plt.get_cmap("viridis")
    for li, (leg, mleg) in enumerate(zip(ctx.legs, mock_legs)):
        axA, axR = axes[0][li], axes[1][li]
        tau0_vec = jnp.asarray(truth_tau0_on_leg(truth, leg))
        P_emu, _ = DL.predict_P_obs_on_leg(
            ctx.model, jnp.asarray(truth["params_unit"]), tau0_vec,
            jnp.asarray(truth["w_c"]), pf_stats=ctx.pf_stats, dla_core=core[leg.name],
            cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
        P_emu = np.asarray(P_emu)
        P_truth = np.zeros(leg.k.shape[0])
        for iz in range(leg.n_z):
            rows = np.where(np.asarray(leg.z_idx) == iz)[0]
            j = int(np.argmin(np.abs(z_sim - float(leg.z[iz]))))
            P_truth[rows] = np.asarray(jnp.interp(
                jnp.asarray(leg.k[rows]), jnp.asarray(cache_k), jnp.asarray(P_sim[j])))
        P_mock = np.asarray(mleg.P_data)
        sig_d = np.sqrt(np.diag(np.asarray(leg.C_data)))
        k = np.asarray(leg.k); zr = np.asarray(leg.z_row)
        kept = np.isfinite(P_mock) & (P_truth > 0)
        zlev = np.unique(zr[kept]); norm = Normalize(zlev.min(), zlev.max())
        fac = k / np.pi
        for zz in zlev:
            m = kept & (zr == zz)
            o = np.argsort(k[m]); c = cmap(norm(zz))
            axA.plot(k[m][o], (fac * P_truth)[m][o], "-", color=c, lw=1.3)
            axA.plot(k[m][o], (fac * P_emu)[m][o], "--", color=c, lw=1.1, alpha=0.9)
            axA.errorbar(k[m], (fac * P_mock)[m], yerr=(fac * sig_d)[m], fmt=".", ms=4,
                         color=c, alpha=0.55, elinewidth=0.6, capsize=0)
            axR.plot(k[m][o], (100.0 * (P_emu / P_truth - 1.0))[m][o], "-", color=c, lw=1.1)
        # grey ±σ_data/P_truth envelope (the per-row data 1σ the emu residual lives inside)
        ok = np.argsort(k[kept]); env = 100.0 * (sig_d / P_truth)[kept][ok]
        axR.fill_between(k[kept][ok], -env, env, color="0.8", alpha=0.6,
                         label=r"$\pm\sigma_{\rm data}/P_{\rm truth}$ (data 1σ)")
        axA.plot([], [], "k-", label="sim-truth"); axA.plot([], [], "k--", label="emu @ truth θ")
        axA.plot([], [], "k.", label=r"noisy mock $\pm\sigma_{\rm data}$")
        axA.set_xscale("log"); axA.set_yscale("log")
        axA.set_ylabel(r"$k\,P_{\rm 1D}/\pi$")
        med = float(np.median(np.abs(P_emu / P_truth - 1.0)[kept]))
        axA.set_title(f"{leg.name}  ({int(kept.sum())} rows, {zlev.size} z-bins; "
                      f"|emu/truth−1| med {100*med:.2f}%)")
        axA.legend(fontsize=8, loc="best"); axA.grid(alpha=0.3, which="both")
        axR.set_xscale("log"); axR.axhline(0, color="k", lw=0.8)
        axR.set_xlabel("k [s/km] (angular)")
        axR.set_ylabel("emu/truth − 1 [%]")
        axR.set_ylim(-20, 20); axR.legend(fontsize=8, loc="upper left")
        axR.grid(alpha=0.3, which="both")
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=[axA, axR], label="z", pad=0.01)
    fig.suptitle(f"Leg-B mock example (sim={sims[0][:22]}…) — truth · emulator-at-truth · "
                 "noisy mock, coloured by z")
    p = Path(figdir) / "legb_mock_example.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(p)


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
