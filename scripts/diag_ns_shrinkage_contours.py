#!/usr/bin/env python3
"""Diagnostic contours for the held-out n_s tilt = benign shrinkage story.

Reads the per-leg all-folds held-out pkls (prod_sbc_loso_fold{0..7}_<LEG>) and
emits two figures that visualize WHY the deployed n_s posterior shrinks toward
the prior center (a calibrated effect amplified by a weak n_s signal):

  Fig A  ns_recovery   : recovered n_s (post mean +- sd) vs truth across the box
                         (regression-to-center tilt) + 1D posteriors at low/mid/
                         high truth (boundary pile-up at the sparse high-n_s tail).
  Fig B  ns_degeneracy : n_s-A_p and n_s-tau0 covariance contours for a low- and a
                         high-n_s mock (the degeneracies that weaken the n_s signal,
                         + the posterior-width vs prior-width 'shrinkage budget').

Draws for ns/Ap are on the UNIT CUBE [0,1]; prior is uniform => Var_prior = 1/12,
sd_prior = 1/sqrt(12) = 0.2887. Physical n_s is parsed from the sim name for labels.
"""
import argparse, glob, pickle, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SD_PRIOR = 1.0 / np.sqrt(12.0)   # uniform [0,1] prior

def parse_phys_ns(sim):
    m = re.search(r"ns([0-9.]+)", str(sim))
    return float(m.group(1)) if m else np.nan

def load_leg(root, leg, prefix):
    """Return list of per-mock dicts with truth/draws/idx, sorted by truth n_s (unit)."""
    out = []
    for k in range(8):
        d = f"{root}/{prefix}fold{k}_{leg}"
        for f in sorted(glob.glob(f"{d}/mock_*.pkl")):
            try:
                r = pickle.load(open(f, "rb"))
            except Exception:
                continue
            names = list(r["names"])
            ix = {n: names.index(n) for n in names}
            truth = np.asarray(r["truth_vec"], float)
            draws = np.asarray(r["draws"], float)
            out.append(dict(
                fold=k, file=os.path.basename(f), names=names, ix=ix,
                truth=truth, draws=draws,
                ns_unit=float(truth[ix["ns"]]),
                ns_phys=parse_phys_ns(r["sim"]),
                n_div=int(r.get("n_div", 0)),
            ))
    out.sort(key=lambda r: r["ns_unit"])
    return out

def cov_ellipse(ax, x, y, nsig=1.0, **kw):
    mu = np.array([x.mean(), y.mean()])
    C = np.cov(x, y)
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    width, height = 2 * nsig * np.sqrt(np.maximum(w[::-1], 0))
    e = Ellipse(mu, width, height, angle=ang, fill=False, **kw)
    ax.add_patch(e)
    return mu

# ---------------------------------------------------------------- Fig A
def fig_recovery(mocks, leg, outpath):
    ns_t = np.array([m["ns_unit"] for m in mocks])
    pm   = np.array([m["draws"][:, m["ix"]["ns"]].mean() for m in mocks])
    psd  = np.array([m["draws"][:, m["ix"]["ns"]].std(ddof=1) for m in mocks])
    ph   = np.array([m["ns_phys"] for m in mocks])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    # -- left: recovered vs truth (shrinkage / regression to center) --
    sc = axL.scatter(ns_t, pm, c=ph, cmap="viridis", s=46, zorder=5,
                     edgecolor="k", linewidth=0.4)
    axL.errorbar(ns_t, pm, yerr=psd, fmt="none", ecolor="0.6", lw=0.8,
                 alpha=0.7, zorder=2)
    lo, hi = -0.02, 1.02
    axL.plot([lo, hi], [lo, hi], "k--", lw=1, label="unbiased (y=x)")
    axL.axhline(0.5, color="crimson", ls=":", lw=1.2, label="prior center (0.5)")
    # OLS fit (pooled) to show the tilt slope
    b1, b0 = np.polyfit(ns_t, pm, 1)
    xx = np.linspace(0, 1, 50)
    axL.plot(xx, b0 + b1 * xx, color="navy", lw=2,
             label=f"fit slope={b1:.2f} (shrinkage<1)")
    axL.axvspan(0.9, 1.02, color="orange", alpha=0.10, zorder=0)
    axL.text(0.95, 0.04, "sparse\nhigh-ns\ntail", color="darkorange",
             ha="center", va="bottom", fontsize=8, transform=axL.get_xaxis_transform())
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
    axL.set_xlabel("truth n_s  (unit cube)")
    axL.set_ylabel("posterior mean n_s  (unit cube)")
    axL.set_title(f"{leg}: n_s recovery regresses toward the center\n"
                  "(slope<1 = shrinkage; ends pull inward)")
    axL.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(sc, ax=axL); cb.set_label("physical n_s (sim)")

    # -- right: 1D posteriors at low / mid / high truth --
    picks = [mocks[0], mocks[len(mocks)//2], mocks[-1]]
    cols = ["#2166ac", "#4d4d4d", "#b2182b"]
    for m, c in zip(picks, cols):
        x = m["draws"][:, m["ix"]["ns"]]
        axR.hist(x, bins=24, range=(0, 1), density=True, histtype="stepfilled",
                 color=c, alpha=0.28)
        axR.hist(x, bins=24, range=(0, 1), density=True, histtype="step",
                 color=c, lw=1.6,
                 label=f"truth n_s={m['ns_phys']:.3f}  (post mean {x.mean():.2f})")
        axR.axvline(m["ns_unit"], color=c, ls="--", lw=1.6)
    axR.axvline(0.5, color="crimson", ls=":", lw=1.2)
    axR.axhline(1.0, color="0.6", ls="-", lw=1, alpha=0.6)
    axR.text(0.02, 1.04, "uniform prior density = 1", color="0.4", fontsize=8)
    axR.set_xlim(0, 1)
    axR.set_xlabel("n_s  (unit cube;  dashed = truth,  dotted = prior center)")
    axR.set_ylabel("posterior density")
    axR.set_title(f"{leg}: high-n_s posterior piles against the cube edge\n"
                  "(weak signal => wide, reflects off boundary => shrinks in)")
    axR.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    return dict(slope=float(b1), psd_med=float(np.median(psd)),
                psd_over_prior=float(np.median(psd)/SD_PRIOR))

# ---------------------------------------------------------------- Fig B
def _tau0_mean(m):
    """Mean of the tau0_z* ladder = the global mean-flux mode (degenerate with n_s)."""
    cols = [i for n, i in m["ix"].items() if n.startswith("tau0_z")]
    return m["draws"][:, cols].mean(axis=1), np.mean([m["truth"][i] for i in cols])

def fig_degeneracy(mocks, leg, outpath):
    lowm  = mocks[1]            # near low edge
    highm = mocks[-2]           # near high edge (skip the very tip if noisy)
    panels = [("Ap", "A_p"), ("__tau0mean__", "mean-flux <tau0>")]

    fig, axes = plt.subplots(1, len(panels)+1, figsize=(5.2*(len(panels)+1), 5.0))
    for ax, (pk, plabel) in zip(axes, panels):
        for m, c, tag in [(lowm, "#2166ac", "low"), (highm, "#b2182b", "high")]:
            xs = m["draws"][:, m["ix"]["ns"]]
            if pk == "__tau0mean__":
                ys, ytruth = _tau0_mean(m)
            else:
                ys, ytruth = m["draws"][:, m["ix"][pk]], m["truth"][m["ix"][pk]]
            ax.scatter(xs, ys, s=6, color=c, alpha=0.18, zorder=2)
            for ns_ in (1.0, 2.0):
                cov_ellipse(ax, xs, ys, nsig=ns_, edgecolor=c, lw=1.6, zorder=4)
            r = np.corrcoef(xs, ys)[0, 1]
            ax.scatter([m["truth"][m["ix"]["ns"]]], [ytruth],
                       marker="*", s=220, color=c, edgecolor="k", linewidth=0.6,
                       zorder=6, label=f"{tag} ns={m['ns_phys']:.3f} (r={r:+.2f}, ★truth)")
        ax.axvline(0.5, color="crimson", ls=":", lw=1.0)
        ax.set_xlabel("n_s (unit cube)"); ax.set_ylabel(f"{plabel} (unit cube)")
        ax.set_title(f"{leg}: n_s vs {plabel}\n(1σ/2σ posterior; tilt = degeneracy)")
        ax.legend(fontsize=8)

    # last panel: posterior-width 'shrinkage budget' across the box
    axW = axes[-1]
    ns_t = np.array([m["ns_unit"] for m in mocks])
    psd  = np.array([m["draws"][:, m["ix"]["ns"]].std(ddof=1) for m in mocks])
    contr = 1 - (psd**2) / (1/12.)
    sc = axW.scatter(ns_t, psd, c=contr, cmap="plasma", s=40, edgecolor="k", lw=0.3)
    axW.axhline(SD_PRIOR, color="k", ls="--", lw=1.2, label=f"prior sd = {SD_PRIOR:.3f}")
    axW.set_ylim(0, SD_PRIOR*1.08)
    axW.set_xlabel("truth n_s (unit cube)")
    axW.set_ylabel("posterior sd of n_s")
    axW.set_title(f"{leg}: n_s posterior width vs prior\n"
                  "(near prior sd => weak signal => strong shrinkage)")
    axW.legend(fontsize=8, loc="lower center")
    cb = fig.colorbar(sc, ax=axW); cb.set_label("contraction 1 - Var_post/Var_prior")
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    return dict(contr_med=float(np.median(1-(psd**2)/(1/12.))))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="KS")
    ap.add_argument("--root", default="/scratch/cavestru_root/cavestru1/mfho")
    ap.add_argument("--prefix", default="prod_sbc_loso_")
    # working diagnostic figures live in the PRIVATE notes repo, not the code repo
    ap.add_argument("--outdir", default="/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation/heldout_ns_shrinkage")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    mocks = load_leg(a.root, a.leg, a.prefix)
    print(f"[{a.leg}] loaded {len(mocks)} mocks; "
          f"truth n_s(unit) {mocks[0]['ns_unit']:.3f}..{mocks[-1]['ns_unit']:.3f}; "
          f"phys {mocks[0]['ns_phys']:.3f}..{mocks[-1]['ns_phys']:.3f}; "
          f"divergent mocks={sum(m['n_div']>0 for m in mocks)}")
    pA = f"{a.outdir}/diag_ns_shrink_recovery_{a.leg}.png"
    pB = f"{a.outdir}/diag_ns_shrink_degeneracy_{a.leg}.png"
    rA = fig_recovery(mocks, a.leg, pA)
    rB = fig_degeneracy(mocks, a.leg, pB)
    print(f"  recovery slope={rA['slope']:.3f}  median post_sd={rA['psd_med']:.3f} "
          f"= {rA['psd_over_prior']*100:.0f}% of prior sd")
    print(f"  median contraction={rB['contr_med']:.2f}")
    print(f"  wrote {pA}")
    print(f"  wrote {pB}")
