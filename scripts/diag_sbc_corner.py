#!/usr/bin/env python3
"""Held-out SBC corner plots (no `corner` pkg) for representative mocks.

Posterior corner over a physically meaningful subset — cosmology (n_s, A_p),
mean flux (<tau0> = z-ladder mean), IGM (alpha_q), HCD (alpha_LLS) — with the
TRUTH overlaid (red). For the held-out cert: a calibrated posterior should
cover the red truth; the n_s panels show the shrinkage toward 0.5 (prior
center) when the leg constrains n_s weakly (KS) and its absence when it doesn't
(eBOSS). n_s / A_p are on the unit cube [0,1]; nuisances in native units.

  python scripts/diag_sbc_corner.py --leg KS          # low/mid/high-n_s KS corners
  python scripts/diag_sbc_corner.py --leg eBOSS --which mid
"""
import argparse, glob, pickle, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from hcd_analysis.emulator.meanflux_prior import kim_tau0, fit_tau0_alpha_priya

# the 2 SAMPLED mean-flux sites (tau0_amp, dtau0) are recovered from the deterministic
# tau0_z ladder via the deployed mapping; alpha_subdla added so the n_s-subDLA panel shows.
CORNER_PARAMS = ["ns", "Ap", "tau0_amp", "dtau0", "alpha_lls", "alpha_subdla"]
LABELS = {"ns": "n_s", "Ap": "A_p", "tau0_amp": r"$\tau_0$ amp", "dtau0": r"$d\tau_0$ slope",
          "alpha_lls": r"$\alpha_{\rm LLS}$ HCD", "alpha_subdla": r"$\alpha_{\rm subDLA}$"}
UNIT = {"ns", "Ap"}  # unit cube [0,1]
_ZLAD = 2.2 + 0.2 * np.arange(13)
_KIM = np.asarray(kim_tau0(_ZLAD))

def parse_phys_ns(s):
    m = re.search(r"ns([0-9.]+)", str(s)); return float(m.group(1)) if m else np.nan

def _rec_sites(ladder):  # tau_eff(z) -> (amp, slope) via the deployed tau0_alpha_priya mapping
    a = np.clip(np.asarray(ladder) / _KIM, 1e-6, None)
    if a.ndim == 1: return np.array(fit_tau0_alpha_priya(_ZLAD, a))
    return np.array([fit_tau0_alpha_priya(_ZLAD, a[j]) for j in range(len(a))])

def get_arrays(d):
    names = list(d["names"]); D = np.asarray(d["draws"], float); T = np.asarray(d["truth_vec"], float)
    ix = {n: i for i, n in enumerate(names)}
    ti = [ix[f"tau0_z{i}"] for i in range(13)]
    amp_d = _rec_sites(D[:, ti]); amp_t = _rec_sites(T[ti])    # (N,2) and (2,)
    cols, tcols = [], []
    for p in CORNER_PARAMS:
        if p == "tau0_amp":
            cols.append(amp_d[:, 0]); tcols.append(amp_t[0])
        elif p == "dtau0":
            cols.append(amp_d[:, 1]); tcols.append(amp_t[1])
        else:
            cols.append(D[:, ix[p]]); tcols.append(T[ix[p]])
    return np.array(cols).T, np.array(tcols)

def load_leg(root, leg, prefix="prod_sbc_loso_"):
    out = []
    for k in range(8):
        for f in sorted(glob.glob(f"{root}/{prefix}fold{k}_{leg}/mock_*.pkl")):
            try:
                d = pickle.load(open(f, "rb"))
                nm = list(d["names"])
                if any(p not in nm for p in [f"tau0_z{i}" for i in range(13)]
                       + ["ns", "Ap", "alpha_lls", "alpha_subdla"]):
                    continue
                X, T = get_arrays(d)
            except Exception:
                continue
            out.append(dict(X=X, T=T, fold=k, ns_phys=parse_phys_ns(d["sim"]),
                            ns_unit=float(T[0]), n_div=int(d.get("n_div", 0)),
                            file=os.path.basename(f)))
    out.sort(key=lambda r: r["ns_unit"]); return out

def cov_ellipse(ax, x, y, nsig, **kw):
    mu = [x.mean(), y.mean()]; C = np.cov(x, y); w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    wd, ht = 2 * nsig * np.sqrt(np.maximum(w[::-1], 0))
    ax.add_patch(Ellipse(mu, wd, ht, angle=ang, fill=False, **kw))

def corner(rec, leg, outpath, color="#2c6fbb"):
    X, truth = rec["X"], rec["T"]; P = X.shape[1]
    labs = [LABELS[p] for p in CORNER_PARAMS]
    fig, axes = plt.subplots(P, P, figsize=(2.45 * P, 2.45 * P))
    # coverage flag per param (is truth inside the central 68%?)
    for r in range(P):
        for c in range(P):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                if r == 0 and c == P - 1:
                    ax.text(0.5, 0.6, "red = truth", color="red", ha="center", fontsize=11)
                    ax.text(0.5, 0.35, "1σ/2σ contours", color=color, ha="center", fontsize=10)
                continue
            if r == c:
                ax.hist(X[:, r], bins=22, density=True, color=color, alpha=0.35)
                ax.hist(X[:, r], bins=22, density=True, histtype="step", color=color, lw=1.5)
                ax.axvline(truth[r], color="red", lw=1.8)
                lo, hi = np.percentile(X[:, r], [16, 84])
                inside = lo <= truth[r] <= hi
                ax.set_title(("✓" if inside else "✗") + f" {labs[r]}", fontsize=9,
                             color=("green" if inside else "crimson"))
                if CORNER_PARAMS[r] in UNIT: ax.set_xlim(0, 1)
                ax.set_yticks([])
            else:
                ax.scatter(X[:, c], X[:, r], s=5, color=color, alpha=0.13, zorder=2)
                for s in (1, 2): cov_ellipse(ax, X[:, c], X[:, r], s, edgecolor=color, lw=1.3, zorder=4)
                ax.scatter([truth[c]], [truth[r]], marker="*", s=210, color="red",
                           edgecolor="k", linewidth=0.5, zorder=6)
                if CORNER_PARAMS[c] in UNIT: ax.set_xlim(0, 1)
                if CORNER_PARAMS[r] in UNIT: ax.set_ylim(0, 1)
            if r == P - 1: ax.set_xlabel(labs[c], fontsize=9)
            else: ax.set_xticklabels([])
            if c == 0 and r > 0: ax.set_ylabel(labs[r], fontsize=9)
            elif r != c: ax.set_yticklabels([])
    fig.suptitle(f"{leg} held-out SBC corner — sim n_s={rec['ns_phys']:.3f} "
                 f"(fold {rec['fold']}, n_div={rec['n_div']})\n"
                 f"red ★/line = truth;  n_s,A_p on unit cube [0,1]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(outpath, dpi=120); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="KS")
    ap.add_argument("--which", default="all", choices=["all", "low", "mid", "high"])
    ap.add_argument("--root", default="/scratch/cavestru_root/cavestru1/mfho")
    # working diagnostic figures live in the PRIVATE notes repo, not the code repo
    ap.add_argument("--outdir", default="/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation/heldout_ns_shrinkage")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    M = load_leg(a.root, a.leg)
    picks = {"low": M[0], "mid": M[len(M)//2], "high": M[-1]}
    todo = picks if a.which == "all" else {a.which: picks[a.which]}
    for tag, rec in todo.items():
        p = f"{a.outdir}/sbc_corner_{a.leg}_{tag}.png"
        corner(rec, a.leg, p)
        print(f"  {a.leg} {tag}: sim n_s={rec['ns_phys']:.3f} ndiv={rec['n_div']} -> {p}")
