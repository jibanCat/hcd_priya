#!/usr/bin/env python3
"""Figures for the 2026-06-21 prior-arm results doc:
  (1) tau0-informative prior A/B (KS held-out, uniform vs informative),
  (2) subDLA x1.5 displaced-truth (self-draw eBOSS) n_s leak.
Working diagnostic figs -> notes repo. Recovery slope <1 = shrinkage; the informative
arm should lift the A_p (and n_s) slope toward 1 and shrink the posterior."""
import pickle, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hcd_analysis.emulator.meanflux_prior import kim_tau0, fit_tau0_alpha_priya

SCR = "/scratch/cavestru_root/cavestru1/mfho"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation/prior_arms"
os.makedirs(OUT, exist_ok=True)
Z = 2.2 + 0.2 * np.arange(13); KIM = np.asarray(kim_tau0(Z))

def _dtau0(ladder):
    return fit_tau0_alpha_priya(Z, np.clip(np.asarray(ladder) / KIM, 1e-6, None))[1]

def load_ks(leg):
    out = []
    for k in range(8):
        for f in sorted(glob.glob(f"{SCR}/prod_sbc_loso_fold{k}_{leg}/mock_*.pkl")):
            d = pickle.load(open(f, "rb")); nm = list(d["names"])
            if any(f"tau0_z{i}" not in nm for i in range(13)): continue
            D = np.asarray(d["draws"]); T = np.asarray(d["truth_vec"])
            ix = {p: nm.index(p) for p in ("ns", "Ap")}; ti = [nm.index(f"tau0_z{i}") for i in range(13)]
            out.append(dict(ns_t=T[ix["ns"]], ns_pm=D[:, ix["ns"]].mean(),
                            Ap_t=T[ix["Ap"]], Ap_pm=D[:, ix["Ap"]].mean(), Ap_ps=D[:, ix["Ap"]].std(ddof=1),
                            dtau_pm=np.array([_dtau0(D[j, ti]) for j in range(len(D))]).mean(),
                            dtau_ps=np.array([_dtau0(D[j, ti]) for j in range(len(D))]).std(ddof=1),
                            dtau_t=_dtau0(T[ti])))
    return out

def arr(M, k): return np.array([m[k] for m in M])

uni, inf = load_ks("KS"), load_ks("KS_tau0inf")

# ---------- Fig 1: tau0-informative A/B ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for M, c, lab in [(uni, "#999999", "uniform tau0"), (inf, "#1f6fb4", "informative tau0")]:
    s = np.polyfit(arr(M, "Ap_t"), arr(M, "Ap_pm"), 1)[0]
    ax[0].scatter(arr(M, "Ap_t"), arr(M, "Ap_pm"), s=26, color=c, alpha=0.7,
                  edgecolor="k", lw=0.3, label=f"{lab} (slope {s:.2f})")
    s = np.polyfit(arr(M, "ns_t"), arr(M, "ns_pm"), 1)[0]
    ax[1].scatter(arr(M, "ns_t"), arr(M, "ns_pm"), s=26, color=c, alpha=0.7,
                  edgecolor="k", lw=0.3, label=f"{lab} (slope {s:.2f})")
for a, lab in zip(ax[:2], ["A_p", "n_s"]):
    a.plot([0, 1], [0, 1], "k--", lw=1, label="unbiased (y=x)")
    a.axhline(0.5, color="crimson", ls=":", lw=1, label="prior center")
    a.set_xlim(-.02, 1.02); a.set_ylim(-.02, 1.02)
    a.set_xlabel(f"truth {lab} (unit cube)"); a.set_ylabel(f"posterior mean {lab}")
    a.set_title(f"{lab} recovery: informative lifts the slope\n(less shrinkage)", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")
# dtau0 pull
for M, c, lab in [(uni, "#999999", "uniform"), (inf, "#1f6fb4", "informative")]:
    pull = (arr(M, "dtau_pm") - arr(M, "dtau_t")) / arr(M, "dtau_ps")
    ax[2].hist(pull, bins=16, range=(-2, 3), histtype="stepfilled", color=c, alpha=0.45,
               label=f"{lab} (mean {pull.mean():+.2f})")
    ax[2].axvline(pull.mean(), color=c, lw=2)
ax[2].axvline(0, color="crimson", ls=":", lw=1.2, label="truth (Kim-central)")
ax[2].set_xlabel("dtau0 pull = (mean - truth)/sd"); ax[2].set_ylabel("# mocks")
ax[2].set_title("Mean-flux slope bias:\ninformative pulls it toward truth", fontsize=10)
ax[2].legend(fontsize=8)
fig.suptitle("tau0-informative prior A/B  —  KS held-out, all 8 folds (Kim-centered TruncatedNormal sigma=0.05 vs uniform)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); p1 = f"{OUT}/tau0inf_AB.png"; fig.savefig(p1, dpi=130); plt.close(fig)

# ---------- Fig 2: subDLA displaced-truth ----------
fs = sorted(glob.glob(f"{SCR}/prod_sbc_selfdraw_eBOSS_subdla15/mock_*.pkl"))
sub_t = []; sub_pm = []; ns_pull = []
for f in fs:
    d = pickle.load(open(f, "rb")); nm = list(d["names"])
    i_s = nm.index("alpha_subdla"); i_n = nm.index("ns")
    Ds = np.asarray(d["draws"])[:, i_s]; Dn = np.asarray(d["draws"])[:, i_n]
    sub_t.append(float(d["truth_vec"][i_s])); sub_pm.append(Ds.mean())
    ns_pull.append((Dn.mean() - float(d["truth_vec"][i_n])) / Dn.std(ddof=1))
sub_t = np.array(sub_t); sub_pm = np.array(sub_pm); ns_pull = np.array(ns_pull)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(sub_t, sub_pm, s=34, color="#b2182b", alpha=0.75, edgecolor="k", lw=0.3)
lo, hi = sub_t.min() * .9, sub_t.max() * 1.05
ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="unbiased")
ax[0].set_xlabel("displaced subDLA truth (x1.5 above pin)"); ax[0].set_ylabel("recovered subDLA")
ax[0].set_title(f"subDLA under-recovered\n(prior pulls the x1.5 truth down; pull {ns_pull.mean():+.2f} is n_s)", fontsize=10)
ax[0].legend(fontsize=8)
ax[1].hist(ns_pull, bins=12, range=(-3, 2), histtype="stepfilled", color="#b2182b", alpha=0.5)
ax[1].axvline(ns_pull.mean(), color="#b2182b", lw=2.5, label=f"mean {ns_pull.mean():+.2f} sigma")
ax[1].axvline(0, color="k", ls=":", lw=1.2, label="unbiased")
ax[1].set_xlabel("n_s pull = (mean - truth)/sd"); ax[1].set_ylabel("# mocks")
ax[1].set_title("n_s LEAKS low when subDLA truth\nsits above the prior center", fontsize=10)
ax[1].legend(fontsize=8)
fig.suptitle("subDLA x1.5 displaced-truth  —  self-draw eBOSS (the real-fit subDLA-center -> n_s risk)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); p2 = f"{OUT}/subdla_displaced_eBOSS.png"; fig.savefig(p2, dpi=130); plt.close(fig)

print("wrote", p1); print("wrote", p2)
print(f"N: uniform={len(uni)} informative={len(inf)} subdla={len(fs)}")

# ============================ contour-only corners (no points) ============================
from scipy.stats import gaussian_kde

def kde_1d(ax, x, color, lw=1.8, ls="-"):
    g = np.linspace(x.min(), x.max(), 200); ax.plot(g, gaussian_kde(x)(g), color=color, lw=lw, ls=ls)

def kde_contour(ax, x, y, color, ls="-"):
    kde = gaussian_kde(np.vstack([x, y]))
    px = (x.max()-x.min())*0.25 + 1e-9; py = (y.max()-y.min())*0.25 + 1e-9
    xx, yy = np.mgrid[x.min()-px:x.max()+px:70j, y.min()-py:y.max()+py:70j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zs = np.sort(zz.ravel())[::-1]; cum = np.cumsum(zs)/zs.sum()
    lv = sorted(zs[np.searchsorted(cum, L)] for L in (0.95, 0.68))
    ax.contour(xx, yy, zz, levels=lv, colors=color, linestyles=ls, linewidths=1.6)

def corner(ax, cols, draws_sets, truth, labels):
    """draws_sets = list of (label,color,ls,DRAWS[ndraw,ncol]); contours only, KDE 1-D diagonal."""
    P = len(cols)
    for r in range(P):
        for c in range(P):
            a = ax[r, c]
            if c > r: a.axis("off"); continue
            for lab, col, ls, D in draws_sets:
                if r == c:
                    kde_1d(a, D[:, r], col, ls=ls)
                else:
                    kde_contour(a, D[:, c], D[:, r], col, ls=ls)
            if truth is not None:
                if r == c: a.axvline(truth[r], color="crimson", ls=":", lw=1.3)
                else:
                    a.axvline(truth[c], color="crimson", ls=":", lw=1.0)
                    a.axhline(truth[r], color="crimson", ls=":", lw=1.0)
            if r == c: a.set_yticks([])
            if r == P-1: a.set_xlabel(labels[c], fontsize=9)
            else: a.set_xticklabels([])
            if c == 0 and r > 0: a.set_ylabel(labels[r], fontsize=9)
            elif r != c: a.set_yticklabels([])

def sites_from_pkl(path, want):
    d = pickle.load(open(path, "rb")); nm = list(d["names"]); D = np.asarray(d["draws"]); T = np.asarray(d["truth_vec"])
    ti = [nm.index(f"tau0_z{i}") for i in range(13)]
    cols = []; tr = []
    for w in want:
        if w == "tau0_amp":
            cols.append(np.array([fit_tau0_alpha_priya(Z, np.clip(D[j, ti]/KIM, 1e-6, None))[0] for j in range(len(D))]))
            tr.append(fit_tau0_alpha_priya(Z, np.clip(T[ti]/KIM, 1e-6, None))[0])
        elif w == "dtau0":
            cols.append(np.array([_dtau0(D[j, ti]) for j in range(len(D))])); tr.append(_dtau0(T[ti]))
        else:
            i = nm.index(w); cols.append(D[:, i]); tr.append(T[i])
    return np.array(cols).T, np.array(tr)

# --- corner A: tau0-informative A/B on the SAME mock (fold4 mock0), uniform vs informative ---
want = ["ns", "Ap", "tau0_amp", "dtau0"]; labs = [r"$n_s$", r"$A_p$", r"$\tau_0$amp", r"$d\tau_0$"]
Du, tr = sites_from_pkl(f"{SCR}/prod_sbc_loso_fold4_KS/mock_0000.pkl", want)
Di, _ = sites_from_pkl(f"{SCR}/prod_sbc_loso_fold4_KS_tau0inf/mock_0000.pkl", want)
fig, ax = plt.subplots(4, 4, figsize=(9.5, 9.5))
corner(ax, want, [("uniform", "#888888", "-", Du), ("informative", "#1f6fb4", "-", Di)], tr, labs)
fig.legend(handles=[plt.Line2D([], [], color="#888888", label="uniform τ₀"),
                    plt.Line2D([], [], color="#1f6fb4", label="informative τ₀"),
                    plt.Line2D([], [], color="crimson", ls=":", label="truth")],
           loc="upper right", fontsize=10, frameon=False)
fig.suptitle("τ₀-informative A/B — one KS mock (same data), contours = 68/95%\n"
             "informative tightens A_p + pins τ₀amp/dτ₀", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); p3 = f"{OUT}/tau0inf_corner.png"; fig.savefig(p3, dpi=130); plt.close(fig)

# --- corner B: subDLA displaced — one eBOSS mock, n_s/A_p/alpha_subdla degeneracy ---
want = ["ns", "Ap", "alpha_subdla"]; labs = [r"$n_s$", r"$A_p$", r"$\alpha_{\rm subDLA}$"]
Ds, trs = sites_from_pkl(f"{SCR}/prod_sbc_selfdraw_eBOSS_subdla15/mock_0000.pkl", want)
fig, ax = plt.subplots(3, 3, figsize=(7.5, 7.5))
corner(ax, want, [("", "#b2182b", "-", Ds)], trs, labs)
fig.legend(handles=[plt.Line2D([], [], color="#b2182b", label="posterior (68/95%)"),
                    plt.Line2D([], [], color="crimson", ls=":", label="displaced truth")],
           loc="upper right", fontsize=10, frameon=False)
fig.suptitle("subDLA displaced-truth — one eBOSS mock\nn_s↔α_subDLA tilt = the live degeneracy", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); p4 = f"{OUT}/subdla_corner.png"; fig.savefig(p4, dpi=130); plt.close(fig)
print("wrote", p3); print("wrote", p4)
