#!/usr/bin/env python3
"""Per-leg resolution 4-arm bracket: corner OVERPLOT of the arms (bias) + a good/bad diagnostic panel.

For each leg with a resolution bracket (DESI: a/b/d; eBOSS: a/b/c/d -- KS has NO bracket, R_z-blocked),
overplot each arm's INJECTED-arm posterior in BIAS space (draw - per-mock truth), 68/95% KDE contours,
truth at the origin. Params: n_s, A_p (cosmology) + h, tau0 (the mechanism -- arm-D leaks via h, option-a
dumps into tau0). A second panel shows per-arm |Delta|+2SE (n_s, A_p) and sigma(n_s) = the good/bad summary.

Arms: a = option-a (res in cov, no float); b = option-b TIGHT N(0,0.02); c = option-b WIDE N(0,0.05);
d = arm-D (coherent cross-z cov). Gate = |mean Delta_bias|+2SE < 0.30 sigma_post on n_s AND A_p.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu ... python3 this.py
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
BASE = "/scratch/cavestru_root/cavestru0/mfho/res_bracket"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/08_resolution"
KDE_BW, GRID_SMOOTH = 0.9, 1.0

LEGS = {
    "DESI":  dict(d="desi_bres_002",  arms=[("a", "resolution_desi"), ("b", "resolution_b_desi"),
                                            ("d", "resolution_d_desi")]),
    "eBOSS": dict(d="eboss_bres_0044", arms=[("a", "resolution_a_eboss"), ("b", "resolution_b_eboss"),
                                             ("c", "resolution_c_eboss"), ("d", "resolution_d_eboss")]),
}
ALAB = {"a": "option-a (res in cov)", "b": "option-b tight N(0,.02)", "c": "option-b WIDE N(0,.05)",
        "d": "arm-D (coherent cov)"}
ACOL = {"a": "#D62728", "b": "#1F77B4", "c": "#2CA02C", "d": "#9467BD"}
# CERTIFIED per-arm gate budget |mean Delta|+2SE in sigma_post (analyze_dnuis_bias.py, FIXED sigma_ref =
# the M1 normalization so an arm cannot "pass" by ballooning sigma). (n_s, A_p). DESI b = the N=10 N-bump.
CERT = {"DESI":  {"a": (2.227, 1.442), "b": (0.256, 0.144), "d": (2.214, 1.707)},
        "eBOSS": {"a": (0.867, 4.214), "b": (0.410, 0.750), "c": (0.221, 0.102), "d": (0.915, 0.847)}}
COLS = ["ns", "Ap", "hub", "tau0_mean"]
LAB = {"ns": r"$\Delta n_s$", "Ap": r"$\Delta A_p$", "hub": r"$\Delta h$", "tau0_mean": r"$\Delta \bar\tau_0$"}


def load_arm(legd, prefix):
    clean, inj = [], []
    for f in sorted(glob.glob(f"{BASE}/{legd}/{prefix}_shard_*.pkl")):
        d = pickle.load(open(f, "rb")); clean += d["clean_per_mock"]; inj += d["inj_per_mock"]
    return clean, inj


def _col_vals(rec, col):
    names = list(rec["names"]); draws = np.asarray(rec["draws"], float); truth = np.asarray(rec["truth_vec"], float)
    if col == "tau0_mean":
        idx = [i for i, n in enumerate(names) if n.startswith("tau0_z")]
        return draws[:, idx].mean(1), float(truth[idx].mean())
    i = names.index(col); return draws[:, i], float(truth[i])


def resid_pool(recs, col):                     # pooled (draw - truth) over mocks (bias space)
    out = [(_col_vals(r, col)[0] - _col_vals(r, col)[1]) for r in recs]
    return np.concatenate(out) if out else np.array([])


def budget(clean, inj, col):                   # |mean Delta|+2SE in sigma_post (sref = median clean sd)
    dm, sd = [], []
    for c, ii in zip(clean, inj):
        vc, _ = _col_vals(c, col); vi, _ = _col_vals(ii, col)
        dm.append(vi.mean() - vc.mean()); sd.append(vc.std())
    dm = np.asarray(dm); sref = np.median(sd)
    se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
    return (abs(dm.mean()) + 2 * se) / sref, np.median(sd)


def kde_1d(ax, x, color, lw=2.0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 6 or np.ptp(x) == 0:
        return
    kde = gaussian_kde(x, bw_method=KDE_BW); g = np.linspace(x.min() - 0.3 * np.ptp(x), x.max() + 0.3 * np.ptp(x), 220)
    ax.plot(g, kde(g), color=color, lw=lw); ax.fill_between(g, kde(g), color=color, alpha=0.12)


def kde_contour(ax, x, y, color, lw=2.0):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if x.size < 6 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=KDE_BW)
    except np.linalg.LinAlgError:
        return
    px, py = np.ptp(x) * 0.3 + 1e-9, np.ptp(y) * 0.3 + 1e-9
    xx, yy = np.mgrid[x.min() - px:x.max() + px:130j, y.min() - py:y.max() + py:130j]
    zz = gaussian_filter(kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape), GRID_SMOOTH)
    zs = np.sort(zz.ravel())[::-1]; cum = np.cumsum(zs) / zs.sum()
    lv = sorted(zs[np.searchsorted(cum, L)] for L in (0.95, 0.68))
    if lv[0] == lv[1]:
        lv[1] = lv[1] * 1.0001 + 1e-12
    ax.contourf(xx, yy, zz, levels=lv + [zz.max()], colors=[color, color], alpha=0.16, zorder=1)
    ax.contour(xx, yy, zz, levels=lv, colors=color, linewidths=[lw * 0.8, lw], zorder=3)


for leg, cfg in LEGS.items():
    arms = cfg["arms"]
    R = {a: {c: None for c in COLS} for a, _ in arms}; BUD = {}
    for a, pref in arms:
        clean, inj = load_arm(cfg["d"], pref)
        for c in COLS:
            R[a][c] = resid_pool(inj, c)
        BUD[a] = {c: budget(clean, inj, c)[0] for c in ("ns", "Ap")}
        BUD[a]["sig_ns"] = budget(clean, inj, "ns")[1]

    # ---- corner ----
    P = len(COLS); fig, ax = plt.subplots(P, P, figsize=(3.0 * P, 3.0 * P))
    for r in range(P):
        for c in range(P):
            a = ax[r, c]
            if c > r:
                a.axis("off"); continue
            for atag, _ in arms:
                if r == c:
                    kde_1d(a, R[atag][COLS[r]], ACOL[atag])
                else:
                    kde_contour(a, R[atag][COLS[c]], R[atag][COLS[r]], ACOL[atag])
            if r == c:
                a.axvline(0, color="k", ls=(0, (4, 2)), lw=1.3, zorder=4); a.set_yticks([]); a.set_ylim(bottom=0)
            else:
                a.axvline(0, color="k", ls=(0, (4, 2)), lw=0.9); a.axhline(0, color="k", ls=(0, (4, 2)), lw=0.9)
                a.plot(0, 0, marker="*", ms=12, color="k", mec="white", mew=0.8, zorder=5)
            a.grid(True, color="0.92", lw=0.5); a.tick_params(length=3)
            a.set_xlabel(LAB[COLS[c]]) if r == P - 1 else a.set_xticklabels([])
            (a.set_ylabel(LAB[COLS[r]]) if (c == 0 and r > 0) else (None if r == c else a.set_yticklabels([])))
    handles = []
    for atag, _ in arms:
        bn, ba = CERT[leg][atag]; pf = "PASS" if (bn < 0.3 and ba < 0.3) else "FAIL"
        handles.append(Line2D([], [], color=ACOL[atag], lw=4,
                              label=f"{ALAB[atag]}:  n_s {bn:.2f} / A_p {ba:.2f}  [{pf}]"))
    handles.append(Line2D([], [], color="k", ls=(0, (4, 2)), marker="*", ms=12, mec="white",
                          label="truth (origin)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.98), fontsize=12,
               frameon=True, framealpha=0.95, edgecolor="0.8")
    fig.suptitle(f"Resolution 4-arm bracket -- {leg}: injected-arm bias (draw - truth), 68/95% contours "
                 f"[gate |Delta|+2SE < 0.30 sigma_post]", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/resolution_arms_corner_{leg.lower()}.png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[corner] {leg} -> {p}")

    # ---- good/bad diagnostic panel: |Delta|+2SE (ns, Ap) + sigma(ns) ----
    figd, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(arms)); w = 0.38
    a1.bar(x - w / 2, [min(CERT[leg][a][0], 1.4) for a, _ in arms], w, color="#1F77B4", label="n_s", edgecolor="k", lw=0.4)
    a1.bar(x + w / 2, [min(CERT[leg][a][1], 1.4) for a, _ in arms], w, color="#F58518", label="A_p", edgecolor="k", lw=0.4)
    for i, (a, _) in enumerate(arms):
        for off, ki in ((-w / 2, 0), (w / 2, 1)):
            if CERT[leg][a][ki] > 1.4:
                a1.text(x[i] + off, 1.42, f"{CERT[leg][a][ki]:.1f}", ha="center", fontsize=8, color="dimgray")
    a1.axhline(0.30, color="crimson", ls="--", lw=1.5); a1.text(len(arms) - 0.5, 0.32, "0.30 gate", color="crimson", ha="right", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels([a for a, _ in arms]); a1.set_ylim(0, 1.55)
    a1.set_ylabel(r"$|\Delta_{\rm bias}|+2\,$SE  [$\sigma_{\rm post}$]"); a1.set_title(f"{leg}: bias (lower=better)"); a1.legend(fontsize=9); a1.grid(True, alpha=0.25)
    a2.bar(x, [BUD[a]["sig_ns"] for a, _ in arms], 0.55, color="#54A24B", edgecolor="k", lw=0.4)
    a2.set_xticks(x); a2.set_xticklabels([a for a, _ in arms]); a2.set_ylabel(r"$\sigma(n_s)$ posterior width")
    a2.set_title(f"{leg}: precision cost (a 'pass' by ballooning $\\sigma$ is not a win)"); a2.grid(True, alpha=0.25)
    figd.suptitle(f"Resolution arms -- {leg}: good/bad summary (a=option-a b=opt-b-tight"
                  f"{' c=opt-b-wide' if leg=='eBOSS' else ''} d=arm-D)", fontsize=12)
    figd.tight_layout(); pd = f"{OUT}/resolution_arms_diag_{leg.lower()}.png"; figd.savefig(pd, dpi=130, bbox_inches="tight"); plt.close(figd)
    print(f"[diag]   {leg} -> {pd}")
