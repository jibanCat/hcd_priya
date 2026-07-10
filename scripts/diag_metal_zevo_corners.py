#!/usr/bin/env python3
"""Model C+ metal-injection CORNER plots, <=3 runs per corner (READ-ONLY on the pkls).

Adapts scripts/diag_dnuis_cell_corners.py to the metal_zevo campaign. Each corner overlays a CLEAN
reference (no injection) with TWO injected arms (so at most 3 runs per corner), grouped by survey x
class: DESI/eBOSS x {in-class arm1+arm2, out-of-class arm3+arm4}. Axes are (Delta n_s, Delta A_p,
Delta tau0_amp, Delta dtau0), each draw plotted as (draw - per-mock truth) and pooled over mocks so
truth sits at the origin; the injected cloud's offset off origin (relative to clean) is the paired
bias the analyzer certifies. tau0_amp/dtau0 are the ACTUAL sampled sites (rec['sites_extra']), not
the tau0_z ladder proxy. a_SiIII is absent under flatlog2node (metal nodes not packed), so it is not
an axis. KDE 68/95% contours, reusing the dnuis-corner style.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_metal_zevo_corners.py
"""
import argparse, glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 14,
                     "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
CLEAN_C = "#555555"
ARM_C = {"a": "#c0392b", "b": "#1f77b4"}   # two injected arms per corner
TRU_C = "#111111"
KDE_BW, KDE_BW_2D, GRID_SMOOTH = 1.35, 1.45, 1.1
BASE = "/scratch/cavestru_root/cavestru1/mfho/metal_zevo"
FIG_DIR = "/home/mfho/hcd_priya_notes/figures/analysis/07_gate_b_metal"
LABS = {"ns": r"$\Delta n_s$", "Ap": r"$\Delta A_p$",
        "tau0_amp": r"$\Delta\tau_{0,\rm amp}$", "dtau0": r"$\Delta(d\tau_0)$"}
COLS = ["ns", "Ap", "tau0_amp", "dtau0"]

# corner groups: (key, survey, title, [(arm, "in/out label"), ...])  -- clean + 2 arms = 3 runs
GROUPS = [
    ("desi_inclass", "desi", "DESI in-class (sigmoid-fittable SiIII+SiII+cross)",
     [("arm1_decreasing", "arm1: flat f, k=0.05"), ("arm2_increasing", "arm2: rising f, k=0.012")]),
    ("desi_ooc", "desi", "DESI out-of-class (envelope the sigmoid cannot match)",
     [("arm3_ma2025", "arm3: Ma+2025 exp"), ("arm4_decreasing_ooc", "arm4: gauss damp")]),
    ("eboss_inclass", "eboss", "eBOSS in-class (SiIII-only, coarse k)",
     [("arm1_decreasing", "arm1: flat f, k=0.05"), ("arm2_increasing", "arm2: rising f, k=0.012")]),
    ("eboss_ooc", "eboss", "eBOSS out-of-class",
     [("arm3_ma2025", "arm3: Ma+2025 exp"), ("arm4_decreasing_ooc", "arm4: gauss damp")]),
]


def load_cell(arm, survey):
    fs = sorted(glob.glob(os.path.join(BASE, f"metal_zevo_{arm}_{survey}_shard_*.pkl")))
    clean, inj = [], []
    for p in fs:
        d = pickle.load(open(p, "rb"))
        clean.extend(d["clean_per_mock"]); inj.extend(d["inj_per_mock"])
    return clean, inj


def _col(rec, key):
    """(draws_col, truth) for an axis. ns/Ap by name from draws; tau0_amp/dtau0 from sites_extra."""
    if key in ("tau0_amp", "dtau0"):
        se = rec.get("sites_extra") or {}
        if key not in se:
            return None, None
        return np.asarray(se[key]["draws"], float), float(se[key]["truth"])
    names = list(rec["names"])
    if key not in names:
        return None, None
    j = names.index(key)
    return np.asarray(rec["draws"], float)[:, j], float(np.asarray(rec["truth_vec"], float)[j])


def stacked(recs, cols):
    out = {c: [] for c in cols}
    for rec in recs:
        for c in cols:
            d, t = _col(rec, c)
            if d is not None:
                out[c].append(d - t)
    return {c: (np.concatenate(v) if v else np.array([])) for c, v in out.items()}


def bias_z(recs, key):
    zs = []
    for rec in recs:
        d, t = _col(rec, key)
        if d is not None and d.std() > 0:
            zs.append((d.mean() - t) / d.std())
    return np.array(zs)


def kde_1d(ax, x, color, lw=2.2, fa=0.16):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 3 or np.ptp(x) == 0:
        if x.size: ax.axvline(float(np.median(x)), color=color, lw=lw)
        return
    pad = np.ptp(x) * 0.10 + 1e-9
    g = np.linspace(x.min() - pad, x.max() + pad, 256)
    try:
        d = gaussian_kde(x, bw_method=KDE_BW)(g)
    except np.linalg.LinAlgError:
        ax.axvline(float(np.median(x)), color=color, lw=lw); return
    ax.fill_between(g, d, color=color, alpha=fa, zorder=1)
    ax.plot(g, d, color=color, lw=lw, zorder=3, solid_capstyle="round")


def kde_contour(ax, x, y, color, fa=0.20, lw=2.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if x.size < 6 or np.ptp(x) == 0 or np.ptp(y) == 0:
        if x.size: ax.scatter(x, y, s=12, color=color, alpha=0.7, zorder=3, edgecolors="none")
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=KDE_BW_2D)
    except np.linalg.LinAlgError:
        ax.scatter(x, y, s=12, color=color, alpha=0.7, zorder=3, edgecolors="none"); return
    px = np.ptp(x) * 0.30 + 1e-9; py = np.ptp(y) * 0.30 + 1e-9
    xx, yy = np.mgrid[x.min() - px:x.max() + px:130j, y.min() - py:y.max() + py:130j]
    zz = gaussian_filter(kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape), GRID_SMOOTH)
    zs = np.sort(zz.ravel())[::-1]; cum = np.cumsum(zs) / zs.sum()
    lv = sorted(zs[np.searchsorted(cum, L)] for L in (0.95, 0.68))
    if lv[0] == lv[1]: lv[1] = lv[1] * 1.0001 + 1e-12
    ax.contourf(xx, yy, zz, levels=lv + [zz.max()], colors=[color, color], alpha=fa, zorder=1)
    ax.contour(xx, yy, zz, levels=lv, colors=color, linestyles="-", linewidths=[lw * 0.8, lw], zorder=3)


def make_corner(group, path):
    key, survey, title, arms = group
    # clean reference = the first arm's clean (cleans are ~identical at matched seed/mock/ctx; on
    # eBOSS all arms literally share it). 2 injected arms overlaid -> at most 3 runs per corner.
    clean0, _ = load_cell(arms[0][0], survey)
    rc = stacked(clean0, COLS)
    arm_res, arm_bz = [], []
    for slot, (arm, lab) in zip(("a", "b"), arms):
        _, inj = load_cell(arm, survey)
        arm_res.append((slot, lab, stacked(inj, COLS)))
        arm_bz.append((lab, bias_z(inj, "ns").mean() - bias_z(clean0, "ns").mean(),
                       bias_z(inj, "Ap").mean() - bias_z(clean0, "Ap").mean()))
    P = len(COLS)
    fig, ax = plt.subplots(P, P, figsize=(3.0 * P, 3.0 * P))
    for r in range(P):
        for c in range(P):
            a = ax[r, c]
            if c > r:
                a.axis("off"); continue
            ci, cj = COLS[r], COLS[c]
            if r == c:
                kde_1d(a, rc[ci], CLEAN_C, lw=1.8, fa=0.10)
                for slot, lab, ri in arm_res:
                    kde_1d(a, ri[ci], ARM_C[slot], lw=2.2, fa=0.16)
                a.axvline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=1.4, zorder=4)
                a.set_yticks([]); a.set_ylim(bottom=0)
            else:
                kde_contour(a, rc[cj], rc[ci], CLEAN_C, fa=0.10, lw=1.5)
                for slot, lab, ri in arm_res:
                    kde_contour(a, ri[cj], ri[ci], ARM_C[slot], fa=0.20, lw=2.0)
                a.axvline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=0.9, zorder=4)
                a.axhline(0.0, color=TRU_C, ls=(0, (4, 2)), lw=0.9, zorder=4)
                a.plot(0.0, 0.0, marker="*", ms=12, color=TRU_C, mec="white", mew=0.8, zorder=5)
            a.grid(True, color="0.9", lw=0.5, zorder=0); a.tick_params(length=3)
            if r == P - 1:
                a.set_xlabel(LABS[COLS[c]]); a.tick_params(axis="x", labelrotation=30)
                for lbl in a.get_xticklabels(): lbl.set_ha("right")
            else:
                a.set_xticklabels([])
            if c == 0 and r > 0:
                a.set_ylabel(LABS[COLS[r]])
            elif r != c:
                a.set_yticklabels([])
    handles = [plt.Line2D([], [], color=CLEAN_C, lw=4, label="CLEAN reference (no injection)")]
    for (slot, lab, _), (lab2, dz_ns, dz_ap) in zip(arm_res, arm_bz):
        handles.append(plt.Line2D([], [], color=ARM_C[slot], lw=4,
                       label=f"{lab}   ($\\Delta n_s$={dz_ns:+.2f}, $\\Delta A_p$={dz_ap:+.2f})"))
    handles.append(plt.Line2D([], [], color=TRU_C, ls=(0, (4, 2)), lw=1.5, marker="*", ms=12,
                   mec="white", mew=0.8, label="truth (origin; per-mock truth subtracted)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.895), frameon=True,
               framealpha=0.95, edgecolor="0.8", borderpad=0.8, labelspacing=0.6)
    fig.suptitle(f"Model C+ metal injection corner: {title}", fontsize=14, y=0.985)
    fig.text(0.5, 0.955, r"contours 68/95%; axes = draw $-$ per-mock truth pooled over mocks; "
             "injected clouds shift up in $\\Delta A_p$ (anti-correlated with $\\Delta\\tau_0$) = the leak",
             ha="center", fontsize=11, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=135, bbox_inches="tight"); plt.close(fig)
    print(f"[corner] {key:16s} clean={max(v.size for v in rc.values()):4d} draws -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=FIG_DIR)
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    for g in GROUPS:
        make_corner(g, os.path.join(a.fig_dir, f"corner_{g[0]}.png"))


if __name__ == "__main__":
    main()
