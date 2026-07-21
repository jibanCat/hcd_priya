#!/usr/bin/env python3
"""Figures for the held-out subDLA/LLS dN/dX consistency finding (PI directive 2026-06-19).

Story: the alpha_subdla/-lls sigma-pulls (-1.7/-1.5) are a TIGHT-POSTERIOR artifact on a
STABLE, modest physical under-recovery (prior centered on data dN/dX < PRIYA sim incidence),
and cosmology (ns,Ap) stays unbiased -> not a worry.

Writes 2 figures into the NOTES repo 05_likelihood dir.
Usage: PYTHONPATH=/home/mfho/hcd_priya python3 scripts/plot_heldout_dndx_consistency.py
"""
import pickle, glob, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
os.makedirs(OUT, exist_ok=True)
ARMS = [
    ("0.40 pilot", "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout", "#c44"),
    ("0.20 arm",   "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout_amp020", "#268"),
]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "mock_*.pkl"))):
        try:
            out[os.path.basename(f).replace("mock_", "").replace(".pkl", "")] = pickle.load(open(f, "rb"))
        except Exception:
            pass
    return out


arms = [(lbl, load(d), c) for lbl, d, c in ARMS]
names = None
for _, a, _ in arms:
    for x in a.values():
        names = list(x["names"]); break
    if names: break
JI = {n: names.index(n) for n in ["alpha_subdla", "alpha_lls", "ns", "Ap"]}


def col(x, j):
    dr = np.asarray(x["draws"])[:, j]
    return float(dr.mean()), float(dr.std()), float(x["truth_vec"][j])


# ===== Figure 1: recovered vs sim-truth incidence, per mock =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, cls, title in [(axes[0], "alpha_subdla", r"subDLA incidence  $\alpha_{\rm subDLA}$"),
                       (axes[1], "alpha_lls", r"LLS incidence  $\alpha_{\rm LLS}$")]:
    j = JI[cls]
    for k, (lbl, a, c) in enumerate(arms):
        mids = sorted(a.keys())
        xs = np.array([int(m) for m in mids]) + (k - 0.5) * 0.16
        rec = np.array([col(a[m], j)[0] for m in mids])
        sd = np.array([col(a[m], j)[1] for m in mids])
        ax.errorbar(xs, rec, yerr=sd, fmt="o", color=c, capsize=3, ms=6,
                    label=f"{lbl} recovered  (rec/truth≈{np.mean(rec/np.array([col(a[m],j)[2] for m in mids])):.2f})")
    # truth (use the 0.20 arm = all 8 mocks)
    a8 = arms[1][1]
    mids = sorted(a8.keys())
    tx = np.array([int(m) for m in mids])
    tt = np.array([col(a8[m], j)[2] for m in mids])
    ax.plot(tx, tt, "k*", ms=13, label="sim truth", zorder=5)
    ax.set_title(title); ax.set_xlabel("held-out mock id"); ax.set_ylabel(r"$\alpha$ (incidence weight)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
fig.suptitle("Recovered subDLA/LLS incidence sits COHERENTLY below sim truth — central value STABLE across prior width,\n"
             "only the error bar shrinks 0.40→0.20 (data sets the recovered dN/dX; prior sets the error bar)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p1 = os.path.join(OUT, "sbc_heldout_dndx_recovered_vs_truth.png")
fig.savefig(p1, dpi=120); plt.close(fig)
print("wrote", p1)

# ===== Figure 2: pull = offset/post_sd decomposition  +  cosmology stays clean =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# left: decomposition bars
ax = axes[0]
classes = ["alpha_subdla", "alpha_lls"]
clabels = ["subDLA", "LLS"]
W = 0.35
for k, (lbl, a, c) in enumerate(arms):
    pulls, offs, sds = [], [], []
    for cls in classes:
        j = JI[cls]
        mids = sorted(a.keys())
        rec = np.array([col(a[m], j)[0] for m in mids])
        sd = np.array([col(a[m], j)[1] for m in mids])
        tt = np.array([col(a[m], j)[2] for m in mids])
        pulls.append(np.mean((rec - tt) / sd))
        offs.append(np.mean(np.abs(rec - tt)))
        sds.append(np.mean(sd))
    xs = np.arange(len(classes)) + (k - 0.5) * W
    ax.bar(xs, np.abs(pulls), W, color=c, label=f"{lbl}  |σ-pull|")
    for xi, p, o, s in zip(xs, pulls, offs, sds):
        ax.annotate(f"|Δ|={o:.3f}\nσ_post={s:.3f}", (xi, abs(p)), ha="center", va="bottom", fontsize=7)
ax.axhline(1.0, ls="--", c="grey", lw=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(clabels)
ax.set_ylabel(r"$|$mean $\sigma$-pull$|$"); ax.set_ylim(0, 2.3)
ax.set_title("Tightening 0.40→0.20: |Δ| (abs offset) ≈ unchanged,\nσ_post SHRINKS → σ-pull GROWS (artifact)")
ax.legend(fontsize=8)

# right: cosmology pulls clean with gate band
ax = axes[1]
cosmo = ["ns", "Ap"]
for k, (lbl, a, c) in enumerate(arms):
    for ci, cp in enumerate(cosmo):
        j = JI[cp]
        mids = sorted(a.keys())
        pull = np.array([(col(a[m], j)[0] - col(a[m], j)[2]) / col(a[m], j)[1] for m in mids])
        x = ci + (k - 0.5) * W
        ax.errorbar(x, pull.mean(), yerr=pull.std() / np.sqrt(len(pull)), fmt="s", color=c, capsize=4, ms=9,
                    label=lbl if ci == 0 else None)
        ax.annotate(f"{pull.mean():+.2f}σ", (x, pull.mean()), ha="center", va="bottom", fontsize=8)
ax.axhspan(-0.3, 0.3, color="green", alpha=0.12, label="n_s gate |mean|≤0.3σ")
ax.axhline(0, c="k", lw=0.8)
ax.set_xticks(range(len(cosmo))); ax.set_xticklabels([r"$n_s$", r"$A_p$"])
ax.set_ylabel("cosmology mean σ-pull"); ax.set_ylim(-1.0, 1.0)
ax.set_title("Cosmology stays UNBIASED at both widths\n(despite the coherent HCD under-recovery)")
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("The α_subDLA/α_LLS pull is a tight-posterior artifact on a stable offset — it does NOT leak into cosmology",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
p2 = os.path.join(OUT, "sbc_heldout_dndx_pull_artifact_cosmo_clean.png")
fig.savefig(p2, dpi=120); plt.close(fig)
print("wrote", p2)
