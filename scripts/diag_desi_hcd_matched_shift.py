#!/usr/bin/env python3
"""Matched-pair (prior off-on, SAME data) cosmology shift across the DESI HCD-prior mocks.

The right metric for "how much does removing the dN/dX prior move cosmology": per mock,
(off_mean - on_mean)/on_sd. (The pooled/across-mock-truth normalization washes the
systematic out — that was the bug in the first cosmo-shift figure.)"""
import pickle, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCR = "/scratch/cavestru_root/cavestru1/mfho"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation/desi_hcd_prior"
N = 6
def arm(d):
    o = {}
    for m in range(N):
        r = pickle.load(open(f"{SCR}/{d}/mock_{m:04d}.pkl", "rb")); nm = list(r["names"]); D = np.asarray(r["draws"])
        for p in ("ns", "Ap", "alpha_lls"):
            o.setdefault(p, []).append((D[:, nm.index(p)].mean(), D[:, nm.index(p)].std(ddof=1)))
    return o
on, off = arm("desi_hcd_prior_on"), arm("desi_hcd_prior_off")
sh = {p: np.array([(off[p][m][0]-on[p][m][0])/on[p][m][1] for m in range(N)]) for p in on}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
# left: per-mock n_s & A_p matched shift
w = 0.38; x = np.arange(N)
ax[0].bar(x-w/2, sh["ns"], w, color="#2166ac", label="n_s")
ax[0].bar(x+w/2, sh["Ap"], w, color="#b2182b", label="A_p")
ax[0].axhline(0, color="k", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"mock {m}" for m in range(N)], rotation=20)
ax[0].set_ylabel("matched shift  (off − on)/σ_on")
ax[0].set_title("Per-mock cosmology shift when the HCD prior is removed\n"
                "(5/6 mocks: n_s↑, A_p↓; mock 2 flips — its data prefers low LLS)", fontsize=10)
ax[0].legend(fontsize=9)
# right: the systematic (mean ± SEM) for the 3 params
labs = ["n_s", "A_p", r"$\alpha_{\rm LLS}$"]; keys = ["ns", "Ap", "alpha_lls"]
mu = [sh[k].mean() for k in keys]; sem = [sh[k].std(ddof=1)/np.sqrt(N) for k in keys]
cols = ["#2166ac", "#b2182b", "#4d4d4d"]
ax[1].bar(range(3), mu, yerr=sem, color=cols, alpha=0.85, capsize=5)
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labs)
ax[1].set_ylabel("matched shift  (off − on)/σ_on")
ax[1].set_title(f"Systematic (N={N} matched pairs)\n"
                f"n_s {mu[0]:+.2f}σ · A_p {mu[1]:+.2f}σ · α_LLS {mu[2]:+.1f}σ", fontsize=10)
for i, (m, s) in enumerate(zip(mu, sem)):
    ax[1].text(i, m + (0.1 if m >= 0 else -0.1)*np.sign(m), f"{m:+.2f}", ha="center",
               va="bottom" if m >= 0 else "top", fontsize=9)
fig.suptitle("DESI HCD-prior — matched-pair cosmology shift (identical data, prior off vs on)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = f"{OUT}/desi_hcd_prior_matched_shift.png"; fig.savefig(p, dpi=130); plt.close(fig)
print(f"matched-pair systematic: n_s {mu[0]:+.2f}±{sem[0]:.2f}σ | A_p {mu[1]:+.2f}±{sem[1]:.2f}σ | "
      f"alpha_LLS {mu[2]:+.1f}±{sem[2]:.1f}σ  (n_s>0: {(sh['ns']>0).sum()}/6, A_p<0: {(sh['Ap']<0).sum()}/6)")
print("wrote", p)
