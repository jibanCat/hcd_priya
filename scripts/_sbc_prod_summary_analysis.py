#!/usr/bin/env python3
"""Post-process the production-ensemble Leg-A SBC run (48 mocks).
Pure post-processing: read pkls + numpy + matplotlib. No NUTS.
Produces the full SBC diagnostic figure set + npz arrays + a stats JSON
to /home/mfho/hcd_priya_notes/figures/analysis/05_likelihood/ (prefix sbc_).
"""
import pickle, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats as sst

SRC = "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_corrected"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC, "mock_*.pkl")))
assert len(files) == 48, len(files)

# ---- load ----
mocks = []
names = None
for f in files:
    d = pickle.load(open(f, "rb"))
    if names is None:
        names = list(d["names"])
    mocks.append(d)
P = len(names)            # 26
M = len(mocks)            # 48
print(f"loaded {M} mocks, {P} params")

# pretty labels
PRETTY = {
    "ns": r"$n_s$", "Ap": r"$A_p$", "herei": "herei", "heref": "heref",
    "alphaq": r"$\alpha_q$", "hub": r"$h$", "omegamh2": r"$\Omega_m h^2$",
    "hireionz": r"$z_{\rm reion}$", "bhfeedback": "bhfb",
    "alpha_lls": r"$\alpha_{\rm LLS}$", "alpha_subdla": r"$\alpha_{\rm subDLA}$",
    "alpha_dla": r"$\alpha_{\rm DLA}$", "a_SiIII": r"$a_{\rm SiIII}$",
}
for z in range(13):
    PRETTY[f"tau0_z{z}"] = rf"$\tau_0^{{z{z}}}$"
def lab(n): return PRETTY.get(n, n)

TAU0_IDX = [names.index(f"tau0_z{z}") for z in range(13)]
# redshift centers for the 13 tau0 bins (PRIYA/DESI Lya grid z=2.0..4.4 step 0.2)
Z_CENTERS = np.array([2.0 + 0.2*i for i in range(13)])

# ---- per-mock arrays ----
truth = np.array([m["truth_vec"] for m in mocks])              # (M,P)
ndiv = np.array([m["n_div"] for m in mocks])                   # (M,)
Larr = np.array([m["L"] for m in mocks])                       # (M,)
ll_true = np.array([m["ll_true"] for m in mocks])             # (M,)

# draws have variable length per mock -> keep as list
draws_list = [np.array(m["draws"]) for m in mocks]            # each (L_i, P)
ll_draws_list = [np.array(m["ll_draws"]) for m in mocks]     # each (L_i,)

# ---- core SBC quantities ----
# rank of truth among draws (#{draw < truth}), 0..L
rank = np.zeros((M, P))
rank_frac = np.zeros((M, P))   # rank/L in [0,1]
pmean = np.zeros((M, P))
psd = np.zeros((M, P))
pull = np.zeros((M, P))        # (mean - truth)/sd
# central credible coverage flags
cov50 = np.zeros((M, P), bool)
cov68 = np.zeros((M, P), bool)
cov90 = np.zeros((M, P), bool)
# PIT = posterior CDF at truth ~ rank_frac (with tie jitter handled by < )
pit = np.zeros((M, P))

for i in range(M):
    dr = draws_list[i]           # (L,P)
    L = dr.shape[0]
    t = truth[i]                 # (P,)
    r = np.sum(dr < t[None, :], axis=0)   # (P,)
    rank[i] = r
    rank_frac[i] = r / L
    pit[i] = r / L
    pmean[i] = dr.mean(0)
    psd[i] = dr.std(0, ddof=1)
    pull[i] = (pmean[i] - t) / np.where(psd[i] > 0, psd[i], np.nan)
    # central credible intervals
    for frac, flag in ((0.50, cov50), (0.68, cov68), (0.90, cov90)):
        lo = np.percentile(dr, 100*(1-frac)/2, axis=0)
        hi = np.percentile(dr, 100*(1+frac)/2, axis=0)
        flag[i] = (t >= lo) & (t <= hi)

# likelihood rank: rank of ll_true among ll_draws
ll_rank = np.array([np.sum(ll_draws_list[i] < ll_true[i]) for i in range(M)])
ll_rank_frac = ll_rank / Larr

# ---- per-param summary stats ----
mean_pull = pull.mean(0)
sem_pull = pull.std(0, ddof=1) / np.sqrt(M)
std_pull = pull.std(0, ddof=1)
pull_z = mean_pull / sem_pull     # significance of mean-pull from 0

# rank-uniformity test on rank_frac (KS vs U(0,1))
ks_p = np.array([sst.kstest(rank_frac[:, j], "uniform").pvalue for j in range(P)])
ks_D = np.array([sst.kstest(rank_frac[:, j], "uniform").statistic for j in range(P)])

cov50_frac = cov50.mean(0)
cov68_frac = cov68.mean(0)
cov90_frac = cov90.mean(0)

print("\n=== per-param summary ===")
print(f"{'param':14s} {'meanpull':>9s} {'sem':>6s} {'z':>6s} {'stdpull':>8s} {'KSp':>7s} "
      f"{'cov50':>6s} {'cov68':>6s} {'cov90':>6s}")
for j in range(P):
    print(f"{names[j]:14s} {mean_pull[j]:+9.3f} {sem_pull[j]:6.3f} {pull_z[j]:+6.2f} "
          f"{std_pull[j]:8.3f} {ks_p[j]:7.3f} {cov50_frac[j]:6.2f} {cov68_frac[j]:6.2f} {cov90_frac[j]:6.2f}")

# ---- within-posterior correlations (averaged over mocks) ----
# build per-mock within-posterior corr for a focus set
focus = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla", "a_SiIII"]
# add a tau0 z-mean column
fi = [names.index(n) for n in focus]
corr_stack = []
for i in range(M):
    dr = draws_list[i]
    tau0_mean = dr[:, TAU0_IDX].mean(1, keepdims=True)   # (L,1)
    block = np.concatenate([dr[:, fi], tau0_mean], axis=1)  # (L, len(focus)+1)
    c = np.corrcoef(block, rowvar=False)
    corr_stack.append(c)
corr_stack = np.array(corr_stack)             # (M, K, K)
corr_mean = corr_stack.mean(0)
corr_std = corr_stack.std(0)
focus_lbl = [lab(n) for n in focus] + [r"$\bar\tau_0$"]

# key within-posterior correlations vs n_s
ns_pos = focus.index("ns")
def wpcorr(other):
    k = (focus + ["tau0bar"]).index(other) if other != "tau0bar" else len(focus)
    return corr_stack[:, ns_pos, k]
wp_ns_subdla = wpcorr("alpha_subdla")
wp_ns_lls = wpcorr("alpha_lls")
wp_ns_ap = wpcorr("Ap")
wp_ns_dla = wpcorr("alpha_dla")
wp_ns_tau0 = wpcorr("tau0bar")
print("\n=== within-posterior corr(n_s, .) averaged over mocks ===")
print(f"  alpha_subdla: {wp_ns_subdla.mean():+.3f} +/- {wp_ns_subdla.std():.3f}")
print(f"  alpha_lls   : {wp_ns_lls.mean():+.3f} +/- {wp_ns_lls.std():.3f}")
print(f"  alpha_dla   : {wp_ns_dla.mean():+.3f} +/- {wp_ns_dla.std():.3f}")
print(f"  Ap          : {wp_ns_ap.mean():+.3f} +/- {wp_ns_ap.std():.3f}")
print(f"  tau0bar     : {wp_ns_tau0.mean():+.3f} +/- {wp_ns_tau0.std():.3f}")

# across-mock pull correlations
def acorr(a, b):
    return np.corrcoef(pull[:, names.index(a)], pull[:, names.index(b)])[0, 1]
am_ns_subdla = acorr("ns", "alpha_subdla")
am_ns_lls = acorr("ns", "alpha_lls")
am_ns_ap = acorr("ns", "Ap")
am_ns_dla = acorr("ns", "alpha_dla")
print("\n=== across-mock pull corr ===")
print(f"  ns vs alpha_subdla: {am_ns_subdla:+.3f}")
print(f"  ns vs alpha_lls   : {am_ns_lls:+.3f}")
print(f"  ns vs Ap          : {am_ns_ap:+.3f}")
print(f"  ns vs alpha_dla   : {am_ns_dla:+.3f}")

# ---- save master npz ----
np.savez(os.path.join(OUT, "sbc_master_arrays.npz"),
         names=np.array(names), truth=truth, pmean=pmean, psd=psd, pull=pull,
         rank=rank, rank_frac=rank_frac, pit=pit, L=Larr, ndiv=ndiv,
         ll_true=ll_true, ll_rank=ll_rank, ll_rank_frac=ll_rank_frac,
         mean_pull=mean_pull, sem_pull=sem_pull, std_pull=std_pull, pull_z=pull_z,
         ks_p=ks_p, ks_D=ks_D, cov50=cov50_frac, cov68=cov68_frac, cov90=cov90_frac,
         corr_mean=corr_mean, corr_std=corr_std, focus=np.array(focus_lbl),
         z_centers=Z_CENTERS, tau0_idx=np.array(TAU0_IDX),
         wp_ns_subdla=wp_ns_subdla, wp_ns_lls=wp_ns_lls, wp_ns_ap=wp_ns_ap,
         wp_ns_dla=wp_ns_dla, wp_ns_tau0=wp_ns_tau0)

# stats JSON for the doc table
stats = {
    "n_mocks": M, "n_params": P, "total_div": int(ndiv.sum()),
    "L_distribution": {int(k): int((Larr == k).sum()) for k in sorted(set(Larr.tolist()))},
    "names": names,
    "per_param": {names[j]: {
        "mean_pull": float(mean_pull[j]), "sem_pull": float(sem_pull[j]),
        "pull_z": float(pull_z[j]), "std_pull": float(std_pull[j]),
        "ks_p": float(ks_p[j]), "ks_D": float(ks_D[j]),
        "cov50": float(cov50_frac[j]), "cov68": float(cov68_frac[j]),
        "cov90": float(cov90_frac[j])} for j in range(P)},
    "within_post_corr_ns": {
        "alpha_subdla": [float(wp_ns_subdla.mean()), float(wp_ns_subdla.std())],
        "alpha_lls": [float(wp_ns_lls.mean()), float(wp_ns_lls.std())],
        "alpha_dla": [float(wp_ns_dla.mean()), float(wp_ns_dla.std())],
        "Ap": [float(wp_ns_ap.mean()), float(wp_ns_ap.std())],
        "tau0bar": [float(wp_ns_tau0.mean()), float(wp_ns_tau0.std())]},
    "across_mock_pull_corr": {
        "ns_alpha_subdla": float(am_ns_subdla), "ns_alpha_lls": float(am_ns_lls),
        "ns_Ap": float(am_ns_ap), "ns_alpha_dla": float(am_ns_dla)},
    "ll_rank_ks_p": float(sst.kstest(ll_rank_frac, "uniform").pvalue),
}
json.dump(stats, open(os.path.join(OUT, "sbc_stats.json"), "w"), indent=2)
print("\nwrote sbc_stats.json + sbc_master_arrays.npz")

# =====================================================================
# FIGURES
# =====================================================================
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

def uniform_band(L_eff, nbins, conf=0.99):
    """Binomial band on a uniform-rank histogram with M draws into nbins bins."""
    expected = M / nbins
    lo = sst.binom.ppf((1-conf)/2, M, 1/nbins)
    hi = sst.binom.ppf((1+conf)/2, M, 1/nbins)
    return expected, lo, hi

# --- FIG 1: rank histograms, all 26 params ---
NB = 10
fig, axes = plt.subplots(6, 5, figsize=(16, 16))
axes = axes.ravel()
exp, lo, hi = uniform_band(M, NB)
flagged = []
for j in range(P):
    ax = axes[j]
    ax.hist(rank_frac[:, j], bins=np.linspace(0, 1, NB+1), color="#4477AA",
            edgecolor="white", alpha=0.85)
    ax.axhspan(lo, hi, color="grey", alpha=0.25, zorder=0)
    ax.axhline(exp, color="k", lw=0.8, ls="--")
    bad = ks_p[j] < 0.05
    if bad:
        flagged.append(names[j])
    title_c = "crimson" if bad else "black"
    ax.set_title(f"{lab(names[j])}  KSp={ks_p[j]:.2f}", color=title_c, fontsize=9)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
for j in range(P, len(axes)):
    axes[j].axis("off")
fig.suptitle(f"SBC rank-of-truth histograms (rank/L), all 26 params  |  N={M} mocks, "
             f"grey=99% binomial band  |  RED title = KS p<0.05", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(os.path.join(OUT, "sbc_rank_histograms_all.png"), bbox_inches="tight")
plt.close(fig)
print("FIG1 sbc_rank_histograms_all.png  flagged(KSp<0.05):", flagged)

# --- FIG 2: PIT/ECDF-difference per param with simultaneous bands ---
def ecdf_band(M, conf=0.95):
    # simultaneous band approx via DKW-style sqrt(ln(2/alpha)/2M)
    alpha = 1 - conf
    eps = np.sqrt(np.log(2/alpha) / (2*M))
    return eps
eps = ecdf_band(M, 0.95)
fig, axes = plt.subplots(6, 5, figsize=(16, 16))
axes = axes.ravel()
xx = np.linspace(0, 1, 200)
ecdf_breach = []
for j in range(P):
    ax = axes[j]
    s = np.sort(pit[:, j])
    ecdf = np.searchsorted(s, xx, side="right") / M
    diff = ecdf - xx
    ax.plot(xx, diff, color="#AA3377", lw=1.4)
    ax.fill_between(xx, -eps, eps, color="grey", alpha=0.22)
    ax.axhline(0, color="k", lw=0.6)
    breach = np.any(np.abs(diff) > eps)
    if breach:
        ecdf_breach.append(names[j])
    ax.set_title(f"{lab(names[j])}", color="crimson" if breach else "black", fontsize=9)
    ax.set_ylim(-0.35, 0.35); ax.set_xlim(0, 1)
    ax.tick_params(labelsize=7)
for j in range(P, len(axes)):
    axes[j].axis("off")
fig.suptitle(f"SBC PIT ECDF-difference (ECDF(PIT) - uniform), all 26 params  |  "
             f"grey = 95% DKW simultaneous band  |  RED = breach", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(os.path.join(OUT, "sbc_pit_ecdf_all.png"), bbox_inches="tight")
plt.close(fig)
print("FIG2 sbc_pit_ecdf_all.png  breach:", ecdf_breach)

# --- FIG 3: pull summary bar (mean +/- SEM) + per-param pull violin-ish ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11),
                               gridspec_kw={"height_ratios": [1, 1.2]})
order = np.arange(P)
cols = ["crimson" if abs(pull_z[j]) > 2 else "#4477AA" for j in range(P)]
ax1.bar(order, mean_pull, yerr=sem_pull, color=cols, edgecolor="k", lw=0.4, capsize=2)
ax1.axhline(0, color="k", lw=0.8)
ax1.axhspan(-0.3, 0.3, color="green", alpha=0.08)
ax1.set_xticks(order); ax1.set_xticklabels([lab(n) for n in names], rotation=60, ha="right", fontsize=8)
ax1.set_ylabel(r"mean pull $\pm$ SEM  [$(\hat\mu-\theta)/\hat\sigma$]")
ax1.set_title(f"Per-param mean pull across {M} mocks (RED = |mean|/SEM > 2; green band = $\\pm0.3\\sigma$)")
# annotate flagged
for j in range(P):
    if abs(pull_z[j]) > 2:
        ax1.annotate(f"{pull_z[j]:+.1f}", (j, mean_pull[j]),
                     textcoords="offset points", xytext=(0, 8 if mean_pull[j]<0 else -12),
                     ha="center", fontsize=7, color="crimson")
# bottom: pull-std per param (over-dispersion)
ax2.bar(order, std_pull, color=["crimson" if std_pull[j] > 1.15 else "#66CCAA" for j in range(P)],
        edgecolor="k", lw=0.4)
ax2.axhline(1.0, color="k", lw=0.9, ls="--", label="ideal std=1")
ax2.axhspan(0.85, 1.15, color="grey", alpha=0.15)
ax2.set_xticks(order); ax2.set_xticklabels([lab(n) for n in names], rotation=60, ha="right", fontsize=8)
ax2.set_ylabel("pull std (dispersion)")
ax2.set_title("Per-param pull std (RED > 1.15 = over-dispersed / posterior too narrow)")
ax2.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sbc_pull_summary_bar.png"), bbox_inches="tight")
plt.close(fig)
print("FIG3 sbc_pull_summary_bar.png")

# --- FIG 4: coverage curve ---
fig, ax = plt.subplots(figsize=(15, 7))
x = np.arange(P)
w = 0.26
for off, frac, fc, lbl_ in ((-w, cov50_frac, "#4477AA", "50%"),
                            (0, cov68_frac, "#EE6677", "68%"),
                            (w, cov90_frac, "#228833", "90%")):
    ax.bar(x+off, frac, width=w, color=fc, edgecolor="k", lw=0.3, label=f"central {lbl_} CI")
for nom, fc in ((0.50, "#4477AA"), (0.68, "#EE6677"), (0.90, "#228833")):
    ax.axhline(nom, color=fc, lw=1.1, ls="--")
ax.set_xticks(x); ax.set_xticklabels([lab(n) for n in names], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("fraction of mocks covering truth")
ax.set_ylim(0, 1.05)
ax.set_title(f"Calibration coverage: fraction of {M} mocks with truth inside the central "
             f"50/68/90% CI (dashed = nominal)")
ax.legend(loc="lower left", ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sbc_coverage_curve.png"), bbox_inches="tight")
plt.close(fig)
print("FIG4 sbc_coverage_curve.png")

# --- FIG 5: n_s headline pull histogram ---
j_ns = names.index("ns")
j_ap = names.index("Ap")
fig, ax = plt.subplots(figsize=(9, 6.5))
p = pull[:, j_ns]
bins = np.linspace(-4, 4, 21)
ax.hist(p, bins=bins, density=True, color="#EE6677", alpha=0.7, edgecolor="white",
        label=f"$n_s$ pull (N={M})")
xx = np.linspace(-4, 4, 300)
ax.plot(xx, sst.norm.pdf(xx), "k-", lw=1.6, label=r"$\mathcal{N}(0,1)$ (ideal)")
ax.plot(xx, sst.norm.pdf(xx, p.mean(), p.std(ddof=1)), "b--", lw=1.6,
        label=rf"fit $\mathcal{{N}}({p.mean():.2f}, {p.std(ddof=1):.2f}^2)$")
ax.axvline(p.mean(), color="crimson", lw=2,
           label=rf"mean = {p.mean():+.2f}$\sigma$ ({p.mean()/ (p.std(ddof=1)/np.sqrt(M)):+.1f}$\sigma$ from 0)")
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"$n_s$ pull  $(\hat\mu-\theta)/\hat\sigma$")
ax.set_ylabel("density")
ax.set_title(f"$n_s$ SBC headline: biased LOW (mean {p.mean():+.2f}$\\sigma$) "
             f"+ over-dispersed (std {p.std(ddof=1):.2f} → posterior ~{(1-1/p.std(ddof=1))*100:.0f}% too narrow)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sbc_ns_headline_pull.png"), bbox_inches="tight")
plt.close(fig)
print(f"FIG5 sbc_ns_headline_pull.png  ns pull mean={p.mean():+.3f} std={p.std(ddof=1):.3f}")

# --- FIG 6: mechanism panel ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# (a) across-mock pull scatter ns vs subdla
ax = axes[0, 0]
ax.scatter(pull[:, names.index("alpha_subdla")], pull[:, j_ns], c="#AA3377", s=30, alpha=0.8)
ax.set_xlabel(r"$\alpha_{\rm subDLA}$ pull"); ax.set_ylabel(r"$n_s$ pull")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_title(f"across-mock: corr($n_s$,$\\alpha_{{\\rm subDLA}}$) = {am_ns_subdla:+.2f}")
# regression line
sl, ic = np.polyfit(pull[:, names.index("alpha_subdla")], pull[:, j_ns], 1)
xr = np.array([pull[:, names.index("alpha_subdla")].min(), pull[:, names.index("alpha_subdla")].max()])
ax.plot(xr, sl*xr+ic, "k--", lw=1)
# (b) contrast: ns vs lls and ns vs Ap
ax = axes[0, 1]
ax.scatter(pull[:, names.index("alpha_lls")], pull[:, j_ns], c="#4477AA", s=26, alpha=0.7,
           label=rf"$\alpha_{{\rm LLS}}$  (r={am_ns_lls:+.2f})")
ax.scatter(pull[:, j_ap], pull[:, j_ns], c="#228833", s=26, alpha=0.7, marker="s",
           label=rf"$A_p$  (r={am_ns_ap:+.2f})")
ax.set_xlabel("HCD/$A_p$ pull"); ax.set_ylabel(r"$n_s$ pull")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_title("contrast: $n_s$ vs $\\alpha_{\\rm LLS}$ / $A_p$ (weaker)")
ax.legend()
# (c) within-posterior corr distributions
ax = axes[1, 0]
parts = [wp_ns_subdla, wp_ns_lls, wp_ns_dla, wp_ns_ap, wp_ns_tau0]
plbls = [r"$\alpha_{\rm subDLA}$", r"$\alpha_{\rm LLS}$", r"$\alpha_{\rm DLA}$", r"$A_p$", r"$\bar\tau_0$"]
bp = ax.boxplot(parts, labels=plbls, showmeans=True, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#FFD08A"); patch.set_alpha(0.7)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel(r"within-posterior corr($n_s$, $\cdot$)")
ax.set_title("within-posterior corr with $n_s$, per-mock distribution (box=IQR, ▲=mean)")
for i_, pp in enumerate(parts):
    ax.annotate(f"{pp.mean():+.2f}", (i_+1, pp.mean()), textcoords="offset points",
                xytext=(8, 0), fontsize=8)
# (d) correlation matrix (within-posterior, averaged over mocks)
ax = axes[1, 1]
im = ax.imshow(corr_mean, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(focus_lbl))); ax.set_xticklabels(focus_lbl, rotation=45, ha="right")
ax.set_yticks(range(len(focus_lbl))); ax.set_yticklabels(focus_lbl)
for a in range(len(focus_lbl)):
    for b in range(len(focus_lbl)):
        ax.text(b, a, f"{corr_mean[a,b]:+.2f}", ha="center", va="center",
                fontsize=7, color="k" if abs(corr_mean[a,b]) < 0.6 else "white")
ax.set_title("within-posterior corr matrix (mock-averaged)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("MECHANISM: $n_s\\,\\leftrightarrow\\,\\alpha_{\\rm subDLA}$ normalization coupling", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "sbc_mechanism_panel.png"), bbox_inches="tight")
plt.close(fig)
print("FIG6 sbc_mechanism_panel.png")

# --- FIG 7: per-z tau0 calibration ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
# (a) tau0 mean-pull vs z
ax = axes[0, 0]
mp = mean_pull[TAU0_IDX]; sp = sem_pull[TAU0_IDX]
ax.errorbar(Z_CENTERS, mp, yerr=sp, marker="o", color="#4477AA", capsize=3)
ax.axhline(0, color="k", lw=0.8); ax.axhspan(-0.3, 0.3, color="green", alpha=0.08)
ax.set_xlabel("z"); ax.set_ylabel(r"$\tau_0(z)$ mean pull $\pm$ SEM")
ax.set_title("per-z $\\tau_0$ mean pull (load-bearing = low-z)")
for i_, z in enumerate(Z_CENTERS):
    if abs(mp[i_]/sp[i_]) > 2:
        ax.annotate(f"{mp[i_]/sp[i_]:+.1f}", (z, mp[i_]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="crimson")
# (b) tau0 pull std vs z
ax = axes[0, 1]
ax.plot(Z_CENTERS, std_pull[TAU0_IDX], marker="s", color="#EE6677")
ax.axhline(1.0, color="k", lw=0.8, ls="--"); ax.axhspan(0.85, 1.15, color="grey", alpha=0.15)
ax.set_xlabel("z"); ax.set_ylabel(r"$\tau_0(z)$ pull std")
ax.set_title("per-z $\\tau_0$ dispersion")
# (c) tau0 coverage vs z
ax = axes[1, 0]
ax.plot(Z_CENTERS, cov68_frac[TAU0_IDX], "o-", label="68%", color="#EE6677")
ax.plot(Z_CENTERS, cov90_frac[TAU0_IDX], "s-", label="90%", color="#228833")
ax.axhline(0.68, color="#EE6677", ls="--", lw=0.8); ax.axhline(0.90, color="#228833", ls="--", lw=0.8)
ax.set_xlabel("z"); ax.set_ylabel("coverage"); ax.set_ylim(0.4, 1.02)
ax.set_title("per-z $\\tau_0$ coverage vs nominal"); ax.legend()
# (d) tau0 rank-frac KS p-value vs z
ax = axes[1, 1]
ax.bar(Z_CENTERS, ks_p[TAU0_IDX], width=0.15,
       color=["crimson" if ks_p[k] < 0.05 else "#66CCAA" for k in TAU0_IDX], edgecolor="k", lw=0.3)
ax.axhline(0.05, color="k", ls="--", lw=0.8, label="p=0.05")
ax.set_xlabel("z"); ax.set_ylabel("rank-uniformity KS p"); ax.set_title("per-z $\\tau_0$ rank-uniformity")
ax.legend()
fig.suptitle(r"Per-$z$ $\tau_0$ SBC calibration (13 bins, z=2.0..4.4)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "sbc_tau0_per_z.png"), bbox_inches="tight")
plt.close(fig)
print("FIG7 sbc_tau0_per_z.png")

# --- FIG 8: likelihood-rank SBC ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
NBll = 10
ax1.hist(ll_rank_frac, bins=np.linspace(0, 1, NBll+1), color="#332288",
         edgecolor="white", alpha=0.85)
exp, lo, hi = uniform_band(M, NBll)
ax1.axhspan(lo, hi, color="grey", alpha=0.25)
ax1.axhline(exp, color="k", ls="--", lw=0.8)
ll_ks = sst.kstest(ll_rank_frac, "uniform").pvalue
ax1.set_xlabel(r"rank of $\ell\ell_{\rm true}$ among $\ell\ell_{\rm draws}$ (/L)")
ax1.set_ylabel("# mocks")
ax1.set_title(f"Likelihood-rank SBC histogram  (KS p={ll_ks:.2f})")
# ecdf
s = np.sort(ll_rank_frac); xx = np.linspace(0, 1, 200)
ecdf = np.searchsorted(s, xx, side="right") / M
ax2.plot(xx, ecdf, color="#332288", lw=1.6, label="empirical")
ax2.plot([0, 1], [0, 1], "k--", lw=1, label="uniform")
ax2.fill_between(xx, xx-eps, xx+eps, color="grey", alpha=0.2)
ax2.set_xlabel("rank/L"); ax2.set_ylabel("ECDF")
ax2.set_title("Likelihood-rank ECDF vs uniform (95% DKW)")
ax2.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sbc_likelihood_rank.png"), bbox_inches="tight")
plt.close(fig)
print(f"FIG8 sbc_likelihood_rank.png  KSp={ll_ks:.3f}")

# --- FIG 9: health (divergences + L + draw counts) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.bar(range(M), ndiv, color="#228833")
ax.set_xlabel("mock index"); ax.set_ylabel("# divergences")
ax.set_title(f"Divergences per mock (total = {int(ndiv.sum())})")
ax.set_ylim(-0.5, max(1, ndiv.max()+0.5))
ax = axes[1]
ax.bar(range(M), Larr, color="#4477AA")
ax.set_xlabel("mock index"); ax.set_ylabel("L (# stored draws)")
ax.set_title(f"Stored posterior draws per mock (median {int(np.median(Larr))})")
ax = axes[2]
ax.hist(Larr, bins=[60, 90, 110, 130, 175, 250, 320], color="#AA3377", edgecolor="white")
ax.set_xlabel("L"); ax.set_ylabel("# mocks"); ax.set_title("L distribution")
fig.suptitle("Run health: 0 divergences across all 48 mocks; variable stored-draw count", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "sbc_health.png"), bbox_inches="tight")
plt.close(fig)
print("FIG9 sbc_health.png")

# --- FIG 10: A_p vs n_s contrast panel ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
# rank hists
for ax, j, nm in ((axes[0,0], j_ns, "ns"), (axes[0,1], j_ap, "Ap")):
    ax.hist(rank_frac[:, j], bins=np.linspace(0, 1, 11), color="#EE6677" if nm=="ns" else "#228833",
            edgecolor="white", alpha=0.85)
    exp, lo, hi = uniform_band(M, 10)
    ax.axhspan(lo, hi, color="grey", alpha=0.25); ax.axhline(exp, color="k", ls="--", lw=0.8)
    ax.set_title(f"{lab(nm)} rank-of-truth  (KS p={ks_p[j]:.2f})")
    ax.set_xlabel("rank/L")
# pull hists
for ax, j, nm in ((axes[1,0], j_ns, "ns"), (axes[1,1], j_ap, "Ap")):
    p = pull[:, j]
    ax.hist(p, bins=np.linspace(-4, 4, 17), density=True,
            color="#EE6677" if nm=="ns" else "#228833", alpha=0.7, edgecolor="white")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, sst.norm.pdf(xx), "k-", lw=1.4)
    ax.axvline(p.mean(), color="crimson", lw=1.6)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_title(f"{lab(nm)} pull: mean {p.mean():+.2f}$\\sigma$, std {p.std(ddof=1):.2f}")
    ax.set_xlabel("pull")
fig.suptitle("$A_p$ is well-calibrated, $n_s$ is not — side-by-side", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "sbc_ap_vs_ns_contrast.png"), bbox_inches="tight")
plt.close(fig)
print("FIG10 sbc_ap_vs_ns_contrast.png")

# --- FIG 11 (bonus): pull vs truth (prior-edge effects) for key params ---
key = ["ns", "Ap", "alpha_subdla", "alpha_lls", "alpha_dla", "a_SiIII"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, nm in zip(axes.ravel(), key):
    j = names.index(nm)
    ax.scatter(truth[:, j], pull[:, j], s=26, c="#4477AA", alpha=0.75)
    ax.axhline(0, color="k", lw=0.6)
    # linear trend
    sl, ic = np.polyfit(truth[:, j], pull[:, j], 1)
    xr = np.array([truth[:, j].min(), truth[:, j].max()])
    rr = np.corrcoef(truth[:, j], pull[:, j])[0, 1]
    ax.plot(xr, sl*xr+ic, "r--", lw=1, label=f"r={rr:+.2f}")
    ax.set_xlabel(f"{lab(nm)} truth"); ax.set_ylabel("pull")
    ax.set_title(lab(nm)); ax.legend(fontsize=8)
fig.suptitle("Pull vs truth — prior-edge / trend check (r = corr; flat ≈ no edge effect)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "sbc_pull_vs_truth.png"), bbox_inches="tight")
plt.close(fig)
print("FIG11 sbc_pull_vs_truth.png")

# --- FIG 12 (bonus): worst-offending mocks (sum of |pull| over cosmology+HCD) ---
focus_bad = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"]
fbi = [names.index(n) for n in focus_bad]
absp = np.abs(pull[:, fbi])
score = absp.sum(1)
ord_worst = np.argsort(score)[::-1]
fig, ax = plt.subplots(figsize=(15, 6))
im = ax.imshow(pull[ord_worst][:, fbi].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_yticks(range(len(focus_bad))); ax.set_yticklabels([lab(n) for n in focus_bad])
ax.set_xticks(range(M)); ax.set_xticklabels([f"{i:02d}" for i in ord_worst], rotation=90, fontsize=6)
ax.set_xlabel("mock index (sorted by total |pull|, worst→best)")
ax.set_title("Per-mock pull heatmap (cosmology + HCD), worst offenders left")
fig.colorbar(im, ax=ax, fraction=0.02, label="pull")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sbc_pull_heatmap_mocks.png"), bbox_inches="tight")
plt.close(fig)
print("FIG12 sbc_pull_heatmap_mocks.png  worst 5 mocks:", ord_worst[:5].tolist())

print("\nALL FIGURES DONE")
print("flagged rank-nonuniform (KSp<0.05):", flagged)
print("ECDF breach:", ecdf_breach)
