#!/usr/bin/env python3
"""Phase-2b emulator data/preprocessing diagnostics (from REAL cache shards).

Produces five PNGs under figures/analysis/04_emulator/ that document the
emulator's input parameter distributions, tau0 coverage, output channel
transforms, per-class structure / sample-variance weights, and the
Nyquist mask coverage. See the Phase-2b spec (sec.4, 7, 9, 10).

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_emulator_data_diagnostics.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hcd_analysis.emulator.data import load_cache, signed_log, COARSE_NAMES

SHARD_GLOB = "/scratch/cavestru_root/cavestru0/mfho/tau0_shards/observables_tau0_lf.shard*.h5"
N_SHARDS = 10
OUTDIR = "/home/mfho/hcd_priya/figures/analysis/04_emulator"

PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback"]
DELTA_NAMES = ("LLS", "subDLA", "DLA")  # delta = (Pu-Pf)[:,1:]  -> classes 1..3


def load_many(n=N_SHARDS):
    files = sorted(glob.glob(SHARD_GLOB))[:n]
    assert files, f"no shards at {SHARD_GLOB}"
    dicts = [load_cache(f) for f in files]
    keys_cat = ["params", "kfkms", "P_tier_p", "target_F", "scale", "z_grid",
                "z_meta", "alpha_idx", "P_filt", "delta", "coarse_counts",
                "tau0", "mask", "w_c_cache", "inv_nc"]
    out = {}
    for k in keys_cat:
        out[k] = np.concatenate([d[k] for d in dicts], axis=0)
    print(f"loaded {len(files)} shards -> {out['params'].shape[0]} rows")
    return out, files


# ---------------------------------------------------------------------------
def fig1_param_hist(d):
    P = d["params"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    for i in range(9):
        ax = axes[i]
        x = P[:, i]
        ax.hist(x, bins=40, color="C0", alpha=0.8)
        lo, hi = x.min(), x.max()
        ax.set_title(f"{PARAM_NAMES[i]}\n[{lo:.4g}, {hi:.4g}]", fontsize=10)
        ax.set_xlabel(PARAM_NAMES[i]); ax.set_ylabel("rows")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
    # 10th panel: z_grid
    ax = axes[9]
    ax.hist(d["z_grid"], bins=np.arange(3.1, 5.6, 0.2), color="C3", alpha=0.8)
    ax.set_title("z_grid (redshift coord)"); ax.set_xlabel("z"); ax.set_ylabel("rows")
    spread = "; ".join(f"{n}:[{P[:,i].min():.3g},{P[:,i].max():.3g}]"
                       for i, n in enumerate(PARAM_NAMES))
    fig.suptitle(
        "Fig 1 - Raw input parameter distributions (9 params + z)\n"
        "Params span VERY different scales (Ap~1e-9 vs herei~4 vs ns~0.8) "
        "=> input normalization (fit_norm) is REQUIRED",
        fontsize=13)
    fig.text(0.5, 0.005, spread, ha="center", fontsize=7, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, "data_params_hist.png")


def fig2_tau0_coverage(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    tau0, z, ai = d["tau0"], d["z_grid"], d["alpha_idx"]
    sc = a1.scatter(z + np.random.uniform(-0.04, 0.04, z.size), tau0,
                    c=ai, cmap="viridis", s=8, alpha=0.6)
    a1.set_xlabel("z_grid"); a1.set_ylabel(r"$\tau_0=-\ln\,F$")
    a1.set_title("tau0 vs z, colored by alpha_idx")
    fig.colorbar(sc, ax=a1, label="alpha_idx")
    # trace a few alphas across z to show same alpha -> different tau0 at each z
    for a in [0, 5, 10, 15, 19]:
        m = ai == a
        zs = np.unique(z[m])
        med = [np.median(tau0[m & (z == zz)]) for zz in zs]
        a2.plot(zs, med, "-o", ms=4, label=f"alpha={a}")
    a2.set_xlabel("z_grid"); a2.set_ylabel(r"median $\tau_0$")
    a2.set_title("same alpha_idx maps to different tau0 across z\n"
                 "(=> hold out tau0-edges, not alpha-edges; spec sec.7)")
    a2.legend(fontsize=8)
    fig.suptitle("Fig 2 - tau0 (target optical depth) coverage", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "data_tau0_coverage.png")


def fig3_output_transforms(d):
    rng = np.random.default_rng(0)
    R = d["P_filt"].shape[0]
    rows = rng.choice(R, size=min(60, R), replace=False)
    k = d["kfkms"][rows].T  # (K, nrows) per-row k
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: P_filt for 4 classes, linear vs log; plus log-log
    Pf = d["P_filt"][rows]  # (n,4,K)
    a = axes[0, 0]
    for ci, nm in enumerate(COARSE_NAMES):
        a.plot(d["kfkms"][rows[0]], Pf[0, ci], label=nm)
    a.set_title("P_filt (one row), 4 classes - LINEAR")
    a.set_xlabel("k [s/km]"); a.set_ylabel("P_filt"); a.legend(fontsize=8)

    a = axes[0, 1]
    for ci, nm in enumerate(COARSE_NAMES):
        a.semilogy(d["kfkms"][rows[0]], np.maximum(Pf[0, ci], 1e-30), label=nm)
    a.set_title("P_filt (one row), 4 classes - LOG-y\n(dynamic range ~3 decades => safe_log channel)")
    a.set_xlabel("k [s/km]"); a.set_ylabel("log P_filt"); a.legend(fontsize=8)

    a = axes[0, 2]
    for ci, nm in enumerate(COARSE_NAMES):
        med = np.nanmedian(Pf[:, ci], axis=0)
        a.loglog(d["kfkms"][rows[0]], np.maximum(med, 1e-30), label=nm)
    a.set_title("median P_filt per class - LOG-LOG")
    a.set_xlabel("k [s/km]"); a.set_ylabel("P_filt"); a.legend(fontsize=8)

    # Bottom row: delta linear vs arcsinh; emphasise DLA sign flip at low k
    Dl = d["delta"][rows]  # (n,3,K)
    a = axes[1, 0]
    for ci, nm in enumerate(DELTA_NAMES):
        a.plot(d["kfkms"][rows[0]], Dl[0, ci], label=nm)
    a.axhline(0, color="k", lw=0.6)
    a.set_title(r"$\Delta=P_{unfilt}-P_{filt}$ (one row) - LINEAR")
    a.set_xlabel("k [s/km]"); a.set_ylabel(r"$\Delta$"); a.legend(fontsize=8)

    a = axes[1, 1]
    for ci, nm in enumerate(DELTA_NAMES):
        a.plot(d["kfkms"][rows[0]], signed_log(Dl[0, ci]), label=nm)
    a.axhline(0, color="k", lw=0.6)
    a.set_title(r"$\mathrm{arcsinh}(\Delta)$ (one row) - smooth through 0")
    a.set_xlabel("k [s/km]"); a.set_ylabel(r"$\mathrm{arcsinh}\,\Delta$"); a.legend(fontsize=8)

    # show where delta goes negative (per class, vs k) -> motivates arcsinh
    a = axes[1, 2]
    kref = d["kfkms"][rows[0]]
    for ci, nm in enumerate(DELTA_NAMES):
        frac_neg = np.nanmean(d["delta"][:, ci, :] < 0, axis=0)
        a.plot(kref, frac_neg, label=nm)
    a.set_xscale("log")
    a.set_title("fraction of rows with Delta<0 vs k\n"
                "DLA flips sign at low k => signed (arcsinh) transform needed")
    a.set_xlabel("k [s/km]"); a.set_ylabel("frac rows Delta<0"); a.legend(fontsize=8)

    fig.suptitle("Fig 3 - Output channel transforms (P_filt: log; Delta: arcsinh)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "data_output_transforms.png")


def fig4_class_counts_weights(d):
    cc = d["coarse_counts"]   # (R,4)
    inv = d["inv_nc"]         # (R,4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    a = axes[0, 0]
    for ci, nm in enumerate(COARSE_NAMES):
        x = cc[:, ci]
        x = x[x > 0]
        a.hist(np.log10(np.maximum(x, 1)), bins=40, alpha=0.5, label=nm)
    a.set_xlabel("log10(coarse_counts)"); a.set_ylabel("rows")
    a.set_title("Per-class absorber counts (n_c)\nDLA is rarest by ~1 decade")
    a.legend(fontsize=9)

    a = axes[0, 1]
    for ci, nm in enumerate(COARSE_NAMES):
        x = inv[:, ci]
        x = x[x > 0]
        a.hist(np.log10(x), bins=40, alpha=0.5, label=nm)
    a.set_yscale("log")
    a.set_xlabel("log10(inv_nc = 1/n_c)"); a.set_ylabel("rows (log)")
    a.set_title("Sample-variance weights 1/n_c\nDLA gets LARGEST weight (rarest)")
    a.legend(fontsize=9)

    a = axes[1, 0]
    med = [np.median(cc[:, ci]) for ci in range(4)]
    a.bar(COARSE_NAMES, med, color=["C0", "C1", "C2", "C3"])
    a.set_yscale("log")
    a.set_ylabel("median n_c (log)")
    a.set_title("Median count per class\n(DLA rarity motivates 1/n_c down-weighting)")
    for i, v in enumerate(med):
        a.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    a = axes[1, 1]
    empty_frac = (cc == 0).mean(0)
    a.bar(COARSE_NAMES, empty_frac, color=["C0", "C1", "C2", "C3"])
    a.set_ylabel("fraction of rows with class EMPTY")
    a.set_title("Fraction of rows where class has zero counts")
    for i, v in enumerate(empty_frac):
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Fig 4 - Per-class structure, counts, and 1/n_c weights", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "data_class_counts_weights.png")


def fig5_nyquist_mask(d):
    mask = d["mask"]               # (R,K) finite(P_tier_p)
    # per-class finite (P_filt) -> the bins that genuinely survive native Nyquist
    pf_fin = np.isfinite(d["P_filt"]).all(1)  # (R,K)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    a = axes[0]
    nanfrac = 1.0 - mask.mean(0)
    a.plot(np.arange(mask.shape[1]), nanfrac, label="P_tier_p")
    a.plot(np.arange(pf_fin.shape[1]), 1.0 - pf_fin.mean(0), label="P_filt (all classes)")
    a.set_xlabel("k-index"); a.set_ylabel("fraction NaN (masked)")
    a.set_title("Masked-bin fraction vs k-index")
    a.legend(fontsize=9)

    a = axes[1]
    nyq = np.array([int(np.where(r)[0].max()) + 1 if r.any() else 0 for r in mask])
    a.hist(nyq, bins=np.arange(nyq.min() - 1, mask.shape[1] + 2), color="C2")
    a.set_xlabel("native Nyquist cutoff (last finite k-index + 1)")
    a.set_ylabel("rows")
    a.set_title(f"Distribution of native Nyquist cutoffs\n"
                f"(K={mask.shape[1]} bins; all-finite={np.mean(nyq==mask.shape[1]):.2%} of rows)")

    a = axes[2]
    nfin = mask.sum(1)
    a.hist(nfin, bins=40, color="C4")
    a.set_xlabel("# finite (usable) k-bins per row")
    a.set_ylabel("rows")
    a.set_title("Usable-bin count per row")

    fig.suptitle("Fig 5 - Mask / native-Nyquist coverage", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "data_nyquist_mask.png")


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    sz = os.path.getsize(p)
    print(f"wrote {p}  ({sz/1024:.0f} KB)  {'OK' if sz > 20480 else 'TOO SMALL'}")


def main():
    d, _ = load_many()
    fig1_param_hist(d)
    fig2_tau0_coverage(d)
    fig3_output_transforms(d)
    fig4_class_counts_weights(d)
    fig5_nyquist_mask(d)


if __name__ == "__main__":
    main()
