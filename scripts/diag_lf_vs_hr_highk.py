"""PI multi-fidelity hypothesis — DIRECT forward LF-vs-HR cached P1D comparison.

QUESTION (PI): is the low-fidelity (LF, 1536**3) clean-forest P1D numerically
unreliable at high k (k>0.04 s/km), especially at low z, relative to the
high-fidelity (HR, 3072**3) sims — in a way that matches the emulator residual
driving the -0.65 sigma n_s bias (localized to the small-scale / KS leg at low z)?

This is a FORWARD comparison of the two CACHED per-class P1D vectors. NO training,
no emulator. We measure the RAW LF->HR resolution discrepancy of the clean forest
P1D directly from the caches.

Clean-forest P1D used by the production forward model is P_filt[0] (the CLEAN
coarse class of P_tier_c_filtered, count-weighted collapse of the first fine bin;
see hcd_analysis/emulator/data.py COARSE_SLICES). We use exactly that here.

Matching: HR's 6 design cosmologies are EXACT LF design points. We match
HR<->LF rows on (params_unit rounded, z rounded, alpha_idx) — IDENTICAL to
hcd_analysis.emulator.multifidelity.match_hr_to_lf. The alpha_idx axis is the
tau0/mean-flux rung; alpha_slope is bit-identical per idx across the two caches
(verified), so the mean-flux match is EXACT (no closest-rung approximation).

delta(z,k) = (P_LF - P_HR)/P_HR  for the clean forest P1D, at matched (sim,z,alpha).
HR is interpolated (log-log) onto the LF native k-grid so both live on the SAME
k axis; HR k-range (~0.2) strictly contains the LF range (~0.069), so this is
INTERPOLATION not extrapolation everywhere we report.

Outputs:
  figures/analysis/04_emulator/lf_vs_hr_highk.png
  figures/analysis/04_emulator/lf_vs_hr_highk.txt

Env (MANDATORY): h5py + numpy suffice (pure cache read, no jax/emulator import).
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_lf_vs_hr_highk.py
"""
from __future__ import annotations

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"

# clean coarse-class collapse: first fine bin is the clean forest (data.py COARSE_SLICES[0]).
CLEAN_SLICE = slice(0, 1)

# PRIYA design box (data.py PARAM_LIMITS) for the unit-cube match key.
PARAM_LIMITS = np.array([
    [0.8, 1.05], [1.2e-9, 2.6e-9], [3.5, 4.5], [2.2, 3.2], [1.3, 3.0],
    [0.65, 0.75], [0.14, 0.146], [6.5, 8.0], [0.03, 0.07]], dtype=np.float64)

K_HI = 0.04           # s/km — the high-k threshold the PI flagged
K_LO = 0.02           # s/km — low-k band upper edge
K_CMP_MAX = 0.069     # compare up to the LF Nyquist (shared band)
LOWZ = (2.0, 2.4)     # low-z band
HIZ = (3.2, 3.6)      # high-z anchor band (z=3.4 +/- 0.2)


def _norm(p):
    return (p - PARAM_LIMITS[:, 0]) / (PARAM_LIMITS[:, 1] - PARAM_LIMITS[:, 0])


def load(path):
    with h5py.File(path, "r") as f:
        Pf15 = f["P_tier_c_filtered"][:]            # (R,15,K)
        cnt15 = f["tier_c_counts"][:]               # (R,15)
        d = dict(
            kfkms=f["kfkms"][:],
            z=f["z_grid"][:],
            aidx=f["alpha_idx"][:].astype(int),
            aslope=f["alpha_slope"][:],
            target_F=f["target_F"][:],
            params=f["params"][:],
            sim=np.array([s.decode() if isinstance(s, bytes) else s
                          for s in f["sim_name"][:]]),
        )
    # clean-class count-weighted collapse over the clean fine bins (here just bin 0):
    c = cnt15[:, CLEAN_SLICE].astype(float)
    w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
    seg = Pf15[:, CLEAN_SLICE, :]
    contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
    P_clean = contrib.sum(1)                        # (R,K)
    allnan = (~np.isfinite(seg)).all(1)
    P_clean[allnan] = np.nan
    d["P_clean"] = P_clean
    return d


def rowkey(d):
    pu = np.round(_norm(d["params"]), 6)
    z = np.round(d["z"], 4)
    return [(tuple(pu[i]), z[i], int(d["aidx"][i])) for i in range(len(z))]


def match(lf, hr):
    idx = {k: i for i, k in enumerate(rowkey(lf))}
    return [(h, idx[k]) for h, k in enumerate(rowkey(hr)) if k in idx]


def main():
    lf = load(LF_CACHE)
    hr = load(HR_CACHE)
    pairs = match(lf, hr)                            # [(hr_row, lf_row), ...]
    print(f"matched rows (hr_row, lf_row): {len(pairs)}")

    # the LF native k-grid we report on: take a single canonical LF k-grid per z
    # (varies microscopically per row via dv); we just use each pair's own LF k.
    rows = []
    overlap_sims = set()
    for hrow, lrow in pairs:
        klf = lf["kfkms"][lrow]
        Plf = lf["P_clean"][lrow]
        khr = hr["kfkms"][hrow]
        Phr = hr["P_clean"][hrow]
        z = lf["z"][lrow]
        sim = lf["sim"][lrow]
        # restrict to LF bins that are finite, positive, and within the shared band
        m = (np.isfinite(klf) & (klf > 0) & (klf <= K_CMP_MAX)
             & np.isfinite(Plf) & (Plf > 0))
        if m.sum() < 5:
            continue
        # log-log interp HR onto the LF k-grid (HR range contains LF range -> interp)
        mh = np.isfinite(khr) & (khr > 0) & np.isfinite(Phr) & (Phr > 0)
        Phr_on_lf = np.exp(np.interp(np.log(klf[m]), np.log(khr[mh]),
                                     np.log(Phr[mh]),
                                     left=np.nan, right=np.nan))
        delta = (Plf[m] - Phr_on_lf) / Phr_on_lf      # (P_LF - P_HR)/P_HR
        kk = klf[m]
        overlap_sims.add(sim)
        for kv, dv in zip(kk, delta):
            if np.isfinite(dv):
                rows.append((sim, float(z), float(kv), float(dv)))

    rows = np.array(rows, dtype=object)
    sims = np.array([r[0] for r in rows])
    zz = np.array([r[1] for r in rows], float)
    kk = np.array([r[2] for r in rows], float)
    dd = np.array([r[3] for r in rows], float)
    print(f"overlap sims: {len(overlap_sims)}  finite delta samples: {len(dd)}")

    # ---- band statistics ---------------------------------------------------
    def band(zmask, kmask):
        m = zmask & kmask & np.isfinite(dd)
        if m.sum() == 0:
            return np.nan, np.nan, 0
        return float(np.mean(dd[m])), float(np.std(dd[m])), int(m.sum())

    lowz = (zz >= LOWZ[0]) & (zz <= LOWZ[1])
    hiz = (zz >= HIZ[0]) & (zz <= HIZ[1])
    z22 = np.isclose(zz, 2.2, atol=0.01) | np.isclose(zz, 2.0, atol=0.01)
    z34 = np.isclose(zz, 3.4, atol=0.01)
    klo = kk < K_LO
    khi = kk > K_HI
    allz = np.ones_like(zz, bool)

    # z-band x k-band grid: mean delta (signed) and sign-coherence across sims
    zbins = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    kedges = [0.005, 0.01, 0.02, 0.04, 0.069]

    lines = []
    def emit(s):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("LF vs HR cached clean-forest P1D — direct forward resolution discrepancy")
    emit("delta(z,k) = (P_LF - P_HR)/P_HR   [clean class, matched sim/z/alpha-rung]")
    emit("=" * 78)
    emit(f"HR cache: {HR_CACHE}")
    emit(f"  per-class P1D cache (analogous to LF). n_k_HR=525 (Nyquist ~0.20 s/km).")
    emit(f"LF cache: {LF_CACHE}  (n_k_LF=172, Nyquist ~0.069 s/km)")
    emit(f"overlap sims = {len(overlap_sims)} (all 6 HR design points are exact LF points;")
    emit(f"  alpha/tau0 rung matched EXACTLY — alpha_slope bit-identical per idx).")
    emit(f"matched (sim,z,alpha) rows = {len(pairs)}; finite delta(k) samples = {len(dd)}.")
    emit(f"compare band: k in (LF k_min, {K_CMP_MAX}] s/km; HR log-log interpolated onto LF k.")
    emit("")
    emit("--- (2) low-k vs high-k, low-z vs high-z (signed mean delta, %) ---")
    for nm, zm in (("ALL z", allz), (f"low-z {LOWZ}", lowz),
                   (f"high-z {HIZ}", hiz), ("z=2.0-2.2", z22), ("z=3.4", z34)):
        mlo, slo, nlo = band(zm, klo)
        mhi, shi, nhi = band(zm, khi)
        emit(f"  {nm:16s}: low-k(k<{K_LO}) mean={mlo*100:+7.3f}% (n={nlo:5d})   "
             f"high-k(k>{K_HI}) mean={mhi*100:+7.3f}% (n={nhi:5d})")
    emit("")

    # sign-coherence: fraction of sims with negative mean delta in low-z high-k
    emit("--- (3) sign-coherence across sims (low-z high-k band k>%.2f) ---" % K_HI)
    for nm, zm in ((f"low-z {LOWZ}", lowz), ("z=2.0-2.2", z22),
                   (f"high-z {HIZ}", hiz), ("ALL z", allz)):
        per_sim = []
        for s in sorted(overlap_sims):
            m = (sims == s) & zm & khi & np.isfinite(dd)
            if m.sum() > 0:
                per_sim.append(np.mean(dd[m]))
        per_sim = np.array(per_sim)
        if len(per_sim) == 0:
            emit(f"  {nm:16s}: no samples")
            continue
        nneg = int((per_sim < 0).sum())
        emit(f"  {nm:16s}: per-sim mean delta = "
             f"[{', '.join(f'{v*100:+.2f}' for v in per_sim)}]% ; "
             f"{nneg}/{len(per_sim)} sims negative")
    emit("")

    # k-growth: mean |delta| and signed delta in successive k-bins, low-z vs high-z
    emit("--- (2b) k-growth of delta (signed mean %, low-z vs high-z) ---")
    emit(f"  {'k-band [s/km]':22s} {'low-z mean':>12s} {'high-z mean':>12s}")
    for i in range(len(kedges) - 1):
        ka, kb = kedges[i], kedges[i + 1]
        km = (kk >= ka) & (kk < kb)
        mlo, _, nlo = band(lowz, km)
        mhi, _, nhi = band(hiz, km)
        emit(f"  [{ka:.3f},{kb:.3f})        {mlo*100:+10.3f}%  {mhi*100:+10.3f}%")
    emit("")

    # full z-trend of the high-k deficit (averaged over 6 sims x 20 alpha rungs)
    emit("--- (2c) full z-trend of high-k LF->HR deficit (avg over 6 sims x 20 alpha) ---")
    emit(f"  {'z':>5} {'delta(k>0.04)':>14} {'delta@k~0.069':>14}")
    for zt in sorted(set(np.round(zz, 1))):
        zm = np.isclose(zz, zt, atol=0.05)
        mhi, _, nhi = band(zm, khi)
        # near-Nyquist bin
        knyq = (kk >= 0.06) & (kk <= K_CMP_MAX)
        mny, _, _ = band(zm, knyq)
        if nhi > 0:
            emit(f"  {zt:5.1f} {mhi*100:+13.2f}% {mny*100:+13.2f}%")
    emit("")

    # PRIYA sanity check: ~7% convergence at k=0.07, worst HeII reion z~3-4
    emit("--- sanity vs PRIYA published convergence (~7% @ k=0.07, worst z~3-4) ---")
    khi_edge = (kk >= 0.06) & (kk <= K_CMP_MAX)
    m_all, s_all, n_all = band(allz, khi_edge)
    m_lz, _, _ = band(lowz, khi_edge)
    m_hz, _, _ = band(hiz, khi_edge)
    emit(f"  |delta| near k=0.06-0.069: ALL-z signed mean={m_all*100:+.2f}% "
         f"(rms {s_all*100:.2f}%, n={n_all}); low-z {m_lz*100:+.2f}% ; high-z {m_hz*100:+.2f}%")
    emit("")

    # ---- emulator-residual comparison --------------------------------------
    emit("--- (3/4) comparison to the emulator residual driving the -0.65sigma n_s ---")
    emit("  Emulator coherent held-out residual P_hat/P_true-1 (embias_stats_summary.json):")
    emit("    low-k +0.025% , mid-k +0.013% , high-k -0.023%  (emulator UNDER-predicts at high k).")
    emit("  n_s bias is NEGATIVE -0.65sigma (diag_emu_bias_allfolds.py; 50/60 sims neg).")
    emit("  Sign convention here: delta=(P_LF-P_HR)/P_HR>0 means LF cache has EXCESS power vs HR.")

    # ---- figure ------------------------------------------------------------
    make_fig(rows, sims, zz, kk, dd, overlap_sims, zbins, kedges,
             lowz, hiz, band, klo, khi)

    with open(f"{OUT}/lf_vs_hr_highk.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/lf_vs_hr_highk.txt")


def make_fig(rows, sims, zz, kk, dd, overlap_sims, zbins, kedges,
             lowz, hiz, band, klo, khi):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    zband_specs = [("z = 2.0-2.2 (low-z)", (1.99, 2.21), "C3"),
                   ("z = 2.4-2.8", (2.39, 2.81), "C1"),
                   ("z = 3.2-3.6 (high-z)", (3.19, 3.61), "C0")]

    # top row: delta(k) per z-band, one line per sim + band mean
    for col, (title, (za, zb), c) in enumerate(zband_specs):
        ax = fig.add_subplot(gs[0, col])
        zm = (zz >= za) & (zz <= zb)
        # per-sim mean delta vs k (bin in k)
        kgrid = np.array(kedges)
        kc = np.sqrt(kgrid[:-1] * kgrid[1:])
        for s in sorted(overlap_sims):
            sm = (sims == s) & zm
            prof = []
            for i in range(len(kgrid) - 1):
                km = (kk >= kgrid[i]) & (kk < kgrid[i + 1]) & sm & np.isfinite(dd)
                prof.append(np.mean(dd[km]) * 100 if km.any() else np.nan)
            ax.plot(kc, prof, "-", color="0.6", lw=0.8, alpha=0.7)
        # band mean over all sims
        prof = []
        for i in range(len(kgrid) - 1):
            km = (kk >= kgrid[i]) & (kk < kgrid[i + 1]) & zm & np.isfinite(dd)
            prof.append(np.mean(dd[km]) * 100 if km.any() else np.nan)
        ax.plot(kc, prof, "o-", color=c, lw=2.2, label="band mean")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0.04, color="m", ls="--", lw=1.1, label="k=0.04")
        ax.axhspan(-7, 7, color="0.85", alpha=0.4, zorder=0,
                   label="+/-7% PRIYA conv.")
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km]")
        ax.set_ylabel("delta = (P_LF-P_HR)/P_HR  [%]")
        ax.set_title(title)
        ax.set_ylim(-20, 20)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    # bottom-left: full delta(z,k) heatmap (mean over sims)
    ax = fig.add_subplot(gs[1, 0])
    zb_c = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
    kgrid = np.array([0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.069])
    kc = np.sqrt(kgrid[:-1] * kgrid[1:])
    H = np.full((len(zb_c), len(kc)), np.nan)
    for iz, zv in enumerate(zb_c):
        zm = np.isclose(zz, zv, atol=0.05)
        for ik in range(len(kc)):
            km = (kk >= kgrid[ik]) & (kk < kgrid[ik + 1]) & zm & np.isfinite(dd)
            if km.any():
                H[iz, ik] = np.mean(dd[km]) * 100
    im = ax.pcolormesh(np.arange(len(kc) + 1), np.arange(len(zb_c) + 1),
                       H, cmap="RdBu_r", vmin=-10, vmax=10)
    ax.set_xticks(np.arange(len(kc)) + 0.5)
    ax.set_xticklabels([f"{v:.3f}" for v in kc], rotation=60, fontsize=7)
    ax.set_yticks(np.arange(len(zb_c)) + 0.5)
    ax.set_yticklabels([f"{v:.1f}" for v in zb_c], fontsize=7)
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("z")
    ax.set_title("delta(z,k) mean over sims [%]")
    # mark k=0.04 column
    ik04 = int(np.searchsorted(kgrid, 0.04)) - 0.5
    ax.axvline(ik04 + 0.5, color="m", lw=1.2, ls="--")
    fig.colorbar(im, ax=ax, label="delta [%]")

    # bottom-mid: summary bars low-k/high-k x low-z/high-z
    ax = fig.add_subplot(gs[1, 1])
    cats = [("low-z\nlow-k", lowz, klo), ("low-z\nhigh-k", lowz, khi),
            ("high-z\nlow-k", hiz, klo), ("high-z\nhigh-k", hiz, khi)]
    means = []
    errs = []
    for _, zm, km in cats:
        m, s, n = band(zm, km)
        means.append(m * 100)
        errs.append((s / np.sqrt(max(n, 1))) * 100)
    colors = ["C0", "C3", "C0", "C3"]
    ax.bar(range(4), means, yerr=errs, color=colors, alpha=0.8, capsize=4)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in cats], fontsize=8)
    ax.set_ylabel("signed mean delta [%]")
    ax.set_title("LF-HR discrepancy: band means\n(red=high-k, blue=low-k)")
    ax.grid(alpha=0.3, axis="y")

    # bottom-right: |delta| growth vs k, low-z vs high-z
    ax = fig.add_subplot(gs[1, 2])
    kgrid2 = np.array([0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.069])
    kc2 = np.sqrt(kgrid2[:-1] * kgrid2[1:])
    for nm, zm, c in [("low-z 2.0-2.4", lowz, "C3"), ("high-z 3.2-3.6", hiz, "C0")]:
        prof = []
        for i in range(len(kgrid2) - 1):
            km = (kk >= kgrid2[i]) & (kk < kgrid2[i + 1]) & zm & np.isfinite(dd)
            prof.append(np.mean(dd[km]) * 100 if km.any() else np.nan)
        ax.plot(kc2, prof, "o-", color=c, lw=2, label=nm)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0.04, color="m", ls="--", lw=1.1, label="k=0.04")
    ax.set_xscale("log")
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("signed mean delta [%]")
    ax.set_title("delta growth vs k: low-z vs high-z")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "PI MF hypothesis test: cached LF (1536^3) vs HR (3072^3) clean-forest P1D "
        f"(forward, no training; {len(overlap_sims)} overlap sims)",
        fontsize=12, y=0.98)
    p = f"{OUT}/lf_vs_hr_highk.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
