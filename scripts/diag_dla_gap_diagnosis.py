"""Diagnostic: why does the PRIYA DLA P1D template show almost no low-k boost
in the DESI data band (k>=1e-3) while Rogers+2018 (Illustris-1) predicts ~3x?

Verdict (see module docstring of figure / final report): the boost is REAL and
present (ratio ~2.96 at k_min=1e-3, matching Rogers' 3.1), but it is concentrated
in the lowest 1-2 in-band modes and falls below 1 by k~2e-3, so the band-MEDIAN
ratio looks ~0.5-0.9. This is the Rogers c(z) high-k plateau (c_smallDLA=0.66,
c_largeDLA=0.33 -> genuine sub-1 suppression), NOT a missing boost. The very
lowest mode k0=5e-4 (BELOW k_min) carries a 10x boost from the mean-flux offset
(DLA sightlines are darker: <F>~0.33-0.51 vs global 0.69).

Read-only: reads the committed LF cache; writes ONE figure.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/diag_dla_gap_diagnosis.py
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUT = ROOT / "figures/analysis/05_likelihood/dla_gap_diagnosis.png"
K_MIN = 1e-3
Z_FID = 3.0

# Rogers+2018 Table 2 (verbatim): [LLS, Sub-DLA, Small-DLA, Large-DLA]
RA0 = np.array([2.2001, 1.5083, 1.1415, 0.8633]); RA1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
RB0 = np.array([36.449, 81.388, 162.95, 429.58]); RB1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
RC0 = np.array([0.9849, 0.8667, 0.6572, 0.3339]); RC1 = np.array([-0.0631, 0.0196, 0.1169, 0.4653])
RZ0 = 2.0
COARSE = (slice(0, 1), slice(1, 8), slice(8, 13), slice(13, 15))


def rogers_ratio(k_ang, z, i):
    """Rogers Eq.6 per-class ratio P_i/P_forest at ANGULAR k (s/km), class i."""
    zf = (1.0 + z) / (1.0 + RZ0)
    a = RA0[i] * zf ** RA1[i]; b = RB0[i] * zf ** RB1[i]; c = RC0[i] * zf ** RC1[i]
    return zf ** (-3.55) * (a * np.exp(b * k_ang) - 1.0) ** -2 + c


def collapse(P15, c15, s):
    c = c15[s].astype(float)
    if c.sum() == 0:
        return np.zeros(P15.shape[1])
    w = c / c.sum()
    seg = P15[s]
    return (np.where(np.isfinite(seg), seg, 0.0) * w[:, None]).sum(0)


def main():
    with h5py.File(CACHE, "r") as h:
        zg = h["z_grid"][:]; asl = h["alpha_slope"][:]
        sel = np.where(np.abs(zg - Z_FID) < 0.05)[0]
        r = sel[np.argmin(np.abs(asl[sel] - 1.0))]   # alpha~1 -> observed mean flux
        Pu = h["P_tier_c"][r]; Pf = h["P_tier_c_filtered"][r]
        cnt = h["tier_c_counts"][r]; mf = h["mean_F_by_bin"][r]
        k = h["kfkms"][r]; tF = float(h["target_F"][r]); alpha = float(asl[r])

    g = np.isfinite(k) & (k > 0) & (k <= 0.06)
    k = k[g]; Pu = Pu[:, g]; Pf = Pf[:, g]
    ikmin = int(np.searchsorted(k, K_MIN))

    clean = collapse(Pu, cnt, COARSE[0])             # unfiltered clean baseline
    dla_u = collapse(Pu, cnt, COARSE[3])             # unfiltered coarse DLA
    dla_f = collapse(Pf, cnt, COARSE[3])             # filtered coarse DLA
    core = dla_u - dla_f                             # dla_core add-back
    asused = dla_f + core                            # == dla_u (identity)

    r_used = dla_u / clean                           # AS-USED (== unfilt coarse)
    r_filt = dla_f / clean
    r_t13 = Pu[13] / clean
    r_t14 = Pu[14] / clean
    r_t13f = Pf[13] / clean
    r_t14f = Pf[14] / clean

    rog_small = rogers_ratio(k, Z_FID, 2)
    rog_large = rogers_ratio(k, Z_FID, 3)

    nDLA = int(cnt[13:].sum()); N = int(cnt.sum())
    inb = k >= K_MIN

    # ---- console summary ----
    print(f"row={r} z={Z_FID} alpha={alpha:.3f} target_F={tF:.4f}  nDLA={nDLA}/{N}")
    print(f"counts tier13={cnt[13]} tier14={cnt[14]}")
    print(f"mean_F clean={mf[0]:.3f} t13={mf[13]:.3f} t14={mf[14]:.3f} (global {tF:.3f})")
    print(f"identity max|filt+core - unfilt|/unfilt = {np.nanmax(np.abs(asused/dla_u-1)):.1e}")
    print(f"\nAS-USED DLA/clean:  k0={r_used[0]:.2f}  k_min={r_used[ikmin]:.2f}  "
          f"band-median={np.median(r_used[inb]):.2f}  band-min={r_used[inb].min():.2f}")
    print(f"FILTERED DLA/clean: k0={r_filt[0]:.2f}  k_min={r_filt[ikmin]:.2f}  "
          f"band-median={np.median(r_filt[inb]):.2f}")
    print(f"tier14 unfilt/clean: k0={r_t14[0]:.2f}  k_min={r_t14[ikmin]:.2f}")
    print(f"tier13 unfilt/clean: k0={r_t13[0]:.2f}  k_min={r_t13[ikmin]:.2f}")
    print(f"Rogers small@k_min={rog_small[ikmin]:.2f}  large@k_min={rog_large[ikmin]:.2f}")
    print(f"Rogers small@k0={rog_small[0]:.2f}  large@k0={rog_large[0]:.2f}")

    # ---- figure: 2x2 ----
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.5))

    def vline(a):
        a.axvline(K_MIN, color="0.4", ls="--", lw=1.3)
        a.axhline(1.0, color="k", lw=0.6)

    # (a) FULL-k log-x ratios — the whole story
    a = ax[0, 0]
    a.semilogx(k, r_used, "C3-", lw=2.6, label="coarse DLA, as-used (filt+core ≡ unfilt)")
    a.semilogx(k, r_filt, "C3:", lw=1.6, label="coarse DLA, filtered (cores masked)")
    a.semilogx(k, r_t13, "C0-", lw=1.4, label="tier13 [20.3-21.0] unfilt / clean")
    a.semilogx(k, r_t14, "C1-", lw=1.4, label="tier14 [>=21.0] unfilt / clean")
    a.semilogx(k, rog_small, "C2--", lw=1.8, label="Rogers+2018 small-DLA (Eq.6)")
    a.semilogx(k, rog_large, "C4--", lw=1.8, label="Rogers+2018 large-DLA (Eq.6)")
    vline(a)
    a.text(K_MIN * 1.05, 8.5, "DATA k_min", rotation=90, fontsize=8, color="0.3", va="top")
    a.axvspan(k[0], K_MIN, color="0.92", zorder=0)
    a.text(k[0] * 1.05, 0.15, "below k_min\n(cut from likelihood)", fontsize=7.5, color="0.4")
    a.set_ylim(0, 11)
    a.set_xlim(k[0] * 0.9, 0.06)
    a.set_xlabel("k  [s/km, angular]"); a.set_ylabel("P_class / P_clean")
    a.set_title("(a) FULL-k ratios: boost lives at k<2e-3, mostly BELOW the data k_min")
    a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.25)

    # (b) zoom on the data band only — what the eye sees in-range -> looks ~flat/<1
    a = ax[0, 1]
    kb = k[inb]
    a.semilogx(kb, r_used[inb], "C3-", lw=2.6, label="as-used DLA / clean")
    a.semilogx(kb, rog_small[inb], "C2--", lw=1.8, label="Rogers small-DLA")
    a.semilogx(kb, rog_large[inb], "C4--", lw=1.8, label="Rogers large-DLA")
    a.axhline(1.0, color="k", lw=0.6)
    a.axhline(np.median(r_used[inb]), color="C3", ls=":", lw=1.2,
              label=f"band-median = {np.median(r_used[inb]):.2f}")
    a.set_ylim(0, 3.3)
    a.set_xlabel("k  [s/km, angular]"); a.set_ylabel("P_DLA / P_clean")
    a.set_title("(b) IN-BAND view (k>=k_min): ratio ≈3 only at the first bin,\n"
                "then <1 — the 'no boost' impression is a band-median artefact")
    a.legend(fontsize=8.5, loc="upper right"); a.grid(alpha=0.25)
    a.annotate(f"k_min: {r_used[ikmin]:.2f}", (k[ikmin], r_used[ikmin]),
               xytext=(k[ikmin] * 2.5, 2.7), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="C3"))

    # (c) LINEAR power at low-k — the mean-flux offset deposits the boost
    a = ax[1, 0]
    a.loglog(k, clean, "k-", lw=2, label=f"P_clean  (<F>={mf[0]:.2f})")
    a.loglog(k, dla_u, "C3-", lw=2, label=f"P_DLA unfilt (as-used)")
    a.loglog(k, Pu[13], "C0-", lw=1.3, label=f"tier13 (<F>={mf[13]:.2f})")
    a.loglog(k, Pu[14], "C1-", lw=1.3, label=f"tier14 (<F>={mf[14]:.2f})")
    a.loglog(k, dla_f, "C3:", lw=1.4, label="P_DLA filtered")
    a.axvline(K_MIN, color="0.4", ls="--", lw=1.3)
    a.set_xlabel("k  [s/km, angular]"); a.set_ylabel("P1D(k)  [km/s]")
    a.set_title("(c) LINEAR power: DLA low-k EXCESS is real & large\n"
                "(darker sightlines -> mean-flux offset dumps power at k->0)")
    a.legend(fontsize=8.5, loc="lower left"); a.grid(alpha=0.25, which="both")

    # (d) filtered vs unfiltered: how much the masking strips, and add-back restore
    a = ax[1, 1]
    a.semilogx(k, r_used, "C3-", lw=2.4, label="unfilt (filt+add-back) / clean")
    a.semilogx(k, r_filt, "C3:", lw=2.0, label="filtered / clean (wings+cores gone)")
    a.fill_between(k, r_filt, r_used, color="C3", alpha=0.15,
                   label="restored by dla_core add-back")
    a.semilogx(k, r_t14, "C1-", lw=1.2, label="tier14 unfilt/clean")
    a.semilogx(k, r_t14f, "C1:", lw=1.2, label="tier14 filt/clean")
    vline(a)
    a.set_ylim(0, 11); a.set_xlim(k[0] * 0.9, 0.06)
    a.set_xlabel("k  [s/km, angular]"); a.set_ylabel("ratio / clean")
    a.set_title("(d) Filtering removes the saturated-core low-k spike;\n"
                "the dla_core add-back fully restores it (unfilt curve)")
    a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.25)

    fig.suptitle(
        f"DLA P1D low-k boost: PRIYA (LF sim, z={Z_FID:.0f}, target_F={tF:.2f}) vs Rogers+2018 (Illustris-1)\n"
        f"As-used DLA/clean = {r_used[ikmin]:.2f} at k_min=1e-3 (Rogers small={rog_small[ikmin]:.2f}, "
        f"large={rog_large[ikmin]:.2f}) — boost IS present, but confined to k<2e-3; band-median "
        f"{np.median(r_used[inb]):.2f} reflects Rogers' c(z) sub-1 high-k plateau, not a missing boost.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
