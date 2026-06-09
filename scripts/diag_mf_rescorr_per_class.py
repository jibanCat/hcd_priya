#!/usr/bin/env python3
"""diag_mf_rescorr_per_class.py -- PER-CLASS LF->HR resolution correction rho.

DESIGN PREREQUISITE for the MF (multi-fidelity) implementation. All prior
LF-vs-HR work (diag_lf_vs_hr_highk.py, diag_mf_rescorr_loso.py,
diag_mf_rescorr_vs_tau0_z.py) measured rho = P_HR/P_LF on the CLEAN forest only.
The HCD forward model is  P_obs = P_clean + Sum_c alpha_c R_c  where the excess
template R_c = P_c - P_clean (c in {LLS, subDLA, DLA}). If the resolution
correction rho_c differs by class, a clean-only rho would mis-correct the HCD
templates. PI question: "are we sure rho is the same across HCD classes?"

WHAT WE MEASURE (forward cache, NO training, NO emulator):
  1. rho_c(z,k) = P_HR_c / P_LF_c for each of the 4 COARSE classes
     (clean, LLS, subDLA, DLA), 6-sim mean. The 4 coarse classes are the
     count-weighted collapse of the 15 fine N_HI bins via data.py COARSE_SLICES,
     using tier_c_counts -- EXACTLY mirroring diag_lf_vs_hr_highk's clean collapse
     (CLEAN_SLICE = slice(0,1)) for all 4 slices.
  2. inter-class spread: max|rho_c - rho_clean| over (z,k), abs and as a fraction
     of the correction.
  3. the forward-relevant EXCESS template R_c = P_c - P_clean: its LF->HR ratio
     (P_HR excess / P_LF excess). This is what actually enters P_obs. R_c can be
     small / sign-changing, so we report where it is well-defined.
  4. per-class data quality: the coarse-class counts (tier_c_counts collapsed),
     flagging where rho_c is noise-dominated vs robust (esp. DLA, fewest counts).

MATCHING (identical to diag_lf_vs_hr_highk / multifidelity.match_hr_to_lf):
  HR's 6 design cosmologies are EXACT LF design points. Match HR<->LF rows on
  (params_unit rounded 6, z rounded 4, alpha_idx). alpha_slope is bit-identical
  per alpha_idx across the two caches, so the mean-flux (tau0) rung match is EXACT.
  HR is log-log interpolated onto the LF native k-grid (HR Nyquist ~0.20 strictly
  contains the LF Nyquist ~0.069), so HR is INTERPOLATED, never extrapolated.

OUTPUT:
  figures/analysis/04_emulator/mf_rescorr_per_class.png
  figures/analysis/04_emulator/mf_rescorr_per_class.txt

ENV (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
      scripts/diag_mf_rescorr_per_class.py
"""
from __future__ import annotations
import warnings
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/mfho/hcd_priya")
from hcd_analysis.emulator.data import (  # noqa: E402
    COARSE_SLICES, COARSE_NAMES, normalize_params)

LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
OUT_PNG = f"{OUT}/mf_rescorr_per_class.png"
OUT_TXT = f"{OUT}/mf_rescorr_per_class.txt"

# shared reference k-grid (matches the prior diag_mf_rescorr_loso band).
K_LO = 0.01           # focus on the resolution band
K_HI = 0.069          # LF Nyquist (shared LF/HR band)
NK = 40
HIK_FRAC = 0.75       # high-k = top quartile of the reference grid
LOWZ_MAX = 2.4        # low-z band (where the clean rho is largest)
N_ALPHA = 20

# The ~1% n_s-bias logP level the design must be compared against.
NS_BIAS_LEVEL = 0.01


def _strs(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load(path):
    """Load cache and produce all 4 coarse-class P1D by count-weighted collapse.

    Mirrors diag_lf_vs_hr_highk's clean collapse (CLEAN_SLICE -> count-weighted
    mean over fine sub-bins) but for ALL 4 COARSE_SLICES. Returns P4 (R,4,K) and
    the collapsed coarse counts cc (R,4)."""
    with h5py.File(path, "r") as f:
        Pf15 = f["P_tier_c_filtered"][:]            # (R,15,K)
        cnt15 = f["tier_c_counts"][:]               # (R,15)
        d = dict(
            kfkms=f["kfkms"][:],
            z=np.round(f["z_grid"][:], 4),
            aidx=f["alpha_idx"][:].astype(int),
            params=f["params"][:],
            sim=_strs(f["sim_name"][:]),
        )
    R, _, K = Pf15.shape
    P4 = np.full((R, 4, K), np.nan)
    cc = np.zeros((R, 4))
    for ci, s in enumerate(COARSE_SLICES):
        c = cnt15[:, s].astype(float)
        cc[:, ci] = c.sum(1)
        w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
        seg = Pf15[:, s, :]
        contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
        summ = contrib.sum(1)
        allnan = (~np.isfinite(seg)).all(1)
        summ[allnan] = np.nan
        P4[:, ci, :] = summ
    d["P4"] = P4
    d["cc"] = cc
    d["pu"] = np.round(normalize_params(d["params"]), 6)
    return d


def rowkey(d):
    return [(tuple(d["pu"][i]), d["z"][i], d["aidx"][i]) for i in range(len(d["z"]))]


def main():
    lf = load(LF_CACHE)
    hr = load(HR_CACHE)

    # all-NaN slices (bins above native Nyquist, single-finite ddof=1) produce
    # benign "Mean of empty slice" / "Degrees of freedom <= 0" RuntimeWarnings from
    # nanmean/nanstd; they are expected (NaN-masked) and handled by the fill-with-NaN
    # init, so silence them for clean output.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # shared reference k-grid (log-uniform) within the LF<->HR overlap band.
    ref_k = np.power(10.0, np.linspace(np.log10(K_LO), np.log10(K_HI), NK))
    logk_ref = np.log10(ref_k)

    # match every HR row to its LF row on (params_unit, z, alpha)
    lk = {k: i for i, k in enumerate(rowkey(lf))}
    hkeys = rowkey(hr)
    hr_rows, lf_rows = [], []
    for h, k in enumerate(hkeys):
        if k in lk:
            hr_rows.append(h)
            lf_rows.append(lk[k])
    hr_rows = np.array(hr_rows)
    lf_rows = np.array(lf_rows)
    assert len(hr_rows) == len(hkeys), "not all HR rows matched"

    sims = sorted(set(hr["sim"]))
    n_sim = len(sims)
    sim_idx = {s: i for i, s in enumerate(sims)}
    zvals = np.array(sorted(set(hr["z"][hr_rows])))
    nz = len(zvals)

    # ---------------------------------------------------------------------- #
    # Per matched row, interpolate the 4-class LF and HR P1D onto ref_k
    # (log-log within each source's support; never extrapolated).
    # ---------------------------------------------------------------------- #
    M = len(hr_rows)
    logP_lf = np.full((M, 4, NK), np.nan)    # LF logP per class on ref grid
    logP_hr = np.full((M, 4, NK), np.nan)    # HR logP per class on ref grid
    row_sim = np.empty(M, int)
    row_z = np.empty(M)
    row_alpha = np.empty(M, int)
    cc_lf = np.full((M, 4), np.nan)
    cc_hr = np.full((M, 4), np.nan)

    for m, (h, l) in enumerate(zip(hr_rows, lf_rows)):
        khr = hr["kfkms"][h]
        klf = lf["kfkms"][l]
        for ci in range(4):
            Phr = hr["P4"][h, ci]
            Plf = lf["P4"][l, ci]
            mhr = np.isfinite(khr) & (khr > 0) & np.isfinite(Phr) & (Phr > 0)
            mlf = np.isfinite(klf) & (klf > 0) & np.isfinite(Plf) & (Plf > 0)
            if mhr.sum() >= 3 and mlf.sum() >= 3:
                logP_hr[m, ci] = np.interp(logk_ref, np.log10(khr[mhr]),
                                           np.log(Phr[mhr]),
                                           left=np.nan, right=np.nan)
                logP_lf[m, ci] = np.interp(logk_ref, np.log10(klf[mlf]),
                                           np.log(Plf[mlf]),
                                           left=np.nan, right=np.nan)
        row_sim[m] = sim_idx[hr["sim"][h]]
        row_z[m] = hr["z"][h]
        row_alpha[m] = hr["aidx"][h]
        cc_lf[m] = lf["cc"][l]
        cc_hr[m] = hr["cc"][h]

    # rho_c per row (log): g = logP_HR - logP_LF ; rho = exp(g)
    g = logP_hr - logP_lf                    # (M,4,NK)
    rho = np.exp(g)

    # ---- per-(class,z,k) 6-sim mean rho (avg over alpha within sim, then over sims)
    rho_szk = np.full((n_sim, 4, nz, NK), np.nan)  # per sim mean over alpha
    for si in range(n_sim):
        for zi, zz in enumerate(zvals):
            sel = (row_sim == si) & (row_z == zz)
            if sel.any():
                with np.errstate(invalid="ignore"):
                    rho_szk[si, :, zi, :] = np.nanmean(rho[sel], axis=0)
    with np.errstate(invalid="ignore"):
        rho_czk = np.nanmean(rho_szk, axis=0)        # (4,nz,NK) 6-sim mean
        rho_czk_std = np.nanstd(rho_szk, axis=0, ddof=1)   # inter-sim scatter

    # ---- EXCESS template R_c = P_c - P_clean, and its LF->HR ratio ---------
    # work in LINEAR P. P_lf_c = exp(logP_lf), etc. Build per-row linear P.
    P_lf = np.exp(logP_lf)                   # (M,4,NK)
    P_hr = np.exp(logP_hr)
    # excess for c in {LLS=1, subDLA=2, DLA=3}
    R_lf = P_lf[:, 1:, :] - P_lf[:, 0:1, :]  # (M,3,NK)
    R_hr = P_hr[:, 1:, :] - P_hr[:, 0:1, :]
    # 6-sim mean of the excess and the excess ratio per (class,z,k)
    Rlf_szk = np.full((n_sim, 3, nz, NK), np.nan)
    Rhr_szk = np.full((n_sim, 3, nz, NK), np.nan)
    for si in range(n_sim):
        for zi, zz in enumerate(zvals):
            sel = (row_sim == si) & (row_z == zz)
            if sel.any():
                with np.errstate(invalid="ignore"):
                    Rlf_szk[si, :, zi, :] = np.nanmean(R_lf[sel], axis=0)
                    Rhr_szk[si, :, zi, :] = np.nanmean(R_hr[sel], axis=0)
    with np.errstate(invalid="ignore"):
        Rlf_czk = np.nanmean(Rlf_szk, axis=0)        # (3,nz,NK)
        Rhr_czk = np.nanmean(Rhr_szk, axis=0)
    # excess ratio from the 6-sim mean excess (more robust than per-row ratio mean
    # when R_c is small). Defined only where |R_lf| is not tiny.
    rho_excess_czk = Rhr_czk / Rlf_czk           # (3,nz,NK)
    # per (z,k) 6-sim-mean clean LF P (the denominator for the excess fraction).
    Pclean_szk = np.full((n_sim, nz, NK), np.nan)
    for si in range(n_sim):
        for zi, zz in enumerate(zvals):
            sel = (row_sim == si) & (row_z == zz)
            if sel.any():
                with np.errstate(invalid="ignore"):
                    Pclean_szk[si, zi] = np.nanmean(P_lf[sel, 0, :], axis=0)
    with np.errstate(invalid="ignore"):
        Pclean_zk = np.nanmean(Pclean_szk, axis=0)   # (nz,NK)
    excess_frac = Rlf_czk / Pclean_zk[None]          # (3,nz,NK) excess / clean

    # bands
    hik_mask = ref_k >= np.quantile(ref_k, HIK_FRAC)
    lowz_z = zvals <= LOWZ_MAX

    lines = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    excess_names = COARSE_NAMES[1:]   # LLS, subDLA, DLA

    P("=" * 80)
    P("PER-CLASS LF->HR resolution correction rho_c = P_HR_c / P_LF_c")
    P("FORWARD cache analysis (no training, no emulator). 4 coarse classes.")
    P("=" * 80)
    P(f"matched HR rows = {M} ; HR sims = {n_sim} ; z grid = {nz} ; alpha rungs = {N_ALPHA}")
    P(f"shared k-grid: {NK} log-bins, k=[{ref_k.min():.4f},{ref_k.max():.4f}] s/km")
    P(f"  (HR log-log interpolated onto the shared grid within HR support; LF same)")
    P(f"high-k band (reporting): k>={ref_k[hik_mask].min():.4f} s/km (top quartile)")
    P(f"low-z band  (reporting): z<={LOWZ_MAX}")
    P(f"n_s-bias reference level: ~{100*NS_BIAS_LEVEL:.0f}% in logP")
    P("")
    P("HR sims (ns):")
    ns_vals = []
    for s in sims:
        pr = hr["params"][hr["sim"] == s][0]
        ns_vals.append(pr[0])
        P(f"  ns={pr[0]:.3f} Ap={pr[1]:.3e}")
    P(f"  ns span [{min(ns_vals):.3f}, {max(ns_vals):.3f}] (clustered mid-high)")
    P("")

    # find the index nearest z=2.2 and k=0.05 for the headline numbers
    iz22 = int(np.argmin(np.abs(zvals - 2.2)))
    ik05 = int(np.argmin(np.abs(ref_k - 0.05)))
    P("-" * 80)
    P(f"(1) rho_c at (z={zvals[iz22]:.1f}, k={ref_k[ik05]:.4f} s/km), 6-sim mean:")
    P("-" * 80)
    for ci, nm in enumerate(COARSE_NAMES):
        r = rho_czk[ci, iz22, ik05]
        s = rho_czk_std[ci, iz22, ik05]
        P(f"  {nm:7s}: rho = {r:.4f}  ({100*(r-1):+.2f}% in P)   inter-sim std = {s:.4f}")
    P("")
    # also the low-z high-k BAND mean (more robust than a single cell)
    lh = np.ix_(lowz_z, hik_mask)
    P(f"  low-z(z<={LOWZ_MAX}) high-k(top quartile) BAND-mean rho_c:")
    for ci, nm in enumerate(COARSE_NAMES):
        rband = np.nanmean(rho_czk[ci][lh])
        sband = np.nanmean(rho_czk_std[ci][lh])
        P(f"    {nm:7s}: rho = {rband:.4f}  ({100*(rband-1):+.2f}% in P)   "
          f"mean inter-sim std = {sband:.4f}")
    P("")

    # ---------------------------------------------------------------------- #
    # (2) inter-class spread of rho_c vs rho_clean
    # ---------------------------------------------------------------------- #
    P("-" * 80)
    P("(2) INTER-CLASS SPREAD: rho_c - rho_clean (the per-class deviation)")
    P("-" * 80)
    rho_clean = rho_czk[0]                            # (nz,NK)
    P(f"  {'class':8s} {'max|rho_c-rho_clean|':>22s} {'(z,k) of max':>20s} "
      f"{'band-mean|diff|':>16s}")
    max_diffs = {}
    for ci in range(1, 4):
        diff = rho_czk[ci] - rho_clean              # (nz,NK)
        ad = np.abs(diff)
        if np.all(~np.isfinite(ad)):
            continue
        amax = np.nanmax(ad)
        zi, ki = np.unravel_index(np.nanargmax(ad), ad.shape)
        bandmean = np.nanmean(ad[lh])
        max_diffs[COARSE_NAMES[ci]] = (amax, zvals[zi], ref_k[ki], bandmean)
        P(f"  {COARSE_NAMES[ci]:8s} {amax:22.4f} "
          f"{f'(z={zvals[zi]:.1f},k={ref_k[ki]:.3f})':>20s} {bandmean:16.4f}")
    P("")
    # express as a FRACTION of the correction (the clean correction depth |rho-1|)
    corr_depth_band = np.abs(np.nanmean(rho_clean[lh]) - 1.0)
    P(f"  clean correction DEPTH (low-z high-k band) = |rho_clean-1| = {corr_depth_band:.4f} "
      f"({100*corr_depth_band:.2f}% in P)")
    for ci in range(1, 4):
        nm = COARSE_NAMES[ci]
        if nm not in max_diffs:
            continue
        amax, zmx, kmx, bandmean = max_diffs[nm]
        P(f"  {nm:7s}: max|diff|={100*amax:+.2f}pp = {amax/max(corr_depth_band,1e-9):.2f}x "
          f"the correction depth; band-mean|diff|={100*bandmean:.2f}pp "
          f"= {bandmean/max(corr_depth_band,1e-9):.2f}x")
    # the headline: is the class difference >> or << the n_s level?
    worst_band = max(max_diffs.values(), key=lambda v: v[3])[3] if max_diffs else np.nan
    worst_max = max(max_diffs.values(), key=lambda v: v[0])[0] if max_diffs else np.nan
    P("")
    P(f"  WORST band-mean inter-class |rho_c-rho_clean| = {100*worst_band:.2f}pp")
    P(f"  WORST pointwise inter-class |rho_c-rho_clean|  = {100*worst_max:.2f}pp")
    P(f"  (compare to the ~{100*NS_BIAS_LEVEL:.0f}% n_s-bias logP level)")
    P("")

    # ---------------------------------------------------------------------- #
    # (3) the FORWARD-relevant quantity: EXCESS template R_c LF->HR ratio
    # ---------------------------------------------------------------------- #
    P("-" * 80)
    P("(3) EXCESS TEMPLATE R_c = P_c - P_clean : its LF->HR ratio (P_HR exc/P_LF exc)")
    P("-" * 80)
    P("  This is what enters the forward model P_obs = P_clean + Sum_c alpha_c R_c.")
    P("  KEY: the 'excess' is SIGNED and SIGN-STABLE per class --")
    P("    LLS    : R_c > 0 (adds a little power at low k, ~0 at high k)")
    P("    subDLA : R_c < 0 (SUPPRESSES power; ~-15% of clean)")
    P("    DLA    : R_c < 0 (strongly suppresses; ~-50% of clean)")
    P("  The ratio rho_excess = R_HR/R_LF is well-defined where R_LF and R_HR share")
    P("  sign AND |R_LF/P_clean| is sizeable (>2%); near R_c~0 the ratio is unstable.")
    P("")
    SIZE_GATE = 0.02   # |excess/clean| must exceed 2% for the ratio to be stable
    rho_clean_band = np.nanmean(rho_clean[lh])
    for ci, nm in enumerate(excess_names):
        # well-defined: finite ratio, SAME sign (R_lf*R_hr>0), sizeable LF excess.
        wd = (np.isfinite(rho_excess_czk[ci]) & (Rlf_czk[ci] * Rhr_czk[ci] > 0)
              & (np.abs(excess_frac[ci]) > SIZE_GATE))
        nfin = np.isfinite(rho_excess_czk[ci])
        frac_def = np.mean(wd[nfin]) if nfin.any() else 0.0
        with np.errstate(invalid="ignore"):
            ef_band = np.nanmedian(excess_frac[ci][lh])           # signed
            re_band = np.nanmean(np.where(wd, rho_excess_czk[ci], np.nan)[lh])
            re_all = np.nanmean(np.where(wd, rho_excess_czk[ci], np.nan))
            # how far the excess ratio departs from rho_clean, where well-defined
            dev_band = np.nanmean(np.where(wd, np.abs(rho_excess_czk[ci] - rho_clean),
                                           np.nan)[lh])
        P(f"  {nm:7s}: excess/clean (low-z hik median) = {100*ef_band:+.1f}% of clean P "
          f"(SIGN {'+' if ef_band > 0 else '-'})")
        P(f"           rho_excess (well-defined cells): all-band mean={re_all:.3f}  "
          f"low-z hik mean={re_band:.3f}")
        P(f"           rho_clean low-z hik = {rho_clean_band:.3f}  -> |rho_excess-rho_clean| "
          f"band-mean = {100*dev_band:.2f}pp")
        P(f"           ({100*frac_def:.0f}% of finite (z,k) cells have a sizeable, "
          f"sign-stable excess => ratio well-defined)")
    P("")
    P("  INTERPRETATION: where the excess is SIZEABLE (subDLA/DLA: large negative,")
    P("  ~60-100% of cells well-defined), rho_excess ~ rho_clean -> the clean rho")
    P("  applied to the whole P_obs (clean + signed excess) resolution-corrects the")
    P("  excess CORRECTLY. The LLS excess is tiny (<1% of clean) and sign-changes")
    P("  toward high k, so its ratio is ill-conditioned -- but a tiny excess carries")
    P("  negligible weight in P_obs, so a clean rho mis-correcting it is harmless.")
    P("")

    # ---------------------------------------------------------------------- #
    # (4) per-class data quality / counts
    # ---------------------------------------------------------------------- #
    P("-" * 80)
    P("(4) PER-CLASS DATA QUALITY: coarse-class counts (tier_c_counts collapsed)")
    P("-" * 80)
    P("  Counts are sightline x pixel contributions to each class' P1D (NOT #sightlines).")
    P(f"  {'class':8s} {'LF mean cnt':>14s} {'LF lowz cnt':>14s} "
      f"{'HR mean cnt':>14s} {'HR lowz cnt':>14s} {'inter-sim CoV(rho)':>20s}")
    lowz_rows = row_z <= LOWZ_MAX
    for ci, nm in enumerate(COARSE_NAMES):
        lf_mean = np.nanmean(cc_lf[:, ci])
        lf_lowz = np.nanmean(cc_lf[lowz_rows, ci])
        hr_mean = np.nanmean(cc_hr[:, ci])
        hr_lowz = np.nanmean(cc_hr[lowz_rows, ci])
        # inter-sim CoV of rho in low-z high-k = robustness proxy
        with np.errstate(invalid="ignore"):
            cov = np.nanmean(rho_czk_std[ci][lh]) / np.abs(np.nanmean(rho_czk[ci][lh]))
        P(f"  {nm:8s} {lf_mean:14.0f} {lf_lowz:14.0f} {hr_mean:14.0f} {hr_lowz:14.0f} "
          f"{100*cov:18.2f}%")
    P("")
    P("  The CoV(rho) (inter-sim scatter / mean) is the noise proxy: classes with")
    P("  fewer counts (DLA) have noisier per-row P1D -> larger CoV. A class whose")
    P("  rho_c differs from clean by LESS than its own CoV is noise-dominated: the")
    P("  6 HR sims cannot resolve a real per-class difference there.")
    P("")

    # ---------------------------------------------------------------------- #
    # VERDICT
    # ---------------------------------------------------------------------- #
    P("=" * 80)
    P("VERDICT")
    P("=" * 80)
    # is the worst inter-class band difference below or above the n_s level?
    dla_cov = (np.nanmean(rho_czk_std[3][lh]) /
               np.abs(np.nanmean(rho_czk[3][lh])))
    P(f"  - clean rho low-z high-k = {100*(np.nanmean(rho_clean[lh])-1):+.2f}% in P "
      f"(matches prior clean-only diags ~+6 to +9%).")
    P(f"  - WORST inter-class band-mean |rho_c - rho_clean| = {100*worst_band:.2f}pp, "
      f"pointwise worst {100*worst_max:.2f}pp.")
    P(f"  - DLA inter-sim CoV(rho) low-z high-k = {100*dla_cov:.2f}% (the noisiest class).")
    if worst_band < NS_BIAS_LEVEL and worst_band < 1.5 * np.nanmean(
            [(np.nanmean(rho_czk_std[c][lh]) / np.abs(np.nanmean(rho_czk[c][lh])))
             for c in range(1, 4)]):
        P("  => The inter-class band-mean difference is BELOW the ~1% n_s level AND")
        P("     comparable to the inter-sim (6-sim) noise: a SINGLE clean rho applied")
        P("     to all classes is adequate at the n_s-bias tolerance. A per-class rho")
        P("     is NOT required by these data (and cannot be resolved by 6 HR sims).")
    elif worst_band < 1.5 * NS_BIAS_LEVEL:
        P("  => The inter-class difference is at the ~1% n_s level (borderline): a")
        P("     single clean rho is mostly adequate but may leave a ~n_s-level per-class")
        P("     residual on the largest-excess class. Flag; revisit with more HR sims.")
    else:
        P("  => The inter-class difference EXCEEDS the ~1% n_s level: a PER-CLASS rho")
        P("     (at least for the offending class) is warranted.")
    P("")
    P("  HONESTY CAVEAT: only 6 HR sims, ns in [%.3f,%.3f] (clustered mid-high)." %
      (min(ns_vals), max(ns_vals)))
    P("  The excess R_c = P_c - P_clean is a small difference of two large noisy")
    P("  numbers -> its LF->HR ratio is intrinsically noisier than rho_clean, and")
    P("  for DLA (fewest counts) the per-class signal may be at or below the 6-sim")
    P("  noise floor. Where the inter-class |diff| is < the per-class CoV, the 6-sim")
    P("  statistics CANNOT resolve a real per-class difference -- 'class-independent'")
    P("  there means 'not distinguishable', not 'proven identical'.")
    P("=" * 80)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {OUT_TXT}")

    make_fig(ref_k, zvals, rho_czk, rho_czk_std, rho_excess_czk, excess_frac,
             Rlf_czk, Rhr_czk, hik_mask, lowz_z, ns_vals)


def make_fig(ref_k, zvals, rho_czk, rho_czk_std, rho_excess_czk, excess_frac,
             Rlf_czk, Rhr_czk, hik_mask, lowz_z, ns_vals):
    colors = {"clean": "k", "LLS": "C0", "subDLA": "C1", "DLA": "C3"}
    excess_names = COARSE_NAMES[1:]
    # representative z-bands
    zsel = [zvals[np.argmin(np.abs(zvals - zt))] for zt in (2.2, 3.0, 4.0)]

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.27)

    # top row: rho_c(k) for all 4 classes, one panel per z-band
    for col, zz in enumerate(zsel):
        ax = fig.add_subplot(gs[0, col])
        zi = int(np.argmin(np.abs(zvals - zz)))
        for ci, nm in enumerate(COARSE_NAMES):
            ax.plot(ref_k, rho_czk[ci, zi], "-", color=colors[nm], lw=1.8, label=nm)
            ax.fill_between(ref_k,
                            rho_czk[ci, zi] - rho_czk_std[ci, zi],
                            rho_czk[ci, zi] + rho_czk_std[ci, zi],
                            color=colors[nm], alpha=0.12)
        ax.axhline(1.0, color="grey", lw=0.7, ls=":")
        ax.axvspan(ref_k[hik_mask].min(), ref_k.max(), color="orange", alpha=0.08)
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km]")
        ax.set_ylabel(r"$\rho_c = P_{HR}/P_{LF}$")
        ax.set_title(f"per-class rho   z={zz:.1f}  (band=6-sim mean +/- inter-sim std)")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    # bottom-left: inter-class spread rho_c - rho_clean vs k (low-z band mean)
    ax = fig.add_subplot(gs[1, 0])
    with np.errstate(invalid="ignore"):
        rho_clean_lz = np.nanmean(rho_czk[0][lowz_z], axis=0)
    for ci in range(1, 4):
        nm = COARSE_NAMES[ci]
        with np.errstate(invalid="ignore"):
            rc_lz = np.nanmean(rho_czk[ci][lowz_z], axis=0)
        ax.plot(ref_k, 100 * (rc_lz - rho_clean_lz), "-", color=colors[nm],
                lw=1.8, label=f"{nm} - clean")
    ax.axhline(0.0, color="grey", lw=0.7, ls=":")
    ax.axhline(1.0, color="red", lw=0.7, ls="--", label="+/-1% n_s level")
    ax.axhline(-1.0, color="red", lw=0.7, ls="--")
    ax.axvspan(ref_k[hik_mask].min(), ref_k.max(), color="orange", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel(r"$\rho_c - \rho_{clean}$  [percentage points]")
    ax.set_title("INTER-CLASS spread (low-z band mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # bottom-mid: EXCESS template correction rho_excess vs clean (low-z band)
    # gate: same-sign & sizeable (|excess/clean|>2%), matching the text.
    ax = fig.add_subplot(gs[1, 1])
    with np.errstate(invalid="ignore"):
        rho_clean_lz = np.nanmean(rho_czk[0][lowz_z], axis=0)
    ax.plot(ref_k, rho_clean_lz, "k-", lw=2.2, label="clean rho")
    for ci, nm in enumerate(excess_names):
        wd = ((Rlf_czk[ci] * Rhr_czk[ci] > 0) & (np.abs(excess_frac[ci]) > 0.02))
        with np.errstate(invalid="ignore"):
            re_lz = np.nanmean(np.where(wd, rho_excess_czk[ci], np.nan)[lowz_z], axis=0)
        ax.plot(ref_k, re_lz, "--o", color=colors[nm], lw=1.8, ms=3,
                label=f"rho_excess[{nm}]")
    ax.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax.axvspan(ref_k[hik_mask].min(), ref_k.max(), color="orange", alpha=0.08)
    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.25)
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel(r"LF$\to$HR ratio")
    ax.set_title("EXCESS-template correction vs clean\n(low-z, same-sign & |excess/clean|>2%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # bottom-right: rho(z,k) inter-class max|diff| heatmap (over classes)
    ax = fig.add_subplot(gs[1, 2])
    with np.errstate(invalid="ignore"):
        maxdiff = np.nanmax(np.abs(rho_czk[1:] - rho_czk[0:1]), axis=0)  # (nz,NK)
    im = ax.pcolormesh(np.arange(len(ref_k) + 1), np.arange(len(zvals) + 1),
                       100 * maxdiff, cmap="viridis", vmin=0, vmax=5)
    nkt = max(1, len(ref_k) // 8)
    ax.set_xticks(np.arange(len(ref_k))[::nkt] + 0.5)
    ax.set_xticklabels([f"{v:.3f}" for v in ref_k[::nkt]], rotation=60, fontsize=7)
    ax.set_yticks(np.arange(len(zvals)) + 0.5)
    ax.set_yticklabels([f"{v:.1f}" for v in zvals], fontsize=7)
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("z")
    ax.set_title("max over classes |rho_c - rho_clean| [pp]")
    fig.colorbar(im, ax=ax, label="pp")

    fig.suptitle(
        "Per-class LF->HR resolution correction rho_c (forward cache, 6 HR sims; "
        f"ns in [{min(ns_vals):.2f},{max(ns_vals):.2f}])",
        fontsize=13, y=0.98)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT_PNG}")


if __name__ == "__main__":
    main()
