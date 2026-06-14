#!/usr/bin/env python3
"""diag_mf_rescorr_loso.py -- LOSO validation of the LF->HR resolution correction's
theta-insensitivity, FORWARD cache analysis (no training, no emulator backbone).

WHAT THIS TESTS (the repo's actual MF correction form, multifidelity.py):
  The default validated MF model (delta_mode='none', FixedMeanHead + log_rho) applies
  a THETA-INDEPENDENT, per-(z,k) mean LF->HR log-ratio:
      g_correction(z,k) = gbar(z,k) = < log P_HR - log P_LF >_{sims,alpha at z}
  carried on top of the per-k log_rho (g == log_rho + (gbar-log_rho) == gbar(z,k)).
  ALL theta-dependence is assumed to live in f_LF; the correction itself is fixed.
  (There is ALSO a separate FIXED particle-convergence res_corr factor, which is
  parameter-independent by construction and NOT what this test interrogates.)

  We test exactly that correction, but measured DIRECTLY ON THE RAW CACHE P1D (no
  frozen-emulator backbone, no log-log extrapolation) so that the test is a clean
  forward comparison free of emulator interpolation error:
      rho_log(sim,z,k,alpha) = log P_HR_clean - log P_LF_clean   (the per-row log-ratio)
  on the SHARED k-grid: HR is log-log interpolated DOWN onto the LF k-grid (LF k_max
  ~0.069 s/km), so HR is interpolated within its support, never extrapolated, and the
  LF is used on its native bins (no LF tail extrapolation).

  NB: because the LF native k_max is ~0.069 s/km, this raw-cache test covers the
  band the LF can actually resolve. The high-k KODIAQ band (0.07-0.2 s/km) is where
  the LF prediction is a TAIL EXTRAPOLATION corrected by the head; that band is NOT
  probed by a raw-cache ratio (no LF truth there) and is flagged, not tested, here.

TESTS:
  1. theta-sensitivity of rho:
     (a) inter-sim scatter std_over_sims(rho)/mean(rho) at fixed (z,k[,alpha]);
     (b) regress log-rho on the 9 unit-cube theta params across the 6 sims per (z,k);
         report R^2 (variance explained) and which params drive it;
     (c) tau0-dependence: rho at alpha-rung 0 vs 19.
  2. 6-fold LOSO: hold out sim h, build gbar_{-h}(z,k) from the other 5 sims (mean
     log-ratio at each z, matching FixedMeanHead/fixed_mean_table), apply to the
     held-out LF: logP_pred = logP_LF(h) + gbar_{-h}(z,k); compare to true logP_HR(h):
         eps(z,k) = exp(logP_pred - logP_HR) - 1  (fractional P error).
     Report eps low-z high-k and pooled vs the ~-6% correction and the ~1% n_s level.

OUTPUT: figures/analysis/04_emulator/mf_rescorr_loso.{png,txt}
"""
from __future__ import annotations
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/mfho/hcd_priya")
from hcd_analysis.emulator.data import normalize_params, COARSE_NAMES  # noqa: E402

LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
OUT_PNG = "/home/mfho/hcd_priya/figures/analysis/04_emulator/mf_rescorr_loso.png"
OUT_TXT = "/home/mfho/hcd_priya/figures/analysis/04_emulator/mf_rescorr_loso.txt"

PARAM_NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub",
               "omegamh2", "hireionz", "bhfeedback"]
CLEAN = 0                      # clean-forest class index in P_tier_c_filtered
N_ALPHA = 20
# the "low-z high-k" band for reporting. low-z = z<=2.6 (where the LF deficit is
# largest, matching the prior diag_lf_vs_hr_highk 2.0-2.4 band); high-k = top
# quartile of the reference k-grid (the band the LF cannot resolve well).
LOWZ_MAX = 2.6


def _strs(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_cache(fn):
    with h5py.File(fn, "r") as f:
        d = dict(
            P=f["P_tier_c_filtered"][:, CLEAN, :],   # (N, K) clean-forest P1D
            kfkms=f["kfkms"][:],                       # (N, K)
            z=np.round(f["z_grid"][:], 4),
            alpha=f["alpha_idx"][:].astype(int),
            params=f["params"][:],
            sim=_strs(f["sim_name"][:]),
        )
    d["pu"] = np.round(normalize_params(d["params"]), 6)   # unit-cube params (N,9)
    return d


def rowkey(d):
    return [(tuple(d["pu"][i]), d["z"][i], d["alpha"][i]) for i in range(len(d["z"]))]


def main():
    lf = load_cache(LF_CACHE)
    hr = load_cache(HR_CACHE)

    # The LF k-grid in s/km varies per row (k = angular_bin / box_velocity, which
    # depends on the cosmology & z through the Hubble flow). So we build a FIXED
    # reference k-grid and interpolate BOTH the LF and HR P1D (each on its OWN native
    # k-grid) onto it by log-log interpolation, NEVER extrapolating beyond either
    # source. Reference grid = log-uniform from the DATA k_min up to the SMALLEST LF
    # k_max over all matched rows (so every LF row supports the whole reference band).
    # This keeps the ratio a clean within-support comparison on a common grid.
    # reference band: 0.01 s/km up to the LF Nyquist 0.069 (matches the prior
    # diag_lf_vs_hr_highk K_CMP_MAX). Rows whose own k_max falls below a reference
    # bin contribute NaN there (np.interp right=NaN -> nanmean ignores); only ~5% of
    # LF rows roll off below 0.0645, so the high-k bins remain well populated.
    k_lo = 0.01                                          # focus on the resolution band
    k_hi = 0.069                                         # LF Nyquist (shared LF/HR band)
    K = 40
    lf_k = np.power(10.0, np.linspace(np.log10(k_lo), np.log10(k_hi), K))  # reference k
    logk_lf = np.log10(lf_k)

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

    # ---- per-row log-ratio g = logP_HR - logP_LF on the LF grid ---------------- #
    # HR log-log interpolated onto logk_lf within HR support (no extrapolation).
    M = len(hr_rows)
    g = np.full((M, K), np.nan)            # log-ratio per matched row
    row_sim = np.empty(M, dtype=int)
    row_z = np.empty(M)
    row_alpha = np.empty(M, dtype=int)
    row_pu = np.empty((M, 9))
    logPlf_row = np.full((M, K), np.nan)   # LF truth on reference grid (per row)
    logPhr_row = np.full((M, K), np.nan)   # HR truth on reference grid (per row)
    for m, (h, l) in enumerate(zip(hr_rows, lf_rows)):
        Phr = hr["P"][h]
        khr = hr["kfkms"][h]
        Plf = lf["P"][l]
        klf = lf["kfkms"][l]
        mhr = np.isfinite(Phr) & (Phr > 0) & np.isfinite(khr)
        mlf = np.isfinite(Plf) & (Plf > 0) & np.isfinite(klf)
        # interpolate BOTH onto reference logk_lf within their own support (NaN outside)
        logPhr_i = np.interp(logk_lf, np.log10(khr[mhr]), np.log(Phr[mhr]),
                             left=np.nan, right=np.nan)
        logPlf_i = np.interp(logk_lf, np.log10(klf[mlf]), np.log(Plf[mlf]),
                             left=np.nan, right=np.nan)
        logPlf_row[m] = logPlf_i
        logPhr_row[m] = logPhr_i
        g[m] = logPhr_i - logPlf_i
        row_sim[m] = sim_idx[hr["sim"][h]]
        row_z[m] = hr["z"][h]
        row_alpha[m] = hr["alpha"][h]
        row_pu[m] = hr["pu"][h]

    rho = np.exp(g)                         # multiplicative resolution correction
    zvals = np.array(sorted(set(row_z)))
    # high-k = top quartile of the LF k-grid (the band where LF is power-deficient)
    hik_mask = lf_k >= np.quantile(lf_k, 0.75)
    lowz_mask_z = zvals <= LOWZ_MAX

    lines = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    P("=" * 78)
    P("MF LF->HR resolution-correction LOSO validation (FORWARD cache, no training)")
    P("=" * 78)
    P(f"correction form tested: g(sim,z,k,alpha) = log P_HR_clean - log P_LF_clean")
    P(f"  (the repo default delta_mode='none' applies gbar(z,k) = <g>_sims,alpha @ z,")
    P(f"   theta-INDEPENDENT; this test measures g on RAW cache P1D, no emulator)")
    P(f"shared k-grid: LF native {K} bins, k=[{lf_k.min():.4f},{lf_k.max():.4f}] s/km")
    P(f"  (HR log-log interpolated DOWN onto LF grid; never extrapolated)")
    P(f"matched HR rows: {M} ; HR sims: {n_sim}")
    P(f"high-k band (reporting): k>={lf_k[hik_mask].min():.4f} s/km (top quartile)")
    P(f"low-z band (reporting):  z<={LOWZ_MAX}")
    P("")
    P("HR sims and their ns / Ap (unit-cube ns at index 0):")
    ns_vals, ap_vals = [], []
    for s in sims:
        m = hr["sim"] == s
        pr = hr["params"][m][0]
        pu = hr["pu"][m][0]
        ns_vals.append(pr[0]); ap_vals.append(pr[1])
        P(f"  {s[:34]:34s} ns={pr[0]:.3f} Ap={pr[1]:.3e}  ns_unit={pu[0]:.3f}")
    P(f"  ns span: [{min(ns_vals):.3f}, {max(ns_vals):.3f}]  "
      f"Ap span: [{min(ap_vals):.2e}, {max(ap_vals):.2e}]")
    P("")

    # ===================================================================== #
    # TEST 1a: inter-sim scatter of rho at fixed (z,k), pooled over alpha
    # ===================================================================== #
    # mean rho over alpha per (sim,z,k), then scatter over sims.
    rho_szk = np.full((n_sim, len(zvals), K), np.nan)
    for si in range(n_sim):
        for zi, zz in enumerate(zvals):
            sel = (row_sim == si) & (row_z == zz)
            if sel.sum() == 0:
                continue
            with np.errstate(invalid="ignore"):
                rho_szk[si, zi] = np.nanmean(rho[sel], axis=0)
    with np.errstate(invalid="ignore"):
        mean_zk = np.nanmean(rho_szk, axis=0)                 # (Nz,K)
        std_zk = np.nanstd(rho_szk, axis=0, ddof=1)           # (Nz,K)
        cov_zk = std_zk / np.abs(mean_zk)                     # coefficient of variation

    P("-" * 78)
    P("TEST 1a: inter-sim scatter of rho  std_over_sims(rho)/|mean(rho)| per (z,k)")
    P("-" * 78)
    lowz_hik = np.ix_(lowz_mask_z, hik_mask)
    P(f"  mean(rho) overall: {np.nanmean(mean_zk):.4f}  "
      f"(so the mean correction is {100*(np.nanmean(mean_zk)-1):+.1f}% in P)")
    P(f"  inter-sim CoV(rho) pooled all (z,k):   median={np.nanmedian(cov_zk):.4f}  "
      f"max={np.nanmax(cov_zk):.4f}")
    P(f"  inter-sim CoV(rho) LOW-z HIGH-k band:  median={np.nanmedian(cov_zk[lowz_hik]):.4f}  "
      f"max={np.nanmax(cov_zk[lowz_hik]):.4f}")
    # the mean correction magnitude in the low-z high-k band:
    corr_lowz_hik = np.nanmean(mean_zk[lowz_hik])
    P(f"  mean correction LOW-z HIGH-k:          rho={corr_lowz_hik:.4f} "
      f"({100*(corr_lowz_hik-1):+.1f}% in P)")
    P("")

    # ===================================================================== #
    # TEST 1b: regress log-rho on the 9 unit-cube theta across the 6 sims
    # ===================================================================== #
    # per (z,k): y_sim = mean_alpha log-rho ; X = sim unit-cube params (centered).
    # R^2 = fraction of inter-sim log-rho variance the 9 params explain.
    # With 6 sims & 9 params the full regression is under-determined (R^2->1 trivially),
    # so we report (i) a SINGLE-param univariate R^2 per param (which param best
    # predicts rho), averaged over the low-z high-k band, AND (ii) note the rank deficit.
    # sim-level params (one row per sim)
    sim_pu = np.zeros((n_sim, 9))
    for s in sims:
        si = sim_idx[s]
        sim_pu[si] = hr["pu"][hr["sim"] == s][0]
    sim_pu_c = sim_pu - sim_pu.mean(0)

    g_szk = np.log(rho_szk)                                   # (n_sim,Nz,K) log-rho
    # univariate R^2 per param, pooled over the low-z high-k cells
    P("-" * 78)
    P("TEST 1b: regress inter-sim log-rho on unit-cube theta (6 sims)")
    P("-" * 78)
    P("  per (z,k) cell, univariate R^2 of log-rho vs each single param, then")
    P("  averaged over LOW-z HIGH-k cells (which param drives any theta-dependence):")
    cells = [(zi, ki) for zi in np.where(lowz_mask_z)[0] for ki in np.where(hik_mask)[0]]
    uni_r2 = {p: [] for p in PARAM_NAMES}
    for (zi, ki) in cells:
        y = g_szk[:, zi, ki]
        if not np.all(np.isfinite(y)):
            continue
        yv = y - y.mean()
        sst = np.sum(yv ** 2)
        if sst < 1e-30:
            continue
        for pj, pn in enumerate(PARAM_NAMES):
            xj = sim_pu_c[:, pj]
            if np.sum(xj ** 2) < 1e-30:
                uni_r2[pn].append(0.0)
                continue
            beta = np.sum(xj * yv) / np.sum(xj ** 2)
            resid = yv - beta * xj
            uni_r2[pn].append(1.0 - np.sum(resid ** 2) / sst)
    P(f"  (cells used: {len(uni_r2['ns'])} of {len(cells)} low-z high-k)")
    rank = sorted(PARAM_NAMES, key=lambda p: -np.mean(uni_r2[p]) if uni_r2[p] else 0)
    for pn in rank:
        if uni_r2[pn]:
            P(f"    {pn:12s} mean univariate R^2 = {np.mean(uni_r2[pn]):.3f}  "
              f"(median {np.median(uni_r2[pn]):.3f})")
    P(f"  NOTE: with only 6 sims and 9 params, a FULL multivariate fit is rank-")
    P(f"  deficient (R^2->1 trivially, no DOF) -- univariate R^2 above is the honest")
    P(f"  signal. High univariate R^2 for a param just means rho CORRELATES with it")
    P(f"  across these 6 clustered sims; it cannot isolate causation from 6 points.")
    P("")

    # ===================================================================== #
    # TEST 1c: tau0 (alpha-rung) dependence of rho: rung 0 vs rung 19
    # ===================================================================== #
    P("-" * 78)
    P("TEST 1c: tau0 (alpha-rung) dependence of rho -- rung 0 vs rung 19")
    P("-" * 78)
    def rho_at_rung(a):
        sel = row_alpha == a
        with np.errstate(invalid="ignore"):
            r_zk = np.full((len(zvals), K), np.nan)
            for zi, zz in enumerate(zvals):
                s2 = sel & (row_z == zz)
                if s2.sum():
                    r_zk[zi] = np.nanmean(rho[s2], axis=0)
        return r_zk
    r0 = rho_at_rung(0)
    r19 = rho_at_rung(N_ALPHA - 1)
    P(f"  rung0  (alpha low)  mean rho all(z,k): {np.nanmean(r0):.4f} "
      f"({100*(np.nanmean(r0)-1):+.1f}%);  LOW-z HIGH-k: {np.nanmean(r0[lowz_hik]):.4f} "
      f"({100*(np.nanmean(r0[lowz_hik])-1):+.1f}%)")
    P(f"  rung19 (alpha high) mean rho all(z,k): {np.nanmean(r19):.4f} "
      f"({100*(np.nanmean(r19)-1):+.1f}%);  LOW-z HIGH-k: {np.nanmean(r19[lowz_hik]):.4f} "
      f"({100*(np.nanmean(r19[lowz_hik])-1):+.1f}%)")
    drung = np.nanmean(r0[lowz_hik]) - np.nanmean(r19[lowz_hik])
    P(f"  rung0 - rung19 (LOW-z HIGH-k) in rho:  {drung:+.4f}  "
      f"({100*drung:+.1f} percentage-points in P)")
    P(f"  => tau0 shifts the correction by ~{abs(100*drung):.1f}pp; the repo carries")
    P(f"     a per-(z,k) mean POOLED over alpha (gbar pools tau0). If this shift is")
    P(f"     non-negligible vs the n_s level, a tau0-resolved correction is warranted.")
    P("")

    # ===================================================================== #
    # TEST 2: 6-fold LOSO -- build gbar from 5 sims, predict the held-out HR
    # ===================================================================== #
    # gbar_{-h}(z,k) = mean over (the 5 train sims, all alpha) of g at z.
    # apply: logP_pred(h) = logP_LF(h) + gbar_{-h}(z,k); eps = exp(pred-logP_HR)-1.
    P("-" * 78)
    P("TEST 2: 6-fold LOSO  -- gbar from 5 sims applied to held-out HR's LF")
    P("-" * 78)
    P("  (mirrors FixedMeanHead/fixed_mean_table: per-z mean log-ratio, theta-indep;")
    P("   eps(z,k) = P_pred/P_HR - 1, the residual a FIXED correction leaves)")
    eps_all = []          # (rows, K) fractional error over all held-out rows
    eps_lowz_hik = []
    eps_by_z = {zz: [] for zz in zvals}
    # logPlf_row / logPhr_row are the per-row LF/HR truths on the reference grid
    # (computed up front). LOSO predicts logP_pred = logPlf_row + gbar_{-h}(z,k).

    for hsim in range(n_sim):
        train = row_sim != hsim
        test = row_sim == hsim
        # gbar_{-h}(z,k): per-z mean log-ratio over train rows
        gbar = np.full((len(zvals), K), np.nan)
        for zi, zz in enumerate(zvals):
            sel = train & (row_z == zz)
            if sel.sum():
                with np.errstate(invalid="ignore"):
                    gbar[zi] = np.nanmean(g[sel], axis=0)
        # predict held-out rows
        for m in np.where(test)[0]:
            zi = int(np.searchsorted(zvals, row_z[m]))
            zi = min(zi, len(zvals) - 1)
            # exact z match (zvals are the discrete grid)
            zi = int(np.argmin(np.abs(zvals - row_z[m])))
            logP_pred = logPlf_row[m] + gbar[zi]
            eps = np.exp(logP_pred - logPhr_row[m]) - 1.0
            eps_all.append(eps)
            eps_by_z[row_z[m]].append(eps)
            if row_z[m] <= LOWZ_MAX:
                eps_lowz_hik.append(eps[hik_mask])
    eps_all = np.array(eps_all)                          # (rows,K)
    eps_lowz_hik = np.concatenate(eps_lowz_hik) if eps_lowz_hik else np.array([])

    with np.errstate(invalid="ignore"):
        rms_all = np.sqrt(np.nanmean(eps_all ** 2))
        med_abs_all = np.nanmedian(np.abs(eps_all))
        max_abs_all = np.nanmax(np.abs(eps_all))
        rms_lh = np.sqrt(np.nanmean(eps_lowz_hik ** 2))
        med_lh = np.nanmedian(np.abs(eps_lowz_hik))
        max_lh = np.nanmax(np.abs(eps_lowz_hik))
        # coherent (mean, signed) error in low-z high-k -- the n_s-bias-relevant quantity
        mean_lh = np.nanmean(eps_lowz_hik)
    P(f"  LOSO eps pooled ALL (z,k):      RMS={100*rms_all:.2f}%  "
      f"median|eps|={100*med_abs_all:.2f}%  max|eps|={100*max_abs_all:.2f}%")
    P(f"  LOSO eps LOW-z HIGH-k:          RMS={100*rms_lh:.2f}%  "
      f"median|eps|={100*med_lh:.2f}%  max|eps|={100*max_lh:.2f}%")
    P(f"  LOSO eps LOW-z HIGH-k COHERENT (signed mean): {100*mean_lh:+.2f}%  "
      f"(this is the ~n_s-bias-relevant coherent logP residual)")
    P("")
    # per-held-out-sim coherent low-z high-k error. The POOLED signed mean (+0.01%)
    # cancels opposite-sign sims; the per-sim coherent error is the honest
    # n_s-bias-relevant quantity (a coherent logP offset for that held-out cosmology).
    P("  per-held-out-sim coherent (mean) low-z high-k eps:")
    persim_coh = []
    for hsim in range(n_sim):
        train = row_sim != hsim
        test = row_sim == hsim
        gbar = np.full((len(zvals), K), np.nan)
        for zi, zz in enumerate(zvals):
            sel = train & (row_z == zz)
            if sel.sum():
                with np.errstate(invalid="ignore"):
                    gbar[zi] = np.nanmean(g[sel], axis=0)
        ev = []
        for m in np.where(test)[0]:
            if row_z[m] > LOWZ_MAX:
                continue
            zi = int(np.argmin(np.abs(zvals - row_z[m])))
            eps = np.exp(logPlf_row[m] + gbar[zi] - logPhr_row[m]) - 1.0
            ev.append(eps[hik_mask])
        if ev:
            ev = np.concatenate(ev)
            with np.errstate(invalid="ignore"):
                coh = float(np.nanmean(ev))
                persim_coh.append(coh)
                P(f"    sim {hsim} (ns={ns_vals[hsim]:.3f}): coherent={100*coh:+.2f}%  "
                  f"RMS={100*np.sqrt(np.nanmean(ev**2)):.2f}%")
    persim_coh = np.array(persim_coh)
    worst_coh = persim_coh[np.argmax(np.abs(persim_coh))]
    P(f"  WORST per-sim coherent low-z high-k: {100*worst_coh:+.2f}%  "
      f"(spread {100*persim_coh.min():+.2f}% .. {100*persim_coh.max():+.2f}%)")
    P("  (the pooled signed mean cancels opposite-sign sims; the worst per-sim")
    P("   coherent offset is the honest n_s-bias-relevant residual.)")
    P("")

    # ===================================================================== #
    # VERDICT
    # ===================================================================== #
    P("=" * 78)
    P("VERDICT (numbers above):")
    P("-" * 78)
    corr_pct = 100 * (corr_lowz_hik - 1)
    P(f"  - the resolution correction itself is ~{corr_pct:+.1f}% in P (low-z high-k);")
    P(f"    i.e. the LF is ~{100*(1/corr_lowz_hik - 1):+.1f}% deficient there (matches the")
    P(f"    prior diag_lf_vs_hr_highk ~-6%).")
    P(f"  - inter-sim CoV(rho) low-z high-k median ~{100*np.nanmedian(cov_zk[lowz_hik]):.1f}% "
      f": the raw theta(+alpha)-spread of the correction across the 6 sims (SMALL).")
    P(f"  - LOSO: WORST per-sim coherent low-z high-k residual ~{100*worst_coh:+.2f}%, "
      f"per-sim RMS up to ~{100*rms_lh:.1f}%.")
    P(f"    Compare to: correction effect ~|{corr_pct:.1f}|%, and the n_s-bias level ~1% logP.")
    if abs(100 * worst_coh) < 1.0 and 100 * rms_lh < 0.5 * abs(corr_pct):
        P("  -> WORST per-sim coherent residual < 1% (the n_s level) and RMS well below")
        P("     the correction: over THIS 6-sim sub-volume the THETA-INDEPENDENT")
        P("     correction GENERALIZES well enough that the FIXED gbar does NOT itself")
        P("     re-introduce a coherent n_s-level bias. MF does not merely reshuffle it.")
    elif abs(100 * worst_coh) < 1.5 and 100 * rms_lh < abs(corr_pct):
        P("  -> WORST per-sim coherent residual is ~1-1.5% (BORDERLINE vs the ~1% n_s")
        P("     level) though still well below the |%.1f|%% correction. The fixed gbar" % corr_pct)
        P("     mostly removes the deficit but can leave a ~n_s-level coherent residual")
        P("     on an individual held-out cosmology -> MF substantially fixes, but does")
        P("     not provably ELIMINATE, the n_s bias inside this cluster; flag for the")
        P("     tau0-resolved correction + more HR sims at the edges.")
    else:
        P("  -> WORST per-sim coherent residual is comparable to the n_s level/correction:")
        P("     a fixed gbar leaves a theta-dependent coherent residual -> a theta-aware")
        P("     (or at least tau0-resolved) correction or more HR sims are needed.")
    P("")
    P("  COVERAGE CAVEAT (critical): the 6 HR sims span ns in [%.3f, %.3f] only," %
      (min(ns_vals), max(ns_vals)))
    P("  Ap %.2e-%.2e, clustered mid-high. This LOSO tests theta-insensitivity ONLY" %
      (min(ap_vals), max(ap_vals)))
    P("  over that sub-volume. LOW-ns and the sparse ns>0.98 / >1.0 edge are UNTESTED.")
    P("  With 6 points this is a WEAK (necessary, not sufficient) test: leave-one-out")
    P("  over 6 clustered sims cannot certify theta-insensitivity globally; it can only")
    P("  show whether a fixed correction self-consistently predicts a held-out sim")
    P("  INSIDE this cluster. Extrapolation to the prior edges is not validated here.")
    P("=" * 78)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {OUT_TXT}")

    # =============================== FIGURE ============================== #
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # pick representative z-bands
    zlo = zvals[np.argmin(np.abs(zvals - 2.4))]    # low z
    zmid = zvals[np.argmin(np.abs(zvals - 3.4))]
    zhi = zvals[np.argmin(np.abs(zvals - 4.4))]

    # row 0: rho(k) per sim overlaid, at 3 z-bands (inter-sim scatter = theta-sens)
    for ax, zz in zip(axes[0], [zlo, zmid, zhi]):
        zi = int(np.argmin(np.abs(zvals - zz)))
        for si, s in enumerate(sims):
            ax.plot(lf_k, rho_szk[si, zi], lw=1.2,
                    label=f"ns={ns_vals[si]:.3f}")
        ax.plot(lf_k, mean_zk[zi], "k--", lw=2.0, label="mean (gbar)")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.axvspan(lf_k[hik_mask].min(), lf_k.max(), color="orange", alpha=0.08)
        ax.set_xscale("log")
        ax.set_title(f"rho = P_HR/P_LF per sim   z={zz:.1f}")
        ax.set_xlabel("k [s/km]"); ax.set_ylabel("rho")
        ax.grid(alpha=0.3)
        if zz == zlo:
            ax.legend(fontsize=7, ncol=2)

    # row 1, col0: inter-sim CoV(rho) heatmap-ish: CoV vs k for a few z
    ax = axes[1, 0]
    for zz in [zlo, zmid, zhi]:
        zi = int(np.argmin(np.abs(zvals - zz)))
        ax.plot(lf_k, 100 * cov_zk[zi], label=f"z={zz:.1f}")
    ax.axvspan(lf_k[hik_mask].min(), lf_k.max(), color="orange", alpha=0.08)
    ax.set_xscale("log"); ax.set_xlabel("k [s/km]")
    ax.set_ylabel("inter-sim CoV(rho) [%]")
    ax.set_title("theta(+alpha)-spread of rho across 6 sims")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # row 1, col1: tau0 dependence rung0 vs rung19 (low-z high-k mean over z<=3)
    ax = axes[1, 1]
    with np.errstate(invalid="ignore"):
        r0z = np.nanmean(r0[lowz_mask_z], axis=0)
        r19z = np.nanmean(r19[lowz_mask_z], axis=0)
    ax.plot(lf_k, r0z, label="alpha rung 0 (low tau0)")
    ax.plot(lf_k, r19z, label="alpha rung 19 (high tau0)")
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.axvspan(lf_k[hik_mask].min(), lf_k.max(), color="orange", alpha=0.08)
    ax.set_xscale("log"); ax.set_xlabel("k [s/km]"); ax.set_ylabel("rho (z<=3 mean)")
    ax.set_title("tau0-dependence of the correction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # row 1, col2: LOSO eps vs k, low-z (per held-out sim coherent + band)
    ax = axes[1, 2]
    # recompute per-sim coherent eps(k) in low-z for plotting
    for hsim in range(n_sim):
        train = row_sim != hsim
        test = row_sim == hsim
        gbar = np.full((len(zvals), K), np.nan)
        for zi, zz in enumerate(zvals):
            sel = train & (row_z == zz)
            if sel.sum():
                with np.errstate(invalid="ignore"):
                    gbar[zi] = np.nanmean(g[sel], axis=0)
        epsk = []
        for m in np.where(test)[0]:
            if row_z[m] > LOWZ_MAX:
                continue
            zi = int(np.argmin(np.abs(zvals - row_z[m])))
            epsk.append(np.exp(logPlf_row[m] + gbar[zi] - logPhr_row[m]) - 1.0)
        if epsk:
            with np.errstate(invalid="ignore"):
                ek = 100 * np.nanmean(np.array(epsk), axis=0)
            ax.plot(lf_k, ek, lw=1.2, label=f"ns={ns_vals[hsim]:.3f}")
    ax.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax.axhline(1.0, color="red", lw=0.6, ls="--")
    ax.axhline(-1.0, color="red", lw=0.6, ls="--")
    ax.axvspan(lf_k[hik_mask].min(), lf_k.max(), color="orange", alpha=0.08)
    ax.set_xscale("log"); ax.set_xlabel("k [s/km]")
    ax.set_ylabel("LOSO eps [%] (z<=3 mean)")
    ax.set_title("6-fold LOSO error (red=+/-1%% n_s level)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("MF LF->HR resolution-correction: theta-sensitivity + 6-fold LOSO "
                 "(forward cache, ns in [%.2f,%.2f] only)" %
                 (min(ns_vals), max(ns_vals)), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[wrote] {OUT_PNG}")


if __name__ == "__main__":
    main()
