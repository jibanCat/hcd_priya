#!/usr/bin/env python3
"""Caveats on the LF-emulator k-coherence (Phase-5a, before building the 60-sim C_emu):

  C1 reconcile  — band-coupling corr(low-k coherent, high-k coherent) across sims (the
                  diag_emu_lowk_investigation PART-D quantity) + within-cell vs pooled coherence.
  C2 decompose  — split the dominant pooled k-mode into FLAT (A_p-like) + log-k TILT (n_s-like)
                  + curvature, to see how much is A_p-degenerate vs n_s-degenerate.
  C3 artifact   — within-fixed-z cross-k coherence (rule out a z-mixing pooling artifact).

Also CACHES the pooled per-(sim,z) coherent residual → mf_cemu_lfcoh_pool.npz for the build.

OUTPUT: figures/analysis/04_emulator/lf_emu_kcoherence_caveats.{png}, the npz cache, console.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import make_splits, load_cache
from hcd_analysis.emulator.closure_legb import CACHE_PATH
from scripts.build_xclass_error_vector import fold_rfrac

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/final_fold"
OUT_PNG = f"{REPO}/figures/analysis/04_emulator/lf_emu_kcoherence_caveats.png"
POOL_NPZ = f"{REPO}/hcd_analysis/_emulator_data/mf_cemu_lfcoh_pool.npz"
K_LO, K_HI = 0.01, 0.069
# Build the coherence over the FULL leg z-range (≤4.6), NOT just low-z — the emulator step-review
# (2026-06-12) flagged that zeroing z>2.6 could make C_emu over-confident where DESI has most of
# its leverage (He-II reion z≈3–4). We MEASURE the per-z within-z coherence across the full range
# (report C3 per z) and cache f over all z; LOWZ_REPORT is only the band for the C1/C2 prior-art
# comparison numbers.
LOWZ_MAX = 4.6        # build/cache the coherence table over all leg z (was 2.6)
LOWZ_REPORT = 2.6     # the low-z band used only for the C1/C2 prior-art-comparable summary
CLEAN = 0


def main():
    t0 = time.time()
    d = load_cache(CACHE_PATH)
    kf = np.asarray(d["kfkms"]); kcol = np.nanmedian(kf, axis=0)
    kband = (kcol >= K_LO) & (kcol <= K_HI)
    kb = kcol[kband]
    sim_all = np.array([s.decode() if isinstance(s, bytes) else s for s in d["sim_name"]])
    # pool clean rfrac + per-row (sim, z) across 8 folds
    R, zr, sr = [], [], []
    for f in range(8):
        model, meta, norm = T.load_checkpoint(f"{CKPT}{f}")
        _tr, va, _ho = make_splits(d, f, n_folds=8)
        rf = fold_rfrac(d, model, va, norm, datarange=True)[:, CLEAN, :]
        R.append(rf); zr.append(np.asarray(d["z_grid"])[va]); sr.append(sim_all[va])
        print(f"  fold {f}: {len(va)} rows ({time.time()-t0:.0f}s)")
    R = np.concatenate(R)[:, kband]                       # (Npool, Kb)
    zr = np.concatenate(zr); sr = np.concatenate(sr)
    sims = sorted(set(sr)); n_sim = len(sims)
    print(f"pooled {R.shape} clean rfrac, {n_sim} sims")

    def nan_corr(X):
        Xc = X - np.nanmean(X, 0, keepdims=True); n = X.shape[1]
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                m = np.isfinite(Xc[:, i]) & np.isfinite(Xc[:, j])
                if m.sum() > 5: C[i, j] = np.mean(Xc[m, i] * Xc[m, j])
        dg = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        return C, C / (dg[:, None] * dg[None, :])

    # ---- per-(sim,z) COHERENT residual (alpha/tau0-pooled) — the n_s-relevant object ----
    # mean over the sim's rows at each z (low-z band), → (n_sim, Nz_lowz, Kb)
    zlowz = sorted(set(zr[zr <= LOWZ_MAX]))
    coh = np.full((n_sim, len(zlowz), kb.size), np.nan)
    for si, s in enumerate(sims):
        for zi, zz in enumerate(zlowz):
            m = (sr == s) & (np.abs(zr - zz) < 1e-6)
            if m.sum(): coh[si, zi] = np.nanmean(R[m], 0)
    # ===== C3: within-fixed-z coherence (across sims), per low-z z-bin =====
    print("\n# C3 — within-fixed-z cross-k |corr| (across sims; rules out z-mixing artifact):")
    within = []
    for zi, zz in enumerate(zlowz):
        _, Rc = nan_corr(coh[:, zi, :])
        off = np.nanmedian(np.abs(Rc - np.diag(np.diag(Rc))))
        within.append(off); print(f"    z={zz:.2f}: median|off-diag corr| across {n_sim} sims = {off:.3f}")
    print(f"    => within-z median {np.nanmedian(within):.3f} (high ⇒ genuine, not a pooling artifact)")

    # ===== C1: band-coupling corr(low-k coherent, high-k coherent) across sims =====
    klo_m = kb <= 0.02; khi_m = kb >= 0.0442
    lo_co = np.array([np.nanmean(coh[si][:, klo_m]) for si in range(n_sim)])
    hi_co = np.array([np.nanmean(coh[si][:, khi_m]) for si in range(n_sim)])
    rr = np.corrcoef(lo_co, hi_co)[0, 1]
    print(f"\n# C1 — band-coupling: corr(low-k coherent, high-k coherent) across {n_sim} sims = {rr:+.3f}")
    print("    (the diag_emu_lowk_investigation PART-D quantity; nonzero ⇒ the global basis couples k = k-coherent)")

    # ===== C2: decompose the dominant pooled coherent k-mode into flat + tilt + curvature =====
    # build the per-(sim,z) coherent second-moment over the (z,k) flat vector, take mode0's k-shape.
    V = np.nan_to_num(coh).reshape(n_sim, len(zlowz) * kb.size)
    fsh = (V.T @ V) / n_sim
    w, U = np.linalg.eigh(fsh); U = U[:, ::-1]; w = w[::-1]
    mode0 = U[:, 0].reshape(len(zlowz), kb.size)
    vk = np.nanmean(mode0, 0)                              # low-z-averaged dominant k-mode
    # least-squares onto {1, centered log10 k}: flat (A_p-like) + tilt (n_s-like); residual = curvature
    x = np.log10(kb); x = x - x.mean()
    A = np.vstack([np.ones_like(x), x]).T
    ccoef, *_ = np.linalg.lstsq(A, vk, rcond=None)
    fit = A @ ccoef; resid = vk - fit
    tot = np.sum(vk**2)
    f_flat = (ccoef[0]**2 * len(vk)) / tot
    f_tilt = (ccoef[1]**2 * np.sum(x**2)) / tot
    f_curv = np.sum(resid**2) / tot
    print(f"\n# C2 — dominant coherent k-mode decomposition (fraction of mode variance):")
    print(f"    FLAT (A_p-like)   = {100*f_flat:5.1f}%")
    print(f"    log-k TILT (n_s)  = {100*f_tilt:5.1f}%")
    print(f"    curvature/resid   = {100*f_curv:5.1f}%")
    print(f"    => the coherent emu residual is {'TILT-dominated (n_s-degenerate)' if f_tilt>f_flat else 'FLAT-dominated (A_p-degenerate)'}")
    print(f"    mode0 captures {100*w[0]/w.sum():.0f}% of the per-(sim,z) coherent variance")

    np.savez_compressed(POOL_NPZ, coh=coh, sims=np.array(sims), z=np.array(zlowz), k=kb,
                        f_shape_lfcoh=fsh, band_coupling_r=rr)
    print(f"\nwrote {POOL_NPZ}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax[0].bar(range(len(zlowz)), within, color="C0"); ax[0].axhline(np.nanmedian(within), color="r", ls="--")
    ax[0].set_xticks(range(len(zlowz))); ax[0].set_xticklabels([f"{z:.1f}" for z in zlowz], fontsize=7)
    ax[0].set_xlabel("z"); ax[0].set_ylabel("median |off-diag corr|")
    ax[0].set_title(f"C3: within-fixed-z k-coherence (med {np.nanmedian(within):.2f})")
    ax[1].scatter(lo_co, hi_co, c="C4"); ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
    ax[1].set_xlabel("low-k coherent (per sim)"); ax[1].set_ylabel("high-k coherent")
    ax[1].set_title(f"C1: band coupling  r={rr:+.2f}")
    ax[2].plot(kb, vk, "o-", label="dominant coherent mode")
    ax[2].plot(kb, fit, "r--", label=f"flat+tilt fit (tilt {100*f_tilt:.0f}%)")
    ax[2].set_xscale("log"); ax[2].set_xlabel("k [s/km]"); ax[2].set_ylabel("mode")
    ax[2].set_title(f"C2: flat {100*f_flat:.0f}% / tilt {100*f_tilt:.0f}% / curv {100*f_curv:.0f}%")
    ax[2].legend(fontsize=8)
    fig.suptitle("LF-emulator k-coherence caveats: within-z (C3), band coupling (C1), amplitude-vs-tilt (C2)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120); print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
