#!/usr/bin/env python3
"""Is the LF EMULATOR residual k-coherent? (Phase-5a follow-up to the shape floor.)

The shape floor's off-diagonal mode is the LF→HR RESOLUTION residual, measurable only on the
6 HR sims (needs HR truth) → rank ≤ 6. The user's question: the LF EMULATOR residual is
measurable on all 60 LF sims (8-fold LOSO, no HR needed) — is IT k-coherent, and does its
k-mode resemble the resolution tilt? If so, the 60-sim set could better-estimate the
off-diagonal mode amplitude that the rank-6 HR set leaves to an empirical `infl`.

The production C_emu (`build_xclass_error_vector.py`) pooled these same 60-sim residuals but
reduced to a per-k 4×4 CROSS-CLASS block ("CS confirmed no k-correlation"). This RE-CHECKS the
k-coherence rigorously (cross-k correlation matrix + eigenspectrum) and compares to f_shape.

OUTPUT: figures/analysis/04_emulator/lf_emu_kcoherence.{png,txt} (code repo, foundational diag).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # noqa: F401  x64 before jax
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import make_splits, load_cache
from hcd_analysis.emulator.closure_legb import CACHE_PATH
from scripts.build_xclass_error_vector import fold_rfrac

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/final_fold"
OUT_PNG = f"{REPO}/figures/analysis/04_emulator/lf_emu_kcoherence.png"
SHAPE_NPZ = f"{REPO}/hcd_analysis/_emulator_data/mf_cemu_shape.npz"
K_LO, K_HI = 0.01, 0.069        # the LF Nyquist band (where the resolution tilt lives)
LOWZ_MAX = 2.6
CLEAN = 0                        # clean-forest class (the cosmology-bearing one)


def main():
    t0 = time.time()
    d = load_cache(CACHE_PATH)
    kf = np.asarray(d["kfkms"])           # (N,K) per-row k (s/km)
    z = np.asarray(d["z_grid"])
    # representative k-grid (rows share the cache angular-k binning up to cosmology rescale);
    # use the median per-column k for labeling + band masks.
    kcol = np.nanmedian(kf, axis=0)       # (K,)
    kband = (kcol >= K_LO) & (kcol <= K_HI)
    print(f"k cols in [{K_LO},{K_HI}]: {kband.sum()} of {kcol.size}")

    # --- pool the held-out clean-class fractional residual across 8 folds (all 60 sims) ---
    zrow = np.asarray(d["z_grid"])
    rfrac_pool, zrow_pool = [], []
    for f in range(8):
        model, meta, norm = T.load_checkpoint(f"{CKPT}{f}")
        _tr, va, _ho = make_splits(d, f, n_folds=8)
        rf = fold_rfrac(d, model, va, norm, datarange=True)   # (Nval,4,K)
        rfrac_pool.append(rf[:, CLEAN, :])                    # clean class (Nval,K)
        zrow_pool.append(np.asarray(d["z_grid"])[va])
        print(f"  fold {f}: {len(va)} rows ({time.time()-t0:.0f}s)")
    R = np.concatenate(rfrac_pool, axis=0)                    # (Npool,K) clean rfrac
    zr = np.concatenate(zrow_pool, axis=0)                    # (Npool,)
    # restrict to the low-z high-k resolution band
    lowz = zr <= LOWZ_MAX
    Rb = R[np.ix_(lowz, kband)]                               # (Nlowz, Kband)
    kb = kcol[kband]
    print(f"pooled clean rfrac in low-z band: {Rb.shape} rows×k")

    # --- cross-k covariance / correlation (pairwise-complete, NaN-safe) ---
    def nan_cov(X):
        Xc = X - np.nanmean(X, axis=0, keepdims=True)
        n = X.shape[1]
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                m = np.isfinite(Xc[:, i]) & np.isfinite(Xc[:, j])
                if m.sum() > 5:
                    C[i, j] = np.mean(Xc[m, i] * Xc[m, j])
        return C
    C_lf = nan_cov(Rb)
    dlf = np.sqrt(np.clip(np.diag(C_lf), 1e-300, None))
    Rcorr = C_lf / (dlf[:, None] * dlf[None, :])
    off = Rcorr - np.diag(np.diag(Rcorr))
    w = np.linalg.eigvalsh(np.nan_to_num(C_lf))[::-1]
    frac = w / w.sum()
    print("\n=== LF EMULATOR residual k-coherence (clean class, low-z high-k) ===")
    print(f"  effective independent samples ≈ 60 sims (rank ≫ the 6-HR resolution set)")
    print(f"  cross-k |correlation| median={np.nanmedian(np.abs(off)):.3f}  max={np.nanmax(np.abs(off)):.3f}")
    print(f"  eigenspectrum: mode0={100*frac[0]:.1f}%  mode0+1={100*(frac[0]+frac[1]):.1f}%  "
          f"(diagonal/k-incoherent ⇒ flat spectrum; coherent ⇒ one big mode)")
    diag_frac = np.nanmean(np.abs(np.diag(Rcorr)))   # ~1
    print(f"  mean |diag corr|={diag_frac:.2f} (sanity, ≈1)")

    # --- compare the LF dominant k-mode to the 6-HR RESOLUTION tilt mode ---
    sh = np.load(SHAPE_NPZ, allow_pickle=True)
    f_shape = sh["f_shape"]; zc = sh["z"]; kc = sh["k"]
    Nz, Kr = len(zc), len(kc)
    # resolution f_shape dominant eigvec, reshaped (Nz,Kr); take the low-z-averaged k-mode
    wv, U = np.linalg.eigh(f_shape); U = U[:, ::-1]
    res_mode_zk = U[:, 0].reshape(Nz, Kr)
    res_kmode = np.nanmean(res_mode_zk[zc <= LOWZ_MAX], axis=0)   # (Kr,) low-z k-mode
    # LF dominant k-mode on its band kb; interp the resolution k-mode onto kb for comparison
    U_lf = np.linalg.eigh(np.nan_to_num(C_lf))[1][:, ::-1]
    lf_kmode = U_lf[:, 0]
    res_on_kb = np.interp(np.log10(kb), np.log10(kc), res_kmode)
    # sign-free cosine similarity of the two dominant k-modes
    cos = abs(float(np.dot(lf_kmode, res_on_kb) /
                    (np.linalg.norm(lf_kmode) * np.linalg.norm(res_on_kb) + 1e-30)))
    print(f"\n  |cosine(LF dominant k-mode, RESOLUTION tilt k-mode)| = {cos:.2f}")
    print("   (high ⇒ same k-shape, 60-sim set could constrain the off-diagonal amplitude;")
    print("    low ⇒ DISTINCT residuals, the 6-HR set is the only handle on the resolution tilt)")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    im = ax[0].imshow(Rcorr, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    ax[0].set_title(f"(a) LF emu residual k-CORRELATION (60 sims)\nmedian|off|={np.nanmedian(np.abs(off)):.2f}")
    ax[0].set_xlabel("k bin"); ax[0].set_ylabel("k bin"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(range(min(8, len(frac))), 100 * frac[:8], color="C1")
    ax[1].set_title(f"(b) LF eigenspectrum (mode0={100*frac[0]:.0f}%)")
    ax[1].set_xlabel("mode"); ax[1].set_ylabel("% var")
    ax[2].plot(kb, lf_kmode / np.sign(lf_kmode[np.argmax(np.abs(lf_kmode))]), "o-", label="LF emu mode0")
    ax[2].plot(kb, res_on_kb / np.sign(res_on_kb[np.argmax(np.abs(res_on_kb))]), "s--", label="HR resolution tilt")
    ax[2].set_xscale("log"); ax[2].set_xlabel("k [s/km]"); ax[2].set_ylabel("mode (sign-normed)")
    ax[2].set_title(f"(c) dominant k-modes  |cos|={cos:.2f}"); ax[2].legend(fontsize=8)
    fig.suptitle("LF emulator residual k-coherence (60 sims) vs the HR resolution tilt (6 sims)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print("\nwrote", OUT_PNG)


if __name__ == "__main__":
    main()
