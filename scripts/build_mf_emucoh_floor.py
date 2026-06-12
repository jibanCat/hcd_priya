#!/usr/bin/env python3
"""Build the 60-sim LF-EMULATOR-COHERENCE C_emu term (Phase-5a, Q from PI 2026-06-12).

The production C_emu is DIAGONAL in (z,k) — it discards the LF emulator's k-COHERENT
generalization gap (jax-traps §24: clean low-z coherent 0.45% TRAIN vs 1.22% VAL, the
finite-60-sim LOSO sampling of the low-k Jacobian, the source of the per-fold A_p/n_s
scatter). The diagnostic `diag_lf_emu_kcoherence_caveats.py` measured this from all 60 LF
sims (8-fold LOSO, clean class, FULL leg-z z∈[2.0,4.4], k 0.01–0.069):
  - within-fixed-z cross-k |corr| 0.59–0.79 across ALL z (incl. He-II reion z≈3–4) —
    genuine (not a pooling artifact), so the term is built over the full z range (not just low-z);
  - bands decoupled (low↔high coherent corr ≈ 0 across sims);
  - full-z dominant mode = 53% log-k TILT (n_s) + 41% FLAT (A_p) + 6% curvature (tilt-dominated);
    the per-z structure raises the rank (mode0 27%), so top-15 (not top-5) is needed for ~97%.

This re-keys the cached `mf_cemu_lfcoh_pool.npz` into a leg-binder-shaped data product
`mf_cemu_emucoh.npz`, TRUNCATED to the top-m eigenmodes (drop the rank-tail sampling noise;
the discarded variance is conservatively re-folded via the likelihood's diagonal top-up).
It is consumed EXACTLY like the resolution shape floor (DL.load_mf_emucoh +
DL.mf_shape_cov_for_leg + the fixed-P_data on-leg amplitude) — a SECOND k-coherent off-
diagonal term in C_emu, complementary to the 6-HR resolution floor.

Design = the Bayesian+CS spec (2026-06-12): second-moment (uncentered; mean≈1.8% of trace),
full (z,k) vector (band-decoupling comes out naturally), clean-class only (first build),
FACE-VALUE amplitude (no empirical infl — 60 sims pin it), top-5 truncation.

OUTPUT: hcd_analysis/_emulator_data/mf_cemu_emucoh.npz (+ a diagnostic figure, NOTES repo).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya")

POOL_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_lfcoh_pool.npz"
OUT_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_multifidelity/mf_emucoh_modes.png"
N_MODES = 15                     # top-m truncation (full-z table, 13 z-bins: top-15 ≈ 96.8% of trace;
                                 # the per-z structure raises the rank vs the low-z-only view, where
                                 # top-5 sufficed. The discarded ~3% folds into the diagonal top-up.)


def build():
    d = np.load(POOL_NPZ, allow_pickle=True)
    f_full = np.asarray(d["f_shape_lfcoh"], float)      # (Nz·Nk, Nz·Nk) second moment, PSD
    z = np.asarray(d["z"], float); k = np.asarray(d["k"], float)
    coh = np.asarray(d["coh"], float)
    n_sim = int(coh.shape[0])
    band_r = float(d["band_coupling_r"]) if "band_coupling_r" in d else np.nan

    # symmetrize (float64 matmul roundoff) then top-m truncate.
    f_full = 0.5 * (f_full + f_full.T)
    w, U = np.linalg.eigh(f_full)
    w = w[::-1]; U = U[:, ::-1]
    w = np.clip(w, 0.0, None)                            # guard tiny negative eigenvalues
    m = min(N_MODES, np.sum(w > 0))
    f_trunc = (U[:, :m] * w[:m]) @ U[:, :m].T            # rank-m reconstruction (PSD)
    f_trunc = 0.5 * (f_trunc + f_trunc.T)
    kept = w[:m].sum() / w.sum()

    np.savez_compressed(
        OUT_NPZ, z=z, k=k, f_shape=f_trunc, n_sim=n_sim, z_max=float(z.max()),
        n_modes=m, kept_var_frac=kept, band_coupling_r=band_r,
        source="mf_cemu_lfcoh_pool.npz", clean_class_only=True)
    print(f"wrote {OUT_NPZ}")
    print(f"  f_shape {f_trunc.shape}  (Nz={len(z)} z∈[{z.min():.2f},{z.max():.2f}], Nk={len(k)})")
    print(f"  n_sim={n_sim}  top-{m} modes kept {100*kept:.1f}% of trace  (band-coupling r={band_r:+.2f})")
    print(f"  PSD min-eig of truncated f = {np.linalg.eigvalsh(f_trunc).min():.2e}")
    return z, k, f_trunc, w, U, m, n_sim


def make_fig(z, k, f_trunc, w, U, m, n_sim):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Nz, Nk = len(z), len(k)
    diag = np.sqrt(np.clip(np.diag(f_trunc), 0, None)).reshape(Nz, Nk)
    frac = w / w.sum()
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax[0].bar(range(min(8, len(w))), 100 * frac[:8], color="C1")
    ax[0].axvline(m - 0.5, color="r", ls="--", label=f"keep top-{m}")
    ax[0].set_xlabel("mode"); ax[0].set_ylabel("% var")
    ax[0].set_title(f"(a) eigenspectrum (mode0={100*frac[0]:.0f}%, top-{m}={100*frac[:m].sum():.0f}%)")
    ax[0].legend(fontsize=8)
    # dominant mode k-shapes per z
    for zi in range(Nz):
        v = U[:, 0].reshape(Nz, Nk)[zi]
        if np.any(v): ax[1].plot(k, v, label=f"z={z[zi]:.1f}")
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_xscale("log")
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("mode0 (per z)")
    ax[1].set_title("(b) dominant mode k-shape (flat→rising tilt)"); ax[1].legend(fontsize=7)
    im = ax[2].pcolormesh(k, z, 100 * diag, shading="auto", cmap="viridis")
    ax[2].set_xscale("log"); ax[2].set_xlabel("k [s/km]"); ax[2].set_ylabel("z")
    ax[2].set_title("(c) sqrt(diag f_trunc) = coherent RMS [%]"); fig.colorbar(im, ax=ax[2])
    fig.suptitle(f"60-sim LF-emulator-coherence C_emu term (clean class, top-{m} modes)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs(os.path.dirname(NOTES_FIG), exist_ok=True)
    fig.savefig(NOTES_FIG, dpi=120); print("wrote", NOTES_FIG)


if __name__ == "__main__":
    z, k, f_trunc, w, U, m, n_sim = build()
    if "--fig" in sys.argv:
        make_fig(z, k, f_trunc, w, U, m, n_sim)
