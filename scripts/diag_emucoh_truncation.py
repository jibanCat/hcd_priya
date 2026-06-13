#!/usr/bin/env python3
"""Truncation robustness (Bayesian referee, 2026-06-12): is the emucoh inference effect insensitive
to top-5 vs top-15 eigen-truncation? The inference enters ONLY through the on-DESI-leg fractional
covariance ctx.mf_emucoh_per_leg["DESI"] (= S·f·Sᵀ, then ×P_data⊗P_data common to both). So compare
that BOUND-ON-LEG matrix at top-5 vs top-15 via the PRODUCTION binding path (build_legb_ctx) — no
NUTS needed. Agreement to within MC noise ⇒ the production result does not depend on the truncation.
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import hcd_analysis.emulator  # noqa: F401  x64
from hcd_analysis.emulator.closure_legb import build_legb_ctx

POOL = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_lfcoh_pool.npz"
TOP15_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"   # production (top-15)


def trunc(f_full, m):
    # match build_mf_emucoh_floor.py: symmetrize, eigh, REVERSE to descending, clip, top-m.
    f = 0.5 * (f_full + f_full.T)
    w, U = np.linalg.eigh(f)
    w = w[::-1]; U = U[:, ::-1]               # eigh is ascending → flip to descending
    w = np.clip(w, 0.0, None)
    m = min(m, int(np.sum(w > 0)))
    fr = (U[:, :m] * w[:m]) @ U[:, :m].T
    return 0.5 * (fr + fr.T), float(w[:m].sum() / w.sum())


def leg_cov(npz_path):
    ctx, _ = build_legb_ctx(ckpt="/home/mfho/hcd_priya/checkpoints/final_fold6", with_mf=False,
                            mf_fold=6, mf_emucoh=True, mf_emucoh_legs=("DESI",), mf_emucoh_npz=npz_path)
    return np.asarray(ctx.mf_emucoh_per_leg["DESI"])


def main():
    d = np.load(POOL, allow_pickle=True)
    f_full = np.asarray(d["f_shape_lfcoh"], float)
    z = np.asarray(d["z"], float); k = np.asarray(d["k"], float)
    rank_pos = int(np.sum(np.clip(np.linalg.eigvalsh(0.5 * (f_full + f_full.T)), 0, None) > 0))
    print(f"pool f_full {f_full.shape}, positive rank {rank_pos}")

    tmp = tempfile.mkdtemp(prefix="emucoh_trunc_")

    def leg_for_m(m):
        fm, kept = trunc(f_full, m)
        p = os.path.join(tmp, f"emucoh_top{m}.npz")
        np.savez(p, z=z, k=k, f_shape=fm, n_sim=60, z_max=float(z.max()),
                 source="mf_cemu_lfcoh_pool.npz", clean_class_only=True)
        return leg_cov(p), kept

    MS = [5, 10, 15, 20, 30]
    Sfull, _ = leg_for_m(rank_pos)               # full-rank on-leg cov = the "truth"
    tr_full = float(np.trace(Sfull))
    print(f"\nConvergence of the ON-DESI-LEG C_emucoh vs the FULL-RANK leg cov (the quantity the fit sees):")
    print(f"  {'top-m':>6} {'f-trace%':>9} {'leg-trace/full':>14} {'Frob Δ vs full':>15} {'lead-3 overlap':>15}")
    wf, Uf = np.linalg.eigh(Sfull); topf = Uf[:, -3:]
    prev = None
    for m in MS:
        Sm, kept = leg_for_m(m)
        leg_tr = float(np.trace(Sm)) / tr_full
        fro = float(np.linalg.norm(Sm - Sfull) / np.linalg.norm(Sfull))
        proj = Uf  # overlap of top-m's leading dirs with full's
        wm, Um = np.linalg.eigh(Sm)
        ov = float(np.linalg.norm(Um[:, -3:] @ (Um[:, -3:].T @ topf)) / np.linalg.norm(topf))
        tag = ""
        if prev is not None:
            tag = f"  (Δ vs top-{prev[0]}: leg-trace {100*(leg_tr-prev[1]):+.1f}pp, Frob {100*(prev[2]-fro):+.1f}pp closer)"
        print(f"  {m:>6} {100*kept:>8.1f}% {leg_tr:>13.3f} {100*fro:>14.1f}% {ov:>15.3f}{tag}")
        prev = (m, leg_tr, fro)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("\nRead: top-15's leg-trace fraction + how much top-20/top-30 ADD over top-15 (small ⇒ top-15")
    print("converged ⇒ modes 16+ are noise; top-15 justified). Directions stable if overlap→1.")


if __name__ == "__main__":
    main()
