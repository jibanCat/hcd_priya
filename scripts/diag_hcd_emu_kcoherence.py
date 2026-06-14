#!/usr/bin/env python3
"""Gate A of the HCD-class coherent C_emu build (#9): is the per-HCD-class emulator residual
k-COHERENT (and cross-class coherent)? — the diagnose-first check, NO NUTS.

The production C_emu is DIAGONAL IN k for the HCD classes (the 4×4 cross-class ρ is per-k).
emucoh adds k-coherence for the CLEAN class only. The closure shows α_subDLA under-covers
(−1 to −3σ). TWO root causes were proposed: (i) a MISSING k-coherent HCD-class covariance term,
or (ii) a pure subDLA↔LLS↔DLA identifiability/prior-projection effect a covariance term can't fix.

This discriminates them at the SOURCE (no inference): pool the 8-fold/60-sim LOSO held-out
fractional residual for ALL 4 classes (fold_rfrac already returns (Nval,4,K)), build the per-(sim,z)
coherent vector per class (mean over the sim's rows at each (z,k), exactly the emucoh construction),
and measure:
  (A1) WITHIN-class cross-k coherence per class: the within-z (demeaned) cross-k correlation +
       eigenspectrum mode0% (clean was 0.59–0.79 / one big mode). If subDLA is ALSO k-coherent,
       the diagonal-in-k C_emu under-counts its true error → α_subDLA overconfident → under-coverage
       → the COVARIANCE hypothesis is VIABLE. If subDLA is k-INCOHERENT (flat spectrum, |off|≲0.2),
       the diagonal ρ already captures it → a coherent term WON'T help → it's identifiability → STOP.
  (A2) CROSS-class coherence: same-(z,k) 4×4 correlation (the clean↔HCD cancellation + the named
       subDLA↔LLS↔DLA coupling) — context for whether the clean class must stay in the covariance.

VERDICT printed + a figure (notes repo). PASS (build) iff subDLA shows cross-k |corr|≳0.4 AND a
dominant mode; FAIL (escalate to a prior/identifiability fix) otherwise.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_hcd_emu_kcoherence.py
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
OUT_PNG = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator/hcd_emu_kcoherence.png"
OUT_NPZ = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator/hcd_emu_kcoherence.npz"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]
HCD = [1, 2, 3]
K_LO, K_HI = 0.01, 0.069          # the LF band where the resolution/HCD coherence lives
COH_PASS = 0.40                   # cross-k |corr| threshold for "k-coherent" (clean was 0.59–0.79)


def within_z_crossk_corr(coh_c, z, k):
    """coh_c (n_sim, nz, nk) per-(sim,z) coherent residual for ONE class. Build the within-z
    DEMEANED cross-k correlation (subtract each z's across-sim mean k-shape so a z-trend can't
    masquerade as cross-k coherence — the 'genuine, not a pooling artifact' check), pooling the
    (sim,z) demeaned rows. Returns (Rcorr (nk,nk), median|off|, eigmode0_frac)."""
    n_sim, nz, nk = coh_c.shape
    rows = []
    for zi in range(nz):
        block = coh_c[:, zi, :]                                   # (n_sim, nk)
        mu = np.nanmean(block, axis=0, keepdims=True)            # across-sim mean k-shape at this z
        rows.append(block - mu)                                   # demeaned within z
    X = np.concatenate(rows, axis=0)                              # (n_sim*nz, nk)
    n = nk
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if m.sum() > 8:
                C[i, j] = np.mean(X[m, i] * X[m, j])
    dlf = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    Rcorr = C / (dlf[:, None] * dlf[None, :])
    off = Rcorr - np.diag(np.diag(Rcorr))
    # clip negatives: nan_to_num can zero a sparsely-paired off-diagonal → mildly indefinite C;
    # mode0% is a secondary readout (the verdict is median|off|), but keep the eigval ratio sane.
    w = np.clip(np.linalg.eigvalsh(np.nan_to_num(C))[::-1], 0.0, None)
    frac0 = float(w[0] / w.sum()) if w.sum() > 0 else np.nan
    return Rcorr, float(np.nanmedian(np.abs(off))), frac0


def main():
    t0 = time.time()
    d = load_cache(CACHE_PATH)
    kf = np.asarray(d["kfkms"])
    kcol = np.nanmedian(kf, axis=0)                               # (K,)
    kband = (kcol >= K_LO) & (kcol <= K_HI)
    kb = kcol[kband]
    zgrid = np.asarray(d["z_grid"])
    print(f"k cols in [{K_LO},{K_HI}]: {kband.sum()} of {kcol.size}")

    # --- pool the 8-fold/60-sim held-out fractional residual for ALL 4 classes ---
    rfrac_pool, zrow_pool, sim_pool = [], [], []
    sim_name = np.asarray(d["sim_name"])
    for f in range(8):
        model, meta, norm = T.load_checkpoint(f"{CKPT}{f}")
        _tr, va, _ho = make_splits(d, f, n_folds=8)
        rf = fold_rfrac(d, model, va, norm, datarange=True)       # (Nval,4,K)
        rfrac_pool.append(rf[:, :, kband])                        # (Nval,4,Kband)
        zrow_pool.append(zgrid[va])
        sim_pool.append(sim_name[va])
        print(f"  fold {f}: {len(va)} rows ({time.time()-t0:.0f}s)")
    R = np.concatenate(rfrac_pool, axis=0)                        # (Npool,4,Kband)
    zr = np.concatenate(zrow_pool, axis=0)                        # (Npool,)
    sr = np.concatenate(sim_pool, axis=0)                         # (Npool,)
    zu = np.unique(zr)
    sims = np.unique(sr)
    nk = kb.size
    print(f"pooled residual: {R.shape}  ({len(sims)} sims, {len(zu)} z)")

    # --- per-(sim,z) coherent vector per class: coh[c] (n_sim, nz, nk) ---
    coh = np.full((4, len(sims), len(zu), nk), np.nan)
    for ci in range(4):
        for si, s in enumerate(sims):
            for zi, zz in enumerate(zu):
                m = (sr == s) & (np.abs(zr - zz) < 1e-6)
                if m.any():
                    coh[ci, si, zi] = np.nanmean(R[m, ci, :], axis=0)

    # --- (A1) within-class cross-k coherence per class ---
    print("\n=== (A1) WITHIN-class cross-k coherence (clean = reference; subDLA = the decisive one) ===")
    Rcorrs, results = {}, {}
    for ci, nm in enumerate(CLASSES):
        Rcorr, medoff, frac0 = within_z_crossk_corr(coh[ci], zu, kb)
        Rcorrs[nm] = Rcorr
        results[nm] = (medoff, frac0)
        flag = "k-COHERENT" if medoff >= COH_PASS else "k-incoherent"
        print(f"  {nm:7s}: cross-k |corr| median={medoff:.3f}   mode0={100*frac0:4.1f}%   → {flag}")

    # --- (A2) same-(z,k) cross-class correlation (the clean↔HCD + subDLA↔LLS↔DLA coupling) ---
    # at each (z,k) cell, correlation across sims of (ε_clean,ε_LLS,ε_subDLA,ε_DLA); average |.|.
    print("\n=== (A2) same-(z,k) CROSS-class correlation (avg over z,k cells) ===")
    xcorr = np.full((4, 4), np.nan)
    for a in range(4):
        for b in range(4):
            vals = []
            for zi in range(len(zu)):
                for ki in range(nk):
                    xa, xb = coh[a, :, zi, ki], coh[b, :, zi, ki]
                    m = np.isfinite(xa) & np.isfinite(xb)
                    if m.sum() > 8:
                        sa, sb = xa[m].std(), xb[m].std()
                        if sa > 0 and sb > 0:
                            vals.append(np.mean((xa[m] - xa[m].mean()) * (xb[m] - xb[m].mean())) / (sa * sb))
            if vals:
                xcorr[a, b] = np.mean(vals)
    hdr = "        " + "".join(f"{n:>8s}" for n in CLASSES)
    print(hdr)
    for a, nm in enumerate(CLASSES):
        print(f"  {nm:6s}" + "".join(f"{xcorr[a, b]:+8.2f}" for b in range(4)))

    # --- VERDICT ---
    sub_off, sub_mode = results["subDLA"]
    sub_coherent = sub_off >= COH_PASS
    lls_off, _ = results["LLS"]
    dla_off, _ = results["DLA"]
    any_hcd_coherent = max(sub_off, lls_off, dla_off) >= COH_PASS
    cleanhcd = np.nanmean([abs(xcorr[0, c]) for c in HCD])
    subhcd = np.nanmean([abs(xcorr[2, c]) for c in (1, 3)])
    print("\n=== GATE A VERDICT ===")
    print(f"  subDLA cross-k |corr|={sub_off:.3f} (threshold {COH_PASS}) → "
          f"{'COHERENT' if sub_coherent else 'INCOHERENT'}")
    print(f"  clean↔HCD same-k |corr|≈{cleanhcd:.2f} (cancellation channel → keep clean in the Gram)")
    print(f"  subDLA↔(LLS,DLA) same-k |corr|≈{subhcd:.2f} (the named cross-class coupling)")
    if sub_coherent:
        print("  ✅ PASS: the subDLA-class residual IS k-coherent → the diagonal-in-k C_emu under-counts")
        print("     its true error → α_subDLA overconfident. The covariance hypothesis is VIABLE → BUILD.")
        print("     (Gate B next: confirm the term's GLS-Fisher σ_subDLA widening is the right SIZE.)")
    elif any_hcd_coherent:
        print("  ⚠️  MIXED: subDLA itself is not k-coherent but another HCD class is — re-scope the term.")
    else:
        print("  ❌ FAIL: HCD-class residuals are k-INCOHERENT → the diagonal ρ already captures them.")
        print("     A coherent term won't move α_subDLA coverage → escalate to a PRIOR/identifiability fix.")

    # --- figure ---
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ci, nm in enumerate(CLASSES):
        ax = axes[0, ci]
        im = ax.imshow(Rcorrs[nm], vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        medoff, frac0 = results[nm]
        ax.set_title(f"{nm}: cross-k corr\nmed|off|={medoff:.2f}  mode0={100*frac0:.0f}%",
                     fontsize=10, color=("C3" if (nm == "subDLA" and medoff >= COH_PASS) else "k"))
        ax.set_xlabel("k bin"); ax.set_ylabel("k bin")
        fig.colorbar(im, ax=ax, fraction=0.046)
    # bottom: per-class diag RMS(k), eigenspectra, the cross-class 4×4, and the dominant subDLA k-mode
    axd = axes[1, 0]
    for ci, nm in enumerate(CLASSES):
        rms = np.sqrt(np.clip(np.nanmean(coh[ci] ** 2, axis=(0, 1)), 0, None)) * 100
        axd.plot(kb, rms, "o-", ms=3, label=nm)
    axd.set_xscale("log"); axd.set_xlabel("k [s/km]"); axd.set_ylabel("coherent RMS [%]")
    axd.set_title("(e) per-class coherent RMS(k)"); axd.legend(fontsize=8)
    axe = axes[1, 1]
    for ci, nm in enumerate(CLASSES):
        w = np.clip(np.linalg.eigvalsh(np.nan_to_num(Rcorrs[nm]))[::-1], 0.0, None)
        axe.plot(range(min(8, len(w))), 100 * w[:8] / w.sum(), "o-", ms=3, label=nm)
    axe.set_xlabel("mode"); axe.set_ylabel("% var"); axe.set_title("(f) cross-k eigenspectra")
    axe.legend(fontsize=8)
    axx = axes[1, 2]
    im = axx.imshow(np.nan_to_num(xcorr), vmin=-1, vmax=1, cmap="RdBu_r")
    axx.set_xticks(range(4)); axx.set_xticklabels(CLASSES, rotation=45, fontsize=8)
    axx.set_yticks(range(4)); axx.set_yticklabels(CLASSES, fontsize=8)
    for a in range(4):
        for b in range(4):
            axx.text(b, a, f"{xcorr[a, b]:+.2f}", ha="center", va="center", fontsize=8)
    axx.set_title("(g) same-(z,k) cross-class corr")
    axm = axes[1, 3]
    Usub = np.linalg.eigh(np.nan_to_num(Rcorrs["subDLA"]))[1][:, ::-1]
    Ucln = np.linalg.eigh(np.nan_to_num(Rcorrs["clean"]))[1][:, ::-1]
    axm.plot(kb, Usub[:, 0] / np.sign(Usub[np.argmax(np.abs(Usub[:, 0])), 0]), "o-", ms=3, label="subDLA mode0")
    axm.plot(kb, Ucln[:, 0] / np.sign(Ucln[np.argmax(np.abs(Ucln[:, 0])), 0]), "s--", ms=3, label="clean mode0")
    axm.axhline(0, color="k", lw=.5); axm.set_xscale("log")
    axm.set_xlabel("k [s/km]"); axm.set_ylabel("mode (sign-normed)")
    axm.set_title("(h) dominant k-mode shapes"); axm.legend(fontsize=8)
    fig.suptitle("Gate A — HCD-class emulator residual k-coherence (8-fold/60-sim LOSO)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nwrote {OUT_PNG}")
    np.savez_compressed(OUT_NPZ, kb=kb, zu=zu, coh=coh, xcorr=xcorr,
                        results={k: results[k] for k in CLASSES}, classes=np.array(CLASSES))
    print(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
