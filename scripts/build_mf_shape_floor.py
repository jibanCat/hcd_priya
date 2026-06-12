#!/usr/bin/env python3
"""Build the SHAPE-AWARE MF C_emu floor (Phase-5a fix).

Phase-5a Test B showed the LF->HR MF resolution correction leaves a coherent *k-tilt*
residual (RMS ~1.5% low-z high-k) that biases n_s up to +2.8sigma in the NUTS closure,
and the existing DIAGONAL floor (sized to the ~0.9% coherent MEAN) does not envelope it
because a diagonal inflation is the wrong SHAPE for a coherent tilt.

This builds the natural fix: the per-held-out-sim LOSO eps OUTER-PRODUCT, a fractional
shape covariance

    f_shape[(z,k),(z',k')] = (1/N_sim) * sum_sim  eps_sim(z,k) * eps_sim(z',k')

whose DIAGONAL is the per-(z,k) mean-square eps (the RMS the diagonal floor undersized)
and whose OFF-DIAGONAL is the coherent tilt the n_s direction reads. On a leg this becomes
C_shape[i,j] = infl^2 * f_shape[i,j] * P_obs[i] * P_obs[j]  (fractional, scales with the model).

eps_sim(z,k) is the LOSO residual a FIXED gbar leaves on held-out sim h:
    eps = exp( logP_LF(h) + gbar_{-h}(z,k) - logP_HR(h) ) - 1     (alpha-pooled per z)
(identical construction to scripts/diag_mf_rescorr_loso.py TEST 2.)

We STORE the per-sim eps table on the cache (z,k) grid (so the leg binder can interpolate
onto each leg's exact k rows and form the outer product there) + a diagnostic of the
eigenspectrum / dominant mode. Output: hcd_analysis/_emulator_data/mf_cemu_shape.npz
(+ a diagnostic figure in the NOTES repo when --fig).

Usage:
    python3 scripts/build_mf_shape_floor.py            # build + save npz + diagnose
    python3 scripts/build_mf_shape_floor.py --fig      # also write the diagnostic figure
"""
from __future__ import annotations
import os, sys
import numpy as np
import h5py

sys.path.insert(0, "/home/mfho/hcd_priya")
from hcd_analysis.emulator.data import normalize_params  # noqa: E402

LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
OUT_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_shape.npz"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_multifidelity/mf_shape_floor_modes.png"

CLEAN = 0
K = 40
K_LO, K_HI = 0.01, 0.069          # LF Nyquist band (shared LF/HR support)
Z_MAX_FLOOR = 4.6                 # the leg binding never indexes z>4.6


def _strs(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def load_cache(fn):
    with h5py.File(fn, "r") as f:
        d = dict(P=f["P_tier_c_filtered"][:, CLEAN, :], kfkms=f["kfkms"][:],
                 z=np.round(f["z_grid"][:], 4), alpha=f["alpha_idx"][:].astype(int),
                 params=f["params"][:], sim=_strs(f["sim_name"][:]))
    d["pu"] = np.round(normalize_params(d["params"]), 6)
    return d


def rowkey(d):
    return [(tuple(d["pu"][i]), d["z"][i], d["alpha"][i]) for i in range(len(d["z"]))]


def measure_eps_per_sim():
    """Return (sims, zvals, lf_k, eps_sim[(n_sim, Nz, K)]) — the alpha-pooled LOSO eps."""
    lf, hr = load_cache(LF_CACHE), load_cache(HR_CACHE)
    lf_k = np.power(10.0, np.linspace(np.log10(K_LO), np.log10(K_HI), K))
    logk = np.log10(lf_k)
    lk = {k: i for i, k in enumerate(rowkey(lf))}
    hkeys = rowkey(hr)
    hr_rows, lf_rows = [], []
    for h, k in enumerate(hkeys):
        if k in lk:
            hr_rows.append(h); lf_rows.append(lk[k])
    hr_rows, lf_rows = np.array(hr_rows), np.array(lf_rows)
    sims = sorted(set(hr["sim"])); n_sim = len(sims)
    sidx = {s: i for i, s in enumerate(sims)}
    M = len(hr_rows)
    g = np.full((M, K), np.nan)
    logPlf = np.full((M, K), np.nan); logPhr = np.full((M, K), np.nan)
    row_sim = np.empty(M, int); row_z = np.empty(M)
    for m, (h, l) in enumerate(zip(hr_rows, lf_rows)):
        Phr, khr, Plf, klf = hr["P"][h], hr["kfkms"][h], lf["P"][l], lf["kfkms"][l]
        mhr = np.isfinite(Phr) & (Phr > 0) & np.isfinite(khr)
        mlf = np.isfinite(Plf) & (Plf > 0) & np.isfinite(klf)
        logPhr[m] = np.interp(logk, np.log10(khr[mhr]), np.log(Phr[mhr]), left=np.nan, right=np.nan)
        logPlf[m] = np.interp(logk, np.log10(klf[mlf]), np.log(Plf[mlf]), left=np.nan, right=np.nan)
        g[m] = logPhr[m] - logPlf[m]
        row_sim[m] = sidx[hr["sim"][h]]; row_z[m] = hr["z"][h]
    zvals = np.array(sorted(set(row_z)))
    # LOSO: gbar_{-h}(z,k) from the 5 train sims (alpha-pooled per z); eps on held-out rows,
    # then alpha-pool the held-out eps per (sim, z).
    eps_sim = np.full((n_sim, len(zvals), K), np.nan)
    for h in range(n_sim):
        train = row_sim != h
        gbar = np.full((len(zvals), K), np.nan)
        for zi, zz in enumerate(zvals):
            sel = train & (row_z == zz)
            if sel.sum():
                gbar[zi] = np.nanmean(g[sel], axis=0)
        for zi, zz in enumerate(zvals):
            sel = (row_sim == h) & (row_z == zz)
            if not sel.sum():
                continue
            eps_rows = np.exp(logPlf[sel] + gbar[zi] - logPhr[sel]) - 1.0   # (nrow, K)
            eps_sim[h, zi] = np.nanmean(eps_rows, axis=0)
    return sims, zvals, lf_k, eps_sim


def build():
    sims, zvals, lf_k, eps_sim = measure_eps_per_sim()
    zkeep = zvals <= Z_MAX_FLOOR
    zv = zvals[zkeep]
    eps = np.nan_to_num(eps_sim[:, zkeep, :], nan=0.0)        # (n_sim, Nz, K)
    n_sim, Nz, _ = eps.shape
    # flatten (z,k) z-major; second-moment shape covariance (NOT mean-subtracted: a floor
    # treats the residual as an unknown zero-mean systematic with this covariance).
    V = eps.reshape(n_sim, Nz * K)                            # (n_sim, Nz*K)
    f_shape = (V.T @ V) / n_sim                               # (Nz*K, Nz*K) fractional

    np.savez_compressed(OUT_NPZ,
                        sims=np.array(sims), z=zv, k=lf_k,
                        eps_sim=eps[:, :, :], f_shape=f_shape,
                        n_sim=n_sim, z_max=Z_MAX_FLOOR)
    print(f"wrote {OUT_NPZ}: eps_sim{eps.shape}, f_shape{f_shape.shape}, n_sim={n_sim}")
    return sims, zv, lf_k, eps, f_shape


def diagnose(sims, zv, lf_k, eps, f_shape):
    n_sim, Nz, Kk = eps.shape
    w, U = np.linalg.eigh(f_shape)
    w = w[::-1]; U = U[:, ::-1]
    frac = w / w.sum()
    print("\n=== shape-covariance eigenspectrum (fractional units) ===")
    print("  total variance (trace):", f"{np.trace(f_shape):.3e}")
    print("  top modes (eigval, %var, cumulative):")
    cum = 0.0
    for i in range(min(6, len(w))):
        cum += frac[i]
        print(f"    mode {i}: lam={w[i]:.3e}  {100*frac[i]:5.1f}%  cum {100*cum:5.1f}%")
    # diagonal RMS per (z,k) vs the current diagonal floor's ~0.9-1.5%
    diag = np.sqrt(np.clip(np.diag(f_shape), 0, None)).reshape(Nz, Kk)
    hik = lf_k >= np.quantile(lf_k, 0.75)
    print(f"\n  sqrt(diag) [per-(z,k) RMS eps]:  median={100*np.median(diag):.2f}%  "
          f"max={100*diag.max():.2f}%")
    print(f"  sqrt(diag) HIGH-k (top quartile): median={100*np.median(diag[:, hik]):.2f}%  "
          f"max={100*diag[:, hik].max():.2f}%")
    # cross-z coherence of the dominant mode: reshape mode0 to (Nz,K), check sign-coherence
    m0 = U[:, 0].reshape(Nz, Kk)
    # per-z high-k mean of mode0 (is the tilt the same sign across z?)
    m0_hik_byz = np.array([np.mean(m0[zi, hik]) for zi in range(Nz)])
    print(f"\n  dominant mode (mode0) high-k per-z mean signs: "
          f"{np.array2string(np.sign(m0_hik_byz).astype(int), separator=',')}")
    print(f"    -> {'COHERENT across z (cross-z blocks matter)' if np.all(m0_hik_byz>=0) or np.all(m0_hik_byz<=0) else 'mixed sign across z'}")
    # per-sim coherent high-k eps (cross-check vs mf_rescorr_loso.txt)
    print("\n  per-sim coherent low-z(<=2.6) high-k eps (cross-check):")
    lowz = zv <= 2.6
    for si, s in enumerate(sims):
        ns = s.split("Ap")[0].replace("ns", "")
        coh = 100 * np.mean(eps[si][np.ix_(lowz, hik)])
        rms = 100 * np.sqrt(np.mean(eps[si][np.ix_(lowz, hik)] ** 2))
        print(f"    ns={ns}: coherent={coh:+.2f}%  RMS={rms:.2f}%")
    return w, U, diag


def make_fig(sims, zv, lf_k, eps, f_shape, w, U, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Nz, Kk = diag.shape
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    # (a) per-sim eps vs k at a z where the residual is sizeable (the tilt)
    zi = int(np.argmax(diag.mean(axis=1)))               # the worst z (largest mean RMS)
    a = ax[0]
    for si, s in enumerate(sims):
        ns = s.split("Ap")[0].replace("ns", "")
        a.plot(lf_k, 100 * eps[si, zi], label=f"ns={ns}", lw=1.3)
    # dominant mode shape at this z (scaled by sqrt(lam0), sign for display)
    m0 = (U[:, 0] * np.sqrt(w[0])).reshape(Nz, Kk)[zi]
    a.plot(lf_k, 100 * m0, "k--", lw=2.0, label="mode0 √λ")
    a.axhline(0, color="k", lw=0.6); a.set_xscale("log")
    a.set_xlabel("k [s/km]"); a.set_ylabel("eps [%]")
    a.set_title(f"(a) per-sim LOSO eps tilt (z={zv[zi]:.2f})"); a.legend(fontsize=7)
    # (b) eigenspectrum
    b = ax[1]
    frac = w / w.sum()
    b.bar(range(len(w)), 100 * frac, color="C0")
    b.set_xlabel("mode"); b.set_ylabel("% of shape variance")
    b.set_title(f"(b) eigenspectrum (mode0={100*frac[0]:.0f}%)"); b.set_xlim(-0.5, 6.5)
    # (c) sqrt(diag) per (z,k): the RMS the shape floor now covers vs ~0.9% coherent
    c = ax[2]
    im = c.pcolormesh(lf_k, zv, 100 * diag, shading="auto", cmap="viridis")
    c.set_xscale("log"); c.set_xlabel("k [s/km]"); c.set_ylabel("z")
    c.set_title("(c) sqrt(diag f_shape) = RMS eps [%]"); fig.colorbar(im, ax=c)
    fig.suptitle("Shape-aware MF floor — the LOSO eps outer-product (fractional)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(NOTES_FIG), exist_ok=True)
    fig.savefig(NOTES_FIG, dpi=130)
    print("\nwrote", NOTES_FIG)


if __name__ == "__main__":
    s, zv, lfk, eps, fsh = build()
    w, U, diag = diagnose(s, zv, lfk, eps, fsh)
    if "--fig" in sys.argv:
        make_fig(s, zv, lfk, eps, fsh, w, U, diag)
