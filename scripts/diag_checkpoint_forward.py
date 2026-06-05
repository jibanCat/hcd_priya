"""Phase-C checkpoint diagnostic — the CORRECTED HCD forward model + the C_emu budget.

A single multi-panel figure for the checkpoint review:
  (1) per-class excess templates R_c = P_c − P_clean (the corrected HCD object) — confirms
      Δ_LLS ≠ 0 now (the old P_c^unf−P_c^filt gave Δ_LLS≡0).
  (2) P_obs = P_clean + Σ α_c·R_c at α=0 (clean) / α=sim-w_c / α=obs-prior — the HCD
      contamination level on the total P1D.
  (3) the per-class fractional emulator error σ_c(k) at z=3 (the τ₀-aware error vector).
  (4) C_emu sub-dominance: diag(C_emu)^½ / P_obs vs k vs a DESI-like 5% data error
      (the field-standard "is the emulator error sub-dominant" check).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_checkpoint_forward.py
"""
from __future__ import annotations
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401  x64
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import inference as I
from hcd_analysis.emulator.predict import predict_P_filt, _excess_from_P_filt, predict_P_obs
from hcd_analysis.emulator.likelihood import sigma_at_tau0

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
CKPT = str(ROOT / "checkpoints/decomp_nb24_fold0")
OUT = ROOT / "figures/analysis/05_likelihood/checkpoint_forward.png"
CLS = ("LLS", "subDLA", "DLA")
DATA_FRAC = 0.05   # DESI-like per-mode fractional data error (illustrative)


def main():
    d = load_cache(CACHE)
    model, meta, norm = T.load_checkpoint(CKPT)
    pf = norm["P_filt"]
    ev = np.load(str(ROOT / "checkpoints/error_vector.npz"))
    sigma = ev["sigma"]; alpha_centres = jnp.asarray(ev["tau0_band_centres"])
    import sys; sys.path.insert(0, str(ROOT / "scripts"))
    from run_loso_sweep import make_z_bands
    zb_of, _ = make_z_bands(d["z_grid"], sigma.shape[2])

    z_fid = 3.0
    rows = np.where(np.isclose(d["z_grid"], z_fid, atol=0.05))[0]
    r0 = rows[np.argmin(np.abs(d["tau0"][rows] - np.median(d["tau0"][rows])))]
    z_unit = float(d["x"][r0, 9]); tau0 = float(d["tau0"][r0]); zb = int(zb_of[r0])
    w_c = np.asarray(d["w_c_cache"][r0]).reshape(4)
    dla_core = jnp.asarray(d["delta"][r0, 2])
    kf = np.asarray(d["kfkms"][r0]); g = np.isfinite(kf)
    theta0 = jnp.full(9, 0.5)

    P_filt = predict_P_filt(model, theta0, z_unit, tau0, pf)         # (4,K)
    P_clean = np.asarray(P_filt[0])
    excess = np.asarray(_excess_from_P_filt(P_filt, dla_core))       # (3,K) R_c
    # α references
    a_sim = jnp.asarray(w_c[1:])                                     # sim incidence
    a_obs, sig_a = I.hcd_incidence_prior(jnp.asarray(w_c[1:]), z=z_fid)  # observed-centered
    P_clean_only = np.asarray(predict_P_obs(model, theta0, z_unit, tau0, jnp.zeros(3), pf, dla_core))
    P_sim = np.asarray(predict_P_obs(model, theta0, z_unit, tau0, a_sim, pf, dla_core))
    P_obs = np.asarray(predict_P_obs(model, theta0, z_unit, tau0, a_obs, pf, dla_core))

    # per-class fractional σ at z=3 (τ₀-interp'd), and C_emu via the per-class coefficients
    sig_ck = np.asarray(sigma_at_tau0(jnp.asarray(sigma[:, :, zb, :]), alpha_centres, z_fid, tau0))  # (4,K)
    P_dla_unf = np.asarray(P_filt[3]) + np.asarray(dla_core)
    P_cls = np.stack([P_clean, np.asarray(P_filt[1]), np.asarray(P_filt[2]), P_dla_unf])  # (4,K)
    coef = np.concatenate([[1.0 - a_obs.sum()], np.asarray(a_obs)])  # [clean,LLS,sub,DLA]
    emu_var = np.einsum("c,ck,ck->k", coef**2, np.nan_to_num(sig_ck)**2, P_cls**2)
    sig_emu = np.sqrt(np.maximum(emu_var, 0))
    data_sig = DATA_FRAC * np.abs(P_obs)

    lls_rms = float(np.sqrt(np.nanmean((excess[0, g]) ** 2)))
    print(f"LLS excess RMS = {lls_rms:.3g}  (old broken template had Δ_LLS≡0)")
    print(f"obs α center @z3 = {np.asarray(a_obs)}  (sim w_c = {w_c[1:]})")
    frac = sig_emu[g] / np.maximum(np.abs(P_obs[g]), 1e-30)
    print(f"median emu frac error = {np.nanmedian(frac):.3%}; median C_emu/C_data = "
          f"{np.nanmedian((sig_emu[g]/np.maximum(data_sig[g],1e-30))**2):.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    for i, c in enumerate(CLS):
        a.semilogx(kf[g], excess[i, g], lw=1.8, label=f"{c}  (RMS {np.sqrt(np.nanmean(excess[i,g]**2)):.2g})")
    a.axhline(0, color="k", lw=0.6); a.set_title("(1) corrected excess R_c = P_c − P_clean (z=3)\n"
              "LLS now NON-zero (old P_c^unf−P_c^filt gave Δ_LLS≡0)")
    a.set_xlabel("k [s/km]"); a.set_ylabel("R_c [P1D units]"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.loglog(kf[g], P_clean_only[g], "k-", lw=2, label="clean (α=0)")
    a.loglog(kf[g], P_sim[g], "--", color="tab:orange", lw=1.8, label="α=sim w_c (P_tier_p)")
    a.loglog(kf[g], P_obs[g], "-", color="tab:red", lw=1.8, label="α=obs prior (data-like)")
    a.set_title("(2) P_obs = P_clean + Σ α_c·R_c — the HCD contamination level")
    a.set_xlabel("k [s/km]"); a.set_ylabel("P_obs"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1, 0]
    for i, c in enumerate(("clean",) + CLS):
        a.loglog(kf[g], np.nan_to_num(sig_ck[i, g]), lw=1.6, label=c)
    a.set_title("(3) per-class fractional emulator error σ_c(k) at z=3 (τ₀-aware)")
    a.set_xlabel("k [s/km]"); a.set_ylabel("σ_frac"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    a.loglog(kf[g], sig_emu[g], "-", color="tab:purple", lw=2, label="√diag(C_emu)")
    a.loglog(kf[g], data_sig[g], "--", color="tab:gray", lw=1.8, label="DESI-like 5% data error")
    a.set_title("(4) C_emu sub-dominance: emulator error vs data error\n"
                f"(median C_emu/C_data = {np.nanmedian((sig_emu[g]/np.maximum(data_sig[g],1e-30))**2):.2f})")
    a.set_xlabel("k [s/km]"); a.set_ylabel("σ (P1D units)"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    fig.suptitle("Phase-C checkpoint — corrected HCD forward model + the C_emu error budget (z=3)", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
