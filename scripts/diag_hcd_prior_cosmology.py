"""How much COSMOLOGY bias does the HCD incidence prior introduce? (PI verification)

The α_c incidence prior pulls α toward its center/width; through the α–θ degeneracy that
leaks into the cosmology posterior. This propagates, at a fiducial point (Fisher):
  J_θ = ∂P_obs/∂θ (9), J_α = ∂P_obs/∂α (3, = the excess templates);
  joint Fisher F over [θ,α] with a DESI-like diagonal cov; the α prior Λ=diag(1/σ_α²).
Reports:
  (1) BIAS from MIS-CENTERING — if the prior were centered on the SIM weight (the old wrong
      choice) instead of the observed incidence, the induced θ shift δθ = (F'⁻¹[0;Λ·Δμ])_θ,
      in σ_θ units (Δμ = sim − observed). Shows why centering on the observed matters.
  (2) CONSTRAINT change — σ_θ with the prior vs α-free (flat) vs α-fixed: how much the prior
      tightens/loosens cosmology (the broad subDLA vs tight LLS effect).
  (3) the α–θ Fisher correlation (which params each α_c is degenerate with).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_hcd_prior_cosmology.py
Writes figures/analysis/05_likelihood/hcd_prior_cosmology.png + .json.
"""
from __future__ import annotations

import json
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401  x64
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import inference as I
from hcd_analysis.emulator.predict import predict_P_obs

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
CKPT = str(ROOT / "checkpoints/decomp_nb24_fold0")
OUT = ROOT / "figures/analysis/05_likelihood"
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
HCD = ("LLS", "subDLA", "DLA")
SIGMA_FRAC = 0.05         # DESI-like per-mode fractional error on P_obs (illustrative)


def _ridge_inv(F):
    F = F + 1e-10 * np.trace(F) / F.shape[0] * np.eye(F.shape[0])
    return np.linalg.inv(F)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cache(CACHE)
    model, meta, norm = T.load_checkpoint(CKPT)
    pf = norm["P_filt"]
    z_fid = 3.0
    zsel = np.isclose(d["z_grid"], z_fid, atol=0.05)
    rows = np.where(zsel)[0]
    r0 = rows[np.argmin(np.abs(d["tau0"][rows] - np.median(d["tau0"][rows])))]
    z_unit = float(d["x"][r0, 9]); tau0 = float(d["tau0"][r0])
    w_c4 = np.asarray(d["w_c_cache"][r0]).reshape(4)
    w_hcd = jnp.asarray(w_c4[1:])                       # (3,) sim weights LLS,subDLA,DLA
    dla_core = jnp.asarray(d["delta"][r0, 2])           # DLA-core add-back (K,)
    kf = np.asarray(d["kfkms"][r0]); g = np.isfinite(kf)
    theta0 = jnp.full(9, 0.5)

    # observed-centered prior + the sim-centered alternative (the offset)
    alpha_obs, sig_alpha = I.hcd_incidence_prior(w_hcd, z=z_fid)        # observed center
    alpha_sim = jnp.stack([w_hcd[0], w_hcd[1], I.HCD_DLA_RESIDUAL_FRAC * w_hcd[2]])  # sim center
    alpha_fid = alpha_obs                                                # fiducial truth = observed

    # Jacobians at the fiducial
    f_theta = lambda th: predict_P_obs(model, th, z_unit, tau0, alpha_fid, pf, dla_core)
    f_alpha = lambda a: predict_P_obs(model, theta0, z_unit, tau0, a, pf, dla_core)
    P_obs = np.asarray(predict_P_obs(model, theta0, z_unit, tau0, alpha_fid, pf, dla_core))
    Jth = np.asarray(jax.jacfwd(f_theta)(theta0))[g]                    # (M,9)
    Jal = np.asarray(jax.jacfwd(f_alpha)(alpha_fid))[g]                 # (M,3)
    Cinv = 1.0 / (SIGMA_FRAC * np.abs(P_obs[g])) ** 2                   # (M,)

    # joint likelihood Fisher over [θ(9), α(3)]
    J = np.concatenate([Jth, Jal], axis=1)                             # (M,12)
    F = (J.T * Cinv) @ J                                                # (12,12)
    Lam = np.zeros((12, 12))
    Lam[9:, 9:] = np.diag(1.0 / np.asarray(sig_alpha) ** 2)            # α prior precision
    Finv_prior = _ridge_inv(F + Lam)
    Finv_free = _ridge_inv(F)                                          # α free (flat prior)
    Fthth_inv = _ridge_inv(F[:9, :9])                                  # α fixed
    sig_theta = np.sqrt(np.diag(Finv_prior))[:9]

    # (1) bias from centering on SIM instead of OBSERVED: prior pulls α by Δμ = sim − obs
    dmu = np.zeros(12); dmu[9:] = np.asarray(alpha_sim) - np.asarray(alpha_obs)
    dtheta_offset = (Finv_prior @ (Lam @ dmu))[:9]
    bias_sigma = dtheta_offset / sig_theta

    # (2) constraint change: σ_θ prior vs free vs fixed
    sig_free = np.sqrt(np.diag(Finv_free))[:9]
    sig_fixed = np.sqrt(np.diag(Fthth_inv))[:9]

    # (3) α–θ correlation (from the α-free joint inverse)
    s12 = np.sqrt(np.diag(Finv_free))
    corr = Finv_free / np.outer(s12, s12)
    corr_ath = corr[:9, 9:]                                            # (9,3)

    print("HCD-prior → cosmology bias (mis-centering on sim vs observed), in σ_θ:")
    for i, p in enumerate(PARAMS):
        print(f"  {p:11} bias={bias_sigma[i]:+.3f}σ   σ_θ(prior)/σ_θ(free)={sig_theta[i]/sig_free[i]:.3f}"
              f"   /σ_θ(fixed)={sig_theta[i]/max(sig_fixed[i],1e-30):.2f}")
    worst = PARAMS[int(np.argmax(np.abs(bias_sigma)))]
    print(f"  worst mis-centering bias: {worst} = {bias_sigma[np.argmax(np.abs(bias_sigma))]:+.3f}σ")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].bar(range(9), bias_sigma, color="tab:red")
    ax[0].set_xticks(range(9)); ax[0].set_xticklabels(PARAMS, rotation=90, fontsize=8)
    ax[0].axhline(0.2, color="k", ls=":", lw=1, label="0.2σ gate"); ax[0].axhline(-0.2, color="k", ls=":", lw=1)
    ax[0].set_ylabel("θ bias [σ_θ]"); ax[0].legend(fontsize=8)
    ax[0].set_title("(1) cosmology bias if prior centered on SIM not observed")
    rel = sig_theta / sig_free
    ax[1].bar(range(9), rel, color="tab:blue"); ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xticks(range(9)); ax[1].set_xticklabels(PARAMS, rotation=90, fontsize=8)
    ax[1].set_ylabel("σ_θ(prior) / σ_θ(α-free)")
    ax[1].set_title("(2) constraint change from the α prior\n(<1 = prior tightens θ)")
    im = ax[2].imshow(corr_ath, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax[2].set_yticks(range(9)); ax[2].set_yticklabels(PARAMS, fontsize=8)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(HCD, fontsize=8)
    ax[2].set_title("(3) α–θ correlation"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle("Does the HCD incidence prior bias cosmology? (z=3 fiducial, DESI-like cov)", fontsize=12)
    fig.tight_layout()
    fp = OUT / "hcd_prior_cosmology.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print("wrote", fp)
    json.dump({"bias_sigma": {PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
               "sigtheta_prior_over_free": {PARAMS[i]: float(rel[i]) for i in range(9)},
               "alpha_obs": np.asarray(alpha_obs).tolist(), "alpha_sim": np.asarray(alpha_sim).tolist(),
               "sigma_alpha": np.asarray(sig_alpha).tolist()},
              open(OUT / "hcd_prior_cosmology.json", "w"), indent=2)
    print("wrote", OUT / "hcd_prior_cosmology.json")


if __name__ == "__main__":
    main()
