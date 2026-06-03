"""Post-recipe production-checkpoint verification (Phase-2b productionization).

Loads the PRODUCTION ``checkpoints/walkthrough_fold0`` (trained by train_fold with the
productionized recipe: deep θ-blind BaselineHead pre-fit to its floor + frozen, the
Σ(weight·mask) joint loss, hard low-rank residual head) and confirms — on the SAME
LF fold-0 LOSO val split the model was validated against — that the recipe delivers:

  1. DEPLOYED median |P̂/P−1| < 1% per class (clean/LLS/subDLA), and p95 (LF k-range).
  2. LF Fisher-projected emulator-error bias < 0.2σ on ns/Ap (reusing the
     scripts/diag_tilt_bias_lf_hr.py fisher_bias logic, but on the PRODUCTION model's
     residual path: J = σ_cosmo·∂(model P_filt_resid)/∂θ via jax.jacfwd; the baseline
     is θ-blind so ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ exactly).

This is the deployed-model check that complements diag_theta_tracking_honest.py
(term (a)/(b) decomposition). The JAX-trap caveat (jax-traps-log #: degenerate J on
few-sim suites) is reported alongside (abs unit-cube shift + Fisher cond number).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/verify_production_recipe.py
"""
from __future__ import annotations
import json

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import numpy as np
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import load_cache, make_splits, safe_log
from hcd_analysis.emulator.train import load_checkpoint

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CKPT = "checkpoints/walkthrough_fold0"
OUT = "figures/analysis/04_emulator/production_recipe_verify.json"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
# DESI-DR1-like diagonal per-mode covariance (same assumption as diag_tilt_bias_lf_hr).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}


def main():
    d = load_cache(CACHE)
    model, meta, norm = load_checkpoint(CKPT)
    pf = norm["P_filt"]
    sig_marg, mu_marg, sig_cosmo = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    fold = int(meta.get("fold", 0))
    nfold = 8
    tr, va, ho = make_splits(d, fold, nfold)
    n_k = int(meta.get("n_k") or d["P_tier_p"].shape[1])

    # ---- deployed reconstruction on val rows ----------------------------------
    Xva = jnp.asarray(d["x"][va])
    tau_va = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(Xva, tau_va)
    base = np.asarray(pred["P_filt_base"])           # (n,4,K) standardized m̂
    resid = np.asarray(pred["P_filt_resid"])         # (n,4,K) standardized r̂
    logP_hat = (base * sig_marg + mu_marg) + sig_cosmo * resid
    logP_true = safe_log(d["P_filt"][va])
    mask = np.isfinite(logP_true)
    frac = np.exp(logP_hat - logP_true) - 1.0

    print(f"fold={fold}/{nfold}  n_val={len(va)}  n_k={n_k}")
    print("\n=== DEPLOYED |P̂/P−1| on LF fold-0 val (full LF k-range) ===")
    dep = {}
    for ci, nm in enumerate(CLS):
        f = frac[:, ci, :][mask[:, ci, :]]
        dep[nm] = dict(median=float(np.median(np.abs(f))),
                       rms=float(np.sqrt(np.mean(f ** 2))),
                       p95=float(np.percentile(np.abs(f), 95)))
        print(f"  {nm:7}: median={dep[nm]['median']*100:.3f}%  "
              f"rms={dep[nm]['rms']*100:.3f}%  p95={dep[nm]['p95']*100:.3f}%")

    # ---- Fisher bias on the PRODUCTION model (z=3 fiducial, cube centre) -------
    z_fid = 3.0
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - z_fid))], atol=1e-6)
    tau0_fid = float(np.median(d["tau0"][zsel]))
    z_unit_fid = float(np.median(d["x"][zsel, 9]))
    theta0 = jnp.full(9, 0.5, dtype=jnp.float64)

    def rhat_of_theta(theta9):
        x = jnp.concatenate([theta9, jnp.asarray([z_unit_fid])])
        return model(x, jnp.asarray(tau0_fid))["P_filt_resid"]   # (4, n_k)

    Jr = np.asarray(jax.jacfwd(rhat_of_theta)(theta0))           # (4,n_k,9)
    J_logP = Jr * sig_cosmo[:, :, None]                          # ∂logP̂/∂θ

    # δ = deployed log error at the fiducial z-slice, per (class,k)
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    dlog = logP_hat[vz] - logP_true[vz]
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(mask[vz][:, ci, j], dlog[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])
    kvalid = (mask[vz].mean(0) > 0.5)                            # (4,K)

    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not (kvalid[ci, j] and np.all(np.isfinite(J_logP[ci, j]))
                    and np.isfinite(delta[ci, j])):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    J = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    F = (J.T * Cinv) @ J
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    cond = float(np.linalg.cond(F + ridge * np.eye(9)))

    print(f"\n=== Fisher-bias (production model, z={z_fid}, cube centre) ===")
    print(f"  {len(dv)} modes  Fisher cond={cond:.2e}")
    for i, p in enumerate(PARAMS):
        flag = "  <-- ns/Ap" if i < 2 else ""
        print(f"  {p:11}: bias={bias_sigma[i]:+.3f}σ  "
              f"(σ_Fisher={sigma[i]:.4f}, δθ_unit={dtheta[i]:+.4e}){flag}")

    summary = {
        "fold": fold, "n_val": int(len(va)), "n_k": n_k,
        "deployed_frac": dep,
        "fisher_bias": {
            "z_fid": z_fid, "tau0_fid": tau0_fid, "n_modes": int(len(dv)),
            "fisher_cond": cond,
            "bias_in_sigma": {PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
            "sigma_fisher": {PARAMS[i]: float(sigma[i]) for i in range(9)},
            "dtheta_unit": {PARAMS[i]: float(dtheta[i]) for i in range(9)},
        },
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
