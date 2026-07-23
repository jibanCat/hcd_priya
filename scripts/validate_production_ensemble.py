"""Validate the PRODUCTION (all-sims) emulator ensemble: per-member + ensemble fit quality on the
held-out 10% ROW-val (in-distribution but unseen rows), + the inter-member (seed) scatter the ensemble
suppresses. Compares to the LOSO folds' in-range val RMS (1.0-2.4% overall, clean ~0.4-1.6%).

NOT a generalization test (all sims are in training — generalization is certified by the LOSO closure +
inherited via the C_emu). This is the fit-quality + ensemble-benefit gate.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/validate_production_ensemble.py
"""
import glob, os
import numpy as np
import hcd_analysis.emulator  # noqa: F401  x64
import jax, jax.numpy as jnp
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache, datarange_mask
from hcd_analysis.emulator.predict import predict_P_filt
from hcd_analysis.emulator.prod_ensemble import production_member_paths
from hcd_analysis.emulator.closure_legb import CACHE_PATH

VAL_SEED, VAL_FRAC = 12345, 0.10
CKPT = "/home/mfho/hcd_priya/checkpoints/final_prod_seed"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]


def member_pred(model, norm, d, idx):
    """(Nval,4,K) predicted P_filt for rows idx (vmap over rows)."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    def one(xi, ti):
        return predict_P_filt(model, xi[:9], xi[9], ti, pf)
    return np.asarray(jax.vmap(one)(x, tau0))


def main():
    d = load_cache(CACHE_PATH)
    n = d["P_tier_p"].shape[0]
    perm = np.random.default_rng(VAL_SEED).permutation(n)
    val_idx = np.sort(perm[:int(VAL_FRAC * n)])
    P_true = d["P_filt"][val_idx]                                   # (Nval,4,K)
    mask_k = d["mask"][val_idx]; inr = datarange_mask(d)[val_idx]   # (Nval,K)
    keep = mask_k & inr                                            # in-range finite

    # PINNED members (freeze decision 6): manifest-verified (sha256 + exact pairing + count +
    # stray-member tripwire) via checkpoints/production_ensemble_manifest.json, NOT a glob.
    members = [p + ".eqx" for p in production_member_paths(checkpoints_dir=os.path.dirname(CKPT))]
    print(f"members: {len(members)}  | val rows: {len(val_idx)}")
    preds = []
    for p in members:
        s = p.split("seed")[1].split(".")[0]
        model, meta, norm = T.load_checkpoint(p[:-4])
        pr = member_pred(model, norm, d, val_idx)
        preds.append(pr)
        # per-member in-range fractional RMS per class
        rms = []
        for c in range(4):
            r = (pr[:, c, :] - P_true[:, c, :]) / np.where(P_true[:, c, :] != 0, P_true[:, c, :], np.nan)
            r = np.where(keep & np.isfinite(r), r, np.nan)
            rms.append(np.sqrt(np.nanmean(r ** 2)))
        print(f"  seed {s}: in-range RMS  " + "  ".join(f"{CLASSES[c]} {100*rms[c]:.2f}%" for c in range(4)))
    P = np.stack(preds)                                            # (M,Nval,4,K)
    ens = P.mean(axis=0)                                           # ENSEMBLE = mean P_filt over members

    print("\n=== ENSEMBLE (mean of members) in-range fractional RMS ===")
    ens_rms = []
    for c in range(4):
        r = (ens[:, c, :] - P_true[:, c, :]) / np.where(P_true[:, c, :] != 0, P_true[:, c, :], np.nan)
        r = np.where(keep & np.isfinite(r), r, np.nan)
        ens_rms.append(np.sqrt(np.nanmean(r ** 2)))
    print("  " + "  ".join(f"{CLASSES[c]} {100*ens_rms[c]:.2f}%" for c in range(4)) +
          f"   [LOSO folds: clean ~0.4-1.6%, overall 1.0-2.4%]")

    # inter-member (seed) scatter the ensemble averages out: std across members / mean, in-range
    sc = []
    for c in range(4):
        rel = P[:, :, c, :].std(axis=0) / np.where(ens[:, c, :] != 0, ens[:, c, :], np.nan)
        rel = np.where(keep, rel, np.nan)
        sc.append(np.nanmedian(rel))
    print("\n=== inter-member (seed) scatter [median rel std across members] — the ensemble benefit ===")
    print("  " + "  ".join(f"{CLASSES[c]} {100*sc[c]:.2f}%" for c in range(4)) +
          f"   (ensemble of {len(members)} suppresses this √M≈{np.sqrt(len(members)):.1f}×)")
    print("\n  VERDICT: production ensemble fit "
          f"{'OK — in-range RMS comparable to the LOSO folds' if max(ens_rms) < 0.05 else 'CHECK — RMS high'}; "
          "generalization inherited from LOSO (+ C_emu).")


if __name__ == "__main__":
    main()
