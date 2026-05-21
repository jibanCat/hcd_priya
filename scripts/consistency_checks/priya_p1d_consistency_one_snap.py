"""Single-(sim, snap) PRIYA P1D consistency check, suitable for sbatch fan-out.

For one (sim_idx, snap, priya_z_idx): read tau, apply fake_spectra's filter,
loop over the 10 alpha rows PRIYA stored for sim_idx, compute fake_spectra
P1D, compare to PRIYA's stored row. Write one .npz with ratios for all 10
alphas.

Run in emu-3.9 env with GSL loaded:
    module load gcc/10.3.0 gsl/2.7
    conda activate emu-3.9
    python3 scripts/consistency_checks/priya_p1d_consistency_one_snap.py \\
        --sim-idx 44 --sim-folder ns0.803... --snap 17 --z-idx 8 \\
        --out docs/superpowers/figs/multipoint_sim44_snap17.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py

from fake_spectra.fluxstatistics import flux_power, obs_mean_tau
from fake_spectra.spectra import Spectra

_filter_single_tau_complex = Spectra._filter_single_tau_complex


PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
HCD_ROOT = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
EMU_ROOT = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
N_K = 172
N_ALPHA = 10
N_SIM = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-idx", type=int, required=True,
                    help="PRIYA sim index 0..59 (col within 10×60 alpha-major layout)")
    ap.add_argument("--sim-folder", type=str, required=True,
                    help="HCD output sim folder name (ns...)")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--z-idx", type=int, required=True,
                    help="PRIYA zout index (0..12)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"sim_idx={args.sim_idx}  sim={args.sim_folder}  snap={args.snap}  z_idx={args.z_idx}",
          flush=True)

    # Load PRIYA reference
    with h5py.File(PRIYA_FILE, "r") as f:
        priya_params = f["params"][...]
        priya_zout = f["zout"][...]
        priya_kfkms = f["kfkms"][...]
        priya_fv = f["flux_vectors"][...]
    priya_z = float(priya_zout[args.z_idx])

    # Load meta + tau
    meta_p = os.path.join(HCD_ROOT, args.sim_folder, f"snap_{args.snap:03d}", "meta.json")
    tau_p = os.path.join(EMU_ROOT, args.sim_folder, "output",
                         f"SPECTRA_{args.snap:03d}", "lya_forest_spectra_grid_480.hdf5")
    with open(meta_p) as fp:
        m = json.load(fp)
    nbins, dv_kms = int(m["nbins"]), float(m["dv_kms"])
    z_actual = float(m["z"])
    vmax = nbins * dv_kms
    assert abs(z_actual - priya_z) < 1e-2, f"z mismatch: meta={z_actual} priya={priya_z}"

    t0 = time.time()
    with h5py.File(tau_p, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    print(f"  loaded tau {tau.shape} in {time.time()-t0:.1f}s  vmax={vmax:.1f} km/s",
          flush=True)

    # Apply fake_spectra filter once
    tau_eff = -np.log(np.mean(np.exp(-tau)))
    ii = np.where(np.max(tau, axis=1) > 1e6)[0]
    t0 = time.time()
    self_stub = SimpleNamespace(nbins=nbins)
    for i in ii:
        tau[i], _ = _filter_single_tau_complex(self_stub, tau[i], tau_eff,
                                               tau_thresh=1e6, thresh2=0.25)
    print(f"  filtered {len(ii)} sightlines in {time.time()-t0:.1f}s  taueff={tau_eff:.4f}",
          flush=True)

    # Loop over the 10 PRIYA alphas for this sim_idx
    alpha_rows = [args.sim_idx + N_SIM * a for a in range(N_ALPHA)]
    alphas = np.array([float(priya_params[r, 0]) for r in alpha_rows])
    ratio_grid = np.full((N_ALPHA, N_K), np.nan)
    target_F_arr = np.zeros(N_ALPHA)
    medians = np.zeros(N_ALPHA)
    maxabs = np.zeros(N_ALPHA)
    stds = np.zeros(N_ALPHA)

    for a_idx, prow in enumerate(alpha_rows):
        alpha = float(priya_params[prow, 0])
        target_F = float(np.exp(-alpha * obs_mean_tau(z_actual)))
        target_F_arr[a_idx] = target_F
        t0 = time.time()
        kf, P = flux_power(tau, vmax, spec_res=0.0,
                           mean_flux_desired=target_F, window=False)
        kf, P = kf[1:], P[1:]
        # PRIYA's row
        P_priya = priya_fv[prow, args.z_idx*N_K:(args.z_idx+1)*N_K].astype(np.float64)
        kp = priya_kfkms[prow, args.z_idx, :].astype(np.float64)
        # k-grid alignment
        assert np.max(np.abs(kf[:N_K] - kp)/kp) < 1e-10
        r = P[:N_K] / P_priya
        ratio_grid[a_idx] = r
        medians[a_idx] = float(np.median(r))
        maxabs[a_idx] = float(np.max(np.abs(r - 1)))
        stds[a_idx] = float(np.std(r))
        print(f"  α={alpha:.4f} target_F={target_F:.4f} med={medians[a_idx]:.7f} "
              f"std={stds[a_idx]:.2e} max|r-1|={maxabs[a_idx]:.2e} ({time.time()-t0:.0f}s)",
              flush=True)

    # Save (also store k-grid for plotting)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             sim_idx=args.sim_idx, sim_folder=args.sim_folder,
             snap=args.snap, z_idx=args.z_idx, z=z_actual,
             alphas=alphas, target_F=target_F_arr,
             k_priya=priya_kfkms[alpha_rows[0], args.z_idx, :],
             ratio_grid=ratio_grid,
             medians=medians, stds=stds, maxabs=maxabs,
             nbins=nbins, dv_kms=dv_kms, vmax=vmax)
    print(f"\nSaved {args.out}", flush=True)
    print(f"summary: worst_max|r-1| = {maxabs.max():.3e}, worst_std = {stds.max():.3e}",
          flush=True)


if __name__ == "__main__":
    main()
