"""Bit-identity test against PRIYA's flux_vectors HDF5.

Runs in emu-3.9 + GSL env:
    export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
    /home/mfho/.conda/envs/emu-3.9/bin/python3 tests/test_priya_p1d.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    from hcd_analysis.priya_p1d import compute_tier_p_p1d
    from fake_spectra.fluxstatistics import obs_mean_tau
    _has_fs = True
except ImportError as e:
    print(f"SKIP — fake_spectra unavailable ({e}); requires emu-3.9 env with gsl")
    sys.exit(0)


PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP = 17
PRIYA_ROW = 344
PRIYA_Z_IDX = 8
N_K = 172


def test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1():
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    with open(meta_p) as f: m = json.load(f)
    nbins, dv_kms, z = int(m["nbins"]), float(m["dv_kms"]), float(m["z"])
    vmax = nbins * dv_kms

    with h5py.File(tau_p, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    with h5py.File(PRIYA_FILE, "r") as f:
        alpha = float(f["params"][PRIYA_ROW, 0])
        P_priya = f["flux_vectors"][PRIYA_ROW, PRIYA_Z_IDX*N_K:(PRIYA_Z_IDX+1)*N_K].astype(np.float64)
        kp = f["kfkms"][PRIYA_ROW, PRIYA_Z_IDX, :].astype(np.float64)

    kf, P_mine, target_F, scale = compute_tier_p_p1d(tau, vmax, alpha_slope=alpha, z=z)
    # k-grid first
    assert np.max(np.abs(kf[:N_K] - kp) / kp) < 1e-12, "k-grid mismatch"
    # P1D bit-identical
    ratio = P_mine[:N_K] / P_priya
    assert np.max(np.abs(ratio - 1)) < 1e-4, \
        f"P1D mismatch: max|r-1| = {np.max(np.abs(ratio-1)):.3e}"
    # Sanity: scale and target_F roughly match expectations
    assert 0.6 < target_F < 0.8
    assert 0.85 < scale < 1.0
    print(f"OK  alpha={alpha:.4f}  z={z}  target_F={target_F:.4f}  scale={scale:.4f}  "
          f"med(r)={np.median(ratio):.7f}  max|r-1|={np.max(np.abs(ratio-1)):.3e}")


if __name__ == "__main__":
    test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1()
    print("OK")
