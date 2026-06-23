"""PRIYA P1D consistency check v3 — use fake_spectra's actual filter, not my port.

v2 ported _filter_single_tau_complex into pure Python. The port truncated walks
at the box boundary; fake_spectra uses Python negative indexing so walks wrap
periodically. That subtle difference left a residual ~0.1 % slope in the
v2 ratio and a 1.3 % spike at mode m=1.

v3: import _filter_single_tau_complex from fake_spectra directly as an unbound
method and call it via a SimpleNamespace stand-in for `self` (it only reads
`self.nbins`). Everything else (mean-flux inversion, flux_power) is unchanged.

Run in emu-3.9 env with GSL loaded:
    module load gcc/10.3.0 gsl/2.7
    conda activate emu-3.9
    python3 scripts/consistency_checks/priya_p1d_consistency_v3.py

Outputs:
    hcd_priya_notes/docs/superpowers/figs/2026-05-20-priya-p1d-consistency-v3.npz
"""
import os, sys, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import h5py

# fake_spectra imports (emu-3.9 + GSL needed)
from fake_spectra.fluxstatistics import flux_power, mean_flux, obs_mean_tau
from fake_spectra._spectra_priv import _rescale_mean_flux
from fake_spectra.spectra import Spectra
# unbound method — only reads self.nbins
_filter_single_tau_complex = Spectra._filter_single_tau_complex


# --------------------------------------------------------------------- INPUTS
PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP = 17
TAU_HDF5 = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
META = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"

PRIYA_ROW = 344
PRIYA_Z_IDX = 8  # zout[8] = 3.0
N_K_PRIYA = 172


# ===================================================== Load PRIYA reference
with h5py.File(PRIYA_FILE, "r") as f:
    priya_alpha = float(f["params"][PRIYA_ROW, 0])
    priya_z = float(f["zout"][PRIYA_Z_IDX])
    priya_kfkms = f["kfkms"][PRIYA_ROW, PRIYA_Z_IDX, :].astype(np.float64)
    fv_flat = f["flux_vectors"][PRIYA_ROW, :].astype(np.float64)
    P_priya = fv_flat[PRIYA_Z_IDX * N_K_PRIYA:(PRIYA_Z_IDX + 1) * N_K_PRIYA]
print(f"PRIYA row {PRIYA_ROW}: alpha={priya_alpha:.6f}, z={priya_z}")


# ===================================================== Load tau + meta
with open(META) as fp:
    m = json.load(fp)
nbins = int(m["nbins"])
dv_kms = float(m["dv_kms"])
assert abs(float(m["z"]) - priya_z) < 1e-3
vmax = nbins * dv_kms

t0 = time.time()
with h5py.File(TAU_HDF5, "r") as f:
    tau = f["tau/H/1/1215"][...].astype(np.float64)
print(f"loaded tau {tau.shape} in {time.time()-t0:.1f}s; vmax={vmax:.2f} km/s")


# ============================== Apply fake_spectra's actual _filter_single_tau
print("\n=== Filter using fake_spectra._filter_single_tau_complex (periodic BC) ===")
TAU_THRESH = 1.0e6
THRESH2 = 0.25
mean_F_pre = np.mean(np.exp(-tau))
taueff = -np.log(mean_F_pre)
print(f"  pre-filter <F>={mean_F_pre:.6f}  taueff={taueff:.4f}")

# Stand-in self with the only attribute the method uses: nbins.
self_stub = SimpleNamespace(nbins=nbins)

t0 = time.time()
tausum = np.max(tau, axis=1)
ii = np.where(tausum > TAU_THRESH)[0]
print(f"  {len(ii)}/{tau.shape[0]} sightlines flagged")
total_filled = 0
for i in ii:
    tt, tot = _filter_single_tau_complex(self_stub, tau[i], taueff,
                                         tau_thresh=TAU_THRESH, thresh2=THRESH2)
    tau[i] = tt
    total_filled += tot
assert np.max(tau) < TAU_THRESH * 1.01
print(f"  filtered in {time.time()-t0:.1f}s; pixels filled = {total_filled} "
      f"({total_filled / tau.size * 100:.3f}% of pixels)")


# =================================================== PRIYA-style mean_flux
tau_obs_kim = obs_mean_tau(priya_z)
target_F = float(np.exp(-priya_alpha * tau_obs_kim))
print(f"\ntau_obs_Kim2013(z={priya_z}) = {tau_obs_kim:.6f}")
print(f"mean_flux_desired = exp(-alpha * tau_obs) = {target_F:.6f}")


# ============================================== fake_spectra flux_power
t0 = time.time()
kf_kms, P_mine = flux_power(tau, vmax, spec_res=0.0,
                            mean_flux_desired=target_F, window=False)
kf_kms = kf_kms[1:]
P_mine = P_mine[1:]
print(f"flux_power done in {time.time()-t0:.1f}s, {len(kf_kms)} k-bins")

scale = mean_flux(tau, target_F)
print(f"scale (final, post-filter inversion) = {scale:.6f}")


# ============================================== Direct mode-by-mode ratio
n = N_K_PRIYA
ratio_direct = P_mine[:n] / P_priya
# also direct k-grid alignment check
dk_rel = (kf_kms[:n] - priya_kfkms) / priya_kfkms
print(f"\n=== k-grid alignment ===")
print(f"  max |dk/k| over {n} modes: {np.max(np.abs(dk_rel)):.3e} (should be ~1e-16)")

print(f"\n=== Direct mode-by-mode ratio (no interpolation) ===")
print(f"  median = {np.median(ratio_direct):.6f}")
print(f"  mean   = {np.mean(ratio_direct):.6f}")
print(f"  std    = {np.std(ratio_direct):.6f}")
print(f"  min, max  = {ratio_direct.min():.6f}, {ratio_direct.max():.6f}")
print(f"  max|r-1|  = {np.max(np.abs(ratio_direct - 1)):.4g}")
print(f"  p95|r-1|  = {np.percentile(np.abs(ratio_direct - 1), 95):.5f}")
print(f"  p99|r-1|  = {np.percentile(np.abs(ratio_direct - 1), 99):.5f}")
print(f"\n  Sample bins:")
for i in [0, 1, 5, 20, 50, 100, 150, 170]:
    print(f"    m={i+1:3d}  k={priya_kfkms[i]:.4e}  mine={P_mine[i]:.4e}  "
          f"PRIYA={P_priya[i]:.4e}  ratio={ratio_direct[i]:.6f}")


# Save
_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
out_npz = _FIGS / "2026-05-20-priya-p1d-consistency-v3.npz"
np.savez(out_npz,
         priya_kfkms=priya_kfkms, priya_p1d=P_priya,
         kf_kms=kf_kms, p1d_mine=P_mine,
         ratio_direct=ratio_direct,
         alpha=priya_alpha, z=priya_z, target_F=target_F, scale=scale,
         n_filtered=len(ii), total_pixels_filled=total_filled)
print(f"\nSaved {out_npz}")
