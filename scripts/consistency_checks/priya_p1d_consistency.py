"""PRIYA P1D consistency check v2 — uses fake_spectra's *actual* code.

Pipeline:
1. Assert sim/snap params match PRIYA's params row (via SimulationICs.json).
2. Load raw tau into memory (~3.5 GB for 691200x1250 float32).
3. Apply fake_spectra's _filter_single_tau_complex (multi-DLA loop per sightline).
4. Compute mean_flux_desired = exp(-alpha * obs_mean_tau_Kim2013(z)) per PRIYA's
   convention (mean_flux.py / fluxstatistics.py both use the same formula).
5. Solve for the actual tau-multiplier via fake_spectra's _rescale_mean_flux
   (Newton-Raphson C extension).
6. Compute P1D via fake_spectra.fluxstatistics.flux_power.
7. Compare to flux_vectors[row, z_idx*172:(z_idx+1)*172].

Run in emu-3.9 env with GSL loaded:
    module load gcc/10.3.0 gsl/2.7
    conda activate emu-3.9
    python3 scripts/consistency_checks/priya_p1d_consistency.py

Outputs:
    docs/superpowers/figs/2026-05-20-priya-p1d-consistency.npz
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

# fake_spectra imports (require emu-3.9 + GSL)
from fake_spectra.fluxstatistics import (
    flux_power, mean_flux, obs_mean_tau, _powerspectrum, _flux_power_bins,
)
from fake_spectra._spectra_priv import _rescale_mean_flux


# ----------------------------------------------------------------------- INPUTS
PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP = 17
TAU_HDF5 = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
SIM_ICS = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/SimulationICs.json"
META = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"

PRIYA_ROW = 344
PRIYA_Z_IDX = 8  # zout[8] = 3.0
N_K_PRIYA = 172


# ====================================================== 1. ASSERT SIM/SNAP MATCH
print("=== 1. Sim/snap parameter assertion ===")
with h5py.File(PRIYA_FILE, "r") as f:
    priya_params = f["params"][PRIYA_ROW, :].astype(float)
    priya_zout = f["zout"][...]
    priya_z = float(priya_zout[PRIYA_Z_IDX])
priya_alpha = priya_params[0]
priya_cosmo = priya_params[1:]
print(f"PRIYA row {PRIYA_ROW}: alpha (tau0/Kim) = {priya_alpha:.6f}, z = {priya_z}")

# PRIYA's params col order: [alpha, ns, Ap, herei, heref, alphaq, hub, omegamh2,
# hireionz, bhfeedback] per the json (param_names index 0..8 -> cols 1..9).

with open(SIM_ICS) as fp:
    ics = json.load(fp)

omegamh2_mine = ics["omega0"] * ics["hubble"]**2
mine = {
    "ns":        ics["ns"],
    "Ap":        2.2e-9,                         # not in SimulationICs.json (different parameterisation);
                                                 # use folder-name value as PRIYA's training-grid sampling parameter
    "herei":     ics["here_i"],
    "heref":     ics["here_f"],
    "alphaq":    ics["alpha_q"],
    "hub":       ics["hubble"],
    "omegamh2":  omegamh2_mine,
    "hireionz":  ics["hireionz"],
    "bhfeedback":ics["bhfeedback"],
}
priya_named = dict(zip(["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback"], priya_cosmo))
print(f"{'param':<12} {'mine':>14} {'PRIYA':>14} {'rel diff':>12}")
maxrel = 0.0
for k in mine:
    rel = abs(mine[k] - priya_named[k]) / max(abs(priya_named[k]), 1e-30)
    maxrel = max(maxrel, rel)
    print(f"{k:<12} {mine[k]:>14.6g} {priya_named[k]:>14.6g} {rel:>12.4g}")
print(f"Worst relative diff: {maxrel:.4g}")
# Allow Ap to be a known parameterisation difference; everything else should match to <1e-3.
nonAp_max = max(abs(mine[k]-priya_named[k])/max(abs(priya_named[k]),1e-30)
                for k in mine if k != "Ap")
assert nonAp_max < 1e-3, f"non-Ap params disagree at {nonAp_max:.3g} — wrong sim/row pairing!"
print(f"OK (non-Ap params agree to {nonAp_max:.3g})")


# ==================================================== 2. LOAD META & RAW TAU
print("\n=== 2. Load meta + raw tau ===")
with open(META) as fp:
    m = json.load(fp)
nbins = int(m["nbins"])
dv_kms = float(m["dv_kms"])
my_z = float(m["z"])
print(f"nbins={nbins}  dv_kms={dv_kms}  z={my_z}")
assert abs(my_z - priya_z) < 1e-3, f"z mismatch: mine={my_z}, PRIYA={priya_z}"

t0 = time.time()
with h5py.File(TAU_HDF5, "r") as f:
    tau_native = f["tau/H/1/1215"][...].astype(np.float64)  # (n_skewers, nbins)
print(f"  loaded tau shape {tau_native.shape}, dtype {tau_native.dtype},"
      f" {tau_native.nbytes/1e9:.2f} GB, in {time.time()-t0:.1f}s")
n_skewers = tau_native.shape[0]
vmax = nbins * dv_kms
print(f"  vmax = nbins * dv_kms = {vmax:.2f} km/s")


# =============================================== 3. APPLY fake_spectra's _filter_tau
# Port of fake_spectra/spectra.py:_filter_single_tau_complex + _filter_tau,
# multi-DLA loop. This MODIFIES tau in-place.
print("\n=== 3. Apply _filter_tau (multi-DLA per sightline) ===")
TAU_THRESH = 1.0e6
THRESH2 = 0.25
# First-pass tau_eff = -ln<exp(-tau)> over ALL pixels (PRIYA's convention).
mean_F_pre = np.mean(np.exp(-tau_native))
tau_eff = -np.log(mean_F_pre)
print(f"  pre-filter <F> = {mean_F_pre:.6f}, tau_eff = {tau_eff:.4f}")

def filter_single_tau(tt, taueff, tau_thresh=TAU_THRESH, thresh2=THRESH2):
    """Exact port of fake_spectra._filter_single_tau_complex (multi-DLA loop)."""
    n = tt.size
    newthresh = taueff + thresh2
    while np.max(tt) > tau_thresh:
        maxx = int(np.argmax(tt))
        j = 0
        while tt[maxx-j] > newthresh:
            tt[maxx-j] = taueff
            j += 1
            if maxx-j < 0:
                break
        j = 1
        while maxx+j < n and tt[maxx+j] > newthresh:
            tt[maxx+j] = taueff
            j += 1
    return tt

t0 = time.time()
tausum = np.max(tau_native, axis=1)
ii = np.where(tausum > TAU_THRESH)[0]
print(f"  {len(ii)} / {n_skewers} sightlines flagged (max(tau) > {TAU_THRESH:g})")
for i in ii:
    filter_single_tau(tau_native[i], tau_eff)
print(f"  filter took {time.time()-t0:.1f}s; new max(tau) = {tau_native.max():.4g}")


# =============================================== 4. Target mean flux per PRIYA
print("\n=== 4. PRIYA's mean_flux_desired (Kim 2013 obs_mean_tau) ===")
tau_obs_kim = obs_mean_tau(priya_z)         # = 0.0023 * (1+z)^3.65 (Kim 2013 / 0711.1862)
target_tau = priya_alpha * tau_obs_kim       # PRIYA convention: alpha is multiplier on Kim
mean_flux_desired = np.exp(-target_tau)
print(f"  obs_mean_tau(z={priya_z}) [Kim 2013] = {tau_obs_kim:.6f}")
print(f"  target_tau    = alpha * obs_mean_tau = {target_tau:.6f}")
print(f"  target <F>    = exp(-target_tau) = {mean_flux_desired:.6f}")
print(f"  measured <F>α=1     (pre-filter)  = {mean_F_pre:.6f}")
print(f"  measured <F>α=1.011 (direct mult) = {np.mean(np.exp(-priya_alpha*tau_native[:1024])):.6f} (subset, no filter)")


# =========== 5. Solve for the actual tau-multiplier via _rescale_mean_flux
print("\n=== 5. Solve actual tau-multiplier (PRIYA's iterative scale) ===")
t0 = time.time()
# Use the EXACT same call as fake_spectra.fluxstatistics.flux_power
scale = _rescale_mean_flux(tau_native.astype(np.float64), float(mean_flux_desired),
                           tau_native.size, 1e-5, 1e30)
print(f"  scale = {scale:.6f}  (alpha_input = {priya_alpha:.6f})")
print(f"  effective tau-mult = scale (i.e. F = exp(-scale * tau))")
print(f"  Verify: <exp(-scale*tau)> = {np.mean(np.exp(-scale*tau_native[:8192])):.6f} (subset)")
print(f"  solved in {time.time()-t0:.1f}s")


# =================== 6. Compute P1D via fake_spectra.fluxstatistics.flux_power
print("\n=== 6. P1D via fake_spectra.fluxstatistics.flux_power ===")
t0 = time.time()
kf_kms, P_mine = flux_power(tau_native, vmax, spec_res=0.0,
                            mean_flux_desired=float(mean_flux_desired),
                            window=False)
# flux_power returns kf[0..nbins/2] including k=0 mode; PRIYA's
# get_flux_power_1D drops k=0 (it returns kf[1:], avg_flux_power[1:]).
kf_kms = kf_kms[1:]
P_mine = P_mine[1:]
print(f"  fake_spectra P1D done in {time.time()-t0:.1f}s")
print(f"  output: {len(kf_kms)} k-bins, k range [{kf_kms[0]:.4e}, {kf_kms[-1]:.4e}] (angular s/km)")
print(f"  P1D range: [{P_mine.min():.3e}, {P_mine.max():.3e}] (s/km)")


# ============================================= 7. Compare to PRIYA flux_vectors
print("\n=== 7. Compare to PRIYA flux_vectors[344, z=3.0 slab] ===")
with h5py.File(PRIYA_FILE, "r") as f:
    priya_kfkms = f["kfkms"][PRIYA_ROW, PRIYA_Z_IDX, :].astype(np.float64)
    fv_flat = f["flux_vectors"][PRIYA_ROW, :].astype(np.float64)
    P_priya = fv_flat[PRIYA_Z_IDX * N_K_PRIYA:(PRIYA_Z_IDX + 1) * N_K_PRIYA]

# Interpolate my (kf_kms, P_mine) onto PRIYA's kfkms grid (loglog)
valid = (kf_kms > 0) & (P_mine > 0) & np.isfinite(P_mine)
in_range = (priya_kfkms >= kf_kms[valid].min()) & (priya_kfkms <= kf_kms[valid].max())
log_kt = np.log(priya_kfkms)
log_k = np.log(kf_kms[valid])
log_p = np.log(P_mine[valid])
P_mine_on_priya = np.full(N_K_PRIYA, np.nan)
P_mine_on_priya[in_range] = np.exp(np.interp(log_kt[in_range], log_k, log_p))
ratio = P_mine_on_priya / P_priya

both = np.isfinite(ratio) & (P_priya > 0)
print(f"  k bins compared: {both.sum()} / {N_K_PRIYA}")
r = ratio[both]
print(f"  ratio (mine / PRIYA):")
print(f"    median = {np.median(r):.6f}")
print(f"    mean   = {np.mean(r):.6f}")
print(f"    std    = {np.std(r):.6f}")
print(f"    min    = {r.min():.6f}")
print(f"    max    = {r.max():.6f}")
print(f"    p1, p99 = {np.percentile(r,1):.6f}, {np.percentile(r,99):.6f}")
print(f"    fractional excursion from 1.0:  max|r-1| = {np.max(np.abs(r-1)):.4g}")

# Sample bins
print("\n  Sample bins:")
for i in [0, 5, 20, 50, 100, 150, 170]:
    if i < N_K_PRIYA and both[i]:
        print(f"    k={priya_kfkms[i]:.4e}: mine={P_mine_on_priya[i]:.4e}  "
              f"PRIYA={P_priya[i]:.4e}  ratio={ratio[i]:.6f}")

# Save outputs for plotting (alongside the markdown doc that references them)
_OUT_NPZ = (Path(__file__).resolve().parent.parent.parent
            / "docs" / "superpowers" / "figs"
            / "2026-05-20-priya-p1d-consistency.npz")
np.savez(_OUT_NPZ,
         priya_kfkms=priya_kfkms, priya_p1d=P_priya,
         kf_kms=kf_kms, p1d_mine=P_mine,
         p1d_mine_on_priya=P_mine_on_priya, ratio=ratio,
         alpha=priya_alpha, z=priya_z,
         mean_flux_desired=mean_flux_desired, scale_inverted=scale,
         tau_eff_unfiltered=tau_eff, mean_F_unfiltered=mean_F_pre,
         n_filtered=len(ii), n_total=n_skewers)
print(f"\nSaved {_OUT_NPZ}")
