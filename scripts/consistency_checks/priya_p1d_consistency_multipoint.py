"""Multi-point PRIYA P1D consistency check.

For each (sim, snap) pair, read tau once, apply fake_spectra's filter once,
then loop over the 10 α values PRIYA stored for that sim. Compare to PRIYA's
stored flux_vectors row by row.

Runs in emu-3.9 env with GSL loaded:
    module load gcc/10.3.0 gsl/2.7
    conda activate emu-3.9
    python3 scripts/consistency_checks/priya_p1d_consistency_multipoint.py
"""
import os, sys, json, time, glob
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import h5py

from fake_spectra.fluxstatistics import flux_power, mean_flux, obs_mean_tau
from fake_spectra.spectra import Spectra

_filter_single_tau_complex = Spectra._filter_single_tau_complex


# ----------------------------------------------------------------- inputs
PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
HCD_ROOT = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
EMU_ROOT = "/nfs/turbo/umor-yueyingn/mfho/emu_full"

# (sim_idx, folder_name, snap_num) — snap_num maps to a PRIYA z (zout has 13 z: 4.6..2.2 step 0.2)
# Test 3 sims × 4 redshifts (z=4.6 high-z DLA-rich; z=4.0; z=3.0 mid; z=2.4 low-z)
SIM_INDICES = {
    44: "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056",
    0:  "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453",
    29: "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068",
}
# (snap_num, z, priya_z_idx)
SNAPS = [
    (8,  4.6,  0),
    (11, 4.0,  3),
    (17, 3.0,  8),
    (21, 2.4, 11),
]
N_K = 172
N_ALPHA = 10
N_SIM = 60


# ============================================================ load PRIYA
print("Loading PRIYA reference...")
with h5py.File(PRIYA_FILE, "r") as f:
    priya_params = f["params"][...]
    priya_zout = f["zout"][...]
    priya_kfkms = f["kfkms"][...]
    priya_fv = f["flux_vectors"][...]


# Diagnostic accumulator: each row is one (sim, snap, alpha)
rows = []
SAVED_RATIOS = {}   # (sim_idx, snap) -> (alpha_grid, ratio array shape (10, 172))

for sim_idx, sim_folder in SIM_INDICES.items():
    for snap, zexp, zidx in SNAPS:
        meta_p = os.path.join(HCD_ROOT, sim_folder, f"snap_{snap:03d}", "meta.json")
        tau_p = os.path.join(EMU_ROOT, sim_folder, "output", f"SPECTRA_{snap:03d}",
                             "lya_forest_spectra_grid_480.hdf5")
        if not os.path.exists(meta_p):
            print(f"  [skip] no meta: {meta_p}")
            continue
        if not os.path.exists(tau_p):
            print(f"  [skip] no tau: {tau_p}")
            continue

        with open(meta_p) as fp:
            m = json.load(fp)
        nbins = int(m["nbins"])
        dv_kms = float(m["dv_kms"])
        z_actual = float(m["z"])
        vmax = nbins * dv_kms
        if abs(z_actual - zexp) > 1e-2:
            print(f"  [skip] z mismatch: sim_idx={sim_idx} snap={snap} z={z_actual} (wanted {zexp})")
            continue

        # PRIYA's stored kfkms / fv for this (sim, z)
        print(f"\n=== sim_idx={sim_idx}  snap={snap}  z={z_actual} ===")
        t0 = time.time()
        with h5py.File(tau_p, "r") as f:
            tau = f["tau/H/1/1215"][...].astype(np.float64)
        print(f"  loaded tau {tau.shape} in {time.time()-t0:.1f}s  vmax={vmax:.1f} km/s")

        # apply fake_spectra's filter (periodic BC) — ONCE per (sim, snap)
        tau_eff = -np.log(np.mean(np.exp(-tau)))
        ii = np.where(np.max(tau, axis=1) > 1e6)[0]
        t0 = time.time()
        self_stub = SimpleNamespace(nbins=nbins)
        for i in ii:
            tau[i], _ = _filter_single_tau_complex(self_stub, tau[i], tau_eff,
                                                   tau_thresh=1e6, thresh2=0.25)
        print(f"  filtered {len(ii)} sightlines in {time.time()-t0:.1f}s  taueff={tau_eff:.4f}")

        # PRIYA's 10 α-rows for this sim_idx
        alpha_rows = [sim_idx + N_SIM * a for a in range(N_ALPHA)]
        alphas = [float(priya_params[r, 0]) for r in alpha_rows]
        ratio_grid = np.full((N_ALPHA, N_K), np.nan)

        for a_idx, prow in enumerate(alpha_rows):
            alpha = float(priya_params[prow, 0])
            target_F = float(np.exp(-alpha * obs_mean_tau(z_actual)))
            t0 = time.time()
            kf, P = flux_power(tau, vmax, spec_res=0.0,
                               mean_flux_desired=target_F, window=False)
            kf, P = kf[1:], P[1:]
            dt = time.time() - t0

            # PRIYA stored row
            P_priya = priya_fv[prow, zidx*N_K:(zidx+1)*N_K].astype(np.float64)
            kp = priya_kfkms[prow, zidx, :].astype(np.float64)
            # k-grid sanity check
            dk = (kf[:N_K] - kp)
            assert np.max(np.abs(dk)/kp) < 1e-10, f"k-grid mismatch! max |dk/k|={np.max(np.abs(dk)/kp):.3e}"

            r = P[:N_K] / P_priya
            ratio_grid[a_idx] = r
            row = dict(
                sim_idx=sim_idx, snap=snap, z=z_actual, alpha=alpha,
                priya_row=prow,
                median=float(np.median(r)),
                mean=float(np.mean(r)),
                std=float(np.std(r)),
                max_abs_dev=float(np.max(np.abs(r - 1))),
                p99_abs_dev=float(np.percentile(np.abs(r - 1), 99)),
                dt_s=dt,
            )
            rows.append(row)
            print(f"   α={alpha:.4f}  target_F={target_F:.4f}  "
                  f"med={row['median']:.7f}  std={row['std']:.2e}  "
                  f"max|r-1|={row['max_abs_dev']:.3e}  ({dt:.0f}s)")

        SAVED_RATIOS[(sim_idx, snap)] = (np.array(alphas), ratio_grid)
        del tau

# ============================================================ summary
print(f"\n\n========== SUMMARY ({len(rows)} test points) ==========")
print(f"{'sim_idx':>7} {'snap':>5} {'z':>5} {'α':>7} {'median':>11} {'std':>10} {'max|r-1|':>10}")
for r in rows:
    print(f"{r['sim_idx']:>7} {r['snap']:>5} {r['z']:>5.2f} {r['alpha']:>7.4f} "
          f"{r['median']:>11.7f} {r['std']:>10.2e} {r['max_abs_dev']:>10.2e}")

# Overall metrics
all_max = [r["max_abs_dev"] for r in rows]
all_std = [r["std"] for r in rows]
all_med = [r["median"] for r in rows]
print(f"\nOverall:")
print(f"  worst max|r-1| across all test points: {max(all_max):.3e}")
print(f"  worst std across all test points:      {max(all_std):.3e}")
print(f"  median of medians:                     {np.median(all_med):.7f}")
print(f"  all max|r-1| < 1e-4?  {all(m < 1e-4 for m in all_max)}")
print(f"  all max|r-1| < 1e-5?  {all(m < 1e-5 for m in all_max)}")

# Save for plot
_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
np.savez(_FIGS / "2026-05-20-priya-p1d-multipoint.npz",
         rows=np.array([(r['sim_idx'], r['snap'], r['z'], r['alpha'], r['priya_row'],
                          r['median'], r['std'], r['max_abs_dev'], r['p99_abs_dev'])
                         for r in rows],
                        dtype=[("sim_idx","i4"),("snap","i4"),("z","f4"),("alpha","f8"),
                               ("priya_row","i4"),("median","f8"),("std","f8"),
                               ("max_abs_dev","f8"),("p99_abs_dev","f8")]),
         saved_ratios_keys=np.array(list(SAVED_RATIOS.keys()), dtype="i4"),
         **{f"sim{k[0]}_snap{k[1]}_alphas": v[0] for k, v in SAVED_RATIOS.items()},
         **{f"sim{k[0]}_snap{k[1]}_ratios": v[1] for k, v in SAVED_RATIOS.items()},
)
print(f"\nSaved {_FIGS}/2026-05-20-priya-p1d-multipoint.npz")
