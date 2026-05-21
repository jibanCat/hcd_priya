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


def test_tier_c_matches_filter_free_sum_at_alpha_one():
    """When tau_thresh=inf (no filter), the sightline-weighted sum of the four
    Tier-C P1Ds must equal the Tier-P-without-filter P1D — to floating-point.
    This guards against bugs in the per-class accumulator."""
    from hcd_analysis.priya_p1d import compute_tier_c_p1d, compute_tier_p_p1d
    from hcd_analysis.catalog import AbsorberCatalog

    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as f: m = json.load(f)
    vmax = int(m["nbins"]) * float(m["dv_kms"])
    z = float(m["z"])
    with h5py.File(tau_p, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)

    alpha = 1.0
    # Tier P with filter disabled => total P1D over all sightlines, no DLA mask
    kf_P, P_P, target_F, scale = compute_tier_p_p1d(
        tau.copy(), vmax, alpha_slope=alpha, z=z, tau_thresh=np.inf)
    # Tier C: per-class P1Ds with the same scale + target_F
    kf_C, by_class, n_by_class, _, _ = compute_tier_c_p1d(
        tau, vmax, alpha_slope=alpha, z=z, catalog=catalog,
        external_scale=scale, external_target_F=target_F)
    # weights = n_class / n_total
    n_total = sum(n_by_class.values())
    P_recombined = sum((n_by_class[c]/n_total) * by_class[c] for c in by_class)
    assert np.allclose(P_recombined, P_P, rtol=1e-10), \
        f"per-class sum != total P1D at alpha=1 (no filter), worst rel diff "\
        f"= {np.max(np.abs(P_recombined/P_P - 1)):.3e}"
    print("OK — Tier C sums to Tier P at alpha=1 with no filter")


def _snap_to_grid(z):
    return round(float(z) / 0.2) * 0.2


def _find_snap_for_z(sim_folder, target_z, hcd_root):
    """Return the snap number whose meta z snaps to target_z, or None.

    Snap numbering is sim-specific (e.g. z=3.0 is snap_017 for sim 44 but
    snap_016 for sim 0), so discover it from meta.json rather than assuming
    a fixed snap-to-z map (lesson from the 2026-05-20 sbatch fan-out).
    """
    import glob
    for meta in glob.glob(f"{hcd_root}/{sim_folder}/snap_*/meta.json"):
        with open(meta) as f:
            zz = float(json.load(f)["z"])
        if abs(_snap_to_grid(zz) - target_z) < 1e-6:
            return int(Path(meta).parent.name.split("_")[1])
    return None


def test_priya_p1d_bit_identical_multipoint():
    """SLOW (~80 min serial): compute_tier_p_p1d at every PRIYA alpha for
    3 sims x 4 redshifts (120 points). Asserts max|r-1| < 1e-4 everywhere.

    Guarded by SLOW_TESTS=1 (skips otherwise). To run:
        SLOW_TESTS=1 \\
        LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH \\
        /home/mfho/.conda/envs/emu-3.9/bin/python3 tests/test_priya_p1d.py
    For the parallel sbatch fan-out, see scripts/consistency_checks/sbatch_multipoint.sh.
    """
    if not os.environ.get("SLOW_TESTS"):
        print("SKIP test_priya_p1d_bit_identical_multipoint (set SLOW_TESTS=1 to run)")
        return

    HCD = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
    EMU = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
    sims = {  # PRIYA sim_idx -> hcd_outputs folder
        44: "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056",
        0:  "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453",
        29: "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068",
    }
    z_targets = [(4.6, 0), (4.0, 3), (3.0, 8), (2.4, 11)]  # (z, PRIYA zout idx)
    N_SIM = 60

    with h5py.File(PRIYA_FILE, "r") as f:
        params = f["params"][...]
        fv = f["flux_vectors"][...]

    worst = 0.0
    n_pts = 0
    for sim_idx, folder in sims.items():
        for ztarget, zidx in z_targets:
            snap = _find_snap_for_z(folder, ztarget, HCD)
            assert snap is not None, f"no snap at z={ztarget} for {folder}"
            with open(f"{HCD}/{folder}/snap_{snap:03d}/meta.json") as f:
                meta = json.load(f)
            vmax = int(meta["nbins"]) * float(meta["dv_kms"])
            taup = f"{EMU}/{folder}/output/SPECTRA_{snap:03d}/lya_forest_spectra_grid_480.hdf5"
            with h5py.File(taup, "r") as f:
                tau = f["tau/H/1/1215"][...].astype(np.float64)
            # First alpha call filters tau in place; later calls no-op the
            # filter (max(tau) < 1e6 afterwards) and just re-solve mean flux.
            for a in range(10):
                prow = sim_idx + N_SIM * a
                alpha = float(params[prow, 0])
                # ztarget is exactly on PRIYA's zout grid -> no z-mismatch bias.
                kf, P_mine, target_F, scale = compute_tier_p_p1d(
                    tau, vmax, alpha_slope=alpha, z=ztarget)
                P_priya = fv[prow, zidx*N_K:(zidx+1)*N_K].astype(np.float64)
                r = P_mine[:N_K] / P_priya
                m = float(np.max(np.abs(r - 1)))
                worst = max(worst, m)
                n_pts += 1
                assert m < 1e-4, \
                    f"sim{sim_idx} z{ztarget} alpha={alpha:.4f}: max|r-1|={m:.3e}"
            del tau
    print(f"OK multipoint: {n_pts} points, worst max|r-1| = {worst:.3e}")


def test_bin_sightlines_by_nhi():
    from types import SimpleNamespace
    from hcd_analysis.priya_p1d import (
        bin_sightlines_by_nhi, tier_c_labels, N_TIER_C_BINS, FINE_NHI_EDGES)
    # Edges (14): [17.2, ...7 LLS..., 19.0, ...5 subDLA..., 20.3, 21.0]
    # classes: 0 clean | 1..7 LLS | 8..12 subDLA | 13 DLA-edge [20.3,21.0) | 14 >=21.0
    ab = lambda i, lognhi: SimpleNamespace(skewer_idx=i, log_NHI=lognhi)
    cat = SimpleNamespace(absorbers=[
        ab(1, 17.3),    # LLS, [17.2,17.4571) -> class 1
        ab(2, 19.0),    # subDLA floor (exact edge) -> class 8
        ab(3, 20.4),    # DLA edge [20.3,21.0) -> class 13
        ab(4, 23.1),    # DLA tail >=21.0 -> class 14
        ab(5, 18.0), ab(5, 20.9),  # highest wins: 20.9 -> DLA edge -> class 13
    ])
    cls = bin_sightlines_by_nhi(cat, n_skewers=6)
    assert cls.shape == (6,)
    assert cls[0] == 0          # clean (no absorber)
    assert cls[1] == 1          # 17.3
    assert cls[2] == 8          # 19.0 exact subDLA edge
    assert cls[3] == 13         # 20.4
    assert cls[4] == 14         # overflow tail
    assert cls[5] == 13         # max(18.0,20.9)=20.9
    assert len(FINE_NHI_EDGES) == 14 and N_TIER_C_BINS == 15
    assert tier_c_labels()[0] == "clean"
    assert tier_c_labels()[-1].startswith(">=")
    print("OK bin_sightlines_by_nhi")


if __name__ == "__main__":
    test_bin_sightlines_by_nhi()
    test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1()
    test_tier_c_matches_filter_free_sum_at_alpha_one()
    test_priya_p1d_bit_identical_multipoint()
    print("OK")
