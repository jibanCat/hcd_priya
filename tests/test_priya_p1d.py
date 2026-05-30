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


def test_tier_c_fine_bins_sum_to_total_at_alpha_one():
    """Count-weighted sum of all fine-N_HI bin P1Ds == Tier-P-no-filter P1D."""
    from hcd_analysis.priya_p1d import compute_tier_c_p1d, compute_tier_p_p1d
    from hcd_analysis.catalog import AbsorberCatalog
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as fh: m = json.load(fh)
    vmax = int(m["nbins"]) * float(m["dv_kms"]); z = float(m["z"])
    with h5py.File(tau_p, "r") as fh:
        tau = fh["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)
    kf_P, P_P, target_F, scale = compute_tier_p_p1d(
        tau.copy(), vmax, alpha_slope=1.0, z=z, tau_thresh=np.inf)
    kf_C, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau, vmax, alpha_slope=1.0, z=z, catalog=catalog,
        external_scale=scale, external_target_F=target_F)
    n_total = int(n_by_bin.sum())
    P_recombined = (n_by_bin[:, None] / n_total * P_by_bin).sum(axis=0)
    assert np.allclose(P_recombined, P_P, rtol=1e-10), \
        f"fine-bin sum != total; worst {np.max(np.abs(P_recombined/P_P-1)):.3e}"
    assert np.array_equal(kf_C, kf_P)
    print(f"OK fine-bin sum==total ({P_by_bin.shape[0]} bins, n={n_total})")


def test_fine_bins_reconstruct_subdla_split():
    """Merging fine bins at the aligned 19.0 edge reproduces a direct
    clean+LLS vs subDLA+DLA split (mechanics of count-weighted reconstruction)."""
    from hcd_analysis.priya_p1d import (
        compute_tier_c_p1d, compute_tier_p_p1d, merge_fine_to_classes,
        _per_class_p1d_at_scale, bin_sightlines_by_nhi, FINE_NHI_EDGES)
    from hcd_analysis.catalog import AbsorberCatalog
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as fh: m = json.load(fh)
    vmax = int(m["nbins"]) * float(m["dv_kms"]); z = float(m["z"])
    with h5py.File(tau_p, "r") as fh:
        tau = fh["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)
    _, _, tF, sc = compute_tier_p_p1d(tau.copy(), vmax, 1.0, z, tau_thresh=np.inf)
    kf_C, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau, vmax, 1.0, z, catalog=catalog, external_scale=sc, external_target_F=tF)
    # 19.0 is FINE_NHI_EDGES[7]; class index just above it is 8. Split classes
    # [0..8) (clean+LLS, max log_NHI < 19.0) vs [8..15) (subDLA+DLA).
    split_idx = int(np.where(np.isclose(FINE_NHI_EDGES, 19.0))[0][0]) + 1   # = 8
    P_lo = merge_fine_to_classes(P_by_bin, n_by_bin, [(0, split_idx)])[0]
    cls = bin_sightlines_by_nhi(catalog, tau.shape[0])
    _, P_direct = _per_class_p1d_at_scale(tau[cls < split_idx], vmax, sc, tF)
    assert np.allclose(P_lo, P_direct, rtol=1e-10)
    print("OK fine->class reconstruction at 19.0 edge")


def test_tier_c_nonDLA_reproduces_priya_filtered():
    from hcd_analysis.priya_p1d import compute_tier_p_p1d, compute_tier_c_p1d
    from hcd_analysis.catalog import AbsorberCatalog
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP=17; NK=172; ZIDX=8; ROW=344
    tau_p=f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    cat_p=f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    meta_p=f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    priya="/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
    if not all(os.path.exists(p) for p in (tau_p,cat_p,meta_p,priya)):
        print("SKIP tier_c_nonDLA (data unavailable)"); return
    with open(meta_p) as fh: m=json.load(fh)
    vmax=int(m["nbins"])*float(m["dv_kms"]); z=round(float(m["z"])/0.2)*0.2  # PRIYA grid z
    with h5py.File(tau_p,"r") as fh: tau=fh["tau/H/1/1215"][...].astype(np.float64)
    catalog=AbsorberCatalog.load_npz(cat_p)
    with h5py.File(priya,"r") as fh:
        alpha=float(fh["params"][ROW,0])
        P_priya=fh["flux_vectors"][ROW,ZIDX*NK:(ZIDX+1)*NK].astype(np.float64)
    # Tier P filters tau in place; reuse the SAME filtered tau for Tier C so the
    # per-class partition is exact against Tier P.
    tau_f=tau.copy()
    kf_p,P_p,tF,scale=compute_tier_p_p1d(tau_f, vmax, alpha_slope=alpha, z=z)
    assert np.max(np.abs(P_p[:NK]/P_priya-1))<1e-4, "Tier P != PRIYA (precondition)"
    kf_c,P_by_bin,n_by_bin,_,_=compute_tier_c_p1d(
        tau_f, vmax, alpha, z, catalog=catalog, external_scale=scale, external_target_F=tF)
    N=int(n_by_bin.sum())
    P_all=(n_by_bin[:,None]/N*P_by_bin).sum(0)
    P_nonDLA=(n_by_bin[:13,None]/N*P_by_bin[:13]).sum(0)   # classes 0..12 = forest+LLS+subDLA
    n_dla=int(n_by_bin[13:].sum())
    # EXACT: all classes on the filtered tau reconstruct Tier P (= PRIYA)
    assert np.allclose(P_all, P_p, rtol=1e-10), \
        f"filtered partition != Tier P, worst {np.max(np.abs(P_all/P_p-1)):.2e}"
    # DIAGNOSTIC: non-DLA classes vs PRIYA's filtered P1D
    res=np.abs(P_nonDLA[:NK]/P_priya-1)
    print(f"RESULT Tier-C decomposition: filtered partition EXACT; "
          f"non-DLA vs PRIYA max|r-1|={res.max():.3e} med={np.median(res):.3e} "
          f"(n_DLA={n_dla}/{N}={n_dla/N:.2%})  [target <1%]")
    assert res.max() < 0.05, f"non-DLA residual {res.max():.3e} >5% — investigate"
    print("OK tier_c_nonDLA_reproduces_priya_filtered")


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


def test_per_class_freeze_core_thin_and_frozen():
    """tau_freeze=inf reproduces the uniform rescale exactly; a finite
    tau_freeze leaves frozen pixels at native tau and scales the rest, which
    changes the per-class P1D. The freeze must bite in the TRANSITION regime
    (moderate tau): fully-saturated cores (tau>~745) underflow exp(-tau)->0
    either way, so the physically-relevant difference lives at moderate tau."""
    import numpy as np
    from hcd_analysis.priya_p1d import _per_class_p1d_at_scale
    rng = np.random.default_rng(0)
    tau = rng.uniform(0.0, 5.0, size=(64, 128))   # forest + transition-regime tau
    vmax, scale, tF = 1000.0, 0.8, 0.7
    tfreeze = 2.0                                  # splits the [0,5] range

    kf_u, P_u = _per_class_p1d_at_scale(tau, vmax, scale, tF)                 # uniform
    kf_i, P_i = _per_class_p1d_at_scale(tau, vmax, scale, tF, tau_freeze=np.inf)
    assert np.allclose(P_u, P_i, rtol=0, atol=0), "tau_freeze=inf must equal uniform"

    kf_f, P_f = _per_class_p1d_at_scale(tau, vmax, scale, tF, tau_freeze=tfreeze)
    tau_eff = np.where(tau > tfreeze, tau, scale * tau)
    frozen, thin = tau > tfreeze, tau <= tfreeze
    assert frozen.any() and thin.any(), "both regimes must be populated"
    assert np.all(tau_eff[frozen] == tau[frozen]), "frozen pixels keep native tau"
    assert np.allclose(tau_eff[thin], scale * tau[thin]), "thin pixels scaled by scale"
    assert not np.allclose(P_f, P_u), "freeze must change the per-class P1D"
    print("OK freeze-core: inf==uniform, finite freeze bites in the transition regime")


if __name__ == "__main__":
    test_bin_sightlines_by_nhi()
    test_per_class_freeze_core_thin_and_frozen()
    test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1()
    test_tier_c_fine_bins_sum_to_total_at_alpha_one()
    test_fine_bins_reconstruct_subdla_split()
    test_tier_c_nonDLA_reproduces_priya_filtered()
    test_priya_p1d_bit_identical_multipoint()
    print("OK")
