"""
Numerical-hypothesis tests for the CDDF / dN/dX excess vs observations.

Hypothesis 1 (τ_threshold): detection threshold may catch forest-blend
"systems" that aren't real physical halos.  Vary τ_threshold ∈ {100, 300,
1000, 3000, 10000, 30000, 100000} and watch DLA count.

Hypothesis 2 (merge_dv_kms): the 100 km/s merge criterion may split
broadened DLA cores with brief τ-dips into multiple catalog entries.
Vary merge_dv_kms ∈ {100, 200, 500, 1000, 2000} and watch DLA count.

Hypothesis 3 (sightline overlap): 480×480 grid ≈ 0.25 Mpc/h lateral
spacing at comoving; DLA host haloes have R_vir ~ 50–200 kpc proper
(~0.05–0.2 Mpc comoving at z=3).  Close sightlines may pierce the same
halo.  Downsample to 240×240 and 120×120 and see if dN/dX drops.

Hypothesis 4 (per-sightline multiplicity): how often does one sightline
host >1 DLA within 1000 km/s?  Distribution of v-separations.

One flagship snapshot: ns0.803 snap_017 (z=3).  Each test writes a
result file under tests/out/cddf_excess/ and prints summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import AbsorberCatalog, build_catalog
from hcd_analysis.io import read_header, pixel_dv_kms

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT_EXISTING = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs") / SIM / "snap_017" / "catalog.npz"
OUT = ROOT / "tests" / "out" / "cddf_excess"
OUT.mkdir(parents=True, exist_ok=True)


def summary_counts(cat: AbsorberCatalog) -> dict:
    return {
        "n_total": len(cat.absorbers),
        **cat.summary(),
    }


# ------------------------------------------------------------------------
# H1: vary tau_threshold (detection)
# ------------------------------------------------------------------------
def test_H1_tau_threshold(n_workers=8, n_skewers=None):
    print("=" * 72)
    print("H1: vary tau_threshold (detection sensitivity)")
    print("=" * 72)
    header = read_header(HDF5); dv = pixel_dv_kms(header)
    results = []
    for tau_thr in [100, 300, 1000, 3000, 10000, 30000, 100000]:
        t0 = time.time()
        cat = build_catalog(
            hdf5_path=HDF5, sim_name=SIM, snap=17, z=header.redshift, dv_kms=dv,
            tau_threshold=float(tau_thr), merge_dv_kms=100.0, min_pixels=2,
            fast_mode=True, n_workers=n_workers, min_log_nhi=17.2,
            batch_size=4096, n_skewers=n_skewers,
        )
        s = summary_counts(cat)
        s["tau_threshold"] = tau_thr; s["dt_s"] = round(time.time()-t0, 1)
        results.append(s)
        print(f"  τ_thr={tau_thr:>6}: {s}")
    (OUT / "H1_tau_threshold.json").write_text(json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------------
# H2: vary merge_dv_kms (one-halo line-blending recovery)
# ------------------------------------------------------------------------
def test_H2_merge_dv_kms(n_workers=8, n_skewers=None):
    print("=" * 72)
    print("H2: vary merge_dv_kms (should reduce double-counting per halo)")
    print("=" * 72)
    header = read_header(HDF5); dv = pixel_dv_kms(header)
    results = []
    for merge in [50, 100, 200, 500, 1000, 2000, 5000]:
        t0 = time.time()
        cat = build_catalog(
            hdf5_path=HDF5, sim_name=SIM, snap=17, z=header.redshift, dv_kms=dv,
            tau_threshold=100.0, merge_dv_kms=float(merge), min_pixels=2,
            fast_mode=True, n_workers=n_workers, min_log_nhi=17.2,
            batch_size=4096, n_skewers=n_skewers,
        )
        s = summary_counts(cat)
        s["merge_dv_kms"] = merge; s["dt_s"] = round(time.time()-t0, 1)
        results.append(s)
        print(f"  merge={merge:>5} km/s: {s}")
    (OUT / "H2_merge_dv_kms.json").write_text(json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------------
# H3: sightline-grid downsampling (halo-overlap test)
# ------------------------------------------------------------------------
def test_H3_sightline_grid(n_workers=8):
    print("=" * 72)
    print("H3: downsample sightline grid — does close-spacing inflate count?")
    print("=" * 72)
    # The 480×480×3 grid maps to rows 0..691199 in the HDF5.  Axes 1,2,3
    # interleave (see spectra/axis).  Simpler: just use a uniformly-spaced
    # subset of sightline indices.
    header = read_header(HDF5); dv = pixel_dv_kms(header)
    n_total = header.n_skewers  # 691200
    results = []

    # Load existing full catalog (no rebuild needed — we're just subsampling).
    cat_full = AbsorberCatalog.load_npz(CAT_EXISTING)
    si_arr = np.array([a.skewer_idx for a in cat_full.absorbers])
    logN_arr = np.array([a.log_NHI for a in cat_full.absorbers])
    print(f"  Baseline full catalog: total={len(cat_full.absorbers)}, "
          f"DLA={int((logN_arr>=20.3).sum())}")

    # Read spectra/axis once so we can restrict to a uniform subset along each axis.
    with h5py.File(HDF5, "r") as f:
        axis = f["spectra/axis"][:]
        cofm = f["spectra/cofm"][:]    # (N,3) kpc/h comoving

    # For each axis (x,y,z LOS), sightlines form a 480×480 transverse grid.
    # Stride k means 1/k² of sightlines kept per axis.
    for stride in [1, 2, 4, 8]:
        kept = np.zeros(n_total, dtype=bool)
        for ax in (1, 2, 3):
            sel_axis = np.where(axis == ax)[0]
            # 480×480 within one axis.  Transverse positions in the two axes
            # perpendicular to LOS.  Use cofm to find the unique transverse
            # coordinates (pairs), then keep 1 every `stride` along each.
            other_dims = [d for d in (0, 1, 2) if d != (ax - 1)]
            transverse = cofm[sel_axis][:, other_dims]
            # Snap to grid
            u = np.unique(np.round(transverse[:, 0], 2))
            v = np.unique(np.round(transverse[:, 1], 2))
            u_keep = u[::stride]; v_keep = v[::stride]
            keep_mask = (np.isin(np.round(transverse[:, 0], 2), u_keep)
                        & np.isin(np.round(transverse[:, 1], 2), v_keep))
            kept[sel_axis[keep_mask]] = True
        n_kept = int(kept.sum())
        # Now count catalog entries only on these kept sightlines
        sel_cat = np.isin(si_arr, np.where(kept)[0])
        logN_sub = logN_arr[sel_cat]
        n_dla = int((logN_sub >= 20.3).sum())
        n_sub = int(((logN_sub >= 19.0) & (logN_sub < 20.3)).sum())
        n_lls = int(((logN_sub >= 17.2) & (logN_sub < 19.0)).sum())
        frac_sl = n_kept / n_total
        # If no sightline-overlap: per-sightline density unchanged,
        #   count scales as n_kept / n_total.
        # If overlap: count scales slower than n_kept — suggests multiple
        #   sightlines piercing each halo.
        # Normalise to expected-scaling count:
        expected_DLA = int((logN_arr >= 20.3).sum() * frac_sl)
        ratio = n_dla / expected_DLA if expected_DLA > 0 else float("nan")
        results.append({"stride": stride, "n_kept": n_kept,
                         "frac_sl": round(frac_sl, 4),
                         "DLA": n_dla, "subDLA": n_sub, "LLS": n_lls,
                         "DLA_expected": expected_DLA,
                         "ratio_observed_to_expected": round(ratio, 4)})
        print(f"  stride={stride} (frac_sl={frac_sl:.3f}):  DLA={n_dla:>6}  "
              f"expected(if uniform)={expected_DLA}  ratio={ratio:.3f}")
    (OUT / "H3_sightline_grid.json").write_text(json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------------
# H4: per-sightline DLA multiplicity + v-separation distribution
# ------------------------------------------------------------------------
def test_H4_per_sightline_multiplicity():
    print("=" * 72)
    print("H4: per-sightline multiplicity & v-separation")
    print("=" * 72)
    header = read_header(HDF5); dv = pixel_dv_kms(header)
    cat = AbsorberCatalog.load_npz(CAT_EXISTING)
    si_arr = np.array([a.skewer_idx for a in cat.absorbers])
    logN_arr = np.array([a.log_NHI for a in cat.absorbers])
    pix_start = np.array([a.pix_start for a in cat.absorbers])
    pix_end = np.array([a.pix_end for a in cat.absorbers])

    # DLA multiplicity per sightline
    for cls_name, cls_mask in [("DLA", logN_arr >= 20.3),
                                ("subDLA", (logN_arr >= 19.0) & (logN_arr < 20.3))]:
        si_cls = si_arr[cls_mask]
        unique, counts = np.unique(si_cls, return_counts=True)
        print(f"\n  {cls_name} multiplicity per sightline:")
        for k in range(1, 6):
            n_k = int((counts == k).sum())
            print(f"    exactly {k}: {n_k:>6}  sightlines  ({100*n_k/len(unique):.2f}%)")
        n_ge6 = int((counts >= 6).sum())
        print(f"    ≥ 6:       {n_ge6:>6}  sightlines  ({100*n_ge6/len(unique):.2f}%)")

        # v-separation between adjacent entries of the same class on the same sightline
        seps = []
        for sl in unique[counts > 1]:
            sel = si_cls == sl   # wait: si_cls filters already; need pix coords
            idx = np.where((si_arr == sl) & cls_mask)[0]
            ps = np.sort(pix_start[idx])
            for j in range(len(ps) - 1):
                seps.append((ps[j+1] - ps[j]) * dv)
        seps = np.array(seps)
        print(f"    n pairs: {len(seps)}")
        if len(seps) > 0:
            print(f"    v-sep (km/s) between {cls_name} pairs: "
                  f"median={np.median(seps):.0f}, "
                  f"p25={np.percentile(seps, 25):.0f}, "
                  f"p75={np.percentile(seps, 75):.0f}")
            print(f"    v-sep < 500 km/s: {(seps<500).sum()}  (candidates for line-blending)")
            print(f"    v-sep < 200 km/s: {(seps<200).sum()}")
            print(f"    v-sep < 100 km/s: {(seps<100).sum()}")

    summary = {"sim": SIM[:40], "snap": 17, "z": header.redshift}
    (OUT / "H4_multiplicity.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+",
                    default=["H1", "H2", "H3", "H4"],
                    choices=["H1","H2","H3","H4"])
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--n-skewers", type=int, default=100000,
                    help="H1/H2 subsample — full 691k is much slower")
    args = ap.parse_args()

    if "H1" in args.tests: test_H1_tau_threshold(args.n_workers, args.n_skewers)
    if "H2" in args.tests: test_H2_merge_dv_kms(args.n_workers, args.n_skewers)
    if "H3" in args.tests: test_H3_sightline_grid(args.n_workers)
    if "H4" in args.tests: test_H4_per_sightline_multiplicity()


if __name__ == "__main__":
    main()
