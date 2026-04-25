"""
Fit Rogers+2018 four-parameter α vector per (sim, snap) for the LF suite
and the HR suite.

For each available `p1d_per_class.h5`:
  - Reconstruct the "HCD-dirty" P1D by weighting P_class_only by per-class
    sightline fractions:  P_total(k) = Σ_c (n_c / n_tot) · P_c_only(k)
  - Use P_clean as the reference forest P1D.
  - Call `hcd_template.fit_alpha(k_angular, P_total, P_forest, z)` with
    the PRIYA angular k convention (k_ang = 2π · k_cyclic).
  - Restrict the fit to the PRIYA emulator range (0.0009 ≤ k_ang ≤ 0.20
    rad·s/km).

Outputs one HDF5 per sim under `data_dir()/rogers_alpha/<sim>.h5` with
per-snap groups:
  {snap}/alpha    (4,)  α_LLS, α_Sub, α_Small, α_Large
  {snap}/alpha_err (4,)
  {snap}/chi2     scalar
  {snap}/z        scalar
  {snap}/k_ang    (n_k_in_window,)
  {snap}/ratio_obs  (n_k_in_window,)
  {snap}/ratio_fit  (n_k_in_window,)

Also a master summary HDF5 `data_dir()/rogers_alpha_summary.h5` with
flat 1-D arrays (sim, snap, z, α_LLS, α_Sub, α_Small, α_Large,
alpha_err_*, chi2, dof) for easy (sim, z) scans.

Safe to run in the background: skips sim/snap pairs already in the
per-sim output file unless `--force` is passed.

Run:
    python3 scripts/fit_rogers_alpha.py                # LF + HR
    python3 scripts/fit_rogers_alpha.py --suite lf
    python3 scripts/fit_rogers_alpha.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir
from hcd_analysis.hcd_template import fit_alpha

SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")

# PRIYA emulator k range in angular units
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20

# HDF5 dataset names inside p1d_per_class.h5
CLASS_KEYS = {
    "LLS":     ("P_LLS_only",    "n_sightlines_LLS"),
    "subDLA":  ("P_subDLA_only", "n_sightlines_subDLA"),
    "DLA":     ("P_DLA_only",    "n_sightlines_DLA"),
}


def _enumerate_snaps(sim_dir: Path) -> list[tuple[str, Path, float]]:
    """List completed snap dirs that have a per-class h5 present."""
    out = []
    if not sim_dir.is_dir():
        return out
    for p in sorted(sim_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("snap_"):
            continue
        pc = p / "p1d_per_class.h5"
        meta = p / "meta.json"
        if not (pc.exists() and meta.exists() and (p / "done").exists()):
            continue
        try:
            z = float(json.load(open(meta))["z"])
        except Exception:
            continue
        out.append((p.name, p, z))
    return out


def _fit_one_snap(pc_path: Path, z: float) -> dict | None:
    """Fit Rogers α for a single snap's p1d_per_class.h5.  Returns the
    result dict or None on failure."""
    try:
        with h5py.File(pc_path, "r") as f:
            k_cyc = f["k"][:]
            P_clean = f["P_clean"][:]
            n_clean = int(f["n_sightlines_clean"][()])
            pieces = {}
            n_tot = n_clean
            for cls, (pkey, nkey) in CLASS_KEYS.items():
                pieces[cls] = (f[pkey][:], int(f[nkey][()]))
                n_tot += pieces[cls][1]
    except Exception as exc:
        return {"error": str(exc)}

    if n_tot <= 0 or P_clean.size == 0:
        return {"error": "empty snap"}

    # Reconstruct "HCD-dirty" P1D as sightline-weighted mean of the per-
    # class contributions (each already averaged over its own subset).
    P_total = (n_clean / n_tot) * P_clean
    for cls, (p, n) in pieces.items():
        P_total = P_total + (n / n_tot) * p

    # PRIYA angular k
    k_ang = 2.0 * np.pi * k_cyc
    sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX) & (k_cyc > 0)
    # Drop non-positive P_clean bins (no forest power → undefined ratio)
    sel &= (P_clean > 0)
    sel &= (P_total > 0)
    if not sel.any():
        return {"error": "no k bins in window"}

    res = fit_alpha(
        k_angular=k_ang[sel],
        P_total=P_total[sel],
        P_forest=P_clean[sel],
        z=z,
    )
    # Extract errors if present
    alpha_err = res.get("alpha_err")
    if alpha_err is None:
        alpha_err = np.full(4, np.nan)
    return {
        "alpha": np.asarray(res["alpha"], dtype=np.float64),
        "alpha_err": np.asarray(alpha_err, dtype=np.float64),
        "chi2": float(res.get("chi2", np.nan)),
        "dof": int(sel.sum() - 4),
        "k_ang": k_ang[sel],
        "ratio_obs": np.asarray(res["ratio_obs"], dtype=np.float64),
        "ratio_fit": np.asarray(res["ratio_fit"], dtype=np.float64),
        "z": z,
    }


def _fit_sim(sim: str, sim_dir: Path, out_dir: Path, force: bool) -> tuple[int, int]:
    """Fit every snap for one sim; write a per-sim HDF5.  Returns (n_ok, n_skip)."""
    per_sim_h5 = out_dir / f"{sim}.h5"
    existing: set[str] = set()
    if per_sim_h5.exists() and not force:
        try:
            with h5py.File(per_sim_h5, "r") as f:
                existing = set(f.keys())
        except Exception:
            existing = set()

    n_ok = n_skip = n_fail = 0
    with h5py.File(per_sim_h5, "a") as f:
        for snap_name, snap_dir, z in _enumerate_snaps(sim_dir):
            if snap_name in existing:
                n_skip += 1
                continue
            res = _fit_one_snap(snap_dir / "p1d_per_class.h5", z)
            if res is None or "error" in res:
                n_fail += 1
                continue
            g = f.create_group(snap_name)
            g.create_dataset("alpha",     data=res["alpha"])
            g.create_dataset("alpha_err", data=res["alpha_err"])
            g.create_dataset("k_ang",     data=res["k_ang"])
            g.create_dataset("ratio_obs", data=res["ratio_obs"])
            g.create_dataset("ratio_fit", data=res["ratio_fit"])
            g.attrs["z"]    = res["z"]
            g.attrs["chi2"] = res["chi2"]
            g.attrs["dof"]  = res["dof"]
            n_ok += 1
    return n_ok, n_skip


def _scan_and_fit(suite_root: Path, label: str, out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = sorted(p for p in suite_root.iterdir()
                  if p.is_dir() and p.name.startswith("ns"))
    print(f"\n[{label}] fitting {len(sims)} sims …")
    total_ok = total_skip = 0
    for i, sim_dir in enumerate(sims, 1):
        ok, skip = _fit_sim(sim_dir.name, sim_dir, out_dir, force)
        total_ok += ok; total_skip += skip
        if i % 10 == 0 or i == len(sims):
            print(f"  {i:3d}/{len(sims)}  |  ok={total_ok}  skip={total_skip}")


def _build_master_summary(base_dir: Path, out_path: Path) -> None:
    """Flatten all per-sim HDF5s under base_dir/{lf,hr}/ into a single
    HDF5 with 1-D arrays.  Skips LF/HR group merging — each sim appears
    once per (snap, suite)."""
    rows = []
    for suite in ["lf", "hr"]:
        d = base_dir / suite
        if not d.is_dir():
            continue
        for per_sim in sorted(d.glob("*.h5")):
            try:
                with h5py.File(per_sim, "r") as f:
                    for snap_name, g in f.items():
                        rows.append({
                            "suite": suite,
                            "sim": per_sim.stem,
                            "snap": snap_name,
                            "z": float(g.attrs["z"]),
                            "chi2": float(g.attrs["chi2"]),
                            "dof": int(g.attrs["dof"]),
                            "alpha":     g["alpha"][:],
                            "alpha_err": g["alpha_err"][:],
                        })
            except Exception as exc:
                print(f"  skip {per_sim}: {exc}")

    if not rows:
        print("No rows to summarise.")
        return

    n = len(rows)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("suite", data=np.array([r["suite"] for r in rows], dtype="S"))
        f.create_dataset("sim",   data=np.array([r["sim"]   for r in rows], dtype="S"))
        f.create_dataset("snap",  data=np.array([r["snap"]  for r in rows], dtype="S"))
        f.create_dataset("z",     data=np.array([r["z"]     for r in rows], dtype=np.float64))
        f.create_dataset("chi2",  data=np.array([r["chi2"]  for r in rows], dtype=np.float64))
        f.create_dataset("dof",   data=np.array([r["dof"]   for r in rows], dtype=np.int32))
        a_arr = np.stack([r["alpha"] for r in rows])
        e_arr = np.stack([r["alpha_err"] for r in rows])
        f.create_dataset("alpha",      data=a_arr)
        f.create_dataset("alpha_err",  data=e_arr)
        f.attrs["columns_alpha"] = "LLS, Sub, Small, Large"
        f.attrs["n_rows"] = n
    print(f"  wrote master summary with {n} rows: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["lf", "hr", "both"], default="both")
    ap.add_argument("--force", action="store_true",
                    help="refit even if the snap is already in the per-sim HDF5")
    args = ap.parse_args()

    out_base = data_dir() / "rogers_alpha"
    print(f"Output dir: {out_base}")

    if args.suite in ("lf", "both"):
        _scan_and_fit(SCRATCH, label="LF", out_dir=out_base / "lf", force=args.force)
    if args.suite in ("hr", "both"):
        _scan_and_fit(SCRATCH / "hires", label="HR", out_dir=out_base / "hr", force=args.force)

    # Master flat summary
    summary_path = data_dir() / "rogers_alpha_summary.h5"
    _build_master_summary(out_base, summary_path)


if __name__ == "__main__":
    main()
