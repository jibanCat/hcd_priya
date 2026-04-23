"""
Build a per-(sim, z) summary of HCD-related scalars from the fresh
catalogs on /scratch/.../hcd_outputs/, for both LF (60 sims) and
HiRes (4 sims) campaigns.

For each (sim, snap) we record:
  - counts   : absorbers per class (LLS / subDLA / DLA)
  - sum_NHI  : Σ N_HI per class (cm⁻²)
  - dX_total : total absorption path across all sightlines
  - dndx     : counts / dX per class
  - Omega_HI : Σ N_HI · m_H · H_0 / (c · ρ_crit,0 · dX_total)  per class
  - params   : 9 PRIYA emulator parameters parsed from the sim name
  - meta     : z, hubble, box_kpc_h, n_skewers, omegam (from meta.json)

Outputs: figures/analysis/hcd_summary_lf.h5, hcd_summary_hr.h5

Usage:
    python3 scripts/build_hcd_summary.py

Tested against the catalog/meta format in place as of 2026-04-22.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hcd_analysis.cddf import absorption_path_per_sightline


# -----------------------------------------------------------------------------
# Physical constants (cgs), used for Omega_HI
# -----------------------------------------------------------------------------

_M_H_G = 1.67353e-24      # hydrogen atom mass (g)
_C_CM_S = 2.99792458e10   # speed of light (cm/s)
_MPC_CM = 3.0857e24       # 1 Mpc in cm
# H_0 in 1/s = h · 100 km/s/Mpc = h · 1e7 cm/s / (3.0857e24 cm/Mpc)
#            = 3.2408e-18 h /s
_H0_PER_H_1_S = 1.0e7 / _MPC_CM   # = 3.2408e-18  (H0 prefactor / h)
# ρ_crit,0 = 3 H_0^2 / (8πG) = 1.87847e-29 h^2 g/cm^3
_RHO_CRIT_0_PER_H2 = 1.87847e-29  # g/cm^3 per h²

LLS_MIN, SUBDLA_MIN, DLA_MIN = 17.2, 19.0, 20.3
CLASSES = [("LLS", LLS_MIN, SUBDLA_MIN),
           ("subDLA", SUBDLA_MIN, DLA_MIN),
           ("DLA", DLA_MIN, 30.0)]

_SIM_PARAM = {
    "ns":         r"ns([0-9.]+)",
    "Ap":         r"Ap([0-9.e+-]+)",
    "herei":      r"herei([0-9.]+)",
    "heref":      r"heref([0-9.]+)",
    "alphaq":     r"alphaq([0-9.]+)",
    "hub":        r"hub([0-9.]+)",
    "omegamh2":   r"omegamh2([0-9.]+)",
    "hireionz":   r"hireionz([0-9.]+)",
    "bhfeedback": r"bhfeedback([0-9.]+)",
}
PARAM_KEYS = list(_SIM_PARAM)


def parse_params(sim: str) -> dict:
    out = {}
    for key, pat in _SIM_PARAM.items():
        m = re.search(pat, sim)
        if m:
            out[key] = float(m.group(1))
    return out


def omega_hi_from_catalog(sum_NHI_cm2: float, dX_total: float, hubble: float) -> float:
    """
    Ω_HI = (m_H · H_0 / c) / ρ_crit,0 · Σ N_HI / ΔX

    Σ N_HI and ΔX are summed across all sightlines (per class).  The
    prefactor uses H_0 = 100 h km/s/Mpc, ρ_crit,0 = 1.87847e-29 h² g/cm³,
    so the combination is (const / h) independent of h otherwise.
    """
    if dX_total <= 0.0:
        return float("nan")
    # H_0 / c in cm^-1 · /h   → divide by h^2 from rho_crit gives /h final
    # prefactor (cgs): (m_H · H_0 / c) / ρ_crit,0
    prefactor = (_M_H_G * (_H0_PER_H_1_S * hubble)) / (
        _C_CM_S * (_RHO_CRIT_0_PER_H2 * hubble ** 2)
    )
    return prefactor * sum_NHI_cm2 / dX_total


# -----------------------------------------------------------------------------
# Per-(sim, snap) record builder
# -----------------------------------------------------------------------------

def build_record(sim_dir: Path, snap_dir: Path) -> dict | None:
    meta_p = snap_dir / "meta.json"
    cat_p = snap_dir / "catalog.npz"
    done_p = snap_dir / "done"
    if not (done_p.exists() and meta_p.exists() and cat_p.exists()):
        return None

    meta = json.load(open(meta_p))

    try:
        d = np.load(cat_p, allow_pickle=True)
        NHI = d["NHI"].astype(np.float64)
    except Exception:
        return None
    logN = np.log10(np.maximum(NHI, 1.0))

    # Ω_m per sim: use ics omega0 if available, else default to 0.31
    ics = meta.get("sim_ics", {})
    omega_m = float(ics.get("omega0", 0.31))
    omega_l = 1.0 - omega_m
    z = float(meta["z"])
    hubble = float(meta["hubble"])
    box_kpc_h = float(meta["box_kpc_h"])
    n_skewers = int(meta["n_skewers"])

    dX_per_sightline = absorption_path_per_sightline(
        box_kpc_h, hubble, omega_m, omega_l, z
    )
    dX_total = dX_per_sightline * n_skewers

    # Per-class aggregates
    counts = {}
    sum_NHI = {}
    dndx = {}
    OmegaHI = {}
    for cls, lo, hi in CLASSES:
        mask = (logN >= lo) & (logN < hi)
        counts[cls] = int(mask.sum())
        sum_NHI[cls] = float(NHI[mask].sum()) if mask.any() else 0.0
        dndx[cls] = counts[cls] / dX_total if dX_total > 0 else np.nan
        OmegaHI[cls] = omega_hi_from_catalog(sum_NHI[cls], dX_total, hubble)

    # Total (all HCDs above LLS_MIN)
    mask_tot = logN >= LLS_MIN
    counts["total"] = int(mask_tot.sum())
    sum_NHI["total"] = float(NHI[mask_tot].sum()) if mask_tot.any() else 0.0
    dndx["total"] = counts["total"] / dX_total if dX_total > 0 else np.nan
    OmegaHI["total"] = omega_hi_from_catalog(sum_NHI["total"], dX_total, hubble)

    params = parse_params(sim_dir.name)

    return {
        "sim": sim_dir.name,
        "snap": int(meta["snap"]),
        "z": z,
        "hubble": hubble,
        "omega_m": omega_m,
        "box_kpc_h": box_kpc_h,
        "n_skewers": n_skewers,
        "dX_per_sightline": dX_per_sightline,
        "dX_total": dX_total,
        "counts": counts,
        "sum_NHI": sum_NHI,
        "dndx": dndx,
        "Omega_HI": OmegaHI,
        "params": params,
    }


def scan_root(root: Path) -> list[dict]:
    recs = []
    for sim_dir in sorted(root.iterdir()):
        if not sim_dir.is_dir() or not sim_dir.name.startswith("ns"):
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            if not snap_dir.is_dir() or not snap_dir.name.startswith("snap_"):
                continue
            r = build_record(sim_dir, snap_dir)
            if r is not None:
                recs.append(r)
    return recs


def save_hdf5(recs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    sim_names = np.array([r["sim"] for r in recs], dtype="S")
    snaps = np.array([r["snap"] for r in recs], dtype=np.int32)
    zs = np.array([r["z"] for r in recs], dtype=np.float64)
    hubbles = np.array([r["hubble"] for r in recs], dtype=np.float64)
    omega_ms = np.array([r["omega_m"] for r in recs], dtype=np.float64)
    box = np.array([r["box_kpc_h"] for r in recs], dtype=np.float64)
    nsk = np.array([r["n_skewers"] for r in recs], dtype=np.int64)
    dX_tot = np.array([r["dX_total"] for r in recs], dtype=np.float64)
    dX_per = np.array([r["dX_per_sightline"] for r in recs], dtype=np.float64)

    with h5py.File(path, "w") as f:
        f.create_dataset("sim", data=sim_names)
        f.create_dataset("snap", data=snaps)
        f.create_dataset("z", data=zs)
        f.create_dataset("hubble", data=hubbles)
        f.create_dataset("omega_m", data=omega_ms)
        f.create_dataset("box_kpc_h", data=box)
        f.create_dataset("n_skewers", data=nsk)
        f.create_dataset("dX_total", data=dX_tot)
        f.create_dataset("dX_per_sightline", data=dX_per)

        for cls in ["LLS", "subDLA", "DLA", "total"]:
            f.create_dataset(f"counts/{cls}",
                             data=np.array([r["counts"][cls] for r in recs], dtype=np.int64))
            f.create_dataset(f"sum_NHI/{cls}",
                             data=np.array([r["sum_NHI"][cls] for r in recs], dtype=np.float64))
            f.create_dataset(f"dndx/{cls}",
                             data=np.array([r["dndx"][cls] for r in recs], dtype=np.float64))
            f.create_dataset(f"Omega_HI/{cls}",
                             data=np.array([r["Omega_HI"][cls] for r in recs], dtype=np.float64))

        for pk in PARAM_KEYS:
            col = np.array([r["params"].get(pk, np.nan) for r in recs], dtype=np.float64)
            f.create_dataset(f"params/{pk}", data=col)

        f.attrs["n_records"] = n
        f.attrs["class_defs"] = f"LLS=[{LLS_MIN},{SUBDLA_MIN}), subDLA=[{SUBDLA_MIN},{DLA_MIN}), DLA>={DLA_MIN}"


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import data_dir
    DATA = data_dir()
    SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")

    print("Scanning LF (60-sim) root…")
    lf_recs = scan_root(SCRATCH)
    print(f"  {len(lf_recs)} (sim, snap) records")
    save_hdf5(lf_recs, DATA / "hcd_summary_lf.h5")
    print(f"  saved {DATA / 'hcd_summary_lf.h5'}")

    print("Scanning HiRes root…")
    hr_recs = scan_root(SCRATCH / "hires")
    print(f"  {len(hr_recs)} (sim, snap) records")
    save_hdf5(hr_recs, DATA / "hcd_summary_hr.h5")
    print(f"  saved {DATA / 'hcd_summary_hr.h5'}")


if __name__ == "__main__":
    main()
