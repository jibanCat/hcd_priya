"""
Per-simulation cosmic-variance bootstrap for dN/dX per class at z ≈ 3.

Replaces the earlier across-sim bootstrap (which averaged out cosmic
variance).  For each LF simulation we resample sightlines with
replacement using `catalog.skewer_idx`, giving the cosmic-variance
error on dN/dX at that sim's cosmological parameter point.

Also highlights a "fiducial slice" of sims nearest the eBOSS PRIYA
best-fit (A_p ≈ 1.7e-9, n_s ≈ 1.0, Fernandez+2024 / Bourboux+2020),
and reports whether the fiducial subset still sits significantly
below the observational dN/dX(DLA).

Outputs
-------
  figures/analysis/bootstrap_dndx_per_sim.png  — scatter + error bars + obs
  figures/analysis/bootstrap_dndx_per_sim.csv  — per-sim estimate + σ_cosmic

Run:
    python3 scripts/bootstrap_dndx_per_sim.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hcd_analysis.cddf import absorption_path_per_sightline

SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
import sys as _sys_common
_sys_common.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir
DATA = data_dir()
OUT = ROOT / "figures" / "analysis" / "04_hcd_mf"
OUT.mkdir(parents=True, exist_ok=True)
# Classes
LLS_MIN, SUBDLA_MIN, DLA_MIN = 17.2, 19.0, 20.3
CLASSES = [("LLS", LLS_MIN, SUBDLA_MIN),
           ("subDLA", SUBDLA_MIN, DLA_MIN),
           ("DLA", DLA_MIN, 30.0)]

# PRIYA eBOSS best-fit (Fernandez+2024, Bourboux+2020 follow-up):
# A_p ≈ 1.7e-9, n_s ≈ 1.0.  Fiducial slice = sims nearest this point.
FIDUCIAL_AP = 1.7e-9
FIDUCIAL_NS = 1.0
# Fiducial slice sizes to check robustness: stability of the mean dN/dX(DLA)
# as we widen from 10 → 30 sims tells us whether the fiducial answer is
# driven by the 2-3 nearest points or is robust to slice-width choice.
N_FIDUCIAL_LIST = [10, 15, 20, 30]
N_FIDUCIAL = N_FIDUCIAL_LIST[0]   # default slice used in the scatter figure

_PARAM_PATTERNS = {
    "ns": r"ns([0-9.]+)",
    "Ap": r"Ap([0-9.e+-]+)",
    "herei": r"herei([0-9.]+)",
    "heref": r"heref([0-9.]+)",
    "alphaq": r"alphaq([0-9.]+)",
    "hub": r"hub([0-9.]+)",
    "omegamh2": r"omegamh2([0-9.]+)",
    "hireionz": r"hireionz([0-9.]+)",
    "bhfeedback": r"bhfeedback([0-9.]+)",
}


def parse_params(sim: str) -> dict:
    return {k: float(re.search(p, sim).group(1))
            for k, p in _PARAM_PATTERNS.items()
            if re.search(p, sim)}


# Observational dN/dX(DLA) at z ≈ 3 (sbird/dla_data verbatim)
OBS_Z3_DLA = {
    "PW09 z∈[2.7,3.0]": (0.067, 0.006),
    "PW09 z∈[3.0,3.5]": (0.084, 0.006),
    "N12 z=3.05":        (0.0805, None),
    "Ho21 z=3.08":       (0.0706, 0.0022),
    "Ho21 z=3.25":       (0.0748, 0.0028),
}


def per_sim_bootstrap(
    skewer_idx: np.ndarray,
    logN: np.ndarray,
    n_skewers: int,
    dX_per_sightline: float,
    n_boot: int = 0,          # retained for API compatibility; ignored
    rng: np.random.Generator | None = None,  # unused with analytical formula
) -> dict:
    """
    Per-sim cosmic-variance error on dN/dX via analytical CLT:

        If c_i is the count of absorbers on sightline i, then the
        nonparametric bootstrap of n_skewers draws-with-replacement
        gives σ(Σ c_b) = √n_skewers · std(c), so σ(dN/dX) =
        std(c) / (√n_skewers · dX_per_sightline).

    This is identical to the finite-sample bootstrap in the n_boot → ∞
    limit and avoids the O(n_skewers · n_boot · n_class) resampling cost.
    """
    results = {cls: {"point": 0.0, "sigma": 0.0} for cls, _, _ in CLASSES}

    # Per-sightline counts as shape (n_skewers, n_class)
    n_cls = len(CLASSES)
    # skewer_idx may be 1-based; normalise so every index lands in [0, n_skewers).
    offset = int(skewer_idx.min())
    sk_all = skewer_idx - offset
    # Defensive clip to avoid IndexError if offset logic ever changes.
    sk_all = np.clip(sk_all, 0, n_skewers - 1)

    counts = np.zeros((n_skewers, n_cls), dtype=np.int32)
    for j, (_, lo, hi) in enumerate(CLASSES):
        m = (logN >= lo) & (logN < hi)
        if not m.any():
            continue
        np.add.at(counts[:, j], sk_all[m], 1)

    dX_total = n_skewers * dX_per_sightline
    point = counts.sum(axis=0)                  # totals per class
    # σ(sum over n_skewers resampled sightlines) = √n_skewers · std(c)
    std_per_sightline = counts.std(axis=0, ddof=1)
    sigma_sum = np.sqrt(n_skewers) * std_per_sightline
    sigma_dndx = sigma_sum / dX_total

    for j, (cls, _, _) in enumerate(CLASSES):
        results[cls]["point"] = float(point[j] / dX_total)
        results[cls]["sigma"] = float(sigma_dndx[j])
    return results


def z3_records_with_catalog(z_tol=0.05):
    """Yield (sim_name, params, z, meta, catalog_npz) at z ≈ 3 per LF sim."""
    for sim_dir in sorted(SCRATCH.iterdir()):
        if sim_dir.name in {"hires"} or not sim_dir.name.startswith("ns"):
            continue
        params = parse_params(sim_dir.name)
        for snap_dir in sorted(sim_dir.iterdir()):
            if not snap_dir.name.startswith("snap_"):
                continue
            if not (snap_dir / "done").exists():
                continue
            meta_p = snap_dir / "meta.json"
            cat_p = snap_dir / "catalog.npz"
            if not (meta_p.exists() and cat_p.exists()):
                continue
            meta = json.load(open(meta_p))
            z = float(meta["z"])
            if abs(z - 3.0) > z_tol:
                continue
            yield (sim_dir.name, params, z, meta, cat_p)
            break   # one snap per sim


def main():
    print("Loading LF catalogs at z ≈ 3…")
    rng = np.random.default_rng(42)
    rows = []
    for sim, params, z, meta, cat_p in z3_records_with_catalog():
        d = np.load(cat_p, allow_pickle=True)
        skewer_idx = d["skewer_idx"].astype(np.int64)
        NHI = d["NHI"].astype(np.float64)
        logN = np.log10(np.maximum(NHI, 1.0))

        ics = meta.get("sim_ics", {})
        omega_m = float(ics.get("omega0", 0.31))
        omega_l = 1.0 - omega_m
        dX_per_sl = absorption_path_per_sightline(
            meta["box_kpc_h"], meta["hubble"], omega_m, omega_l, z
        )

        r = per_sim_bootstrap(
            skewer_idx=skewer_idx,
            logN=logN,
            n_skewers=int(meta["n_skewers"]),
            dX_per_sightline=dX_per_sl,
            n_boot=500,
            rng=rng,
        )
        rows.append({
            "sim": sim,
            "Ap": params["Ap"],
            "ns": params["ns"],
            "z": z,
            "dX_per_sightline": dX_per_sl,
            "dndx_LLS": r["LLS"]["point"],
            "sigma_LLS": r["LLS"]["sigma"],
            "dndx_subDLA": r["subDLA"]["point"],
            "sigma_subDLA": r["subDLA"]["sigma"],
            "dndx_DLA": r["DLA"]["point"],
            "sigma_DLA": r["DLA"]["sigma"],
        })
    rows.sort(key=lambda r: r["Ap"])
    print(f"  {len(rows)} sims with catalog at z ≈ 3")

    # Fiducial slice — N_FIDUCIAL sims nearest to (A_p, n_s) = (1.7e-9, 1.0).
    # Normalise distances by each parameter's range across the suite so Euclidean
    # distance is meaningful in mixed-units (A_p, n_s) space.
    Ap_arr = np.array([r["Ap"] for r in rows])
    ns_arr = np.array([r["ns"] for r in rows])
    d = np.sqrt(
        ((Ap_arr - FIDUCIAL_AP) / (np.ptp(Ap_arr) + 1e-30))**2 +
        ((ns_arr - FIDUCIAL_NS) / (np.ptp(ns_arr) + 1e-30))**2
    )
    order = np.argsort(d)
    fid_set = set(rows[i]["sim"] for i in order[:N_FIDUCIAL])
    print(f"  fiducial slice: {N_FIDUCIAL} sims nearest "
          f"(A_p={FIDUCIAL_AP:.2e}, n_s={FIDUCIAL_NS:.2f})")
    for i in order[:N_FIDUCIAL]:
        print(f"    {rows[i]['sim'][:40]}…  A_p={rows[i]['Ap']:.2e}, n_s={rows[i]['ns']:.3f}  "
              f"dN/dX(DLA) = {rows[i]['dndx_DLA']:.4f} ± {rows[i]['sigma_DLA']:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, cls, key, col in zip(
        axes, ["LLS", "subDLA", "DLA"],
        [("dndx_LLS", "sigma_LLS"), ("dndx_subDLA", "sigma_subDLA"),
         ("dndx_DLA", "sigma_DLA")],
        ["C2", "C1", "C3"]
    ):
        Ap = np.array([r["Ap"] for r in rows])
        y = np.array([r[key[0]] for r in rows])
        e = np.array([r[key[1]] for r in rows])
        is_fid = np.array([r["sim"] in fid_set for r in rows])
        # non-fiducial
        ax.errorbar(Ap[~is_fid], y[~is_fid], yerr=e[~is_fid], fmt="o",
                    color=col, alpha=0.4, ms=5, capsize=2,
                    label=f"PRIYA {cls} (non-fiducial)")
        # fiducial
        ax.errorbar(Ap[is_fid], y[is_fid], yerr=e[is_fid], fmt="s",
                    color=col, ms=9, capsize=3, markeredgecolor="k",
                    label=f"Fiducial slice ({N_FIDUCIAL} sims nearest eBOSS BF)")
        # Annotate fiducial mean:
        fid_mean = float(y[is_fid].mean())
        fid_sigma_cosmic = float(np.sqrt(np.mean(e[is_fid] ** 2)))
        fid_sigma_spread = float(y[is_fid].std(ddof=1))
        ax.axhline(fid_mean, color="k", lw=0.8, ls="-",
                   label=f"fiducial mean = {fid_mean:.4f}")
        ax.axhline(fid_mean + fid_sigma_spread, color="k", lw=0.5, ls=":")
        ax.axhline(fid_mean - fid_sigma_spread, color="k", lw=0.5, ls=":")
        if cls == "DLA":
            for name, (val, err) in OBS_Z3_DLA.items():
                ax.axhline(val, color="gray", lw=1.0, ls="--", alpha=0.8)
                ax.text(Ap.max() * 1.02, val, name, fontsize=7, va="center")
                if err:
                    ax.axhspan(val - err, val + err, color="gray", alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("A_p")
        ax.set_ylabel(f"dN/dX ({cls})")
        ax.set_title(f"{cls} — per-sim bootstrap at z ≈ 3")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Per-sim cosmic-variance bootstrap (500 draws) of dN/dX at z ≈ 3\n"
        f"Fiducial slice: {N_FIDUCIAL} sims closest to eBOSS best-fit "
        f"(A_p={FIDUCIAL_AP:.1e}, n_s={FIDUCIAL_NS:.2f})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outp = OUT / "bootstrap_dndx_per_sim.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")

    # --- CSV ---
    import csv as _csv
    csv_p = DATA / "bootstrap_dndx_per_sim.csv"
    with open(csv_p, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["in_fiducial"])
        w.writeheader()
        for r in rows:
            rr = dict(r); rr["in_fiducial"] = r["sim"] in fid_set
            w.writerow(rr)
    print(f"  wrote {csv_p}")

    # --- Verdict: does the fiducial slice reach obs?  Check at several slice widths ---
    print("\n" + "=" * 72)
    print("Significance vs obs — fiducial-slice robustness check")
    print("=" * 72)
    print("Widening the fiducial slice reveals whether the under-prediction is")
    print("driven by 2-3 nearest points or is stable to slice-size choice.")
    print()

    # For each N, extract fiducial rows and compute significance
    Ap_arr_full = np.array([r["Ap"] for r in rows])
    ns_arr_full = np.array([r["ns"] for r in rows])
    d_full = np.sqrt(
        ((Ap_arr_full - FIDUCIAL_AP) / (np.ptp(Ap_arr_full) + 1e-30))**2 +
        ((ns_arr_full - FIDUCIAL_NS) / (np.ptp(ns_arr_full) + 1e-30))**2
    )
    order_full = np.argsort(d_full)

    header = f"  {'N_fid':>5s} {'<dN/dX>':>9s} {'σ_spread':>10s} {'σ_cosmic':>10s} {'σ_total':>9s}"
    for name in OBS_Z3_DLA:
        header += f" {name[:12]:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for N in N_FIDUCIAL_LIST:
        keep = [rows[i] for i in order_full[:N]]
        y = np.array([r["dndx_DLA"] for r in keep])
        e = np.array([r["sigma_DLA"] for r in keep])
        mean = float(y.mean())
        spread = float(y.std(ddof=1))
        cosmic = float(np.sqrt(np.mean(e**2)))
        total = float(np.sqrt(spread**2 + cosmic**2))
        line = (f"  {N:>5d} {mean:>9.4f} {spread:>10.4f} "
                f"{cosmic:>10.4f} {total:>9.4f}")
        for obs_name, (val, _) in OBS_Z3_DLA.items():
            delta = (val - mean) / total
            line += f" {delta:>12.2f}σ"
        print(line)
    print()
    print("σ_spread  = 1σ spread of fiducial-slice dN/dX(DLA) values across sims")
    print("σ_cosmic  = per-sim cosmic σ (CLT bootstrap), averaged over slice")
    print("σ_total   = quadrature combined  (spread dominates)")


if __name__ == "__main__":
    main()
