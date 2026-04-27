"""
Validate τ-peak DLA finder against particle-based colden truth.

Walks every `rand_spectra_DLA.hdf5` under the HiRes tree and, for each file:

1. Reads `colden/H/1` and identifies "truth" DLAs by contiguous-pixel
   integration above 2 × 10²⁰ cm⁻² (`hcd_analysis.dla_truth`).
2. Reads `tau/H/1/1215` and runs the production τ-peak finder
   (`catalog.find_systems_in_skewer` + `catalog.process_skewer_batch`)
   with the production parameters from `config/default.yaml` (fast_mode,
   τ-threshold=100, merge_dv_kms=100, min_log_nhi=17.2).
3. Matches truth ↔ recovered with a ±max(merge_dv_kms / dv_pix, 5 px)
   tolerance and computes per-(sim, snap) summary stats:
       N_truth, N_recovered_DLA, N_matched, completeness, purity,
       mean Δlog NHI, σ Δlog NHI.
4. Aggregates into an HDF5 summary at
       figures/analysis/data/dla_truth_summary.h5
   and a multi-panel scatter PNG at
       figures/analysis/05_truth_validation/dla_truth_recovery.png.

Run
---
    python3 scripts/validate_dla_truth.py             # all 4 hires sims
    python3 scripts/validate_dla_truth.py --max-sims 1  # debug
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.catalog import (
    LOG_NHI_DLA_MIN,
    LOG_NHI_LLS_MIN,
    classify_system,
    process_skewer_batch,
)
from hcd_analysis.dla_truth import (
    RecoveredDLA,
    find_truth_dlas_from_colden,
    match_dla_lists,
    summary_stats,
)
from hcd_analysis.io import pixel_dv_kms, SpectraHeader

# Production absorber parameters (from config/default.yaml)
TAU_THRESHOLD = 100.0
MERGE_DV_KMS = 100.0
MIN_PIXELS = 2
B_INIT = 30.0
B_BOUNDS = (1.0, 300.0)
TAU_FIT_CAP = 1.0e6
VOIGT_MAX_ITER = 200
FAST_MODE = True
MIN_LOG_NHI = 17.2

# Truth-DLA finder parameters
DLA_THRESHOLD = 2.0e20      # canonical Wolfe et al. — sets "is this a DLA?"
PIXEL_FLOOR = 1.0e17        # well below LLS threshold for run stitching
TRUTH_MERGE_GAP_PX = 0      # strict contiguity for truth-side runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIRES_ROOT = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires")


def discover_rand_spectra() -> List[Tuple[str, int, Path]]:
    """Return sorted list of (sim_name, snap, path) for every rand_spectra_DLA."""
    out = []
    for sim_dir in sorted(HIRES_ROOT.iterdir()):
        if not sim_dir.is_dir():
            continue
        out_dir = sim_dir / "output"
        if not out_dir.is_dir():
            continue
        for spec_dir in sorted(out_dir.glob("SPECTRA_*")):
            f = spec_dir / "rand_spectra_DLA.hdf5"
            if f.is_file():
                try:
                    snap = int(spec_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                out.append((sim_dir.name, snap, f))
    return out


def read_rand_spectra_header(path: Path) -> SpectraHeader:
    """SpectraHeader for a rand_spectra HDF5 (same format as grid file)."""
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        tau_ds = f["tau/H/1/1215"]
        n_skewers = tau_ds.shape[0]
        redshift = float(h["redshift"])
        hubble = float(h["hubble"])
        omegam = float(h["omegam"])
        omegal = float(h["omegal"])
        Hz = float(h["Hz"]) if "Hz" in h else (
            100.0 * hubble * (omegam * (1 + redshift) ** 3 + omegal) ** 0.5
        )
        return SpectraHeader(
            redshift=redshift, Hz=Hz, box=float(h["box"]),
            hubble=hubble, nbins=int(h["nbins"]),
            omegam=omegam, omegal=omegal, omegab=float(h["omegab"]),
            n_skewers=n_skewers,
        )


def run_tau_finder(
    tau: np.ndarray,
    dv_kms: float,
) -> List[RecoveredDLA]:
    """
    Run the production τ-peak DLA finder on a (n_skewers, nbins) τ array
    and return RecoveredDLA records (LLS-or-stronger).
    """
    absorbers = process_skewer_batch(
        tau_batch=tau,
        batch_start=0,
        dv_kms=dv_kms,
        tau_threshold=TAU_THRESHOLD,
        merge_dv_kms=MERGE_DV_KMS,
        min_pixels=MIN_PIXELS,
        b_init=B_INIT,
        b_bounds=B_BOUNDS,
        tau_fit_cap=TAU_FIT_CAP,
        max_iter=VOIGT_MAX_ITER,
        fast_mode=FAST_MODE,
        min_log_nhi=MIN_LOG_NHI,
    )
    rec: List[RecoveredDLA] = []
    for a in absorbers:
        rec.append(
            RecoveredDLA(
                skewer_idx=a.skewer_idx,
                pix_start=a.pix_start,
                pix_end=a.pix_end,
                NHI_recovered=a.NHI,
                log_NHI=a.log_NHI,
                absorber_class=a.absorber_class,
            )
        )
    return rec


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def process_file(
    sim_name: str, snap: int, path: Path,
) -> Dict[str, object]:
    """
    Run the truth-vs-recovered comparison on one rand_spectra_DLA.hdf5.

    Returns a dict with scalar diagnostics and the matched-pair scatter
    arrays (kept small — typically O(few × 100) DLAs per file).
    """
    header = read_rand_spectra_header(path)
    z = header.redshift
    dv_kms = pixel_dv_kms(header)

    # Tolerance: convert merge_dv_kms to pixels with a 5-px floor so that
    # very high-resolution rand_spectra (dv_pix ≈ 1 km/s → 100 px tolerance)
    # don't dominate, and very low-res (dv_pix ≈ 10 km/s → 10 px) don't shrink
    # below a sensible minimum.
    tol_pixels = max(int(round(MERGE_DV_KMS / dv_kms)), 5)

    with h5py.File(path, "r") as f:
        colden = f["colden/H/1"][:]            # (n_skew, nbins) cm^-2
        tau = f["tau/H/1/1215"][:].astype(np.float64)

    # 1) truth DLAs from colden
    truth = find_truth_dlas_from_colden(
        colden, dla_threshold=DLA_THRESHOLD, pixel_floor=PIXEL_FLOOR,
        merge_gap_pixels=TRUTH_MERGE_GAP_PX, min_pixels=1,
    )

    # 2) production τ finder, then keep DLA-class only for the primary purity
    recovered_all = run_tau_finder(tau, dv_kms)
    recovered_dla = [r for r in recovered_all if r.absorber_class == "DLA"]
    recovered_lls_or_stronger = [
        r for r in recovered_all if r.absorber_class in ("LLS", "subDLA", "DLA")
    ]

    # 3) match DLA-class recovered against truth — primary "DLA finder" gauge
    res_dla = match_dla_lists(truth, recovered_dla, tol_pixels=tol_pixels)
    stats_dla = summary_stats(truth, recovered_dla, res_dla)

    # 4) match LLS-or-stronger against truth — completeness gauge for H3
    res_loose = match_dla_lists(
        truth, recovered_lls_or_stronger, tol_pixels=tol_pixels,
    )
    stats_loose = summary_stats(truth, recovered_lls_or_stronger, res_loose)

    return {
        "sim_name": sim_name,
        "snap": snap,
        "z": z,
        "dv_kms": dv_kms,
        "n_skewers": header.n_skewers,
        "nbins": header.nbins,
        "tol_pixels": tol_pixels,
        "n_truth": len(truth),
        "n_recovered_dla": len(recovered_dla),
        "n_recovered_lls_or_stronger": len(recovered_lls_or_stronger),
        # DLA-vs-DLA bookkeeping (the "DLA finder" gauge)
        "stats_dla": stats_dla,
        # Truth-vs-LLS+ (the "did we at least flag the DLA as something?" gauge)
        "stats_loose": stats_loose,
        # Scatter arrays for the figure: only matched DLA-vs-DLA pairs
        "log_nhi_truth_matched": np.array(
            [np.log10(max(m.truth.NHI_truth, 1.0)) for m in res_dla.matched]
        ),
        "log_nhi_recovered_matched": np.array(
            [m.recovered.log_NHI for m in res_dla.matched]
        ),
        "delta_log_nhi": np.array([m.delta_log_nhi for m in res_dla.matched]),
        "delta_pix": np.array([m.delta_pix for m in res_dla.matched]),
    }


# ---------------------------------------------------------------------------
# Aggregation: HDF5 + figure
# ---------------------------------------------------------------------------

def write_summary_h5(records: List[Dict], out_path: Path) -> None:
    """Save per-file diagnostics + scatter arrays to a single HDF5."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # File-level metadata for downstream consumers
        f.attrs["script"] = "scripts/validate_dla_truth.py"
        f.attrs["dla_threshold_cm2"] = DLA_THRESHOLD
        f.attrs["pixel_floor_cm2"] = PIXEL_FLOOR
        f.attrs["tau_threshold"] = TAU_THRESHOLD
        f.attrs["merge_dv_kms"] = MERGE_DV_KMS
        f.attrs["min_log_nhi"] = MIN_LOG_NHI
        f.attrs["fast_mode"] = int(FAST_MODE)
        f.attrs["n_files"] = len(records)

        # Flat summary table (one row per file)
        n = len(records)
        cols = {
            "sim_name": np.array([r["sim_name"] for r in records], dtype="S128"),
            "snap": np.array([r["snap"] for r in records], dtype=np.int32),
            "z": np.array([r["z"] for r in records], dtype=np.float64),
            "dv_kms": np.array([r["dv_kms"] for r in records], dtype=np.float64),
            "n_skewers": np.array([r["n_skewers"] for r in records], dtype=np.int32),
            "tol_pixels": np.array([r["tol_pixels"] for r in records], dtype=np.int32),
            "n_truth": np.array([r["n_truth"] for r in records], dtype=np.int32),
            "n_recovered_dla": np.array([r["n_recovered_dla"] for r in records], dtype=np.int32),
            "n_recovered_lls_or_stronger": np.array(
                [r["n_recovered_lls_or_stronger"] for r in records], dtype=np.int32
            ),
            "n_matched_dla": np.array(
                [r["stats_dla"]["N_matched"] for r in records], dtype=np.int32
            ),
            "n_matched_loose": np.array(
                [r["stats_loose"]["N_matched"] for r in records], dtype=np.int32
            ),
            "completeness_dla": np.array(
                [r["stats_dla"]["completeness"] for r in records], dtype=np.float64
            ),
            "completeness_loose": np.array(
                [r["stats_loose"]["completeness"] for r in records], dtype=np.float64
            ),
            "purity_dla": np.array(
                [r["stats_dla"]["purity"] for r in records], dtype=np.float64
            ),
            "mean_dlog_nhi": np.array(
                [r["stats_dla"]["mean_dlog_nhi"] for r in records], dtype=np.float64
            ),
            "std_dlog_nhi": np.array(
                [r["stats_dla"]["std_dlog_nhi"] for r in records], dtype=np.float64
            ),
            "median_dlog_nhi": np.array(
                [r["stats_dla"]["median_dlog_nhi"] for r in records], dtype=np.float64
            ),
        }
        g = f.create_group("summary")
        for k, v in cols.items():
            g.create_dataset(k, data=v)

        # Scatter arrays — variable length, store per (sim, snap) sub-group
        scatter = f.create_group("scatter")
        for r in records:
            key = f"{r['sim_name']}__snap_{r['snap']:03d}"
            sg = scatter.create_group(key)
            sg.attrs["sim_name"] = r["sim_name"]
            sg.attrs["snap"] = r["snap"]
            sg.attrs["z"] = r["z"]
            sg.create_dataset("log_nhi_truth", data=r["log_nhi_truth_matched"])
            sg.create_dataset("log_nhi_recovered", data=r["log_nhi_recovered_matched"])
            sg.create_dataset("delta_log_nhi", data=r["delta_log_nhi"])
            sg.create_dataset("delta_pix", data=r["delta_pix"])

    print(f"  wrote {out_path}")


def make_recovery_figure(records: List[Dict], out_path: Path) -> None:
    """Multi-panel scatter: NHI_recovered vs NHI_truth, one panel per sim."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sims = sorted({r["sim_name"] for r in records})
    n = len(sims)
    if n == 0:
        print("  no records to plot")
        return
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 5.5 * nrows), squeeze=False,
    )

    # Common z-colour scale across panels for cross-sim comparison
    all_z = [r["z"] for r in records if r["log_nhi_truth_matched"].size]
    z_lo = min(all_z) if all_z else 2.0
    z_hi = max(all_z) if all_z else 6.0

    for k, sim in enumerate(sims):
        ax = axes[k // ncols][k % ncols]
        sim_recs = [r for r in records if r["sim_name"] == sim]
        for r in sim_recs:
            x = r["log_nhi_truth_matched"]
            y = r["log_nhi_recovered_matched"]
            if x.size == 0:
                continue
            ax.scatter(
                x, y, c=[r["z"]] * x.size, vmin=z_lo, vmax=z_hi,
                cmap="viridis", alpha=0.5, s=10, edgecolors="none",
            )
        # 1:1 line
        lo, hi = 20.0, 22.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"log$_{10}$ N$_{\rm HI}^{\rm truth}$ [cm$^{-2}$]")
        ax.set_ylabel(r"log$_{10}$ N$_{\rm HI}^{\rm recovered}$ [cm$^{-2}$]")
        ax.set_title(sim[:48], fontsize=9)
        ax.grid(True, alpha=0.25)
        # in-panel summary text
        all_dlog = np.concatenate(
            [r["delta_log_nhi"] for r in sim_recs if r["delta_log_nhi"].size]
        ) if sim_recs else np.array([])
        if all_dlog.size:
            ax.text(
                0.04, 0.96,
                f"N={all_dlog.size}\n"
                f"⟨Δlog⟩={np.mean(all_dlog):+.3f}\n"
                f"σ={np.std(all_dlog, ddof=1):.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

    # Hide any unused panels
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    # Single colour-bar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=z_lo, vmax=z_hi))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="redshift z", shrink=0.7)
    fig.suptitle(
        "τ-peak finder NHI vs particle-based truth (rand_spectra_DLA.hdf5)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-sims", type=int, default=None, help="debug: cap sim count")
    p.add_argument("--max-files", type=int, default=None, help="debug: cap file count")
    args = p.parse_args()

    files = discover_rand_spectra()
    if args.max_sims:
        keep_sims = sorted({s for s, _, _ in files})[: args.max_sims]
        files = [f for f in files if f[0] in set(keep_sims)]
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print("No rand_spectra_DLA.hdf5 files found under", HIRES_ROOT)
        sys.exit(1)

    print(f"Found {len(files)} rand_spectra_DLA.hdf5 files across "
          f"{len(set(s for s, _, _ in files))} sims")

    records: List[Dict] = []
    t0 = time.time()
    for k, (sim, snap, path) in enumerate(files, 1):
        ts = time.time()
        try:
            rec = process_file(sim, snap, path)
        except Exception as exc:
            print(f"  [{k:3d}/{len(files)}] {sim[:30]} snap_{snap:03d}: ERROR {exc}")
            continue
        records.append(rec)
        s_dla = rec["stats_dla"]
        s_loose = rec["stats_loose"]
        print(
            f"  [{k:3d}/{len(files)}] {sim[:30]} snap_{snap:03d} z={rec['z']:.2f} "
            f"({time.time()-ts:.1f}s): "
            f"N_truth={rec['n_truth']:4d} N_rec_DLA={rec['n_recovered_dla']:4d} "
            f"comp={s_dla['completeness']:.2f} comp_loose={s_loose['completeness']:.2f} "
            f"pure={s_dla['purity']:.2f} ⟨Δlog⟩={s_dla['mean_dlog_nhi']:+.3f} "
            f"σ={s_dla['std_dlog_nhi']:.3f}"
        )

    print(f"\nProcessed {len(records)} files in {time.time()-t0:.1f}s")

    if records:
        out_h5 = REPO_ROOT / "figures" / "analysis" / "data" / "dla_truth_summary.h5"
        write_summary_h5(records, out_h5)
        out_png = (
            REPO_ROOT / "figures" / "analysis" / "05_truth_validation"
            / "dla_truth_recovery.png"
        )
        make_recovery_figure(records, out_png)


if __name__ == "__main__":
    main()
