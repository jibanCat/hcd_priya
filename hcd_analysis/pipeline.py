"""
Pipeline orchestration: one (sim, snap), one sim all z, all sims all z.

Execution model
---------------
Level 1 – (sim, snap) pair:
  1. Load header + compute dv_kms.
  2. Build AbsorberCatalog (catalog.py).
  3. Compute all P1D variants (p1d.py).
  4. Compute P1D ratios.
  5. Measure CDDF.
  6. Optionally compute perturbed P1D.
  7. Save all outputs.

Level 2 – one sim, all z:
  Iterate over snapshots; run Level 1 for each.
  Optional intra-sim parallelism via joblib.

Level 3 – all sims, all z:
  Iterate over all sims; run Level 2 for each.
  Parallelism over sims via joblib.

Resume / idempotency
--------------------
Each (sim, snap) writes a sentinel file <output_dir>/<sim>/<snap>/done.
If this file exists and cfg.resume=True, the pair is skipped.

Failure handling
----------------
Failures are caught per (sim, snap), logged, and written to
<output_dir>/<sim>/<snap>/error.txt. The pipeline continues.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .catalog import AbsorberCatalog, build_catalog, LOG_NHI_LLS_MIN
from .cddf import CDDFPerturbation, compute_perturbed_p1d, measure_cddf
from .config import PipelineConfig, save_config
from .io import pixel_dv_kms, read_header
from .p1d import (
    ALL_VARIANTS,
    compute_all_p1d_variants,
    compute_p1d_ratios,
    _DEFAULT_K_BINS,
)
from .snapshot_map import SimSnapshot, SnapEntry, build_snapshot_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def snap_output_dir(cfg: PipelineConfig, sim_name: str, snap: int) -> Path:
    d = Path(cfg.output_root) / sim_name / f"snap_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_done(cfg: PipelineConfig, sim_name: str, snap: int) -> bool:
    sentinel = snap_output_dir(cfg, sim_name, snap) / "done"
    return sentinel.exists()


def mark_done(cfg: PipelineConfig, sim_name: str, snap: int) -> None:
    (snap_output_dir(cfg, sim_name, snap) / "done").touch()


def write_error(cfg: PipelineConfig, sim_name: str, snap: int, msg: str) -> None:
    path = snap_output_dir(cfg, sim_name, snap) / "error.txt"
    with open(path, "w") as f:
        f.write(msg)


# ---------------------------------------------------------------------------
# Level 1: one (sim, snap)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SnapResult:
    sim_name: str
    snap: int
    z: float
    dv_kms: float
    catalog: AbsorberCatalog
    p1d_variants: Dict   # variant_name → (k, p1d, mean_F)
    p1d_ratios: Dict
    cddf_result: Dict
    timing: Dict[str, float]
    perturbed_p1d: Optional[Dict] = None


def run_one_snap(
    sim_snapshot: SimSnapshot,
    entry: SnapEntry,
    cfg: PipelineConfig,
) -> Optional[SnapResult]:
    """
    Run the full analysis pipeline for one (sim, snap) pair.

    Returns SnapResult on success, None on failure (error is logged and written).
    """
    sim_name = sim_snapshot.sim.name
    snap = entry.snap
    z = entry.z
    out_dir = snap_output_dir(cfg, sim_name, snap)

    if cfg.resume and is_done(cfg, sim_name, snap):
        logger.info("SKIP (done): sim=%s snap=%03d z=%.2f", sim_name[:30], snap, z)
        return _load_snap_result(out_dir, sim_name, snap, z)

    logger.info("START: sim=%s snap=%03d z=%.2f", sim_name[:30], snap, z)
    timing: Dict[str, float] = {}

    try:
        # --- Load header --------------------------------------------------
        t0 = time.perf_counter()
        header = read_header(entry.path)
        dv_kms = pixel_dv_kms(header)
        timing["header"] = time.perf_counter() - t0

        # --- Build absorber catalog ---------------------------------------
        t0 = time.perf_counter()
        catalog_path = out_dir / "catalog.npz"

        if cfg.resume and catalog_path.exists():
            catalog = AbsorberCatalog.load_npz(catalog_path)
            logger.info("  loaded catalog from cache (%d absorbers)", len(catalog.absorbers))
        else:
            n_skewers = cfg.p1d.n_skewers if cfg.debug else None
            catalog = build_catalog(
                hdf5_path=entry.path,
                sim_name=sim_name,
                snap=snap,
                z=z,
                dv_kms=dv_kms,
                tau_threshold=cfg.absorber.tau_threshold,
                merge_dv_kms=cfg.absorber.merge_dv_kms,
                min_pixels=cfg.absorber.min_pixels,
                b_init=cfg.absorber.b_init_kms,
                b_bounds=tuple(cfg.absorber.b_bounds),
                tau_fit_cap=cfg.absorber.tau_fit_cap,
                voigt_max_iter=cfg.absorber.voigt_max_iter,
                batch_size=cfg.skewer_batch_size,
                n_skewers=n_skewers,
                fast_mode=cfg.benchmark,
                n_workers=cfg.n_workers_skewer,
                min_log_nhi=cfg.absorber.min_log_nhi,
            )
            catalog.save_npz(catalog_path)

        timing["catalog"] = time.perf_counter() - t0
        logger.info("  catalog: %s", catalog.summary())

        # --- Compute P1D variants -----------------------------------------
        t0 = time.perf_counter()
        n_skewers_p1d = cfg.p1d.n_skewers if cfg.debug else None

        p1d_variants = compute_all_p1d_variants(
            hdf5_path=entry.path,
            nbins=header.nbins,
            dv_kms=dv_kms,
            catalog=catalog,
            variants=ALL_VARIANTS,
            batch_size=cfg.skewer_batch_size,
            n_skewers=n_skewers_p1d,
            fill_strategy=cfg.absorber.mask_fill_strategy,
        )
        timing["p1d"] = time.perf_counter() - t0

        # --- P1D ratios ---------------------------------------------------
        p1d_ratios = compute_p1d_ratios(p1d_variants)

        # --- Measure CDDF -------------------------------------------------
        t0 = time.perf_counter()
        cddf_result = measure_cddf(catalog, header)
        timing["cddf"] = time.perf_counter() - t0

        # --- Perturbed P1D (if CDDF perturbation requested) ---------------
        perturbed_p1d = None
        cddf_cfg = cfg.cddf
        if cddf_cfg.A != 1.0 or cddf_cfg.alpha != 0.0:
            t0 = time.perf_counter()
            perturbation = CDDFPerturbation(
                A=cddf_cfg.A,
                alpha=cddf_cfg.alpha,
                N_pivot=cddf_cfg.N_pivot,
            )
            perturbed_p1d = compute_perturbed_p1d(
                hdf5_path=entry.path,
                nbins=header.nbins,
                dv_kms=dv_kms,
                catalog=catalog,
                perturbation=perturbation,
                base_mask_classes=["DLA", "subDLA", "LLS"],
                batch_size=cfg.skewer_batch_size,
                n_skewers=n_skewers_p1d,
                fill_strategy=cfg.absorber.mask_fill_strategy,
            )
            timing["perturbed_p1d"] = time.perf_counter() - t0

        # --- Save outputs -------------------------------------------------
        _save_snap_outputs(out_dir, catalog, p1d_variants, p1d_ratios, cddf_result,
                           perturbed_p1d, timing, header, sim_name, snap, z, dv_kms)
        mark_done(cfg, sim_name, snap)

        result = SnapResult(
            sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms,
            catalog=catalog, p1d_variants=p1d_variants, p1d_ratios=p1d_ratios,
            cddf_result=cddf_result, timing=timing, perturbed_p1d=perturbed_p1d,
        )
        logger.info("DONE: sim=%s snap=%03d z=%.2f  timing=%s",
                    sim_name[:30], snap, z,
                    {k: f"{v:.1f}s" for k, v in timing.items()})
        return result

    except Exception:
        tb = traceback.format_exc()
        logger.error("FAILED: sim=%s snap=%03d z=%.2f\n%s", sim_name[:30], snap, z, tb)
        write_error(cfg, sim_name, snap, tb)
        return None


def _save_snap_outputs(
    out_dir: Path,
    catalog: AbsorberCatalog,
    p1d_variants: Dict,
    p1d_ratios: Dict,
    cddf_result: Dict,
    perturbed_p1d: Optional[Dict],
    timing: Dict,
    header,
    sim_name: str,
    snap: int,
    z: float,
    dv_kms: float,
) -> None:
    """Save all outputs for one (sim, snap) to out_dir."""

    # P1D variants → npz
    p1d_save = {}
    for var, (k, p1d, mf) in p1d_variants.items():
        p1d_save[f"k_{var}"] = k
        p1d_save[f"p1d_{var}"] = p1d
        p1d_save[f"meanF_{var}"] = np.array([mf])
    np.savez(out_dir / "p1d.npz", **p1d_save)

    # Ratios → npz
    ratio_save = {k: v for k, v in p1d_ratios.items() if isinstance(v, np.ndarray)}
    if ratio_save:
        np.savez(out_dir / "p1d_ratios.npz", **ratio_save)

    # CDDF → npz
    cddf_save = {k: np.array(v) if not isinstance(v, np.ndarray) else v
                 for k, v in cddf_result.items()
                 if not isinstance(v, (str, dict))}
    np.savez(out_dir / "cddf.npz", **cddf_save)

    # Perturbed P1D → npz
    if perturbed_p1d is not None:
        pert_save = {k: v for k, v in perturbed_p1d.items()
                     if isinstance(v, np.ndarray)}
        np.savez(out_dir / "p1d_perturbed.npz", **pert_save)

    # Timing + metadata → JSON
    meta = {
        "sim_name": sim_name,
        "snap": snap,
        "z": z,
        "dv_kms": dv_kms,
        "nbins": int(header.nbins),
        "n_skewers": int(header.n_skewers),
        "box_kpc_h": float(header.box),
        "hubble": float(header.hubble),
        "n_absorbers": {cls: len(catalog.by_class(cls)) for cls in ["LLS", "subDLA", "DLA", "forest"]},
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _load_snap_result(out_dir: Path, sim_name: str, snap: int, z: float) -> Optional[SnapResult]:
    """Load a previously-saved SnapResult (lightweight, without re-reading HDF5)."""
    try:
        meta_path = out_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        catalog = AbsorberCatalog.load_npz(out_dir / "catalog.npz")

        p1d_data = np.load(out_dir / "p1d.npz")
        p1d_variants = {}
        # Reconstruct variant dict
        for key in p1d_data.files:
            if key.startswith("k_"):
                var = key[2:]
                k = p1d_data[f"k_{var}"]
                p1d = p1d_data[f"p1d_{var}"]
                mf = float(p1d_data[f"meanF_{var}"][0])
                p1d_variants[var] = (k, p1d, mf)

        ratio_path = out_dir / "p1d_ratios.npz"
        p1d_ratios = dict(np.load(ratio_path)) if ratio_path.exists() else {}

        cddf_path = out_dir / "cddf.npz"
        cddf_result = dict(np.load(cddf_path)) if cddf_path.exists() else {}

        return SnapResult(
            sim_name=sim_name, snap=snap, z=z,
            dv_kms=meta["dv_kms"],
            catalog=catalog,
            p1d_variants=p1d_variants,
            p1d_ratios=p1d_ratios,
            cddf_result=cddf_result,
            timing=meta.get("timing_s", {}),
        )
    except Exception as exc:
        logger.warning("Could not load cached result for snap %03d: %s", snap, exc)
        return None


# ---------------------------------------------------------------------------
# Level 2: one sim, all z
# ---------------------------------------------------------------------------

def run_sim_all_z(
    sim_snapshot: SimSnapshot,
    cfg: PipelineConfig,
) -> List[Optional[SnapResult]]:
    """Run all redshifts for one simulation. Sequential."""
    results = []
    for entry in sim_snapshot.entries:
        r = run_one_snap(sim_snapshot, entry, cfg)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Level 3: all sims, all z
# ---------------------------------------------------------------------------

def run_all(cfg: PipelineConfig) -> Dict[str, List[Optional[SnapResult]]]:
    """
    Run all (sim, snap) pairs.

    Parallelism: outer loop over sims uses cfg.n_workers processes via joblib.
    Each sim runs sequentially through its snapshots to avoid memory overload.
    """
    from joblib import Parallel, delayed

    snap_map = build_snapshot_map(
        data_root=cfg.data_root,
        z_min=cfg.z_min,
        z_max=cfg.z_max,
        sim_filter=cfg.sim_filter if cfg.sim_filter else None,
        prefer_grid=cfg.prefer_grid,
    )

    if cfg.debug:
        snap_map = snap_map[:cfg.debug_n_sims]
        for ss in snap_map:
            ss.entries[:] = ss.entries[:cfg.debug_n_snaps]

    logger.info("Running %d simulations, %d workers",
                len(snap_map), cfg.n_workers)

    # Save config for reproducibility
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(Path(cfg.output_root) / "config_used.yaml"))

    if cfg.n_workers > 1:
        all_results_flat = Parallel(n_jobs=cfg.n_workers, verbose=10)(
            delayed(run_sim_all_z)(ss, cfg)
            for ss in snap_map
        )
    else:
        all_results_flat = [run_sim_all_z(ss, cfg) for ss in snap_map]

    return {ss.sim.name: results
            for ss, results in zip(snap_map, all_results_flat)}


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(cfg: PipelineConfig, n_sims: int = 1, n_snaps: int = 1) -> Dict:
    """
    Run a timed benchmark on a small subset and extrapolate to full campaign.

    Returns dict with timing measurements and extrapolated totals.
    """
    snap_map = build_snapshot_map(
        data_root=cfg.data_root,
        z_min=cfg.z_min,
        z_max=cfg.z_max,
        prefer_grid=cfg.prefer_grid,
    )

    if not snap_map:
        return {"error": "No simulations found"}

    # Use first n_sims simulations, first n_snaps snapshots each
    test_cases = []
    for ss in snap_map[:n_sims]:
        for entry in ss.entries[:n_snaps]:
            test_cases.append((ss, entry))

    timing_records = []
    for ss, entry in test_cases:
        header = read_header(entry.path)
        dv_kms = pixel_dv_kms(header)
        n_total = header.n_skewers

        # Time catalog build on 10k skewers
        t0 = time.perf_counter()
        _ = build_catalog(
            hdf5_path=entry.path,
            sim_name=ss.sim.name,
            snap=entry.snap,
            z=entry.z,
            dv_kms=dv_kms,
            batch_size=cfg.skewer_batch_size,
            n_skewers=min(10000, n_total),
            fast_mode=True,
            min_log_nhi=LOG_NHI_LLS_MIN,
        )
        t_cat_10k = time.perf_counter() - t0

        # Time P1D on 10k skewers
        from .p1d import compute_p1d_single
        t0 = time.perf_counter()
        _ = compute_p1d_single(
            entry.path, header.nbins, dv_kms,
            n_skewers=min(10000, n_total),
            batch_size=cfg.skewer_batch_size,
        )
        t_p1d_10k = time.perf_counter() - t0

        scale = n_total / 10000.0
        timing_records.append({
            "sim": ss.sim.name[:40],
            "snap": entry.snap,
            "z": entry.z,
            "n_skewers": n_total,
            "t_catalog_10k_s": round(t_cat_10k, 2),
            "t_p1d_10k_s": round(t_p1d_10k, 2),
            "t_catalog_full_est_s": round(t_cat_10k * scale, 1),
            "t_p1d_full_est_s": round(t_p1d_10k * scale, 1),
            "t_snap_est_s": round((t_cat_10k + t_p1d_10k) * scale, 1),
        })

    # Extrapolate to full campaign
    if timing_records:
        avg_snap_s = np.mean([r["t_snap_est_s"] for r in timing_records])
        n_total_snaps = sum(len(ss.entries) for ss in snap_map)
        n_total_sims = len(snap_map)
        campaign_serial_hr = avg_snap_s * n_total_snaps / 3600.0
        campaign_parallel_hr = campaign_serial_hr / max(cfg.n_workers, 1)
    else:
        avg_snap_s = campaign_serial_hr = campaign_parallel_hr = 0.0
        n_total_snaps = n_total_sims = 0

    return {
        "timing_per_snap": timing_records,
        "n_total_sims": n_total_sims,
        "n_total_snaps": n_total_snaps,
        "avg_time_per_snap_s": round(avg_snap_s, 1),
        "campaign_serial_hr": round(campaign_serial_hr, 1),
        "campaign_parallel_hr": round(campaign_parallel_hr, 1),
        "n_workers_assumed": cfg.n_workers,
    }
