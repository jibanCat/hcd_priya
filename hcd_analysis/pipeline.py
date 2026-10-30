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

Catalogue-only mode (cfg.catalogue_only, a product setting)
-------------------------------------------------------------
Each (sim, snap) writes catalog.npz (with its lineage record), cddf.npz (with a record naming the catalogue's sha256)
and meta.json, then the marker done_catalogue; no P1D product is computed. The setting is recorded in the output
root's run_settings.json, so one root holds either catalogue-only or full runs.

Resume / idempotency
--------------------
Each (sim, snap) writes a sentinel file <output_dir>/<sim>/<snap>/done (done_catalogue in catalogue-only mode), after
meta.json, which lists the sha256 and size of every product written for the snapshot ("products") and a lineage
record of the snapshot. If the sentinel exists and cfg.resume=True, the pair is not recomputed: its catalogue must pass
load_validated_catalog (lineage record from an allowed finder, the run's finder settings, the same raw spectra file,
then the value checks) and every listed product must match its sha256; otherwise the run stops. A snapshot directory
holding the other mode's sentinel is refused (RunSettingsMismatch), touching nothing.

Failure handling
----------------
Failures are caught per (sim, snap), logged, and written to
<output_dir>/<sim>/<snap>/error.txt. The pipeline continues.
For a snapshot marked done: if load_validated_catalog refuses its cached
catalogue (built before the box-edge fix, or holding a non-finite N_HI),
CatalogIntegrityError is raised and the run stops, with nothing written for that
snapshot and no stack for its simulation (snapshots finished earlier keep their
outputs); if its cached outputs cannot be read, it is skipped with a WARNING (no
error.txt) and left out of the simulation's cddf_stacked.npz.
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

from . import lineage as L
from .catalog import (AbsorberCatalog, CatalogIntegrityError, CatalogProvenanceError, RELEASE_FINDER_IDS,
                      build_catalog, finder_settings, load_validated_catalog, reconcile_counts, LOG_NHI_LLS_MIN)
from .cddf import (
    CDDFPerturbation,
    compute_perturbed_p1d,
    measure_cddf,
    measure_cddf_from_dataframe,
    stack_cddf_for_sim,
)
from .config import PipelineConfig, save_config
from .run_settings import RUN_SETTINGS_FILE, RunSettingsMismatch, read_record, record_or_verify
from .io import pixel_dv_kms, read_header, read_sim_ics
from .p1d import (
    ALL_VARIANTS,
    compute_all_p1d_variants,
    compute_p1d_excl_nhi,
    compute_p1d_ratios,
    _DEFAULT_K_BINS,
)
from .snapshot_map import SimSnapshot, SnapEntry, build_snapshot_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def _repo_dirty() -> Optional[bool]:
    """Whether the hcd_priya checkout has uncommitted changes (None when git cannot tell). Asked at each snapshot,
    with _repo_commit, so the two describe one moment; without git's optional index lock, so a run never writes into
    the checkout's git metadata."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return bool(out.strip())
    except Exception:
        return None


def _repo_commit() -> str:
    """Return short git commit hash of the hcd_priya repo, or 'unknown'."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Run settings (hcd_analysis/run_settings.py)
# ---------------------------------------------------------------------------

# The settings that may determine a run's products, compared on every run into one output root. The list is
# deliberately conservative: it holds whole sections, so some keys that no product reads today are compared too, and
# data_root is compared as written. Selection (z range, sim filter, debug sizes), parallelism, batching, resume and the
# output location are not product settings and may change between runs; the per-snapshot products do not depend on
# them, while <sim>/cddf_stacked.npz is rebuilt from the snapshots each invocation selects and lists them
# (z_snapshots). A change of code alone is not refused; each snapshot's meta.json records the code commit.
PRODUCT_SETTING_KEYS = ("data_root", "prefer_grid", "absorber_classes", "absorber", "p1d", "cddf", "benchmark", "debug",
                        "catalogue_only")
HIRES_SUBDIR = "hires"     # run_hires writes under <output_root>/hires, an output root with its own record
DONE_FULL, DONE_CATALOGUE = "done", "done_catalogue"     # the completion markers of the two modes
CDDF_KIND_VERSION = 1
SNAPSHOT_KIND_VERSION = 1
# the files a snapshot directory may list in meta.json["products"], with their lineage kinds
PRODUCT_KINDS = {"catalog.npz": "catalog", "cddf.npz": "cddf", "p1d.npz": "p1d", "p1d_ratios.npz": "p1d_ratios",
                 "p1d_excl.npz": "p1d_excl", "p1d_perturbed.npz": "p1d_perturbed", "p1d_per_class.h5": "p1d_per_class"}


class SnapshotIntegrityError(CatalogIntegrityError):
    """A completed snapshot's products do not match the sha256 its meta.json records, or its records disagree."""


def product_settings(cfg: PipelineConfig) -> dict:
    """The product-determining part of ``cfg``, as recorded in ``<output_root>/run_settings.json``."""
    d = dataclasses.asdict(cfg)
    return {k: d[k] for k in PRODUCT_SETTING_KEYS}


def check_run_settings(cfg: PipelineConfig) -> Path:
    """Record the run's product settings at its output root, or verify them against the record there.

    Raises RunSettingsMismatch, before anything is read or written, when the output root holds results made with
    different settings (for example another ``absorber.tau_threshold``), or holds files but no record (results of
    unknown settings, such as an output root written before records existed). The hires subdirectory, an output root
    with its own record, does not count as the low-resolution root's contents."""
    return record_or_verify(Path(cfg.output_root), product_settings(cfg), info=lambda: {"code_commit": _repo_commit()},
                            ignore=(HIRES_SUBDIR,))


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def snap_output_dir(cfg: PipelineConfig, sim_name: str, snap: int) -> Path:
    d = Path(cfg.output_root) / sim_name / f"snap_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_marker(cfg: PipelineConfig) -> str:
    return DONE_CATALOGUE if cfg.catalogue_only else DONE_FULL


def is_done(cfg: PipelineConfig, sim_name: str, snap: int) -> bool:
    sentinel = snap_output_dir(cfg, sim_name, snap) / done_marker(cfg)
    return sentinel.exists()


def mark_done(cfg: PipelineConfig, sim_name: str, snap: int) -> None:
    (snap_output_dir(cfg, sim_name, snap) / done_marker(cfg)).touch()


def _refuse_other_mode(out_dir: Path, cfg: PipelineConfig) -> None:
    """A snapshot directory completed in the other mode (copied in; the root's record keeps the modes apart) is
    neither resumed nor overwritten."""
    other = DONE_FULL if cfg.catalogue_only else DONE_CATALOGUE
    if (out_dir / other).exists():
        mode = "catalogue-only" if cfg.catalogue_only else "full"
        raise RunSettingsMismatch(f"{out_dir} holds the marker {other} of the other mode; this {mode} run neither "
                                  "resumes nor overwrites it. Nothing was read or written.")


def write_error(cfg: PipelineConfig, sim_name: str, snap: int, msg: str) -> None:
    path = snap_output_dir(cfg, sim_name, snap) / "error.txt"
    with open(path, "w") as f:
        f.write(msg)


# ---------------------------------------------------------------------------
# Catalogue and lineage helpers
# ---------------------------------------------------------------------------

def _catalog_kwargs(cfg: PipelineConfig) -> dict:
    """The catalogue-determining arguments of build_catalog for this run (the keyword names of finder_settings)."""
    a = cfg.absorber
    return dict(tau_threshold=a.tau_threshold, merge_dv_kms=a.merge_dv_kms, min_pixels=a.min_pixels,
                b_init=a.b_init_kms, b_bounds=tuple(a.b_bounds), tau_fit_cap=a.tau_fit_cap,
                voigt_max_iter=a.voigt_max_iter, n_skewers=cfg.p1d.n_skewers if cfg.debug else None,
                fast_mode=a.fast_mode or cfg.benchmark, min_log_nhi=a.min_log_nhi)


def _n_skewers_built(kw: dict, n_skewers_file: int) -> int:
    return int(n_skewers_file) if kw["n_skewers"] is None else min(int(kw["n_skewers"]), int(n_skewers_file))


def _catalog_for_snap(cfg, entry, out_dir, header, dv_kms, sim_name, snap, z, run_settings_sha256) -> AbsorberCatalog:
    """The snapshot's catalogue: a cached catalog.npz on resume, refused unless load_validated_catalog accepts it
    with this run's finder settings, raw spectra file and skewer count; else built (stamped) and saved."""
    catalog_path = out_dir / "catalog.npz"
    kw = _catalog_kwargs(cfg)
    if cfg.resume and catalog_path.exists():
        catalog = load_validated_catalog(catalog_path, int(header.nbins), settings=finder_settings(**kw),
                                         raw_path=entry.path, n_skewers_built=_n_skewers_built(kw, header.n_skewers))
        logger.info("  loaded catalog from cache (%d absorbers)", len(catalog.absorbers))
        return catalog
    catalog = build_catalog(hdf5_path=entry.path, sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms,
                            batch_size=cfg.skewer_batch_size, n_workers=cfg.n_workers_skewer,
                            run_settings_sha256=run_settings_sha256, **kw)
    catalog.save_npz(catalog_path)
    return catalog


def _catalog_input(out_dir: Path, catalog: AbsorberCatalog) -> dict:
    """The lineage input entry of the snapshot's catalog.npz, refused unless the file on disk carries the record of
    the catalogue in memory."""
    path = out_dir / "catalog.npz"
    with np.load(path, allow_pickle=False) as d:
        on_disk = L.parse_record(str(d[L.NPZ_KEY][()])) if L.NPZ_KEY in d.files else None
    if catalog.lineage is None or on_disk != catalog.lineage:
        raise CatalogProvenanceError(f"{path}: the catalogue on disk is not the one this snapshot used")
    ident = L.file_identity(path)
    return L.input_entry("catalog", "catalog", {"sha256": ident["sha256"], "size": ident["size"]},
                         record=catalog.lineage)


def _write_cddf(out_dir: Path, cddf_result: Dict, catalog: AbsorberCatalog, catalog_entry: dict,
                run_settings_sha256: Optional[str]) -> dict:
    """cddf.npz with its lineage record: the catalogue (sha256 of catalog.npz) and the raw spectra file whose header
    gives the absorption path. Returns the record."""
    arrays = {k: np.asarray(v) for k, v in cddf_result.items() if not isinstance(v, (str, dict))}
    raw = L.find_input(catalog.lineage, "raw_spectra")
    rec = L.build_record(
        "cddf", CDDF_KIND_VERSION,
        inputs=[catalog_entry, L.input_entry("raw_spectra_header", "raw_spectra", raw)],
        finder_ids=None, settings={"log_nhi_edges": [float(x) for x in np.asarray(cddf_result["log_nhi_edges"])]},
        code=L.code_identity(), allowed_finder_ids=RELEASE_FINDER_IDS, run_settings_sha256=run_settings_sha256,
        product={"sim_name": catalog.sim_name, "snap": int(catalog.snap), "z": float(catalog.z),
                 "n_absorbers": int(np.sum(arrays["n_absorbers"])), "payload_sha256": L.arrays_sha256(arrays)})
    L.write_npz(out_dir / "cddf.npz", arrays, rec)
    return rec


def _write_meta(out_dir: Path, meta: dict, products: List[str], records: Dict[str, dict], mode: str,
                run_settings_sha256: Optional[str]) -> None:
    """meta.json, written last before the completion marker (atomically): ``mode``, ``products`` (sha256 and size of
    every product written for the snapshot) and ``lineage``, the snapshot's record listing those products as its
    inputs (with the records of the lineage-bearing ones, ``records``)."""
    prods, entries = {}, []
    for name in products:
        ident = L.file_identity(out_dir / name)
        prods[name] = {"sha256": ident["sha256"], "size": ident["size"]}
        entries.append(L.input_entry(name, PRODUCT_KINDS[name], prods[name], record=records.get(name)))
    meta = dict(meta, mode=mode, products=prods)
    meta["lineage"] = L.build_record(
        "snapshot", SNAPSHOT_KIND_VERSION, inputs=entries, finder_ids=None, settings={"mode": mode},
        code=L.code_identity(), allowed_finder_ids=RELEASE_FINDER_IDS, run_settings_sha256=run_settings_sha256,
        product={"sim_name": meta["sim_name"], "snap": int(meta["snap"]), "z": float(meta["z"])})
    tmp = out_dir / f".meta.json.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, out_dir / "meta.json")
    finally:
        if tmp.exists():
            tmp.unlink()


def _meta_base(header, sim_name, snap, z, dv_kms, n_absorbers, timing, ics_meta) -> dict:
    meta = {
        "sim_name": sim_name,
        "snap": snap,
        "z": z,
        "dv_kms": dv_kms,
        "nbins": int(header.nbins),
        "n_skewers": int(header.n_skewers),
        "box_kpc_h": float(header.box),
        "hubble": float(header.hubble),
        "n_absorbers": n_absorbers,
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
        "code_commit": _repo_commit(),
        "code_dirty": _repo_dirty(),
    }
    if ics_meta:
        meta["sim_ics"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in ics_meta.items()}
    return meta


def _verify_snapshot(out_dir: Path, meta: dict, cfg: PipelineConfig) -> Dict:
    """Refuse (SnapshotIntegrityError) a completed snapshot unless meta.json is of this run's mode, lists the sha256
    of every product the mode writes, every listed product matches it, the snapshot record agrees with the list, and
    cddf.npz's record names the listed catalogue. Returns the CDDF arrays."""
    mode = "catalogue" if cfg.catalogue_only else "full"

    def refuse(why):
        raise SnapshotIntegrityError(f"{out_dir}: completed snapshot refused ({why}); rebuild it")
    if meta.get("mode") != mode:
        refuse(f"meta.json mode {meta.get('mode')!r}, this run {mode!r}")
    prods = meta.get("products")
    if not isinstance(prods, dict) or not prods:
        refuse("meta.json lists no product hashes (written before snapshots recorded them)")
    need = {"catalog.npz", "cddf.npz"} | (set() if cfg.catalogue_only else {"p1d.npz"})
    if need - set(prods) or set(prods) - set(PRODUCT_KINDS):
        refuse(f"meta.json products {sorted(prods)}, expected {sorted(need)} and only known names")
    for name, ident in sorted(prods.items()):
        p = out_dir / name
        if not (isinstance(ident, dict) and isinstance(ident.get("sha256"), str) and isinstance(ident.get("size"), int)):
            refuse(f"malformed product entry for {name}")
        if not p.is_file() or not L.same_file(p, {"sha256": ident["sha256"], "size": ident["size"]}):
            refuse(f"{name} does not match the sha256 meta.json records")
    try:
        rec = L.check_record(meta.get("lineage"), RELEASE_FINDER_IDS)
        if rec["kind"] != "snapshot" or {i["role"]: i["sha256"] for i in rec["inputs"]} != \
                {k: v["sha256"] for k, v in prods.items()}:
            refuse("the snapshot record does not list the products meta.json lists")
        cddf, crec = L.read_npz(out_dir / "cddf.npz")
        L.check_record(crec, RELEASE_FINDER_IDS)
        if crec["kind"] != "cddf" or L.find_input(crec, "catalog")["sha256"] != prods["catalog.npz"]["sha256"]:
            refuse("cddf.npz was not made from the listed catalog.npz")
    except L.LineageError as e:
        refuse(str(e))
    return cddf


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

    In catalogue-only mode (cfg.catalogue_only) only the catalogue, its CDDF and
    meta.json are written, and the marker is done_catalogue.

    Returns SnapResult on success, None on failure (logged; written to error.txt
    unless the snapshot is marked done). Raises CatalogIntegrityError, writing
    nothing, when the snapshot is marked done and its cached catalogue or products
    are refused, and RunSettingsMismatch, touching nothing, when the output root's
    recorded run settings differ from cfg's (check_run_settings) or the snapshot
    directory holds the other mode's marker.
    """
    sim_name = sim_snapshot.sim.name
    snap = entry.snap
    z = entry.z
    rs_path = check_run_settings(cfg)          # refuses a rerun with other settings before anything is touched
    out_dir = snap_output_dir(cfg, sim_name, snap)
    _refuse_other_mode(out_dir, cfg)

    if cfg.resume and is_done(cfg, sim_name, snap):
        logger.info("SKIP (done): sim=%s snap=%03d z=%.2f", sim_name[:30], snap, z)
        return _load_snap_result(out_dir, sim_name, snap, z, cfg, entry.path)

    logger.info("START: sim=%s snap=%03d z=%.2f", sim_name[:30], snap, z)
    timing: Dict[str, float] = {}

    try:
        # --- Load header + SimulationICs ----------------------------------
        t0 = time.perf_counter()
        header = read_header(entry.path)
        dv_kms = pixel_dv_kms(header)
        ics_meta = read_sim_ics(sim_snapshot.sim)  # empty dict if file absent
        rs_sha = L.sha256_file(rs_path)
        timing["header"] = time.perf_counter() - t0

        # --- Build absorber catalog (or reuse a validated cached one) -----
        t0 = time.perf_counter()
        catalog = _catalog_for_snap(cfg, entry, out_dir, header, dv_kms, sim_name, snap, z, rs_sha)
        timing["catalog"] = time.perf_counter() - t0
        logger.info("  catalog: %s", catalog.summary())

        if cfg.catalogue_only:
            t0 = time.perf_counter()
            cddf_result = measure_cddf(catalog, header)
            timing["cddf"] = time.perf_counter() - t0
            _save_catalogue_outputs(out_dir, catalog, cddf_result, ics_meta, timing, header, sim_name, snap, z, dv_kms,
                                    run_settings_sha256=rs_sha)
            mark_done(cfg, sim_name, snap)
            logger.info("DONE (catalogue): sim=%s snap=%03d z=%.2f  timing=%s", sim_name[:30], snap, z,
                        {k: f"{v:.1f}s" for k, v in timing.items()})
            return SnapResult(sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms, catalog=catalog, p1d_variants={},
                              p1d_ratios={}, cddf_result=cddf_result, timing=timing)

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

        # --- NHI exclusion sweep (P1D vs sightline cut) -------------------
        t0 = time.perf_counter()
        nhi_cuts = cfg.p1d.nhi_excl_thresholds
        excl_result = None
        if nhi_cuts and catalog is not None:
            excl_result = compute_p1d_excl_nhi(
                hdf5_path=entry.path,
                nbins=header.nbins,
                dv_kms=dv_kms,
                catalog=catalog,   # AbsorberCatalog accepted directly; no pandas needed
                log_nhi_cuts=nhi_cuts,
                batch_size=cfg.skewer_batch_size,
                n_skewers=n_skewers_p1d,
            )
        timing["excl_sweep"] = time.perf_counter() - t0

        # --- Per-class subset P1D (Rogers template input) -----------------
        t0 = time.perf_counter()
        from .p1d import compute_p1d_per_class, save_p1d_per_class_hdf5
        per_class = compute_p1d_per_class(
            hdf5_path=entry.path,
            nbins=header.nbins,
            dv_kms=dv_kms,
            catalog=catalog,
            batch_size=cfg.skewer_batch_size,
            n_skewers=n_skewers_p1d,
        )
        save_p1d_per_class_hdf5(
            out_dir / "p1d_per_class.h5",
            per_class,
            sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms,
            extra_attrs={
                "nbins": int(header.nbins),
                "n_skewers_file": int(header.n_skewers),
                "tau_threshold": float(cfg.absorber.tau_threshold),
                "min_log_nhi": float(cfg.absorber.min_log_nhi),
                "fast_mode": bool(cfg.absorber.fast_mode),
                "source_commit": _repo_commit(),
            },
        )
        timing["per_class_p1d"] = time.perf_counter() - t0

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
                           perturbed_p1d, excl_result, ics_meta,
                           timing, header, sim_name, snap, z, dv_kms,
                           run_settings_sha256=rs_sha, written_before=("p1d_per_class.h5",))
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
    excl_result: Optional[Dict],
    ics_meta: Dict,
    timing: Dict,
    header,
    sim_name: str,
    snap: int,
    z: float,
    dv_kms: float,
    *,
    run_settings_sha256: Optional[str] = None,
    written_before=(),
) -> None:
    """Save all outputs for one (sim, snap) to out_dir; meta.json last, listing the sha256 of every product written
    for the snapshot (``written_before``: the ones the caller wrote earlier, besides catalog.npz)."""
    # reconcile the absorber counts against the CDDF first: none of the files below is written if that fails
    # (catalog.npz was written earlier, only after the finder, overlap and value checks of build_catalog)
    n_absorbers = reconcile_counts(catalog, cddf_result["n_absorbers"], cddf_result["log_nhi_edges"])
    catalog_entry = _catalog_input(out_dir, catalog)
    written = ["catalog.npz", *written_before, "p1d.npz", "cddf.npz"]

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
        written.append("p1d_ratios.npz")

    # CDDF → npz, with its lineage record
    cddf_record = _write_cddf(out_dir, cddf_result, catalog, catalog_entry, run_settings_sha256)

    # NHI exclusion sweep → npz
    if excl_result is not None:
        excl_save = {}
        for key, val in excl_result.items():
            if isinstance(val, np.ndarray):
                excl_save[key] = val
            elif isinstance(val, (list, tuple)):
                excl_save[key] = np.array(val)
        np.savez(out_dir / "p1d_excl.npz", **excl_save)
        written.append("p1d_excl.npz")

    # Perturbed P1D → npz
    if perturbed_p1d is not None:
        pert_save = {k: v for k, v in perturbed_p1d.items()
                     if isinstance(v, np.ndarray)}
        np.savez(out_dir / "p1d_perturbed.npz", **pert_save)
        written.append("p1d_perturbed.npz")

    # Timing + metadata → JSON (includes SimulationICs fields), with the products' sha256 and the snapshot record
    meta = _meta_base(header, sim_name, snap, z, dv_kms, n_absorbers, timing, ics_meta)
    _write_meta(out_dir, meta, written, {"catalog.npz": catalog.lineage, "cddf.npz": cddf_record}, "full",
                run_settings_sha256)


def _save_catalogue_outputs(out_dir: Path, catalog: AbsorberCatalog, cddf_result: Dict, ics_meta: Dict, timing: Dict,
                            header, sim_name: str, snap: int, z: float, dv_kms: float, *,
                            run_settings_sha256: Optional[str] = None) -> None:
    """Catalogue-only mode: cddf.npz (with its lineage) and meta.json (counts, the sha256 of catalog.npz and
    cddf.npz, the code commit and dirty flag, the snapshot record), after the same count reconciliation."""
    n_absorbers = reconcile_counts(catalog, cddf_result["n_absorbers"], cddf_result["log_nhi_edges"])
    catalog_entry = _catalog_input(out_dir, catalog)
    cddf_record = _write_cddf(out_dir, cddf_result, catalog, catalog_entry, run_settings_sha256)
    meta = _meta_base(header, sim_name, snap, z, dv_kms, n_absorbers, timing, ics_meta)
    _write_meta(out_dir, meta, ["catalog.npz", "cddf.npz"], {"catalog.npz": catalog.lineage, "cddf.npz": cddf_record},
                "catalogue", run_settings_sha256)


def _load_snap_result(out_dir: Path, sim_name: str, snap: int, z: float, cfg: PipelineConfig,
                      raw_path) -> Optional[SnapResult]:
    """Load a previously-saved SnapResult (lightweight: the raw spectra file is only stat-ed). The catalogue must pass
    load_validated_catalog with this run's settings and raw file, and every product must match the sha256 meta.json
    records (_verify_snapshot); a refusal raises CatalogIntegrityError."""
    try:
        meta_path = out_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        kw = _catalog_kwargs(cfg)
        catalog = load_validated_catalog(out_dir / "catalog.npz", int(meta["nbins"]), settings=finder_settings(**kw),
                                         raw_path=raw_path,
                                         n_skewers_built=_n_skewers_built(kw, int(meta["n_skewers"])))
        cddf_result = _verify_snapshot(out_dir, meta, cfg)

        p1d_variants, p1d_ratios = {}, {}
        if not cfg.catalogue_only:
            p1d_data = np.load(out_dir / "p1d.npz")
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

        return SnapResult(
            sim_name=sim_name, snap=snap, z=z,
            dv_kms=meta["dv_kms"],
            catalog=catalog,
            p1d_variants=p1d_variants,
            p1d_ratios=p1d_ratios,
            cddf_result=cddf_result,
            timing=meta.get("timing_s", {}),
        )
    except CatalogIntegrityError:
        raise                                 # a refused catalogue is never reused, nor silently dropped from a stack
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
    """Run all redshifts for one simulation, then stack per-z-bin CDDFs."""
    results = []
    for entry in sim_snapshot.entries:
        r = run_one_snap(sim_snapshot, entry, cfg)
        results.append(r)

    # Stack CDDF across snapshots into per-z-bin CDDFs
    _stack_and_save_cddf(sim_snapshot, cfg, results)
    return results


def _stack_and_save_cddf(
    sim_snapshot: SimSnapshot,
    cfg: PipelineConfig,
    results: List[Optional[SnapResult]],
) -> None:
    """
    Collect per-snap CDDFs and stack them into per-redshift-bin CDDFs.
    Saves to <output_root>/<sim>/cddf_stacked.npz.
    """
    cddf_list = []
    for r in results:
        if r is not None and r.cddf_result:
            cddf_list.append(r.cddf_result)

    if not cddf_list:
        return

    try:
        stacked = stack_cddf_for_sim(cddf_list)
        sim_dir = Path(cfg.output_root) / sim_snapshot.sim.name
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Save each z-bin as a group in one npz
        save_dict = {}
        bin_labels = []
        for label, cdata in stacked.items():
            safe_label = label.replace("-", "to").replace(".", "p")
            for field in ("f_nhi", "n_absorbers", "log_nhi_centres", "log_nhi_edges",
                          "f_nhi_per_snap"):
                if field in cdata:
                    save_dict[f"{safe_label}__{field}"] = np.asarray(cdata[field])
            save_dict[f"{safe_label}__z_snapshots"] = np.array(cdata["z_snapshots"])
            save_dict[f"{safe_label}__z_min"] = np.array([cdata["z_min"]])
            save_dict[f"{safe_label}__z_max"] = np.array([cdata["z_max"]])
            save_dict[f"{safe_label}__total_path"] = np.array([cdata["total_path"]])
            bin_labels.append(label)

        np.savez(sim_dir / "cddf_stacked.npz", **save_dict)
        logger.info("  CDDF stacked for sim=%s: %d z-bins", sim_snapshot.sim.name[:30], len(bin_labels))
    except Exception as exc:
        logger.warning("CDDF stacking failed for sim=%s: %s", sim_snapshot.sim.name[:30], exc)


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

    check_run_settings(cfg)                    # before anything is read or written (config_used.yaml included)
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
# HiRes mode: process emu_full_hires and compute convergence ratios
# ---------------------------------------------------------------------------

def run_hires(cfg: PipelineConfig) -> Dict[str, List[Optional[SnapResult]]]:
    """
    Run the full pipeline on HiRes simulations (cfg.hires_data_root).
    Outputs go to <output_root>/hires/<sim>/...
    """
    from joblib import Parallel, delayed

    hires_cfg = dataclasses.replace(cfg, data_root=cfg.hires_data_root)
    hires_cfg = dataclasses.replace(hires_cfg, output_root=str(Path(cfg.output_root) / HIRES_SUBDIR))
    check_run_settings(hires_cfg)              # before anything is read or written (config_hires.yaml included)

    snap_map = build_snapshot_map(
        data_root=hires_cfg.data_root,
        z_min=hires_cfg.z_min,
        z_max=hires_cfg.z_max,
        sim_filter=hires_cfg.sim_filter if hires_cfg.sim_filter else None,
        prefer_grid=hires_cfg.prefer_grid,
    )

    if not snap_map:
        logger.warning("No HiRes simulations found in %s", hires_cfg.data_root)
        return {}

    logger.info("HiRes: running %d simulations from %s", len(snap_map), hires_cfg.data_root)

    Path(hires_cfg.output_root).mkdir(parents=True, exist_ok=True)
    save_config(hires_cfg, str(Path(hires_cfg.output_root) / "config_hires.yaml"))

    if hires_cfg.n_workers > 1:
        all_results_flat = Parallel(n_jobs=hires_cfg.n_workers, verbose=10)(
            delayed(run_sim_all_z)(ss, hires_cfg) for ss in snap_map
        )
    else:
        all_results_flat = [run_sim_all_z(ss, hires_cfg) for ss in snap_map]

    return {ss.sim.name: res for ss, res in zip(snap_map, all_results_flat)}


def _load_snap_z_map(sim_dir: Path) -> Dict[str, Tuple[Path, float]]:
    """
    Return {snap_name: (path, z)} for every completed snap in sim_dir.
    A snap is considered completed iff its `done` sentinel and a parseable
    meta.json with a numeric 'z' field both exist.
    """
    out: Dict[str, Tuple[Path, float]] = {}
    if not sim_dir.is_dir():
        return out
    for p in sorted(sim_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("snap_"):
            continue
        if not (p / "done").exists():
            continue
        meta_path = p / "meta.json"
        if not meta_path.exists():
            continue
        try:
            z = float(json.load(open(meta_path))["z"])
        except Exception:
            continue
        out[p.name] = (p, z)
    return out


def _match_snaps_by_z(
    lf_snaps: Dict[str, Tuple[Path, float]],
    hf_snaps: Dict[str, Tuple[Path, float]],
    z_tol: float,
) -> List[Tuple[str, Path, float, str, Path, float]]:
    """
    For every HR snap, pick the LF snap with the smallest |z_LF - z_HR|,
    provided the gap is ≤ z_tol.  Ties (equal |Δz|) are broken by the
    lower LF snap number so the match is deterministic.

    Returns a list of (hf_name, hf_path, hf_z, lf_name, lf_path, lf_z)
    tuples, ordered by hf_z.
    """
    matches = []
    # Iterate HR snaps in snap-name order for stability; sort output by z below.
    for hf_name, (hf_p, hf_z) in sorted(hf_snaps.items()):
        best = None  # (|Δz|, lf_name, lf_p, lf_z)
        for lf_name, (lf_p, lf_z) in sorted(lf_snaps.items()):
            dz = abs(lf_z - hf_z)
            if dz > z_tol:
                continue
            # Strictly-less keeps the first (lowest-snap) winner on ties.
            if best is None or dz < best[0]:
                best = (dz, lf_name, lf_p, lf_z)
        if best is None:
            continue
        _, lf_name, lf_p, lf_z = best
        matches.append((hf_name, hf_p, hf_z, lf_name, lf_p, lf_z))
    matches.sort(key=lambda m: m[2])
    return matches


def check_convergence_settings(lf_root: Path, hf_root: Path) -> None:
    """Refuse to combine the two resolutions' products when their recorded run settings differ in anything but
    data_root, or when only one of them has a record (RunSettingsMismatch). Two roots without records (written before
    records existed) cannot be compared: a warning is logged."""
    recs = {}
    for name, root in (("low-resolution", Path(lf_root)), ("hires", Path(hf_root))):
        p = root / RUN_SETTINGS_FILE
        recs[name] = read_record(p)["settings"] if os.path.lexists(p) else None
    missing = [n for n, r in recs.items() if r is None]
    if len(missing) == 1:
        raise RunSettingsMismatch(f"convergence: the {missing[0]} root has no {RUN_SETTINGS_FILE}, so its settings are "
                                  f"unknown and cannot be combined with the recorded {lf_root} / {hf_root}")
    if missing:
        logger.warning("convergence: neither %s nor %s has a run-settings record; their settings are not compared",
                       lf_root, hf_root)
        return
    lf, hf = ({k: v for k, v in r.items() if k != "data_root"} for r in recs.values())
    from .run_settings import _differences, _lookup
    diff = _differences(lf, hf)
    if diff:
        detail = "; ".join(f"{k}: low-resolution {_lookup(lf, k)!r}, hires {_lookup(hf, k)!r}" for k in diff[:12])
        raise RunSettingsMismatch(f"convergence: {lf_root} and {hf_root} were made with different settings: {detail}")


def compute_convergence_ratios(
    cfg: PipelineConfig,
    variants: Optional[List[str]] = None,
    z_tol: float = 0.05,
) -> Dict:
    """
    Compute HiRes/LowRes P1D convergence ratios T(k) = P1D_hires / P1D_lowres
    for matching simulation parameter points.

    Sim matching is by sim folder name.  Snapshot matching is by redshift
    (closest |z_LF - z_HR| ≤ z_tol), NOT by snap folder name — the two
    suites' Snapshots.txt tables are offset by one snap, so same-snap
    matching mixes resolution with z-evolution.  HR snaps with no LF
    match inside z_tol are skipped.

    Each resulting (z, variant) entry uses the HR z as its label, since
    the HR set is the anchor (fewer snaps, each tested against its
    closest LF epoch).

    Returns dict: sim_name → z_HR → variant → {'k', 'T_k'}
    """
    if variants is None:
        variants = ["all", "no_DLA_priya"]

    lf_root = Path(cfg.output_root)
    hf_root = Path(cfg.output_root) / HIRES_SUBDIR
    check_convergence_settings(lf_root, hf_root)

    # Find common sims
    lf_sims = {p.name for p in lf_root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    hf_sims = {p.name for p in hf_root.iterdir() if p.is_dir() and not p.name.startswith(".")
               and p.name != "hires"}
    common_sims = lf_sims & hf_sims

    logger.info("Convergence ratios: %d matching sims (z_tol=%.3f)",
                len(common_sims), z_tol)

    ratios = {}
    for sim_name in sorted(common_sims):
        lf_sim_dir = lf_root / sim_name
        hf_sim_dir = hf_root / sim_name

        lf_snaps = _load_snap_z_map(lf_sim_dir)
        hf_snaps = _load_snap_z_map(hf_sim_dir)

        sim_ratios = {}
        for hf_name, hf_p, hf_z, lf_name, lf_p, lf_z in _match_snaps_by_z(
            lf_snaps, hf_snaps, z_tol=z_tol
        ):
            try:
                lf_data = dict(np.load(lf_p / "p1d.npz"))
                hf_data = dict(np.load(hf_p / "p1d.npz"))

                snap_ratios = {}
                for var in variants:
                    k_key = f"k_{var}"
                    p_key = f"p1d_{var}"
                    if k_key in lf_data and k_key in hf_data:
                        k = lf_data[k_key]
                        p_lf = lf_data[p_key]
                        p_hf = hf_data[p_key]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            T_k = np.where(p_lf > 0, p_hf / p_lf, np.nan)
                        snap_ratios[var] = {"k": k, "T_k": T_k}
                sim_ratios[round(hf_z, 3)] = snap_ratios
                logger.debug(
                    "matched %s/%s (z=%.3f) ↔ LF %s (z=%.3f), |Δz|=%.3f",
                    sim_name, hf_name, hf_z, lf_name, lf_z, abs(hf_z - lf_z),
                )
            except Exception as exc:
                logger.warning(
                    "Convergence ratio failed for %s/%s↔%s: %s",
                    sim_name, hf_name, lf_name, exc,
                )

        if sim_ratios:
            ratios[sim_name] = sim_ratios
            # Save convergence ratios per sim
            out_path = hf_root / sim_name / "convergence_ratios.npz"
            save_dict = {}
            for z_val, z_data in sim_ratios.items():
                z_label = f"z{z_val:.3f}".replace(".", "p")
                for var, vdata in z_data.items():
                    save_dict[f"{z_label}__{var}__k"] = vdata["k"]
                    save_dict[f"{z_label}__{var}__T_k"] = vdata["T_k"]
            np.savez(out_path, **save_dict)

    return ratios


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
            record_lineage=False,             # a timing build, never saved: the whole-file hash would distort it
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
