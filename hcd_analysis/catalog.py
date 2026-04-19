"""
Absorber catalog: identify and classify absorption systems from tau arrays.

Architecture
------------
1. find_systems_in_skewer()
   Locate connected velocity regions where tau > tau_threshold.
   Merge regions separated by < merge_dv_kms. Each region is a "system candidate."

2. measure_nhi()
   For each candidate region: estimate NHI via Voigt fitting (or fast approximation).
   Wraps voigt_utils.fit_nhi_from_tau (Voigt fit) or nhi_from_tau_fast (fast mode).

3. classify_system()
   LLS    : 10^17.2 <= NHI < 10^19.0
   subDLA : 10^19.0 <= NHI < 10^20.3
   DLA    : NHI >= 10^20.3

4. build_catalog()
   Processes all skewers in batches (memory-aware). Returns an AbsorberCatalog.

fake_spectra reuse
------------------
- voigt_utils.py (this package) wraps scipy.special.wofz, using the same Faddeeva
  approach as fake_spectra internally.
- If fake_spectra is available, we cross-check absorption parameters (f_lu, lambda, Gamma).

Merging criteria
----------------
Two absorption blobs A (pixels i1..i2) and B (pixels j1..j2) with j1 > i2 are merged
into one system if:
  (j1 - i2) * dv_kms < merge_dv_kms
i.e. the gap between them in velocity space is smaller than merge_dv_kms.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .voigt_utils import fit_nhi_from_tau, nhi_from_tau_fast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification thresholds (log10 NHI)
# ---------------------------------------------------------------------------
LOG_NHI_LLS_MIN = 17.2
LOG_NHI_SUBDLA_MIN = 19.0
LOG_NHI_DLA_MIN = 20.3

CLASSES = ["LLS", "subDLA", "DLA", "forest"]  # forest = below LLS threshold


def classify_system(NHI: float) -> str:
    """Return 'LLS', 'subDLA', 'DLA', or 'forest'."""
    log_N = np.log10(max(NHI, 1e1))
    if log_N >= LOG_NHI_DLA_MIN:
        return "DLA"
    if log_N >= LOG_NHI_SUBDLA_MIN:
        return "subDLA"
    if log_N >= LOG_NHI_LLS_MIN:
        return "LLS"
    return "forest"


# ---------------------------------------------------------------------------
# Absorber data structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Absorber:
    skewer_idx: int       # row index in the HDF5 file
    pix_start: int        # first pixel index (inclusive)
    pix_end: int          # last pixel index (inclusive)
    v_center: float       # velocity of flux minimum (km/s from skewer start)
    NHI: float            # column density (cm^-2)
    b_kms: float          # Doppler parameter (km/s); NaN if fast-mode estimate
    log_NHI: float        # log10(NHI)
    absorber_class: str   # "LLS" | "subDLA" | "DLA" | "forest"
    fit_success: bool     # True if Voigt fit converged
    fast_mode: bool       # True if NHI came from fast estimator


@dataclasses.dataclass
class AbsorberCatalog:
    """Collection of absorbers for one (sim, snap)."""
    sim_name: str
    snap: int
    z: float
    dv_kms: float
    absorbers: List[Absorber] = dataclasses.field(default_factory=list)

    def by_class(self, cls: str) -> List[Absorber]:
        return [a for a in self.absorbers if a.absorber_class == cls]

    def log_nhi_array(self) -> np.ndarray:
        return np.array([a.log_NHI for a in self.absorbers])

    def nhi_array(self) -> np.ndarray:
        return np.array([a.NHI for a in self.absorbers])

    def pix_ranges(self, classes: Optional[List[str]] = None) -> List[Tuple[int, int, int]]:
        """
        Return list of (skewer_idx, pix_start, pix_end) for masking.
        Optionally filter by class list.
        """
        result = []
        for a in self.absorbers:
            if classes is None or a.absorber_class in classes:
                result.append((a.skewer_idx, a.pix_start, a.pix_end))
        return result

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {cls: 0 for cls in CLASSES}
        for a in self.absorbers:
            counts[a.absorber_class] = counts.get(a.absorber_class, 0) + 1
        return counts

    def to_dataframe(self):
        """
        Convert catalog to a pandas DataFrame with columns:
          skewer_idx, pix_start, pix_end, NHI, log_nhi, b_kms, absorber_class,
          fit_success, fast_mode.
        Returns None if pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            return None
        if not self.absorbers:
            return pd.DataFrame(columns=[
                "skewer_idx", "pix_start", "pix_end", "NHI", "log_nhi",
                "b_kms", "absorber_class", "fit_success", "fast_mode",
            ])
        records = [
            {
                "skewer_idx": a.skewer_idx,
                "pix_start": a.pix_start,
                "pix_end": a.pix_end,
                "NHI": a.NHI,
                "log_nhi": a.log_NHI,
                "b_kms": a.b_kms,
                "absorber_class": a.absorber_class,
                "fit_success": a.fit_success,
                "fast_mode": a.fast_mode,
            }
            for a in self.absorbers
        ]
        return pd.DataFrame(records)

    def save_npz(self, path: str | Path) -> None:
        if not self.absorbers:
            np.savez(
                str(path),
                sim_name=self.sim_name, snap=self.snap, z=self.z, dv_kms=self.dv_kms,
                skewer_idx=np.array([], dtype=np.int32),
                pix_start=np.array([], dtype=np.int32),
                pix_end=np.array([], dtype=np.int32),
                NHI=np.array([], dtype=np.float64),
                b_kms=np.array([], dtype=np.float64),
                fit_success=np.array([], dtype=bool),
                fast_mode=np.array([], dtype=bool),
            )
            return
        np.savez(
            str(path),
            sim_name=self.sim_name,
            snap=self.snap,
            z=self.z,
            dv_kms=self.dv_kms,
            skewer_idx=np.array([a.skewer_idx for a in self.absorbers], dtype=np.int32),
            pix_start=np.array([a.pix_start for a in self.absorbers], dtype=np.int32),
            pix_end=np.array([a.pix_end for a in self.absorbers], dtype=np.int32),
            NHI=np.array([a.NHI for a in self.absorbers], dtype=np.float64),
            b_kms=np.array([a.b_kms for a in self.absorbers], dtype=np.float64),
            fit_success=np.array([a.fit_success for a in self.absorbers], dtype=bool),
            fast_mode=np.array([a.fast_mode for a in self.absorbers], dtype=bool),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "AbsorberCatalog":
        d = np.load(str(path), allow_pickle=True)
        cat = cls(
            sim_name=str(d["sim_name"]),
            snap=int(d["snap"]),
            z=float(d["z"]),
            dv_kms=float(d["dv_kms"]),
        )
        n = len(d["skewer_idx"])
        for i in range(n):
            NHI = float(d["NHI"][i])
            ab = Absorber(
                skewer_idx=int(d["skewer_idx"][i]),
                pix_start=int(d["pix_start"][i]),
                pix_end=int(d["pix_end"][i]),
                v_center=float(d["pix_start"][i] + d["pix_end"][i]) / 2.0 * cat.dv_kms,
                NHI=NHI,
                b_kms=float(d["b_kms"][i]),
                log_NHI=np.log10(max(NHI, 1e1)),
                absorber_class=classify_system(NHI),
                fit_success=bool(d["fit_success"][i]),
                fast_mode=bool(d["fast_mode"][i]),
            )
            cat.absorbers.append(ab)
        return cat


# ---------------------------------------------------------------------------
# Per-skewer system finder
# ---------------------------------------------------------------------------

def find_systems_in_skewer(
    tau: np.ndarray,
    tau_threshold: float,
    merge_gap_pixels: int,
    min_pixels: int,
) -> List[Tuple[int, int]]:
    """
    Find absorption systems in a single skewer tau array.

    Returns list of (pix_start, pix_end) tuples (inclusive).

    Algorithm:
    1. Threshold: mark pixels where tau > tau_threshold.
    2. Find connected runs of marked pixels.
    3. Merge runs separated by < merge_gap_pixels.
    4. Drop runs shorter than min_pixels.
    """
    above = tau > tau_threshold
    if not np.any(above):
        return []

    # Find runs
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(above) - 1))

    # Merge nearby runs
    if len(runs) <= 1:
        merged = runs
    else:
        merged: List[Tuple[int, int]] = [runs[0]]
        for s, e in runs[1:]:
            gap = s - merged[-1][1] - 1
            if gap <= merge_gap_pixels:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

    # Filter by length
    return [(s, e) for (s, e) in merged if (e - s + 1) >= min_pixels]


# ---------------------------------------------------------------------------
# NHI measurement per system
# ---------------------------------------------------------------------------

def measure_nhi_for_system(
    tau: np.ndarray,
    pix_start: int,
    pix_end: int,
    dv_kms: float,
    b_init: float,
    b_bounds: Tuple[float, float],
    tau_fit_cap: float,
    max_iter: int,
    fast_mode: bool,
) -> Tuple[float, float, bool]:
    """
    Measure NHI for one absorption system.

    Returns (NHI, b_kms, fit_success).
    """
    seg = tau[pix_start:pix_end + 1].astype(np.float64)

    if fast_mode:
        NHI = nhi_from_tau_fast(seg, dv_kms)
        return NHI, float("nan"), True

    # Build velocity array centred on peak
    n_pix = len(seg)
    v_arr = np.arange(n_pix, dtype=np.float64) * dv_kms
    v_arr -= v_arr[np.argmax(seg)]  # centre on peak

    try:
        NHI, b, success = fit_nhi_from_tau(
            seg, v_arr,
            b_init=b_init,
            b_bounds=b_bounds,
            tau_cap=tau_fit_cap,
            max_iter=max_iter,
        )
    except Exception as exc:
        logger.debug("Voigt fit exception: %s", exc)
        NHI = nhi_from_tau_fast(seg, dv_kms)
        b = b_init
        success = False

    return NHI, b, success


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_skewer_batch(
    tau_batch: np.ndarray,
    batch_start: int,
    dv_kms: float,
    tau_threshold: float,
    merge_dv_kms: float,
    min_pixels: int,
    b_init: float,
    b_bounds: Tuple[float, float],
    tau_fit_cap: float,
    max_iter: int,
    fast_mode: bool,
    lls_only_above: bool = True,
    min_log_nhi: float = 17.0,
) -> List[Absorber]:
    """
    Process a batch of skewers and return all detected absorbers.

    Parameters
    ----------
    tau_batch   : shape (batch_size, nbins)
    batch_start : global row index of tau_batch[0]
    lls_only_above : if True, skip skewers whose max(tau) is below the
                     LLS tau equivalent (very fast pre-filter)
    """
    merge_gap_pixels = max(1, int(merge_dv_kms / dv_kms))
    absorbers: List[Absorber] = []

    # Rough tau floor corresponding to LLS: NHI ~ 10^17.2
    # tau_lls_approx = NHI_lls * sigma_prefactor / (sqrt(pi) * b * 1e5)
    # Using b=30 km/s: ~ 10^17.2 * 3.0e-12 / (sqrt(pi)*3e6) ≈ very rough
    # In practice tau > 1 reliably catches LLS in Lya
    # Pre-filter: skip skewers with max(tau) below the floor for min_log_nhi systems.
    # For NHI=10^min_log_nhi, b=30 km/s: tau_peak ~ NHI * sigma_peak
    # sigma_peak = _SIGMA_PREFACTOR / (sqrt(pi) * b * 1e5)
    from .voigt_utils import _SIGMA_PREFACTOR as _SP
    import math
    _b_ref = 30.0  # km/s reference b
    _tau_min = (10.0**min_log_nhi) * _SP / (math.sqrt(math.pi) * _b_ref * 1e5) * 0.1
    tau_lls_floor = max(tau_threshold, _tau_min)

    for local_idx in range(tau_batch.shape[0]):
        tau_row = tau_batch[local_idx].astype(np.float64)
        global_idx = batch_start + local_idx

        if lls_only_above and tau_row.max() < tau_lls_floor:
            continue

        systems = find_systems_in_skewer(
            tau_row, tau_threshold, merge_gap_pixels, min_pixels
        )

        for pix_start, pix_end in systems:
            NHI, b, success = measure_nhi_for_system(
                tau_row, pix_start, pix_end, dv_kms,
                b_init, b_bounds, tau_fit_cap, max_iter, fast_mode,
            )
            NHI = max(NHI, 1.0)  # clip to positive
            v_center = (pix_start + pix_end) / 2.0 * dv_kms
            log_NHI = np.log10(max(NHI, 1.0))
            if log_NHI < min_log_nhi:
                continue  # below minimum threshold (forest absorber)
            absorbers.append(Absorber(
                skewer_idx=global_idx,
                pix_start=pix_start,
                pix_end=pix_end,
                v_center=v_center,
                NHI=NHI,
                b_kms=b,
                log_NHI=log_NHI,
                absorber_class=classify_system(NHI),
                fit_success=success,
                fast_mode=fast_mode,
            ))

    return absorbers


# ---------------------------------------------------------------------------
# Main catalog builder (iterates file in batches)
# ---------------------------------------------------------------------------

def build_catalog(
    hdf5_path: Path,
    sim_name: str,
    snap: int,
    z: float,
    dv_kms: float,
    tau_threshold: float = 1.0,
    merge_dv_kms: float = 100.0,
    min_pixels: int = 2,
    b_init: float = 30.0,
    b_bounds: Tuple[float, float] = (1.0, 300.0),
    tau_fit_cap: float = 1.0e6,
    voigt_max_iter: int = 200,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    fast_mode: bool = False,
    n_workers: int = 1,
    min_log_nhi: float = 17.0,
) -> AbsorberCatalog:
    """
    Build an AbsorberCatalog by scanning all skewers in hdf5_path.

    Parameters
    ----------
    fast_mode   : use fast NHI estimator instead of full Voigt fit
    n_workers   : number of parallel workers (uses joblib)
    n_skewers   : limit total skewers processed (for benchmarking/debug)
    """
    from .io import iter_tau_batches

    catalog = AbsorberCatalog(sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms)
    t0 = time.perf_counter()

    if n_workers > 1:
        from joblib import Parallel, delayed

        # Collect all batches first (metadata only)
        batches = list(iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers))
        logger.info("Processing %d batches in parallel (n_workers=%d)", len(batches), n_workers)

        results = Parallel(n_jobs=n_workers)(
            delayed(process_skewer_batch)(
                tau_batch, row_start, dv_kms,
                tau_threshold, merge_dv_kms, min_pixels,
                b_init, b_bounds, tau_fit_cap, voigt_max_iter, fast_mode,
                min_log_nhi=min_log_nhi,
            )
            for row_start, _, tau_batch in batches
        )
        for abs_list in results:
            catalog.absorbers.extend(abs_list)

    else:
        n_processed = 0
        for row_start, row_end, tau_batch in iter_tau_batches(
            hdf5_path, batch_size=batch_size, n_skewers=n_skewers
        ):
            batch_abs = process_skewer_batch(
                tau_batch, row_start, dv_kms,
                tau_threshold, merge_dv_kms, min_pixels,
                b_init, b_bounds, tau_fit_cap, voigt_max_iter, fast_mode,
                min_log_nhi=min_log_nhi,
            )
            catalog.absorbers.extend(batch_abs)
            n_processed += (row_end - row_start)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Catalog built: %d absorbers in %.1f s (sim=%s snap=%03d z=%.2f)",
        len(catalog.absorbers), elapsed, sim_name[:20], snap, z,
    )
    return catalog
