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
   Processes all skewers in batches (memory-aware). Returns an AbsorberCatalog carrying its lineage record
   (hcd_analysis/lineage.py): the finder id FINDER_ID, stamped only when every searched skewer was cross-checked by
   the independent reference finder in that run, the raw spectra file's sha256, size and mtime, the finder settings,
   the code commit and the source digests of the finder and the N_HI estimator.

Provenance
----------
catalog.npz holds the arrays and the record (field ``lineage_json``). ``save_npz`` writes only a catalogue stamped by
build_catalog. ``load_npz`` and ``load_validated_catalog`` (the release loader: record, allowed finder id, then the
value checks) refuse a catalogue without a record; ``load_catalog_historical(path, reason=...)`` is the only reader of
an old one, and what it returns is never release-eligible.

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

from . import lineage as L
from .voigt_utils import fit_nhi_from_tau, nhi_from_tau_fast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finder identity (written into every catalogue's lineage record)
# ---------------------------------------------------------------------------
# The id names the catalogue algorithm: the system finder (find_systems_in_skewer, cross-checked against
# find_systems_in_skewer_reference) and the N_HI estimator (measure_nhi_for_system). "hcd_priya.finder/1" was the
# pre-fix finder that listed every box-edge system twice (docs/bugs_found.md, bug 8); it was never stamped. Any change
# that can change a catalogue's output must bump FINDER_ID, add the golden output of the new id
# (tests/test_catalog_systems.py) and, after review, the new id to RELEASE_FINDER_IDS.
FINDER_ID = "hcd_priya.finder/2"
RELEASE_FINDER_IDS = frozenset({FINDER_ID})
CATALOG_KIND_VERSION = 2          # catalog.npz layout: the seven entry arrays, four scalars, lineage_json

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
    lineage: Optional[dict] = None             # the hcd_priya.lineage/1 record (build_catalog, load_npz)
    historical_reason: Optional[str] = None    # set by load_catalog_historical only
    original_lineage: Optional[dict] = None    # load_catalog_historical: the record as stored in the file

    @property
    def release_eligible(self) -> bool:
        return (self.historical_reason is None and self.lineage is not None
                and self.lineage.get("release_eligible") is True)

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
        Convert catalog to a pandas DataFrame (if available) or a plain dict of arrays.
        Columns: skewer_idx, pix_start, pix_end, NHI, log_nhi, b_kms, absorber_class.
        """
        arrays = {
            "skewer_idx": np.array([a.skewer_idx for a in self.absorbers], dtype=np.int32),
            "pix_start":  np.array([a.pix_start  for a in self.absorbers], dtype=np.int32),
            "pix_end":    np.array([a.pix_end    for a in self.absorbers], dtype=np.int32),
            "NHI":        np.array([a.NHI        for a in self.absorbers], dtype=np.float64),
            "log_nhi":    np.array([a.log_NHI    for a in self.absorbers], dtype=np.float64),
            "b_kms":      np.array([a.b_kms      for a in self.absorbers], dtype=np.float64),
            "absorber_class": np.array([a.absorber_class for a in self.absorbers]),
        }
        try:
            import pandas as pd
            return pd.DataFrame(arrays)
        except ImportError:
            return arrays  # plain dict of numpy arrays — works with compute_p1d_excl_nhi

    def _arrays(self) -> Dict[str, np.ndarray]:
        """The arrays catalog.npz holds (its payload, digested into the lineage record)."""
        ab = self.absorbers
        return {
            "sim_name": np.array(str(self.sim_name)),
            "snap": np.array(int(self.snap), dtype=np.int64),
            "z": np.array(float(self.z), dtype=np.float64),
            "dv_kms": np.array(float(self.dv_kms), dtype=np.float64),
            "skewer_idx": np.array([a.skewer_idx for a in ab], dtype=np.int32),
            "pix_start": np.array([a.pix_start for a in ab], dtype=np.int32),
            "pix_end": np.array([a.pix_end for a in ab], dtype=np.int32),
            "NHI": np.array([a.NHI for a in ab], dtype=np.float64),
            "b_kms": np.array([a.b_kms for a in ab], dtype=np.float64),
            "fit_success": np.array([a.fit_success for a in ab], dtype=bool),
            "fast_mode": np.array([a.fast_mode for a in ab], dtype=bool),
        }

    def save_npz(self, path: str | Path) -> None:
        """Write catalog.npz with its lineage record. Refused (CatalogProvenanceError) unless the catalogue carries the
        record build_catalog stamped with a finder id, and its entries are still the ones that record digested: a
        catalogue read by load_catalog_historical, or changed after it was built, cannot be written."""
        rec = self.lineage
        if self.historical_reason is not None:
            raise CatalogProvenanceError(f"{path}: a catalogue read as historical ({self.historical_reason}) is never "
                                         "written again")
        if rec is None or rec.get("kind") != "catalog" or not rec.get("finder_ids"):
            raise CatalogProvenanceError(f"{path}: refusing to save a catalogue without a finder-stamped lineage record "
                                         "(only build_catalog, with the finder cross-check run, stamps one)")
        try:
            L.write_npz(path, self._arrays(), rec)
        except L.LineageError as e:
            raise CatalogProvenanceError(f"{path}: refusing to save: {e} (the catalogue was changed after "
                                         "build_catalog stamped it)") from None

    @classmethod
    def load_npz(cls, path: str | Path) -> "AbsorberCatalog":
        """Read catalog.npz and its lineage record (no pickle). Refused (CatalogProvenanceError) when the file has no
        record, a malformed one, or arrays that differ from the ones the record digested. This checks the record's
        form only; ``load_validated_catalog`` is the release loader, and ``load_catalog_historical`` reads an old
        catalogue."""
        try:
            arrays, rec = L.read_npz(path, require_record=False)
        except L.LineageError as e:
            raise CatalogProvenanceError(f"{path}: {e}") from None
        if rec is None:
            raise CatalogProvenanceError(
                f"{path}: no lineage record: written before catalogues carried one (before the box-edge fix, "
                "docs/bugs_found.md, bug 8, or by a pre-lineage build). It is historical: read it with "
                "load_catalog_historical(path, reason=...), or rebuild it")
        if rec["kind"] != "catalog":
            raise CatalogProvenanceError(f"{path}: the lineage record describes a {rec['kind']!r}, not a catalog")
        cat = cls._from_arrays(arrays)
        cat.lineage = rec
        return cat

    @classmethod
    def _from_arrays(cls, d) -> "AbsorberCatalog":
        cat = cls(
            sim_name=str(d["sim_name"]),
            snap=int(d["snap"]),
            z=float(d["z"]),
            dv_kms=float(d["dv_kms"]),
        )
        # Materialise arrays up front. Indexing an NpzFile (d["NHI"][i])
        # inside a loop re-reads the whole archive each time -> O(n^2).
        si = np.asarray(d["skewer_idx"])
        ps = np.asarray(d["pix_start"])
        pe = np.asarray(d["pix_end"])
        NHI = np.asarray(d["NHI"], dtype=np.float64)
        bkms = np.asarray(d["b_kms"])
        fs = np.asarray(d["fit_success"])
        fm = np.asarray(d["fast_mode"])
        log_NHI_arr = np.log10(np.maximum(NHI, 1e1))
        dv = cat.dv_kms
        absorbers = cat.absorbers
        for i in range(len(si)):
            NHI_i = float(NHI[i])
            absorbers.append(Absorber(
                skewer_idx=int(si[i]),
                pix_start=int(ps[i]),
                pix_end=int(pe[i]),
                v_center=float(ps[i] + pe[i]) / 2.0 * dv,
                NHI=NHI_i,
                b_kms=float(bkms[i]),
                log_NHI=float(log_NHI_arr[i]),
                absorber_class=classify_system(NHI_i),
                fit_success=bool(fs[i]),
                fast_mode=bool(fm[i]),
            ))
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
    For systems that wrap across the periodic box boundary, pix_end >= len(tau);
    the wrap segment occupies pixels [pix_start:] and [:pix_end - len(tau) + 1].

    Every system is listed once. A skewer whose gaps are all merged (or whose tau is above threshold everywhere) is one
    system (0, n - 1), of n pixels, kept only if n >= min_pixels.

    Algorithm:
    1. Threshold: mark pixels where tau > tau_threshold.
    2. Use a doubled array to handle periodic boundary wrap-around.
    3. Merge runs separated by gaps of at most merge_gap_pixels.
    4. Drop runs shorter than min_pixels.
    5. Keep the runs starting in [0, n) (the second copy repeats them), then drop the first-copy runs that a wrapping
       run already covers (its own head, found again).
    """
    n = len(tau)
    above = tau > tau_threshold
    if not np.any(above):
        return []

    # Double the array: [tau | tau] so wrap-around runs are contiguous.
    above2 = np.concatenate([above, above])
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(above2):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, 2 * n - 1))

    # Merge nearby runs
    if len(runs) > 1:
        merged: List[Tuple[int, int]] = [runs[0]]
        for s, e in runs[1:]:
            gap = s - merged[-1][1] - 1
            if gap <= merge_gap_pixels:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
    else:
        merged = list(runs)

    # Keep only runs with pix_start < n (first copy only; second copy is duplicate).
    # pix_end may be >= n for systems wrapping across the boundary.
    result = []
    for s, e in merged:
        if s >= n:
            break  # remaining runs are duplicates from the second copy
        if (e - s + 1) >= min_pixels:
            result.append((s, e))
    # A run that wraps (e >= n) already holds the first copy's pixels [0, e - n]; the first-copy runs that start
    # there are its own head, found again (the box-edge duplicate, docs/bugs_found.md). Drop them. If the wrapping run
    # itself starts at or before e - n, every gap is merged: one system covering the whole skewer (n pixels), reported
    # once and kept only if n >= min_pixels.
    if result and result[-1][1] >= n:
        head_end = result[-1][1] - n
        if result[-1][0] > head_end:
            result = [(s, e) for s, e in result if s > head_end]
        else:
            result = [(0, n - 1)] if n >= min_pixels else []
    return result


def find_systems_in_skewer_reference(
    tau: np.ndarray,
    tau_threshold: float,
    merge_gap_pixels: int,
    min_pixels: int,
) -> List[Tuple[int, int]]:
    """
    Independent implementation of find_systems_in_skewer, used to cross-check it (process_skewer_batch refuses a
    skewer on which the two disagree) and in the tests. Same conventions and output: (pix_start, pix_end) with
    0 <= pix_start < n and pix_end >= n for a system that wraps; a skewer whose gaps are all merged is one system
    (0, n - 1).

    Algorithm: rotate the thresholded mask so that it starts on a below-threshold pixel (no run then crosses the
    array start), find linear runs, merge gaps <= merge_gap_pixels including the circular gap between the last and
    the first run, drop runs shorter than min_pixels, rotate the starts back.
    """
    above = np.asarray(tau) > tau_threshold
    n = above.size
    below = np.flatnonzero(~above)
    if below.size == n:
        return []
    if below.size == 0:
        return [(0, n - 1)] if n >= min_pixels else []
    r = int(below[0])
    rot = np.roll(above, -r)                                    # rot[0] is below threshold
    d = np.diff(np.concatenate([[0], rot.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    runs = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if runs and s - runs[-1][1] - 1 <= merge_gap_pixels:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    wrap_gap = (n - 1 - runs[-1][1]) + runs[0][0]
    if wrap_gap <= merge_gap_pixels:
        if len(runs) == 1:
            return [(0, n - 1)] if n >= min_pixels else []
        first = runs.pop(0)
        runs[-1][1] = first[1] + n                              # the last run continues through the array start
    out = []
    for s, e in runs:
        if e - s + 1 >= min_pixels:
            s0 = (s + r) % n
            out.append((s0, s0 + (e - s)))
    return sorted(out)


class CatalogIntegrityError(ValueError):
    """An absorber catalogue failed an integrity check (independent finders disagree, entries overlap, counts do not
    reconcile, or a cached catalogue loaded for reuse is invalid). Nothing is written past a failed check."""


class CatalogProvenanceError(CatalogIntegrityError):
    """A catalogue's lineage record is missing, malformed, from a finder outside the release allow-list, or does not
    match what the caller expects (settings, raw spectra, skewer count)."""


def find_wrap_duplicates(skewer_idx, pix_start, pix_end, n: int) -> np.ndarray:
    """Boolean mask of the entries the pre-fix finder wrote wrongly: (1) a box-edge duplicate, an entry with
    pix_end < n whose pix_end equals pix_end_w - n of a wrapping entry w (pix_end_w >= n) on the same skewer; (2) an
    entry spanning more than the skewer (pix_end - pix_start + 1 > n, a whole-skewer system listed twice over).
    Read-only audit of existing catalogues (they are not rewritten); a catalogue built by the fixed finder has
    neither."""
    sk = np.asarray(skewer_idx, dtype=np.int64)
    ps = np.asarray(pix_start, dtype=np.int64)
    pe = np.asarray(pix_end, dtype=np.int64)
    wrap = pe >= n
    key = sk * (2 * n) + pe
    target = sk[wrap] * (2 * n) + (pe[wrap] - n)
    return ((~wrap) & np.isin(key, target)) | (pe - ps + 1 > n)


def finder_settings(*, tau_threshold, merge_dv_kms, min_pixels, b_init, b_bounds, tau_fit_cap, voigt_max_iter,
                    n_skewers, fast_mode, min_log_nhi) -> dict:
    """The catalogue-determining settings, as build_catalog records them (and as a caller compares them before reusing
    a catalogue). The Voigt-only settings are recorded in fast mode too."""
    return L.canonicalise({
        "tau_threshold": float(tau_threshold), "merge_dv_kms": float(merge_dv_kms), "min_pixels": int(min_pixels),
        "b_init_kms": float(b_init), "b_bounds": [float(b_bounds[0]), float(b_bounds[1])],
        "tau_fit_cap": float(tau_fit_cap), "voigt_max_iter": int(voigt_max_iter),
        "n_skewers": None if n_skewers is None else int(n_skewers), "fast_mode": bool(fast_mode),
        "min_log_nhi": float(min_log_nhi), "lls_only_above": True,
    })


# The finder settings of the release catalogues (config/default.yaml, absorber section, all skewers). A release consumer
# compares a catalogue's recorded settings with these (release eligibility is provenance only); a test pins them to
# the configuration file.
RELEASE_FINDER_SETTINGS = finder_settings(tau_threshold=100.0, merge_dv_kms=100.0, min_pixels=2, b_init=30.0,
                                          b_bounds=(1.0, 300.0), tau_fit_cap=1.0e6, voigt_max_iter=200,
                                          n_skewers=None, fast_mode=True, min_log_nhi=17.2)


def check_catalog_record(cat: "AbsorberCatalog", nbins: Optional[int] = None, *, settings: Optional[dict] = None,
                         raw_path=None, n_skewers_built: Optional[int] = None,
                         require_release_eligible: bool = False) -> dict:
    """Refuse (CatalogProvenanceError) a catalogue whose lineage record is not that of a catalogue built by an allowed
    finder with the cross-check run, or that differs from what the caller expects: the skewer length ``nbins``, the
    finder ``settings`` (``finder_settings``), the raw spectra file ``raw_path`` (its sha256; a matching size and mtime
    are accepted without rereading), the number of skewers searched, and, with ``require_release_eligible``, release
    eligibility. A release-eligible load must state all three expectations (``settings``, usually
    ``RELEASE_FINDER_SETTINGS``, ``raw_path`` and ``n_skewers_built``, the file's skewer count) and refuses a partial
    build (a recorded skewer limit, or fewer skewers than the raw file holds). Returns the record."""
    if require_release_eligible and (settings is None or raw_path is None or n_skewers_built is None):
        raise CatalogProvenanceError("a release-eligible load must state the expected finder settings, raw spectra "
                                     "file and skewer count (settings, raw_path, n_skewers_built)")
    rec = cat.lineage
    if cat.historical_reason is not None:
        raise CatalogProvenanceError(f"read as historical ({cat.historical_reason}); never reused")
    if rec is None:
        raise CatalogProvenanceError("no lineage record")
    try:
        L.check_record(rec, RELEASE_FINDER_IDS)
    except L.LineageError as e:
        raise CatalogProvenanceError(str(e)) from None
    if rec["kind"] != "catalog" or rec["kind_version"] != CATALOG_KIND_VERSION:
        raise CatalogProvenanceError(f"a {rec['kind']} record of version {rec['kind_version']}, not catalog "
                                     f"{CATALOG_KIND_VERSION}")
    bad = [f for f in rec["finder_ids"] if f not in RELEASE_FINDER_IDS]
    if not rec["finder_ids"] or bad:
        raise CatalogProvenanceError(f"finder ids {rec['finder_ids']} are not in the release allow-list "
                                     f"{sorted(RELEASE_FINDER_IDS)}")
    prod = rec["product"]
    if not (isinstance(prod.get("payload_sha256"), str) and len(prod["payload_sha256"]) == 64
            and prod["payload_sha256"] == L.arrays_sha256(cat._arrays())):
        raise CatalogProvenanceError("the record's payload_sha256 is missing or is not the digest of the entries")
    finder = prod.get("finder") or {}
    if finder.get("verified") is not True:
        raise CatalogProvenanceError("the record does not say the finder cross-check ran")
    mine = {"sim_name": cat.sim_name, "snap": cat.snap, "n_absorbers": len(cat.absorbers)}
    theirs = {k: prod.get(k) for k in mine}
    if mine != theirs:
        raise CatalogProvenanceError(f"the record describes {theirs}, the file holds {mine}")
    if nbins is not None and prod.get("nbins") != int(nbins):
        raise CatalogProvenanceError(f"built on skewers of {prod.get('nbins')} pixels, expected {int(nbins)}")
    if settings is not None:
        want = L.canonicalise(settings)
        if rec["settings"] != want:
            from .run_settings import _differences, _lookup
            diff = _differences(rec["settings"], want)
            detail = "; ".join(f"{k}: recorded {_lookup(rec['settings'], k)!r}, expected {_lookup(want, k)!r}"
                               for k in diff[:12])
            raise CatalogProvenanceError(f"built with other finder settings: {detail}")
    if n_skewers_built is not None and prod.get("n_skewers_built") != int(n_skewers_built):
        raise CatalogProvenanceError(f"built from {prod.get('n_skewers_built')} skewers, expected {int(n_skewers_built)}")
    if raw_path is not None:
        try:
            L.verify_input(rec, "raw_spectra", raw_path)
        except (L.LineageError, OSError) as e:
            raise CatalogProvenanceError(str(e)) from None
    if require_release_eligible:
        if not rec["release_eligible"]:
            raise CatalogProvenanceError("not release-eligible: " + "; ".join(rec["ineligible_reasons"]))
        raw_info = L.find_input(rec, "raw_spectra").get("info") or {}
        if rec["settings"].get("n_skewers") is not None or prod.get("n_skewers_built") != raw_info.get("n_skewers"):
            raise CatalogProvenanceError(f"a partial build ({prod.get('n_skewers_built')} of "
                                         f"{raw_info.get('n_skewers')} skewers, limit "
                                         f"{rec['settings'].get('n_skewers')}); a release catalogue covers the file")
    return rec


def load_catalog_historical(path, *, reason: str) -> "AbsorberCatalog":
    """Read a catalogue as it is, with or without a lineage record and with no value check: the only reader of a
    catalogue written before records existed (a pre-fix catalogue, with its box-edge duplicates). ``reason`` (a
    non-empty string) is logged and kept on the result, which is never release-eligible and cannot be saved, reused
    by the pipeline or passed to a release loader. Its ``lineage`` is None (no record in the file) or a copy of the
    stored record marked not release-eligible and historical, which ``lineage.input_entry`` refuses, so nothing
    built from it can be eligible; the stored record is kept as ``original_lineage``."""
    if not (isinstance(reason, str) and reason.strip()):
        raise ValueError("load_catalog_historical needs a non-empty reason")
    try:
        arrays, rec = L.read_npz(path, require_record=False)
    except L.LineageError as e:
        raise CatalogProvenanceError(f"{path}: {e}") from None
    cat = AbsorberCatalog._from_arrays(arrays)
    cat.original_lineage = rec
    cat.lineage = None if rec is None else _historical_record(rec, reason.strip())
    cat.historical_reason = reason.strip()
    logger.warning("historical read of %s (%s): not release-eligible", path, cat.historical_reason)
    return cat


def _historical_record(rec: dict, reason: str) -> dict:
    """A copy of ``rec`` that says what a historical read makes of it: not release-eligible, with the reason."""
    out = L.canonicalise(rec)
    out["product"] = dict(out["product"], historical_reason=reason)
    out["release_eligible"] = False
    out["ineligible_reasons"] = list(out["ineligible_reasons"]) + [f"read by load_catalog_historical: {reason}"]
    return L.check_record(out)


def load_validated_catalog(path, n: int, *, settings: Optional[dict] = None, raw_path=None,
                           n_skewers_built: Optional[int] = None,
                           require_release_eligible: bool = False) -> "AbsorberCatalog":
    """The release loader of a cached catalogue. Refused (CatalogIntegrityError, or its subclass
    CatalogProvenanceError for the record) unless:
      1. it carries a lineage record from build_catalog with an allowed finder id and the finder cross-check run, and
         its arrays are those the record digested; the caller's expectations (``check_catalog_record``: the skewer
         length ``n``, ``settings``, ``raw_path``, ``n_skewers_built``, ``require_release_eligible``) hold;
      2. (second line, independent of the record) its entries are distinct systems (``validate_catalog_entries``
         with ``n``) and its N_HI values pass ``check_catalog_values``.
    A catalogue built by the pre-fix finder, or written without a record, is refused, never re-used silently; read it
    with ``load_catalog_historical``, or rebuild it."""
    try:
        cat = AbsorberCatalog.load_npz(path)
        check_catalog_record(cat, n, settings=settings, raw_path=raw_path, n_skewers_built=n_skewers_built,
                             require_release_eligible=require_release_eligible)
    except CatalogProvenanceError as e:
        msg = str(e)
        prefix = f"{path}: "
        raise CatalogProvenanceError(f"{path}: cached catalogue refused "
                                     f"({msg[len(prefix):] if msg.startswith(prefix) else msg})") from None
    if cat.absorbers:
        try:
            validate_catalog_entries([a.skewer_idx for a in cat.absorbers], [a.pix_start for a in cat.absorbers],
                                     [a.pix_end for a in cat.absorbers], n)
            check_catalog_values(cat)
        except CatalogIntegrityError as e:
            raise CatalogIntegrityError(f"{path}: cached catalogue refused ({e}); it was built before the box-edge "
                                        f"fix (docs/bugs_found.md, bug 8) or is damaged; rebuild it") from None
    return cat


def _class_counts(catalog: "AbsorberCatalog"):
    nhi = np.array([a.NHI for a in catalog.absorbers], dtype=np.float64)
    if not np.all(np.isfinite(nhi)):
        raise CatalogIntegrityError(f"{int(np.sum(~np.isfinite(nhi)))} absorbers with a non-finite N_HI")
    labels = [a.absorber_class for a in catalog.absorbers]
    by_label = {c: labels.count(c) for c in ("LLS", "subDLA", "DLA", "forest")}
    logn = np.log10(np.maximum(nhi, 1e1))
    by_value = {
        "LLS": int(np.sum((logn >= LOG_NHI_LLS_MIN) & (logn < LOG_NHI_SUBDLA_MIN))),
        "subDLA": int(np.sum((logn >= LOG_NHI_SUBDLA_MIN) & (logn < LOG_NHI_DLA_MIN))),
        "DLA": int(np.sum(logn >= LOG_NHI_DLA_MIN)),
        "forest": int(np.sum(logn < LOG_NHI_LLS_MIN)),
    }
    if by_label != by_value:
        raise CatalogIntegrityError(f"class counts from labels {by_label} != from NHI values {by_value}")
    return by_label, logn


def check_catalog_values(catalog: "AbsorberCatalog") -> None:
    """Refuse non-finite N_HI, and class labels that disagree with the classes of N_HI. Run before a catalogue is saved
    (``build_catalog``) and when a cached one is loaded (``load_validated_catalog``). ``catalog.npz`` stores no labels:
    ``load_npz``, like ``process_skewer_batch``, assigns them with ``classify_system``, so on every catalogue the label
    check compares ``classify_system`` with the class thresholds (drift between the two)."""
    _class_counts(catalog)


def validate_catalog_entries(skewer_idx, pix_start, pix_end, n: int) -> None:
    """Refuse a catalogue whose entries are not distinct systems: every entry must satisfy 0 <= pix_start < n and
    pix_start <= pix_end < pix_start + n, and no two entries of one skewer may share a pixel modulo n (this catches
    the box-edge duplicate, a head inside its own wrapping system). Independent of the finder."""
    sk = np.asarray(skewer_idx, dtype=np.int64)
    ps = np.asarray(pix_start, dtype=np.int64)
    pe = np.asarray(pix_end, dtype=np.int64)
    bad = np.flatnonzero((ps < 0) | (ps >= n) | (pe < ps) | (pe >= ps + n))
    if bad.size:
        i = int(bad[0])
        raise CatalogIntegrityError(f"{bad.size} entries with impossible pixel spans, first: skewer {sk[i]} "
                                    f"({ps[i]}, {pe[i]}), n = {n}")
    o = np.lexsort((ps, sk))
    s, a, b = sk[o], ps[o], pe[o]
    lin = np.flatnonzero((s[1:] == s[:-1]) & (a[1:] <= b[:-1]))
    if lin.size:
        i = int(lin[0])
        raise CatalogIntegrityError(f"{lin.size} overlapping neighbours, first on skewer {s[i]}: ({a[i]}, {b[i]}) and "
                                    f"({a[i + 1]}, {b[i + 1]})")
    keys = s * (2 * n) + a
    w = np.flatnonzero(b >= n)
    lo = np.searchsorted(keys, s[w] * (2 * n), "left")
    hi = np.searchsorted(keys, s[w] * (2 * n) + (b[w] - n), "right")
    hit = np.flatnonzero(hi > lo)
    if hit.size:
        j = int(w[hit[0]])
        raise CatalogIntegrityError(f"{hit.size} wrapping entries overlap an entry at the start of their skewer "
                                    f"(box-edge duplicate), first on skewer {s[j]}: ({a[j]}, {b[j]})")


def reconcile_counts(catalog: "AbsorberCatalog", cddf_counts, cddf_edges) -> Dict[str, int]:
    """Count the catalogue's absorbers three ways and refuse unless they agree: (1) the stored per-entry class labels,
    (2) the classes recomputed from each entry's NHI with the class thresholds, (3) the CDDF histogram (entries inside
    the CDDF edges). All three come from the same entries, so they guard the labels, the thresholds and the CDDF against
    drift, not against a duplicated entry (the finder cross-check and validate_catalog_entries do that). Returns the
    per-class counts written to meta.json."""
    by_label, logn = _class_counts(catalog)
    edges = np.asarray(cddf_edges, dtype=np.float64)
    inside = int(np.sum((logn >= edges[0]) & (logn <= edges[-1])))
    if int(np.sum(cddf_counts)) != inside:
        raise CatalogIntegrityError(f"CDDF holds {int(np.sum(cddf_counts))} absorbers, the catalogue {inside} inside "
                                    f"its edges [{edges[0]}, {edges[-1]}]")
    return by_label


# ---------------------------------------------------------------------------
# NHI measurement per system
# ---------------------------------------------------------------------------

_WING_PIXELS = 200  # ±2000 km/s at dv≈10 km/s — captures DLA damping wings


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

    The fitting window is expanded by _WING_PIXELS on each side of the
    detected segment so the Voigt fitter can see the damping wings.
    For DLAs (logNHI≥20.3) the wings extend ±1000–2000 km/s and carry
    most of the NHI information; fitting only the saturated core produces
    large underestimates.

    Returns (NHI, b_kms, fit_success).
    """
    n = len(tau)
    if pix_end >= n:
        # Periodic boundary wrap: build the full wrapped segment
        core = np.concatenate([tau[pix_start:], tau[:pix_end - n + 1]])
    else:
        core = tau[pix_start:pix_end + 1]

    if fast_mode:
        NHI = nhi_from_tau_fast(core.astype(np.float64), dv_kms)
        return NHI, float("nan"), True

    # Expand window by wing pixels to include damping wings
    if pix_end >= n:
        # For wrapped systems use the core only (wing expansion too complex)
        seg = core.astype(np.float64)
        peak_idx = np.argmax(seg)
    else:
        lo = max(0, pix_start - _WING_PIXELS)
        hi = min(n - 1, pix_end + _WING_PIXELS)
        seg = tau[lo:hi + 1].astype(np.float64)
        # Peak index within the expanded segment
        peak_idx = np.argmax(seg)

    v_arr = (np.arange(len(seg), dtype=np.float64) - peak_idx) * dv_kms

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
        NHI = nhi_from_tau_fast(core.astype(np.float64), dv_kms)
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
    verify_finder: bool = True,
) -> List[Absorber]:
    """
    Process a batch of skewers and return all detected absorbers.

    Parameters
    ----------
    tau_batch   : shape (batch_size, nbins)
    batch_start : global row index of tau_batch[0]
    lls_only_above : if True, skip skewers whose max(tau) is below the
                     LLS tau equivalent (very fast pre-filter)
    verify_finder : if True (the default), every skewer's systems are found a second time by the independent
                    find_systems_in_skewer_reference and the batch is refused (CatalogIntegrityError) unless the two
                    agree exactly
    """
    return _scan_skewer_batch(tau_batch, batch_start, dv_kms, tau_threshold, merge_dv_kms, min_pixels, b_init, b_bounds,
                              tau_fit_cap, max_iter, fast_mode, lls_only_above=lls_only_above, min_log_nhi=min_log_nhi,
                              verify_finder=verify_finder)[0]


def _scan_skewer_batch(
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
    verify_finder: bool = True,
) -> Tuple[List[Absorber], int, int]:
    """process_skewer_batch, returning also the number of skewers searched (those past the pre-filter) and the
    number of those whose systems the reference finder found identically: the evidence build_catalog needs before it
    stamps FINDER_ID."""
    merge_gap_pixels = max(1, int(merge_dv_kms / dv_kms))
    absorbers: List[Absorber] = []
    n_searched = n_verified = 0

    # Pre-filter: skip skewers whose max(tau) is clearly below the tau_peak
    # expected for an absorber at NHI=10^min_log_nhi, b=30 km/s.
    # tau_peak = NHI * _SIGMA_PREFACTOR / (sqrt(pi) * b_kms)
    from .voigt_utils import _SIGMA_PREFACTOR as _SP
    import math
    _b_ref = 30.0  # km/s reference b
    _tau_min = (10.0**min_log_nhi) * _SP / (math.sqrt(math.pi) * _b_ref) * 0.1
    tau_lls_floor = max(tau_threshold, _tau_min)

    for local_idx in range(tau_batch.shape[0]):
        tau_row = tau_batch[local_idx].astype(np.float64)
        global_idx = batch_start + local_idx

        if lls_only_above and tau_row.max() < tau_lls_floor:
            continue

        systems = find_systems_in_skewer(
            tau_row, tau_threshold, merge_gap_pixels, min_pixels
        )
        n_searched += 1
        if verify_finder:
            ref = find_systems_in_skewer_reference(tau_row, tau_threshold, merge_gap_pixels, min_pixels)
            if systems != ref:
                raise CatalogIntegrityError(f"skewer {global_idx}: the two system finders disagree: {systems} != {ref}")
            n_verified += 1

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

    return absorbers, n_searched, n_verified


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
    verify_finder: bool = True,
    run_settings_sha256: Optional[str] = None,
    record_lineage: bool = True,
) -> AbsorberCatalog:
    """
    Build an AbsorberCatalog by scanning all skewers in hdf5_path.

    Parameters
    ----------
    fast_mode   : use fast NHI estimator instead of full Voigt fit
    n_workers   : number of parallel workers (uses joblib)
    n_skewers   : limit total skewers processed (for benchmarking/debug)
    verify_finder : cross-check every searched skewer with the reference finder (the default). FINDER_ID is stamped
                    only when every batch reports that check for every skewer it searched; without it the catalogue
                    carries no finder id and save_npz refuses it
    run_settings_sha256 : sha256 of the output root's run_settings.json, recorded in the lineage
    record_lineage : hash the raw spectra file and attach the lineage record (the default). False (timing runs that
                     never save) leaves ``lineage`` None, so the catalogue cannot be saved
    """
    from .io import iter_tau_batches

    catalog = AbsorberCatalog(sim_name=sim_name, snap=snap, z=z, dv_kms=dv_kms)
    t0 = time.perf_counter()
    raw = L.file_identity(hdf5_path) if record_lineage else None    # before the scan: the bytes it reads
    scan_args = (dv_kms, tau_threshold, merge_dv_kms, min_pixels, b_init, b_bounds, tau_fit_cap, voigt_max_iter,
                 fast_mode)
    n_rows = n_searched = n_verified = 0

    if n_workers > 1:
        from joblib import Parallel, delayed

        # Collect all batches first (metadata only)
        batches = list(iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers))
        logger.info("Processing %d batches in parallel (n_workers=%d)", len(batches), n_workers)

        results = Parallel(n_jobs=n_workers)(
            delayed(_scan_skewer_batch)(tau_batch, row_start, *scan_args, min_log_nhi=min_log_nhi,
                                        verify_finder=verify_finder)
            for row_start, _, tau_batch in batches
        )
        for abs_list, n_s, n_v in results:
            catalog.absorbers.extend(abs_list)
            n_searched += n_s
            n_verified += n_v
        n_rows = sum(row_end - row_start for row_start, row_end, _ in batches)
        nbins = batches[0][2].shape[1] if batches else None

    else:
        nbins = None
        for row_start, row_end, tau_batch in iter_tau_batches(
            hdf5_path, batch_size=batch_size, n_skewers=n_skewers
        ):
            batch_abs, n_s, n_v = _scan_skewer_batch(tau_batch, row_start, *scan_args, min_log_nhi=min_log_nhi,
                                                     verify_finder=verify_finder)
            catalog.absorbers.extend(batch_abs)
            n_searched += n_s
            n_verified += n_v
            n_rows += (row_end - row_start)
            nbins = tau_batch.shape[1]

    if nbins is not None and catalog.absorbers:                # refuse overlapping entries (independent of the finder)
        validate_catalog_entries([a.skewer_idx for a in catalog.absorbers], [a.pix_start for a in catalog.absorbers],
                                 [a.pix_end for a in catalog.absorbers], nbins)
    check_catalog_values(catalog)                              # before any caller can save it
    if record_lineage:
        settings = finder_settings(tau_threshold=tau_threshold, merge_dv_kms=merge_dv_kms, min_pixels=min_pixels,
                                   b_init=b_init, b_bounds=b_bounds, tau_fit_cap=tau_fit_cap,
                                   voigt_max_iter=voigt_max_iter, n_skewers=n_skewers, fast_mode=fast_mode,
                                   min_log_nhi=min_log_nhi)
        catalog.lineage = _catalog_record(hdf5_path, raw, catalog, settings, n_rows, n_searched, n_verified,
                                          verify_finder, run_settings_sha256)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Catalog built: %d absorbers in %.1f s (sim=%s snap=%03d z=%.2f)",
        len(catalog.absorbers), elapsed, sim_name[:20], snap, z,
    )
    return catalog


def _source_digests() -> Dict[str, Optional[str]]:
    """sha256 of the finder's and the N_HI estimator's module files (information: what ran)."""
    from . import voigt_utils
    out: Dict[str, Optional[str]] = {}
    for name, f in (("catalog.py", __file__), ("voigt_utils.py", voigt_utils.__file__)):
        try:
            out[name] = L.sha256_file(f)
        except (OSError, TypeError):
            out[name] = None
    return out


def _estimator_source_sha256() -> Optional[str]:
    """sha256 of the N_HI estimator's source: measure_nhi_for_system, the voigt_utils functions it calls and their
    constants (information: what ran)."""
    import hashlib
    import inspect
    from . import voigt_utils as V
    try:
        parts = [inspect.getsource(f) for f in (measure_nhi_for_system, V.nhi_from_tau_fast, V.fit_nhi_from_tau,
                                                V._residuals, V.tau_voigt, V.voigt_profile_phi)]
    except (OSError, TypeError):
        return None
    parts.append(repr((_WING_PIXELS, V._SIGMA_PREFACTOR, V._VOIGT_A_PREFACTOR)))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _catalog_record(hdf5_path, raw, catalog, settings, n_rows, n_searched, n_verified, verify_finder,
                    run_settings_sha256) -> dict:
    """The catalogue's lineage record. FINDER_ID only when the reference finder checked every searched skewer."""
    from .io import read_header
    st = Path(hdf5_path).stat()
    if (int(st.st_size), int(st.st_mtime_ns)) != (raw["size"], raw["mtime_ns"]):
        raise CatalogProvenanceError(f"{hdf5_path}: the raw spectra file changed while the catalogue was built")
    hdr = read_header(hdf5_path)
    verified = bool(verify_finder) and n_verified == n_searched
    product = {
        "sim_name": catalog.sim_name, "snap": int(catalog.snap), "z": float(catalog.z),
        "dv_kms": float(catalog.dv_kms), "nbins": int(hdr.nbins), "n_skewers_built": int(n_rows),
        "n_absorbers": len(catalog.absorbers), "payload_sha256": L.arrays_sha256(catalog._arrays()),
        "finder": {"verified": verified, "n_skewers_searched": int(n_searched), "n_skewers_verified": int(n_verified),
                   "source_sha256": _source_digests(), "estimator_source_sha256": _estimator_source_sha256()},
    }
    info = {"path": str(Path(hdf5_path).resolve()), "nbins": int(hdr.nbins), "n_skewers": int(hdr.n_skewers),
            "redshift": float(hdr.redshift), "box_kpc_h": float(hdr.box)}
    return L.build_record(
        "catalog", CATALOG_KIND_VERSION,
        inputs=[L.input_entry("raw_spectra", "raw_spectra", raw, info=info)],
        finder_ids=[FINDER_ID] if verified else [],
        settings=settings, code=L.code_identity(), allowed_finder_ids=RELEASE_FINDER_IDS,
        run_settings_sha256=run_settings_sha256, product=product)
