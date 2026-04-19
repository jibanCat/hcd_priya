"""
I/O layer: file discovery, HDF5 reading, header parsing.

Assumptions (confirmed by inspection):
  - Simulation folders live directly under data_root and match the pattern ns*.
  - Each sim has output/ containing Snapshots.txt and SPECTRA_NNN/ subdirs.
  - Each SPECTRA_NNN/ contains lya_forest_spectra_grid_480.hdf5 (primary) and
    optionally lya_forest_spectra.hdf5 (secondary, fewer skewers).
  - HDF5 datasets:
      tau/H/1/1215   shape (N_skewers, nbins)  float32 – raw optical depth
      spectra/cofm   shape (N_skewers, 3)       float64 – comoving positions (kpc/h)
      spectra/axis   shape (N_skewers,)          int32   – LOS axis (1, 2, or 3)
  - All other groups (colden, tau_obs, temperature, velocity, ...) are EMPTY.
  - Header attributes: Hz, box, hubble, nbins, npart, omegab, omegal, omegam, redshift
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Named return types
# ---------------------------------------------------------------------------

class SpectraHeader(NamedTuple):
    redshift: float
    Hz: float          # H(z) in km/s/Mpc
    box: float         # comoving box size in kpc/h
    hubble: float      # dimensionless h
    nbins: int         # pixels per skewer
    omegam: float
    omegal: float
    omegab: float
    n_skewers: int     # actual number of sightlines in file


class SimInfo(NamedTuple):
    name: str          # folder name (the unique sim identifier)
    path: Path         # absolute path to sim root
    params: Dict[str, float]   # parsed cosmological/astrophysical parameters


# ---------------------------------------------------------------------------
# Simulation folder discovery and parameter parsing
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(
    r"ns(?P<ns>[0-9.e+\-]+)"
    r"Ap(?P<Ap>[0-9.e+\-]+)"
    r"herei(?P<herei>[0-9.e+\-]+)"
    r"heref(?P<heref>[0-9.e+\-]+)"
    r"alphaq(?P<alphaq>[0-9.e+\-]+)"
    r"hub(?P<hub>[0-9.e+\-]+)"
    r"omegamh2(?P<omegamh2>[0-9.e+\-]+)"
    r"hireionz(?P<hireionz>[0-9.e+\-]+)"
    r"bhfeedback(?P<bhfeedback>[0-9.e+\-]+)"
)


def parse_sim_params(folder_name: str) -> Optional[Dict[str, float]]:
    """
    Extract cosmological/astrophysical parameters from simulation folder name.

    Example:
      ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141
        hireionz7.17bhfeedback0.056
    Returns dict with keys: ns, Ap, herei, heref, alphaq, hub, omegamh2,
                             hireionz, bhfeedback.
    Returns None if folder does not match the expected pattern.
    """
    m = _PARAM_RE.match(folder_name)
    if m is None:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}


def discover_simulations(data_root: str | Path) -> List[SimInfo]:
    """
    Walk data_root and return all parseable simulation folders sorted by name.
    Folders that do not match the ns…bhfeedback… pattern are silently skipped.
    """
    root = Path(data_root)
    sims = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        params = parse_sim_params(d.name)
        if params is None:
            continue
        sims.append(SimInfo(name=d.name, path=d, params=params))
    return sims


def filter_simulations(sims: List[SimInfo], name_fragments: List[str]) -> List[SimInfo]:
    """
    Optionally restrict to sims whose folder name contains any fragment in name_fragments.
    If name_fragments is empty, all sims are returned.
    """
    if not name_fragments:
        return sims
    return [s for s in sims if any(frag in s.name for frag in name_fragments)]


# ---------------------------------------------------------------------------
# SPECTRA file discovery
# ---------------------------------------------------------------------------

def spectra_files_for_sim(sim: SimInfo, prefer_grid: bool = True) -> Dict[int, Path]:
    """
    Return {snap_number: Path} for all available SPECTRA directories in sim.

    preference: lya_forest_spectra_grid_480.hdf5 over lya_forest_spectra.hdf5.
    If a SPECTRA_NNN dir has neither file, it is omitted (with a warning).
    """
    output_dir = sim.path / "output"
    if not output_dir.is_dir():
        return {}

    result: Dict[int, Path] = {}
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = re.match(r"SPECTRA_(\d+)$", entry.name)
        if m is None:
            continue
        snap = int(m.group(1))

        grid_file = entry / "lya_forest_spectra_grid_480.hdf5"
        rand_file = entry / "lya_forest_spectra.hdf5"

        if prefer_grid and grid_file.exists():
            result[snap] = grid_file
        elif rand_file.exists():
            result[snap] = rand_file
        elif grid_file.exists():
            result[snap] = grid_file
        else:
            # No HDF5 in this SPECTRA dir
            pass

    return result


# ---------------------------------------------------------------------------
# HDF5 reading
# ---------------------------------------------------------------------------

def read_header(path: Path) -> SpectraHeader:
    """Read Header attributes from a SPECTRA HDF5 file."""
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        tau_ds = f["tau/H/1/1215"]
        n_skewers = tau_ds.shape[0]
        return SpectraHeader(
            redshift=float(h["redshift"]),
            Hz=float(h["Hz"]),
            box=float(h["box"]),
            hubble=float(h["hubble"]),
            nbins=int(h["nbins"]),
            omegam=float(h["omegam"]),
            omegal=float(h["omegal"]),
            omegab=float(h["omegab"]),
            n_skewers=n_skewers,
        )


def read_tau_chunk(
    path: Path,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """
    Read a contiguous row slice of tau/H/1/1215 from an HDF5 file.
    Returns float32 array of shape (row_end - row_start, nbins).
    Keeps the file open only for the duration of the read.
    """
    with h5py.File(path, "r") as f:
        return f["tau/H/1/1215"][row_start:row_end, :]


def read_cofm_chunk(
    path: Path,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """
    Read a contiguous row slice of spectra/cofm.
    Returns float64 array of shape (row_end - row_start, 3) in kpc/h.
    """
    with h5py.File(path, "r") as f:
        return f["spectra/cofm"][row_start:row_end, :]


def read_axis_chunk(
    path: Path,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """
    Read skewer axes (1, 2, or 3 = x, y, z).
    Returns int32 array of shape (row_end - row_start,).
    """
    with h5py.File(path, "r") as f:
        return f["spectra/axis"][row_start:row_end]


def iter_tau_batches(
    path: Path,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Iterate over the tau array in row batches.
    Yields (row_start, row_end, tau_batch) with tau_batch shape (batch, nbins).

    n_skewers: if set, stops after this many rows (for debugging).
    """
    with h5py.File(path, "r") as f:
        total = f["tau/H/1/1215"].shape[0]
    if n_skewers is not None:
        total = min(total, n_skewers)
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        yield start, end, read_tau_chunk(path, start, end)
        start = end


# ---------------------------------------------------------------------------
# Snapshots.txt parsing
# ---------------------------------------------------------------------------

def read_snapshots_txt(sim: SimInfo) -> Dict[int, float]:
    """
    Parse output/Snapshots.txt → {snap_index: scale_factor a}.

    File format (two columns, space-separated):
      000 0.1
      001 0.111111
      ...
    """
    snap_file = sim.path / "output" / "Snapshots.txt"
    if not snap_file.exists():
        return {}
    result: Dict[int, float] = {}
    with open(snap_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    snap = int(parts[0])
                    a = float(parts[1])
                    result[snap] = a
                except ValueError:
                    continue
    return result


def snap_to_z(a: float) -> float:
    """Convert scale factor a to redshift z = 1/a - 1."""
    return 1.0 / a - 1.0


# ---------------------------------------------------------------------------
# SimulationICs.json parsing
# ---------------------------------------------------------------------------

_ICS_FIELDS = {
    # JSON key → output key (some sims may use slightly different keys)
    "npart": "npart",
    "box": "ics_box",          # kpc/h (ICs file uses same box unit)
    "omega0": "omega0",
    "omegab": "ics_omegab",
    "hubble": "ics_hubble",
    "ns": "ics_ns",
    "scalar_amp": "scalar_amp",
    "here_i": "here_i",
    "here_f": "here_f",
    "alpha_q": "alpha_q",
    "hireionz": "ics_hireionz",
    "bhfeedback": "ics_bhfeedback",
    "seed": "seed",
    "uvb": "uvb",
}


def read_sim_ics(sim: SimInfo) -> Dict[str, Any]:
    """
    Parse SimulationICs.json in the sim root folder.

    Returns a flat dict with standardised keys (see _ICS_FIELDS above).
    Returns empty dict if the file is missing or malformed (non-fatal).
    """
    ics_path = sim.path / "SimulationICs.json"
    if not ics_path.exists():
        return {}
    try:
        with open(ics_path) as fh:
            raw: Dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    result: Dict[str, Any] = {}
    for json_key, out_key in _ICS_FIELDS.items():
        # Handle case-insensitive or underscore variants
        for candidate in (json_key, json_key.replace("_", ""), json_key.lower()):
            if candidate in raw:
                result[out_key] = raw[candidate]
                break
    return result


def pixel_dv_kms(header: SpectraHeader) -> float:
    """
    Compute pixel velocity width in km/s.

    Derivation:
      v_box = H(z) * L_phys
            = H(z) [km/s/Mpc] * (box [kpc/h] / 1000 / h) [Mpc] / (1+z)
    where the (1+z) converts comoving → physical distance.
    Each pixel covers v_box / nbins km/s.

    For the observed data: H(z=5.4)=610 km/s/Mpc, box=120000 kpc/h, h=0.735
    → dv = 610 * (120000/1000/0.735) / 1556 / 6.4 ≈ 10.0 km/s.
    """
    box_mpc = header.box / 1000.0 / header.hubble  # comoving Mpc
    v_box = header.Hz * box_mpc / (1.0 + header.redshift)  # km/s
    return v_box / header.nbins
