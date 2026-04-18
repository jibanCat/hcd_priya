"""
Snapshot-to-redshift mapping.

For each simulation, builds a table of all available (snap, a, z, hdf5_path)
tuples and filters to the requested redshift range.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

from .io import (
    SimInfo,
    SpectraHeader,
    discover_simulations,
    filter_simulations,
    pixel_dv_kms,
    read_header,
    read_snapshots_txt,
    snap_to_z,
    spectra_files_for_sim,
)


@dataclasses.dataclass
class SnapEntry:
    snap: int
    a: float
    z: float
    path: Path
    header: Optional[SpectraHeader] = None  # loaded lazily

    @property
    def dv_kms(self) -> float:
        if self.header is None:
            raise RuntimeError("Header not loaded; call load_header() first.")
        return pixel_dv_kms(self.header)

    def load_header(self) -> "SnapEntry":
        self.header = read_header(self.path)
        return self


@dataclasses.dataclass
class SimSnapshot:
    """All available snapshots within the z range for one simulation."""
    sim: SimInfo
    entries: List[SnapEntry]  # sorted by snap number

    def filter_z(self, z_min: float, z_max: float) -> "SimSnapshot":
        filtered = [e for e in self.entries if z_min <= e.z <= z_max]
        return SimSnapshot(sim=self.sim, entries=filtered)

    def snaps_in_range(self, z_min: float, z_max: float) -> List[SnapEntry]:
        return [e for e in self.entries if z_min <= e.z <= z_max]


def build_snapshot_map(
    data_root: str,
    z_min: float = 2.0,
    z_max: float = 6.0,
    sim_filter: Optional[List[str]] = None,
    prefer_grid: bool = True,
) -> List[SimSnapshot]:
    """
    Discover all simulations and return a list of SimSnapshot objects,
    each containing only the snapshots in [z_min, z_max].

    Simulations with zero valid snapshots in range are excluded.
    Mismatches between Snapshots.txt and actual SPECTRA folders are handled
    gracefully: only (snap, file) pairs that both exist are included.
    """
    sims = discover_simulations(data_root)
    if sim_filter:
        sims = filter_simulations(sims, sim_filter)

    result: List[SimSnapshot] = []
    for sim in sims:
        snap_to_a = read_snapshots_txt(sim)
        file_map = spectra_files_for_sim(sim, prefer_grid=prefer_grid)

        entries: List[SnapEntry] = []
        for snap, path in sorted(file_map.items()):
            if snap not in snap_to_a:
                # SPECTRA folder exists but not in Snapshots.txt: use Header redshift
                try:
                    hdr = read_header(path)
                    a = 1.0 / (1.0 + hdr.redshift)
                except Exception:
                    continue
            else:
                a = snap_to_a[snap]
            z = snap_to_z(a)
            if not (z_min <= z <= z_max):
                continue
            entries.append(SnapEntry(snap=snap, a=a, z=round(z, 6), path=path))

        if entries:
            result.append(SimSnapshot(sim=sim, entries=sorted(entries, key=lambda e: e.snap)))

    return result


def print_coverage_table(snap_map: List[SimSnapshot]) -> None:
    """Print a human-readable coverage table (sim x snap)."""
    all_snaps = sorted({e.snap for ss in snap_map for e in ss.entries})
    header_row = f"{'Simulation':<30} " + " ".join(f"{s:>5}" for s in all_snaps)
    print(header_row)
    print("-" * len(header_row))
    for ss in snap_map:
        available = {e.snap for e in ss.entries}
        row = f"{ss.sim.name[:30]:<30} " + " ".join(
            "  [x]" if s in available else "   . " for s in all_snaps
        )
        print(row)
