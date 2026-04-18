"""
hcd_analysis — HCD absorber analysis pipeline for fake_spectra emulator outputs.

High-level package structure:
  config.py       — Config dataclass, YAML loading, CLI overrides
  io.py           — HDF5 reading, file discovery
  snapshot_map.py — Snapshot-to-redshift mapping per simulation
  catalog.py      — Absorber identification from tau (LLS / subDLA / DLA)
  masking.py      — Tau / flux masking for absorber classes
  p1d.py          — P1D(k) computation and ratio products
  cddf.py         — CDDF measurement and continuous perturbation model
  pipeline.py     — Orchestration: one (sim, z), one sim, all sims
  report.py       — Figures and markdown outputs
"""

__version__ = "0.1.0"
