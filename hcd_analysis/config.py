"""
Configuration dataclass with YAML loading and CLI override support.

All physical thresholds, paths, and tuning knobs live here.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclasses.dataclass
class CDDFConfig:
    """Parameters for the continuous CDDF perturbation model f'(N) = A * f(N) * (N/N_pivot)^alpha."""
    A: float = 1.0          # amplitude multiplier (1.0 = no change)
    alpha: float = 0.0      # power-law tilt (0.0 = no tilt)
    N_pivot: float = 1.0e20 # pivot column density (cm^-2)


@dataclasses.dataclass
class AbsorberConfig:
    """Thresholds and fitting parameters for absorber identification."""
    # Tau threshold to define the boundary of an absorption system
    tau_threshold: float = 1.0
    # Use the sum-rule (tau-integral) NHI estimator instead of a Voigt fit.
    # See docs/fast_mode_physics.md. Recommended default = True.
    fast_mode: bool = True
    # Maximum velocity gap (km/s) to merge adjacent components into one system
    merge_dv_kms: float = 100.0
    # Minimum pixels for a valid system (avoids noise spikes)
    min_pixels: int = 2
    # Voigt fit: maximum iterations per system
    voigt_max_iter: int = 200
    # b parameter initial guess (km/s) for Voigt fit
    b_init_kms: float = 30.0
    # b parameter bounds for Voigt fit (km/s)
    b_bounds: tuple = (1.0, 300.0)
    # Cap tau at this value before fitting to avoid inf in optimization
    tau_fit_cap: float = 1.0e6
    # Minimum log10(NHI) to store in catalog. Systems below this are discarded.
    # Default 17.0 = just below LLS threshold, so only HCD systems are kept.
    min_log_nhi: float = 17.2  # = LOG_NHI_LLS_MIN
    # For P1D masking: how to handle masked pixels
    # Options: "zero_tau"   → set tau=0 in masked pixels
    #          "mean_flux"  → set F=mean(F) in masked pixels then recompute tau
    #          "contiguous" → linear interpolation over masked region
    mask_fill_strategy: str = "mean_flux"


@dataclasses.dataclass
class P1DConfig:
    """P1D computation parameters."""
    # k bins for output (s/km). None → use native FFT bins.
    k_min: float = 1.0e-3   # s/km
    k_max: float = 5.0e-2   # s/km  (Nyquist for dv≈10 km/s)
    n_k_bins: int = 35
    log_k_bins: bool = True
    # Number of skewers to use (None → all; for debugging use e.g. 10000)
    n_skewers: Optional[int] = None
    # Axis selection: 1, 2, 3 or None (all axes)
    axes: Optional[List[int]] = None
    # NHI thresholds for sightline-exclusion sweep (log10 NHI, cm^-2).
    # Sightlines with ANY absorber above this cut are excluded entirely.
    nhi_excl_thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.3, 20.5, 21.0]
    )


@dataclasses.dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Data root (low-force simulations, 60 sims)
    data_root: str = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
    # HiRes data root (3 matching sims, 2× npart)
    hires_data_root: str = "/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires"
    # Output root
    output_root: str = "./outputs"
    # Redshift range filter (inclusive)
    z_min: float = 2.0
    z_max: float = 6.0
    # Subset of simulations (by folder name fragment) – empty means all
    sim_filter: List[str] = dataclasses.field(default_factory=list)
    # Absorber classes to compute
    absorber_classes: List[str] = dataclasses.field(
        default_factory=lambda: ["LLS", "subDLA", "DLA"]
    )
    # Sub-configs
    absorber: AbsorberConfig = dataclasses.field(default_factory=AbsorberConfig)
    p1d: P1DConfig = dataclasses.field(default_factory=P1DConfig)
    cddf: CDDFConfig = dataclasses.field(default_factory=CDDFConfig)
    # Parallelism
    n_workers: int = 4           # parallel workers (sims or redshifts)
    n_workers_skewer: int = 1    # intra-file parallelism (skewer batches)
    skewer_batch_size: int = 4096  # number of skewers per batch (memory control)
    # Prefer grid_480 file over non-grid file
    prefer_grid: bool = True
    # Benchmark / debug
    benchmark: bool = False
    debug: bool = False
    debug_n_sims: int = 2
    debug_n_snaps: int = 3
    # Resume: skip (sim, snap) pairs that already have output
    resume: bool = True


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _nested_set(d: dict, key_path: str, value: Any) -> None:
    """Set d[a][b][c] = value from key_path='a.b.c'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _dataclass_from_dict(cls, d: dict):
    """Recursively build a dataclass from a dict, matching nested dataclass fields."""
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        # Resolve type annotation string (Python 3.9 compat)
        ftype = f.type if not isinstance(f.type, str) else eval(f.type)
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _dataclass_from_dict(ftype, val)
        elif ftype is float and not isinstance(val, float):
            # YAML may parse scientific notation without signed exponent (e.g. "1.0e6")
            # as a string. Coerce explicitly.
            kwargs[f.name] = float(val)
        elif ftype is int and not isinstance(val, int):
            kwargs[f.name] = int(val)
        else:
            kwargs[f.name] = val
    return dataclasses.replace(cls(), **kwargs)


def load_config(yaml_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """
    Load config from YAML file + CLI overrides.

    Args:
        yaml_path: Path to YAML config file. If None, uses built-in defaults.
        overrides:  Dict of dot-notation key paths → values, e.g.
                    {"p1d.k_min": 5e-4, "cddf.A": 1.2}

    Returns:
        PipelineConfig dataclass.
    """
    # Start from defaults serialized to dict
    base_dict = dataclasses.asdict(PipelineConfig())

    if yaml_path is not None:
        with open(yaml_path) as fh:
            user_dict = yaml.safe_load(fh) or {}
        base_dict = _deep_merge(base_dict, user_dict)

    if overrides:
        for key_path, value in overrides.items():
            _nested_set(base_dict, key_path, value)

    return _dataclass_from_dict(PipelineConfig, base_dict)


def save_config(cfg: PipelineConfig, path: str) -> None:
    """Dump config to YAML for reproducibility."""
    with open(path, "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=True)
