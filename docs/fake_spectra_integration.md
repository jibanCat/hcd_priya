# fake_spectra Integration

## Package

- **Repository:** https://github.com/sbird/fake_spectra
- **Author:** Simeon Bird
- **Purpose:** Compute Lyman-alpha forest spectra from cosmological simulations.

## What this repo reuses from fake_spectra

### Directly reused: physics constants and line parameters

`hcd_analysis/voigt_utils.py` uses the same Lyman-alpha line parameters
as fake_spectra:
  - Lyman-alpha wavelength: lambda = 1215.67 Å
  - Oscillator strength: f_lu = 0.4164
  - Einstein A coefficient: Gamma = 6.265e8 s^-1

These values are embedded as constants (the same values fake_spectra uses)
to avoid a hard dependency on fake_spectra's internal module structure,
which has changed across versions.

### Reused: Voigt profile computation approach

`voigt_utils.voigt_profile_phi()` uses `scipy.special.wofz` (Faddeeva function),
which is the same mathematical approach as fake_spectra's internal Voigt
computation. The Voigt-Hjerting function H(a,u) = Re[wofz(u + i*a)] is
standard in all modern Lyman-alpha codes.

### Wrapped: tau_voigt() model function

`voigt_utils.tau_voigt(v, NHI, b, v0)` computes the theoretical optical depth
for a single Voigt component. This is the model function used in absorber
fitting. It replicates what fake_spectra computes internally when generating
synthetic spectra from particle data.

### NOT reused (why)

| fake_spectra component | Status in this repo | Reason |
|------------------------|---------------------|--------|
| `Spectra` class | Not used | Designed for particle data; we have pre-computed tau |
| `Spectra.get_tau()` | Not applicable | tau is already saved |
| `Spectra.get_col_density()` | Not applicable | Requires particle data (GADGET snapshots) |
| `find_absorbers()` | Re-implemented | Our implementation operates on tau directly |
| `rate_interpolate` | Not used | Designed for post-processing Spectra objects |

## What is newly implemented in this repo

| Component | Location | Description |
|-----------|----------|-------------|
| Tau-based system finder | `catalog.find_systems_in_skewer()` | Connected threshold regions in velocity space |
| Voigt fitting from tau | `voigt_utils.fit_nhi_from_tau()` | scipy L-BFGS-B fit in log space |
| AbsorberCatalog | `catalog.AbsorberCatalog` | Serialisable container for absorber sets |
| P1D accumulator | `p1d.P1DAccumulator` | Streaming, memory-efficient P1D computation |
| CDDF measurement | `cddf.measure_cddf()` | From absorption path and simulation box |
| CDDF perturbation | `cddf.CDDFPerturbation` | Continuous (A, alpha, N_pivot) model |
| Pipeline | `pipeline.run_one_snap()` | End-to-end orchestration with resume/restart |
| Batch scripts | `scripts/` | Great Lakes SLURM job templates |

## Installation

See `docs/` section on environment setup or README for installation instructions.
If fake_spectra is not installed, the pipeline falls back to internal Voigt
utilities (the same physics, just without fake_spectra as a dependency).
