# fake_spectra Integration

## Package

- **Repository:** https://github.com/sbird/fake_spectra
- **Author:** Simeon Bird
- **Purpose:** Compute Lyman-alpha forest spectra from cosmological simulations.

## Status: reference codebase, not a runtime dependency

Production code paths in `hcd_priya` **do not import `fake_spectra` at runtime**. The pre-saved `tau/H/1/1215` HDF5 grids that this pipeline consumes were *originally generated* with `fake_spectra`, but everything downstream of that — system finding, NHI recovery, P1D, CDDF, masking, templates — is reimplemented locally.

The only `import fake_spectra` site in the package is at `hcd_analysis/voigt_utils.py:72`, inside a `try/except ImportError` block. It is used purely as a sanity hook: if `fake_spectra` happens to be installed, the module captures `voigt_profile` for cross-validation; if not, every code path proceeds with our internal Voigt implementation. No production figure or downstream artefact depends on this soft import being satisfied.

## What we ported (with cross-validation)

| Component | Local source | Cross-check |
|-----------|--------------|-------------|
| Lyman-α line constants (λ = 1215.67 Å, f = 0.4164, Γ = 6.265 × 10⁸ s⁻¹) | `hcd_analysis/voigt_utils.py` (compile-time constants) | Bahcall–Peebles 1969 / Wolfe-Gawiser-Prochaska 2005 |
| Voigt-Hjerting `H(a, u) = Re[wofz(u + i·a)]` | `voigt_utils.voigt_profile_phi` (`scipy.special.wofz`) | Same approach as `fake_spectra` — standard Faddeeva form |
| Velocity-space integrated cross section `σ_int = π · e² · f · λ / (m_e · c · 1e5)` (cm² · km/s) | `voigt_utils._SIGMA_PREFACTOR` ≈ 1.3435 × 10⁻¹² | `tests/test_tau_sum_rule.py`: ratio = 1.0000 to FP after bug #1 fix |
| Forward-model `tau_voigt(v, NHI, b, v0)` | `voigt_utils.tau_voigt` | `tests/test_nhi_recovery.py` cross-normalisation |
| Absorption distance `dX = (1+z)² · L_com · H_0/c` | `cddf.absorption_path_per_sightline` | `tests/test_absorption_path.py` (6 tests, includes inline port of `fake_spectra.unitsystem.absorption_distance`) |

## What we did **not** port (and why)

| `fake_spectra` component | Status | Reason |
|--------------------------|--------|--------|
| `Spectra` class | Not used | We start from saved τ, not particle data |
| `Spectra.get_tau()` | Not applicable | τ already stored on disk |
| `Spectra.get_col_density()` | Not applicable | Requires GADGET particle snapshots |
| `find_absorbers()` | Reimplemented | Local `catalog.find_systems_in_skewer` operates on τ directly with our chosen thresholds |
| `rate_interpolate` | Not used | Designed for post-processing `Spectra` objects |

## Newly implemented in this repo

| Component | Location |
|-----------|----------|
| τ-based system finder | `catalog.find_systems_in_skewer` |
| NHI from τ sum rule (production, fast_mode) | `catalog.nhi_from_tau_fast` (uses `_SIGMA_PREFACTOR`) |
| Voigt fit fallback (NHI from τ profile) | `voigt_utils.fit_nhi_from_tau` (scipy L-BFGS-B in log space) |
| `AbsorberCatalog` container | `catalog.AbsorberCatalog` |
| Streaming `P1DAccumulator` | `p1d.P1DAccumulator` |
| Per-class P1D HDF5 output | `p1d.compute_p1d_per_class` + `save_p1d_per_class_hdf5` |
| CDDF measurement | `cddf.measure_cddf`, `cddf.stack_cddf_for_sim` |
| CDDF perturbation model | `cddf.CDDFPerturbation` |
| Rogers+2018 four-parameter HCD template | `hcd_analysis/hcd_template.py` |
| End-to-end pipeline | `pipeline.run_one_snap`, `pipeline.run_all` |
| Great Lakes SLURM launchers | `scripts/batch_*.sh`, `scripts/install_greatlakes.sh` |

## Installation

`fake_spectra` is **optional**. The pipeline imports it inside a `try/except ImportError` and silently falls back to internal utilities if it is not available. `scripts/install_greatlakes.sh` does install it as a convenience for users who want the soft cross-check, but no production target requires it.
