# Pipeline Overview

## Goal

Process all `fake_spectra` outputs for the lyman-alpha forest emulator at Great Lakes.
Two simulation sets are processed:

- **Low-force (LF):** 60 sims in `/nfs/turbo/umor-yueyingn/mfho/emu_full/`
- **HiRes (HF):**      3 sims in `/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/`

For each (simulation, redshift) pair:

1. Parse `SimulationICs.json` for sim parameters (npart, cosmology, UVB, feedback).
2. Identify HCD absorption systems (LLS, subDLA, DLA) from tau via Voigt fitting.
3. Compute P1D(k) for the full forest and with each class pixel-masked.
4. Compute P1D(k) under sightline exclusion at 10 NHI thresholds (logN>17.2…21.0).
5. Measure the CDDF f(NHI, X); stack into per-redshift-bin CDDFs per sim.
6. Compute HiRes/LF convergence ratios T(k) = P1D_hires/P1D_lf at matching sims.
7. Optionally compute P1D under a continuous CDDF perturbation f'(N)=A·f(N)·(N/Npiv)^α.

## Data flow

```
HDF5 tau (691200 × 1556)         SimulationICs.json
         │                               │
         ├─── catalog.py ──→ AbsorberCatalog (.npz) ──→ cddf.py ──→ f(NHI,X)
         │        │                                          │
         │        └─ Voigt fit                    stack_cddf_for_sim() → cddf_stacked.npz
         │
         ├─── p1d.py (pixel masking) ──→ P1D all/no_DLA/no_HCD/...
         │
         ├─── p1d.py (sightline excl) ──→ P1D_excl(k, logN_cut) [10 cuts]
         │
         └─── cddf.py (perturbation)  ──→ P1D_perturbed [optional]

HiRes: same flow → outputs/hires/{sim}/
Convergence: T(k) = P1D_hires / P1D_lf for 2–3 matched sims
```

## Entry points

```bash
# Full LF campaign (60 sims, all z)
sbatch scripts/batch_greatlakes.sh              # 21 CPUs, 8hr

# Full HiRes campaign (3 sims)
sbatch scripts/batch_hires.sh                  # 4 CPUs, 2hr

# Convergence ratios (after both above)
sbatch scripts/batch_convergence.sh

# One simulation, all redshifts
hcd run-sim --sim ns0.803... --config config/default.yaml

# HiRes sims only
hcd run-hires --config config/default.yaml

# Convergence ratios (interactive)
hcd convergence --config config/default.yaml
```

## Output file tree

```
outputs/
  config_used.yaml
  {sim_name}/
    cddf_stacked.npz        ← per-z-bin CDDF (all snaps stacked)
    snap_{NNN}/
      done                  ← sentinel (skip on resume)
      catalog.npz           ← absorber catalog
      p1d.npz               ← P1D variants (all/no_DLA/no_HCD/...)
      p1d_ratios.npz        ← ratio arrays
      p1d_excl.npz          ← sightline exclusion sweep (10 NHI cuts)
      cddf.npz              ← per-snap CDDF
      p1d_perturbed.npz     ← (optional) CDDF-perturbed P1D
      meta.json             ← timing, absorber counts, sim_ics fields
  hires/
    {sim_name}/
      convergence_ratios.npz  ← T(k) = P1D_hires / P1D_lf
      snap_{NNN}/             ← same structure as LF

figures/
  discovery_summary.png
  cddf_per_z_bin.png
  p1d_curves.png
  p1d_ratios.png
  p1d_excl_sweep_z*.png
  convergence_ratios.png
  ...
```

## k grid

Output P1D uses 50 log-spaced bins from 1.08×10⁻³ to 5.0×10⁻² s/km.
The first 35 bins match the PRIYA emulator kf grid; the remaining 15 extend
to the Nyquist frequency (dv≈10 km/s → k_Nyq = 1/(2×10) = 0.05 s/km).
