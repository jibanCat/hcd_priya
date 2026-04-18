# hcd_analysis — HCDs in PRIYA and effects on Lyman-alpha P1D

Production-quality Python pipeline for analyzing High Column Density (HCD)
absorbers in `fake_spectra` outputs from the Lyman-alpha forest emulator
at Great Lakes (`/nfs/turbo/umor-yueyingn/mfho/emu_full/`).

## What it does

For each `(simulation, redshift)` pair:
1. Identifies LLS, subDLA, and DLA absorption systems from saved tau via Voigt fitting
2. Computes P1D(k) for the full forest and with each absorber class masked
3. Measures the Column Density Distribution Function f(NHI, X)
4. Applies a parametric CDDF perturbation model and propagates it to P1D

## Data layout (confirmed by inspection)

- **60 simulations**, each a unique cosmological/astrophysical parameter point
- **~18 snapshots** per simulation covering z = 2.0–5.4
- **691,200 sightlines** per snapshot (480×480×3 grid), 1556 pixels each
- Pixel velocity width: **~10 km/s**
- Key HDF5 dataset: `tau/H/1/1215` — all other groups (colden, velocity, temperature) are **empty**

Full details: [`docs/data_layout.md`](docs/data_layout.md)

## Quick start

```bash
# Install (Great Lakes)
bash scripts/install_greatlakes.sh

# Check coverage
hcd coverage

# One simulation, one redshift (debug)
hcd run-one --sim ns0.803 --snap 17 --debug --verbose

# One simulation, all redshifts
hcd run-sim --sim ns0.803Ap2.2e-09 --config config/default.yaml

# All simulations, all redshifts (parallel)
hcd run-all --n-workers 36

# Benchmark
hcd benchmark --n-sims 2 --n-snaps 3

# Generate figures + docs
hcd report
```

## SLURM batch (Great Lakes)

```bash
# Full campaign on one 36-core node
sbatch scripts/batch_greatlakes.sh run-all

# Array job: one SLURM task per simulation (60 tasks)
sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array

# Single simulation
sbatch scripts/batch_greatlakes.sh run-sim --sim ns0.803Ap2.2e-09
```

## Config overrides

```bash
hcd run-all \
  --config config/default.yaml \
  --set n_workers=16 \
  --set cddf.A=1.2 \
  --set cddf.alpha=0.5 \
  --set absorber.tau_threshold=2.0
```

## P1D definition

```
F(v) = exp(-tau(v))
delta_F(v) = F(v) / <F> - 1
P1D(k) = dv / N × |DFT(delta_F)[n]|²    k in s/km,  P1D in km/s
```

See [`docs/p1d_definition.md`](docs/p1d_definition.md).

## Absorber classification

| Class  | log10(NHI) range |
|--------|-----------------|
| LLS    | 17.2 – 19.0 |
| subDLA | 19.0 – 20.3 |
| DLA    | ≥ 20.3 |

## fake_spectra integration

| Component | Source |
|-----------|--------|
| Lyman-alpha line parameters (f, λ, Γ) | fake_spectra constants |
| Voigt profile via scipy wofz | same approach as fake_spectra |
| Absorber system finder | newly implemented (tau-based) |
| P1D accumulator | newly implemented |
| CDDF + perturbation model | newly implemented |

See [`docs/fake_spectra_integration.md`](docs/fake_spectra_integration.md).

## Outputs

```
outputs/{sim_name}/snap_{NNN}/
  done              ← sentinel (idempotent resume)
  catalog.npz       ← absorber catalog
  p1d.npz           ← P1D variants: all, no_DLA, no_subDLA, no_LLS, no_HCD
  p1d_ratios.npz    ← ratio arrays
  cddf.npz          ← f(NHI, X)
  p1d_perturbed.npz ← perturbed P1D
  meta.json         ← timing + metadata
figures/            ← auto-generated plots
docs/               ← auto-generated markdown
```

## Package structure

```
hcd_analysis/
  config.py        io.py          snapshot_map.py
  voigt_utils.py   catalog.py     masking.py
  p1d.py           cddf.py        pipeline.py     report.py
cli/run.py
config/default.yaml
scripts/batch_greatlakes.sh  install_greatlakes.sh
```

## Explicit assumptions

See [`docs/assumptions.md`](docs/assumptions.md). Key ones:

1. `colden` is **empty** — NHI derived from tau only
2. Only `tau/H/1/1215` (HI Lyman-alpha) is used
3. tau is raw (not rescaled to observed mean flux)
4. Systems = connected tau > 1 regions, merged at < 100 km/s gap
5. k in s/km; P1D in km/s
