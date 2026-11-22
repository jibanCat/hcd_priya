# The research pipeline: catalogue construction

This page was part of the [README](../README.md) until 2026-11. It describes the catalogue construction of the
research pipeline that the emulator was built for. The supported interface of this release is the predictor and
the data likelihood of `hcd_analysis.likelihood` ([README](../README.md#scope-and-limitations-of-this-release)).

## HCD catalog construction

**Requires the full data / Great Lakes (not in this repo).** This step reads the
raw PRIYA τ skewers, which are large and not distributed here; it is a cluster
job. The structure of the products it describes can still be read for reference.

The HCD catalog is the foundation the rest of the pipeline rests on: it records,
for every PRIYA simulation and redshift, which sightlines intersect a dense
absorber and how much neutral hydrogen that absorber carries. From the catalog we
derive the column-density distribution function and the per-class P1D, and those
products are what the emulator is ultimately trained on.

The catalog is built directly from the optical-depth (τ) skewers produced by
`fake_spectra`. For each sightline, `hcd_analysis/catalog.py` locates connected
velocity regions where τ exceeds a detection threshold, merges regions separated
by less than a velocity gap, and estimates the column density N_HI of each system.
In production the column density is obtained by the sum-rule inversion (the fast
estimator, `absorber.fast_mode: true` in the config); a full Voigt fit with a
±2000 km/s wing window is available as an alternative. Each system is then placed
into one of three classes by its N_HI, with everything below the LLS threshold
discarded as forest:

- LLS:    10^17.2 ≤ N_HI < 10^19.0 cm⁻²
- subDLA: 10^19.0 ≤ N_HI < 10^20.3 cm⁻²
- DLA:    N_HI ≥ 10^20.3 cm⁻²

These boundaries are defined in `hcd_analysis/catalog.py` (`classify_system`), and
the detection and measurement parameters (τ threshold, merge gap, the N_HI floor)
live under `absorber:` in `config/default.yaml`.

Build the catalog as part of the per-simulation pipeline (`hcd_env`):

```bash
hcd run-sim --sim ns0.803Ap2.2e-09 --config config/default.yaml   # one sim, all snaps
hcd run-all --n-workers 36 --config config/default.yaml             # full 60-sim LF campaign
```

Each output root holds the results of one set of settings: the first run writes `run_settings.json` there, and a
later run with different settings, or into a root that holds files but no record, is refused before anything is
written (`hcd_analysis/run_settings.py`).

On Great Lakes the full campaign runs as a SLURM array, one task per simulation:

```bash
sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array
```

The input is the τ skewer HDF5 for each (sim, snap), read alongside the
`SimulationICs.json` that records the simulation's cosmology. The output is one
`catalog.npz` per (sim, snap) under `outputs/{sim}/snap_{NNN}/`, holding, for
every detected absorber, its sightline index, pixel range, N_HI, Doppler
parameter and class. The same run also writes the per-class P1D (`p1d.npz`) and
the per-snapshot CDDF (`cddf.npz`), which `cddf.py` stacks per redshift bin into
`cddf_stacked.npz`. Those per-class P1D and CDDF products, resampled across the
mean-flux ladder, are what `scripts/build_emulator_cache_tau0.py` assembles into
the τ₀ training cache.

The P1D files of this catalogue pipeline (`p1d.npz`) use the cyclic wavenumber k_n = n / (N dv)
(`hcd_analysis/p1d.py`), for N pixels of width dv. It is not the angular wavenumber of the predictor, which is
larger by the factor 2π for the same mode.
