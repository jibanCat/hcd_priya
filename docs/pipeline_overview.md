# Pipeline Overview

## Goal

Process all `fake_spectra` outputs for the lyman-alpha forest emulator
at Great Lakes. For each (simulation, redshift) pair:

1. Identify HCD absorption systems (LLS, subDLA, DLA) from tau.
2. Compute P1D(k) for the full forest and with each class masked.
3. Measure the CDDF f(NHI, X).
4. Compute the P1D effect of a parametric CDDF perturbation.

## Data flow

```
HDF5 tau array (691200 × 1556)
         │
         ├─── catalog.py ──→ AbsorberCatalog (.npz)
         │        │               │
         │        └─ Voigt fit    └─── cddf.py ──→ f(NHI, X)
         │                               │
         └─── p1d.py ──────────────────→ P1D all variants
                  │                          │
                  └─ masking.py ─────────────┘
                       │
                       └─ cddf.py (perturbation) ──→ P1D perturbed
```

## Entry points

```bash
# One simulation, one redshift
hcd run-one --sim ns0.803... --snap 017 --config config/default.yaml

# One simulation, all redshifts
hcd run-sim --sim ns0.803... --config config/default.yaml

# All simulations, all redshifts
hcd run-all --config config/default.yaml

# Benchmark mode
hcd benchmark --n-sims 2 --n-snaps 3
```

## Outputs

Each (sim, snap) pair writes to:
```
outputs/{sim_name}/snap_{NNN}/
  done              ← sentinel (skip on resume)
  catalog.npz       ← absorber catalog
  p1d.npz           ← P1D variants
  p1d_ratios.npz    ← ratio arrays
  cddf.npz          ← CDDF measurement
  p1d_perturbed.npz ← (optional) perturbed P1D
  meta.json         ← timing + metadata
```

Figures are written to `figures/`.
Markdown docs are in `docs/`.
