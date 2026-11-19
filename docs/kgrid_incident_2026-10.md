# The 2026 k-grid representation regression (public summary)

This note is the public record of a coordinate defect in the emulator found in late September 2026, and of the
fix (the "gates A to E" of the 2026-10 emulator-debug campaign). Code comments that cite the incident point here.

## What was wrong

Each PRIYA simulation box is 120 Mpc/h comoving. Its one-dimensional flux power spectrum is computed on the box
modes n = 1..172, whose comoving wavenumbers k_com = 2 pi n / 120 h/Mpc are the same for every simulation and
every snapshot. The angular wavenumber of a mode, in s/km, is not: it is

    k_skm = k_com (1 + z) / (100 E(z; theta)),    E(z) = sqrt(Om (1+z)^3 + 1 - Om),    Om = omegamh2 / hub^2,

so it moves with redshift and with the cosmology (hub, omegamh2) of each row.

The cache stores each row's own angular grid (`kfkms`). Older code collapsed it to a single row: the training
script saved row 0's grid in the checkpoint metadata, and downstream code used that one grid for every
prediction request, as if one velocity grid described all of them. Emulator outputs were therefore placed at
the wrong physical k whenever the request's (z, theta) differed from that row. The power values on each mode
were right; their k labels were not. Code that put mode-axis quantities at data k inherited the defect: the
interpolation of predictions onto survey k bins, the multi-fidelity (LF to HF) correction tables and the
stored emulator error vectors. Network training itself did not depend on the collapsed grid (the k-range mask
used each row's own grid, and the edge-emphasis weight depends only on the mode index). The pre-2026-10 forward
model is retired in the code.

## The fix

* `hcd_analysis/emulator/schema.py`: every cache and checkpoint key has a class (`GLOBAL_STATIC`, `PER_ROW`,
  ...). `kfkms` is `PER_ROW`; the comoving modes are exposed as `k_com_hmpc` (`GLOBAL_STATIC`), validated across
  all rows instead of being read from row 0. Checkpoints use schema 2.0, record `k_com_hmpc` and the
  coordinate convention, and never store a velocity grid. `train.load_checkpoint` refuses schema-1 checkpoints.
* `hcd_analysis/emulator/kcoord.py`: the only place that maps modes to physical k, per request (z, theta).
  `kcoord.bind` places mode-axis quantities at the data k of one (survey leg, z).
* `hcd_analysis/emulator/predict.py`: `EmulatorPrediction` returns every array together with the `k_skm` it is
  defined on.
* `hcd_analysis/emulator/mf_modes.py`: the LF to HF correction is measured mode by mode, at equal physical k.
* `hcd_analysis/emulator/forward.py`: the survey-leg forward model on the canonical coordinate.
* The production LF members (`checkpoints/production_ensemble_manifest.json`) were retrained on the repaired
  cache below, as schema-2.0 checkpoints; the manifest pins them by SHA-256.

The same campaign found that, for one simulation, the LF cache's z = 2.8 rows took their velocity scale (and
so their `kfkms`) and their absorber catalogue from a neighbouring snapshot, and its z = 3.0 rows were missing.
The repair (`scripts/splice_tau0_cache.py`) wrote a new cache in which only the affected rows differ. The historical cache is kept unchanged, and `tests/test_cache_repair_s4.py`
checks the difference against the footprint that was specified before the rebuild.

## Tests

* `tests/regression/test_incident_2026_kgrid.py` reproduces the old row-0 collapse in isolation and checks that
  the new schema refuses a collapsed cache.
* `tests/regression/legacy_forward_pre2026_10.py` keeps the pre-2026-10 forward as a test-only fixture, which
  `tests/test_forward_v2.py` compares against. Production code does not import it.
* `tests/test_schema.py`, `tests/test_kcoord.py`, `tests/test_kcoord_binding.py`,
  `tests/test_predict_physical_grid.py`, `tests/test_gateA_blocking.py`, `tests/test_mf_modes.py`,
  `tests/test_forward_v2.py` and `tests/test_gateE_invariants.py` cover the corrected coordinate.
* Tests that need the real caches skip when the caches are absent. Set `HCD_GATE_RUN=1` to make them fail
  instead (`tests/gate_helpers.py`).

## Internal references in the code

Comments cite internal design and review records by name (for example `GATE_E_SPEC`, `S4_REPAIR_SPEC`, gate
review records, the JAX traps log) and provenance-ledger identifiers of the form `PU-NNNN`. The maintainers
keep these records in a private notes repository. This note is their public summary.
