# PRIYA param_limits used for unit-cube input normalization (2026-06-01)

Fix for the Phase-2b preprocessing blocker (raw params span ~9 orders of
magnitude; encoder can't learn). We normalize the 9 sim params + z to [0,1].

## Source (authoritative, NOT invented)

`/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json`

This is the serialized `Emulator.param_limits` / `param_names` for the exact
production grid whose `params` array our cache reproduces (`_PRIYA_FILE =
.../kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5`, same dir).
Preferred over the `coarse_grid.py` hardcoded defaults (L93-124) because the
saved JSON is the *actual* box used for this grid and it widens those defaults
(ns_hi 1.05 vs 0.995, herei_hi 4.5 vs 4.1, heref_lo 2.2 vs 2.6, alphaq_hi 3.0
vs 2.5). `emulator_params.json` and `T0emulator_params.json` agree exactly.

## Alignment

PRIYA `param_names` (from the JSON): `{ns:0, Ap:1, herei:2, heref:3, alphaq:4,
hub:5, omegamh2:6, hireionz:7, bhfeedback:8}`. This is IDENTICAL (by name and
position) to our cache `param_names` / `build_emulator_cache.PARAM_ORDER`:
`(ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback)`. The
by-name alignment is therefore the identity — no reordering needed. (PRIYA's
raw `params` HDF5 array prepends mean-flux `alpha` as col 0; that column is NOT
a design param here and is excluded — see
`tests/test_emulator_cache_tau0.py::test_read_priya_params_matches_priya_array`
which slices `priya_params[sim_idx, 1:]`.)

## The 9 (lo, hi) used — in OUR cache param order

| # | param      | lo      | hi      | JSON param_limits index |
|---|------------|---------|---------|-------------------------|
| 0 | ns         | 0.8     | 1.05    | [0] |
| 1 | Ap         | 1.2e-9  | 2.6e-9  | [1] |
| 2 | herei      | 3.5     | 4.5     | [2] |
| 3 | heref      | 2.2     | 3.2     | [3] |
| 4 | alphaq     | 1.3     | 3.0     | [4] |
| 5 | hub        | 0.65    | 0.75    | [5] |
| 6 | omegamh2   | 0.14    | 0.146   | [6] |
| 7 | hireionz   | 6.5     | 8.0     | [7] |
| 8 | bhfeedback | 0.03    | 0.07    | [8] |

z: `Z_LIMITS = (2.0, 5.4)` from `coarse_grid.py` L153-154 (`max_z=5.4,
min_z=2.0`, the full PRIYA zout grid range). (Note: this grid's JSON records
`max_z=4.6, min_z=2.2` for its own zout span; the spec mandates the wider
PRIYA grid range (2.0, 5.4), used here.)

## Verification

- All 60 PRIYA design `sample_params` map into [0,1] under these limits
  (unit min 0.010, max 0.990).
- Real shard `observables_tau0_lf.shard000.h5` (240 rows): `in_domain`
  fraction = **1.0**, params_unit min/max = 0.013 / 0.850. z_unit in [0.353, 1.0].
