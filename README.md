# hcd_analysis — HCDs in PRIYA and effects on Lyman-α P1D

Production pipeline for High Column Density (HCD) absorbers in the
**PRIYA** Lyman-α emulator simulation suite at Great Lakes
(`/nfs/turbo/umor-yueyingn/mfho/emu_full/` and
`/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/`).

## What it does

For each `(simulation, snapshot)`:

1. Identifies LLS / subDLA / DLA absorption systems from the saved τ grids,
   recovering N_HI via the τ sum rule (`fast_mode = true`).
2. Computes P1D(k) for the full forest and with the PRIYA DLA mask
   applied (`ALL_VARIANTS = ["all", "no_DLA_priya"]`).
3. Computes per-sightline-class P1D contributions
   (`P_clean`, `P_LLS_only`, `P_subDLA_only`, `P_DLA_only`) for the
   Rogers+2018 four-parameter HCD-template fit.
4. Measures the column density distribution function `f(N_HI, X)` per snap
   and stacks across z bins.
5. Fits the Rogers four-parameter α per `(sim, snap)` and emits a flat
   per-sim summary HDF5 for the emulator.

## Data layout

- 60 LF sims + 4 HiRes sims; ~18 snapshots per sim covering z = 2.0–5.4.
- 691,200 sightlines per snapshot (480 × 480 × 3 grid), 1556 pixels each,
  pixel velocity width ≈ 10 km/s.
- Authoritative HDF5 dataset: `tau/H/1/1215`. Other groups (`colden`,
  `velocity`, `temperature`) are empty.

Full details: [`docs/data_layout.md`](docs/data_layout.md).

## Quick start

```bash
# Install (Great Lakes)
bash scripts/install_greatlakes.sh

# One snap (debug)
hcd run-one --sim ns0.803Ap2.2e-09 --snap 17 --debug --verbose

# One sim, all snaps
hcd run-sim --sim ns0.803Ap2.2e-09 --config config/default.yaml

# Full campaign (parallel)
hcd run-all --n-workers 36

# Benchmark
hcd benchmark --n-sims 2 --n-snaps 3
```

## SLURM batch (Great Lakes)

```bash
# All sims on one 36-core node
sbatch scripts/batch_greatlakes.sh run-all

# Array job: one task per sim
sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array

# Single sim
sbatch scripts/batch_greatlakes.sh run-sim --sim ns0.803Ap2.2e-09
```

## Config overrides

```bash
hcd run-all \
  --config config/default.yaml \
  --set n_workers=16 \
  --set absorber.fast_mode=true \
  --set absorber.tau_threshold=100
```

## P1D definition

```
F(v) = exp(-τ(v))
δ_F(v) = F(v)/⟨F⟩ − 1
P1D(k) = (dv / N) · |DFT(δ_F)[n]|²       k in s/km (cyclic),  P1D in km/s
```

PRIYA emulator x-axis convention is `k_angular = 2π · k_cyclic`
(rad·s/km). Production analysis figures use `k_angular ∈ [0.0009, 0.20]
rad·s/km`. See [`docs/p1d_definition.md`](docs/p1d_definition.md) and
[`docs/analysis.md`](docs/analysis.md) §1 for the convention discussion.

## Absorber classification

| Class  | log10(N_HI) range |
|--------|-------------------|
| LLS    | 17.2 – 19.0       |
| subDLA | 19.0 – 20.3       |
| DLA    | ≥ 20.3            |

`min_log_nhi = 17.2` is applied at catalog write time.

## fake_spectra integration

`fake_spectra` is a **reference codebase**, not a runtime dependency. The
only `import fake_spectra` site is a soft optional in
`hcd_analysis/voigt_utils.py:72`; everything works without it. Line
constants and the `(1+z)² · L_com · H_0/c` absorption-distance formula
are ported with cross-validation tests.

See [`docs/fake_spectra_integration.md`](docs/fake_spectra_integration.md).

## Outputs

Per snap (`<output_root>/<sim>/snap_NNN/`):

| File | Description |
|------|-------------|
| `done` | sentinel (idempotent resume) |
| `meta.json` | timing + cosmology + sim metadata |
| `catalog.npz` | absorber catalog (NHI, b, skewer_idx, …); `min_log_nhi=17.2` |
| `p1d.npz` | P1D variants — `all`, `no_DLA_priya` only (production) |
| `p1d_per_class.h5` | HDF5: `P_clean`, `P_LLS_only`, `P_subDLA_only`, `P_DLA_only`, plus mean F per class and metadata attrs (`fast_mode`, `tau_threshold`, `min_log_nhi`, `k_convention='cyclic'`) |
| `cddf.npz` | per-snap `f(N_HI, X)` |
| `cddf_corrected.npz` | post-bug-#7 corrected sibling (2026-04-25 patch) |

Per sim (`<output_root>/<sim>/`):

| File | Description |
|------|-------------|
| `cddf_stacked.npz` | per-z-bin stacked CDDF |
| `cddf_stacked_corrected.npz` | post-bug-#7 corrected sibling |
| `convergence_ratios.npz` | LF/HR T(k) (HR sims only; bug-fixed z-matching, `tests/test_convergence_z_match.py`) |

Suite-level (`<output_root>/_hcd_analysis_data/`):

| File | Description |
|------|-------------|
| `hcd_summary_lf.h5` / `hcd_summary_hr.h5` | flat per-`(sim, z)` HCD scalar + template summary (1076 LF + 70 HR records) |
| `rogers_alpha_summary.h5` | flat Rogers α fit per `(sim, z)` (LF + HR) |

## Tests

```bash
for t in tests/test_*.py; do python3 "$t"; done
```

The suite locks the post-audit science claims: τ sum rule (`tests/test_tau_sum_rule.py`),
NHI recovery (`tests/test_nhi_recovery.py`), Voigt profile normalisation,
absorption-distance formula (`tests/test_absorption_path.py` — six independent
checks), Rogers HCD-template (`tests/test_hcd_template.py`),
LF/HR z-matching in the convergence pipeline
(`tests/test_convergence_z_match.py`), the on-scratch dX patch
(`tests/test_cddf_dx_patch.py`), and an end-to-end pipeline regression
(`tests/sanity_run_one_snap.py`).

## Status (2026-04-25)

After bug #7 was fixed (`hcd_analysis/cddf.py` absorption-distance
formula), PRIYA dN/dX(DLA) now sits modestly *below* PW09 / N12 / Ho21 by
factor 1.5–2 (consistent with moderate-resolution boxed hydro slightly
under-producing DLAs); CDDF tracks Prochaska+14 within 0.1–0.3 dex at
all z. Full forensic narrative of the seven bugs found during the audit
is in [`docs/bugs_found.md`](docs/bugs_found.md). Authoritative science
walkthrough: [`docs/analysis.md`](docs/analysis.md). Process-level
handover for next session: [`docs/SESSION_HANDOVER.md`](docs/SESSION_HANDOVER.md).

## Package structure

```
hcd_analysis/
  config.py        io.py           snapshot_map.py
  voigt_utils.py   catalog.py      masking.py
  p1d.py           cddf.py         hcd_template.py
  pipeline.py      report.py
cli/run.py
config/default.yaml
scripts/   batch_greatlakes.sh, install_greatlakes.sh, plot_*, fit_rogers_alpha.py,
           patch_cddf_dx.py, build_hcd_summary.py, ...
tests/     test_*.py (locks the post-audit science claims; see above)
docs/      analysis.md, bugs_found.md, SESSION_HANDOVER.md, data_layout.md, ...
```
