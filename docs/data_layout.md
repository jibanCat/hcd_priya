# Data Layout

**Root:** `/nfs/turbo/umor-yueyingn/mfho/emu_full/`

## Simulation Folders (60 total)

Each folder encodes the parameter point as a name string:
```
ns{ns}Ap{Ap}herei{herei}heref{heref}alphaq{alphaq}hub{hub}omegamh2{omegamh2}hireionz{hireionz}bhfeedback{bhfeedback}
```

### Parameter definitions
| Symbol | Description | Range |
|--------|-------------|-------|
| ns | spectral index | 0.80 – 1.05 |
| Ap | amplitude A_s (10^-9) | 1.2e-9 – 2.6e-9 |
| herei | HeII reionisation start z | 3.5 – 4.1 |
| heref | HeII reionisation end z | 2.6 – 3.2 |
| alphaq | quasar spectral index | 1.3 – 3.0 |
| hub | Hubble h | 0.65 – 0.75 |
| omegamh2 | Omega_m h^2 | 0.140 – 0.146 |
| hireionz | HI reionisation z | 6.5 – 8.0 |
| bhfeedback | black hole feedback | 0.03 – 0.07 |

## Directory tree

```
{sim_folder}/
  output/
    Snapshots.txt        ← snap_idx  scale_factor_a
    SPECTRA_004/
      lya_forest_spectra_grid_480.hdf5   ← primary (691200 skewers)
    SPECTRA_005/
      lya_forest_spectra_grid_480.hdf5
    ...
    SPECTRA_023/
      lya_forest_spectra_grid_480.hdf5
```

## HDF5 Schema (`lya_forest_spectra_grid_480.hdf5`)

```
/
  Header/
    attrs:
      redshift   float   e.g. 5.4
      Hz         float   H(z) in km/s/Mpc
      box        float   120000.0 (kpc/h, comoving)
      hubble     float   h
      nbins      int     1556  (pixels per skewer)
      npart      array   particle counts
      omegab     float
      omegal     float
      omegam     float
  spectra/
    axis   (691200,)    int32   LOS axis: 1=x, 2=y, 3=z
    cofm   (691200, 3)  float64 skewer position (kpc/h, comoving)
  tau/
    H/
      1/
        1215   (691200, 1556)  float32   HI Lyman-alpha optical depth
  colden/          ← GROUP EXISTS BUT IS EMPTY
  tau_obs/         ← EMPTY
  temperature/     ← EMPTY
  velocity/        ← EMPTY
  density_weight_density/  ← EMPTY
  num_important/   ← EMPTY
```

**Important:** `colden` is an empty group. Column densities are not pre-computed
and must be inferred from `tau/H/1/1215`.

## Snapshots and Redshifts

Snapshots.txt maps snapshot index → scale factor a. Redshift: z = 1/a - 1.

| Snap | a       | z    | In 2≤z≤6 |
|------|---------|------|-----------|
| 004  | 0.15625 | 5.40 | yes |
| 005  | 0.16129 | 5.20 | yes |
| 006  | 0.16667 | 5.00 | yes |
| 007  | 0.17241 | 4.80 | yes |
| 008  | 0.17857 | 4.60 | yes |
| 009  | 0.18519 | 4.40 | yes |
| 010  | 0.19231 | 4.20 | yes |
| 011  | 0.20000 | 4.00 | yes |
| 012  | 0.20833 | 3.80 | yes |
| 013  | 0.21112 | 3.74 | yes |
| 014  | 0.21739 | 3.60 | yes |
| 015  | 0.22727 | 3.40 | yes |
| 016  | 0.23810 | 3.20 | yes |
| 017  | 0.25000 | 3.00 | yes |
| 018  | 0.26316 | 2.80 | yes |
| 019  | 0.27778 | 2.60 | yes |
| 020  | 0.28077 | 2.56 | yes |
| 021  | 0.29412 | 2.40 | yes |
| 022  | 0.31250 | 2.20 | yes |
| 023  | 0.33333 | 2.00 | yes |

## Pixel velocity width

dv = H(z) × (box_Mpc_physical) / nbins ≈ **10.0 km/s** at z=5.4
(this is a coincidental round number; exact value is stored in each file's Header).

## Snapshot availability

Not all simulations have all snapshots. Summary from the 60-sim dataset:

| Snap | z    | N_sims |
|------|------|--------|
| 004  | 5.40 | 60 / 60 |
| 005  | 5.20 | 60 / 60 |
| ...  |      |         |
| 022  | 2.20 | 48 / 60 |
| 023  | 2.00 | 18 / 60 |

Lower-redshift snapshots (snap 022, 023) are missing in many simulations.
The pipeline handles missing snapshots gracefully.

## Secondary file

Some simulations also contain `lya_forest_spectra.hdf5` (32000 skewers,
1747 pixels, float64). This is a random subsample and is used only when
the grid file is absent.

---

## Analysis-output data (separate from the git repo)

Large generated artifacts (HDF5 summary tables, CSVs) live **outside
git** to keep the repo lean.  Only scripts, docs, and figures that
appear inline in the docs are version-controlled.

### Paths

| Where | What |
|---|---|
| `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/` | Production pipeline output (per-sim `catalog.npz`, `p1d.npz`, `p1d_per_class.h5`, …). |
| `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/_hcd_analysis_data/` | HCD summary tables produced by analysis scripts (`hcd_summary_lf.h5`, `hcd_summary_hr.h5`, MF coefficient CSVs, bootstrap tables).  Reached via the `data_dir()` helper in `scripts/common.py`. |
| `/nfs/turbo/umor-yueyingn/mfho/` | Long-term persistent storage (Turbo filesystem).  **TODO**: migrate `_hcd_analysis_data/` here once results are stable. |

### How to override

Every analysis script reads its summary files through

```python
from common import data_dir
DATA = data_dir()
```

By default this returns the scratch path above.  To redirect (e.g. to
Turbo), export the env var:

```bash
export HCD_DATA_DIR=/nfs/turbo/umor-yueyingn/mfho/hcd_analysis_data
```

The helper creates the directory if it doesn't exist, so relocating is
a one-line change — no script edits needed.

### Regenerating the summaries

If you arrive at the repo without any data on scratch, rebuild with:

```bash
python3 scripts/build_hcd_summary.py
```

This scans `hcd_outputs/` (LF) and `hcd_outputs/hires/` (HR) and writes
`hcd_summary_{lf,hr}.h5` into `data_dir()`.  All downstream MF scripts
depend on these two HDF5s being present.

### `.gitignore` entries

The following paths are intentionally not tracked:

```
.claude/                        # local Claude Code state
logs/                           # SLURM / pipeline logs
figures/intermediate/           # pre-audit stale figures
figures/analysis/data/          # generated artefacts (use scratch instead)
__pycache__/                    # Python caches
*.pyc
```
