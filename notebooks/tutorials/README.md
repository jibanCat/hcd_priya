# HCD tutorials

Five short notebooks that walk a new student through the per-(sim, snap)
HCD outputs from the ground up.  Read them in order:

| # | Notebook | What it covers |
|---|----------|----------------|
| 0 | [`00_dataset_layout.ipynb`](00_dataset_layout.ipynb) | Where the raw spectra and processed outputs live, how the 9-D parameter point is encoded in the simulation folder name, the snapshot ↔ redshift mapping, and what files exist for each (sim, snap). |
| 1 | [`01_reading_catalogs_and_spectra.ipynb`](01_reading_catalogs_and_spectra.ipynb) | Loading `meta.json` and `catalog.npz`; inspecting absorber records and the LLS / subDLA / DLA classification; opening the raw `lya_forest_spectra_grid_480.hdf5` file without loading it; visualising one DLA in `tau` and `F`. |
| 2 | [`02_recomputing_cddf_and_dndx.ipynb`](02_recomputing_cddf_and_dndx.ipynb) | Reading `cddf_corrected.npz`; recomputing `f(N_HI)` and per-class `dN/dX` from the catalog; cross-checking against the project-wide `hcd_summary_lf.h5`; overplotting Ho+21. |
| 3 | [`03_recomputing_per_class_p1d.ipynb`](03_recomputing_per_class_p1d.ipynb) | Reading `p1d_per_class.h5`; plotting the four Rogers per-class templates; recomputing `compute_p1d_per_class` from raw spectra + catalog on a 30k-sightline subsample to confirm the round-trip. |
| 4 | [`04_per_spectrum_inspection_and_masking.ipynb`](04_per_spectrum_inspection_and_masking.ipynb) | Per-spectrum view of LLS / subDLA / DLA in `τ(v)` and `F(v)`; step-by-step walkthrough of `find_systems_in_skewer` on a real multi-absorber skewer; fast vs Voigt NHI estimation; comparison of the three masking strategies (`pixrange`, `tauspace`, `priya`) and the three fill strategies (`zero_tau`, `mean_flux`, `contiguous`). |

## Running them yourself

The notebooks are committed with their executed outputs.  To re-run from
scratch::

    for nb in 00_dataset_layout 01_reading_catalogs_and_spectra \
              02_recomputing_cddf_and_dndx 03_recomputing_per_class_p1d \
              04_per_spectrum_inspection_and_masking; do
      jupyter nbconvert --to notebook --execute "$nb.ipynb" --inplace
    done

NB 00–02 and NB 04 each take a few seconds.  NB 03 streams 30 000
sightlines from the raw spectra HDF5 and takes about 1–2 minutes;
reduce `n_skewers` or choose a different sim/snap if you need it faster.

The notebook contents are generated from `_build_notebooks.py`.  If you
want to edit an explanation, prefer to edit the build script and re-run::

    python3 _build_notebooks.py

then re-execute the affected notebook.  This keeps text and code in one
auditable place.

## Project ideas after the tutorials

Once you've worked through the five notebooks and want a self-contained
project to take on, see [`STUDENT_PROJECTS.md`](STUDENT_PROJECTS.md).
It lists ~12 projects organised by theme (catalog diagnostics, CDDF
studies, P1D systematics, clustering, emulator scaffolding) with
difficulty, time estimates, deliverables, and entry-point file pointers
for each.

## What's next after these tutorials

The four observables you've now learned to read — `catalog`,
`cddf_corrected`, `p1d_per_class`, and `meta.json` — are the inputs to
the HCD emulator under construction.  The next milestone is **phase 1
of the emulator work**: building a single in-repo HDF5 cache that stacks
the per-(sim, snap) observables across all 1076 outputs onto a shared
k-grid and NHI-grid.  See `docs/SESSION_HANDOVER_2026_04_28.md` for the
overall project state and the next planning thread for the emulator
design.
