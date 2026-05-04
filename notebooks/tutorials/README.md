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

## Learning objectives

After working through each notebook you should be able to do the
following on a fresh terminal, *without looking back at the notebook*.
If you can't, you haven't finished the notebook — re-read the relevant
section.

### NB 00 — dataset layout
- Locate the raw spectra (Turbo) and processed outputs (scratch) on the
  cluster, and explain why the two live in different filesystems.
- Parse a 9-parameter point from a sim folder name using the
  `parse_sim_params` helper, and recite the names of the 9 parameters.
- Map a snapshot index (e.g. snap 022) to its redshift via
  `Snapshots.txt`, and explain why a few sims are missing snap 022 / 023.
- Name the four files in `<sim>/snap_NNN/` that the rest of the project
  reads from (`catalog.npz`, `cddf_corrected.npz`, `p1d_per_class.h5`,
  `meta.json`) and what each one contains.

### NB 01 — reading catalogs and spectra
- Load `meta.json` and confirm absorber counts agree with the per-class
  thresholds (LLS, subDLA, DLA).
- Open `lya_forest_spectra_grid_480.hdf5` and read **one** sightline's
  τ array without loading the whole 3 GB file.
- Convert τ(v) → F(v) and identify the visual signature of a DLA
  (saturated core + damping wings) versus an LLS (narrow optically-thick
  dip with no wings).
- Translate a catalog row's `(skewer_idx, pix_start, pix_end)` into a
  velocity range on the same skewer.

### NB 02 — recomputing CDDF and dN/dX
- State the definition `f(N_HI) = d²n / (dN_HI dX)` and explain why we
  use the absorption distance `dX` rather than redshift `dz`.
- Recompute `f(N_HI)` from `catalog.npz` using
  `measure_cddf_from_dataframe` and reproduce the cached
  `cddf_corrected.npz` to ~1e-6 relative precision.
- Compute per-class `dN/dX` (LLS, subDLA, DLA) and cross-check against
  `hcd_summary_lf.h5`.
- Overplot one observational CDDF (Ho+21, Prochaska+05) at the matching
  redshift and comment on the agreement / disagreement.

### NB 03 — recomputing per-class P1D
- State the definition of `P1D(k)` and the discrete formula in terms of
  `numpy.fft.rfft`.
- Explain why each per-class P1D uses its **own subset's** mean flux for
  δF normalisation (the Rogers convention) rather than the global ⟨F⟩.
- Run `compute_p1d_per_class` on a 30 000-sightline subsample and
  reproduce the cached `P_clean` to ~1% precision (the noisier classes
  agree on shape but not amplitude on a small subsample).
- Plot the Rogers ratio `P_<class>_only / P_clean` and explain what it
  means physically.

### NB 04 — per-spectrum inspection and masking
- Walk through `find_systems_in_skewer` on a real skewer, naming the
  four stages (threshold → connected runs in the doubled array → merge
  close runs → drop short runs).
- Distinguish `nhi_from_tau_fast` (production LF) from `fit_nhi_from_tau`
  (HiRes Voigt fit) and explain when each is preferable.
- Build all three masks (`pixrange`, `tauspace`, `priya`) on the same
  DLA and explain which is the production choice and why.
- Apply the three fill strategies (`zero_tau`, `mean_flux`,
  `contiguous`) and explain the effect on δF in the masked region.

## Concept-check questions

These are short questions you should be able to answer after each
notebook.  They are **not** code-writing exercises (those are at the
end of each notebook under "Suggested student exercises") — they are
conceptual checks to verify you understood the material.  If you find
yourself stuck, that is a signal to re-read the relevant section or
look at the background reading below.

### After NB 00

1. There are 60 LF sims and 4 HR sims.  Where are the HR sims on disk,
   and how would you load all 4 as `SimInfo` objects?
2. A snap with `a = 0.31250` corresponds to what redshift?  Why do
   sims share the same `(snap, z)` mapping?
3. Why does `nbins` (number of velocity pixels per skewer) vary across
   sims at the *same* snapshot?
4. What's stored in `<sim>/snap_NNN/done`?  Why is it useful?

### After NB 01

1. The example sim's catalog has 81 762 absorbers at snap 022.  Where
   does this number come from in `meta.json`?  Verify the
   bookkeeping holds.
2. `tau` is stored as float32 on disk but the catalog NHI is float64.
   Why is float32 enough for the τ values we deal with?
3. The catalog's `pix_start..pix_end` window for a DLA only covers the
   saturated *core* — not the damping wings.  How would you find the
   full wing extent of one DLA from the τ array alone (no Voigt fit)?
4. Why is `fast_mode = True` in the LF catalog but `False` in the HR
   catalog?  See `docs/fast_mode_physics.md` for the answer.

### After NB 02

1. Derive `dX/dz = (1+z)² · H₀/H(z)` starting from "comoving distance
   travelled by a photon in time `Δt`".  Why does this matter for CDDF
   normalisation?
2. The corrected CDDF differs from the original by a factor of
   `(1+z)·h`.  At z = 2.2 with h = 0.67, what is that factor numerically?
   Why is this the difference between a 0.3 dex offset vs Prochaska+05
   and a 0.0 dex one?
3. dN/dX(DLA) at z = 2.2 is ~0.05 in this sim.  What does that number
   mean physically?  Convert it to "expected DLAs per unit comoving
   Mpc/h sightline length".
4. The simulation tends to over-predict the LLS regime relative to
   Ho+21.  Name two physical reasons this could be (a hint: think
   about UV background and small-scale density fluctuations).

### After NB 03

1. Why does `P_DLA_only` use its own subset's mean flux ⟨F⟩_DLA rather
   than the global ⟨F⟩?  What would change in the Rogers ratio
   `P_DLA_only / P_clean` if we used the global ⟨F⟩?
2. The k-array on disk is `numpy.fft.rfftfreq(nbins, d=dv)`.  PRIYA's
   "angular k" is `2π × this`.  Why are there two conventions, and
   which is which?  (Hint: cyclic frequency vs angular frequency.)
3. The DLA template ratio `P_DLA_only / P_clean` shows a low-k
   enhancement and a high-k upturn.  What real-space feature drives
   each?  See figure caption in the
   `figures/diagnostics/p1d_masking/per_class_realspace_fourier.png`.
4. If you raise the NHI threshold for the "DLA" class from 20.3 to
   20.5, do you expect `P_DLA_only` at low k to *rise* or *fall*?
   Argue from first principles.

### After NB 04

1. The detection algorithm scans a *doubled* `[above | above]` array
   instead of the original `above` array.  What problem does the
   doubling solve?  How would the algorithm fail if you scanned only
   the original?
2. For a DLA with log NHI = 21, b = 25 km/s, where does τ drop below
   1 in the damping wing?  Compute analytically using the Lorentzian
   tail of the Voigt profile, and verify against the data using one
   real DLA from the notebook.
3. The `pixrange` mask covers fewer pixels than `tauspace`.  Why does
   that make `pixrange` a *bad* choice for masked-P1D analysis at
   small k?  Reference `docs/masking_strategy.md` figure 7.
4. The PRIYA mask only kicks in when `max(τ) > 10⁶`.  What NHI
   threshold does that correspond to (assume b = 30 km/s)?  Why is
   that consistent with a "DLA-only" mask?

## Background reading

You don't need any of this to start the tutorials, but if you want
deeper grounding in any of the underlying physics, statistics, or
programming, here are some starting points by topic.

### The Lyman-α forest as a cosmological probe

- **Croft, Weinberg, Katz, Hernquist 1998** — "Recovery of the Power
  Spectrum of Mass Fluctuations from Observations of the Lyα Forest".
  The original Lyα-cosmology paper; reading just §1–§3 gives you the
  conceptual basis for everything else.
- **McDonald, Miralda-Escudé et al. 2000, 2006** — definitive P1D
  papers from the SDSS era.  The 2006 paper formalises the P1D
  estimator we use today.
- **Palanque-Delabrouille et al. 2013** — "The one-dimensional Lyα
  forest power spectrum from BOSS".  Modern P1D measurement and
  systematics.
- **Khaire, Walther, Hennawi et al. 2024** — the PRIYA simulation
  suite paper.  This is the suite our 60 LF + 4 HR sims came from.

### High column-density absorbers (HCDs)

- **Wolfe, Gawiser, Prochaska 2005** (ARA&A 43, 861) — the canonical
  DLA review.  Sections 1–3 cover the LLS / subDLA / DLA
  classification and the physics of damping wings.
- **Prochaska, Herbert-Fort, Wolfe 2005** — the SDSS DLA CDDF that
  notebook 02 compares against.
- **Bird et al. 2014** ("Damped Lyα systems at low redshift") — first
  hydrodynamic prediction of DLA bias.  Section 5 is what feeds the
  scale-dependent-bias TODO in `SESSION_HANDOVER_2026_04_28.md` §4D.
- **Font-Ribera, Kirkby et al. 2012** — the BOSS DLA-Lyα cross
  correlation that the project's clustering pipeline mirrors.
- **Pérez-Ràfols et al. 2018** — DR12 DLA bias measurement.  This is
  the observation our `b_DLA` estimate is being compared against.
- **Ho, Bird, Garnett 2021** — modern neural DLA finder; we use their
  CDDF tables for the observation overlay in NB 02.

### Voigt profile and line-transfer

- **Draine 2011, _Physics of the ISM_, §6** — textbook derivation of
  the Voigt profile from radiative transfer.  This is the
  cleanest single source.
- **Tepper-García 2006** — analytic approximation to the Voigt
  function that's faster than `scipy.special.wofz`.  Several
  fake-spectra implementations use it.

### Statistics / numerics

- **Numerical Recipes** (Press, Teukolsky, Vetterling, Flannery) —
  Ch. 13 (FFT) is the right reference for the P1D estimator.
- **Hamilton 1992** — multipole expansion of the correlation function.
  This is the formalism behind the rmu Hamilton-multipole fit in
  `docs/multipole_jacobian_explained.md`.
- **Kaiser 1987** — redshift-space distortions.  The basis for the
  `(b_DLA, β_DLA)` joint-fit model.
- **Goodman & Weare 2010** — affine-invariant ensemble MCMC (the
  algorithm behind `emcee`).  Useful background for the eventual
  emulator MCMC.

### Programming and tools

- **NumPy, SciPy, matplotlib, h5py official documentation** — the four
  packages that 90 % of this project's code depends on.
- **Software Carpentry** lessons (https://software-carpentry.org) —
  if you are new to scientific Python, work through their NumPy and
  matplotlib lessons first.
- **fake_spectra** package documentation (Bird et al. on GitHub) —
  the upstream simulation post-processing this project sits on top of.
- **Jupyter / nbformat** documentation — for understanding the
  `_build_notebooks.py` pattern this folder uses.

### In-repo references (read these as you go)

| Topic | File |
|---|---|
| Project state, current TODOs | `docs/SESSION_HANDOVER_2026_04_28.md` |
| Older audit context (CDDF / fast-mode / per-class templates) | `docs/SESSION_HANDOVER.md` |
| Bug history (the seven bugs the audit caught) | `docs/bugs_found.md` |
| Why fast-mode NHI estimation works | `docs/fast_mode_physics.md` |
| Mask choice rationale | `docs/masking_strategy.md` |
| Clustering definitions and conventions | `docs/clustering_definitions.md` |
| Catalog truth-validation against particle data | `docs/dla_truth_validation.md` |
| Figure index for the analysis figures | `docs/analysis_index.md` |

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
