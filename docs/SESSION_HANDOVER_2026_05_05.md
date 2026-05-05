# Session handover — 2026-05-05 (joint-emulator-scaffold + student tutorials)

**Read this first** if you are picking up the *emulator scaffolding /
student-tutorial* thread.  This is a separate thread from the
clustering work covered in
[`SESSION_HANDOVER_2026_04_28.md`](SESSION_HANDOVER_2026_04_28.md) —
the clustering thread is independent, on `main`, with PR #8 already
merged at commit `1d5fe40`.  Read the 2026-04-28 handover *only* if
the user is asking about clustering.

---

## 1. Status snapshot (end of 2026-05-05)

| Item | State |
|---|---|
| Working tree | **clean** |
| Current branch | `joint-emulator-scaffold` (6 commits ahead of `main`, all pushed to `origin`) |
| HEAD commit | `ab7acba` |
| Open PRs on this branch | **none** |
| Tutorials package | committed under `notebooks/tutorials/` — 5 NBs (00–04) + `README.md` + `STUDENT_PROJECTS.md` + `_build_notebooks.py` |
| Phase-1 emulator code | **not started** |
| All 5 tutorial NBs executed cleanly | yes (verified at end of session) |

### Most recent commits on this branch

```
ab7acba Tutorials: learning objectives, concept-checks, background reading
3a46a93 Tutorials: recommended starter path A1 → B1 → B2
ca79f14 Tutorials: STUDENT_PROJECTS.md hand-off list
0f5c5e0 Widen x-windows on DLA F(v) plots
3851838 Tutorial NB04: per-spectrum inspection, finder algorithm, masking
9ad8e2f Tutorial notebooks for HCD catalog and per-class P1D
```

---

## 2. What this session accomplished

The session was **entirely student-onboarding work** — no new pipeline
code, no new science.  The user's explicit ask was a hand-off package
they could give a graduate student or RA to self-onboard onto the
project, including the open emulator scaffolding work.

**Five tutorial notebooks** under `notebooks/tutorials/`:

| # | Notebook | What it covers | Cells |
|---|---|---|---|
| 0 | `00_dataset_layout.ipynb` | Where data lives, parameter encoding, snapshot↔z map, per-(sim,snap) file inventory | 13 |
| 1 | `01_reading_catalogs_and_spectra.ipynb` | `meta.json`, `catalog.npz`, raw HDF5 streaming, visualising one DLA | 20 |
| 2 | `02_recomputing_cddf_and_dndx.ipynb` | `cddf_corrected.npz`, recompute via `measure_cddf_from_dataframe`, dN/dX cross-check, Ho+21 overlay | 13 |
| 3 | `03_recomputing_per_class_p1d.ipynb` | `p1d_per_class.h5`, four Rogers per-class templates, full round-trip via `compute_p1d_per_class` on 30k sightlines | 15 |
| 4 | `04_per_spectrum_inspection_and_masking.ipynb` | τ(v) and F(v) for one LLS/subDLA/DLA each; `find_systems_in_skewer` walked stage-by-stage; fast vs Voigt NHI; three masks × three fills | 30 |

**`STUDENT_PROJECTS.md`** — 12 self-contained projects in 6 themes (A
catalog diagnostics, B CDDF studies, C P1D systematics, D clustering,
E HiRes vs LF, F emulator scaffolding) with difficulty band, time
estimate, prereqs, deliverables and pointer files.  At the top there
is a **recommended starter path: A1 → B1 → B2** (4–7 weeks total)
chosen to give the student catalog-level intuition (A1) before they
go into the actual science work (B1, B2).

**Extended `README.md`** — added per-NB *learning objectives*, per-NB
*concept-check questions* (4 conceptual questions each, complementing
the hands-on coding exercises that already live inside each NB), and
a curated *background reading* section (Lyα forest, HCDs, Voigt /
line transfer, statistics/numerics, programming tools, in-repo refs).

**One bugfix during the session**: NB01's DLA F(v) panel and NB04's
DLA panels had hard-coded x-windows that were narrower than the
saturated cores of the chosen DLAs, so the user reported "I only see
a flat line".  Replaced all four hard-coded windows with adaptive
ones (`max(2 × core_width, floor)`).  See `0f5c5e0`.

---

## 3. Where to start reading (for a fresh Claude on this thread)

In approximate priority order:

| File | Why |
|---|---|
| `notebooks/tutorials/README.md` | Index of the tutorials with learning objectives, concept-checks, and background reading. |
| `notebooks/tutorials/STUDENT_PROJECTS.md` | The 12-project hand-off list with the recommended A1 → B1 → B2 starter path at the top. |
| `notebooks/tutorials/_build_notebooks.py` | The single source-of-truth Python file that generates all 5 NBs.  **Edit the build script, never the .ipynb files directly** — direct .ipynb edits are wiped on next regen. |
| `docs/SESSION_HANDOVER_2026_04_28.md` | Older clustering handover.  Independent thread; only relevant if the user asks about clustering. |

---

## 4. Active TODO list (priority order)

### A. Phase-1 emulator data cache (the next concrete piece of work)

The user signed off on this earlier in the session ("3) yes go ahead
phase 1") but redirected to the tutorials first.  Phase 1 has **not
started** — no code yet.  Plan as agreed in conversation:

* One script `scripts/build_emulator_cache.py` that walks all 1076
  (sim, snap) outputs and writes a single in-repo HDF5 file
  `hcd_analysis/_emulator_data/observables.h5` (gitignored — file
  should be small, ~few MB).
* Schema:
  - `params (1076, 9)` — the 9-D parameter vector parsed from folder
    names via `hcd_analysis.io.parse_sim_params`.
  - `z (1076,)`, `sim (1076,)` (string), `snap (1076,)`.
  - `P_clean (1076, n_k_shared)`, `P_LLS_only (1076, n_k_shared)`,
    `P_subDLA_only (1076, n_k_shared)`, `P_DLA_only (1076, n_k_shared)`
    — interpolated onto a single shared cyclic-k grid because the
    on-disk grids vary in length per (sim, snap).
  - Per-class `mean_F` and `n_sightlines` arrays.
  - `cddf (1076, 30)`, `dndx_LLS / dndx_subDLA / dndx_DLA (1076,)`
    pulled from `cddf_corrected.npz`.
* Read-only over existing files; no recomputation from raw spectra
  (mean-flux dimension is phase 3, not phase 1).
* Spot-check by re-loading the cache and comparing one (sim, snap)
  back to the source `.npz` / `.h5` files.

**Open decision before writing code** (asked at end of session,
unanswered): the **shared k-grid** for the per-class P1D arrays.
Two options:

1. The existing 50-bin extended grid `_DEFAULT_K_BINS` in
   `hcd_analysis/p1d.py` (runs to k_Nyq ≈ 0.05 s/km cyclic;
   already used by the upstream Lyα emulator).  My default
   recommendation.
2. A new uniform log-grid covering the union of per-(sim,snap) k
   ranges (more bins, finer resolution, but a new convention).

Confirm with user before proceeding.

### B. Phase 2 — two-head model + training scaffold (after phase 1)

User decisions already made:

* **Framework: JAX** (chosen for downstream HMC compatibility).  Do
  *not* default to PyTorch.
* **P_HCD output: 3 separate classes** (DLA, subDLA, LLS), not
  summed.
* Architecture as discussed: shared encoder over the 9-D parameter
  vector → latent; head A (latent → CDDF, dN/dX); head B
  (latent, τ₀ → 3-class P1D + P_clean).
* Initially train head B with each sim's natural τ₀ only, so phase 2
  doesn't depend on phase 3.

### C. Phase 3 — mean-flux dimension (after phase 2 baseline trains)

User decisions already made:

* **Post-process the mean-flux scaling** rather than re-run
  fakespectra: `tau → α · tau`, recompute the per-class P1D.  This
  is the cheap version; we can validate against the gold-standard
  later.
* **Sample more than 20 α values per sim** because the NN should
  compress better than the upstream GP emulator.  User suggested
  this is acceptable to be "many more" (e.g. 40–80 α's per sim).

### D. Phase 4 — joint-likelihood interface

Wrapper that takes (params, τ₀), returns the four observables and
their covariance, with a `predict_p1d_inputs()` shortcut returning
just (P_HCD per class, P_clean) for the upstream P1D pipeline.

---

## 5. Conventions / gotchas (tutorial-specific)

These supplement the conventions in `SESSION_HANDOVER_2026_04_28.md`
§5 (which still apply for clustering).  Tutorial-specific:

1. **Notebooks are auto-generated.**  Edit `_build_notebooks.py`,
   then run `python3 notebooks/tutorials/_build_notebooks.py`, then
   re-execute the affected notebook with `jupyter nbconvert --to
   notebook --execute`.  *Direct .ipynb edits are wiped on next
   regen.*
2. **NB 03 takes 1–2 minutes** to execute (streams 30 k sightlines
   through `compute_p1d_per_class`).  NBs 00 / 01 / 02 / 04 each run
   in seconds.  Run NB 03 in background when re-executing all five.
3. **k-grid varies per (sim, snap)** because `nbins` does — different
   `H(z)` and box sizes give different velocity-pixel widths.  Phase
   1 must interpolate onto a shared grid before stacking.
4. **Always use `cddf_corrected.npz`** — `cddf.npz` is the older
   buggy version with a missing `(1+z)·h` factor.
5. **Per-class P1D uses per-subset mean flux** (Rogers convention),
   not the global ⟨F⟩.  This is what makes
   `P_<class>_only / P_clean` a clean template independent of forest
   amplitude.
6. **Example sim used throughout the tutorials**:
   `ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333`,
   snap 022 (z = 2.0, dv = 10.0 km/s, nbins = 1228).  Use the same
   one when extending the tutorials so the student doesn't have to
   re-orient.
7. **DLA F(v) plots need wide x-windows.**  The chosen DLAs in NB01
   and NB04 have cores of 60–280 pixels (600–2800 km/s); a fixed
   window narrower than ~2× the core width shows only a flat
   saturated line.  Use the adaptive window pattern
   (`max(2 * core_kms, floor)`) in any new DLA plot.

---

## 6. Open questions / caveats

1. **Phase-1 shared k-grid choice.**  See §4 A.  Default = existing
   `_DEFAULT_K_BINS` unless user prefers otherwise.
2. **Phase-1 start gate.**  User said "yes go ahead" earlier in the
   session but redirected to tutorials.  Confirm before starting
   phase 1, since the conversation has been long.
3. **Tutorial maintenance.**  If anyone changes
   `hcd_analysis.catalog`, `hcd_analysis.cddf`, or
   `hcd_analysis.p1d` in ways that break the NB execution, the NBs
   will go stale.  The build script is the recovery path: regenerate,
   re-execute, fix any new errors.

---

## 7. Quick commands

Verify state in 30 seconds:

```
git status                                          # should be clean
git log --oneline main..HEAD                        # 6 commits if on joint-emulator-scaffold
ls notebooks/tutorials/                             # 5 NBs + 3 .md + _build_notebooks.py
gh pr list --head joint-emulator-scaffold           # no open PRs
```

Re-execute all 5 NBs (NB 03 in the background because it's the slow one):

```
for nb in 00_dataset_layout 01_reading_catalogs_and_spectra \
          02_recomputing_cddf_and_dndx 04_per_spectrum_inspection_and_masking; do
  jupyter nbconvert --to notebook --execute "notebooks/tutorials/$nb.ipynb" --inplace
done
# Run NB 03 in the background or accept ~2 min wall time:
jupyter nbconvert --to notebook --execute notebooks/tutorials/03_recomputing_per_class_p1d.ipynb --inplace
```

Audit all 5 NBs for execution errors:

```
python3 -c "
import nbformat
for n in ['00_dataset_layout','01_reading_catalogs_and_spectra',
          '02_recomputing_cddf_and_dndx','03_recomputing_per_class_p1d',
          '04_per_spectrum_inspection_and_masking']:
    nb = nbformat.read(f'notebooks/tutorials/{n}.ipynb', as_version=4)
    errs = sum(1 for c in nb.cells if c.cell_type=='code'
               for o in c.get('outputs',[]) if o.get('output_type')=='error')
    print(f'{n}: cells={len(nb.cells)}  errors={errs}')
"
```

---

## 8. How to pick up

1. Read `notebooks/tutorials/README.md` end-to-end (~ 5 min) to see
   the shape of the hand-off package.
2. Run the verify-state commands in §7.
3. Ask the user what they want to do next.  Almost certainly one of:
   * Confirm the shared-k-grid choice and start phase 1
     (`scripts/build_emulator_cache.py`).
   * Polish or extend the tutorial package.
   * Pick up a clustering item from §4 of the 2026-04-28 handover.
4. If unsure, default to phase 1 (it's the next thing in the agreed
   plan) and ask the k-grid question first.

**Don't:**

* Start phase-2 model code until phase 1 has produced
  `observables.h5` and you've verified its contents.
* Default to PyTorch.  User picked JAX for HMC compatibility.
* Reduce the τ₀ samples below ~20 in phase 3.  User explicitly wants
  more than that.
* Edit notebook .ipynb files directly — always edit
  `_build_notebooks.py` and regenerate.
