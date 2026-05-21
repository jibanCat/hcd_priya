# Session handover — 2026-05-21

**Branch:** `phase2-emulator-jax` (open PR #10 vs `main`)
**Thread:** HCD emulator — Phase 2 production τ₀ cache build
**Supersedes:** `docs/SESSION_HANDOVER_2026_05_20.md` (refactor complete; this
session prepped the production build and **redesigned the cache schema** before
launching).

**STATUS: PAUSED mid-prep — production build NOT launched.** Waiting on (a) data
transfers to finish (the 6-HR suite) and (b) a final cache-schema rework that
the 2026-05-21 design answers require. Read §3 (decisions) and §4 (next steps)
to resume.

---

## 1. What this session did

1. **Executed + merged the Phase-2a refactor** (6 TDD tasks, commits
   `7e5a444`..`8c853be`, pushed): builder now drives fake_spectra
   (`hcd_analysis/priya_p1d.py`) → Tier P (bit-identical to PRIYA) + Tier C
   (per-class). PR #10 title/body updated.
2. **Resolved the "Ap" parameter** (commit `ab0cf04`, unpushed): PRIYA's
   emulator `Ap` is the primordial amplitude at the Lyα pivot k_p = π/4 /Mpc,
   while CAMB `scalar_amp` (As) is at k_0 = 0.05 /Mpc:
   **Ap = As · (5π)^(ns−1)** (5π = (π/4)/0.05). Builder now reads PRIYA-exact
   params from `SimulationICs.json` (8 direct + this Ap transform), not the
   folder-rounded `parse_sim_params`. Verified to machine precision vs PRIYA's
   params for sims 0/29/44.
3. **Multi-point PRIYA bit-identity** confirmed (120 points: 3 sims × 4 z × 10
   α) worst max|r−1| = 1.89e-5; the lone z=4.6 outlier was a z-grid mismatch
   (now snapped to PRIYA's zout grid). See `docs/superpowers/2026-05-20-priya-p1d-consistency-check.md`.
4. **Production timing pilot** (sbatch 50582218): 2 full-res pairs × 20 α =
   **~70 min/pair**, **34 GB peak**, output schema-valid. So `--mem=48G`,
   FFT is single-threaded (extra cores don't speed one pair).
5. **PRIYA-aligned 20-α grid + HR-aware discovery + merge infra** (commit this
   session, unpushed): see §3 / the commit message.

**Unpushed commits:** `ab0cf04` (Ap fix) + the PRIYA-aligned-grid/HR/merge
commit. Push with `git push origin phase2-emulator-jax` to update PR #10.

---

## 2. Code state (all committed; working tree clean)

- `hcd_analysis/priya_p1d.py` — fake_spectra wrappers: `compute_tier_p_p1d`,
  `compute_tier_c_p1d`, `_apply_priya_filter`, `_classify_sightlines`,
  `_per_class_p1d_at_scale`.
- `scripts/build_emulator_cache_tau0.py` — `build_tau0_rows` (Tier P+C),
  `discover_tau0_pairs` (LF + `hires/`, sorted), `locate_raw_tau_file`
  (+ params-fallback `_match_emu_folder_by_params`), `_read_priya_params`
  (SimICs + Ap transform), `write_cache_tau0` (v2.0 schema), `main`
  (PRIYA-aligned grid, `--alpha-refine` default 2 → 20 α).
- `hcd_analysis/tau0_rescale.py` — `make_alpha_grid_priya_aligned`,
  `slope_alpha_to_target_F`, `obs_mean_tau_kim2013`, `PRIYA_ALPHA_LO/HI`.
  (`freeze_core_rescale`/`tau0_from_mean_flux` are LEGACY.)
- `scripts/merge_tau0_cache.py` — concatenate shards, remap `snap_group_idx`
  (synthetic-shard tested).
- `scripts/consistency_checks/` — drivers + `sbatch_multipoint.sh` (the proven
  parallel pattern) + `sbatch_build_pilot.sh`.
- Tests green: `tests/test_priya_p1d.py` (bit-identity; SLOW_TESTS multipoint),
  `tests/test_emulator_cache_tau0.py` (v2.0 round-trip, HR discovery, params,
  real-data Tier-P-vs-PRIYA), `tests/test_tau0_rescale.py`.

### ⚠️ CRITICAL env (anything importing fake_spectra)
```bash
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
/home/mfho/.conda/envs/emu-3.9/bin/python3 <script>
```
Do NOT `conda activate emu-3.9` (mamba module reload corrupts it). GSL first
(libgsl.so.25), then conda env lib (scipy `_highs` needs its newer libstdc++).
Tests import-guard → print SKIP under the wrong env; **a SKIP is NOT a pass.**

---

## 3. Decisions locked this session (drive the schema rework)

From the user on 2026-05-21 (some in `docs/superpowers/reply_to_open_questions.txt`):

- **z-range:** build full **z = 5.4 → 2.0** (1076 LF pairs), but the
  PRIYA-overlap (z=2.2–4.6) MUST be bit-identical to PRIYA.
- **α = 20**, PRIYA-aligned so **10 of 20 match PRIYA exactly** (DONE:
  `make_alpha_grid_priya_aligned`, PRIYA points at even indices).
- **HR = all 6 sims** from **`/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/`**
  (has the newer 6-sim reference). NOT the 4 in `emu_full`. HR snaps start at
  `SPECTRA_005`; discover by z. (The 4th-HR rounding-fallback in
  `locate_raw_tau_file` was for the `emu_full` layout; the 6-HR source may not
  need it — re-check discovery against the new path.)
- **Cost:** <10,000 CPU-h is fine; **parallelize sbatch for <1–2 day wallclock.**
- **Filtering:** keep the EXPENSIVE per-α filter (no corner-cutting).

### Cache-schema rework REQUIRED before the production build (NOT yet done):

1. **Native k-grid, not interpolated.** Store P1D on the native kfmpc grid
   (LF 172, HR 522), rebin only at the emulator stage. → **separate LF and HR
   caches** (`observables_tau0_lf.h5`, `observables_tau0_hr.h5`), each with its
   kfmpc. This *improves* PRIYA bit-identity (direct native compare). The
   current builder interpolates to a 50-bin angular grid — replace that. NOTE:
   Tier-P bit-identity test is already on the native grid, so it carries over.
2. **Tier C → fine N_HI binning** (the user's flexible/continuous-P_dirty idea).
   Bin sightlines by their highest absorber's log N_HI into **uniform 0.25-dex
   bins (17.0→22.5, ~22 bins) + a ≥22.5 overflow + a clean (no-absorber) bin**;
   store per-bin P1D **and sightline counts**. Any class (Rogers-4 or
   inference-time custom N_HI cuts) reconstructs as a count-weighted sum (P1D is
   sightline-additive — the verified sum-identity). **Occupancy measured**
   (existing catalogs): 0.25 dex is robust (≥~1500 sightlines, <2.5%) up to
   log N_HI ≈ 21.5; the ≥21.5 large-DLA tail is intrinsically sparse (hundreds,
   then tens) and worse at low z. Recommendation = keep uniform 0.25 dex + store
   counts + merge-at-inference (don't pre-merge — preserves flexibility, counts
   flag noisy bins). **PENDING user confirm:** uniform-0.25 vs pre-merge the
   sparse ≥21.5 tail. (Leaning uniform.)
3. **DLA α-response (DECIDED: measure-then-correct).** Physically α should hit
   forest + subDLA/LLS (subDLA/LLS possibly differently from forest) but NOT
   DLA (saturated/self-shielded). Build the **global-α τ-rescale baseline** now
   (what the current code does), and **measure the true per-class UV response**
   from the PART particle fields (`/scratch/yueyingn_root/yueyingn0/mfho/priya/PART/emu_full/`),
   only 3 snaps at z=3 — accept the hope it's parameter- and z-insensitive.
   That study informs any freeze/differential correction later. The current
   global-α Tier C rescales DLA pixels too (an acknowledged approximation).

### Phase-2b design (logged; doesn't gate the cache)
- CDDF: **separate** emulator (like T0); try a joint CDDF/P1D training as an
  accuracy experiment.
- Four-class = mirror Rogers-2018; but prefer the flexible-N_HI scheme above,
  verify against fixed-4-class.
- Mean-flux per class as in #3.
- Multi-fidelity: build **single-fidelity NN first, then a residual MF layer**
  (6 HR now enable it, vs 3 in 2023).
- k: native in training data, rebin at emulator stage (#1).

---

## 4. Next steps (resume order)

1. **Wait for transfers**, then verify the 6-HR data at `emu_full_hires_2`
   (SPECTRA grids present per sim; SimulationICs.json present; PRIYA 6-sim ref).
2. **Confirm the fine-bin decision** (uniform 0.25 dex vs pre-merge tail).
3. **Rework the cache schema**: native k-grid (separate LF/HR), fine-N_HI Tier C
   + counts, 6-HR source. Re-verify Tier-P bit-identity on the native grid;
   update `merge_tau0_cache.py` for the new schema.
4. **Size + launch the production sbatch array** (LF 1076 + HR 6-sims; ~70
   min/pair @ 20 α; shard ~2–3 pairs/job, `--mem=48G`, `--time≈6h`; aim
   <1–2 day wallclock). Per-sim snap-by-z lookup. Account `cavestru0`,
   partition `standard`. (Array/concurrency limits were NOT checked — user
   declined the `scontrol`/`sacctmgr` probe; ask or check at submit time.)
5. **Merge shards** → `observables_tau0_lf.h5` + `observables_tau0_hr.h5`
   (gitignored under `hcd_analysis/_emulator_data/`).
6. **Post-build PRIYA-overlap validation**: z=2.2–4.6 × the 10 PRIYA α rows must
   be bit-identical to PRIYA LF (600-row ref) and HR (6-sim ref at
   `emu_full_hires_2/mf_emulator_flux_vectors_tau1000000.hdf5`).
7. **PART particle study**: per-class UV response (3 snaps, z=3) → inform DLA/
   subDLA/LLS α-response correction.
8. **Write the Phase-2b plan** (JAX loader + encoder + Head A/B + training +
   likelihood). Then cross-validation accuracy is a real metric.

---

## 5. Key data locations

| What | Path |
|---|---|
| LF raw τ | `/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/SPECTRA_NNN/lya_forest_spectra_grid_480.hdf5` |
| LF Phase-1 (meta/cddf/catalog) | `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/<sim>/snap_NNN/` |
| **HR raw τ (6 sims — USE THIS)** | `/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/<sim>/output/SPECTRA_NNN/` |
| HR Phase-1 (4 sims, older) | `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/hires/<sim>/snap_NNN/` |
| PART particle fields (UV study) | `/scratch/yueyingn_root/yueyingn0/mfho/priya/PART/emu_full/` |
| PRIYA LF ref (60×10, 172 k, z2.2–4.6) | `/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5` |
| PRIYA HR ref (old 3-sim, 522 k) | `.../kodiaq_2_2_4_6-48-48/hires/kmax2.0_mf_emulator_flux_vectors_tau1000000.hdf5` |
| PRIYA HR ref (new 6-sim) | `/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/mf_emulator_flux_vectors_tau1000000.hdf5` |

PRIYA params layout: `params[:,0]` = α (Kim slope, sim-independent
`linspace(0.65555638,1.29568388,10)`); `params[:,1:]` = 9 cosmo in
`bec.PARAM_ORDER`. Rows are α-major: row = `sim_idx + n_sim*alpha_idx`.

---

## 6. Reading order to resume
1. This file.
2. `docs/superpowers/2026-05-20-priya-p1d-consistency-check.md` (FINAL VERDICT +
   §6b/§6c) — the bit-identity result + the conventions.
3. `docs/superpowers/reply_to_open_questions.txt` — user's Phase-2b answers.
4. `docs/superpowers/plans/2026-05-20-phase2a-refactor-fake-spectra.md` — the
   executed refactor plan.
5. `hcd_analysis/priya_p1d.py` + `scripts/build_emulator_cache_tau0.py`.

Clustering-track follow-ups (§B/C/D/E in `SESSION_HANDOVER_2026_04_28.md`)
remain untouched on `main`.
