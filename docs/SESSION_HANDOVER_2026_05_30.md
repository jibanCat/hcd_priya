# Session handover — 2026-05-30

**Branch:** `phase2-emulator-jax` (open PR #10 vs `main`), HEAD **`bce73bf`**, in sync with origin.
**Thread:** HCD emulator — Phase-2b. This session: finalized the Phase-2b design,
executed Plan-A (the τ₀ cache), and **launched the production τ₀ cache build**.
**Supersedes:** `docs/SESSION_HANDOVER_2026_05_22.md` (cache v3.1 rework + cddf flip).

**STATUS:** Production τ₀ cache build **RUNNING on the cluster** (LF `51154697`,
HR `51154699`). The immediate next action is in §2: **watch the build → re-run
the one failed shard (44) after fixing an ns0.907 pairing → merge (with guards)
→ validation anchors.** Then Phase B (the emulator NN) — §5.

---

## 1. What this session did

1. **cddf.npz flip (Task 12) DONE + verified** (commit 858bc8d): job 50696155
   (all-60 LF Phase-1 re-run) landed; `discover_sim_snap_pairs`/`read_cddf` now
   read `cddf.npz`; fresh cddf == old cddf_corrected to FP precision.
2. **Phase-2b emulator design spec written + 3-reviewed + committed** (9f339bc):
   `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md`. Architecture:
   JAX/**Equinox**, single-fidelity **LF**, shared encoder [10→256→128→64] +
   Head A (f_nhi[30]+dN/dX[3], τ₀-invariant) + Head B (P_tier_p + 4 filtered
   class P1D + 3 HCD deltas, τ₀-dependent). w_c derived from dN/dX via a
   telescoping-Poisson map (see `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md`).
3. **Plan-A (the cache) executed:** `docs/superpowers/plans/2026-05-29-phase2b-A-cache-freezecore.md`.
   - Cache schema **v3.3 (uniform-rescale Tier-C)**. A per-pixel "freeze-core"
     was implemented, then **proven a numerical no-op** (`scripts/diag_tau_freeze_sensitivity.py`:
     self-shielded pixels have τ≫1 → flux saturated → frozen ≡ scaled), so
     production uses the plain uniform rescale; the `tau_freeze` knob remains a
     no-op option for a Phase-3 absorber-level wing treatment.
   - **Memory/perf optimization** (commit c65ba38): build computes Tier P as the
     count-weighted **sum of the filtered Tier-C pieces** instead of a separate
     `flux_power` call. **VERIFIED bit-identical on real full-res data: 9e-15**
     across 3 nbins, the idx19 extrapolation α, NaN-safe for empty classes
     (`scripts/verify_tierp_sum_identity.py`). HONEST scope: this is a **~30% WALL**
     cut (removes the redundant FFT); it does **NOT** reduce peak memory (still
     ~34 GB — the spike is `np.exp(-tau)` temporaries with both τ arrays alive,
     NOT flux_power, a mis-diagnosis the re-timing caught). `compute_tier_p_p1d`
     (flux_power) is KEPT unchanged as the gold reference (tests + verify script).
4. **α-count decided = 20** (`--alpha-refine 2`). Two reviewers (10-conditional,
   20-GO) + `scripts/diag_alpha_density.py` (10 α interpolates the midpoints to
   ≤0.35%). 20 chosen because the optimization made it cheap AND it enables the
   **held-out-α validation** (train PRIYA-10 / predict the 10 midpoints) across
   the whole param space. 20 α is bit-identical to PRIYA at the even-index nodes.
5. **Profiled + launched the production build** (§2).

---

## 2. ⚠️ PRODUCTION BUILD — current state + resume order

**Jobs (yueyingn0, partition standard):** LF **`51154697`** (`--array=0-89`, 90
tasks), HR **`51154699`** (`--array=0-8`, 9 tasks). Script
`scripts/batch_tau0_production.sh` (env `FIDELITY`/`SHARD_SIZE`; **20 α, 44 GB,
2 core, 12 pairs/shard, 24 h**). Shards →
`/scratch/cavestru_root/cavestru0/mfho/tau0_shards/observables_tau0_{lf,hr}.shardNNN.h5`.
**Cost ~9–10k CPU-h** (re-timed: ~1.3 h/pair @ 20 α, peak ~34 GB; Great Lakes
bills `max(cores, mem/7)·wall`). **Budget cap 20k** ([[cavestru0-compute-budget]]);
~1k already burned on a cancelled 10-α run.

**As of handoff:** 97 RUNNING, 1 COMPLETED, **1 FAILED (shard 44)**. ~4 h into
~16 h tasks. The OnDemand session expires soon, so the in-session waiter will NOT
survive — check manually.

### Resume order
1. **Check completion:** `squeue -j 51154697,51154699` (empty=done);
   `sacct -j 51154697,51154699 -n --format=JobID%18,State | grep -E '_[0-9]+ ' | sort | uniq -c`.
2. **⚠️ Fix + re-run shard 44** (the one known failure). Cause: the z-mismatch
   assertion tripped on the **single mis-paired pair ns0.907 snap_17** (Phase-1
   meta `z=2.800` but its located raw τ is `z=3.000`) — the known ns0.907
   SPECTRA-ladder quirk (gaps at snap 15/18). Only 1 of 1175 pairs. FIX: either
   correctly re-pair ns0.907 snap_17 to its true z=2.8 raw SPECTRA (dig into
   `locate_raw_tau_file`/the ns0.907 ladder), OR drop that single (sim,z) point
   (negligible — many other z=2.8 points). Then re-run:
   `sbatch --array=44 --export=ALL,FIDELITY=lf,SHARD_SIZE=12 scripts/batch_tau0_production.sh`.
   Re-run any OTHER non-COMPLETED tasks the same way.
3. **Merge (with completeness guards):**
   ```
   merge_tau0_cache.py --shards '/scratch/.../tau0_shards/observables_tau0_lf.shard*.h5' \
       --output hcd_analysis/_emulator_data/observables_tau0_lf.h5 --expect-pairs 1072 --expect-rows 21440
   # HR: --expect-pairs 103 --expect-rows 2060 -> observables_tau0_hr.h5
   ```
   The guards REFUSE an incomplete merge (catches a missing/failed shard) — so
   shard 44 must be recovered first.
4. **Validation anchors** (emu-3.9 env): run `tests/test_emulator_cache_tau0.py`;
   on the merged cache confirm Tier-P bit-identity vs PRIYA, HR 1.27e-6, the
   per-row integrity (P_tier_p == Σ w_c·P_filt). Then the **held-out-α test**
   (train an Equinox model on the PRIYA-10 even-index α, predict the 10
   odd-index midpoints) — part of Phase B/validation.

---

## 3. ⚠️ CRITICAL env (unchanged)
- **fake_spectra / tau0 build / anything importing `hcd_analysis.priya_p1d`:**
  ```
  export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-3.9/bin/python3 <script>
  ```
  Do NOT `conda activate`. A SKIP/ImportError is NOT a pass.
- **discovery only (no fake_spectra):** `/sw/pkgs/arc/mamba/py3.11/bin/python3`.
- **This 96 GB OnDemand node ran all verification/timing for FREE** (within an
  already-allocated yueyingn0 session) — good for verification, NOT the build
  (~weeks single-node + session walltime cap).

---

## 4. Key code state (committed + pushed, HEAD bce73bf)
- `scripts/build_emulator_cache_tau0.py` — v3.3 build; hot-path Tier P = Σ
  filtered pieces (drops flux_power); `--spot-check` has a cheap cached-array
  integrity assert.
- `hcd_analysis/priya_p1d.py` — `compute_tier_p_p1d` (flux_power) = GOLD REFERENCE
  (commented; keep exercised); `compute_tier_c_p1d`/`_per_class_p1d_at_scale`
  (chunked) with the `tau_freeze` no-op knob; `_apply_priya_filter`.
- `scripts/merge_tau0_cache.py` — v3.3 + `--expect-rows`/`--expect-pairs` +
  alpha_range cross-shard assert (completeness guards).
- `scripts/batch_tau0_production.sh` — the array (20 α, 44 GB, 12/shard).
- Diagnostics/verification: `verify_tierp_sum_identity.py` (flux_power vs sum,
  KEEP as regression), `diag_alpha_density.py`, `diag_tau_freeze_sensitivity.py`,
  `estimate_tau0_build_cost.py`, `diag_crossclass_coupling.py`, `diag_wc_from_dndx.py`.
- Tests green (emu-3.9): `tests/test_priya_p1d.py`, `tests/test_emulator_cache_tau0.py`.

---

## 5. After the cache: Phase B + C (own plans, not yet written)
- **Phase B — the Equinox emulator** (model + loader + masked loss + training):
  per the spec §3–§9. Loader can be TDD'd against the v3.3 schema fixture NOW
  (before the cache lands). Head B emits Tier P (structural sum) + 4 filtered
  classes + 3 HCD deltas; Head A f_nhi + dN/dX. Per-class `1/n_c` loss weights;
  NaN-safe masked loss; k-fold LOSO; held-out-α validation.
- **Phase C — likelihood** (total-P1D reconstruction, single cosmic-variance
  covariance, PW14 power-law dN/dX→w_c nuisances). Spec §5–§6.
- **Deferred (Phase 3):** absorber-level Voigt wing treatment (the documented
  τ₀-response residual); heteroscedastic head; multi-fidelity; PART UV study.

---

## 6. Reading order to resume
1. This file (§2 = the immediate gated action).
2. `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md` (the design).
3. `docs/superpowers/plans/2026-05-29-phase2b-A-cache-freezecore.md` (the cache plan).
4. `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md` (w_c↔dN/dX).
5. Memory: [[phase2-hr-phase1-and-bins]], [[cavestru0-compute-budget]].
