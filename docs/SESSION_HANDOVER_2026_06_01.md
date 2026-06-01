# Session handover — 2026-06-01

**Branch:** `phase2-emulator-jax` (open PR #10 vs `main`), HEAD **`1469be0`**, in sync with origin pending push.
**Thread:** HCD emulator — Phase-2b. This session: recovered status after a disconnect,
fixed + re-launched the one failed production shard, ran a science-referee faithfulness
audit, wrote the Phase-B implementation plan, and executed its first 6 tasks (the data +
coupling foundation) via subagent-driven development.
**Supersedes:** `docs/SESSION_HANDOVER_2026_05_30.md` (production build launched; shard-44 follow-up).

**STATUS:** Production τ₀ cache build **COMPLETE except LF shard 44**, which is now
**re-running clean** after a fix (job `51214105_44`, ~10h to go). Phase-B (the Equinox
emulator) **plan written + 6/18 tasks done** (env, fixture, loader, transforms, splits,
w_c coupling — all tested). The immediate next actions are in §2 (merge) and §4 (continue Phase B).

---

## 1. What this session did

1. **Recovered the full status.** Two threads exist: the **emulator** (this branch, PR #10
   open, active) and **clustering** (`hcd-clustering`, PR #7 MERGED, Phase-1 gates pass —
   `b_DLA=1.67±0.54` vs Bird+2014, `b_F=−0.141`; next is the 60-sim sweep, blocked on ξ_FF
   scaling). `joint-emulator-scaffold` + the worktree branch are merged-safe-to-delete.
2. **Confirmed the production build:** 89/90 LF + 9/9 HR shards COMPLETE; only LF shard 44 FAILED.
3. **Fixed shard 44 (ns0.907 ladder re-pair)** — commit **`4304001`**. Root cause:
   `locate_raw_tau_file` mapped Phase-1 snap→raw `SPECTRA_<snap>` 1:1, but ns0.907's ladder
   is offset (gaps at snap 15/18): its snap_17 is the z=2.8 snapshot, whose grid_480 tau lives
   in raw **SPECTRA_018** (SPECTRA_017 is a z=3.0 grid). Added a documented
   `_RAW_SPECTRA_DIR_OVERRIDE {(ns0.907,17):18}` + `_raw_spectra_index` helper + 2 regression
   tests. Verified SPECTRA_018 header z=2.8000. Re-launched **job `51214105_44`**; it is now
   running **past the 2:10:37 mark** where the prior attempt failed → re-pair confirmed.
   Keeps the full 1072 pairs / 21440 rows (merge guards unchanged).
4. **Science-referee faithfulness audit** (2 skeptical agents). Core claim **verified on real
   shards**: `Σ_c (counts/N)·P_tier_c_filtered == P_tier_p` to **0.000e+00** (LF 000/040, HR 000),
   zero NaNs. Triage: no BROKEN findings. Genuine items folded into the plan (δ_c across ≥3 sims;
   single-source class edges = Task 17; reframed τ₀-response). The "unfiltered should reconstruct
   Tier P" flag was a misreading (unfiltered feeds the HCD add-back delta, not the total).
5. **Wrote the Phase-B plan** — `docs/superpowers/plans/2026-06-01-phase2b-B-equinox-emulator.md`
   (18 TDD tasks). Commit **`6a663bc`**.
6. **Executed Phase-B Tasks 0–5 + x64 config** (subagent-driven, each spec+quality reviewed).
   See §4. Review caught a real plan defect (empty-class→NaN) and the float32/64 trap.

---

## 2. ⚠️ Cache merge — gated on shard 44, then run

1. **Watch shard 44:** `squeue -j 51214105` (empty = done); confirm `sacct -j 51214105 -X` COMPLETED
   and the file appears: `/scratch/cavestru_root/cavestru0/mfho/tau0_shards/observables_tau0_lf.shard044.h5`.
2. **Merge with completeness guards:**
   ```
   merge_tau0_cache.py --shards '/scratch/.../tau0_shards/observables_tau0_lf.shard*.h5' \
       --output hcd_analysis/_emulator_data/observables_tau0_lf.h5 --expect-pairs 1072 --expect-rows 21440
   # HR: --expect-pairs 103 --expect-rows 2060 -> observables_tau0_hr.h5
   ```
3. **Validation anchors** (emu-3.9 env): `tests/test_emulator_cache_tau0.py`; on the merged cache
   confirm Tier-P bit-identity vs PRIYA, HR 1.27e-6, per-row `P_tier_p == Σ w_c·P_filt`.

---

## 3. ⚠️ CRITICAL envs (two, kept separate)

- **Cache build / fake_spectra / anything importing `hcd_analysis.priya_p1d`** (emu-3.9):
  ```
  export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-3.9/bin/python3 <script>
  ```
  Do NOT `conda activate`. SKIP/ImportError is NOT a pass.
- **Phase-B emulator (NEW, emu-jax)** — pure JAX, never imports fake_spectra:
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest <test> -v
  ```
  ⚠️ **`PYTHONNOUSERSITE=1` is MANDATORY** — else stale numpy<2/h5py leak from `~/.local`.
  jax 0.10.1 (cuda12)/equinox 0.13.8/optax 0.2.8/numpy 2.4.6; pytest 9.0.3 in-env. x64 enabled
  package-wide. GPU training (Task 13) runs under SLURM **account cavestru0, partition gpu**
  (tight budget — profile first). See `docs/superpowers/2026-06-01-emu-jax-env.md`.
- **Discovery only (no fake_spectra):** `/sw/pkgs/arc/mamba/py3.11/bin/python3`.

---

## 4. Phase-B execution state (plan: 2026-06-01-phase2b-B-equinox-emulator.md)

**DONE (8 reviewed commits, HEAD `1469be0`), all tests green (9 emulator tests):**
- **Task 0** — emu-jax GPU env + `hcd_analysis/emulator/__init__.py` (`6a54eb2`); x64 package-wide (`1469be0`).
- **Task 1** — synthetic v3.3 fixture `tests/emulator/_fixture.py` (`7dd5dfe`).
- **Task 2** — loader `hcd_analysis/emulator/data.py` (15→4 collapse, τ₀, Nyquist masks, 1/n_c) (`31759ba`);
  fix: empty-class→0 contract locked (`6a2e246`) — empty classes are 0, not NaN, else the structural sum is poisoned.
- **Task 3** — transforms (arcsinh signed-log) + train-split normalisation (`9456897`).
- **Task 4** — k-fold LOSO + τ₀-edge holdout splits (`8d3ed07`).
- **Task 5** — `hcd_analysis/emulator/dndx_wc.py` M₀ telescoping-Poisson w_c (`6baf995`).

**NEXT — three groups:**
- **Model/loss core, TDD-able NOW (no external inputs):** Task 8 (encoder + Head A), 9 (Head B + structural
  P_tier_p + sign-safe Δ), 10 (NaN-safe masked loss + finite-grad), 11 (joint loss: 1/n_c, per-element terms,
  α-multiplicity, clean mean-F), 12 (Head-A τ₀-invariance grad test), 16 (likelihood: difference form +
  covariance), 17 (single-source class-edge hardening assertion).
- **Need real scientific inputs (DO NOT fabricate — spec: "no invented numbers"):** Task 6 = run
  `scripts/calibrate_delta_c.py` (TBD) against real catalogs in emu-3.9 to get the frozen δ_c(z) coeffs across
  ≥3 sims; Task 7 = pull PW14 (DLA) / Crighton+2015 (subDLA) / LLS-turnover incidence forms + fiducials/widths
  from `sbird/dla_data` or the papers. **Open: is `sbird/dla_data` importable; where are the catalogs?**
- **Gated on the cache merge (§2):** Task 13 (optax train + checkpoint), 14 (k-fold LOSO error vector +
  DLA high-k shot flags), 15 (validation: reframed τ₀-response, cross-class additivity, DLA-shape clustering).

**Execution mode:** subagent-driven (fresh implementer per task → spec review → code-quality review → loop).
Trivial tasks controller-reviewed; substantive ones get a review subagent.

---

## 5. Reading order to resume
1. This file (§2 merge = gated; §4 = Phase-B next).
2. `docs/superpowers/plans/2026-06-01-phase2b-B-equinox-emulator.md` (the Phase-B plan being executed).
3. `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md` (the design).
4. `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md` (w_c↔dN/dX).
5. `docs/superpowers/2026-06-01-emu-jax-env.md` (the new env + PYTHONNOUSERSITE gotcha).
6. Memory: [[phase2-hr-phase1-and-bins]], [[cavestru0-compute-budget]].
