# Session handover — 2026-06-02

**Branch:** `phase2-emulator-jax` (open PR #10 vs `main`). **Thread:** HCD emulator — Phase-2b.
**Supersedes:** `docs/SESSION_HANDOVER_2026_06_01.md`.

**STATUS:** Production τ₀ cache **MERGED + validated**; the Phase-2b emulator (model + loss +
likelihood + coupling + training) is **code-complete and refereed**; a single-fold training
**profiling run is in progress on CPU** (GPU is policy-blocked — see §4). Remaining: review the
profile → full k-fold sweep; then the Phase-C physics gates (§5) before any cosmology.

---

## 1. Cache — DONE
- **Shard 44 re-paired + rebuilt** (ns0.907 snap_17→SPECTRA_018; commit `4304001`); all 90 LF + 9 HR shards present.
- **Merged** (`scripts/merge_tau0_cache.py`, guards passed): `hcd_analysis/_emulator_data/observables_tau0_lf.h5` (21440 rows / 1072 pairs / n_k=172, 956 MB) + `_hr.h5` (2060 / 103 / n_k=525, 278 MB).
- **Validated** via the emulator loader: structural identity `Σ_c w_c·P_filt == P_tier_p` to **1.1e-15** (LF) / 8.9e-16 (HR); 0 NaN; **100% in-domain** (unit-cube param norm); τ₀∈[0.08,2.68], z=2.0–5.4.

## 2. Phase-B emulator — CODE-COMPLETE (all refereed; ~57 tests green)
Package `hcd_analysis/emulator/`: `data.py` (loader, 15→4 collapse, τ₀, masks, 1/n_c, **A1 make_batch
target-transform + train-split norm**, unit-cube param norm, LOSO/τ₀ splits), `model.py` (Encoder +
τ₀-invariant Head A + Head B + **learned low-rank P_filt bottleneck** `n_basis` w/ SVD warm-start +
structural_tier_p + NaN-safe masked `joint_loss`), `dndx_wc.py` (M₀ telescoping-Poisson + **calibrated
δ_c(z) clamped** + `w_c_corrected` + **`alpha_to_dndx` M₀-inverse** + PW14 `dndx_powerlaw` + prior
widths `DELTA_C_RESID_STD`), `likelihood.py` (**single-α_c** `total_p1d_difference` + ratio toggle +
`assemble_covariance` w/ two error channels + off-diagonal), `train.py` (`train_fold`, `train_step`,
checkpoint, `aggregate_error_vector`). CLI `scripts/train_emulator.py` + SLURM `scripts/batch_train_emulator.sh`.
**Every JAX change went implementer → CS+JAX referee → fix-loop** (traps #5–#13 in `docs/superpowers/jax-traps-log.md`).

### Key design decisions (all literature-backed, this session)
- **HCD likelihood reparametrized to single per-class `α_c`** (= effective residual post-masking
  incidence; was a wrong `w_c·A_c` double-amplitude). `P_obs = P_tier_p + Σ_c α_c·Δ_c`. `α_c` posterior
  → per-class effective dN/dX via the analytic `alpha_to_dndx`. `α_c(z)` low-order (DESI 2-node / PW14).
  Field-standard (Rogers&Bird 2018, DESI DR1, PRIYA 2025). Docs: `2026-06-01-hcd-marginalization-literature.md`.
- **Architecture = raw per-k NN + optional learned low-rank bottleneck; NO PCA/poly/GP** (user decision,
  given 60×20 effective rows + NN-dominated broader field). Doc: `2026-06-02-emulator-architecture-literature.md`.
- **Input params unit-cube normalized** to PRIYA's `emulator_params.json` design box (all 60 sims in-domain).

### Reviews this session (docs/superpowers/)
- Architecture: CS/ML + cosmology reviews → `2026-06-02-architecture-review-findings.md` (the must-fix list).
- The "immediate batch" from those reviews is DONE + refereed: **A1** transforms (`8bb199f`,`c1b2715`),
  **Task 2** bottleneck (`6d6f745`,`70ffcc6`), **Task 3** α_c+inverse (`518d76d`,`960c5b2`),
  **A2/A3/A4/A4b/covariance** (`71b7545`). Task 13 training (`bc6a456`).

## 3. ⚠️ CRITICAL envs (TWO, kept separate)
- **emu-jax** (the emulator — pure JAX): `PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya
  /home/mfho/.conda/envs/emu-jax/bin/python3` — **`PYTHONNOUSERSITE=1` MANDATORY** (else stale
  numpy/h5py from ~/.local). jax 0.10.1, x64 on. pytest in-env. See `2026-06-01-emu-jax-env.md`.
- **emu-3.9** (cache build / fake_spectra / priya_p1d / merge): `export LD_LIBRARY_PATH=/sw/pkgs/arc/
  stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH` + that python. No conda activate.

## 4. ⚠️ GPU is POLICY-BLOCKED on cavestru0 — train on CPU
`sbatch -p gpu` under cavestru0 → **`AssocGrpBillingMinutes`** rejection: the gpu partition bills
`GRES/gpu=27391/min`, so any GPU job exceeds the account's billing-minutes cap (RawUsage=0 — a hard
policy ceiling, not used-up budget). Reducing time/cores/mem doesn't help (GPU weight dominates).
**The model is small (~400k params/21k rows) → CPU is fast** (~6 s/epoch; 30-epoch profile ≈ 3 min;
full 8-fold sweep ≈ 1–2 h). Run training on CPU: the allocated OnDemand node (free) for a profile, or
a `standard`-partition CPU job under **yueyingn** (the proven build account) for the sweep. Pursue a
real GPU allocation separately only if scaling demands it (it doesn't for this model).

## 5. NEXT — resume order
1. **Review the profiling run** (`checkpoints/emu_fold0_profile*`, `figures/analysis/04_emulator/*fold0*`):
   per-epoch time, train/val loss, pred-vs-true. Decide batch/epochs/patience for the full sweep.
2. **Full k-fold LOSO sweep** (8 folds, CPU via yueyingn/standard) → `aggregate_error_vector` → the static
   per-(class,k,z) emulator-error vector (Task 14). Surface all fold figures (user wants intermediate figures).
3. **Validation battery** (Task 15): reframed τ₀-response, cross-class additivity, DLA-shape clustering, the
   **Δ_c → Rogers-kernel closure gate** (cosmology review C3).
4. **Phase-C physics gates BEFORE trusting cosmology** (`2026-06-02-architecture-review-findings.md` §C):
   C1 damping-wing covariance bound; **C2 baseline-masking match** (sim `tau_thresh=1e6` clips more than a
   ~70%-complete data finder → α_c may be O(1); needs the target-survey DLA completeness); C4 marginalize
   `w_c` (δ_c residual) in `P_tier_p`; C5 LF→HF resolution in the covariance; M1 metals/continuum k-mask.
5. **Join CDDF (`f_nhi`) + an external LLS-incidence prior** to break the HCD/LLS↔cosmology degeneracy
   (DESI's explicit recommendation; emulator is one step away — Head A already emits f_nhi/dN/dX).
6. **Task 7 external dN/dX priors** (PW14/Crighton from `sbird/dla_data`) — deferred; the FORM is wired
   (`dndx_powerlaw`), only the sourced fiducials/widths remain.

## 6. Reading order
1. This file. 2. `docs/superpowers/2026-06-02-architecture-review-findings.md` (the must-fix list / §C gates).
3. `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md` (§6 revised to single-α_c).
4. `docs/superpowers/2026-06-01-hcd-marginalization-literature.md` + `2026-06-02-emulator-architecture-literature.md`.
5. `docs/superpowers/plans/2026-06-01-phase2b-B-equinox-emulator.md` (the executed plan).
6. `docs/superpowers/jax-traps-log.md` (13 traps). Memory: [[phase2-hr-phase1-and-bins]], [[cavestru0-compute-budget]],
   [[feedback-jax-review-and-traps]], [[feedback-surface-training-figures]].
