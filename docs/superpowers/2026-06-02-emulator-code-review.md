# Phase-2b emulator — final code review (CS/ML + correctness/physics)

**Date:** 2026-06-02. Two parallel read-only reviews of the as-built code
(`hcd_analysis/emulator/{data,model,dndx_wc,likelihood,train}.py`, `scripts/train_emulator.py`,
`calibrate_delta_c.py`, tests). **Verdict: NO critical bugs in either review** — the pipeline is
correct as built (70–79 emulator tests green; `test_emulator_cache_tau0.py` is an emu-3.9 build-env
test, expected to error under emu-jax). Findings below are grouped by **when to act**.

## Confirmed strengths (don't redo)
NaN-safe masked loss + adversarial tests (inf/fully-masked/zero-weight/Hessian-diag); strict
train-only normalization (`test_target_norm_no_val_leak`); structural Head-A τ₀-invariance (exact-zero
Jacobian); A1 transform round-trip exact to ~1e-10 with the standardized-log↔linear boundary respected
(structural_tier_p fed `exp(P_filt)`); the telescoping-Poisson inverse re-derived as the EXACT inverse
of `w_c_from_mu` (μ to 1e-15); additivity exact (`P_obs−P_tier_p=Σα_cΔ_c`, α=0→P_tier_p bit-exact);
two-channel covariance (w_c→σ_Pfilt, α_c→σ_delta, off-diagonal preserved); δ_c coeffs byte-match the
calibration doc + clamped to [2,5.4]; static-field/serialization round-trips; the τ₀-holdout×LOSO CLI
composition excludes the holdout from both train and val.

## A. Fix BEFORE the full sweep (cheap, improves the run / closes correctness gaps)
- **CS-I2 — ragged last minibatch → extra jit retrace.** `_iter_minibatches` keeps a final partial
  batch → a 2nd leading shape → `train_step` recompiles. Fix: pad the final batch to `batch_size` with
  zero-weight rows (masked loss already supports it) → one static shape, no data loss. (Low impact at
  ~1200 train rows / 0.95 s-epoch, but clean.)
- **CS-I1 — `history["lr"]` logged at the epoch's first step**, so the LR figure is shifted vs the LR
  actually applied (optimizer owns the schedule). Diagnostic-only. Log at end-of-epoch step or the mean.
- **Test gaps (close the two the spec leans on):** (1) assert `train_fold` returns the BEST (not last)
  model after a stall; (2) assert `tau0_edge_holdout × kfold_loso` is disjoint from BOTH train and val.
  Also: dense (`n_basis=None`) checkpoint file round-trip; constant-schedule branch.
- **M2 (physics) — the f_nhi/dndx validity predicate is duplicated** in `make_batch` (loss mask) and
  `fit_target_norm` (norm mask). Same definition today; factor into one helper so they can't desync.
- **M4 — checkpoint doesn't record the k-grid / cache identity.** A checkpoint is only meaningful with
  the cache it trained on. Record `kfkms`/n_k + cache path in the `.meta.json`.

## B. Phase-C / likelihood-driver time (NOT blockers for training)
- **Physics-I1 ★ — `DELTA_C_RESID_STD` is exposed but UNCONSUMED.** The δ_c residual prior widths are
  built but nothing wires them into the α_c prior or onto `P_tier_p` via `w_c`. This is the
  structural-baseline marginalization (review gate C4) and it is **absent in code, not merely deferred**.
  Wire when the Phase-C prior/likelihood driver lands: `DELTA_C_RESID_STD` → α_c prior σ AND → a `w_c`
  marginalization that moves the dominant baseline (a ~1% w_c error is degenerate with the cosmo amplitude).
- **Physics-I2 — `alpha_to_dndx(apply_delta=True)` renorm bias is regime-dependent.** Divides out
  `(1+δ_c)` but can't invert the 4-class renorm in `w_c_corrected`; measured 2.3% (LLS) at a high-incidence
  point, not the docstring's "sub-percent." Re-document as `error ≈ HCD-fraction × δ_c`, or invert the
  renorm exactly; add a low-z/high-incidence round-trip test pinning the bound (current `rtol=0.03` is loose).
- **CS-I3 — `inv_nalpha` counts α-siblings over the WHOLE cache** (incl. val/holdout), coupling a fold's
  train weighting to what's held out. Not a leak. Decide + document: keep global (equal-per-snap-block) or
  restrict to train rows.
- **Cleanup:** drop the now-dead `mean_F_clean` from the batch (both reviews; M3) or wire a mean-F head;
  wire the `aggregate_error_vector` producer→consumer (Task 14; M2-CS); `_collapse_p1d` partial-NaN
  under-weighting (documented, bounded — comment that surviving-bin renorm was deliberately not done).

## Recommendation
The sweep is unblocked. Do the **A-list before launching** (small: padding, LR-log, 2 tests, the mask
helper, k-grid-in-meta), then run the full 8-fold CPU sweep (~15–30 min) → `aggregate_error_vector`
(Task 14) → validation (Task 15). The **B-list is the Phase-C likelihood-driver checklist** — track it
with the §C physics gates in `2026-06-02-architecture-review-findings.md`; Physics-I1 (prior-width
plumbing) is the one to not forget, since it's the C4 marginalization and is currently absent in code.
