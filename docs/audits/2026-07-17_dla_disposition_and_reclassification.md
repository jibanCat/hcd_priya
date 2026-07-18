# DLA-completeness: PI disposition + reclassification (2026-07-17)

**Status of record.** Supersedes the gate LANGUAGE (not the numbers) of the 2026-07-12 DLA audit
pair. All numbers in those audits remain valid; what changes is what the campaign is CALLED and
what is treated as the production gate.

## Reclassification of the e_dla injection campaign (PI-adopted language)

The DESI DLA-completeness campaign (shards `dla_completeness_desi_*`, jobs 53297086/53297099/
53336138) is an **out-of-model, fully z-coherent covariance-envelope stress test**, not a
self-drawn 15% PRIYA DLA-completeness SBC:

- It injects DESI's shipped `syst_e_dla_completeness` column (0.15 x fitted no-masking excess
  shape x P_smooth) as a z-coherent additive mean shift into paired self-draw mocks fit with the
  deployed NORC forward. Under DESI's own z-block-diagonal systematic model this injection is
  ~1 sigma per redshift block but jointly chi2 = 10.71 (~3.3 sigma); the quoted n_s/A_p deltas
  are the as-run stress-test denomination (n_s 0.21-0.28 sigma_clean); the covariance-consistent
  1-sigma-typical figure is ~0.10 and the derived same-radius worst case is 1.36x the arm.
- Its FAIL demonstrates **leakage under that adversarial perturbation** (alpha_dla is a
  correct-sign but weak lever, ~10% whitened overlap; ~90% of the mode strands and relocates
  dominantly onto the uniform-prior tau0_amp, the tau0-A_p floor channel).
- It does **not** demonstrate failure of the deployed PRIYA DLA forward model (the 7-agent deep
  audit found every implementation correct, Outcome A), and it does **not** justify adding an
  e_dla-shaped mean nuisance (a floated e_dla amplitude would absorb the injected template by
  construction — circular; PI-rejected, do not implement on this branch).
- It is **not the production DLA-completeness RED gate** unless separately decided that this
  adversarial envelope test is a formal freeze requirement (open PI decision; until then the row
  is booked as a documented stress-test result with the Section-7 ledger-triple denomination).

## Adopted disposition (implemented this date)

**A. Displaced-truth PRIYA self-draw closure arm** — the test that determines whether the deployed
alpha_DLA model and prior can accommodate a ~15% PRIYA-shaped residual without cosmology leakage:
`apply_dla_truth_boost` (closure_legb) + `run_legb` `inject_spec={"dla_truth_boost": 1.5}`;
runner `scripts/run_dla_selfdraw_shard.py`, batch `scripts/batch_dla_selfdraw.sh` (cavestru1,
16 paired mocks, seed 20260615). Boost 1.5 puts the prior-drawn truth median at exactly 0.15 of
the OBSERVED (literature) DLA incidence (= 0.2012 of the PRIYA sim response; +0.408 latent sigma,
in-support). NO e_dla mean template. Both arms (clean refits + boosted) fit under the reduced
covariance below.

**B. Reduced DESI covariance** — modeled-in-mean => removed-from-covariance, cup1d's
`type_analysis="red"` convention (DESI DR1 cosmology paper 2601.21432 Sec 2.1): DESI floats the
PRIYA alpha_DLA mean model, so `syst_e_dla_completeness` is REMOVED from C_data as per-z rank-1
blocks (the shipped z-block-diagonal convention, never one global outer):

    C_reduced = C_fid - sum_z outer(e_dla|z)

Authority: `data_likelihood.DESI_DLA_COV_REDUCE = True` (single constant next to
`DESI_DLA_FORWARD_FRAC`; every DESI leg load inherits it with no per-driver threading; folded into
`closure_legb.forward_signature()`). Surgery: `_subtract_perz_rank1` in `_assemble_leg`, shared
with the resolution removal (one authoritative path). Resolution removal behavior unchanged; the
final DESI covariance removes both floated systematics (`syst_e_resolution` on the option-b path +
`syst_e_dla_completeness`). Guard: removing the DLA term on a leg with `dla_forward_frac == 0`
raises (KS/eBOSS untouched). Measured conditioning: the removal leaves per-z min eigenvalues
essentially unchanged (z=4.0 block 2.5032 -> 2.5032 with inflation; 1.2151 -> 1.2150 without) and
slightly improves condition numbers; **no diagonal inflation is numerically necessary for the
surgery** (`cov_diag_inflation` stays ON in the deployed default for its original chi2-calibration
purpose only). Tests: `tests/test_dla_cov_reduce.py` (fid reconstruction, bit-exact block removal,
cup1d-red agreement to <1e-11, symmetry non-increase [the shipped cov itself is asymmetric at
1.0e-8], SPD + per-z Cholesky with and without inflation, modeled-in-mean guard, opt-out
back-compat pin) + the resolution-mode pin updated in `tests/test_resolution_cup1d_consistency.py`.

This covariance change lands BEFORE freeze and Wave-2 re-SBC; all final closure and SBC fits use
the reduced covariance. Archival reproduction of pre-2026-07-17 results: `dla_cov_reduce=False`.

## Cross-references

- Full audit + mapping tables: notes repo `docs/superpowers/2026-07-17-dla-priya-selfdraw-audit.md`
  and `2026-07-17-dla-completeness-cup1d-double-count-crosscheck.md` (with the PI correction
  banner retracting the floated-e_dla direction).
- The 2026-07-12 audits (`2026-07-12_dla_*` in this directory): numbers valid, gate language
  superseded as above.
