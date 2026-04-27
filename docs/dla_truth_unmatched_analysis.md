# Why is τ-vs-colden completeness only 86 %?

Follow-up analysis on PR #6 (`docs/dla_truth_validation.md`).  Goal: identify
the failure modes of the τ-peak-finder vs particle-based-colden truth
comparison, and decide whether the 86 % completeness / 85 % purity numbers
hide a fixable bug or reflect a fundamental limitation.

**Answer**: 91 % of the unmatched cases are explained by a single physical
effect — a real-space-vs-redshift-space mismatch between fake_spectra's
`colden` and `tau` outputs.  Increasing the matching tolerance from
~100 km/s (= `merge_dv_kms`) to ~300 km/s (the typical halo peculiar
velocity) cleanly resolves it: completeness rises to **97.0 %**, purity to
**95.8 %**, with mean Δlog NHI and σ Δlog NHI both unchanged.  The fix is
in the matcher, not in the τ-peak finder — the production catalog itself
is unbiased (PR #6's H1 / H2 still pass).

---

## 1. Failure-mode breakdown

`scripts/diagnose_dla_truth_unmatched.py` re-runs the validation on all 10
(sim, snap) pairs, classifying every unmatched case.

### Unmatched truth (loose match, 1136 / 8340 = 13.6 %)

| Mode | Count | Description |
|------|------:|-------------|
| **T1** | 0      | No τ system on the sightline at all |
| **T2** | 1027 (12.3 %) | Nearest τ system exists; truth peak falls outside its (span ± tol) |
| **T3** | 109 (1.3 %) | Nearest τ system was already 1-to-1 paired with another truth |

### Unmatched recovered DLA (strict purity, 1240 / 8390 = 14.8 %)

| Mode | Count | Description |
|------|------:|-------------|
| **R1** | 1157 (13.8 %) | Integrated colden over R's span < 2 × 10²⁰ — strong subDLA flipped to DLA by NHI scatter |
| **R2** | 83 (1.0 %) | Integrated colden ≥ 2 × 10²⁰ but no truth peak in (span ± tol) |
| **R3** | 0 | 1-to-1 collision on the recovered side |
| **R4** | 0 | True τ artefact (no colden DLA anywhere near) |

The dominant mode on each side is one specific issue.  Everything else is
sub-percent.

---

## 2. T2 geometry: the real-space vs redshift-space mismatch

`scripts/diagnose_t2_geometry.py` reports for each T2 case:

* truth-peak to nearest-R-edge distance: median **154 px** (~ 154 km/s at
  HiRes 1 km/s/pixel), p25 = 118, p75 = 217, max = 14 450
* span overlap (truth.span ∩ R.span): **median 0 — 91 % have ZERO overlap**
* truth widths: median **4 px** (the truth-DLA finder uses `pixel_floor =
  1 × 10¹⁷`, which only captures the saturated colden core)
* recovered widths: median **295 px** (τ > 100 covers the full damped wing)
* truth NHI: median 10²⁰·⁵⁹ — these are real DLAs, not noise
* recovered NHI of the nearest R: median 10²⁰·⁶² — also a real DLA

Cumulative T2 hit rate as a function of position tolerance:

| tol (px = km/s at HiRes) | T2 cases captured | % of T2 |
|---:|---:|---:|
| 10 | 104 | 9.2 % |
| 50 | 106 | 9.3 % |
| 100 | 107 | 9.4 % |
| **200** | **803** | **70.7 %** |
| **300** | **954** | **84.0 %** |
| 400 | 1065 | 93.8 % |
| 500 | 1114 | 98.1 % |

The cumulative rate has a sharp jump between 100 and 200 km/s and a softer
roll-off out to ~500 km/s.  This is the signature of **peculiar
velocities**.

### Direct visual confirmation

`figures/analysis/05_truth_validation/t2_spotcheck.png` plots `colden(pix)`
and `tau(pix)` for five T2 sightlines spanning 100–450 px offsets.  Every
panel shows a single, clean colden peak (truth) and a single, clean
saturated τ DLA core (recovered) at offsets of 100–450 km/s.  Both are
real, both are at DLA strength, and they describe the **same physical
absorber** — but the absorber's redshift-space position (where τ
saturates) is shifted by the line-of-sight peculiar velocity from its
real-space position (where colden integrates).

### Why colden and τ live in different frames

`fake_spectra` deposits each particle's HI mass into the LOS bin
corresponding to its **real-space** position when computing `colden/H/1`,
but uses **redshift-space** velocity coordinates (real LOS position +
peculiar velocity / H(z)) when convolving the line profile to compute
`tau/H/1/1215`.  An absorber at real-space pixel `x_r` infalling at
peculiar velocity `v_pec` shows up in τ at pixel `x_r − v_pec / dv_pix`.
For galaxy halos hosting DLAs, |v_pec| ~ 100–300 km/s is typical, which
is exactly the offset distribution we measure.

The `velocity` and `density_weight_density` groups in
`rand_spectra_DLA.hdf5` are **empty** (`group: []`), so we cannot directly
recompute colden in redshift space — the velocity field would have to come
from a re-run of fake_spectra with `--save-velocity` or equivalent.

---

## 3. R1: NHI scatter at the 20.3 boundary

R1 (1157 cases) is recovered systems labelled DLA whose colden integration
inside the recovered span yields NHI < 2 × 10²⁰.  The H2 measurement
(σ Δlog NHI ≈ 0.062 dex) tells us how much scatter we have in the τ → NHI
inversion, and at the log NHI = 20.3 classification boundary an `(0.062)`
sigma scatter is enough to flip ~10 % of borderline-DLA systems either
way.  We measured the recovered class on the **τ-derived NHI** (production
behaviour), so any positive scatter pushes a 20.25 → 20.31 system across
the boundary — the corresponding negative scatter on the truth side keeps
it at log NHI 20.25 in the colden integration.

This is a **classification boundary effect, not a bias**.  The scatter
itself is symmetric and covers both directions: an equal number of true
DLAs in the colden ought to have been classified as subDLAs by the
τ-finder, which would show up as missed completeness and is captured in
the loose vs strict completeness gap (we observe loose 86.4 % vs strict
85.7 %, so the boundary effect explains a small slice).

The R1 fraction shrinks from 13.8 % to 1.4 % as we relax the position
tolerance to 300 km/s — confirming most R1 cases are also peculiar-velocity
mismatches in disguise (the recovered τ-span fell on a region where colden
is locally below threshold because the colden peak is shifted, but the
**total integrated** colden across the larger physical absorber would
still be above 2 × 10²⁰).

---

## 4. The fix

Change the matcher's position tolerance from
`tol_pixels = max(merge_dv_kms / dv_pix, 5)` (≈ 100 km/s) to
`tol_pixels = max(300 km/s / dv_pix, 5)`.

Physical justification: 300 km/s is the typical scale of halo peculiar
velocities at z = 2–4 in cosmological simulations, and is well below
typical inter-DLA-on-a-sightline spacing (~ several Mpc/h, ≳ 1000 km/s
LOS).  We checked the spacing distribution: < 0.5 % of sightlines have
two truth-DLAs within 500 km/s of each other, so the new tolerance is
unlikely to introduce spurious cross-matches.

### Tolerance sweep results

`scripts/test_relaxed_tolerance.py` sweeps tol from 10 to 1500 km/s on
all 10 (sim, snap) pairs:

| tol_kms | completeness (loose) | completeness (strict) | purity (strict) | mean Δlog NHI | σ Δlog NHI |
|--------:|--------:|--------:|--------:|--------:|--------:|
|  10  | 0.6905 | 0.6867 | 0.6826 | +0.0094 | 0.0701 |
|  30  | 0.7411 | 0.7368 | 0.7324 | +0.0090 | 0.0684 |
|  50  | 0.7823 | 0.7776 | 0.7729 | +0.0087 | 0.0667 |
| 100  | 0.8638 | 0.8573 | 0.8522 | +0.0081 | 0.0654 |
| 200  | 0.9464 | 0.9405 | 0.9349 | +0.0079 | 0.0622 |
| **300** | **0.9698** | **0.9641** | **0.9584** | **+0.0077** | **0.0611** |
| 500  | 0.9801 | 0.9749 | 0.9691 | +0.0077 | 0.0608 |
| 800  | 0.9811 | 0.9760 | 0.9702 | +0.0077 | 0.0608 |
| 1500 | 0.9818 | 0.9760 | 0.9702 | +0.0077 | 0.0608 |

**At 300 km/s**: completeness 97.0 %, purity 95.8 %, mean Δlog NHI +0.008
dex, σ 0.061 dex.  The bias and scatter are essentially unchanged from the
100 km/s baseline, confirming the additional matches are physical (correct
NHI), not spurious.  Completeness saturates around 97–98 % beyond
300 km/s; the residual 2–3 % unmatched are truly rare (multi-component
truth-DLAs split differently by the two finders, or extreme peculiar
velocities in compact groups).

### What the fix does NOT change

* **The production catalog is unaffected.** The fix is in the validation
  matcher only; the τ-peak finder (`hcd_analysis.catalog.find_systems_in_skewer`)
  and NHI estimator (`voigt_utils.nhi_from_tau_fast`) keep their current
  behaviour. The bias H1 (+0.009 dex) and scatter H2 (0.062 dex) measured
  in PR #6 are unchanged — the issue was never in the τ-finder, it was
  in the matcher's expectation that colden and τ peaks would coincide
  exactly.
* **Spurious matches.** Mean Δlog NHI and σ Δlog NHI are flat across all
  tolerances above 30 km/s, so the new matches at 100–300 km/s offsets
  are NOT noise (they have correct NHI). If they were unrelated
  absorbers, σ would balloon as tolerance grows. It does not.

---

## 5. Could we do better?

**True fix**: bin colden in redshift space.  Rerun `fake_spectra` with the
velocity field saved per pixel, shift each colden bin by its
`v_pec / dv_pix`, then both `tau` and `colden` would live in the same
frame and the matcher could use the original 5–10 px tolerance.

This is **not pursued here** because (a) the velocity field is not in our
existing `rand_spectra_DLA.hdf5` files (`velocity` group is empty), and
(b) the relaxed-tolerance fix already exceeds the user's > 90 % target on
both completeness and purity. We document it as the principled
long-term option in case a future re-run becomes available.

---

## 6. Updated hypothesis tests (replacement for `tests/validate_dla_truth_hypothesis.py`)

| H | Threshold | At 100 km/s tol (PR #6) | At 300 km/s tol (this fix) |
|---|---|---|---|
| H1 (no bias) | ≤ 0.05 dex | +0.0089 dex ✅ | +0.0077 dex ✅ |
| H2 (scatter) | < 0.15 dex | 0.062 dex ✅ | 0.061 dex ✅ |
| **H3 (completeness)** | **≥ 0.80** | **0.865 ✅** | **0.970 ✅** |
| **H4 (purity)** | **≥ 0.70** | **0.842 ✅** | **0.958 ✅** |

All four still pass at the original 100 km/s tolerance; the 300 km/s
tolerance moves H3/H4 from "marginal pass" to "comfortable pass" by
correctly attributing real DLAs to the τ-finder's recoveries.

---

## 7. Action items

1. Update `match_dla_lists` to expose tolerance in km/s (clearer than
   pixels, since dv_pix differs per snap).
2. Update `scripts/validate_dla_truth.py` and
   `tests/validate_dla_truth_hypothesis.py` to use 300 km/s.
3. Re-run the validation; replace the H3/H4 numbers in
   `docs/dla_truth_validation.md` with the new values + the rationale.
4. Add a regression test in `tests/test_dla_truth.py` that asserts the
   matcher recovers a planted peculiar-velocity-shifted DLA at
   ~ 200 km/s offset (synthetic, deterministic).
