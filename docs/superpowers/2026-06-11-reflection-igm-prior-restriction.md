# Reflection — restricting the NUTS prior to the original PRIYA IGM box

*2026-06-11. Decision + rationale + implementation + validation for restricting the sampling
prior on the IGM parameters (herei, heref, alphaq) to the **original PRIYA box**, while keeping
n_s extended. Triggered by the Phase-4b finding that the catastrophic closure biases are pure
emulator extrapolation error at the **widened-box corners**. Companion to the Phase-4b headline
[`2026-06-10-headline-lls-prior-center-Ap.md`](2026-06-10-headline-lls-prior-center-Ap.md).*

---

## The decision, in one line

The NUTS prior now samples the IGM params **herei ≤ 4.1, heref ≥ 2.6, alphaq ≤ 2.5** (the original
Bird+2023 `coarse_grid.py` ranges) instead of the widened training box — so the sampler **cannot
roam into the design-sparse corners where the held-out emulator extrapolates badly**. **n_s is kept
extended to 1.05** (with the C_emu step-inflation above 0.995), per the standing PI decision, because
the real n_s may exceed 0.995 (eBOSS ~1.009). **No retrain** — the emulator's training box is
unchanged; we simply do not *fit* at its edges.

## Why — the Phase-4b evidence

Phase-4b stress-tested 6 IGM-parameter-extreme fiducials. Two blew the ±1σ gate badly, and the
single decisive cross-check showed the bias is **emulator-LOSO error, not HCD marginalization**:

![IGM HCD-sensitivity](../../figures/analysis/05_likelihood/p4b_igm_hcd_sensitivity.png)

- **corr(A_p, α_LLS) is flat** (−0.10…−0.26) across all stress levels → HCD nuisances are not the
  channel.
- The worst case (**QSO-slope-HIGH, A_p +3.74σ**) is reproduced almost exactly by the *standalone*
  emulator-LOSO Fisher bias **+3.45σ** — the likelihood adds nothing. It's the emulator being
  inaccurate at a held-out parameter-space **edge**.

When we locate those fiducials in parameter space, **every large-A_p fiducial sits in the widened-box
extension**, and every in-box fiducial recovers A_p in-gate:

![box restriction rationale](../../figures/analysis/05_likelihood/p4b_box_restriction_rationale.png)

| fiducial | A_p bias | n_s bias | herei | heref | alphaq | inside original PRIYA? |
|---|---|---|---|---|---|---|
| **αq↑** (alphaq_hi) | **+3.74σ** | +2.63σ | 3.71 | 2.96 | **2.98** | ❌ alphaq > 2.5 |
| **herei↑** (herei_hi) | **+2.03σ** | +1.13σ | **4.46** | 3.08 | **2.60** | ❌ herei > 4.1, alphaq > 2.5 |
| heref↓ (heref_lo) | +0.10σ | −0.79σ | 3.96 | **2.24** | **2.56** | ❌ heref < 2.6, alphaq > 2.5 |
| heref↑ (heref_hi) | +0.07σ | +1.59σ | 3.96 | 2.97 | 2.43 | ✅ |
| αq↓ (alphaq_lo) | +0.07σ | +0.06σ | 3.71 | 2.96 | 1.33 | ✅ |
| bhfb↓ (bhfb_lo) | −0.65σ | +0.28σ | 3.76 | 2.96 | 2.40 | ✅ |

The PRIYA design is a Latin hypercube that **densely fills the original box only**; the widening
(below) added just a handful of points, so the held-out emulator has no nearby training support in
the extensions and extrapolates. The closure faithfully propagates that error into cosmology.

## What "original" vs "widened" means

The cache's `data.PARAM_LIMITS` (the emulator's training box) **widened** four of PRIYA's nine
`coarse_grid.py` defaults. Restricting the sampler to the original ranges on the three IGM params
maps, in the emulator's unit cube, to:

| param | original PRIYA | widened (training) | sampling prior now | unit-cube bound |
|---|---|---|---|---|
| **ns** | [0.8, **0.995**] | [0.8, **1.05**] | **[0.8, 1.05] (KEEP extended)** | [0, 1] |
| Ap | [1.2, 2.6]e-9 | [1.2, 2.6]e-9 | unchanged | [0, 1] |
| **herei** | [3.5, **4.1**] | [3.5, **4.5**] | **[3.5, 4.1] (restrict)** | [0, **0.60**] |
| **heref** | [**2.6**, 3.2] | [**2.2**, 3.2] | **[2.6, 3.2] (restrict)** | [**0.40**, 1] |
| **alphaq** | [1.3, **2.5**] | [1.3, **3.0**] | **[1.3, 2.5] (restrict)** | [0, **0.706**] |
| hub, omegamh2, hireionz, bhfeedback | — | — | unchanged | [0, 1] |

**A_p is identical between the two boxes**, so this restriction does not directly touch the A_p
prior — it removes the *IGM* corners whose emulator error *leaks* into A_p.

## Why keep n_s extended (the one parameter we did NOT restrict)

n_s differs from the IGM params: a documented PI decision (2026-06-08) keeps n_s ∈ [0.8, 1.05] with a
**C_emu step-inflation above 0.995** rather than a hard cap, because the *real* n_s can land above the
original 0.995 ceiling (eBOSS measures ~1.009). Hard-capping n_s at 0.995 would **clip the real
posterior**. The sparse-design risk on the n_s ridge is instead absorbed by inflating the emulator
error there, not by forbidding it. So n_s keeps the full unit interval [0, 1]; only herei/heref/alphaq
are sub-intervals.

> This means the interior n_s biases that are **not** edge effects (e.g. heref↑ recovers A_p fine but
> n_s **+1.59σ**, fully inside the box) are **not** addressed by this change — those belong to the
> separate fold-LOSO n_s thread (the contiguous-block partition artifact), tracked elsewhere.

## Implementation

- `data.SAMPLING_LIMITS` (new): the sampling box — original PRIYA on herei/heref/alphaq, extended on
  n_s, identical elsewhere.
- `data.sampling_unit_bounds()` (new): maps `SAMPLING_LIMITS` into the emulator's unit cube via the
  *widened* `PARAM_LIMITS` (so the emulator normalization is untouched), returns `(lo, hi)` clipped to
  [0, 1].
- `closure_legb._THETA_UNIT_LO/_HI`: frozen module defaults from `sampling_unit_bounds()`.
- `LegBCtx.theta_unit_lo/hi` (new, default `None` → the restricted box): the model resolves them via
  `getattr(ctx, ..., None)`, so any ctx (incl. a test `SimpleNamespace`) falls back to the restricted
  default. **Set both to 0/1 to recover the old full-box prior** for a diagnostic arm.
- `_legb_model` **and** `_legb_priors_only` both now sample
  `theta_unit ~ Uniform(theta_unit_lo, theta_unit_hi)` — changed **identically** so the postprocess
  bijector stays consistent with the model.

This is the production real-fit path (DESI-only / KS-only via `_legb_model`). The separate Leg-A SBC
foundation (`inference.unit_box_logprior`) is left on the full box by design.

## Validation

- **New test** `tests/test_sampling_box_restriction.py` (3 cases): the SAMPLING_LIMITS match original
  PRIYA on the IGM params and keep n_s extended; the unit bounds are correct
  (herei [0,0.60], heref [0.40,1], alphaq [0,0.706], rest [0,1]); and a **prior-predictive** draw from
  the production model lands strictly inside the original PRIYA IGM box while n_s still reaches >0.995.
- **Both goldens unaffected:** `test_legb_golden` (forward at fixed θ — no bijector) and
  `test_legb_postprocess_golden` (fast-vs-replay reconstruction on a shared seed — both paths use the
  same new prior) **pass, no regeneration**.
- Full closure/likelihood suite: **47 passed** (`test_sampling_box_restriction`, both goldens,
  `test_legb_tau0_priya`, `test_data_likelihood`, `test_xclass_cemu`).

## Consequence + what's next

- The extended-corner Phase-4b fiducials (αq↑, herei↑, heref↓) are now **inadmissible by construction**
  — their truth lies outside the prior. That is the intended behavior: we do not claim to fit data
  whose IGM truth sits where the emulator is untrustworthy. Real DESI/KS sit in the IGM interior.
- **Still open** (separate threads): (1) the interior fold-LOSO n_s bias (heref↑ +1.59σ); (2) the
  external per-survey LLS-abundance pin + the HCD z-slope pin (in progress — DESI ≈ cosmic-average,
  KODIAQ-SQUAD ≈ 2–3× selection-biased per arXiv:2509.18271 §4.3.3).
