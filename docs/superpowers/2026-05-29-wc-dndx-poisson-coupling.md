# `w_c` ↔ dN/dX translation and cross-class clustering — design note

**Date:** 2026-05-29
**Branch:** `phase2-emulator-jax`
**Context:** Phase-2b emulator brainstorm. How the per-class sightline weights
`w_c` (which weight the unfiltered per-class P1D in the likelihood
reconstruction `P_obs = Σ_c w_c·A_c·P_c`) connect to the per-class incidence
dN/dX, and how to treat absorber clustering between N_HI classes.
**Supporting scripts:** `scripts/diag_wc_from_dndx.py`,
`scripts/diag_crossclass_coupling.py` (+ τ₀-response demos
`scripts/diag_tierc_tau0_response.py`, `scripts/diag_tierc_tau0_ratio.py`).

---

## Decision

- **Head A emits `dN/dX_c(θ, z)`** (per class) **and the CDDF `f_nhi`** —
  the τ₀-invariant physical predictions. It does **not** emit `w_c` as a
  likelihood input (an optional `w_c` output may be kept as a training
  diagnostic only).
- **`w_c` is DERIVED from dN/dX** via the diagonal telescoping-Poisson map
  `M₀` below, plus a small **calibrated, z-dependent correction `δ_c(z)`**
  measured from the sim catalogs.
- **Likelihood nuisances for incidence = a PW14-style power law**
  `dN/dX_c(z) = A_c·(1+z)^{γ_c}` per class, with `(A_c, γ_c)` marginalised
  under observationally-anchored priors (pull the exact PW14 form + fiducial
  values/widths from the paper / `sbird/dla_data` — do NOT invent numbers).
  This summarises the z-evolution, is low-dimensional, and gives the low-z
  clustering scatter a physical place to live.

This is the dN/dX analogue of the per-class P1D amplitude nuisances `A_c`.

## The math (diagonal map `M₀`)

Classes ordered by N_HI: clean < LLS < subDLA < DLA. A sightline's class is its
**highest**-N_HI absorber. dN/dX counts **absorbers** (all of them, including
those hidden behind a higher class); `w_c` counts **sightlines by max**.

Mean class-c absorbers per sightline:

    μ_c = (dN/dX)_c · X̄,   X̄ = X_tot / N_sl   ⇒   μ_c = N_abs,c / N_sl

Sightline max-class = c ⟺ ≥1 class-c absorber AND zero of any higher class:

    w_DLA    = 1 − e^{−μ_DLA}
    w_subDLA = (1 − e^{−μ_subDLA})·e^{−μ_DLA}
    w_LLS    = (1 − e^{−μ_LLS})·e^{−(μ_subDLA + μ_DLA)}
    w_clean  = e^{−(μ_LLS + μ_subDLA + μ_DLA)}

These sum to 1 exactly (telescoping). The `(1 − e^{−μ})` factor captures
absorber **multiplicity** (e.g. LLS sightlines average ~1.25 LLS absorbers; a
naive `w ∝ dN/dX` is ~25% off, the Poisson form ~1%).

**Boundary consistency requirement:** the dN/dX class edges feeding `M₀` MUST
match the Tier-C edges (17.2 / 19.0 / 20.3) so the counts defining `w_c` and the
counts feeding the power law are the same partition. Verify `meta["n_absorbers"]`
class keys use these edges (the diag scripts recompute from the catalog with the
Tier-C edges to guarantee it).

## Verification (single point, full sightlines)

`diag_wc_from_dndx.py`, ns0.803 snap 17, z=3.0, full 691,200 sightlines:

| class | dN/dX | μ | w_direct | w_poisson | reldiff |
|---|---|---|---|---|---|
| clean | 0 | 0 | 0.72520 | 0.72121 | −0.55% |
| LLS | 0.360 | 0.231 | 0.18480 | 0.18715 | +1.27% |
| subDLA | 0.100 | 0.064 | 0.05925 | 0.06027 | +1.73% |
| DLA | 0.050 | 0.032 | 0.03075 | 0.03136 | +2.00% |

Both sum to 1.000000; diagonal Poisson reproduces the counted fractions to ≤2%.

## Cross-class clustering (z-resolved)

`diag_crossclass_coupling.py`, ns0.803, all 18 snaps (z = 2.0–5.4). The
independent-Poisson `w_c` is accurate to **≤2.5% across the whole range**,
degrading smoothly and monotonically toward low z (≈0.5% at z=5.4 → ≈2.5% at
z=2.0), same sign for all classes (Poisson slightly over-predicts the HCD
fractions).

Cross-class co-occurrence `ξ = P(A&B)/[P(A)P(B)]` (>1 = clustered):

| z | ξ(subDLA·DLA) | ξ(LLS·DLA) | ξ(LLS·subDLA) |
|---|---|---|---|
| 5.4 | 0.99 | 0.99 | 0.99 |
| 3.0 | 1.15 | 1.05 | 1.06 |
| 2.0 | 1.56 | 1.25 | 1.29 |

Clustering is **real and grows toward low z** (up to +56% at z=2), confirming
that a max-class sightline is a mixture of classes. **But it acts on rare joint
events** (`P(subDLA)·P(DLA) ≈ 3×10⁻⁴` at z=2), so the absolute effect on `w_c`
is only the ~2.5% residual above. The "hidden" fraction (a class's absorbers
sitting behind a higher class) reaches ~29% for LLS at high z, yet `w_LLS` is
still predicted to 0.5% because the telescoping `e^{−(μ_sub+μ_DLA)}` factor
already absorbs the independent part; only the clustering excess `(ξ−1)` leaks
through.

## Why a full mixing matrix `M` is NOT needed

The off-diagonal (clustering) coupling is genuine but small in absolute `w_c`
terms (≤2.5%, on rare events). The residual is a **smooth, monotonic, measured
systematic** — not noise. So:

- Use the **diagonal telescoping-Poisson `M₀`** as the production map.
- **Calibrate** the residual `δ_c(z)` from the sim catalogs (fit smoothly in z;
  allow weak cosmology dependence) → corrected map unbiased to <0.5%.
- Carry a **small z-dependent prior width** (~0.5% high-z → ~1% low-z) on the
  `w_c` consistency to cover residual cosmology dependence.

Conservative (clustering is measured and accounted for) without the complexity
and extra parameters of a learned 3-input cross-class matrix.

## Implementation TODO (for the Phase-2b plan)

1. Pull the exact PW14 `dN/dX_c(z)` functional form + fiducial `(A_c, γ_c)` and
   prior widths from the paper / `sbird/dla_data` (no invented numbers).
2. Confirm `meta["n_absorbers"]` class boundaries == Tier-C edges
   (17.2/19.0/20.3); reconcile if not.
3. Promote `diag_crossclass_coupling.py` into the `δ_c(z)` calibration step;
   pin `M₀` + the telescoping sum-to-1 identity with a unit test.
4. Pin the dN/dX→`w_c` round-trip (build cache `w_c` vs `M₀`-derived `w_c`)
   to ≤2.5% as a regression test.
