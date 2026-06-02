# Fixing the P1D normalization — triangulated research + recommended redesign

**Date:** 2026-06-02. Three independent agents (Bayesian statistician; ML conditional-normalization/
architecture; beyond-cosmology emulator practice across climate, operator-learning, engineering-UQ,
and molecular/turbulence) converged on the **same** fix for the diagnosed problem (see
`2026-06-02-training-walkthrough.md`): the global per-k standardization buries the cosmology signal
(~0.4% of per-k log-variance) under the τ₀/z spread (~99.5%), so the MSE under-resolves θ.

## The diagnosis (agreed by all three)
**It is loss/target *misspecification*, not under-training.** Squared error on the marginally-standardized
target is the NLL of `logP | inputs ~ N(m̂, σ_k²)` with σ_k = the **marginal** (τ₀/z-dominated) std — so the
implicit precision is set by variation in quantities (z,τ₀) that are *inputs you condition on, not noise*.
The θ-signal sits at ~6% of a unit residual → the optimizer is indifferent to it. PDE/operator learning
has a name for the same thing — **spectral bias** (networks fit the dominant modes, under-resolve the rest).

## The convergent fix — three complementary pieces (all reference-free + differentiable)
Every community uses **delta/residual learning: predict (target − a conditional baseline that absorbs the
dominant predictable variation), standardize the residual to unit variance, and inverse-variance-weight the
loss.** Concretely, for us:

1. **Re-whiten by the CONDITIONAL (within-(z,τ₀)-cell) signal scale `σ_cosmo,k`** — not the marginal σ_k.
   (= GraphCast normalizing the *increment* by the std of the *difference*, not the absolute field; = the
   inverse-variance loss; your loss-reweighting idea.) Fixes the loss *budget*. Cheap, do always. **Necessary
   but not sufficient** (both stats + ML agents: a single MLP still must carve 0.4% out of 99.5%-driven features → gradient interference).
2. **In-network structured mean + zero-mean residual** — a **baseline head `m̂(z,τ₀,k)` that is BLIND to θ**
   + a **residual head `r̂(θ,z,τ₀,k)`**, output `m̂+r̂`. This is the **Kennedy–O'Hagan structured-mean GP**
   (universal kriging `M=μ(x)+Z(x)`), **Δ-learning** (Ramakrishnan+2015), **residual learning** (He+2015), and
   **GraphCast's increment** — all the same idiom, realized **in-network so the net IS the baseline** (no stored
   reference, no interpolation, one differentiable graph for HMC). θ entering *only* the residual makes the
   split structurally identifiable.
3. **FiLM / conditional modulation** (Perez+2018) — the architectural alternative/complement: `(z,τ₀)` generate
   per-feature `(γ,β)` that modulate a θ-content trunk. Operator-learning's standard way to inject a
   parameter-response distinct from a dominant input field. Reference-free, differentiable. Pairs with #2.

**(4) Δ-learning vs an ANALYTIC baseline** (cleanest *if it exists*): if PRIYA's `τ_eff` mean-flux power-law (or
a smooth parametric (z,τ₀) fit) removes most of the τ₀ variance analytically, train on `logP − log B_analytic`;
the residual head then only learns cosmology + leftover. **Only artifact-free if B is analytic/recomputed —
NOT a stored grid you interpolate** (the variant all the clean idioms specifically avoid — validates the "no
interpolation" critique).

## What ALL THREE explicitly reject
- **Stored-(z,τ₀)-reference + interpolation at inference** (my earlier option A): adds unmodeled interpolation
  error into the 99.5% term where it can *bias θ*; the clean idioms design around it. Use the in-network baseline instead.
- **Per-z models** (option B): doesn't touch the within-z τ₀ spread; loses z-borrowing-strength; non-differentiable in z.

## Recommended redesign (the consensus)
**#1 (conditional re-whitening) + #2 (in-network baseline+residual heads), FiLM (#3) optional, and check #4
(analytic τ_eff baseline) as a possible simplification of the baseline.** Train **staged**: fit the baseline head
→ freeze + fit the residual in σ_cosmo units → joint fine-tune (avoids the residual-collapses-to-0 instability).

Implementation surface (localized; architecture you have stays):
- `model.py`: split Head B's P_filt path into `m̂(z,τ₀,k)` (θ-blind) + `r̂(θ,z,τ₀,k)`, output sum; reuse the
  learned low-rank basis (give the residual its own small basis block if needed); optional FiLM trunk.
- `data.py`: compute `σ_cosmo,k` (within-(z,τ₀)-cell std) + the per-cell baseline target; re-whiten the residual
  target by `σ_cosmo,k`. No stored reference for inference (the baseline head supplies it).
- `train.py`: staged schedule (baseline → residual → joint).
- Δ_c (HCD deltas) and Head A (CDDF/dN/dX) get the same treatment if they show the same burial (check).

## ⚠️ New validation criteria (do NOT validate on marginal MSE)
Marginal MSE is already "good" for the wrong reason. The diagnostics that must flip (and become the gate):
- **Within-(z,τ₀)-cell θ-tracking correlation: 0.80 → ≳0.99**, and absolute θ-response error → **~1%**.
- **Gradient check** `∂P/∂θ` (finite-diff vs autodiff) on held-out points — HMC consumes it.
- **Residual stationarity:** held-out residuals show *no* θ-trend (unbiased emulator-error covariance `C_emu`).
- **`C_emu` = the held-out CONDITIONAL residual covariance** (self-consistent with the re-whitening), fed to the likelihood.
- **SBC / posterior-recovery** on mocks: unbiased θ-posterior, width dominated by `C_data` not `C_emu`.

## Sources (per agent)
- **Bayesian:** Kennedy & O'Hagan 2001 (structured mean + zero-mean residual; calibration/error propagation); variance-components/ANCOVA; GLS/whitening.
- **ML arch:** FiLM (1709.07871); residual learning (1512.03385); Δ-ML (1503.04987); β-NLL / inverse-variance weighting (2203.09168, 2212.09184); CosmoPower (2106.03846), LaCE (2305.19064).
- **Beyond-cosmology:** GraphCast (2212.12794: increment target + diff-std normalization + inverse-variance loss); FNO spectral-bias residual boosting (2503.13695, 2404.07200) + frequency-aware loss (2504.04260); universal kriging / KO multi-fidelity (UQLab 1709.09382; co-kriging PMC4528652); Δ-ML & RANS corrections (1503.04987; 2404.09074, 1606.07987).
