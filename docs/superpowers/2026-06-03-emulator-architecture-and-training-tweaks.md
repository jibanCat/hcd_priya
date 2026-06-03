# Phase-2b Lyα P1D emulator — architecture refresh + the training tweaks

**Date:** 2026-06-03 · **Branch:** `phase2-emulator-jax` · **Purpose:** a single readable
account of the *current* architecture and — the part you asked for — **the non-obvious
implementation tweaks that make training easier/better**, several of which went in during
the referee/diagnostic rounds and are easy to miss. Companion to the consolidated design
(`2026-06-03-emulator-design-consolidated.md`), the τ₀ error-model note
(`2026-06-02-tau0-error-model-and-closure-design.md`), and the normalization research
(`2026-06-02-normalization-fix-research.md`).

---

## 0. TL;DR — the tweaks (what you may have overlooked)

| # | Tweak | Why it helps | Evidence |
|---|-------|--------------|----------|
| 1 | **θ-blind baseline + cosmology-residual split** (in-network Kennedy–O'Hagan) | cosmology is only ~0.4% of per-k log-var; a global σ buries it → the net "undertrains" the signal. Splitting puts the (z,τ₀) mean in a θ-blind head and the cosmology in a residual whitened by its *own* scale | within-cell θ-tracking 0.80→0.96; jacfwd(baseline)=0 exactly |
| 2 | **Per-(class,k) conditional whitening** (`σ_cosmo`, `σ_marg`, `μ_marg`, all (4,K)) | targets are ~unit-variance *per k* so the loss weights every k-bin (and the cosmology direction) evenly | σ_cosmo/σ_marg ≈ 0.077 |
| 3 | **Loss normalized by `Σ(weight·mask)`** (not the raw `inv_nc` weighting) | `inv_nc` shrank the baseline-head gradient to ~6e-3 → undertrained baseline → the k-tilt you spotted | tilt slope ±0.15 → ~0 post-fix |
| 4 | **Deeper baseline head (≥3 layers, w256)** | the 1-hidden-layer head plateaus at ~0.10·σ_cosmo regardless of width/epochs (optimization floor); cell-means are ~rank-8 so depth, not width, is the cure | (b) 0.84→0.03–0.05·σ_cosmo |
| 5 | **SVD low-rank output basis (n_basis≈12) + SVD warm-start** | output lives in a smooth 12-dim subspace → fewer params, small-data robustness, good init | SVD-12 recon 0.18–0.64% |
| 6 | **NaN-safe masked loss** (double-`where`) | above-Nyquist NaN bins never poison the gradient | finite grads w/ NaN targets |
| 7 | **Differentiable cubic-spline k-mapping** (interp the 12 basis vectors once per grid) | train on the sim grid, evaluate on *any* survey grid, fully HMC-differentiable, no retrain | spline grad vs finite-diff <1e-6 |
| 8 | **Unit-cube param normalization** (PRIYA `param_limits`) + **arcsinh/signed-log Δ** | inputs in [0,1]; sign-safe HCD-template transform | — |
| 9 | **LOSO k-fold + τ₀-edge holdout** | every sim held out once; the extrapolation probe is in τ₀-space (what HMC queries), not raw-α | — |
| 10 | **Honest validation** (θ-tracking vs the *deployed* baseline; ∂P/∂θ + Fisher-bias gate) | the within-cell corr against the val-mean was *flattered*; the inference-relevant metric is the error's projection onto ∂P/∂θ (a k-tilt biases n_s/A_p) | LF: all params <0.2σ |

Everything below expands these.

---

## 1. The core problem the architecture solves

The target is the per-class filtered P1D, `logP_filt(θ,z,τ₀,k)`, 4 absorber classes
(clean/LLS/subDLA/DLA), 9 PRIYA params `[ns,Ap,herei,heref,alphaq,hub,omegamh2,hireionz,
bhfeedback]` + z + mean-flux τ₀. **Cosmology is ~0.4% of the per-k log-variance**; the
other ~99.5% is the (z,τ₀) spread — and (z,τ₀) are *inputs*, not the signal. A naïve
global per-k standardization makes the MSE dominated by the (z,τ₀) variation, so the
network barely learns the cosmology response (classic spectral-bias undertraining). Every
tweak below exists to put the *cosmology* signal where the optimizer can see it.

## 2. The prediction path (architecture)

```
(θ, z, τ₀)──Encoder──► latent
                         ├─► HeadA      → f_NHI / dN/dX  (τ₀-invariant; → w_c)
                         ├─► BaselineHead(z,τ₀)  → m̂      (θ-BLIND; the (z,τ₀) cell-mean)
                         ├─► HeadB(latent,τ₀)    → r̂, Δ_c (cosmology residual + HCD templates)
                         └─ reconstruct:
   logP̂_filt = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂          [per-(class,k) stats]
   P_filt = exp(logP̂_filt)        →  spline W → data k-grid
   P_tier_p = Σ_c w_c·P_c^filt                            [structural, bit-exact to PRIYA]
   P_obs    = P_tier_p + Σ_{c∈HCD} α_c·Δ_c                [single per-class α_c; α_c→dN/dX]
```

- **θ-blindness is structural:** `BaselineHead` takes only `[z,τ₀]`, so `∂m̂/∂θ ≡ 0`
  (verified by jacfwd). This *forces* all θ-response — including its τ₀-modulation
  `∂²lnP/∂θ∂τ₀` — through `r̂`, and is what makes the baseline/residual split identifiable
  (otherwise it's a gauge ambiguity).
- **The θ-response read-out is exact:** `∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ`.
- **Output basis:** `P = coeffs(4,n_basis)·basis(n_basis,K)` — the encoder is a DeepONet
  *branch* emitting coefficients; the SVD basis is the (currently fixed) set of k-shapes.

## 3. The training tweaks, explained (the heart of this doc)

**(1) Baseline + residual split.** `logP = (z,τ₀)-mean + cosmology-residual`. The mean is
a θ-blind head; the residual is whitened by `σ_cosmo` = the *within-(z,τ₀)-cell* per-k std
(the cosmology scale), not the marginal std. This is the in-network analogue of a
Kennedy–O'Hagan structured-mean GP / GraphCast increment, and it's *the* reason the
cosmology signal is learnable at all.

**(2) Conditional per-k whitening.** `μ_marg, σ_marg, σ_cosmo` are all `(4,K)` (per class,
per k). The baseline target is `(m_cell−μ_marg)/σ_marg` (~unit var) and the residual target
is `(logP−m_cell)/σ_cosmo` (~unit var). Per-k whitening means the loss isn't dominated by
the high-amplitude low-k bins — every k contributes evenly. (This is the per-k
normalization you were checking for — it *is* applied; the residual tilt was a *fit*
artifact, tweak 3/4, not a missing normalization.)

**(3) Loss normalized by `Σ(weight·mask)`.** The earlier loss divided by the raw masked
count while weighting by `inv_nc` (mean ~1e-3–0.2), which shrank the baseline-head gradient
norm to ~6e-3 → the baseline trained ~100× too slowly → it under-resolved the per-(z,τ₀)
k-shape → a coherent k-tilt in the residual. Normalizing by `Σ(weight·mask)` (a true
weighted mean) restores the gradient scale. This single change is most of why the tilt
disappears.

**(4) Deeper baseline head.** Even with (3), a 1-hidden-layer baseline plateaus at
~0.10·σ_cosmo. The cell-means are a smooth (z,τ₀) function (≈rank-8), so the cure is
*depth* (≥3 layers, w256), not width or epochs. Together (3)+(4) drop the baseline mis-fit
term from 0.84·σ_cosmo to 0.03–0.05 — below the finite-sim floor of the cell-mean estimate.

**(5) SVD basis + warm-start.** The P1D manifold is smooth; a rank-12 SVD captures it to
sub-percent. Emulating *coefficients* on a fixed basis (à la Speculator/CosmoPower) is
data-efficient for 60 sims, and warm-starting the basis from the training-data SVD gives a
near-optimal init (faster, stabler). The continuous-trunk (POD-DeepONet) generalization was
prototyped and **deferred** — it cost 1.2–2.4× accuracy without beating a spline.

**(6) NaN-safe masked loss.** Bins above each row's native Nyquist are NaN. `masked_mse`
uses the double-`where` idiom so those bins contribute neither value nor gradient — no NaN
poisoning, verified under `jacfwd`.

**(7) Differentiable spline k-mapping.** For a fixed sim→data grid the cubic spline is a
constant matrix `W`; we precompute `basis_data = basis·Wᵀ` once so the emulator emits
directly on the data k-grid. `∂P_data/∂θ = W·∂P_sim/∂θ` flows through autodiff; the
resolution window is applied pointwise at data-k (no `∂P/∂k` needed). Train once, serve any
survey grid (DESI/KODIAQ), no retraining.

**(8) Inputs/transforms.** Params → unit cube via PRIYA `param_limits`; HCD `Δ_c` →
`signed-log`/`arcsinh` (sign-safe, since `Δ_c` can be small/near-zero).

**(9) Splits.** k-fold LOSO (hold out whole sims, all τ₀ together — no α-sibling leakage);
the extrapolation probe holds out τ₀-*edges* (the continuous values HMC queries), not raw α.

**(10) Honest validation.** (a) within-cell θ-tracking is computed against the *deployed*
baseline `m̂`, not the val-row mean (the latter flatters it 0.80→0.97); (b) the
inference-relevant gate is the **Fisher-bias projection** of the emulator error onto
`∂P/∂θ` — a coherent k-tilt biases n_s/A_p even if it's "θ-blind", so we check the
projection, not the absolute %; (c) the LF emulator, post-tweaks, lands all 9 params at
<0.2σ implied bias.

## 4. The error budget — C_emu (feeds the likelihood)

`C = C_cosmic + C_emu`. `assemble_covariance` now (units-fixed): `emu_var =
Σ_c w_c²·σ_Pfilt²·P_c² + Σ_c α_c²·σ_δ²·Δ_c²` (σ are *fractional*; `P_c²`/`Δ_c²` supply the
absolute scale — the missing factor was a live bug). Still to wire: **τ₀-resolved**
`C_emu(c,k,z,τ₀)` (4,K,Zb,Tb) with a smooth τ₀-index and the state-dependent `−½logdet C`
term; a **k-dependent cosmic-variance floor** `σ_CV(z,k)` (U-shaped: ~0.5–1% low-k, ~0.17%
KODIAQ, ~0.22% small-scale — from the pair-fixed run, *not* a flat 3%); and a **k×k**
(correlated) upgrade, because the rank-12 basis makes emulator errors k-correlated and a
diagonal C_emu under-covers the coherent (tilt) direction.

## 5. The resolution-correction hierarchy (multi-fidelity)

Three resolution tiers, combined so the *cheap, dense* fidelity carries the parameter
dependence and the *expensive, sparse* ones fix resolution:

1. **LF** (large box, 60 sims) — the parameter-dependence backbone `f_LF(θ,z,k)`, evaluated
   (extrapolated via the smooth basis) up to the target k.
2. **LF→HF correction `δ(θ,z,k)`** — a **learned** correction head (decided 2026-06-03).
   Measured `P_HR/P_LF` is a coherent rising resolution tilt (0.94→1.03) that is *strongly
   θ- and z-dependent* (slope 3.5–8.5 %/dex across cosmologies; 1→19 %/dex over z=5→2), so a
   fixed `ρ(k)` AR1 kernel is insufficient — `δ` must be learned and parameter-dependent
   (trained on the 6 HF sims as `HF − ρ·f_LF` in the overlap, extrapolated above the LF
   Nyquist). Reach: **KODIAQ ~0.1 s/km** (most of the KODIAQ band is *above* the LF k_max, so
   `δ` there is constrained by the 6 HF sims + extrapolation — flagged in the budget).
3. **High-k convergence correction (`res_corr`)** — `/home/mfho/lya_emulator_full/
   kodiaq_2_2_4_6-48-48/res_corr/`. It is the **particle-resolution convergence ratio
   `P1D(L15n512)/P1D(L15n384)`** (a 15 Mpc/h box, 512³ vs 384³ particles; `meta.txt`),
   shape **(15 z × 59 k)**, k = 0.003–0.242 s/km (covers the full KODIAQ reach + beyond),
   z = 5.0→2.2 (same grid as our zout), **same `param_limits` as our production grid**.
   Values **0.82–1.06** — i.e. it *suppresses* high-k by up to ~12% (min 0.88 at k≈0.14,
   z=3), departing from 1 already at k≈0.004. It is **parameter-INDEPENDENT** (one
   cosmology) → applied as a **fixed multiplicative factor** `res_corr(z,k)` to the model
   P1D (interpolated to the evaluation k-grid, alongside the spline + the resolution
   window). It corrects the residual small-scale non-convergence that even the HF box
   misses; its single-sim/finite-resolution uncertainty (n512 is not fully converged
   either) is a high-k systematic to carry in the budget. So the full forward model is
   **`P_model(θ,z,k) = ([ρ·f_LF + δ](θ,z,k)) · res_corr(z,k)`**, then spline→data-k →
   window → likelihood.

## 6. Status: validated vs open

- **Validated:** the split + per-k whitening + the (3)+(4) tweaks → LF emulator unbiased
  (Fisher <0.2σ, tilt removed); reconstruction/structural identities bit-exact; the
  differentiable spline; the units fix.
- **In flight / open:** the learned `δ(θ,z,k)` MF head (building now); the τ₀- & CV-aware
  k×k C_emu; the closure/SBC gate (the certification of "unbiased"); the `res_corr` tier
  (pending file inspection); the staged-training path (buggy, unused — non-staged is the
  production path); Δ_subDLA/Δ_DLA conditional split.
