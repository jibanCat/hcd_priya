# τ₀-aware error model (C_emu) + closure/SBC design

**Date:** 2026-06-02 · **Branch:** `phase2-emulator-jax` · **Status:** design (folds the
τ₀×cosmology findings into the Phase-C likelihood + validation before implementation)

This note folds the **τ₀×cosmology investigation** (2026-06-02, 3 agents; memory
`phase2-tau0-cosmology-interaction`) into the concrete design of (1) the emulator-error
covariance `C_emu` consumed by the likelihood, (2) two emulator-side τ₀ conditioning
refinements, and (3) the closure/SBC harness. It supersedes the τ₀-flat error-vector
path in `run_loso_sweep.fold_resid_neff` / `train.aggregate_error_vector` /
`likelihood.assemble_covariance`. Spec cross-refs: §6 (covariance bullet), §7 (τ₀-edge
holdout), §8 (closure test), §10C (tasks 12–15).

## 0. Why (one paragraph + the load-bearing numbers)

The two-stage split `lnP̂ = m̂(z,τ₀) + σ_cosmo(k)·r̂(θ,z,τ₀)` is **structurally sound**
(the cross-derivative ∂²lnP/∂θ∂τ₀ = σ_cosmo·∂²r̂/∂θ∂τ₀ is fully representable; the
θ-blind baseline contributes exactly 0 to it) and **field-standard** (ForestFlow/DESI
DR1 use the same "mean-flux baseline + cosmology/IGM correction" form). The interaction
is empirically **weak and ≈ a scalar rescale for clean/LLS/subDLA** (amplitude spread
≤5% over τ₀, shape corr >0.98, ≤0.3% of cosmology variance in the x×τ₀ term) and the
trained head reproduces their τ₀-trend — so the homoscedastic per-k σ_cosmo is fine
there. **DLA is the exception** (low-k amplitude swing up to ~4×, shape corr →0.90 by
z=4; the head under-resolves it, model/empirical τ₀-spread ≈0.89 at z=3). The
**residual risk is therefore all in the second moments**: the error model is τ₀-flat,
which mis-states the likelihood at the τ₀-ladder edges and biases exactly the τ₀–θ
posterior block the PI flagged. This note makes the error model τ₀-aware, adds the DLA
conditioning fix, and specifies the closure test that *proves* the interaction is
captured.

## 1. The error model: `C_emu(c, k, z, τ₀)`

### 1.1 Producer — `fold_resid_neff` (`scripts/run_loso_sweep.py`)
- **Add a τ₀-band axis.** Today the per-fold output is `sigma (4,K,Zb)` — RMS of the
  fractional residual over *all* val rows in a z-band, which averages over the whole
  τ₀ ladder. Add a τ₀-band stratification mirroring `z_band_of_row`: a
  `tau0_band_of_row` from quantile edges over `d["tau0"]` (or directly over
  `alpha_idx`), `n_tb≈4`, with the **outer bands forced to isolate the ladder
  extremes** (the edges are where the interaction is hardest and the residual largest).
  Output becomes `sigma (4,K,Zb,Tb)`, `neff (4,K,Zb,Tb)`.
- **Keep the residual FRACTIONAL** `(P_pred−P_true)/P_true` (dimensionless, amplitude-
  stable across classes/k) — the units fix lives in the consumer (§1.3), not here.
- Reduction stays RMS over rows within each (zb,tb) cell; empty cells NaN.

### 1.2 Aggregator — `aggregate_error_vector` (`hcd_analysis/emulator/train.py`)
- RMS over folds → `sigma (4,K,Zb,Tb)`; `dla_shot_flag` stays `(K,)` (worst-case DLA
  neff over folds/z/τ₀ bands).
- **Smooth in τ₀** so the consumer is differentiable: fit a low-order σ(τ₀) per
  (c,k,zb) (linear or quadratic in the τ₀-band centers, monotone-safe) OR return the
  banded grid + band centers for the consumer to interpolate linearly. Store the band
  centers (z and τ₀) alongside `sigma` in `error_vector.npz`.

### 1.3 Consumer — `assemble_covariance` (`hcd_analysis/emulator/likelihood.py`)
Three changes, all differentiable / JAX-pure:

**(a) UNITS FIX (currently wrong).** `sigma_Pfilt` is *fractional*, but the current
`emu_var = Σ_c w_c²·σ_Pfilt²` adds it as *absolute* variance to `cosmic_cov` — it is
missing the per-class `P_c²` factor. The error on `P_tier_p=Σ_c w_c·P_c` from class c
is `w_c·(σ_frac,c·P_c)`, so:

    emu_var(k) = Σ_c w_c² · σ_frac,c(k,z,τ₀)² · P_c(k)²            # P_filt channel
               + Σ_c α_c² · σ_δ,c(k,z,τ₀)²  · (Δ_c-scale)²         # delta channel

→ **`assemble_covariance` must take the predicted per-class `P_filt` (4,K)** (and, for
the delta channel, the Δ_c scale) at the evaluation point. (Alternatively define the
producer's σ in absolute units, but fractional is the better-conditioned quantity to
store; do the multiply here.)

**(b) τ₀ INDEXING + smoothness.** `sigma_Pfilt`/`sigma_delta` are now functions of the
sampled `(z, τ₀(z))`. Interpolate σ(c,k,z,τ₀) **smoothly** in τ₀ (linear between band
centers, or evaluate the §1.2 low-order fit) so `C_emu` is C¹ in τ₀ for NUTS. `C_emu`
is now **state-dependent** (depends on sampled τ₀ *and* θ — through both σ and `P_c`).

**(c) THE logdet TERM (easy to miss).** Once `C_emu` depends on sampled parameters, the
Gaussian `−½ logdet C` is **no longer constant** and must enter the log-likelihood with
its gradient. The likelihood driver (new, §1.4) must use:

    logL(θ, τ₀, α) = −½ rᵀ C⁻¹ r − ½ logdet C + const,
        r = P_data − P_obs(θ,τ₀,α),
        C = C_cosmic + C_emu(θ, τ₀)

and let JAX autodiff differentiate through **both** `C⁻¹` and `logdet C` (C depends on
θ,τ₀). Omitting the logdet term while `C` depends on τ₀ biases τ₀(z) toward larger-σ
regions. Pin this with a finite-diff gradient test.

**(d) k×k OFF-DIAGONAL.** The rank-12 P_filt basis forces the emulator error into a
≤12-dim subspace → errors are strongly k-correlated; a diagonal `C_emu` under-states
k-integrated uncertainty and over-constrains the posterior.
- **MVP:** keep `C_emu` diagonal but apply a documented **conservative inflation**
  (envelope so we are never over-confident); `log()` the approximation.
- **Upgrade (follow-up):** estimate a per-(c,zb,tb) **k×k residual covariance** by
  shrinkage (diagonal + low-rank, or Ledoit–Wolf) from the pooled LOSO residual matrix;
  store and propagate it (K=172 is tractable). Gate the upgrade on the closure χ².

**(e) MINIMUM-SAFE fallback** if τ₀-resolution is deferred: inflate `C_emu` to the
**worst-over-τ₀ envelope** per (c,k,z). Biasing toward under-confidence is safe;
τ₀-flat-but-not-inflated is not.

### 1.4 New: a thin likelihood driver
There is no end-to-end consumer today. Add `scripts/build_likelihood.py` (or a
`likelihood.log_prob`) that assembles `P_obs` (`total_p1d_difference`), builds `C` via
the revised `assemble_covariance`, and returns `logL` per §1.3(c) — differentiable, the
object the HMC/closure test calls. This is the contract the closure test (§3) exercises.

## 2. Emulator-side τ₀ conditioning refinements (motivated by the DLA finding)

These live with the training tasks (#13 baseline accuracy / #16), not the likelihood,
but are folded here because the τ₀ investigation motivates them.

### 2.1 Edge-aware `p_resid` loss weight (general)
Homoscedastic σ_cosmo whitening gives the loss no per-τ₀ noise model, so the residual
head tends to **under-fit the interaction at the τ₀-ladder edges** (where it matters
most for the covariance). Add a τ₀-band loss weight on the `p_resid` term that
up-weights the ladder extremes (or an inverse-per-(c,k,τ₀-band)-residual-variance
weight). **Do NOT** τ₀-resolve the σ_cosmo *reconstruction* prefactor for the bulk
classes (no benefit, adds estimator noise — inference-agent §2). Drive this off the
measured edge residual: if `p_resid` RMS is flat in τ₀, skip.

### 2.2 DLA-class τ₀-resolved σ_cosmo (targeted)
For **DLA only**, replace the per-k constant σ_cosmo with `σ_cosmo,DLA(k,τ₀)` so the
residual target is ~unit-variance across τ₀ (the homoscedastic scale forces the head to
carry the full ~4× low-k swing, which it under-resolves). Thread a class-resolved
σ_cosmo through `fit_baseline_residual_norm` → `make_batch` (`t_p_resid`) →
`reconstruct_P_filt`. **Gate:** only if DLA cosmology constraints matter; success
criterion = trained model/empirical τ₀-spread → ~1.0 for DLA (currently ≈0.89 @ z=3).
DLA is the most-masked / lowest-count class, so this is a targeted, not a global, fix.

## 3. Closure / SBC harness (task #15) — τ₀-aware

Goal: **prove the τ₀–θ covariance is correctly propagated**, i.e. fail loudly if the
interaction is mis-captured. No such harness exists yet. New `scripts/closure_sbc.py`.

### 3.1 Mocks (the crux: put the data's τ₀ where the emulator is weakest)
- Use **held-out-sim TRUTH** for `P_obs` (NOT emulator self-prediction — that would
  hide the very emulator error under test), at known `(θ_true, τ₀_true(z))`, plus the
  observational covariance noise.
- Generate at τ₀ **off the training ladder**: the τ₀-edge holdout (`data.tau0_edge_holdout`)
  AND τ₀ **interpolated between ladder rungs**, plus on-ladder controls.

### 3.2 Inference
Run the real HMC/NUTS on the §1.4 driver, **marginalizing τ₀(z) per-z jointly with the
9 θ** (and α_c), with the §4 mean-flux prior attached.

### 3.3 Diagnostics (increasing stringency)
- **(a) θ marginal coverage** — necessary, not sufficient (a wrong τ₀–θ covariance can
  hide if the τ₀ prior absorbs it).
- **(b) JOINT SBC ranks for θ AND τ₀(z)** — rank-uniformity (KS p>0.05, no ∪/∩ trend)
  for all 9 θ and all τ₀(z), **including the τ₀–θ_i correlation-eigenvector projection**
  (the interaction error concentrates there; per-parameter ranks can miss it).
- **(c) Recovered ρ(τ₀, θ_i) + 2D contour** vs the **truth computed two ways**: the
  sim finite-difference Fisher block `F=JᵀC⁻¹J` (gold standard) and the emulator
  autodiff Fisher. HMC posterior ρ must agree with finite-diff within ~10% across the
  whole ladder incl. edges.
- **(d) BIAS-VS-LADDER-POSITION curve** (the smoking-gun deliverable): θ posterior bias
  in posterior-σ units vs the fractional position of τ₀_true in the ladder. Flat-zero =
  pass; bias growing toward the edges = the τ₀-flat-C_emu / edge-under-fit signature.

### 3.4 Pass criteria
SBC ranks uniform; ρ(τ₀,θ) within ~10% of finite-diff across the ladder incl. edges;
bias-vs-ladder curve consistent with zero.

### 3.5 Stress test
Run closure **twice**: with the §4 measurement-width prior AND with a **loose/flat τ₀
prior** (maximally exercises the interaction). If coverage + ρ hold even under the loose
prior, the interaction is *proven* captured and the production tight prior is a clean
information-add, not a crutch. Report θ posteriors for a prior-width scan
(measurement-σ, 3×, ∞).

## 4. Mean-flux prior on τ₀(z)

Attach an **independent per-z Gaussian** on τ₀(z) = −ln⟨F⟩(z), centered on the measured
⟨F⟩(z) at the **measurement width** (external compilation; same definition the code
already uses, `data.py` τ₀=−ln target_F). Rationale: the τ_eff–cosmology degeneracy can
materially bias cosmology — eBOSS/Lyssa's anomalously low A_Lyα was mean-flux-driven and
resolved only by such a prior; PRIYA reports it does not bite with a uniform prior (data
self-constrain τ₀). Which regime we are in is exactly what §3.5 measures.
**Do not artificially tighten** the prior to mask emulator error — use the measurement
width for production and report a prior-width sensitivity scan (§3.5).

## 5. Implementation order (folds into tasks #13–#16)

1. **#14a — units fix** in `assemble_covariance` (multiply by P_c²; pass P_filt) + a
   unit test pinning the dimensional correctness. *(small, correctness-critical, no τ₀)*
2. **#14b — τ₀-band producer/aggregator** (`fold_resid_neff` + `aggregate_error_vector`
   → `(4,K,Zb,Tb)` + band centers). *(small–medium)*
3. **#14c — τ₀-indexed, smooth `C_emu` + the logdet-bearing likelihood driver**
   (`§1.3b/c`, `§1.4`) + finite-diff gradient test through C(θ,τ₀). *(medium)*
4. **#15 — closure/SBC harness** (`§3`) + the mean-flux prior (`§4`). *(medium–large; the gate)*
5. **#16/#13 — DLA τ₀-σ_cosmo + edge loss weight** (`§2`), gated on the closure result
   and the baseline-accuracy fix. *(medium)*
6. **k×k C_emu upgrade** (`§1.3d`) — follow-up, gated on the closure χ².

MVP path to a first closure run: 1 → 2 → 3 (diagonal C_emu with the worst-over-τ₀
envelope, §1.3e) → 4. Upgrade (5,6) after the bias-vs-ladder curve says where it is
needed.
