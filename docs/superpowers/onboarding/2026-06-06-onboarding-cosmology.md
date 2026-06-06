# Cosmology / P1D / parameter-inference-physics onboarding report — 2026-06-06

Lens: the cosmology the emulator+likelihood is meant to recover (A_p, n_s primarily;
the IGM/thermal/feedback nuisances) and whether the machinery from the trained
emulator through C_emu, the data legs, and the Leg-B closure preserves it. This is
context for the documented "closing step" (make the Leg-B forward α(z) shape match
the held-out sim truth's per-z w_c(z)), but the brief is to assess the cosmology
faithfulness, NOT to implement that fix.

Env to run anything:
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## 1. What this subsystem does (your lens)

The forward model recovers cosmology from the Lyα-forest 1D flux power spectrum P1D(k).
The end-to-end chain (forward direction):

```
θ_unit(9) , z_unit , τ₀(z)            ── encoder input
   → Emulator (per-class P_filt, 4 classes: clean/LLS/subDLA/DLA)
   → P_obs(k) = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)          [predict.predict_P_obs]
   → interp cache-k → leg-k (DESI/KS), × metals × resolution     [data_likelihood.predict_P_obs_on_leg]
   → logL = Σ_leg gaussian_loglik(P_data − P_model, C_data + C_emu)
```

The 9 parameters (`data.PARAM_LIMITS`, `inference.PARAM_NAMES`, identical to PRIYA's
`coarse_grid` order), with the design-box unit-cube limits (`data.py:31-41`):

| idx | name        | lo      | hi      | role |
|-----|-------------|---------|---------|------|
| 0 | ns          | 0.8     | 1.05    | **COSMOLOGY** — primordial tilt |
| 1 | Ap          | 1.2e-9  | 2.6e-9  | **COSMOLOGY** — primordial amplitude (Lyα pivot, see §2) |
| 2 | herei       | 3.5     | 4.5     | thermal/reion — He II reion start → T₀, γ |
| 3 | heref       | 2.2     | 3.2     | thermal/reion — He II reion end → T₀, γ |
| 4 | alphaq      | 1.3     | 3.0     | thermal/reion — quasar spectral slope → heating |
| 5 | hub         | 0.65    | 0.75    | cosmology(geom) — degenerate k-rescaling (weak) |
| 6 | omegamh2    | 0.14    | 0.146   | cosmology(geom) — very narrow box (near-fixed) |
| 7 | hireionz    | 6.5     | 8.0     | reion — H I reion redshift → pressure smoothing k_F |
| 8 | bhfeedback  | 0.03    | 0.07    | astro — AGN feedback (weak on P1D) |

The cosmological constraining power lives almost entirely in **A_p and n_s**; the
thermal/pressure state is carried by separate reionization axes (herei/heref/alphaq → T₀, γ;
hireionz → k_F), and bhfeedback is a near-null nuisance. This separation is load-bearing for
the τ₀-design soundness (§4): mean-flux rescaling moves amplitude only, while thermal state
moves on its own axes (consolidated design §6, `2026-06-03-emulator-design-consolidated.md:69-78`).

Mean-flux nuisance: `τ₀(z) = −ln⟨F⟩(z)`, sampled per z on a ladder coordinate
α=τ₀/Kim(z) (`sampler_numpyro.py:54-60`). HCD nuisance: per-class α_c (LLS, subDLA, DLA)
effective post-masking incidence (`predict.py:83-103`).

---

## 2. Load-bearing design decisions & WHY

### 2.1 The A_p definition — primordial amplitude at the Lyα pivot
- `Ap = As · (5π)^(ns−1)`, with `k_p = π/4 /Mpc` (the Lyα pivot) and CAMB's `scalar_amp`
  As at `k_0 = 0.05 /Mpc`; `5π = (π/4)/0.05` is the pivot ratio.
  Source: `docs/SESSION_HANDOVER_2026_05_21.md:23-25` (commit `ab0cf04`),
  `docs/superpowers/2026-06-01-priya-vs-ours-preprocessing.md:49`.
- Verified to machine precision vs PRIYA's saved `params` for sims 0/29/44; the builder
  reads PRIYA-exact `SimulationICs.json` (8 direct + this Ap transform), not folder-rounded
  values. (`SESSION_HANDOVER_2026_05_20.md:118` records the 0.15% match BEFORE the fix.)
- **WHY it matters for inference:** A_p is the cosmology amplitude that the Lyα forest
  most directly constrains; it is degenerate with τ₀ (mean flux) — raising A_p and lowering
  τ_eff both raise low-k power. The blinding (§2.5) protects exactly (A_p, n_s).
- **WATCH:** the cosmology result is reported in the PRIYA A_p convention (Lyα pivot),
  NOT CAMB As at 0.05/Mpc. Any cross-comparison to other surveys must convert with the
  `(5π)^(ns−1)` factor — and the conversion itself depends on n_s, so the two are coupled.

### 2.2 The normalization redesign — un-burying the ~0.4% cosmology signal
`data.fit_baseline_residual_norm` (`data.py:270-336`) +
`predict.reconstruct_P_filt_jax` (`predict.py:29-39`).

- Per-(class,k), ~99.5% of log-P variance is z/τ₀; only ~0.4% is cosmology
  (consolidated design §2, walkthrough §5). A naive global per-k standardization buries it.
- Decomposition (Kennedy–O'Hagan structured-mean):
  `logP̂ = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂`, where:
  - `m̂` = θ-BLIND baseline head, targets the (z,τ₀)-CELL mean — `data.cell_id` (`data.py:246-267`)
    packs (z_rank, alpha_idx). ∂m̂/∂θ ≡ 0 by construction (verified jacfwd(baseline)=0).
  - `r̂` = cosmology residual head, whitened by `σ_cosmo` = the per-(c,k) within-cell
    cosmology scale (`data.py:323-329`, ≈0.077·σ_marg).
- **Identifiability** (`data.py:282-285`, consolidated §2): the split is identifiable ONLY
  because the baseline is θ-blind — that forces ALL θ-response through r̂, so
  `∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ`. Round-trip 8e-16, structural identity to PRIYA 1e-15.
- The cross-term `∂²logP/∂θ∂τ₀ = σ_cosmo·∂²r̂/∂θ∂τ₀` is fully representable (r̂ is a
  joint, non-separable function of θ,z,τ₀); the τ₀–θ posterior covariance is NOT
  structurally lost (consolidated §6).
- **The known bottleneck (consolidated §8):** the DEPLOYED absolute error was
  baseline-LIMITED — the θ-blind baseline m̂ under-resolves the per-(z,τ₀) cell-mean,
  amplified by the tiny σ_cosmo. Honest θ-tracking (referenced to the deployed m̂, not the
  flattered val-mean) is 0.62 (clean) → 0.92 (DLA); term-(b) baseline mis-fit dominates
  (0.64–0.81 of signal). The non-staged (joint) training FIXED this (baseline fit-ratio
  0.09, residual head captures ~90% of cosmology variance); the staged schedule was buggy
  and is NOT the default. Deployed median |P̂/P−1|: clean 0.58% / LLS 0.59% / subDLA 0.67%
  / DLA 0.97% (walkthrough §5). **This is the floor that propagates into C_emu and the
  low-k coherent residual** (the closing-step issue is downstream of this).

### 2.3 The k-grid + the SVD basis + differentiable interpolation
- k is **ANGULAR** (k = 2π/λ_v, s/km) EVERYWHERE — cache `kfkms`, DESI, KS, Rogers all
  agree; never divide by 2π (memory `hcd-template-rogers-normalization`; handoff §7).
- Output = learned SVD low-rank basis (n_basis=12 in spec/24 in walkthrough §5; the
  smoke uses 12) → P = coeffs·basis (consolidated §3). The decoder is a constant matrix
  for a fixed sim→data grid, so `∂P_data/∂θ = W·∂P_sim/∂θ` flows through autodiff.
- The likelihood binds via `jnp.interp(k_sub, cache_k, P_cache)` (linear interp to
  bin-centre; `data_likelihood.py:351`). The model P_obs(k) is smooth, so linear interp is
  sub-percent. **The trunk-vs-spline study (transcript f681e922): SVD + cubic-spline beats
  the learned POD-DeepONet trunk on interpolation; trunk deferred.** The catastrophic
  low-k interp errors (77-90%) are confined to k<1e-3 s/km, BELOW DESI/KS k_min, cut anyway.
- **k ranges:** cache k = 4.04e-4 .. 0.0694 s/km (172 bins). LF Nyquist = 0.069.
  k_max = 0.1 is a MF target (not yet wired). The whole analysis currently sits ≤0.069
  (LF backbone only); KS capped at the cache k_max (memory `phase2-analysis-scope-dla`).
- `DATA_RANGE` (`data.py:52`): z∈[2.2,4.6], k≥1e-3 — restricts the C_emu error budget +
  soft-down-weights out-of-range training bins (factor 0.2, `data.py:55`).

### 2.4 τ₀ / mean-flux — two-stage, separate axes, no extrapolation
- `KIM_AMP, KIM_SLOPE = 2.3e-3, 3.65` (`data.py:650`): the Kim-2013 central curve
  τ_eff = 2.3e-3·(1+z)^3.65 — the cache τ₀-ladder anchor.
- The ladder factor α = τ₀/Kim(z) is the z-INDEPENDENT ladder coordinate
  (`data.tau0_ladder_factor`, `data.py:653-659`). The cache has 20 rungs;
  the production error vector bands them into 4 (`tau0_band_centres = [0.656, 0.833,
  1.153, 1.331]`, verified from `checkpoints/error_vector.npz`). The outer bands
  [0.656, 1.331] bracket the observed mean flux with 2-3× margin → NO extrapolation
  (walkthrough §2; `data.make_tau0_bands`, `data.py:662-697` isolates the extreme rungs).
- **WHY separate axes (Lukić+2015, consolidated §6):** τ-rescaling moves the mean-flux
  amplitude only — it does NOT change T₀/γ/k_F. So the thermal state MUST be on the
  herei/heref/alphaq/hireionz axes; the empirically-measured τ₀×cosmology interaction is
  ≈ a scalar rescale for clean/LLS/subDLA (≤5% amplitude, shape corr >0.98, ≤0.3% of
  cosmology variance), homoscedastic σ_cosmo OK there. **DLA is the exception** (≤4× low-k
  swing, shape corr→0.90) → needs DLA-specific τ₀-resolved σ_cosmo.
- The sampler places a Normal on the ladder coordinate α (well-conditioned, z-isotropic),
  τ₀ = α·Kim(z) deterministic; Kim(z) is constant wrt the sample so the change-of-variable
  is exact + Jacobian-free (`sampler_numpyro.py:54-60`).
- Mean-flux PRIOR: `meanflux_prior.becker13_tau0` (`meanflux_prior.py:49-55`):
  τ_eff = 0.751·((1+z)/4.5)^2.90 − 0.132 (Becker+2013 Eq.6, DLA-masked + metal-corrected).
  Production anchor `center="becker13"`; the Leg-B closure uses it (`closure_legb.py:171`).
  Default frac_sigma = 0.05 (`meanflux_prior.py:25`). **LYA-CONSULT flag still OPEN**
  (`meanflux_prior.py:76-83`): the production ⟨F⟩(z) + its z-dependent error budget must be
  signed off by a Lyα expert; Becker13 is plausible but the σ(z) budget (3-8% z-dependent)
  is a placeholder. **This directly affects A_p** because A_p↔τ₀ is degenerate — a wrong
  τ₀ prior width biases A_p.

### 2.5 Blinding (parameter-blind A_p, n_s ONLY)
- Hidden additive offset `θ_shown = θ_inferred + δ`, `δ_{Ap,ns} ~ U(−3σ_prior,+3σ_prior)`,
  SHA256(project+commit) seed in `blind.lock` (handoff §6, memory `blinding-strategy`).
- **WHY parameter-blind, NOT a data-vector cosmology shift:** a data cosmology shift is a
  smooth (k,z) distortion that the τ₀/thermal/HCD/metal/resolution nuisances are BUILT to
  absorb → it partially self-unblinds AND biases nuisance recovery. (Muir-2020 data-shift
  is safe for 3×2pt, not P1D.) This is a real cosmology-faithfulness decision: the
  nuisance model is flexible enough to eat a coherent shift, which is exactly why the blind
  must be on the 2 cosmology numbers, not on the data.
- Sequence: SBC unblinded on mocks → freeze + commit seed → blind real fit → unblind ONCE.
- DESI real-data cosmology HEADLINES stay LOCAL (gitignored); code + mocks + KS committable
  (memory `desi-results-privacy`).

---

## 3. Verified vs assumed

### VERIFIED (test / figure / proof behind it)
- **A_p = As·(5π)^(ns−1)** to machine precision vs PRIYA params (sims 0/29/44);
  PRIYA bit-identity 120 points worst max|r−1| = 1.89e-5 (`SESSION_HANDOVER_2026_05_21`).
- **Baseline θ-blindness** jacfwd(baseline)=0; reconstruction round-trip 8e-16; structural
  identity to PRIYA 1e-15 (consolidated §2).
- **∂P_obs/∂θ fidelity** autodiff vs finite-diff median 7e-8 (walkthrough §8, T1), finite
  everywhere; A_p Fisher-bias RMS 0.067σ, 0/8 LOSO gate failures (walkthrough §5).
- **dN/dX → w_c → P_tier_p** reconstructs the true total to 0.04% (walkthrough §4); the
  analytic M₀-inverse round-trip is 2.4e-15 (transcript a6248760, dndx_wc referee).
- **Grad physics sane:** ns pivots at k≈0.005 s/km; A_p is a low-k boost declining at high
  k; τ₀ dominates; herei z-localized; hub the degenerate k-rescale — checked vs PRIYA
  covmat by an independent physics agent (walkthrough §8). Weak directions (bhfeedback,
  hireionz) show mild SVD-basis ringing — neutralized by informative priors.
- **HCD forward bug fixed + tested** (commit `86eda00`): old Δ_LLS≡0 → now P_c−P_clean
  with ∂P_obs/∂α_LLS ≠ 0 (walkthrough §6.1-6.2). Difference≡reweight≡multiplicative proven
  algebraically equivalent (walkthrough §6.3).
- **Cross-class C_emu whitening 1.28 → 0.93** (pooled, in-sample) reproduced exactly by the
  4-lens faithfulness review (handoff §3a; transcript a6248760). De-circularized rho
  (folds 1-7) → off-diag clean-LLS +0.917 vs +0.926 in-sample → circularity MILD.
- **C_emu sub-dominant on the real grid:** full-z median C_emu/C_data = 0.0143 DESI /
  0.0114 KS (x-class); LOW-k max 0.518 DESI (handoff §3a item1; transcript a6248760).
- **C_emu DOES vary with τ₀ band:** CV across the 4 τ₀ bands ~7-9% per (c,k,z) (I measured
  this from `error_vector.npz` this session: median CV 0.080; per-class 0.071/0.072/0.082/
  0.093). So it is NOT τ₀-flat — but the variation is modest, and the z-banding is only
  3 bins (z_band_edges = [-inf, 3.2, 4.4, inf]).
- **z-resolved-α forward IMPLEMENTED + verified forward-only**
  (`scripts/diag_legb_zresolved_alpha_check.py`): on DESI low-k the construction Δ goes
  OLD z-median −0.78σ → LIT-shape α(z) −0.24σ → EXACT per-z w_c(z) +0.03σ (handoff RESUME).
- **28 tests pass** (`tests/test_data_likelihood.py tests/test_xclass_cemu.py`, 66s) +
  23 back-compat, byte-identical diagonal path (handoff §3).

### ASSUMED / stated but NOT yet certified (cosmology-relevant)
- **Coverage of (A_p, n_s) under the real likelihood** — NOT certified. Only a single-
  fiducial linearized FISHER forecast exists (handoff §3a-WATCH, transcript a6248760):
  σ in unit-cube cfg1 cosmology-only (A_p 0.376, n_s 0.147) → cfg2 +HCD (0.412, 0.148) →
  cfg3 +τ₀+HCD (0.420, 0.190). HCD marginalization nearly FREE (A_p ×1.10, n_s ×1.01);
  τ₀ is the cost (n_s ×1.28). **This is explicitly DIAGNOSTIC-ONLY** — a Fisher number is
  NOT the coverage verdict (it linearizes at one truth and ignores the prior boundary).
- **The closure-bias FIX is verified forward-only, NOT through NUTS.** The −0.78σ→+0.03σ
  improvement is a construction-residual swap; whether the NUTS posterior pulls on A_p/n_s
  actually collapse to ~0 is NOT yet shown (the smoke with the partial 70% fix was launched
  but no clean N≥99 coverage exists). The pilot 51439170 that showed A_p≈10σ low is
  SUPERSEDED (it ran the buggy z-constant α).
- **The exact per-z w_c(z) "closing step" shape is NOT yet wired into `_legb_model`.** The
  code at `closure_legb.py:453` uses the LITERATURE dN/dX slope `HCD_LIT_OVER_SIM_SLOPE`
  as the α(z) shape, which only removes ~70% of the residual (−0.24σ). The truth's w_c(z)
  rises ~4.5× over z (effective slope ~2.67), MUCH steeper than the lit slope (0.15-1.08).
- **Production mean-flux ⟨F⟩(z) anchor + its σ(z)** — LYA-CONSULT open (`meanflux_prior.py:76`).
- **Δ-channel emulator error (σ_δ) and cross-z covariance** — flagged for before-real-data
  (walkthrough §9 item 5); fine for closure.
- **The cosmology-bias illustrative figure** `hcd_prior_cosmology.png` is single-z,
  diagonal, σ=5%. At σ=2% (DR2-like) worst mis-centering bias rises to **0.192σ — AT the
  0.2σ gate**, and multi-z stacking could push past it
  (`docs/superpowers/2026-06-04-phase-c-checkpoint-review.md:36`). Re-framed "<0.1σ" →
  "<0.2σ at the illustrative fiducial."

---

## 4. Risks / open questions (ranked; flagged for the real-data fit)

### R1 (HIGH) — The low-k coherent residual is the A_p/n_s regime, and the fix is unproven through NUTS
The closure-bias is a coherent, z-correlated, LOW-k over-prediction. Low-k is EXACTLY where
A_p (a low-k boost) and n_s (pivot k≈0.005) sensitivity sits. The diagonal-in-k (and even
the cross-class-but-still-k-diagonal) C_emu CANNOT whiten a coherent low-k shape — so the
NUTS fit absorbs it by lowering HCD-α AND the cosmology amplitude (the α↔cosmo degeneracy;
hub flips negative as A_p's compensator). The pilot showed A_p ≈10σ LOW, α_subDLA ≈19σ LOW.
- The documented fix (exact per-z w_c(z) shape) is verified to drive the CONSTRUCTION
  residual to +0.03σ, but:
  - it is forward-only — the POSTERIOR pull on A_p/n_s has not been shown to collapse;
  - even at +0.03σ construction, a RESIDUAL coherent emulator error (~4-5% low-k shrinkage
    + LOSO fold scatter, z/k-correlated) remains, and the plan's own remedy (low-rank
    correlated-in-k C_emu, §0b-2 of the MF plan) is NOT yet built. Until that exists, ANY
    leftover coherent low-k term marginalizes onto A_p/n_s, NOT whitened.
- **Why this could bias the REAL fit:** the closing-step uses the SIM's per-z w_c(z) shape.
  The real data's HCD incidence z-evolution is NOT the sim's — it is the literature dN/dX
  slope (which is what the production prior `lit_over_sim_at_z` encodes). So the closure can
  be made to pass by matching the sim, while the production forward still uses the lit
  slope. The closure then certifies a DIFFERENT α(z) shape than production runs with. This
  is a subtle self-consistency trap: certify with the same α(z) parametrization production
  uses (sample amplitude + slope), not by hard-wiring the truth's shape, or the coverage
  number does not transfer to real data.

### R2 (HIGH) — The lowest-k DESI bins (k<5e-3, z~3.2) are the one regime where C_emu ≈ C_data
Handoff §3a-WATCH: x-class C_emu/C_data max = 0.518 at low-k DESI — and that is precisely
where A_p/n_s constraining power concentrates. Everywhere else C_emu is sub-dominant
(median 0.014), so the emulator error is benign — but at the cosmology-critical bins it is
a HALF-sized contribution to the budget. Clean-class emulation fidelity there is
load-bearing. The binding diagnostic (handoff §4.2-Q2) shows the low-k emu/truth mismatch
~1.7% DESI ≈ the emulator's own 1σ there (~1.3%), BELOW the cosmic 1σ (~2.8%) — a genuine
emulator floor correctly carried by C_emu's low-k spike. **But it is carried as a DIAGONAL
variance, while the residual is COHERENT** — so the size is right (χ²/dof≈1.1) but the
SHAPE is wrong (it whitens per-k instead of marginalizing a coherent mode). This is the same
mechanism as R1, viewed from the C_emu side.

### R3 (MEDIUM) — τ₀ is the dominant cost on n_s and the mean-flux anchor is unconfirmed
Fisher: τ₀ marginalization inflates n_s by ×1.28 (the single biggest constraining-power
cost; A_p inflates only ×1.10). The A_p–τ₀ anti-correlation is the physical
amplitude↔mean-flux degeneracy (transcript a6248760, 2D-slice diagnostic). So the n_s/A_p
posterior is sensitive to the τ₀ PRIOR. Two unconfirmed pieces:
- the production ⟨F⟩(z) anchor + its σ(z) budget (LYA-CONSULT open, §2.4);
- the prior is built on Becker13 in the closure but the cache ladder is Kim2013-anchored
  (α=1 at Kim centre, NOT Becker centre) — there is a SMALL center mismatch between the
  ladder coordinate and the production prior center. Worth a numerical check that the
  becker13 prior center maps to a ladder α inside the [0.656, 1.331] interior, not near a
  band edge where σ(τ₀) is least accurate.

### R4 (MEDIUM) — Fisher forecast is credible as a RANKING but not as a σ number
The forecast correctly ranks the nuisance costs (HCD free, τ₀ expensive) and the physics is
sane (A_p–τ₀ anti-corr, sharp wells). But: (a) it is a SINGLE fiducial; (b) the chosen sim's
n_s truth sits NEAR the prior edge (unit ≈0.01-0.013) so its posterior piles on the boundary
— a coverage caveat for edge-of-range truths (transcript a6248760); (c) a 6-sim MF Fisher
is rank-deficient (MF plan §4, trap #22) and must NEVER be the gate. **Trust it for "τ₀ is
the cost, HCD is free"; do NOT quote σ(A_p)=0.42·box as a result.**

### R5 (LOW-MEDIUM) — n_s edge-of-prior pile-up will distort coverage for edge truths
Several held-out sims have n_s near the box edge (the box is 0.8-1.05). For those, the
posterior rails against the boundary (Uniform + auto-bijector), so the central-interval
coverage statistic is biased for that param on those mocks. This is a closure-DIAGNOSTIC
caveat, not a real-data bias (the real data's n_s is presumably interior), but it will make
the n_s coverage number look worse than the method warrants. Report per-truth, and flag
edge truths.

### R6 (LOW) — The σ=2% cosmology-bias is AT the 0.2σ gate, with multi-z stacking unquantified
The HCD incidence prior mis-centering bias on cosmology is 0.192σ at σ=2% single-z
(checkpoint-review §6). Multi-z stacking (the real fit has ~11 DESI z) could push past 0.2σ.
The prior centers (lit/sim ratios + z-slope) are themselves uncertain (Zafar-vs-O'Meara
factor-2 on subDLA). A mis-centered HCD prior biases A_p through the α↔cosmo degeneracy.
The DLA-residual≈0 decision (MF plan §0c) removes most DLA leverage, which helps — but
subDLA and LLS centers remain the exposure.

### R7 (LOW) — z-banding of C_emu is coarse (3 bins) and the He II reion z≈3-4 is the worst
PRIYA convergence is worst during He II reionization z≈3-4 (~7% at k≈0.1; percent k<0.05;
memory `reference-priya-sims`). The C_emu z-banding has edges only at z=3.2, 4.4 — so the
z≈3-4 worst-convergence window straddles a single band edge. The MF plan calls for
z-resolved s(k,z) with weight at z≈3-4 (MF plan §0 Q3), but that is future work. For the
LF-only production fit at k≤0.069 this is modest (convergence is good there), but it is the
regime to watch if k_max is pushed toward 0.1.

---

## 5. Pointers for the main agent (the files/functions you MUST know)

1. **`hcd_analysis/emulator/data.py:31-65`** — `PARAM_LIMITS` (the 9-param unit-cube box,
   PRIYA-exact) + `normalize_params`. The cosmology parameter contract starts here. Also
   `DATA_RANGE` (z∈[2.2,4.6], k≥1e-3) at `:52` and the Kim curve at `:650`.

2. **`hcd_analysis/emulator/predict.py:83-103`** — `predict_P_obs`: the differentiable
   forward `P_clean + Σ α_c·(P_c − P_clean)`. ∂P_obs/∂α_c = (P_c − P_clean). The corrected
   HCD model (old Δ_LLS≡0 bug fixed). `reconstruct_P_filt_jax:29-39` is the normalization
   redesign (∂logP/∂θ = σ_cosmo·∂r̂/∂θ — all cosmology flows through r̂).

3. **`hcd_analysis/emulator/inference.py:77-106`** — `lit_over_sim_at_z` +
   `hcd_incidence_prior`: the HCD α prior CENTER (lit/sim ratio × z-slope power-law) and
   the DLA-residual handling. `HCD_LIT_OVER_SIM=(1.06,0.76,1.34)`,
   `HCD_LIT_OVER_SIM_SLOPE=(0.95,0.15,0.40)`. Note §0c of the MF plan SUPERSEDES the
   `HCD_DLA_RESIDUAL_FRAC=0.30`/softplus form (move to α_DLA Gaussian-at-0) — NOT yet done.

4. **`hcd_analysis/emulator/data_likelihood.py`** — the REAL-data binding.
   `load_desi_leg:116` / `load_ks_leg:149` (cuts, covariance); `predict_P_obs_on_leg:294`
   (cache-k → leg-k interp + metals/resolution + C_emu); `data_loglik:384` (block-diagonal
   multi-leg). The lowest-k DESI bins (k<5e-3, z~3.2) are the cosmology-critical regime
   where C_emu ≈ 0.5·C_data.

5. **`hcd_analysis/emulator/likelihood.py:98-175`** — `sigma_at_tau0` / `rho_at_tau0`
   (τ₀-band interp of the error vector; the NaN-BEFORE-interp guard at `:118-122` is
   load-bearing for NUTS τ₀ gradients) + `gaussian_loglik` (the logdet-bearing Gaussian;
   the logdet is load-bearing — omitting it biases τ₀ toward larger-σ regions).

6. **`hcd_analysis/emulator/closure_legb.py`** — the Leg-B closure. `make_truth_from_sim:214`
   (held-out sim → truth, currently z-means the w_c at `:283` — the closing-step touches
   this), `_legb_model:432` (the α(z) shape at `:453` uses the LIT slope, NOT the exact
   w_c(z) — the closing step replaces this), `run_legb:508`, `_aggregate_legb:605`
   (coverage + bias are the ONLY Leg-B headlines; the loglik-rank ECDF is DIAGNOSTIC-ONLY).

7. **`hcd_analysis/emulator/meanflux_prior.py:49-95`** — `becker13_tau0` + `meanflux_tau0_prior`.
   The τ₀ prior center/width that A_p degeneracy is most sensitive to. The LYA-CONSULT at
   `:76-83` (production ⟨F⟩(z) + σ(z) unconfirmed) is an open cosmology risk.

8. **`scripts/diag_legb_zresolved_alpha_check.py`** — the forward-only verification of the
   closing step (OLD −0.78σ → LIT −0.24σ → EXACT per-z w_c +0.03σ). Run this to reproduce
   the construction-residual numbers before/after any α(z) change.

9. **`checkpoints/error_vector.npz`** (diagonal σ, 4×172×3×4) +
   **`error_vector_xclass.npz`** (cross-class rho, 4×4×172×3×4) +
   **`error_vector_xclass_holdout0.npz`** (de-circularized, folds 1-7). tau0_band_centres
   = [0.656,0.833,1.153,1.331]; z_band_edges = [-inf,3.2,4.4,inf]. Matched to
   `checkpoints/final_fold0` (the ONE production model — single fold, LOSO spread already
   pooled in the error vector; do NOT ensemble).

10. **`docs/SESSION_HANDOVER_2026_06_05.md`** (RESUME HERE block) + the MF wiring plan
    `docs/superpowers/plans/2026-06-05-mf-likelihood-wiring-plan.md` (LOCKED claims §0;
    §0b correlated-C_emu fix; §0c DLA-residual≈0). The live state of the closing step + the
    Fisher forecast + the C_emu numbers are here. memory `phase2c-likelihood-progress`.

---

## 6. One-paragraph cosmology verdict (for the meta-reviewer)

The machinery is structurally sound for recovering A_p/n_s: the A_p definition is
PRIYA-exact, the normalization redesign genuinely un-buries the ~0.4% cosmology signal with
the θ-blind-baseline identifiability proof, all θ-response flows through r̂ with verified
gradient fidelity, and the τ₀ two-stage design correctly separates mean-flux amplitude from
thermal state. The HCD marginalization is nearly free (Fisher A_p ×1.10) and the C_emu is
sub-dominant (median 0.014·C_data) EXCEPT at the cosmology-critical low-k DESI bins
(z~3.2, k<5e-3) where it reaches ~0.5·C_data. The single biggest UNRESOLVED risk to the
real-data A_p/n_s is the coherent low-k closure residual (R1+R2): it is the A_p/n_s regime,
the diagonal/cross-class-k-diagonal C_emu cannot whiten a coherent mode (it can only resize
it), the documented fix is verified only forward-only/in-construction (not through the NUTS
posterior), and the low-rank correlated-in-k C_emu that the plan itself calls for to absorb
the residual is NOT yet built. Compounding it: the closing step matches the SIM's per-z
w_c(z), but production uses the literature dN/dX slope — so the closure can pass against a
shape production never uses, and the coverage number may not transfer. Certify with the
production α(z) parametrization (sampled amplitude + slope), build the correlated-in-k
C_emu, and re-run coverage at N≥99 before trusting σ(A_p,n_s) — the Fisher number is a
ranking, not a result.
