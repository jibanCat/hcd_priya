# Phase-C T4 — Closure / SBC harness (v2, post 4-agent review)

**Reviewed by:** stat-software/PPL, CS/JAX, Lyα-cosmology, cosmology-emulator agents (2026-06-04),
all **GO-WITH-CHANGES**. v1 conflated two statistically distinct tests and stress-tested the
wrong axis (the τ₀-ladder edges, which our own `phase2-analysis-scope-dla` shows the data never
visits). v2 re-centers on: a **two-leg split** (true-SBC vs closure-under-misspecification),
the **θ-hull + interior-τ₀** stress (not the τ₀-edge), **validating C_emu as an error model**
(coverage + whitening, not bracketing two wrong models), and **de-circularizing the HCD + adding
the unmodeled physics** (metals, HeII floor).

**Goal:** certify the *emulator-as-likelihood* — the sampler is correct, the emulator-error budget
gives ≥nominal coverage, and cosmology is unbiased in the production regime — before real data.

---

## 0. The two legs (stat-agent M1 — the spine; do not call the second "SBC")

**Leg A — TRUE-SBC (sampler validation, run FIRST).** Mocks from the emulator's OWN generative
model: `ỹ = P_obs_emu(θ̃,τ̃₀,α̃) + ε`, `ε ~ N(0, C)` with the SAME `C = C_cosmic + C_emu` the
likelihood uses. θ̃/τ̃₀/α̃ drawn from EXACTLY the sampler's priors. Now rank-uniformity is the
correct, attributable null → any non-uniformity = a sampler/geometry/coding bug (funnel,
divergence, logdet sign). Cheap (no held-out sims). **If Leg A fails, every closure number is
uninterpretable.**

**Leg B — CLOSURE under emulator misspecification.** Mocks from **held-out-sim TRUTH**, `ε ~ N(0,
C_cosmic ONLY)` (the observational noise; the emulator error IS the sim≠emu discrepancy, not added
noise — CS M5). Judged on **one-sided empirical coverage (≥ nominal)** + the **bias curves**, NOT
rank uniformity (uniformity is false-by-construction here). This is the emulator-misspecification
test the field uses (PRIYA bias-vs-truth, LaCE).

---

## 1. The stress matrix (mock arms) — re-centered on the production-relevant axes

Primary gate = the **interior production regime**; the τ₀-edge is demoted to a robustness appendix
(Lyα S1: our verified ladder brackets the data with 2–3× margin → the τ₀-edge "is a formality").

| arm | what it perturbs | why (reviewer) | pass |
|---|---|---|---|
| **A1 interior-τ₀ (PRIMARY)** | τ₀≈Becker/Turner ⟨F⟩(z), θ interior | the real-data regime | coverage ≥ nominal; bias <0.2σ |
| **A2 θ-hull extrapolation** | θ_true near/outside the 60-sim convex hull | the 60-sim NN is weakest in 9-D θ, NOT τ₀ (emulator M1) | bias-vs-(θ-hull-distance) flat |
| **A3 HCD-template mismatch** | perturbed DLA core (Voigt wing / Rogers kernel) + survey-completeness incidence redistribution | the closure is circular on shape; DESI 15% / our 30% DLA RMSE (Lyα M1) | bias <0.3σ; α absorbs it |
| **A4 metals (SiIII)** | inject a McDonald-form SiIII oscillation (k<0.06) | metals absent from model+mock; largest KODIAQ high-k systematic (Lyα N1) | k<0.06 cosmology unbiased w/ a SiIII nuisance or the cut+inflation |
| **A5 HeII-patchy / IGM-thermal floor** | sightline-to-sightline stochastic variance + 5% bhfeedback departure at high k | proves the k<0.06 cap contains the out-of-scope band (Lyα N2) | in-scope cosmology unbiased |
| **A6 τ₀-ladder edge (robustness)** | off/between-ladder τ₀ | the documented τ₀-flat-C_emu probe (demoted) | reported, not a gate |

Mock-truth must extrapolate the **forward model** (the emulator) past the design too, not just the
error vector (CS N4). The mock noise covariance must include **resolution/window + noise-power
subtraction** at the KODIAQ k-edge, not idealized (Lyα N3).

## 2. Validate C_emu as an error model (emulator M3/M4/M5 — the gap v1 missed)

C_emu is the cosmology-error budget; v1 never validated it. Add (all cheap, mostly no NUTS):
- **Whitening test:** assemble C at each held-out sim's (θ,τ₀); whiten its measured P1D residual
  `r_white = L⁻¹ r`; require mean≈0, var≈1, KS-uniform. The single most diagnostic emulator check.
- **C_emu ≪ C_data:** a `diag(C_emu)/diag(C_cosmic)` heatmap over (k,z) per τ₀-band — the actual
  field-standard certification (PRIYA/DESI/LaCE). Shows WHERE the emulator dominates.
- **k×k reference arm:** run the closure with the empirical **shrinkage k×k LOSO residual
  covariance** (Ledoit–Wolf / diag+low-rank) as a THIRD C_emu arm; show coverage lands inside the
  diagonal/rank-1 bracket (else the bracket is two wrong models, not a bound). Removes the circular
  "gate the k×k upgrade on the closure".
- **Coverage as the certified headline + calibrate `cemu_inflate`:** report the fraction in 68/95%
  CR per param; the plug-in WILL under-cover (no NN-weight uncertainty + a noisy 8-fold estimate),
  and SBC ranks can pass at 88% — so calibrate `cemu_inflate` (per-(class,z) if the under-coverage
  localizes) to restore 95%, don't just bracket.
- **Bias vs variance in the error vector:** report `mean(rfrac)` next to `RMS(rfrac)` per cell; if
  bias-dominated, the symmetric-Gaussian C_emu needs a caveat. Verify the τ₀-banded σ against the
  genuinely off-ladder `tau0_edge_holdout` residual (currently EXCLUDED — never measured).
- **Apply the LF-Nyquist `k_max` cut** (the error vector has `k_min` but no `k_max`; above Nyquist
  C_emu is meaningless and cosmic_cov optimistically small).

## 3. Diagnostics (stat M2 + Lyα M2/M3/S3)

- **Leg A:** ECDF simultaneous confidence bands (Säilynoja+2022), **not** KS; the **log-likelihood
  rank as the PRIMARY joint statistic** (Modrak+2023 — most powerful, avoids the 20–40-marginal
  multiple-testing dilution); N≳400–1000 from a power calc to exclude a ~0.2σ bias; the τ₀–θ (and a
  cross-z τ₀) eigenvector `v` FIXED from the truth Fisher (not the per-mock posterior); bin count
  with `L+1` a multiple of `B`; thin posteriors to near-independence; exclude+re-run divergent mocks.
- **Leg B:** **coverage + bias-vs-{interior-τ₀-position, θ-hull-distance}** as the co-equal smoking
  guns. **Split bias-vs-τ₀ by HCD class AND k-band** — the interaction is DLA-low-k-localized; apply
  the pass on the **DLA low-k panel** (a single curve averages it away). Plus `χ²/dof` of the
  held-out residual under C (the amplitude check SBC misses).
- **Named degeneracy diagnostics:** the **(A_p, α_LLS)** 2D + its degeneracy-eigenvector rank, and
  the **(A_p, τ₀)** block. Run a **loose-α_LLS** arm (mirror the loose-τ₀ stress): if A_p coverage
  only holds with the tight LLS prior, the result is PRIOR-DRIVEN (the eBOSS/Lyssa failure in the
  HCD channel) — KODIAQ's z-dependent LLS selection bias then becomes a carried systematic.
- **τ₀-parametrization arm:** run the closure under BOTH **per-z-free τ₀** (conservative) AND the
  **Kim 2-param slope** (PRIYA's `mean_flux="s"`); report A_p width+bias for each. A divergence is a
  real τ₀–cosmology leak, not an artifact. (Also makes the PRIYA overlay interpretable.)

## 4. The PRIYA overlay (Lyα S4) — three documented caveats or it misleads
(1) mean-flux model differs (per-z-free → your A_p legitimately WIDER); (2) k-range differs (you cap
KODIAQ<0.06, DESI≤0.02); (3) the sim-vs-data offset (A6: subDLA ×1.31, DLA ×0.70). Run a
PRIYA-`mean_flux="s"` arm so the τ₀ axis is matched. Frame as a pipeline consistency check (PRIYA is
a GP on different data), not a certification.

## 5. Software contract (CS M1–M5, S1/S4; numpyro stat-S1)

- **`ctx`** = a frozen `eqx.Module`/`NamedTuple`: static metadata (`n_z`, `K`, flags via
  `eqx.field(static=True)`) + an array pytree (per-z arrays stacked on a leading z-axis). jit only
  over `params`. ONE canonical **pack/unpack** (dict↔vector, `PARAM_NAMES` + fixed `tau0_z…/alpha_*`
  tail, asserted vs `PARAM_LIMITS`) shared by both samplers.
- **`log_lik(params, ctx)`** = LIKELIHOOD-ONLY (vmap `log_lik_single_z` over the z-axis — K is FIXED
  172, no padding — `jnp.sum`); priors a separate tree. numpyro `factor(log_lik)` + `sample` priors;
  Cobaya `logp`=`log_lik`. Test `numpyro potential == log_lik + Σ log_prior` (no double-count) and
  `vmap-sum == loop-sum`. **Hard-assert no soft box in the numpyro path** (Uniform+auto-bijector;
  the soft wall double-penalizes → silent bias).
- **dla_core MVP = FIXED** (no τ₀-interp / no trap-#29); the mock uses the IDENTICAL core; τ₀-interp
  out of scope. State the DLA sector is therefore **un-certified** by Leg B without arm A3.
- **Mocks:** root key → per-mock keys (recorded); Leg-B noise = Cholesky of **cosmic_cov only**
  (jittered as in `gaussian_loglik`); Leg-A noise = `N(0, C_cosmic+C_emu)`.
- **Sampling:** **τ₀ in the α-ladder coordinate** (α=τ₀/Kim(z) — isotropizes prior+C_emu, the
  edge-stiffness cure; belongs here, not a T4d firefight) + **`dense_mass` for the [θ,τ₀] block**
  (the correlation is the physics) + `target_accept≈0.9`. Softplus α_dla (not hard HalfNormal).
- **Parallelism:** shard mocks over **SLURM-array processes** (each caches the NUTS step), NOT
  vmap-over-chains (lockstep leapfrogs = worst-case CPU). Profile one mock × N before launching
  (cavestru0 budget). `--smoke` (N≈20, n_k small, 1 z-bin) exercises the FULL path.
- **ONE production model** for the closure (not the 8-fold ensemble; the C_emu LOSO vector must be
  the matched pair). x64 asserted in the sampler modules; no `donate`.
- **Files:** `inference.log_lik`/`log_prior` trees; `meanflux_prior.py` (μ_z=−ln⟨F⟩(z) match the
  data's masking/metal state — Lyα S2 — + `flat=True` for the ∞-prior scan, not σ=1e30);
  `closure_mocks.py` (pure), `closure_diagnostics.py` (pure), `closure_sbc.py` (orchestration);
  `sampler_numpyro.py`, `sampler_cobaya.py`.

## 6. Pass criteria (v2)
**Leg A:** ECDF bands contain the rank ECDF for all params + the log-lik rank + the fixed
eigenvectors; near-zero divergences. **Leg B:** empirical coverage ≥ nominal (after `cemu_inflate`
calibration); bias <0.2σ interior / flat vs θ-hull-distance; bias <0.3σ under the A3 HCD-template +
A4 metal + A5 HeII perturbations; **survives the loose-α_LLS AND loose-τ₀ stress**; the whitening
test (mean≈0, var≈1) + `χ²/dof≈1` + `C_emu<C_data` in-range; coverage lands inside the
diagonal/rank-1/k×k arms. The DLA-low-k bias panel is the binding test.

## 7. Sequence
Leg-A true-SBC (smoke → N≈400) → the C_emu validation (whitening, C_emu/C_data, k×k arm — no NUTS) →
Leg-B interior-τ₀ + θ-hull (the gate) → the degeneracy + τ₀-param arms → the HCD/metal/HeII
perturbation arms → the PRIYA overlay. Surface coverage, bias-vs-hull, the DLA-low-k bias, and the
whitening figures as they land.
