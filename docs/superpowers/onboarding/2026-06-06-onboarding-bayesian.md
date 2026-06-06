# Bayesian / probabilistic-programming / inference onboarding report — 2026-06-06

Lens = the numpyro model, priors, posterior geometry, the multi-leg Gaussian likelihood,
and the SBC/closure methodology. All file:line refs are to the `phase2c-likelihood` branch
state on 2026-06-06. Verified-vs-assumed is in §3; the closing-step assessment is §4.

---

## 1. What this subsystem does (Bayesian lens)

The Phase-C inference stack turns the merged Phase-2 emulator into a differentiable
posterior that numpyro/NUTS samples, and certifies that posterior with two closure legs
(Leg A = true-SBC of the sampler; Leg B = coverage under emulator misspecification on the
real DESI+KS grids) before the blinded real-data fit.

There are **two parallel generative-model paths** sharing the same priors and the same
emulator forward model but a different likelihood factor:

- **Leg A (cache-grid, sampler validation).** `sampler_numpyro.numpyro_model(ctx)`
  (sampler_numpyro.py:43) places priors as `sample` sites + ONE `factor("loglik")` =
  `closure_ctx.log_lik_from_ctx` (closure_ctx.py:80) → `inference.log_lik_multiz`
  (inference.py:209), which vmaps `log_lik_single_z` over the cache's 172-k × n_z grid.
  Orchestrated by `closure_sbc.run_leg_a_sbc` (closure_sbc.py:173).
- **Leg B (real-survey grid, error-model validation).** `closure_legb._legb_model`
  (closure_legb.py:432) places the SAME priors + ONE `factor("loglik")` =
  `_data_loglik_legcore` (closure_legb.py:384), a thin per-leg-core wrapper over
  `data_likelihood.predict_P_obs_on_leg` (data_likelihood.py:294) +
  `likelihood.gaussian_loglik` (likelihood.py:154). Orchestrated by `closure_legb.run_legb`
  (closure_legb.py:508).

**The generative model (both legs).** The unconstrained latent block is ~25-dim:
- `theta_unit ~ Uniform(0,1)^9` with numpyro's auto-bijector (sampler_numpyro.py:51;
  closure_legb.py:438). PARAM_NAMES order = (ns, Ap, herei, heref, alphaq, hub, omegamh2,
  hireionz, bhfeedback) (inference.py:25).
- `alpha_ladder ~ Normal(tau0_mu/Kim(z), tau0_sigma/Kim(z))` on the z-isotropic
  α-ladder coordinate α=τ₀/Kim(z); `tau0_vec = alpha_ladder·Kim(z)` is deterministic
  (sampler_numpyro.py:58-60; closure_legb.py:440-442). One τ₀ per z-bin (n_z of them).
- `alpha_lls ~ Normal(μ_LLS,σ_LLS)`, `alpha_subdla ~ Normal(μ_sub,σ_sub)`,
  `alpha_dla = softplus(Normal(softplus⁻¹(μ_DLA), 1.0))` (sampler_numpyro.py:63-69;
  closure_legb.py:443-449). The α's are the per-class HCD incidence amplitudes.
- **z-resolved HCD (Leg B only, the in-flight fix).** `_legb_model` builds α_c(z) =
  α_pivot · ((1+z)/(1+z_p))^s_c with s_c = `inference.HCD_LIT_OVER_SIM_SLOPE`
  (closure_legb.py:451-454) → an (n_zg, 3) deterministic `alpha_hcd_z`. Leg A keeps a
  single (3,) α broadcast over z (sampler_numpyro.py:70).

**The likelihood factor.** Gaussian with a carried logdet:
`logL = −½ rᵀC⁻¹r − ½ logdet C`, r = P_data − P_model, C = C_data(or cosmic) + C_emu
(likelihood.py:154-175). The forward mean is the clean-baseline HCD model
`P_obs = P_clean + Σ_c α_c·(P_c − P_clean)` (inference.py:142-143). C_emu is **diagonal in
k** but carries per-class structure: diagonal `Σ_c coef_c²σ_c²P_c²` (default) OR cross-class
`Σ_cc' coef_c·coef_c'·ρ_cc'·P_c·P_c'` when a 4×4×K×Tb `rho_zb` block is supplied
(inference.py:144-154; data_likelihood.py:282-291). The C_emu **depends on the sampled
params** (θ via P_c, τ₀ via the band interp, α via coef) → the logdet term is NOT constant
and is correctly carried (the omission would bias τ₀ toward large-σ regions; documented
likelihood.py:158-160).

**Cobaya/GetDist output.** No live `sampler_cobaya.py` module yet (the closure runs
numpyro only); `scripts/legb_chains_to_cobaya.py` exports per-mock NUTS draws to
GetDist-format `.txt`/`.paramnames`/`.yaml` (the memory `feedback-cobaya-chain-output`
requirement). `minuslogpost` is currently a `−loglik` proxy, theta is exported in the unit
cube (the unit→physical map is a flagged refinement). Leg-A `ctx` carries the canonical
pack/unpack (closure_ctx.py:98-136) used by both samplers.

---

## 2. Load-bearing design decisions & WHY

### 2.1 `potential == loglik + Σ logprior` (no double-counting) — VERIFIED
- The priors are `sample` sites; the likelihood is a `factor`. The contract is pinned by
  `tests/test_closure_sbc.py:93 test_potential_equals_loglik_plus_logprior`, which
  reconstructs `log_density(numpyro_model)` from `log_lik_from_ctx + _manual_log_prior` in
  the CONSTRAINED sample space (test_closure_sbc.py:81-113), `atol=1e-6`.
- The factor is LIKELIHOOD-ONLY by construction (sampler_numpyro.py:73; closure_legb.py:455).
  `inference.log_posterior_single_z` (inference.py:243) is a SEPARATE raw-logp path (for
  blackjax) that DOES add `unit_box_logprior + meanflux_logprior + gaussian_logprior` — it
  is NOT used by the numpyro path. WHY: the soft box `unit_box_logprior` (inference.py:29)
  double-penalizes if combined with Uniform+auto-bijector → silent bias; the numpyro path
  uses the exactly-flat Uniform instead (`box_sharpness=0`, design note inference.py:255-257;
  CS review M1 in the closure plan §5).

### 2.2 θ ~ Uniform^9 + auto-bijector, NOT a soft wall — VERIFIED
- `dist.Uniform(0,1)` is exactly flat in-box (log_prob ≡ 0), so the box never penalizes
  interior moves; the auto-bijector gives finite gradients at the walls. Pinned by
  `test_no_soft_box_uniform_flat_inbox` (test_closure_sbc.py:119). WHY: PRIYA's `-inf` hard
  wall has zero/inf gradient and kills NUTS; the design (closure plan §5, "Hard-assert no
  soft box in the numpyro path") replaced it.

### 2.3 τ₀ in the α-ladder coordinate, dense mass over [θ,τ₀] — DESIGN-MOTIVATED
- Normal is placed on α=τ₀/Kim(z), not τ₀ directly (sampler_numpyro.py:54-60). WHY:
  Kim(z) is a CONSTANT wrt the sample → the change of variable is exact, linear, and
  Jacobian-free (only a constant logp shift); α is well-conditioned and z-isotropic, which
  isotropizes the prior + the C_emu τ₀-bands and cures the ladder edge stiffness (closure
  plan §5 "Sampling"). The mean-flux anchor μ_z is the Becker+2013 Eq.6 curve in Leg B
  (closure_legb.py:171, `center="becker13"`) vs the Kim MVP in Leg A (closure_sbc.py:112).
- `dense_mass=True` (sampler_numpyro.py:80; closure_legb.py:466-468) WHY: the [θ,τ₀]
  correlation is the physics (the τ_eff–cosmology degeneracy); a diagonal mass would
  under-explore that direction. `target_accept=0.9`, `init_to_median`.

### 2.4 α_DLA one-sided via softplus, NOT HalfNormal — DESIGN-MOTIVATED
- `_dla_raw_mu` (sampler_numpyro.py:37) maps the desired centre to the latent so
  `softplus(latent)≈μ_DLA`; the latent prior is `Normal(_dla_raw_mu(μ),1.0)`. WHY:
  HalfNormal's gradient is discontinuous at 0 (breaks NUTS); softplus is smooth and
  strictly positive (pinned `samples["alpha_dla"]>0`, test_closure_sbc.py:172). NOTE this is
  the CURRENT form — §0c (below) replaces it with a Gaussian-at-0 per leg; NOT yet done.

### 2.5 The HCD incidence prior (the dN/dX-based α prior) — the centering machinery
- `hcd_incidence_prior(w_c_fid, z)` (inference.py:86) returns (μ,σ) per class:
  μ = [r(z)·w_LLS, r(z)·w_sub, **HCD_DLA_RESIDUAL_FRAC**·r(z)·w_DLA], r(z) =
  `lit_over_sim_at_z(z)` (inference.py:77), the (literature/sim) dN/dX ratio as a power-law
  in (1+z): r_c(z)=r_c(z_p)·((1+z)/(1+z_p))^s_c, with
  HCD_LIT_OVER_SIM=(1.06,0.76,1.34) at z_p=3 and HCD_LIT_OVER_SIM_SLOPE=(0.95,0.15,0.40)
  (inference.py:70-74). WHY a z-slope: PRIYA's dN/dX evolves differently with z than the
  data, so a single z-independent ratio mis-centers the prior at the z-edges — this is the
  τ₀-analog (a fixed curve + slope, not one number). Widths are fractional:
  HCD_PRIOR_FRAC_SIGMA=(0.15,0.40,0.10) — **LLS TIGHT** because it is cosmology-degenerate
  (DESI DR1), subDLA broad (Zafar-vs-O'Meara factor-2), DLA tight at z≤3.5 and σ-widened
  above via `dla_inflate` (inference.py:62,104). The DLA residual fraction
  HCD_DLA_RESIDUAL_FRAC=0.30 (inference.py:61) assumes ~70% masking completeness.
- Verified numerics (z=3 toy w_c=[0.06,0.02,0.01]): μ=[0.0636,0.0152,0.00402],
  σ=[0.0095,0.0061,0.0004]. The LLS σ/μ=0.15 prior is the SINGLE most cosmology-relevant
  prior (the (A_p,α_LLS) degeneracy — closure plan §3; the "prior-driven A_p" risk).

### 2.6 The Gaussian multi-leg likelihood + block-diagonal Σ
- `data_likelihood.data_loglik` (data_likelihood.py:384) sums `gaussian_loglik` over the
  two legs (block-diagonal: DESI & KS are independent instruments sharing z-VALUES but no
  cross-covariance — data_likelihood.py:391-393). C_total = C_data + diag(C_emu)
  (data_likelihood.py:377). τ₀ is supplied on a GLOBAL z grid and each leg nearest-z-pulls
  its own τ₀ (data_likelihood.py:416-421). C_data for DESI = full 1020×1020 cov (STAT+SYST)
  + `cov_diag_inflation` on the diagonal, sub-selected to kept rows (data_likelihood.py:135-137).
- The cross-class ρ path: `rho_at_tau0` (likelihood.py:125) τ₀-interpolates the 4×4×K block;
  `emu_var = einsum("c,d,cdk,ck,dk->k", coef,coef,ρ,P,P)` is ≥0 because ρ is a sample
  covariance ⇒ SPD ⇒ coefᵀ(P∘ρ∘P)coef ≥ 0 (inference.py:147; data_likelihood.py:284).
  The off-diagonals are large (clean-LLS ρ≈+0.9, subDLA-DLA ρ≈+0.6–0.7; handoff §3a-1) and
  fix the diagonal under-sizing (whitening pooled var 1.28 diagonal → 0.93 cross-class).

### 2.7 The closure-bias diagnosis as a model-misspecification problem (the spine)
- The LF Leg-B pilot showed Ap≈10σ and α_subDLA≈19σ posterior-mean LOW (handoff §6d). Two
  workflows (wf_d9fe7cdb bias, wf_7a6c0636 emulator-bias, 4/4 lenses) concluded: this is a
  CLOSURE-HARNESS / model-spec bug, NOT an emulator bias — held-out `predict_P_filt` is flat
  to 0.03% at low-k. ROOT CAUSE: the forward applied ONE z-constant α while the mock data
  carries the sim's per-z w_c(z), which rises ~3.5–4.5× over z (effective slope ~2.67). So
  the forward over-counts HCD at low z → a COHERENT low-k over-prediction. Because C_emu is
  DIAGONAL IN k, it cannot whiten a coherent (correlated-in-k) offset → the NUTS fit absorbs
  it into A_p/n_s (via the α↔cosmo degeneracy) + α. This is the central Bayesian story: a
  coherent mean-model error that the error model cannot represent biases the posterior.

### 2.8 The CRITICAL Leg-A vs Leg-B distinction (handoff §3a, meta #3) — DESIGN-LOCKED
- **Leg A null IS rank-uniform** (mocks are draws from the emulator's own generative model)
  → the ECDF/rank uniformity gate is the valid verdict; L_FLOOR=99 (the band over-rejects at
  small L: type-I 0.20@L49 → 0.05@L99, closure_sbc.py:49-54).
- **Leg B null is NOT rank-uniform** (truth is a held-out SIM, not an emulator draw). So the
  loglik-rank ECDF is **DIAGNOSTIC-ONLY** for Leg B and must NEVER gate the verdict
  (closure_legb.py:655-658; `ll_ecdf_diagnostic_only=True` returned at :675). The ONLY valid
  Leg-B headlines are **coverage ≥ nominal + per-param bias ≈ 0** (`_aggregate_legb`
  computes both, closure_legb.py:621-653).

### 2.9 De-circularized ρ for the Leg-B cert (meta #2) — DONE
- The 8 fold-0 truth sims are ~1/8 of the pool that built `error_vector_xclass.npz`, so the
  in-sample whitening 0.93 is circular. The fix: `error_vector_xclass_holdout0.npz` (folds
  1-7, EXCLUDES the fold-0 truth sims; SPD min-eig 2.7e-6; off-diag clean-LLS +0.917 vs
  +0.926 in-sample → circularity was mild). Both files exist on disk (verified 2026-06-06).
  The diag check (scripts/diag_legb_zresolved_alpha_check.py:14) uses HOLD0.

---

## 3. Verified vs assumed

### VERIFIED (test / figure / direct numerical proof)
- **`potential == loglik + Σ logprior`** — test_closure_sbc.py:93, atol 1e-6.
- **Uniform exactly flat in-box** — test_closure_sbc.py:119.
- **vmap-sum == loop-sum** for `log_lik_multiz` — test_closure_sbc.py:132, rtol 1e-10.
- **C_mock == C_like at the truth** (Leg-A self-consistency: the noise Cholesky == the
  likelihood's C) — test_closure_sbc.py:178.
- **softplus α_dla > 0** — test_closure_sbc.py:172.
- **ρ SPD ⇒ emu_var ≥ 0; cross-class whitening 1.28→0.93** — reproduced exactly (handoff
  §3, T4c `3b5b20f`); 28 tests pass (test_data_likelihood.py + test_xclass_cemu.py, 66s).
- **The closing-step direction (forward-only, no NUTS)** — scripts/diag_legb_zresolved_alpha_check.py:
  on DESI low-k, low-k whitened construction Δ: OLD z-median **−0.78σ** → LIT-shape α(z)
  **−0.24σ** (70% removed) → EXACT per-z w_c(z) **+0.03σ**. The residual after the lit-shape
  is because the sim's w_c(z) rises ~4.5× over z (slope ~2.67) vs the lit dN/dX slope
  (0.15–1.08). This is a CONSTRUCTION residual (truth-on-leg vs forward at the truth θ),
  NOT a posterior bias.
- **C_emu sub-dominant on the real grid** — full-z median C_emu/C_data 0.0143 DESI / 0.0114
  KS; low-k max **0.52 DESI** at k<5e-3, z~3.2 (handoff §3a-1). This low-k spike is the ONE
  regime where C_emu approaches C_data and is exactly where A_p/n_s sensitivity sits.
- **Fisher forecast (single fiducial, linearized — NOT a coverage verdict)** — σ(A_p,n_s)
  unit-cube: cosmology-only (0.376,0.147) → +HCD (0.412,0.148) → +τ₀+HCD (0.420,0.190). HCD
  marg ~free; τ₀ is the cost (n_s ×1.28) (handoff §4.2).

### ASSUMED / NOT-YET-IMPLEMENTED / STATED-ONLY
- **The closing step (exact per-z g_c(z) in `_legb_model`) is NOT implemented.** The code at
  closure_legb.py:451-454 currently uses the LIT-shape (`HCD_LIT_OVER_SIM_SLOPE`) → the
  −0.24σ partial fix. The handoff "+0.03σ" requires threading the truth's per-z
  g(z)=w_c(z)/w_c(z_p) into `_legb_model` (option B in the handoff; verified forward-only,
  NOT yet wired through run_legb→_run_nuts_legb→_legb_model).
- **§0c (DLA Gaussian-at-0 + DLA-masked mock truth) is NOT implemented.** `inference.py:61`
  still has `HCD_DLA_RESIDUAL_FRAC=0.30` and the sampler/`_legb_model` still use the softplus
  one-sided α_DLA. `make_truth_from_sim` (closure_legb.py:273) still uses the sim's full
  w_c (not DLA-masked). The mock-truth w_c is collapsed to a z-median at closure_legb.py:283
  (the §0b-1 collapse the fix must remove).
- **The low-rank correlated-in-k C_emu (§0b-2) does not exist.** C_emu is diagonal-in-k
  everywhere (inference.py:124 "C stays DIAGONAL IN k either way"); the residual coherent
  offset has no covariance representation yet.
- **Leg-B coverage has NEVER been read as a verdict.** Only smoke (N≈4-8, PATH-only,
  L<L_FLOOR) + an N=50 pilot was approved; the pilot ran the BUGGY z-constant α and is a
  "this is the bug" baseline only (handoff RESUME block + §6d). No certified coverage number
  exists.
- **The `_data_loglik_legcore` per-leg-z DLA core is a documented MVP** — it uses the z-MEAN
  core per leg, not a true per-z core (closure_legb.py:384-394). Stated tiny vs P1D; DLA
  sector "un-certified by Leg B without arm A3".
- **τ₀ prior anchor (Becker13/Kim) is an MVP** — meanflux_prior.py:76-83 LYA-CONSULT: the
  production μ_z must be the observed −ln⟨F⟩(z) in the data's exact DLA-masked/metal state;
  do not ship cosmology on the Kim/Becker MVP without Lyα sign-off.
- **cosmic_cov in Leg A is a 5% fractional floor placeholder** (closure_sbc.py:131); Leg B
  uses the real published C_data so this only affects the sampler-validation leg.
- **`cemu_inflate` calibration not yet run.** The machinery exists
  (closure_diagnostics.calibrate_cemu_inflate:443) but uses a Gaussian-posterior-rescaling
  SURROGATE (NOT a NUTS re-run per inflate, closure_diagnostics.py:408-440); default 1.0.

---

## 4. Risks / open questions (ranked; flag anything that could bias the real-data fit)

### R1 (CRITICAL for the real fit) — the coherent low-k model error must be MARGINALIZED, and the chosen fix must be principled, not just "made the construction Δ go to 0 on one sim."
The closing step makes the FORWARD α(z) match ONE held-out sim's per-z w_c(z). But the real
data's HCD incidence z-shape is NOT the sim's w_c(z) — it is the observed dN/dX evolution
(which is why the lit-slope prior exists in the first place). **Hard-coding the truth's
exact g_c(z) into the forward for the closure is a self-fulfilling construction**: it
guarantees +0.03σ on the closure because the forward and the mock truth now share the same
z-shape BY CONSTRUCTION. That validates the SAMPLER given a correct mean model, but it does
NOT certify that the real fit's mean model is correct — on real data the z-shape is a
nuisance with prior uncertainty. From a Bayesian standpoint the three candidate fixes are:

  - **(B) fixed known g_c(z) shape × sampled amplitude (τ₀-analog).** Principled IF g_c(z) is
    treated as a KNOWN external input with negligible uncertainty (like the Kim curve for
    τ₀). For the closure it is exact (→ +0.03σ). For the real fit it requires committing to a
    g_c(z) shape from the literature dN/dX — and the closure showed the sim's slope (~2.67) is
    far steeper than the lit slope (0.15–1.08), i.e. the shape uncertainty is LARGE and
    real-data-relevant. A fixed shape with the WRONG slope re-introduces exactly the coherent
    low-k error on real data. **Verdict: B-alone is under-conservative unless the shape is
    genuinely known.**
  - **(B+slope) also sample a per-class z-slope nuisance (wide prior).** This is the
    statistically honest marginalization: the z-shape is uncertain → make it a sampled
    nuisance and let the data + a weakly-informative prior set it. It directly addresses the
    "the slope is the thing we got wrong" failure. COST: a per-class slope adds 1-3 params to
    an already ~25-dim posterior and is partially degenerate with α_pivot AND with A_p/n_s at
    low-k (the slope tilts the low-k HCD contribution, same lever as A_p). The Fisher forecast
    suggests HCD marg is nearly free, but a SLOPE nuisance specifically targets the A_p-degenerate
    low-k mode → this needs its OWN coverage check on A_p (the loose-α arm logic, closure plan §3).
    **Verdict: this is the principled marginalization; recommended, with a coverage gate on A_p.**
  - **(C) keep the lit-slope and absorb the residual into a low-rank correlated C_emu.**
    Modeling a COHERENT, NON-ZERO mean offset as a ZERO-MEAN covariance term is the standard
    "marginalize a template you can't predict" trick (e.g. nuisance-mode marginalization, the
    Gaussian-process / shrinkage covariance approach). It is statistically SOUND **only when
    the offset's sign/amplitude is genuinely unknown and symmetric across the ensemble** — a
    zero-mean Gaussian on a mode whose true value is a fixed nonzero number will UNDER-cover
    in the direction of that number (the posterior mean is pulled toward 0 in the
    marginalized direction, which is acceptable, but the credible interval must be wide enough
    to cover the truth). The construction residual here is a DETERMINISTIC function of the
    z-shape mismatch, not random scatter — so option C "works" only if the low-rank term's
    amplitude is set to cover the FULL plausible range of z-shape slopes, i.e. it is really
    option (B+slope) re-expressed as a marginal covariance. **Verdict: C is sound as a
    marginalization IF its amplitude is calibrated to the slope-prior range (not to the
    in-sample residual of one fold, which would be circular and too tight).**

**My ranked recommendation (Bayesian lens): (B+slope) is the principled primary** — make the
z-shape a sampled nuisance with a prior wide enough to bracket [lit-slope, sim-slope], gate
on A_p coverage. Use **(C) as the belt-and-braces** residual-marginalization with an
amplitude tied to the same slope-prior range (NOT the holdout-0 in-sample residual). Avoid
(B-alone) for the real fit — it bakes in a shape the closure itself proved is uncertain by a
factor of ~3 in slope. Do NOT certify on the exact-per-z-g(z) construction alone: it is a
sampler check, not a mean-model check.

### R2 (HIGH) — the §0c DLA decision changes the prior FAMILY (softplus → Gaussian-at-0) and is not yet implemented; it interacts with R1.
Moving α_DLA from `softplus(Normal)` (strictly positive, centred at 0.30·r·w_DLA) to a
small Gaussian centred at 0 (per leg) removes the one-sided floor. Bayesianly this is the
right call IF DLA masking really is ≥90% (DESI) / ~0% residual (KS) — a positive-only prior
on a near-zero quantity artificially inflates the DLA contribution and (because DLA leverage
is low-k) feeds the same A_p mode as R1. But a Gaussian-at-0 ALLOWS NEGATIVE α_DLA, which is
unphysical (a negative incidence). Whether that matters: at ≥90% masking the DLA term is
sub-percent, so a small negative excursion is harmless to P1D but should be checked it does
not create a fake A_p compensation. **Action: when §0c lands, verify the DLA posterior does
not rail negative and that A_p coverage is unchanged with the DLA sector ~removed.**

### R3 (HIGH) — NUTS practicalities on the ~25-dim posterior could silently degrade the Leg-B verdict.
- **ESS ≈ 0.18/sample** (handoff RESUME). At n_samples=250 the thinned L falls to ~28-42
  (< L_FLOOR=99). Coverage is still COMPUTED but coarse, and the central-interval estimate
  per mock has high variance with L<50. **Bump n_samples to reach L≥99 before reading
  coverage** (handoff explicitly: n_samples≳600). The thinning is by the WORST-ESS param
  (closure_diagnostics.thin_to_ess:76) — likely τ₀ or the slope nuisance if added.
- **dense_mass on ~25 dims is expensive** — one loglik+grad ≈ 4.3s un-jitted (the 681×681
  C_total Cholesky-VJP dominates); one cfg3 mock >35 min. N=600 ≈ 4800 CPU-h EXCEEDS the
  ~4000 budget (handoff §4.2; memory cavestru0-compute-budget). Levers: max_tree_depth 8→6,
  diagonal-mass warmup, fewer samples to L≥99 ESS. **Profile before any N≥150 launch.**
- **Divergence-retry escalation** (target_accept 0.9→0.95→0.99, closure_legb.py:542-553;
  closure_sbc.py:209) — sound (divergences cluster on hard geometry; silent exclusion biases
  the kept set toward easy regions). But each retry re-runs the whole NUTS → up to ~2× cost.
  The smoke (handoff) showed div=0 on the pilot mocks, so this may not bite — but a SLOPE
  nuisance (R1) tends to create funnels (α_pivot · g(z) is a product → a classic funnel) and
  could INCREASE divergences. **Watch divergences if (B+slope) is adopted; consider a
  non-centered parametrization for the slope×amplitude product.**

### R4 (MEDIUM) — the closure's C_emu is in-sample even with holdout0; the whitening 0.93 number is mildly circular and the diagonal-in-k assumption is the residual concern.
The holdout0 ρ excludes the fold-0 truth sims (good), but the EMULATOR (final_fold0) was
also trained excluding fold-0, so the truth sims are genuinely held out for the mean — only
the ρ is the relevant in-sample concern, and it was shown mild (0.917 vs 0.926). The deeper
issue: C_emu is diagonal-in-k, so even a perfectly-sized diagonal cannot represent the
coherent low-k mode (R1). The whitening test (closure_diagnostics.whitening_test:499) is the
right diagnostic but reports a SCALAR pooled var; it should be split by k-band and z (the
coherent mode is low-k localized — a pooled var≈1 can hide a low-k var≫1 cancelled by
high-k var≪1). **Action: report whitening var per (k-band, z), not just pooled.**

### R5 (MEDIUM) — coverage is reported only on the cosmo+α block; τ₀ coverage is per-mock-conditional.
`_aggregate_legb` computes coverage for cosmo+α (always present) but τ₀ coverage depends on
which z each mock kept (closure_legb.py:610-638). With different held-out sims keeping
different z-sets, the τ₀ coverage is computed only where present — fine, but the SAMPLE SIZE
per τ₀-z varies and the Wilson CI (closure_diagnostics.empirical_coverage:385) will be wide.
The mean-flux–cosmology degeneracy means τ₀ coverage IS load-bearing for A_p. **Ensure the
final cert reports τ₀ coverage per z with its n, not just cosmo+α.**

### R6 (LOW–MEDIUM) — the Leg-A SBC at L≥99 / N≥400 has not been re-run at production scale.
Leg A is the prerequisite ("if Leg A fails, every closure number is uninterpretable", closure
plan §0). Only `--smoke` (N≈20, PATH-only) is documented. The L_FLOOR=99 + (L+1) highly-composite
bin requirement (closure_diagnostics.rank_histogram_bins:132 warns L=100→101 prime → useless)
is correctly handled (L=99 → L+1=100). **Run the full Leg-A SBC before trusting any Leg-B
coverage** — the sampler must be certified independent of the mean-model fix.

### R7 (LOW) — blinding is parameter-blind on (A_p,n_s) only; the nuisance-absorption argument is correct but the seal timing is the load-bearing rule.
The decision (handoff §6; closure plan §8; memory blinding-strategy) — additive offset
δ_{Ap,ns}~U(−3σ,+3σ) from a committed SHA256 seed, NOT a data-vector shift — is sound (a P1D
data-shift partially self-unblinds via the nuisances). The Bayesian risk is purely process:
commit the seed BEFORE the first real fit (git-detectable), freeze the analysis on mocks,
unblind ONCE. No model-level risk. Results-privacy (DESI real cosmology stays local) is a
governance constraint, not a stats one.

---

## 5. Pointers for the main agent (the files/functions you MUST know)

1. **`hcd_analysis/emulator/closure_legb.py`** — the Leg-B generative model + driver. Know
   `_legb_model` (:432, the priors + the z-resolved α — the closing step lives at :451-454),
   `make_truth_from_sim` (:214, mock truth; the z-median collapse to remove is :283),
   `_data_loglik_legcore` (:384, the per-leg-core factor), `run_legb` (:508), `_aggregate_legb`
   (:605, coverage+bias — the ONLY valid Leg-B headline).
2. **`hcd_analysis/emulator/sampler_numpyro.py`** — the Leg-A model (`numpyro_model`:43),
   the α-ladder + softplus-DLA encoding, `make_nuts`/`run_nuts` (dense_mass, divergence
   tracking). The CANONICAL priors both legs mirror.
3. **`hcd_analysis/emulator/inference.py`** — `hcd_incidence_prior` (:86) + `lit_over_sim_at_z`
   (:77) = the dN/dX α prior + z-slope (the §0c centering to change is :61,101); the forward
   `predict_P_obs_and_cov_single_z` (:109, the cross-class einsum); `log_lik_multiz` (:209).
4. **`hcd_analysis/emulator/data_likelihood.py`** — the real DESI+KS multi-leg Gaussian
   likelihood: `predict_P_obs_on_leg` (:294, per-z forward + diag C_emu interp; per-z α at
   :346), `data_loglik` (:384, block-diagonal Σ). C_total = C_data + diag(C_emu) at :377.
5. **`hcd_analysis/emulator/likelihood.py`** — `gaussian_loglik` (:154, the logdet-bearing
   Gaussian + the SPD jitter — read the WHY on the logdet at :158), `sigma_at_tau0`/`rho_at_tau0`
   (:98/:125, the τ₀-band interp with the sanitize-YDATA-before-interp JAX trap at :116-120).
6. **`hcd_analysis/emulator/closure_diagnostics.py`** — `thin_to_ess` (:76, ESS thinning),
   `ecdf_pit_bands` (:188, the Leg-A gate / Leg-B diagnostic), `loglik_rank` (:245, Modrak),
   `empirical_coverage` (:385, Wilson), `calibrate_cemu_inflate` (:443, the surrogate
   inflation root-find), `whitening_test` (:499). These are the verdict math.
7. **`hcd_analysis/emulator/closure_ctx.py`** — the frozen `Ctx` eqx.Module + the ONE
   pack/unpack (:98-136). The Leg-A `log_lik_from_ctx` entrypoint (:80). `rho_zb` is the
   opt-in cross-class leaf (:71, defaults None = diagonal).
8. **`docs/superpowers/plans/2026-06-05-mf-likelihood-wiring-plan.md` §0b/§0c** — the LOCKED
   PI decisions: §0b-1 z-resolved α, §0b-2 low-rank correlated C_emu, §0c DLA Gaussian-at-0 +
   DLA-masked mock truth. THESE define the closing step; none are wired yet.
9. **`scripts/diag_legb_zresolved_alpha_check.py`** — the forward-only proof of the
   −0.78→−0.24→+0.03σ ladder. Re-run this (it uses the holdout0 ρ) to reproduce the closing-step
   numbers WITHOUT a NUTS run — the cheapest way to validate any α(z) change.
10. **`tests/test_closure_sbc.py`** — the inference contract tests (potential==loglik+logprior
    :93, flat-box :119, vmap==loop :132, C_mock==C_like :178). Run these + test_data_likelihood.py
    + test_xclass_cemu.py (28 pass) after ANY model change.
