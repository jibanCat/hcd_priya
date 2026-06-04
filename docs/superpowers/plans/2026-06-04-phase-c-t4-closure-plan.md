# Phase-C T4 — Closure / SBC harness + samplers + mean-flux prior (PLAN)

**Goal:** prove the *full* differentiable forward model + likelihood (θ + τ₀(z) + α_c, with the
τ₀-aware C_emu and the corrected HCD term) recovers truth **without bias and with calibrated
coverage**, especially where the emulator is weakest (τ₀-ladder edges). This is the gate
before any real-data run.

**Status of inputs (built + reviewed):** the differentiable per-z `log_lik_single_z`
(corrected HCD `P_obs = P_clean + Σ α_c(P_c−P_clean)`, per-class τ₀-aware C_emu, logdet,
NaN-safe, SPD-jittered); `hcd_incidence_prior` (observed-centered, z-slope, subDLA-broad,
DLA-weak-z); `unit_box_logprior` / `meanflux_logprior`; numpyro installed + NUTS smoke-tested
on JAX 0.10.1; the τ₀-banded error vector (`error_vector.npz`); trained 8-fold checkpoints.

---

## T4a — the unified `log_prob` (multi-z, PRIYA-dict) + the mean-flux prior producer

**Files:** `hcd_analysis/emulator/inference.py` (`log_prob`), `hcd_analysis/emulator/meanflux_prior.py` (new), `tests/test_log_prob_multiz.py`.

- `log_prob(params: dict, ctx) -> scalar`: PRIYA-name dict (`ns,Ap,herei,heref,alphaq,hub,
  omegamh2,hireionz,bhfeedback`; per-z `tau0_<z>`; `alpha_lls,alpha_subdla,alpha_dla`).
  Sums `log_lik_single_z` over the data z-bins + ONE box/bijector prior + the per-z τ₀
  Gaussian + the HCD incidence prior. `ctx` holds the fixed arrays: model, pf_stats, per-z
  {w_c, dla_core, sigma_zb (banded σ), alpha_centres, cosmic_cov, P_data, valid_k, z_unit}.
- **`dla_core` wiring:** the DLA-core add-back template = cache `delta[:,2]`, τ₀-interpolated
  per z-bin (reuse the trap-#29-safe `sigma_at_tau0` pattern). For the MVP it is a FIXED
  template (the DLA core is weakly cosmology-dependent); a follow-up may emulate it.
- **Mean-flux prior producer** (`meanflux_prior.py`): ⟨F⟩(z) ± σ from an external compilation
  (Becker+2013 / Turner+2024) → per-z τ₀ Gaussian (μ_z = −ln⟨F⟩(z), σ_z = measurement width).
  NOT artificially tight (design §4). A prior-width scale knob for the §T4e stress.
- **Tests:** multi-z sum = Σ single-z; dict↔vector pack/unpack round-trip; finite + grad
  finite at random params in the prior box.

## T4b — samplers: numpyro NUTS (default) + a Cobaya `Likelihood` (community/cross-check)

**Files:** `hcd_analysis/emulator/sampler_numpyro.py`, `hcd_analysis/emulator/sampler_cobaya.py`, tests.

- **numpyro model:** θ ~ Uniform(0,1)^9 (auto logit-bijector → exactly-flat in-box, finite
  grads — no soft box); τ₀(z) ~ Normal(μ_z, σ_z) (the mean-flux prior); α_lls/subdla ~
  Normal(μ_α, σ_α), α_dla ~ HalfNormal/softplus (one-sided). `numpyro.factor("loglik",
  log_prob_likelihood_only)`. NUTS with window adaptation; report divergences, r̂, ESS.
- **Cobaya adapter:** mirror PRIYA's `CobayaLikelihoodClass.logp(**params_values)` → pack to
  the dict → call `log_prob`. YAML `params` block from `data.PARAM_LIMITS` + the τ₀/α priors.
  A gradient-free cross-check that must agree with NUTS.
- **Tests:** NUTS recovers a Gaussian toy on the real `log_prob` skeleton; Cobaya `logp` ==
  the core `log_prob` to machine precision on a fixed param vector.

## T4c — the mock generator (the crux: held-out-sim TRUTH, on/off the τ₀ ladder)

**Files:** `scripts/closure_sbc.py` (mock section), `tests/test_closure_mocks.py`.

- Mock `P_obs` from **held-out-sim TRUTH** (NOT emulator self-prediction — that would hide
  the very emulator error under test): the cache's true per-class P1D at a known
  `(θ_true, τ₀_true(z), α_true)`, assembled through the SAME corrected HCD forward model,
  + the observational covariance noise draw.
- Generate at τ₀ **on-ladder**, **interpolated between rungs**, and **off-ladder** (the
  edge-holdout / past the extremes) — the τ₀-extrapolation probe.
- α_true drawn from the incidence prior (so SBC samples the prior, as required).

## T4d — the SBC + closure diagnostics (increasing stringency)

**Files:** `scripts/closure_sbc.py`, figures `figures/analysis/07_closure/`.

- **(a) θ marginal coverage** (necessary, not sufficient — a wrong τ₀–θ covariance can hide
  if the τ₀ prior absorbs it).
- **(b) JOINT SBC ranks for θ AND τ₀(z)** over N mocks: rank of θ_true in the posterior
  samples; KS-uniformity (p>0.05), no ∪/∩ trend. **Include the τ₀–θ correlation-eigenvector
  projection** (the interaction error concentrates there; per-param ranks can miss it).
- **(c) recovered ρ(τ₀,θ) vs the TRUTH two ways:** the sim finite-difference Fisher block
  `F=JᵀC⁻¹J` (gold standard) AND the emulator autodiff Fisher; HMC posterior ρ within ~10%
  across the ladder incl. edges.
- **(d) the BIAS-VS-LADDER-POSITION curve (the smoking gun):** θ posterior bias [σ_post] vs
  the fractional position of τ₀_true in the ladder. Flat-zero = pass; bias growing toward
  the edges = the τ₀-flat-C_emu / edge-under-fit signature.

## T4e — stress tests + the PRIYA-posterior overlay

- **cemu_inflate bracketing:** run the closure with the diagonal C_emu AND a PRIYA-style
  rank-1 (fully k-correlated) C_emu; confirm coverage is bracketed (the diagonal is
  anti-conservative — Bayesian review I1).
- **loose-τ₀-prior stress (§3.5):** rerun with a loose/flat τ₀ prior; if coverage + ρ hold,
  the τ₀–θ interaction is *proven* captured (not hidden by the prior). Report a τ₀-prior-width
  scan (measurement-σ, 3×, ∞).
- **PRIYA-posterior overlay:** run on a PRIYA-like mock + overlay the emulator posterior on
  the user's PRIYA chains (`/nfs/turbo/.../chains/simdat/`) — the apples-to-apples check the
  earlier Fisher comparison (Q5) couldn't do.

---

## Pass criteria (design §3.4)
SBC ranks uniform (all θ + τ₀(z) + the τ₀–θ eigenvector, KS p>0.05); ρ(τ₀,θ) within ~10% of
the sim finite-diff Fisher across the ladder incl. edges; bias-vs-ladder consistent with zero;
coverage bracketed by diagonal-vs-rank-1 C_emu; holds under the loose-τ₀ prior.

## Compute / risk
- SBC needs **N≈200–1000 mock inferences × NUTS** (each ~10²–10³ leapfrogs). Mitigations:
  `vmap`/`pmap` the chains; a fast first pass at N≈100 + a small z/α set; the single-z block
  is cheap. Budget + a `--smoke` mode (N≈20).
- Risk: the fixed `dla_core` (not emulated) makes the DLA term θ-static — flag in the closure
  (it's a known MVP simplification; the closure with sim-truth DLA tests it honestly only if
  the mock uses the same fixed core).
- Risk: NUTS divergences at the ladder edges (stiff τ₀ geometry) → reparam / step-size.

## Sequence
T4a → T4b → T4c → T4d (first closure at N≈100, diagonal C_emu) → T4e. Surface the SBC,
bias-vs-ladder, and PRIYA-overlay figures as they land.
