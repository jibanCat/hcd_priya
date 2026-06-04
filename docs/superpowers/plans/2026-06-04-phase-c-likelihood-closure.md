# Phase-C Likelihood + Closure Implementation Plan

> **For agentic workers:** execute task-by-task with TDD + the project's specialist-review convention (Bayesian + Lyα + JAX/Equinox referee per task, see memory `feedback-bayesian-referee` / `feedback-jax-review-and-traps`). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the differentiable, τ₀-aware emulator-error covariance + logdet-bearing likelihood driver and prove the τ₀–θ covariance is correctly propagated via a closure/SBC gate.

**Architecture:** Follows the 3-agent-refereed design `docs/superpowers/2026-06-02-tau0-error-model-and-closure-design.md` (THE detailed spec — read it; this plan is the execution order + test contracts). `logL(θ,τ₀,α) = −½ rᵀC⁻¹r − ½ logdet C + const`, `C = C_cosmic + C_emu(θ,τ₀)`, all JAX-pure/differentiable for NUTS.

**Tech Stack:** JAX 0.10 (x64), Equinox 0.13, optax. **numpyro NOT yet installed** (needed only for Task 4's HMC/SBC run — flag for `pip install numpyro` into `emu-jax` at that point). Trained inputs already on disk: `checkpoints/decomp_nb24_fold{0..7}.eqx` + `checkpoints/error_vector.npz`.

**Status of #14a (units fix):** DONE — merged `likelihood.assemble_covariance` already multiplies by `P_filt²`/`delta_scale²` and is NaN-safe. This plan starts at the gradient gate + #14b.

---

### Task 1: ∂P/∂θ gradient-correctness gate (the HARD pre-gate to SBC)

Proves the assembled `P_obs(θ,τ₀,α)` is smooth and its JAX autodiff Jacobian is correct end-to-end (spline k-map + SVD basis + reconstruct + HCD add-back) — i.e. HMC won't hit a non-differentiable op or NaN gradient. (Design §3.3c's emulator-autodiff-Fisher rests on this.)

**Files:**
- Create: `scripts/diag_grad_fidelity.py`
- Create: `tests/test_grad_fidelity.py`

- [ ] **Step 1: failing test** — build a tiny `Emulator` (synthetic, n_k small) + norm stats; assemble `f(θ_unit)=reconstruct_P_filt(...)→total_p1d_difference`. Assert `jax.jacfwd(f)(θ)` matches central finite-diff (h=1e-3) to `rtol<1e-4` per (param,class,k), AND `f`,`grad` finite across a 9-D `[0,1]` sweep (no NaN/Inf). Run → fails (helper not yet written).
- [ ] **Step 2:** implement `assemble_p_obs_fn(model, norm_stats, z, tau0, alpha_hcd, kfkms)` returning a pure `θ_unit ↦ P_obs` closure (reuse `reconstruct_P_filt`, `model.py` structural_tier_p, `likelihood.total_p1d_difference`).
- [ ] **Step 3:** run test → pass.
- [ ] **Step 4:** `diag_grad_fidelity.py` loads `decomp_nb24_fold0`, evaluates the gate on the real model at a fiducial (cube-centre θ, z=3, mid-ladder τ₀, α=prior centre), sweeps τ₀ across the ladder edges, writes `figures/analysis/05_likelihood/grad_fidelity.png` (rel-err heatmap + finite-vs-NaN map) + a JSON of headline max/median rel-err. **Gate: median rel-err <1e-4, zero non-finite.**
- [ ] **Step 5:** commit.

### Task 2: τ₀-band emulator error vector (#14b)

Add a τ₀-band axis so `C_emu` can be τ₀-aware. Design §1.1–§1.2.

**Files:** Modify `scripts/run_loso_sweep.py` (`fold_resid_neff`), `hcd_analysis/emulator/train.py` (`aggregate_error_vector`); Test `tests/test_emulator_train.py` (+ a new error-vector test).

- [ ] **Step 1: failing test** — `aggregate_error_vector` returns `sigma (4,K,Zb,Tb)` + stored z/τ₀ band centres; outer τ₀ bands isolate the ladder extremes; empty cells NaN; RMS-over-folds preserved.
- [ ] **Step 2:** add `tau0_band_of_row` (quantile edges over `d["tau0"]`/`alpha_idx`, `n_tb=4`, outer bands = ladder extremes) in `fold_resid_neff`; thread the Tb axis through `sigma`/`neff`; keep residual FRACTIONAL.
- [ ] **Step 3:** aggregator RMS over folds → `(4,K,Zb,Tb)`; store band centres in `error_vector.npz`. Run tests → pass.
- [ ] **Step 4:** re-run the LOSO error-vector build (smoke: 2 folds) to regenerate `error_vector.npz` with the Tb axis; sanity-plot σ(τ₀) per (c,k,zb). Commit.

### Task 3: τ₀-indexed smooth C_emu + logdet likelihood driver (#14c)

Design §1.3(b,c) + §1.4. The differentiable object HMC/closure calls.

**Files:** Modify `hcd_analysis/emulator/likelihood.py` (`assemble_covariance` τ₀-interp + a new `log_prob`/driver); Create `tests/test_likelihood_driver.py`.

- [ ] **Step 1: failing test** — (a) `assemble_covariance` accepts the banded σ + a sampled τ₀ and interpolates σ **smoothly (C¹)** in τ₀ between band centres (linear), reducing to today's behaviour when σ is τ₀-flat; (b) a `log_prob(θ,τ₀,α; data, cosmic_cov, ...)` returns `−½ rᵀC⁻¹r − ½ logdet C + const`; (c) **finite-diff gradient test**: `∂logL/∂τ₀` from `jax.grad` matches central FD (incl. the logdet contribution) to `rtol<1e-4` — omitting logdet must FAIL this test (pin it).
- [ ] **Step 2:** implement τ₀-smooth σ interpolation (vmap over (c,k,zb), linear in τ₀-band centres) + `log_prob` (Cholesky solve for `rᵀC⁻¹r` and `logdet`; MVP keeps `C_emu` diagonal with the §1.3e worst-over-τ₀ envelope inflation, `log()` the approximation).
- [ ] **Step 3:** tests → pass (incl. the logdet-omission negative control). Commit.

### Task 4: closure/SBC harness + mean-flux prior (#15) — THE gate

Design §3–§4. **Needs numpyro (NUTS)** — install into `emu-jax` first (flag to user). Uses held-out-sim TRUTH for mocks (NOT emulator self-prediction), at τ₀ on/between/off the ladder.

**Files:** Create `scripts/closure_sbc.py`, `hcd_analysis/emulator/meanflux_prior.py`; Test `tests/test_closure_sbc.py` (mechanics on a cheap mock).

- [ ] **Step 1:** mean-flux prior = independent per-z Gaussian on τ₀(z) at the **measurement width** (verified ladder brackets it, memory `phase2-analysis-scope-dla`); prior-width scan hook (measurement-σ, 3×, ∞).
- [ ] **Step 2:** mock generator from held-out-sim truth at known (θ,τ₀(z)) + obs noise; on/between/off-ladder τ₀.
- [ ] **Step 3:** NUTS on the Task-3 driver, marginalising τ₀(z) jointly with the 9 θ (+ α_c).
- [ ] **Step 4:** diagnostics — θ coverage; JOINT SBC ranks (θ AND τ₀(z), incl. the τ₀–θ correlation-eigenvector projection); recovered ρ(τ₀,θ) vs finite-diff Fisher (±10%); **bias-vs-ladder-position curve** (smoking gun). Stress test: loose/flat τ₀ prior.
- [ ] **Step 5:** pass criteria (§3.4) → figures + report. Commit.

### Gated follow-ups (after the bias-vs-ladder curve says where needed)
- **#16:** DLA-class τ₀-resolved σ_cosmo + edge-aware p_resid loss weight (§2); Δ_subDLA/Δ_DLA burial fix (apply baseline+residual to those Δ channels); staged-training stop-metric bug.
- **k×k C_emu upgrade** (§1.3d): shrinkage (diag+low-rank / Ledoit–Wolf) from the pooled LOSO residual matrix; gate on closure χ².

---

**MVP path to a first closure run:** Task 1 → 2 → 3 (diagonal C_emu, worst-over-τ₀ envelope) → 4. Upgrade (#16, k×k) after closure says where.
