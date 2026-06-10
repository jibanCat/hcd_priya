# Physically-informed τ₀ + HCD priors (break the HCD–A_p–τ₀ degeneracy) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the closure likelihood's over-flexible nuisance parameterizations — the 13 independent per-z τ₀ rungs and the (default) fixed HCD z-slope — with **smooth, physically-informed amplitude+slope models** so neither τ₀ nor the HCD incidence can absorb emulator residual into free per-z wiggle that A_p then trades against. Also make the subDLA template match the data's full-subDLA convention.

**Architecture:** (1) τ₀: 13-dim `alpha_ladder ~ Normal.to_event(1)` → the **PRIYA-native 2-parameter** model `α(z)=τ₀·((1+z)/(1+3))^dτ₀` (so `τ₀(z)=α·Kim07`, NO +C), with **UNIFORM** priors `τ₀∈[0.75,1.25]`, `dτ₀∈[−0.4,0.25]` (Bird+2023 §2.7.1; the *identical* form+priors in the eBOSS fit Fernandez+2024 arXiv:2309.03943 AND the KODIAQ-SQUAD fit arXiv:2509.18271 — so DESI+KS share ONE model, no per-survey mean-flux needed). The closure truth is anchored on a **PRIYA Kim07 curve** (NOT Becker — Becker's +C was what forced the abandoned +C term; a PRIYA-anchored truth fits the pure α-form exactly). The break comes from the SMOOTH structure (no per-z wiggle) + co-varying HCD, NOT a tight prior — arXiv:2509.18271 §5 explicitly warns τ₀ absorbs high-k HCD and must NOT carry a Gaussian prior unless co-varied with HCD. **OPEN (joint-fit choice): single shared (τ₀,dτ₀) [PRIYA default] vs shared dτ₀ + per-leg τ₀ amplitude [+1 param, lets each survey's mean flux float].** (2) HCD: marginalize the per-class z-slope **by default** (not just M3) with the literature dN/dX slope±1σ — amplitude (pivot α) + slope, both physical. (3) subDLA: unfiltered (full-population) excess template + full-incidence prior + full-subDLA mock, matching the P1D-measurement convention (all surveys keep subDLAs). Each phase is golden-guarded and validated forward-only before any NUTS.

**Tech Stack:** JAX/Equinox emulator, numpyro/NUTS, the `closure_legb.py` Leg-B model + `inference.py` forward + `meanflux_prior.py` τ₀ prior + `data.py` cache.

**Why (evidence):** STEP-A (44 chains) — convergence clean, but A_p systematically ~+1.5σ low even at the clean interior M4, traced to (a) the jagged 13-rung τ₀ posterior (2nd-diff RMS ~1–2) leaking into A_p (max|corr(A_p,τ₀)|≈0.5–0.6), and (b) the subDLA↔LLS↔clean identifiability degeneracy + a τ₀-extreme funnel (M2). Literature: all P1D measurements keep the full subDLA population. Full analysis in `docs/superpowers/2026-06-10-stepA-hcd-tau0-investigation.md` + walkthrough §6.

---

## File Structure

- `hcd_analysis/emulator/meanflux_prior.py` — ADD a 2-param `tau0_powerlaw(z, logA, beta)` builder + the physical (A, β) prior spec (center + width); keep the per-z `meanflux_tau0_prior` for back-compat/C_emu banding inputs.
- `hcd_analysis/emulator/closure_legb.py` — `_legb_model` / `_legb_priors_only` / `_legb_reconstruct_deterministics`: replace the `alpha_ladder` (13-dim) sites with `tau0_logA` + `tau0_beta` (2 sites) → reconstruct `tau0_global(z)`; flip `marginalize_zslope` default to True; `LegBCtx` gains `tau0_prior` fields (A/β center+σ) + the closure-vs-realfit centering. `make_truth_from_sim` returns the sim's best-fit (logA, β) truth + the full-subDLA truth.
- `hcd_analysis/emulator/inference.py` — subDLA forward: use the unfiltered excess (`P_filt[2] + delta_subDLA`); `hcd_incidence_prior` subDLA center = full incidence (drop the implicit filtered discount), width per the validation. (DLA/LLS unchanged.)
- `hcd_analysis/emulator/data.py` — expose `delta_subDLA` on the leg grid (already loaded as `out["delta"]`); add a leg-grid interpolation helper if not present.
- `tests/` — `test_data_likelihood.py`, `test_legb_*`, `tests/golden/legb_lf_golden.npz` (regenerate), new `test_tau0_powerlaw.py`, `test_subdla_unfiltered.py`.
- `scripts/run_stepA.py` — param-name lists + dim bookkeeping (25→14/17), the convergence battery names.

---

## Phase 1 — Smooth physically-informed τ₀ (amplitude + slope)

This is the core degeneracy-breaker. Do it first, validate, before Phases 2–3.

### Task 1.1: τ₀ power-law builder + physical prior spec

**Files:** Create test `tests/test_tau0_powerlaw.py`; Modify `hcd_analysis/emulator/meanflux_prior.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tau0_powerlaw.py
import numpy as np
import hcd_analysis.emulator  # noqa: enable x64 before jax
import jax.numpy as jnp
from hcd_analysis.emulator.meanflux_prior import tau0_powerlaw, becker13_tau0, BECKER13_BETA

def test_powerlaw_matches_becker_at_its_params():
    z = jnp.linspace(2.0, 4.6, 14)
    # logA,beta chosen so the power-law (no +C) equals Becker's leading term
    from hcd_analysis.emulator.meanflux_prior import BECKER13_TAU0, BECKER13_ZREF, BECKER13_C
    out = tau0_powerlaw(z, jnp.log(BECKER13_TAU0), BECKER13_BETA, zref=BECKER13_ZREF, c=BECKER13_C)
    np.testing.assert_allclose(np.asarray(out), np.asarray(becker13_tau0(z)), rtol=1e-12)

def test_powerlaw_fits_a_smooth_tau0_curve_to_subpercent():
    # a generic smooth tau0(z) is recovered by a 2-param fit to <1%
    z = jnp.linspace(2.0, 4.6, 14)
    truth = 0.62 * ((1+z)/4.5)**3.1 - 0.10
    from hcd_analysis.emulator.meanflux_prior import fit_tau0_powerlaw
    logA, beta, c = fit_tau0_powerlaw(np.asarray(z), np.asarray(truth))
    rec = np.asarray(tau0_powerlaw(z, logA, beta, c=c))
    assert np.max(np.abs(rec/np.asarray(truth) - 1)) < 0.01
```

- [ ] **Step 2: Run it to verify it fails** — `pytest tests/test_tau0_powerlaw.py -q` → ImportError (`tau0_powerlaw`/`fit_tau0_powerlaw` not defined).

- [ ] **Step 3: Implement** in `meanflux_prior.py`:

```python
def tau0_powerlaw(z, logA, beta, *, zref=BECKER13_ZREF, c=0.0):
    """Smooth τ₀(z) = exp(logA)·((1+z)/(1+zref))^beta + c  (Becker-form power-law)."""
    z = jnp.asarray(z)
    return jnp.exp(logA) * ((1.0 + z) / (1.0 + zref)) ** beta + c

def fit_tau0_powerlaw(z, tau0, *, zref=BECKER13_ZREF):
    """Least-squares (logA, beta, c) for a sim's per-z τ₀(z) — the CLOSURE truth params.
    Fit log(τ₀−c) linear in log((1+z)/(1+zref)); grid a few c then linfit (c is small)."""
    import numpy as _np
    z = _np.asarray(z); tau0 = _np.asarray(tau0)
    best = None
    for c in _np.linspace(-0.20, 0.05, 26):
        y = tau0 - c
        if _np.any(y <= 0):
            continue
        x = _np.log((1.0 + z) / (1.0 + zref))
        b, a = _np.polyfit(x, _np.log(y), 1)         # slope b, intercept a=logA
        resid = _np.sum((a + b * x - _np.log(y)) ** 2)
        if best is None or resid < best[0]:
            best = (resid, a, b, c)
    _, logA, beta, c = best
    return float(logA), float(beta), float(c)

# Physically-informed (A, β) prior — Becker+2013 center, realistic widths.
# Becker reports τ_eff to ~few-% (amplitude) and β well-constrained; widths below are the
# PRODUCTION budget (logA σ ≈ 0.05 ⇒ ~5% on amplitude; β σ ≈ 0.15). LYA-CONSULT to finalize.
TAU0_LOGA_SIGMA = 0.05
TAU0_BETA_SIGMA = 0.15
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_tau0_powerlaw.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add hcd_analysis/emulator/meanflux_prior.py tests/test_tau0_powerlaw.py && git commit -m "feat(tau0): smooth power-law builder + physical (A,beta) prior spec"`

### Task 1.2: verify the sim τ₀(z) is well-fit by 2 params (closure self-consistency precondition)

**Files:** scratch check (no commit).

- [ ] **Step 1:** For all 60 LF sims (the becker13-anchored truth τ₀(z) per z), run `fit_tau0_powerlaw` and record the max |τ₀_fit/τ₀_true − 1| over z. Expected: <1–2% (mean-flux evolution is smooth). Run:
```
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 -c "<<load cache tau0, fit each sim, print residual percentiles>>"
```
- [ ] **Step 2:** GATE — if the residual is >~2% for a meaningful fraction of sims, the 2-param model under-fits the sim τ₀(z); STOP and report (we may need β + a curvature term, or keep the truth as the per-z sim curve with the prior on (A,β) only). If <2%, proceed. **Record the number in the investigation doc.**

### Task 1.3: swap the τ₀ sites in the Leg-B model (13-dim → 2-dim)

**Files:** Modify `hcd_analysis/emulator/closure_legb.py` (`_legb_model`, `_legb_priors_only`, `_legb_reconstruct_deterministics`, `LegBCtx`); Test `tests/test_legb_tau0_powerlaw.py`.

- [ ] **Step 1: Write the failing test** — assert the model now has sites `tau0_logA`, `tau0_beta` (not `alpha_ladder`), and that `tau0_vec` deterministic reconstructs `tau0_powerlaw(zg, logA, beta, c)`:

```python
def test_legb_model_has_powerlaw_tau0_sites():
    # build a minimal ctx, trace the model, assert sites
    import numpyro, jax
    from numpyro import handlers
    ctx, mock_legs, dla_core = _tiny_legb_fixture()  # helper in the test
    tr = handlers.trace(handlers.seed(lambda: _legb_model(ctx, mock_legs, dla_core), jax.random.PRNGKey(0))).get_trace()
    assert "tau0_logA" in tr and "tau0_beta" in tr and "alpha_ladder" not in tr
    assert "tau0_vec" in tr  # deterministic reconstruction retained for the forward/banding
```

- [ ] **Step 2: Run to verify it fails** — `alpha_ladder` still present.

- [ ] **Step 3: Implement.** In `_legb_model` replace lines 611–613:

```python
    # τ₀ as a smooth physical power-law (amplitude + slope), NOT 13 free per-z rungs:
    tau0_logA = numpyro.sample("tau0_logA", dist.Normal(ctx.tau0_logA_mu, ctx.tau0_logA_sigma))
    tau0_beta = numpyro.sample("tau0_beta", dist.Normal(ctx.tau0_beta_mu, ctx.tau0_beta_sigma))
    tau0_global = numpyro.deterministic(
        "tau0_vec", tau0_powerlaw(zg, tau0_logA, tau0_beta, zref=ctx.tau0_zref, c=ctx.tau0_c))
```
Mirror the two sites in `_legb_priors_only` and reconstruct `tau0_vec` in `_legb_reconstruct_deterministics` from `samples["tau0_logA"]`, `samples["tau0_beta"]`. Add `tau0_logA_mu/sigma, tau0_beta_mu/sigma, tau0_zref, tau0_c` to `LegBCtx` (with the builder in `make_legb_ctx`/wherever ctx is constructed). The C_emu τ₀-banding (`sigma_at_tau0`/`rho_at_tau0`) is unchanged — it consumes `tau0_global(z)` exactly as before.

- [ ] **Step 4: Run to verify pass.**

- [ ] **Step 5: Commit.**

### Task 1.4: closure-vs-real-fit centering + mock truth

**Files:** Modify `closure_legb.py` `make_truth_from_sim` + ctx builder.

- [ ] `make_truth_from_sim`: in addition to the per-z truth τ₀, return the sim's best-fit `(tau0_logA_true, tau0_beta_true)` via `fit_tau0_powerlaw`. The CLOSURE prior centers on the sim's truth (`tau0_logA_mu = tau0_logA_true`, etc.) with the PHYSICAL width (`TAU0_LOGA_SIGMA`, `TAU0_BETA_SIGMA`) — tests the likelihood, not the prior, while using the physical informativeness. (The REAL-fit ctx centers on Becker: `logA=log(BECKER13_TAU0)`, `beta=BECKER13_BETA`.) Add a `ctx` flag / builder arg for which centering.
- [ ] Golden update (Task 4.1) will lock the new forward.

---

## Phase 2 — HCD physical amplitude + z-slope by default

### Task 2.1: marginalize the HCD z-slope by default with the literature prior

**Files:** Modify `closure_legb.py` (`_zslope_sites`, the ctx default for `marginalize_zslope`); Test `tests/test_legb_hcd_slope_default.py`.

- [ ] **Step 1: Write the failing test** — with a default ctx, the model samples `s_lls/s_subdla/s_dla` (currently it does NOT unless `marginalize_zslope=True`):
```python
def test_hcd_zslope_marginalized_by_default():
    ctx, mock_legs, dla_core = _tiny_legb_fixture()  # default ctx
    tr = _trace_model(ctx, mock_legs, dla_core)
    assert {"s_lls","s_subdla","s_dla"} <= set(tr)
```
- [ ] **Step 2: Run → fails** (slope fixed by default).
- [ ] **Step 3: Implement** — set the ctx default `marginalize_zslope=True`; keep the prior center `HCD_LIT_OVER_SIM_SLOPE` and width `ZSLOPE_PRIOR_SIGMA` (the literature dN/dX slope ±1σ ×z-edge inflation — already defined). Amplitude (pivot α) already sampled. So HCD now = physical amplitude (pivot, lit/sim-centered) + physical slope (lit-centered, marginalized) — the "physically-informed amplitude+slope" the PI specified. Keep a flag to restore fixed-slope for ablation.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit.**

---

## Phase 3 — subDLA full-population (match the data convention)

### Task 3.1: unfiltered subDLA template + full-incidence prior + full-subDLA mock

**Files:** Modify `inference.py` (subDLA forward + `hcd_incidence_prior`), `data.py` (expose `delta_subDLA` on leg grid), `closure_legb.py` (mock truth keeps full subDLA); Test `tests/test_subdla_unfiltered.py`.

- [ ] **Step 1: Write the failing test** — the forward subDLA excess equals the UNFILTERED excess (`(P_filt[2]−P_clean) + delta_subDLA`) within tol, and the mock truth carries the full subDLA:
```python
def test_subdla_forward_uses_unfiltered_excess():
    # forward at alpha_sub=truth reproduces filtered tier_p + full subDLA add-back
    ...
    np.testing.assert_allclose(excess_sub_forward, filt_excess + delta_sub, rtol=1e-10)
```
- [ ] **Step 2: Run → fails** (forward uses filtered `P_filt[2]` only).
- [ ] **Step 3: Implement** — in the forward (`predict_P_obs_and_cov_single_z` / the LegB excess assembly), the subDLA class power becomes `P_filt[2] + delta_subDLA(z,k)` (a fixed θ-independent add-back from the cache, interpolated to the leg grid — analogous to `dla_core`). `hcd_incidence_prior` subDLA center stays `lit/sim·w_c` (full incidence — no filtered discount), width per the forward-validation recommendation (σ/μ≈0.40 unchanged unless the validation says otherwise). `make_truth_from_sim` builds the mock subDLA from the unfiltered excess (full population). DLA + LLS unchanged.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit.**

---

## Phase 4 — Validation (golden + forward-only + confirming chains)

### Task 4.1: regenerate the golden regression + run the full suite

- [ ] Regenerate `tests/golden/legb_lf_golden.npz` for the NEW forward (the τ₀-powerlaw + unfiltered-subDLA model), document the rtol, and run `pytest tests/test_data_likelihood.py tests/test_legb_*.py tests/test_subdla_unfiltered.py tests/test_tau0_powerlaw.py -q`. Expected: all green. Commit the regenerated golden with a message stating WHY it changed (intentional model change).

### Task 4.2: forward-only bias prediction (zero NUTS)

- [ ] At the interior fiducials (M4 n_s=0.979, M1 1.019, L1a-mid 0.920) compute the linearized/Fisher (A_p, n_s, α) bias for the NEW model vs the OLD (STEP-A) model, AND the proper real-fit mismatch test for subDLA (full-subDLA mock × filtered-forward vs unfiltered-forward). GATE: predicted interior A_p bias drops toward ≤0.5σ and the τ₀-jaggedness channel is removed (2nd-diff RMS → ~0 by construction). Record in the investigation doc. If the forward test does NOT predict improvement, STOP and reconsider before spending chains.

### Task 4.3: confirming closure chains

- [ ] Re-run a focused STEP-A subset through `run_stepA.py` (param-name lists updated for the new 14/17-dim vector): the interior fiducials M4, M1, L1a-mid (4 chains each) + the τ₀-extreme M2 (to confirm the smooth model removes the funnel). GATE: convergence (R-hat<1.01, 0 div, E-BFMI>0.3) AND interior |bias z|<0.2–0.5σ on (A_p, n_s), AND M2's funnel resolved. Compare head-to-head to the STEP-A table in walkthrough §6.3. Insert the new table + verdict.

---

## Self-Review notes
- **Spec coverage:** τ₀ smooth amplitude+slope (Phase 1) ✓; HCD physical amplitude+slope by default (Phase 2) ✓; subDLA full-population (Phase 3) ✓; validation (Phase 4) ✓.
- **Risk — sim τ₀ not 2-param-representable:** Task 1.2 is the explicit gate before the model swap.
- **Risk — dimension change (25→14/17):** touches run_stepA param lists, the convergence battery names, the golden — enumerated in File Structure; Task 4.1 locks it.
- **Closure vs real-fit centering:** Task 1.4 / 3.1 keep the closure centered on the sim truth (test the likelihood) with the physical WIDTH; the real-fit ctx centers on Becker / literature. Do not ship the real fit on the closure centering.
