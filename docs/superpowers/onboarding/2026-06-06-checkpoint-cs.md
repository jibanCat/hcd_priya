# CS / code-correctness / JAX checkpoint review — HCD slope-prior spec — 2026-06-06

Lens: is the spec implementable cleanly + safely? I read the spec in full, the CS + meta
onboarding, the sweep script + its results .txt, and traced every code site the spec
touches (`_legb_model` 432-456, `_data_loglik_legcore` 384-429, `predict_P_obs_on_leg`
294-378, `DataLeg` 70-99, `LegBCtx` 75-103, `make_truth_from_sim` 214-283, `make_legb_mock`
truth_pack 364-368, `dndx_wc.w_c_from_mu`/`alpha_from_dndx_law` 12-86, `inference`
54-106). I ran `diag_legb_zresolved_alpha_check.py` (reproduced −0.780 / +0.034 / −0.239σ
exactly, ~3 min) and confirmed `tests/golden/` does not exist on disk.

## Verdict summary
**GO-WITH-CHANGES.** The factorized α(z) is JAX-clean *in isolation*, but the spec as
written cannot be implemented at the site it names (§1/§3 put `g_fixed,c(z)` and
`δ_{c,leg}` on a model that today computes ONE global-z α array). The per-leg factorization
forces a real plumbing change in `_legb_model` + `_data_loglik_legcore` — not the
"near-zero-risk multiply" the onboarding implied. Two hard prerequisites the spec correctly
defers (golden guard) or under-specifies (per-leg α contract) must land first. Q3 (exact
construction bias) is a <30-line extension of an existing harness and should be RUN before
committing the width, because the linearized bias the spec locks on is the one number that
is both load-bearing and known-13–15% wrong on the relevant quantity.

---

## Q1 — production `g_fixed` (lit-derived) vs sim `w_c(z)`: defer-to-cosmology on the
## science, but a CS-decisive point on differentiability + a hidden plumbing cost

On the **science** of which shape certifies the real fit, defer to [cosmology]/[lya]/[bayesian]
— they own "does coverage transfer." But the CS lens has two concrete inputs:

**(a) The lit-derived `g_fixed` is a PRECOMPUTED CONSTANT, not a differentiable site — and
that is exactly right.** `g_fixed,c(z) = w_c(z)/w_c(z_p)` from a literature incidence law is
built once, host-side, from `dndx_wc.alpha_from_dndx_law(A, γ, Xbar, z)[...,1:]`
(dndx_wc.py:80-86) — `w_c_from_mu` (dndx_wc.py:12-19) and `w_c_corrected` (59-64) are
JAX-pure (no python branch on traced values), but for production the (A_c, γ_c) literature
incidence-law coefficients and Xbar(z) are FIXED numbers, so `g_fixed` should be evaluated
ONCE outside the traced model and closed over as a `(n_z,3)` numpy/jnp constant. It must NOT
be recomputed inside `_legb_model` per leapfrog (that would put `w_c_corrected`'s
renormalisation + the delta_c deg-2 polynomial in the hot path for no benefit — it has no θ
dependence). So Q1's "is the map differentiable / does it need to be?" → **it does not need
to be; precompute it.** Confirmed: `w_c_from_mu` is currently used only in
walkthrough/diagnostic scripts + `calibrate_delta_c`, never in the likelihood hot path, so
there is no precedent forcing it into the traced graph.

**(b) The choice changes the plumbing.** In the **sim-`w_c(z)`** framing the shape is
per-MOCK (a different held-out sim each mock → different `g_fixed`), so it threads as a
per-mock closed-over constant (the route the CS onboarding R3 recommended). In the
**lit-derived** framing the shape is the SAME for every mock and every real-fit run → it can
be a frozen field on `LegBCtx` (computed in `build_legb_ctx`), which is *cleaner* (one source
of truth, set at ctx build, identical mock-to-mock). **CS preference, conditional on the
science: lit-derived, stored on ctx.** It removes the per-mock variation as a degree of
freedom and makes the closure forward byte-identical to the real-fit forward — which is the
whole point of a production-faithful closure and is also what the golden guard then pins.

**One trap the spec hides:** the sweep script (`diag_legb_slope_prior_tradeoff.py:90`)
ALREADY uses the sim framing — `g_fixed = wc_perz[sel_sim] / wc_pivot` — so **the locked
×1.04 / 0.167σ numbers were measured under the closure-only `g_fixed`, NOT the production
lit-derived one.** If Q1 resolves to lit-derived (recommended by the onboarding §5), the
fiducial operating point moves off `s=0` (the lit-vs-lit differential is ~0 only if the lit
incidence law and the lit ratio-slope are mutually consistent — they are *different* fits:
`HCD_LIT_OVER_SIM_SLOPE` vs the dN/dX power-law γ), and the bias decomposition could shift.
This is a CS-flavored consistency point feeding Q3: **re-run the sweep with the production
`g_fixed` before locking the width.** It is the same harness, one line changed (line 90:
swap `wc_perz/wc_pivot` for `alpha_from_dndx_law(...)/g_pivot`).

## Q2 — slope center (WLS 0.76/0.05 vs code 0.95/0.15); fixed const or hyper-prior

Defer the *value* to [bayesian]/[lya]. CS observations:

- The **center is a fixed constant in the sweep** (`S_CENTER`, line 48) but note the sweep's
  bias model centers the *prior* at `s=0` and offsets the *truth* by the gap (line 165:
  `(0 - gap)`) — i.e. the center that enters the code is `s_c=0` as the prior mean of the
  SAMPLED slope, with `g_fixed` carrying the rest. This is a parametrization choice, not a
  number: in the factorized model `α = a_pivot·g_fixed·(1+z)^{s_c}`, the prior on `s_c` is
  `Normal(0, σ_s)` IF `g_fixed` already encodes the center shape, OR `Normal(s_center, σ_s)`
  if `g_fixed` is shape-only. **The spec must state which.** If lit-derived `g_fixed`
  (Q1), then `g_fixed` IS the center and `s_c ~ Normal(0, σ_s)` is correct and clean.
- **A hyper-prior on the center is NOT worth the cost.** It adds a level to the hierarchy →
  a funnel risk on top of the slope×amplitude funnel the onboarding already flagged
  ([bayesian-R3], meta-R6). With ESS≈0.18/sample and a tight compute budget (meta-R15),
  CS recommendation is a **fixed center (=0 under lit-derived `g_fixed`)** — keep the
  hierarchy flat; the per-leg amplitude anchor (§3) already provides the per-leg slack.

## Q3 — is the 0.167σ Fisher bias trustworthy, or compute the exact construction bias first?

**This is the CS lens's strongest recommendation: RUN the exact check before committing the
width. It is cheap and the linearization is known-wrong on the load-bearing axis.**

The spec locks the width using `bias(A_p)=+0.167σ` at the literature width. That number is
computed *linearly* (`diag_legb_slope_prior_tradeoff.py:165`):
```
bias(theta) = Σ_c [Cpost]_{theta,s_c} · Pp[s_c] · (-gap_c)
```
i.e. `(F+P)⁻¹ P Δs` — a first-order propagation through `J_s = ∂P/∂s` evaluated at `s=0`.
The script's OWN nonlinear cross-check (lines 133-140, results .txt lines 6-8) measures
`||J_s·gap − (P(gap)−P(0))|| / ||P(gap)−P(0)|| = 0.1294 (DESI) / 0.1524 (KS)` — **the
linearization is 13–15% off on the forward ΔP over the full sim-vs-lit gap.** The gap (0.76
LLS) is a large step and `(1+z)^s` is convex in `s`, so `J_s·gap` *under*-estimates the true
ΔP, which means the construction bias the prior must absorb is plausibly *larger* than
0.167σ. 0.167σ is already 83% of the 0.2σ gate — a 13–15% under-estimate could push the
honest bias over the gate.

**The check is a <30-line extension of an existing harness.** `diag_legb_zresolved_alpha_check.py`
already substitutes three α arrays (`a_median`, `a_exact`, `a_lit`) directly into
`predict_P_obs_on_leg` and recomputes the whitened residual. To get the EXACT construction
bias at the literature width you do not even need a new harness — `diag_legb_slope_prior_tradeoff.py`
already computes `P_0 = fwd(s0)` and `P_s = fwd(SIM_LIT_GAP)` per leg (lines 136-137). The
missing step is: instead of comparing the *forward ΔP* norms, propagate the EXACT
`ΔP = P_s − P_0` through the SAME `(F+P)⁻¹` map a Fisher-MLE would (`bias_exact =
(F+P)⁻¹ Jᵀ Cinv ΔP_exact` for the slope-pinned-truth construction), and read off the A_p
component at the literature width. That is ~15 lines added to the existing loop (you already
have `J`, `Cinv`, `Cpost`, `ΔP_exact` in scope). **Recommendation: do it; gate the locked
width on the EXACT A_p bias < 0.2σ, not the linear one.** If the exact bias is, say, 0.19σ,
the spec's "free headroom to go wider" claim flips to "already near the gate — do NOT widen,"
which materially changes §2's conclusion.

(Caveat for [bayesian]: the Fisher MLE-bias propagation is itself a linear-in-the-posterior
approximation; the *fully* honest number comes from the NUTS closure. But the exact-ΔP
Fisher bias is strictly better than the J_s·gap one and is the right pre-NUTS gate.)

## Q4 — per-leg anchor float on AMPLITUDE only, or also SLOPE?

**Amplitude-only, as specced — CONFIRM yes, and CS adds a reason beyond the sweep.** The
sweep (results .txt lines 14-35) shows the A_p bias is amplitude-driven (the per-class A_p
bias is dominated by LLS and is flat in the slope-prior width: +0.111σ at m=0.78 vs +0.119σ
at m=0.10), and σ(A_p) barely moves with the slope width (×1.04 over the whole scan). So a
per-leg SLOPE float would buy almost nothing on the binding A_p bias while adding 2 params/leg
(×2 legs = 4 extra slope nuisances) to an already 26-dim posterior with a known funnel + ESS
problem (meta-R6). **CS verdict: amplitude-only.** Additional CS reason: a per-leg slope
float is *degenerate with the per-leg amplitude float at the leg's z-mean* — both shift the
leg's α level — so the two would fight in the sampler (a banana), inflating warmup cost for no
identifiability gain. Keep slope GLOBAL (shared across legs, the physical lit-correction is
not survey-specific) + amplitude PER-LEG (absorbs survey selection). This matches the spec.

---

## Errors or risks in the LOCKED parts

**E1 (REAL — the spec's §1/§3 model cannot be implemented at the site it names).** This is
the one genuine error. The factorized α has TWO per-leg factors: `g_fixed,c(z)` (per-leg z
grid; if lit-derived it's still evaluated on each leg's z) and `exp(δ_{c,leg})` (explicitly
per-leg). But today `_legb_model` (closure_legb.py:450-456) builds ONE `alpha_hcd` of shape
`(n_zg, 3)` on the GLOBAL z grid and passes it to `_data_loglik_legcore`, which RE-SLICES it
per leg by nearest-z (`alpha_hcd[sel]`, line 403). There is no per-leg α path through that
contract. To apply `δ_{c,leg}` you must either:
  - **(route A, recommended)** move the α(z) assembly OUT of `_legb_model` and INTO
    `_data_loglik_legcore`'s per-leg loop: sample `a_pivot` (3,), `s` (3,), and
    `δ_leg` (3,) per leg in the model; pass them as scalars/small vectors; inside the leg
    loop build `alpha_leg = a_pivot[None,:]·exp(δ_leg)·g_fixed_leg·exp(lnz_leg[:,None]·s)`
    → `(n_z_leg, 3)`. This keeps `predict_P_obs_on_leg`'s (n_z,3) path UNCHANGED (the
    onboarding's key invariant), but it CHANGES `_data_loglik_legcore`'s signature (it now
    takes `a_pivot, s, δ_per_leg, g_fixed_per_leg` instead of a prebuilt `alpha_hcd`), and
    the `numpyro.deterministic("alpha_hcd_z", ...)` site at line 454 must move or split.
  - **(route B)** keep `alpha_hcd` global but make it `(n_leg, n_zg, 3)` — rejected: wastes
    the global grid, and `_data_loglik_legcore` still has to pick the leg axis. Route A is
    cleaner.
**Consequence:** this is NOT a "1-line multiply" / "no new shape path" change as the meta
synthesis (§C3, [CS] "low-risk") implied for the *sim-exact-shape* closing step. That earlier
assessment was for threading ONE global `g_zc` with NO per-leg δ. The spec's per-leg anchor
(§3) breaks that assumption. The change is still JAX-clean (no tracer control flow), but it is
a signature/contract change to `_data_loglik_legcore` and `run_legb`'s truth packing, ~40-60
lines, and MUST be golden-guarded.

**E2 (the truth-vector α shape collision — confirmed, matches CS-onboarding R4).**
`make_truth_from_sim` returns `w_c = median(...)` (3,) (closure_legb.py:283) and
`make_legb_mock` packs `truth_pack["alpha_hcd"] = w_c` (3,) (line 366), which becomes the α
COVERAGE truth (the 3 α columns in `_draws_matrix`, lines 502-505). With the new model the
SAMPLED quantities are `a_pivot` (3,) + `s` (3,) + `δ_leg` (3,×2 legs). The coverage truth
vector must rank `a_pivot` against `w_c` at the PIVOT z (z=3), NOT the z-median — so
`make_truth_from_sim` must additionally return `w_c_pivot = w_c[keep_rows where z≈3, 1:]`
(the sweep already does this, line 79: `wc_pivot = wc_perz[ip]`). And the new `s`, `δ_leg`
sites have NO sim truth (the sim has no "lit-correction slope") → their coverage truth is the
PRIOR mean (s=0, δ=0) by construction, which is a *degenerate* coverage test (you're checking
the sampler recovers a parameter whose true value you set to the prior center). **Flag for
[bayesian]:** the slope/anchor coverage is not a real null; only the θ and a_pivot coverage
is. Do not report s/δ coverage as a pass/fail.

**E3 (z-edge inflation `edge(z)` is host-side and fine, but applying it to σ_anchor is
new and unspecified in code).** The spec §2 says edge-inflate σ_s "and, for consistency, the
amplitude σ — extending the current DLA-only inflation to LLS/subDLA." The current DLA-only
inflation is `dla_inflate = 1 + clip(z − 3.5, 0)` (inference.py:104), applied ONLY to the DLA
σ and ONLY on the high-z side. Extending it to LLS/subDLA AND the low-z side AND to σ_anchor
touches `hcd_incidence_prior` (the prior builder), not just the closure. That's a real edit to
the production prior, with its own golden/regression implications (the α-prior μ/σ feeding
`alpha_hcd_mu/sigma` on `LegBCtx`). Not an error, but it is broader scope than "a per-z scalar
on σ_s" reads; flag it as a prior-builder change that needs its own unit test (the (μ,σ) at a
z inside vs outside the trust window).

Otherwise: **none found** in the locked width logic itself — the ×1.04 / per-class
decomposition / endpoint math in the sweep is correct (I checked `metrics`/`post_cov`, lines
153-169: `(F+P)⁻¹`, slope-block prior, the bias propagation sign and indices are right).

## Additional findings from this lens

**F1 — the golden guard does not exist; I confirmed it.** `ls tests/golden/` → no such
directory. The spec §5 correctly lists it as a prerequisite "out of scope here," but given
E1 (this is a real contract change, not a 1-line multiply), the golden guard is now MORE
load-bearing than the onboarding assumed. **Exactly what to freeze:** at a FIXED
`(θ9, τ₀_vec, α=(3,))` — pick the test fixture's values from `tests/test_data_likelihood.py`
(`alpha_hcd=[0.06,0.02,0.003]`, the existing `_ctx()` θ/τ₀) — call `predict_P_obs_on_leg` on
BOTH legs (DESI + KS) with the production `sigma_zb`/`rho_zb` and save `(P_model, C_total)`
for each leg to `tests/golden/legb_lf_golden.npz`. A new test `test_legb_golden_unchanged`
loads it and asserts `jnp.allclose(P_model, golden_P, rtol=1e-12, atol=0)` and same for
`C_total`. Freeze the (3,)-broadcast path (NOT (n_z,3)) because that is the legacy path the
refactor must not move. After the refactor, the legacy (3,) call must still hit the same code
and reproduce byte-for-byte. **This is the single highest-leverage CS action and is a hard
gate before E1's contract change.**

**F2 — x64 ordering is safe for the new code path.** All entry points
(`closure_legb.py:37`, the sweep `diag_legb_slope_prior_tradeoff.py:31`, the harness
`diag_legb_zresolved_alpha_check.py:6`) `import hcd_analysis.emulator` before `import jax`,
so x64 is live before any array. The new `g_fixed` precompute (Q1) and the new `s`/`δ` sites
inherit this. No new x64 fragility IF the precompute lives in `build_legb_ctx` (already
post-import) or a new module that imports the package first. Add the package-import-first
comment to any new file.

**F3 — `jnp.exp(lnz·s)` is safe.** `lnz = ln((1+z)/(1+z_p))` over z∈[2.2,4.6] is in
[−0.215, +0.40]; with `s` capped by a `Normal(0, σ_s≈0.65)` prior at, say, ±4σ ≈ ±2.6, the
exponent is in [−0.56, +1.04] → `exp` in [0.57, 2.83]. No overflow, no underflow, smooth and
fully differentiable in `s`. The `exp(δ)` amplitude factor with `δ ~ Normal(0, 0.30)` at ±4σ
is `exp(±1.2)` ∈ [0.30, 3.3] — also safe. No clamp needed; the priors bound it. (Contrast
the `reconstruct_P_filt_jax` exp which has a ~700 margin — these are far safer.)

**F4 — no new NaN-into-interp trap.** The α factorization is multiplicative and feeds
`predict_P_obs` → `predict_P_obs_on_leg`'s `jnp.interp(k_sub, cache_k, P_cache)` (line 351)
on the model side. The α value never enters an interp ABSCISSA (it's a coefficient on P_cls),
so the jax-traps #29 interp-slope-NaN class is not reachable through this change. The
C_emu coef `[1−Σα, α]` (predict.py:142) now carries `α_c(z)` per z — already handled (the
sweep exercises this with `sigma_zb`/`rho_zb` on, runs finite). Confirmed by the harness
running clean (no NaN warnings except the KS empty-low-k-slice cosmetic divide, which is the
script's own `lowk.sum()==0` print, not the forward).

**F5 — `_data_loglik_legcore` nearest-z slice has no z-tolerance (matches CS-onboarding R6).**
Line 403 `alpha_hcd[sel]` with `sel = argmin|zg − zz|` and NO tolerance — a leg z far from any
global z would pull the wrong α row. Safe today (DESI/KS z ⊂ global), but under route A the
α is built per-leg from `g_fixed_leg` directly so this nearest-z slice DISAPPEARS for α
(it stays only for τ₀). Route A actually REMOVES this latent risk for the α path — a small
correctness win, worth noting in the plan.

**F6 — the sweep's bias uses `Cpost` from the SLOPE-FREE Fisher but the operating C_total
from `s=0`.** Lines 123-131 build `Ctot` at `al0 = alpha_of_s(..., s0)` (s=0) then the Fisher
`F` at the same point. The bias (line 165) uses the marginalized `Cpost`. This is internally
consistent (a local linearization at s=0), but it means the locked bias is a *local* number
at the de-biased point — which is exactly why Q3's full-gap nonlinearity matters (the truth is
a full gap away from s=0, not local). Reinforces the Q3 recommendation.

## Concrete recommendations (numbered, actionable)

1. **Implement the golden guard FIRST (F1), before ANY α refactor.** Freeze
   `(P_model, C_total)` on BOTH legs at the test fixture's fixed `(θ, τ₀, α=(3,))` with
   production `rho_zb`, to `tests/golden/legb_lf_golden.npz`; add `test_legb_golden_unchanged`
   asserting `allclose(rtol=1e-12, atol=0)`. Hard gate.

2. **Rewrite the spec's §1/§3 to put α(z) assembly in `_data_loglik_legcore`'s per-leg loop
   (route A), not in `_legb_model` (E1).** The model samples `a_pivot (3,)`, `s (3,)`,
   `δ_leg (3,)` per leg; the leg loop builds `alpha_leg = a_pivot·exp(δ_leg)·g_fixed_leg·
   exp(lnz_leg·s)` → `(n_z_leg,3)`, which `predict_P_obs_on_leg` already accepts UNCHANGED.
   Update `run_legb`/`_draws_matrix` truth packing accordingly (E2). State this is a
   signature change, ~40-60 lines, not a 1-line multiply.

3. **Store `g_fixed` as a frozen `(n_z_leg,3)` constant on `LegBCtx` per leg (Q1 = lit-derived),
   computed once in `build_legb_ctx` via `dndx_wc.alpha_from_dndx_law(A_lit, γ_lit, Xbar(z), z)`
   normalised to its z=3 value.** Do NOT recompute inside the traced model. Add a ctx field
   `g_fixed_per_leg: dict`. If Q1 resolves to sim-`w_c` instead, thread it per-mock as a
   closed-over constant (route from CS-onboarding R3).

4. **RE-RUN `diag_legb_slope_prior_tradeoff.py` with the production `g_fixed` (one line: 90)
   before locking the width**, because the locked ×1.04 / 0.167σ were measured under the
   sim-`w_c` `g_fixed`, not the production one (Q1 trap).

5. **EXTEND the same sweep to compute the EXACT (non-linearized) construction A_p bias at the
   literature width and gate the width on `bias_exact(A_p) < 0.2σ`** (Q3). ~15 lines: you
   already have `P_s`, `P_0`, `J`, `Cinv`, `Cpost` in scope — propagate `ΔP_exact = P_s − P_0`
   through `(F+P)⁻¹ Jᵀ Cinv` instead of `J_s·gap`. If the exact bias ≥ ~0.19σ, do NOT widen
   per §2's "headroom" claim — re-examine.

6. **Keep the slope center FIXED at s=0 under a lit-derived `g_fixed` (Q2); no hyper-prior.**
   Avoids a second hierarchy level / funnel on the budget- and ESS-constrained sampler.

7. **Amplitude-only per-leg anchor; slope stays GLOBAL (Q4).** Per-leg slope is degenerate
   with the per-leg amplitude at the leg z-mean (a banana) and buys ~0 on the binding A_p bias.

8. **Add the edge-inflation as a PRIOR-BUILDER change with its own unit test (E3):** extend
   `hcd_incidence_prior` (or a wrapper) to two-sided edge inflation on LLS/subDLA σ and σ_s;
   unit-test the (μ,σ) inside vs outside [2.5,3.5].

9. **Unit tests gating the refactor (in addition to the golden guard):**
   (a) `test_alpha_factorized_shape`: route-A α builder returns `(n_z_leg,3)` for each leg;
   (b) `test_alpha_factorized_finite_and_grad`: `_data_loglik_legcore` finite + grad-finite in
   `(a_pivot, s, δ_leg)` on the real cov (mirror `test_data_loglik_finite_and_grads_on_real_cov`);
   (c) `test_alpha_s0_delta0_equals_g_fixed`: at `s=0, δ=0`, `alpha_leg == a_pivot·g_fixed_leg`
   exactly (rtol 1e-12) — pins the de-biased operating point;
   (d) `test_alpha_g_fixed_precomputed_no_trace`: assert `g_fixed` is a concrete array on ctx
   (not a tracer) — guards Q1's precompute discipline;
   (e) keep the existing (3,)-broadcast tests passing UNCHANGED (the golden guard backs this).

10. **State in the spec that s/δ coverage is NOT a valid null (E2):** the sim has no
    lit-correction slope, so s/δ truth = prior center by construction; only θ and a_pivot
    coverage is interpretable. Flag to [bayesian].
