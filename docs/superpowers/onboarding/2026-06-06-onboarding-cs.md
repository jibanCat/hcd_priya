# CS / code-correctness / JAX-numerics onboarding report — 2026-06-06

Lens: implementation correctness + JAX traps. The user came from PyTorch; the project
has a disciplined JAX-traps log (`docs/superpowers/jax-traps-log.md`, 30 numbered
entries) and a "every JAX task gets an independent JAX-specialist review" working
practice. This report verifies what is actually correct in the code on disk, what is
test-backed, and what the documented "closing step" (per-z `alpha(z)` shape match) risks
from a code-safety angle.

Verification performed this session: ran `pytest tests/test_data_likelihood.py
tests/test_xclass_cemu.py` → **28 passed in 137 s (exit 0)** — the handoff's "28 pass"
claim is current. Read every file in the brief's code list end-to-end.

---

## 1. What this subsystem does (CS/JAX lens)

The Phase-C likelihood turns the merged Equinox emulator into a JAX-pure,
differentiable log-likelihood that numpyro/NUTS can sample. There are TWO parallel
likelihood paths that must not be conflated:

- **Leg-A (cache-grid) path:** `inference.py` (`log_lik_single_z` / `log_lik_multiz` /
  `log_posterior_single_z`) + `closure_ctx.py` (frozen `Ctx` eqx.Module) +
  `sampler_numpyro.py` (`numpyro_model`) + `closure_sbc`/`closure_mocks`. Evaluates the
  likelihood on the cache's own 172 angular-k grid; this is where the **SBC rank-uniform
  null** is valid.
- **Leg-B (real-grid) path:** `data_likelihood.py` (`load_desi_leg`/`load_ks_leg`/
  `predict_P_obs_on_leg`/`data_loglik`) binds the same emulator forward to the OBSERVED
  DESI DR1 (85 angular-k × ≤12 z) + KODIAQ-SQUAD (13 k × 14 z) grids, applies published
  cuts + covariances, and is driven by `closure_legb.py` (held-out-sim mock truth +
  cosmic-only noise). The Leg-B null is NOT rank-uniform (truth is a held-out sim, not an
  emulator draw) → coverage≥nominal + bias≈0 is the only valid verdict (enforced in
  `_aggregate_legb`, line 655-676; ECDF tagged DIAGNOSTIC-ONLY).

The forward model itself (`predict.py`) is the CORRECTED HCD form (redesign 2026-06-04):
`P_obs = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)`, with per-class P_filt reconstructed as
`P_filt = exp((m̂·sig_marg + mu_marg) + sig_cosmo·r̂)` — a θ-blind baseline `m̂`
(`BaselineHead`) plus a σ_cosmo-whitened residual `r̂` (`HeadB`), so `∂logP/∂θ =
sig_cosmo·∂r̂/∂θ` (the baseline is structurally θ-blind, jax-traps #14).

C_emu (emulator-error covariance) is diagonal-in-k by construction; it has TWO forms
selected statically: per-class diagonal (`sigma_zb`, `Σ_c coef_c²·σ_c²·P_c²`) or
opt-in cross-class 4×4 (`rho_zb`, `Σ_{c,c'} coef_c·coef_c'·ρ_cc'·P_c·P_c'`). The
cross-class form is the production C_emu and fixes the diagonal under-sizing
(whitening 1.28→0.93).

---

## 2. Load-bearing design decisions & WHY (with pointers)

### x64 enforcement — the bedrock
- `hcd_analysis/emulator/__init__.py:11-12` calls `jax.config.update("jax_enable_x64",
  True)` on PACKAGE IMPORT, before any array is created. The structural identities
  (Σ_c w_c·P_filt sum-to-1, cache reproduction to ~1e-15) are bit-level and die under
  float32 (jax-traps #1, HIT). Every Phase-C module re-asserts it at import:
  `data_likelihood.py:40`, `inference`-consumers via `closure_ctx.py:33`,
  `sampler_numpyro.py:33`, `closure_legb.py:56`. The assert reads
  `jax.config.read("jax_enable_x64")` and fails LOUD if a caller imported jax first.
- CS note: the assert pattern is correct (it checks the live config, not a local flag).
  The one fragility is ordering — any new entry-point that does `import jax` before
  `import hcd_analysis.emulator` silently runs float32 until the assert. All current
  entry-points import the package first (verified: `closure_legb.py:37` comment "enables
  x64 BEFORE jax").

### The θ-blind baseline split (normalization redesign)
- `model.py` `BaselineHead.__call__` (line 125-135): input is ONLY `jnp.stack([z, tau0])`
  — it NEVER sees the latent. `Emulator.__call__` (line 241-247) passes `z = x[..., 9]`
  (NOT a param) and the latent only to HeadA/HeadB. This is what makes `∂m̂/∂θ ≡ 0`
  EXACTLY (jax-traps #14 verified `jacfwd==0.0` on the real param axis). Load-bearing
  because the baseline's mis-fit is amplified by the small σ_cosmo, so it MUST stay
  identifiable; a refactor that fed the latent into BaselineHead would pass every
  forward-shape test and silently break identifiability.
- `predict.py::reconstruct_P_filt_jax` (line 29-39): `logP = (base·sig_marg + mu_marg) +
  sig_cosmo·resid`, then `exp`. The exp is safe by a wide margin (real recon max ~1.7e3 vs
  f64 overflow ~1.8e308; jax-traps #14).

### Static fields for everything that sets a shape
- `n_k`, `n_basis`, `n_layers`, `width` are `eqx.field(static=True)` on `HeadB`/
  `BaselineHead`/`DeltaHead` (model.py:91-94, 171-172; multifidelity.py:279-280). A bare
  `int` leaf breaks plain `value_and_grad` (jax-traps #8, HIT — `grad requires inexact`)
  and bloats the differentiable pytree. Static = part of the jit cache key, reconstructed
  from the skeleton on deserialise. `coherent_debias_term` (model.py:369) reads
  `int(batch["n_cells"])` as `num_segments` — that MUST be a python int, not a traced
  leaf (jax-traps #25, HIT — `ConcretizationTypeError`); fixed at the `make_batch`
  source.

### The NaN-through-where / interp-slope traps (the NUTS killers)
- `likelihood.py::sigma_at_tau0` (line 98-122) and `rho_at_tau0` (line 125-151):
  `jnp.nan_to_num(sigma_zb, nan=0.0)` is applied to the YDATA **before** `jnp.interp`,
  NOT to the output after. Interpolating between two NaN y-points gives value=NaN
  (sanitizable) AND slope=NaN (NOT sanitizable by a downstream `nan_to_num`) →
  `jax.grad` wrt τ₀ returns NaN → NUTS dies on the first leapfrog (jax-traps #29, HIT,
  found on the REAL error_vector.npz with 4 all-NaN-over-τ₀ rows per z-band). This is the
  single most dangerous trap in the likelihood and it is correctly guarded. Mirrored
  exactly in the 4×4 cross-class path.
- `inference.py::log_lik_single_z` (line 196-198): `r = nan_to_num(P_data) - P_obs`, then
  `valid_k` masks out-of-range bins — sanitise the input to the subtract, not the output.
- `data_likelihood.py:286, 290`: `jnp.nan_to_num(rho_ck)` / `jnp.nan_to_num(sigma_ck)`
  inside `_emu_var_on_cache` — out-of-range cache cells σ/ρ→0 (zero variance, not NaN).

### gaussian_loglik SPD jitter
- `likelihood.py::gaussian_loglik` (line 154-175): adds `jitter·mean(diag C)·I` BEFORE the
  Cholesky. `jnp.linalg.cholesky` RETURNS NaN (does not raise) on a non-SPD / zero-diagonal
  C, silently poisoning value AND gradient. A zero diagonal arises when a bin has zero
  cosmic var AND zero emu var (NaN-zeroed σ row). The logdet is carried (`Σ log diag(L)`)
  because C depends on (θ,τ₀) — omitting it biases τ₀ toward larger-σ regions. Correct.

### The cross-class einsum (the brief's named concern)
- `data_likelihood.py:285` and `inference.py:148`: `jnp.einsum("c,d,cdk,ck,dk->k", coef,
  coef, jnp.nan_to_num(rho_ck), P_cls, P_cls)`. This is `emu_var(k) = Σ_{c,c'}
  coef_c·coef_c'·ρ_cc'(k)·P_c·P_c'`. Because ρ is a sample covariance (SPD), this is
  `coefᵀ(P∘ρ∘P)coef ≥ 0` GUARANTEED — emu_var can never go negative. The diagonal
  `ρ_cc=σ_c²` recovers the diagonal `Σ coef²σ²P²` form EXACTLY (pinned by
  `test_leg_xclass_diag_equals_diagonal_path_at_band_centre`, rtol=1e-10). The fixed
  `c,d,cdk,ck,dk` indices are NOT an ellipsis form, so they can't silently mis-broadcast
  (contrast jax-traps #11/#21, which were about ellipsis batching in `likelihood.py`'s
  single-alpha form).

### The per-z loop + alpha shape handling in predict_P_obs_on_leg
- `data_likelihood.py:294-378`. `alpha_hcd = jnp.asarray(...)` then `alpha_z = alpha_hcd if
  alpha_hcd.ndim == 1 else alpha_hcd[iz]` (line 346). The `.ndim` test is a PYTHON branch
  on a STATIC shape (decided at trace time, not on a tracer value) — JAX-clean. (3,)
  broadcasts to all z; (n_z,3) is per-z incidence. This is the backward-compatible hook
  the z-resolved-α fix already added (handoff: "(3,) still broadcasts → 28 tests pass").
- The loop is a PYTHON `for iz in range(leg.n_z)` (line 338) using `.at[rows].set(P_z)`
  immutable updates (line 358, 375). `leg.n_z` is a python int (host-side DataLeg field),
  so the loop is unrolled at trace time — correct, but it means the traced graph grows
  with n_z (≤12 for DESI, ≤14 for KS — fine; this is why the per-step cost is dominated by
  the 681×681 Cholesky-VJP, not the loop). `rows = np.where(z_idx == iz)[0]` is host-side
  numpy (z_idx is numpy), so the row gather indices are concrete. The `jnp.interp(k_sub,
  cache_k, P_cache)` (line 351) maps the cache grid onto the leg's bin centres — the model
  P_obs(k) is smooth so linear interp to bin centre is the documented choice.
- `data_likelihood.py:369-374`: when metals/resolution are ON, `ev_z` (emu var) is scaled
  by the SAME multiplicative factor SQUARED (variance units) — correct unit handling.

### make_truth_from_sim z-median collapse (the closing-step target)
- `closure_legb.py:280-283`: the return packs `w_c=np.median(w_c[keep_rows, 1:], axis=0)`
  — a (3,) z-MEDIAN of the sim's per-row structural weights. This is the collapse the
  documented closing step must undo: the per-z `w_c(z)` array (`d["w_c_cache"][rows,1:]`,
  shape (nZs,3)) IS available at the `keep_rows` indices but is thrown away in favour of
  the median. The closure forward then applies `α_pivot · literature-slope-shape(z)`
  (`closure_legb.py:453-454`) which is too shallow vs the sim's actual ~4.5× rise.

### The z-resolved α(z) site (already half-built)
- `closure_legb.py:450-456` in `_legb_model`: samples `alpha_pivot = [a_lls,a_sub,a_dla]`
  at z=3, then `shape_zg = ((1+zg)/(1+HCD_Z_PIVOT))^HCD_LIT_OVER_SIM_SLOPE` (the literature
  dN/dX power-law, `inference.py:74` = (0.95,0.15,0.40)), and `alpha_hcd =
  alpha_pivot[None,:]·shape_zg` → a (n_zg,3) per-z incidence threaded into
  `_data_loglik_legcore`. `_data_loglik_legcore` (line 403) already handles both (3,) and
  (n_z,3) via `np.ndim(alpha_hcd)==1`.

### MF layer: frozen LF + stop_gradient (NOT static field)
- `multifidelity.py::_freeze` (line 153-166): the LF backbone is frozen by
  `eqx.partition(model, eqx.is_array)` → `tree_map(stop_gradient)` → `eqx.combine`, NOT by
  `eqx.field(static=True)`. A static field holding a sub-Module does NOT freeze it
  (equinox recurses into the pytree and its arrays stay dynamic leaves; the static-captured
  copy desyncs from the differentiated leaves so an in-forward stop_gradient doesn't bind)
  — jax-traps #26, HIT. `lf_model` is a NORMAL dynamic field; `lf_norm` (numpy dict) is the
  genuine static field. `test_mf_lf_backbone_is_frozen` pins grad.lf_model leaves = 0.
- `P_mf` docstring (line 529-538): never `jax.jit(mf.P_mf)` (bound-method jit hashes the
  module as a static arg → `unhashable type: 'list'` on DeltaHead.layers); use
  `eqx.filter_jit` or a lambda (jax-traps #27, #28).
- **MF wiring status: built + validated but NOT wired into the likelihood.** `data_loglik`
  / `predict_P_obs_on_leg` call `predict.predict_P_obs` (the LF path) directly — there is
  no `mf=` kwarg yet. The wiring is fully planned (plan §1.1-1.6) but unimplemented. This
  is the documented "silent inheritance" risk: the closure currently runs LF-only; the MF
  resolution correction (`delta_mode='none'` drops PRIYA's δ(θ), carried in C_emu) is NOT
  in the certified path. Flag for the main agent: the Leg-B certification is of the LF
  likelihood, and MF must be re-certified after wiring.

---

## 3. Verified vs assumed

### Verified (test / figure / proof behind it)
- **28 tests pass** (ran this session, exit 0): loaders shape/cuts/SPD, binding finite+SPD
  each leg, `jnp.interp` cache-grid identity (`test_interp_roundtrips_cache_grid_to_itself`,
  atol 1e-12), metal factor identity@0 + oscillation + differentiable, resolution factor
  identity + grad, `data_loglik` finite + finite grads in (θ9,τ₀,α,a_SiIII,b_res) on the
  REAL cov, block-diagonal == Σ legs (rtol 1e-10), Becker13 closed form + differentiable,
  cross-class diag==diagonal at band centre (rtol 1e-10), positive off-diagonal raises
  emu_var, x-class differentiable in (τ₀,α), `rho_zb_per_leg` plumbing reaches each leg's
  C_emu + lowers χ².
- **Backward-compat for the (3,)→(n_z,3) α change:** verified at the unit level —
  `predict_P_obs_on_leg`'s `alpha_z = alpha_hcd if ndim==1 else alpha_hcd[iz]` and
  `_data_loglik_legcore`'s `np.ndim==1` branch both keep the legacy (3,) path; all 28 tests
  (which use (3,) α) pass unchanged. The forward-only Δ-swap is verified by
  `scripts/diag_legb_zresolved_alpha_check.py`: OLD z-median −0.78σ → LIT-shape −0.24σ →
  EXACT per-z w_c +0.03σ (read the script: it directly substitutes the three α arrays into
  `predict_P_obs_on_leg` and recomputes the whitened residual).
- **C_emu sub-dominant on the real grid** (handoff §3a item 1): full-z median C_emu/C_data
  = 0.0143 DESI / 0.0114 KS, low-k max 0.52 DESI.
- **jax-traps #1-#14, #25-#29 each have a named regression test** (listed in the log).

### Assumed / stated but NOT independently re-derived here
- The whitening 1.28→0.93 number is **in-sample** (the 8 fold-0 truth sims are ~1/8 of the
  pool that built `rho`). The de-circularized `error_vector_xclass_holdout0.npz` (folds 1-7)
  is built (off-diag clean-LLS +0.917 vs +0.926 in-sample → mild circularity) but the
  Leg-B coverage under it has NOT been run to a verdict (the N=50 pilot is the next step;
  smoke is PATH-only at the tiny N).
- The +0.03σ "EXACT per-z" result is **forward-only** (no NUTS). The full closure with the
  closing step + DLA-mask + correlated C_emu has NOT been run to a coverage/bias verdict.
- The block-diagonal (no DESI–KS cross-covariance) is a stated LYA-CONSULT assumption
  (`data_likelihood.py:407`).
- **No golden-regression guard exists on disk.** `tests/golden/` does not exist; the
  handoff §RESUME step 1 ("freeze (P_model,C_total) → tests/golden/legb_lf_golden.npz,
  assert allclose rtol=1e-12") is a PLAN, not implemented. The byte-identity of the
  unchanged LF path currently rests only on the 28 functional tests passing.

---

## 4. Risks / open questions (ranked; flag anything that could bias the real fit)

**R1 (HIGH, could bias the real fit) — the closing-step change has NO regression guard.**
The documented step threads a per-z `g_zc(z)` into the forward, replacing the fixed
`shape_zg` (closure_legb.py:453). There is no golden file pinning the current `(P_model,
C_total)` on the unchanged LF path, so a refactor that accidentally changes the (3,)
broadcast path (e.g. mis-indexing `alpha_hcd[iz]`, or changing the einsum) would not be
caught byte-for-byte — only by the looser functional asserts. **Recommendation: implement
the golden guard FIRST (handoff RESUME step 1) before touching the α threading.** This is
the single most important CS action; freeze `(P_model, C_total)` for both legs at a fixed
(θ,τ₀,α=(3,)) and assert `allclose rtol=1e-12` on the legacy path after the change.

**R2 (HIGH, physics-of-certification) — MF is not wired into the certified likelihood.**
The Leg-B cert is of the LF likelihood; `data_loglik` never calls the MF forward. When MF
is wired (plan §1), the resolution correction `delta_mode='none'` SILENTLY inherits the
LF C_emu scaled by an MF-LOSO factor — and the dropped δ(θ) must be carried in C_emu
(plan §0b-2). A real fit run with MF but certified only on LF would have an
under-validated high-k channel. Flag: re-run the closure after MF wiring; do not assume
the LF cert transfers.

**R3 (MEDIUM) — where to thread `g_zc`: ctx vs model closure.** Two routes for the
closing step. (a) Thread `g_zc` (n_zg,3) through the truth pack → `run_legb` →
`_run_nuts_legb` → `_legb_model` and replace `shape_zg` with the passed `g(z)`. This keeps
`g_zc` a host-side numpy constant captured by the model closure (like `mock_legs`,
`dla_core_per_leg`) — JAX-clean, no new traced leaf. (b) Put it on `LegBCtx`. Route (a) is
safer: `g_zc` is per-MOCK (different sims have different w_c(z)), so it belongs with the
mock, not the frozen ctx. **CS trap to avoid:** `g_zc` must be passed as a numpy array
closed over (or `jnp.asarray`'d once outside the traced fn), NOT computed from
`d["w_c_cache"]` inside the numpyro model — and `alpha_hcd = alpha_pivot[None,:]·g_zc`
keeps the SAME (n_zg,3) shape the downstream already handles, so no new shape path. The
multiply is fully differentiable in `alpha_pivot` (g_zc is a constant), so the sampler's
∂logL/∂α is unaffected. This is a near-zero-risk change IF the golden guard is in place.

**R4 (MEDIUM) — make_truth_from_sim still returns the z-median w_c.** The closing step
needs the per-z `w_c[keep_rows,1:]` (shape (nZs,3)) added to the truth dict, not just the
median `w_c` (line 283). The rows are already kept (`keep_rows`), so this is a one-line
addition (`w_c_perz=w_c[keep_rows,1:]`) — but the existing `w_c` (3,) is consumed by
`make_legb_mock` (line 366, `truth_pack["alpha_hcd"]`) and `run_legb`'s `truth_vec`
(line 566) for the RANK/bias of the α PARAMETER (the sampled pivot, (3,)). Changing the
truth vector's α from (3,) to (n_zg,3) would break `_draws_matrix` / coverage (which expect
3 α columns). So: keep the (3,) `w_c` (the pivot-z truth for the α coverage) AND add the
per-z `w_c_perz` separately for the forward shape. Do NOT conflate them.

**R5 (LOW-MEDIUM) — the `_data_loglik_legcore` per-leg z-mean DLA core is an MVP.** The
docstring (closure_legb.py:384-394) admits it passes each leg a z-MEAN core, not per-z.
The DLA sector is uncertified by Leg-B without arm A3 anyway, and the forward/truth cores
cancel in the emu-error sizing — but this is a documented approximation, not exact.

**R6 (LOW) — `np.argmin(np.abs(z_global - zz))` nearest-z mapping is everywhere**
(closure_legb.py:400, 420; data_likelihood.py:420). It is host-side numpy on concrete z, so
JAX-safe, but it silently maps a leg z to the nearest global z even if no good match exists.
The `z_tol=0.15` guard (closure_legb.py:314) catches drops in the mock, but `data_loglik`'s
`sel` (line 420) has no tolerance — a leg z far from any global z would pull a wrong τ₀.
In practice DESI/KS z ⊂ global by construction, so it's safe; flag for any new leg.

**R7 (LOW) — exp overflow / amplitude head.** `reconstruct_P_filt_jax` exps `logP`; safe
today (margin ~700 in the exponent). If a future fit pushes θ to a regime where the
residual head extrapolates wildly, `exp` could overflow to inf and poison the Cholesky.
The unit-box soft prior (`inference.unit_box_logprior`) / numpyro Uniform bijector keeps θ
in-box, so this is bounded — but it is an assumption, not a clamp.

---

## 5. Pointers for the main agent (the files/functions you MUST know)

1. **`hcd_analysis/emulator/__init__.py:11-12`** — x64 enabled on package import; ALWAYS
   `import hcd_analysis.emulator` before `import jax` in any new entry-point, or identities
   silently run float32.
2. **`hcd_analysis/emulator/data_likelihood.py::predict_P_obs_on_leg` (line 294-378)** —
   the Leg-B model→leg binding; the per-z python loop, the `alpha_hcd.ndim` (3,)/(n_z,3)
   branch (line 346), the `jnp.interp` onto leg k (351), the cross-class einsum
   (`_emu_var_on_cache`, 285). The closing-step change lives upstream of this; this fn
   already accepts per-z α.
3. **`hcd_analysis/emulator/closure_legb.py`** — `_legb_model` (line 432-456, the
   `shape_zg`→`g(z)` swap point), `make_truth_from_sim` (line 214-283, the z-median collapse
   at 283 to undo), `run_legb`→`_run_nuts_legb`→`_legb_model` call chain (508-588) where a
   per-mock `g_zc` would thread as a closed-over numpy constant.
4. **`hcd_analysis/emulator/likelihood.py::sigma_at_tau0 / rho_at_tau0` (line 98-151)** — the
   NaN-before-interp guard that keeps NUTS alive (jax-traps #29). Touch with extreme care;
   any τ₀-interp must sanitise YDATA before `jnp.interp`, never the output after.
5. **`hcd_analysis/emulator/likelihood.py::gaussian_loglik` (154-175)** — SPD jitter before
   Cholesky (Cholesky returns NaN, not raises) + the carried logdet. Used by both legs.
6. **`hcd_analysis/emulator/predict.py::predict_P_obs (83-103) / reconstruct_P_filt_jax
   (29-39)`** — the corrected HCD forward `P_clean + Σ α_c·(P_c−P_clean)` and the
   baseline+residual reconstruction. `∂P_obs/∂α_c = (P_c − P_clean)`.
7. **`hcd_analysis/emulator/model.py::BaselineHead (65-135) / Emulator.__call__
   (241-247)`** — the θ-blind split (`z=x[9]`, latent never to baseline); the structural
   identifiability the gradient gate proved (jax-traps #14).
8. **`hcd_analysis/emulator/multifidelity.py::_freeze (153-166) / MultiFidelity (405-542)`**
   — LF frozen via stop_gradient not static field (jax-traps #26); MF is BUILT but NOT
   wired into `data_loglik` (the silent-inheritance flag, plan §1).
9. **`docs/superpowers/jax-traps-log.md`** — read entries #1, #7, #8, #14, #25, #26, #27,
   #29 before writing any JAX in this repo. They are all HIT or GUARDED with a named test.
10. **`docs/SESSION_HANDOVER_2026_06_05.md` RESUME block (line 8-39)** — the unified fix
    order: golden guard FIRST, then z-resolved α(z), then DLA-mask, then α_DLA Gaussian-at-0,
    then correlated C_emu. The closing step is item 2 of that list; the +0.03σ result is the
    forward-only verification.

### CS bottom line on the closing-step change
Threading the truth's exact per-z `g_c(z) = w_c(z)/w_c(z_p)` into `_legb_model` (route B) is
a **low-risk, JAX-clean change** PROVIDED: (a) the golden-regression guard is implemented
first (R1); (b) `g_zc` is passed as a closed-over numpy/jnp constant per-mock, not computed
inside the numpyro model from a fresh cache read (R3); (c) `make_truth_from_sim` ADDS a
per-z `w_c_perz` field rather than replacing the (3,) `w_c` the α-coverage truth vector
needs (R4); (d) the multiply keeps the existing (n_zg,3) shape so no downstream shape path
changes. No JAX trap is introduced — there is no new tracer-dependent control flow, no new
NaN-into-interp, no new shape-setting traced leaf. The one thing a test must catch is a
regression in the (3,) broadcast path (the golden guard) and a shape mismatch if `g_zc` is
threaded at the wrong rank (add a unit test: `_legb_model` with a passed `g_zc` of shape
(n_zg,3) produces `alpha_hcd` of shape (n_zg,3) and the loglik is finite + grad-finite in
`alpha_pivot`).
