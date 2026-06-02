# JAX / Equinox traps log — Phase-2b emulator

A running log of JAX-specific traps encountered (or deliberately guarded against)
while building the `hcd_analysis/emulator/` JAX/Equinox emulator. Read this before
writing new JAX code in this project. **Working practice:** every task that writes
JAX code gets an independent JAX-specialist review/unit-test pass (a dedicated
background agent) on top of the normal spec + code-quality review.

Status legend: **HIT** = actually bit us; **GUARDED** = known trap, designed around
it preemptively (test pins it); **WATCH** = not yet relevant, flagged for later.

---

## 1. float32 by default vs required float64 — **HIT**
- **Where:** Task 5 (`hcd_analysis/emulator/dndx_wc.py`), surfaced by the implementer.
- **Symptom:** the telescoping `w_c` sum-to-1 assertion (`|Σw − 1| < 1e-12`) and, by
  extension, the structural identity `Σ_c w_c·P_filt == P_tier_p` fail — error sits at
  ~1e-7 instead of <1e-12.
- **Cause:** JAX silently defaults to **float32** regardless of the dtype you wrote.
  Our identities are bit-level (the cache reproduces PRIYA to ~1e-15), so float32
  destroys them.
- **Fix:** enable double precision **before any array is created** —
  `jax.config.update("jax_enable_x64", True)` in `hcd_analysis/emulator/__init__.py`
  (runs on package import) **and** in `tests/conftest.py` (collection time). Commit `1469be0`.
- **Lesson:** for any bit-level / high-precision identity, enable x64 globally and
  assert dtype in a test. Don't rely on writing `np.float64` — JAX ignores it under x32.

## 2. NaN gradient through `jnp.where` (the "double-where trap") — **GUARDED**
- **Where:** Task 10 masked loss (`masked_mse` in `model.py`) — designed in the plan;
  ~2% of P1D bins are NaN above each row's native Nyquist.
- **Symptom (if unguarded):** the forward loss is finite, but `jax.grad` returns **NaN**
  gradients → training diverges with no obvious cause.
- **Cause:** reverse-mode autodiff differentiates **both** branches of
  `jnp.where(mask, f(x), 0.0)`. If `f(x)` is NaN/inf on the masked-out elements
  (here: NaN target bins), the chain rule produces `0 * NaN = NaN` that flows back.
  Same failure for `sqrt(0)`, `log(0)`, `x/0` even when masked away in the forward pass.
- **Fix:** sanitise inputs to finite **before** the masked branch, mask after, and floor
  every masked-mean denominator:
  ```python
  target_safe = jnp.nan_to_num(target)
  diff = jnp.where(mask, pred - target_safe, 0.0)
  denom = jnp.maximum(mask.sum(), 1.0)
  ```
  A `jax.grad` finite-gradient test on a NaN-containing batch pins it.
- **Lesson:** never let a NaN/inf reach a `where`/masked op even on the discarded branch.

## 3. `0 * NaN` poisons the structural sum (empty classes) — **HIT (caught in review)**
- **Where:** Task 2 loader collapse (`_collapse_p1d`), caught by the spec reviewer.
- **Symptom:** an empty coarse class with a NaN P1D would, in `P_tier_p = Σ_c w_c·P_filt`,
  give `w_c=0 · NaN = NaN` and poison the entire total.
- **Cause:** the plan's docstring wrongly said empty class → NaN. The cache convention
  (and the safe choice) is empty class → **0.0** finite.
- **Fix:** empty coarse class (zero counts) collapses to 0.0, NOT NaN; genuine
  above-Nyquist NaN (all fine sub-bins NaN) stays NaN for masking. Locked by a test.
  Commits `6a2e246`, `32f15ed`.
- **Lesson:** in JAX, prefer finite sentinels (0) over NaN wherever a value will be
  multiplied/summed; reserve NaN strictly for "masked, never touched arithmetically."

## 4. user-site (`~/.local`) package leakage — **HIT** (env, not JAX-specific but bit us)
- **Where:** Task 0/1 env setup.
- **Symptom:** `pip install` reports packages "already satisfied" and skips installing
  them into the env; the interpreter then silently imports a stale **numpy<2 / h5py**
  from `~/.local` — which can break JAX (numpy 2.x ABI) and is non-reproducible.
- **Cause:** Python user-site is on `sys.path` and pip's "already satisfied" check sees it.
- **Fix:** run **every** python/pip/pytest with `PYTHONNOUSERSITE=1`; the env was made
  self-contained by reinstalling under that flag. Documented in
  `docs/superpowers/2026-06-01-emu-jax-env.md`.
- **Lesson:** isolate the JAX env hard; `PYTHONNOUSERSITE=1` is mandatory here.

## 5. PRNG key reuse across `Linear` layers — **GUARDED**
- **Where:** Task 8 (`hcd_analysis/emulator/model.py`, `Encoder`/`HeadA`).
- **Symptom (if unguarded):** every `eqx.nn.Linear` built from the *same* key gets
  identically-correlated init weights → reduced effective capacity / a silent
  init bug that never raises and is invisible in a forward-shape test.
- **Cause:** JAX RNG is explicit and stateless; passing one key to N constructors
  reuses it. `Encoder` correctly does `ks = jax.random.split(key, len(widths))`
  and `HeadA` does `k1,k2,k3 = jax.random.split(key, 3)`, so each `Linear` gets a
  distinct subkey. (Note: `Encoder` and `HeadA` are *constructed* with the same
  top-level key in the test — harmless, they are separate modules; the risk is
  *within* a module.)
- **Fix:** split once per module, one subkey per `Linear`; pinned by
  `test_encoder_layer_keys_distinct` (reproducible from same key, differs on a new
  key). Verified the `layers: list` field is a proper pytree (6 array leaves for
  optax) and that latent stays float64 even on float32 input (weights are f64 →
  matmul promotes). Commit `5f64b6f`.
- **Lesson:** one `split` per module, one subkey per parameterised submodule;
  add a key-sensitivity test, never just a shape test.

## 6. scalar `tau0` reshaped inside a vmapped fn — **GUARDED**
- **Where:** Task 9 (`hcd_analysis/emulator/model.py`, `HeadB.__call__` / `Emulator`).
- **Symptom (if unguarded):** Head B builds its trunk input as
  `concatenate([latent, <tau0-as-1d>])`. The forward pass with an unbatched scalar
  tau0 looks fine, but under `jax.vmap(lambda x,t: m(x,t))(X(B,10), T(B,))` a naive
  `tau0.reshape(1)` (or `jnp.array([tau0])`) corrupts: vmap strips the leading batch
  axis so each traced tau0 is 0-d, and a hard `reshape(1)` either errors or, worse,
  re-materialises a wrong shape that broadcasts silently.
- **Cause:** under `vmap` the per-row tau0 is a 0-d traced scalar; rank-changing ops
  must be rank-relative, not absolute. `jnp.atleast_1d(tau0)` promotes 0-d→(1,) and
  leaves (1,) untouched, so it composes correctly with the batched concatenate.
- **Fix:** keep `jnp.atleast_1d(tau0)` (NOT `.reshape(1)`); pinned by
  `test_emulator_vmap_over_tau0_and_structural_grad` (vmap over a (B,) tau0 →
  (B,4,n_k)/(B,3,n_k), batched output matches per-row eager). Commit fcde747.
- **Lesson:** inside a function that will be `vmap`-ped, use rank-relative shape ops
  (`atleast_1d`/`[..., None]`), never absolute `reshape(k)`; always add a batched
  (vmap) test, not just an unbatched forward test. (Reshape correctness separately
  verified: Head B's `out.reshape(7,n_k)` is row-major → 4 contiguous P_filt rows then
  3 delta rows, NOT interleaved; structural_tier_p einsum batches + grads finitely on
  both args, all-ones→sum(w) identity holds.)

## 7. `masked_mse` only sanitises NaN, not inf — precondition: mask ⊇ non-finite — **WATCH**
- **Where:** Task 10 (`masked_mse` in `model.py`), surfaced by the adversarial JAX review.
- **Symptom:** `nan_to_num(target, nan=0.0)` replaces NaN with 0 but leaves ±inf untouched.
  An ±inf target at a **masked-out** position is harmless (the `where` zeros it: grad 0,
  verified at 1st and 2nd order). But an ±inf target at an **unmasked** position survives:
  forward `inf`, gradient `-inf` (`pred − inf` diff). `0·weight` does **not** rescue it
  (`0 · inf = NaN` — confirmed: zero-weight on an unmasked-inf gives a NaN grad).
- **Cause:** the function sanitises only the NaN sentinel; inf is not in the contract.
- **Why it's WATCH, not HIT:** the data pipeline (`data.py`) uses **NaN** as the sole
  above-Nyquist sentinel (`np.full(...,nan)`, `summ[allnan]=nan`) and sets
  `mask = isfinite(P_tier_p)`, so by construction `mask == isfinite(target)` and no inf
  ever reaches a finite/unmasked bin. The joint loss also materialises the full mask
  (`m3 & ones_like(target,bool)`) so `sum(mask)` counts the right element total.
- **Fix (optional hardening, not yet applied):** `nan_to_num(target)` with default
  posinf/neginf clamps inf to a huge finite too — strictly safer at zero cost to the
  in-contract path. Left as-is per spec ("NaN-safe"); pinned that masked-inf is safe by
  `test_masked_mse_inf_target_at_masked_position`.
- **Lesson:** `nan_to_num(nan=0.0)` is NaN-only; if a non-finite value can ever sit at an
  *unmasked* element, clamp inf too. Keep the invariant `mask == isfinite(target)` enforced
  at the data boundary so the loss never has to.

## 8. dynamic `int` leaf breaks plain `value_and_grad` — `eqx.field(static=True)` — **HIT**
- **Where:** Task 11 (`HeadB.n_k` in `model.py`), commit `6fa9fee`, caught building the joint loss.
- **Symptom:** plain `jax.value_and_grad(joint_loss)(model, ...)` raises
  `TypeError: grad requires real- or complex-valued inputs (... a sub-dtype of np.inexact),
  but got int64`. A bare `n_k: int` field is a **dynamic pytree leaf**, so reverse-mode AD
  tries to differentiate the integer. (Confirmed by reconstructing the pre-commit dynamic-int
  `HeadB`: plain `value_and_grad` fails; `eqx.filter_value_and_grad`, which filters to
  `eqx.is_array`/`is_inexact`, silently skips the int and works — which is why earlier
  filter-based tests never exposed it.)
- **Cause:** Equinox treats annotated fields as pytree leaves unless marked static; an int
  leaf is a non-inexact leaf that JAX's `grad` refuses.
- **Fix:** `n_k: int = eqx.field(static=True)` — moves `n_k` into the static aux-data, off the
  differentiable leaf set. Verified regression-free: (a) `filter_jit` traces ONCE and does not
  recompile on new values / a re-keyed model with the same `n_k`, but DOES retrace when `n_k`
  changes (correct — static = part of the jit cache key); (b) `jax.vmap(model)` still batches;
  (c) `tree_serialise/deserialise` round-trips — static `n_k` is rebuilt from the **skeleton**
  (bytes hold only array leaves), restored model reproduces preds bit-for-bit, and
  deserialising into a wrong-`n_k` skeleton correctly raises a shape/sha mismatch; (d) plain
  `value_and_grad(joint_loss)` now returns a finite scalar + all-finite grads. Pinned by
  `test_headB_n_k_static_field_serialise_roundtrip`. Commit `<this commit>`.
- **Lesson:** any non-array hyperparameter stored on an `eqx.Module` (ints, shapes, flags,
  tuples) must be `eqx.field(static=True)`, NOT a bare leaf — else it (a) breaks plain
  `value_and_grad` and (b) bloats the differentiable pytree. Static fields are reconstructed
  from the skeleton on deserialise, so the load-time skeleton must carry the right value.
  Don't let `filter_value_and_grad` mask the bug: test plain `value_and_grad` too.

## 9. likelihood grad must flow through ALL inputs + the Head-A↔Head-B w_c coupling — **GUARDED**
- **Where:** Task 16 (`hcd_analysis/emulator/likelihood.py`), commit `<this commit>`, caught by the JAX review.
- **Symptom (if untested):** the only differentiability test pinned `jax.grad` w.r.t. `delta`
  only. Two real risks went unpinned: (a) grad w.r.t. `w_hcd`/`A_hcd` in both forms; (b) the
  spec-§6 coupling where the **same** `w_c` feeds BOTH the structural `P_tier_p = Σ_c w_c·P_filt`
  (Head A) and the HCD add-back `Σ_HCD w_c·A_c·Δ_c`. If a future refactor accidentally detached
  `w_c` from one path (e.g. `stop_gradient`, or recomputing `P_tier_p` from a constant), the
  forward stays correct and the delta-only test stays green, but the inference gradient is wrong.
- **Cause:** nothing JAX-pathological here (einsum is fully differentiable, x64 on, finite) — the
  trap is **test coverage**: a grad test that exercises one argument silently certifies the others.
- **Fix:** added `test_difference_grad_through_w_and_A` (finite + float64 grads of both forms w.r.t.
  w/A/delta; checks `∂/∂A_c = w_c·Σ_k Δ_c` so the A-path is provably live) and
  `test_structural_coupling_grad_through_wc` (grad of `total_p1d_difference(structural_tier_p(w),
  w[1:], A, Δ)` w.r.t. the full 4-vector `w_c` equals `struct + addback`; clean class[0] = struct
  only, HCD classes[1:] strictly differ from struct-only → both paths contribute). Verified jit==eager
  for all three fns, cov symmetric/diagonal/PSD, ratio A=1→no deviation, additivity to 1e-12.
  Note the **intentional** 3-vs-4 asymmetry: the add-back is HCD-only (3 classes) because the clean
  class lives entirely in the structural `P_tier_p`; the covariance correctly uses the 4-class
  `sigma_emu`/`w_c` (clean emu error is real and belongs in the budget). Sound.
- **Lesson:** for a multi-input differentiable contract, write one finite-grad assertion **per
  argument**, and where two outputs share a parameter, pin that the grad picks up **both**
  contributions (compare against the analytic struct+addback), not just that it is finite.

## 10. `delta_c(z)` polynomial: batched-z + grad-through-z were untested — **GUARDED**
- **Where:** Task 6 (`delta_c`/`w_c_corrected` in `dndx_wc.py`), commit `c94efc0`, caught by the JAX review.
- **Symptom (if untested):** the shipped tests only exercised **scalar** z and grad w.r.t.
  `dndx`. Two real, distinct risks went unpinned: (a) **batched z** — `delta_c` builds
  `jnp.stack([z**2,z,1], axis=-1) @ coeffs.T`, so a (N,) z must give (N,4); an `axis`
  slip would only surface batched (the vmap/likelihood path), never scalar. (b) **grad
  through the continuous z polynomial path** — the likelihood differentiates `δ_c(z)`;
  a scalar `dndx`-only grad test certifies neither z nor the all-zero-`dndx` renorm edge
  (`w/sum(w)` is a latent 0/0 if a class ever collapsed).
- **Cause:** not JAX-pathological — pure **test coverage**. Verified concretely: code is
  float64, JAX-pure (no python branch on traced z), jit==eager for both fns scalar+batched;
  sum-to-1 holds to ≤2.2e-16 at z∈{2.0,3.0,5.4} incl. tiny/zero `dndx`; grad w.r.t. each
  of the 4 components, w.r.t. continuous z, the full `dndx→w_c→scalar` chain, and the
  all-zero-`dndx` z-grad are all finite (the 0/0 does not materialise — `w_clean=1` keeps
  the denom = 1+δ_clean ≠ 0). Coeffs cross-checked **byte-exact** against
  `2026-06-01-delta_c-coeffs.md` (max abs diff 0.0); `delta_c` == `np.polyval` to 6.9e-18
  (matmul-vs-Horner rounding, no-op). Physics direction correct: at the coupling point
  (dndx≈[0.36,0.10,0.05], Xbar=0.642, z=3) δ_c=[+0.0066,−0.0104,−0.0154,−0.0166] moves
  w_c clean↑ / HCD↓ — the sign the calibrated mean δ_c implies.
- **Fix:** added `test_wc_corrected_batched_z_shape_and_sum_at_extremes` (batched (N,4),
  sum-to-1 ≤1e-10 at z-extremes + zero `dndx` → [1,0,0,0]) and
  `test_wc_corrected_grad_through_continuous_z_is_finite` (finite grad through z, incl.
  all-zero-`dndx` renorm edge). Commit `<this commit>`.
- **Lesson:** a polynomial-in-z map differentiated by the likelihood needs a batched-shape
  test AND a grad-through-z test, not just scalar-value + grad-w.r.t.-the-other-arg.
  Cross-check frozen coeffs byte-exact against the calibration record, not "looks small".

## 11. single-alpha likelihood: `...c,...ck->...k` batching + ratio alpha=0 identity untested — **GUARDED**
- **Where:** `hcd_analysis/emulator/likelihood.py` (`total_p1d_difference`, `total_p1d_ratio`),
  refactor `58b0dc7` (drop redundant `w_c*A_c` -> single `alpha_c`).
- **Symptom:** none observed; the refactor swapped the explicit `c,c,ck->k` einsum for an
  ellipsis `...c,...ck->...k` (to support batched sampler calls) but the test file only
  exercised unbatched (3,)/(3,K) inputs, and neither the ratio-form `alpha=0 -> P_obs=P_tier_p`
  identity nor jit-vs-eager were guarded.
- **Cause:** ellipsis einsum silently accepts/mis-broadcasts wrong leading-dim layouts;
  an `alpha=0` "identity" that is only `allclose` rather than exact can mask a stray
  `+eps` or reordering. Both are exactly the kind of contract the refactor newly relies on.
- **Fix:** added `test_batched_alpha_matches_vmap_and_loop` (B,3)/(B,3,K)->(B,K) == vmap,
  `test_ratio_form_alpha_zero_identity` (`jnp.array_equal`, both forms), and
  `test_jit_matches_eager_both_forms` (unbatched + batched). All green.  (commit 20ab26a)
- **Lesson:** when you switch a fixed-index einsum to an ellipsis form for batching, add a
  vmap-equivalence test at the new rank immediately — the ellipsis won't error on a bad
  layout, it'll just return the wrong shape/numbers. Assert reduction identities (alpha=0)
  with `array_equal`, not `allclose`.

---

## Trap template (append new entries above this line)
```
## N. <short name> — HIT | GUARDED | WATCH
- **Where:** <task / file>
- **Symptom:** <what you observe>
- **Cause:** <the JAX mechanism>
- **Fix:** <the code change / config>  (commit <sha>)
- **Lesson:** <the general rule>
```

## Standing JAX pitfalls to keep checking (from the JAX intro)
- Immutability: no in-place writes — use `x.at[idx].set(...)`.
- Explicit PRNG keys — never reuse a key; `jax.random.split`.
- No Python control flow on traced values — `jnp.where`/`lax.cond`/`lax.scan`.
- `jit` recompiles on new input shapes — keep shapes fixed; static args must be hashable.
- Async dispatch — `.block_until_ready()` to time; `jax.debug.print` inside `jit`.
- Debug with `jax.disable_jit()` / `JAX_DEBUG_NANS=1` to localise NaNs.
- Equinox: split params vs static with `eqx.filter(..., eqx.is_array)`;
  use `eqx.filter_jit` / `eqx.filter_value_and_grad`.
