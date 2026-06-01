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
  (B,4,n_k)/(B,3,n_k), batched output matches per-row eager). Commit <SHA>.
- **Lesson:** inside a function that will be `vmap`-ped, use rank-relative shape ops
  (`atleast_1d`/`[..., None]`), never absolute `reshape(k)`; always add a batched
  (vmap) test, not just an unbatched forward test. (Reshape correctness separately
  verified: Head B's `out.reshape(7,n_k)` is row-major → 4 contiguous P_filt rows then
  3 delta rows, NOT interleaved; structural_tier_p einsum batches + grads finitely on
  both args, all-ones→sum(w) identity holds.)

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
