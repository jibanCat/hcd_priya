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
