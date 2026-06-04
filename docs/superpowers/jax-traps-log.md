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

## 12. low-rank P_filt basis: serialise round-trip + structural-composition were untested — **GUARDED**
- **Where:** Task 2 (Phase-2b) low-rank P_filt output bottleneck (`HeadB`/`Emulator` `n_basis`,
  trainable `p_filt_basis`, `svd_basis_init` in `model.py`), commit `6d6f745`, caught by the JAX review.
- **Symptom (if untested):** the shipped tests pinned dense-unchanged, low-rank shapes/param-reduction,
  SVD warm-start, finite grads/jit/vmap, and Δ_c-independence — but two real risks went unpinned:
  (a) **serialisation** — the new `p_filt_basis` is an array leaf and `n_basis` is a *static* field;
  `tree_serialise/deserialise` must round-trip the basis bit-for-bit while rebuilding `n_basis` from
  the SKELETON (the existing round-trip test covered only the DENSE model), and a wrong-`n_basis`
  (or dense) skeleton must RAISE rather than mis-load; (b) **structural composition** — the whole point
  of the bottleneck is that linear-space P_filt from the low-rank head still feeds `structural_tier_p`'s
  `(…,4,n_k)` einsum and the all-ones-`w_c`→sum-over-classes identity.
- **Cause:** not JAX-pathological — pure **test coverage** of the contract the new code relies on.
  Verified concretely (this review): x64 on; all 17 differentiable leaves float (basis included);
  round-trip bit-exact; deserialise into `n_basis-1` and `n_basis=None` skeletons both raise `RuntimeError`;
  `structural_tier_p(ones, exp(P_filt_lowrank))` == sum-over-4-classes, grad through the basis finite+nonzero.
  Also confirmed the dense path is byte-for-byte the literal pre-change code (reconstructed `OldHeadB`:
  trunk/out weights+bias and outputs `array_equal`) — the `split(key,2)→split(key,3)` change is a no-op
  here because this jaxlib's `split` yields the same first-k subkeys for any total ≥ k (the unused `k3`
  is harmless on the dense branch); `n_basis=int` resolves the `if self.n_basis is None` branch at trace
  time (filter_jit traces once across re-keyed same-`n_basis` models, retraces on `n_basis` change).
  SVD `n_basis > rank` still returns orthonormal full-rank rows (rank-collapse guard holds).
- **Fix:** added `test_lowrank_serialise_roundtrip_and_structural_composition` (round-trip bit-exact,
  wrong-skeleton raises, structural einsum + sum identity + finite grad-through-basis). All 21 green. (commit `<this commit>`)
- **Lesson:** when a feature adds a NEW array leaf gated by a NEW static field, the dense serialise test
  does NOT cover it — add a round-trip test at the new config AND assert a wrong-static skeleton raises.
  For an output-reparametrisation bottleneck, pin that the downstream consumer (`structural_tier_p`) still
  composes on the reparametrised output, not just that the head's own forward/grad is finite.

---

## 13. `alpha_to_dndx` M0-inverse: infeasible-w edge + clamp-grad were untested — **GUARDED**
- **Where:** Task 3 (Phase-2b) `alpha_to_dndx` analytic telescoping inverse + `delta_c(z)` `jnp.clip`
  z-range clamp in `hcd_analysis/emulator/dndx_wc.py`, commit `518d76d`, caught by the JAX review.
- **Symptom (if untested):** the shipped tests pinned the exact inverse (≈1e-13), the renorm-caveat
  roundtrip, clamp boundary VALUES, and jit/vmap/float64 + grad-wrt-alpha — but three real risks were
  unpinned: (a) **infeasible-w edge** — `alpha_c` is a FREE flat-log HMC param, so proposals can imply
  `w_DLA>1` or `w_sub>1-w_DLA` (negative telescoping arg); the `_LOG_FLOOR=1e-12` clip must keep BOTH
  value and grad finite, not NaN; (b) **clamp-grad** of `delta_c` — 0 on the extrapolation plateau,
  finite/nonzero inside; (c) **grad-through-(A,γ)** of the PW14 forward law that feeds the likelihood.
- **Cause:** not JAX-pathological — test coverage of the contract HMC relies on. Verified (this review):
  the top-down inversion (`mu_DLA=-ln(1-w_DLA)`, `mu_sub=-ln(1-w_sub/(1-w_DLA))`,
  `mu_LLS=-ln(1-w_LLS/((1-w_DLA)-w_sub))`) is the exact inverse of `w_c_from_mu`; index map
  (w_LLS,w_sub,w_DLA)=alpha[...,0,1,2] correct; the apply_delta=True roundtrip's ~2.3% LLS error is a
  TEST-ONLY artifact (forward `w_c_corrected` renormalises over 4 classes; production `alpha_c` is a free
  amplitude read out by the telescoping inverse, not un-renormalised) — sound framing. NOTE: at the
  infeasible edge the clip leaks a stiff but finite grad (`~1e12` from `1/_LOG_FLOOR` in the LLS/sub
  denominators) — finite, so HMC won't NaN, but the floor magnitude is what bounds that gradient.
- **Fix:** added `test_alpha_to_dndx_edge_domain_and_clamp_grads` (finite value+grad at w_DLA>1,
  w_sub>1-w_DLA, w_DLA==1; clamp grad 0-outside/finite-inside) and
  `test_alpha_from_dndx_law_grad_through_A_gamma` (finite grad wrt A and γ + end-to-end chain). 13 green.
  (commit `<this commit>`)
- **Lesson:** when a clipped log-inverse sits behind a FREE flat-(log)-prior HMC param, test the
  infeasible-domain proposals explicitly — "exact inverse on valid inputs" does NOT cover the edge the
  sampler will actually visit. A `jnp.clip` z-clamp gives a 0 grad on the plateau (correct, by design),
  but assert it so a future refactor to a python branch / unclamped poly is caught.

---

## 14. baseline+residual P_filt redesign: θ-blindness is structural, sig_cosmo=0 div, exp overflow — **GUARDED**
- **Where:** Phase-2b normalization REDESIGN — `BaselineHead`/`Emulator`/`reconstruct_P_filt`/
  `fit_baseline_residual_norm` in `hcd_analysis/emulator/{model,data}.py`, HEAD `6098dcf`, JAX-referee pass.
- **Symptom (if untested):** the redesign splits P_filt into a θ-BLIND baseline `m̂(z,τ₀)` + a residual
  `r̂(θ,z,τ₀)`, with `logP=(m̂·σ_marg+μ_marg)+σ_cosmo·r̂`. Three real risks: (a) **θ-blindness is only as
  good as the wiring** — if `BaselineHead` ever saw the latent (which encodes θ), or the Emulator passed a
  param instead of `z=x[9]`, `∂m̂/∂θ≠0` and the baseline stops being structurally identifiable; the forward
  shapes stay correct and no test fails. (b) **σ_cosmo=0 division** — `σ_cosmo` = sqrt(mean-over-cells of
  within-cell variance); a cell with a SINGLE train sim has within-var 0, and an all-NaN (c,k) has 0
  contributing cells → `σ_cosmo=0` → `t_p_resid=(logP−m)/0 = inf/NaN`. (c) **exp overflow** — the clean
  class spans logP up to ~9.2 (linear ~1e4); an untrained/large standardized output could overflow `exp`.
- **Cause:** not JAX-pathological — pure **contract coverage** of the new normalization. Verified by RUNNING
  (this review): (a) `jax.jacfwd(P_filt_base)` w.r.t. the 9 params is **EXACTLY 0.0** (`all(J==0)`, finite)
  for BOTH dense and low-rank, and the full `∂(reconstructed logP)/∂θ == σ_cosmo·∂r̂/∂θ` to **max abs diff
  0.0** — the θ-response flows ONLY through the residual; baseline correctly DOES depend on z/τ₀ (nonzero
  d/dz, d/dτ₀). `z=x[9]` wiring confirmed (changing a param leaves base EXACTLY unchanged; changing z
  perturbs it). (b) the `np.where(sig_cosmo>=1e-12, sig_cosmo, 1.0)` floor + `within_var_cnt>0` guard makes
  a single-sim-per-cell fit (synthetic n_sims=1) produce `σ_cosmo≡1.0` with NO RuntimeWarning escaping and
  `t_p_resid` finite (max|.|=0, since logP==cell_mean for a lone sim); `nanvar` of a single finite value is
  0 (not NaN) so it pools as 0, and an all-NaN (c,k) gives NaN var → `ok=False` → excluded. The σ_cosmo/
  σ_marg (4,K) broadcast over (n,4,K) is correct (trailing-dim). (c) exp is safe by a wide margin — real
  recon max ~1.7e3; even a 10σ stress (logP≈16.8) → exp 2e7, far below f64 overflow (~709 in the exponent /
  1.8e308). Reconstruction round-trips to **~1e-15** on the real LF+HR cache (train AND val — LOSO holds out
  sims, not cells, so no mu_marg fallback fires), `reconstruct_P_filt` grad finite (matches σ_marg·P /
  σ_cosmo·P exactly), and `structural_tier_p` consumes the LINEAR recon (not log), reproducing cache
  `P_tier_p` to 5e-16. `n_k`/`n_basis` are STATIC (jit traces once, retraces only on `n_basis` change);
  serialise round-trips bit-exact and a wrong-`n_basis`/dense skeleton RAISES. NaN-grad double-where trap:
  the masked-MSE invariant `mask ⊇ NaN(t_p_resid)` HOLDS (all NaN target bins masked out), so `joint_loss`/
  `p_resid_loss` give all-finite grads even on a batch with above-Nyquist NaN (synthetic n_k=16). NOTE: both
  PRODUCTION caches (`observables_tau0_lf.h5` n_k=172, `_hr.h5` n_k=525) currently carry NO above-Nyquist
  NaN bins — the NaN path is exercised only by the synthetic fixture; if a future cache reintroduces Nyquist
  NaN, the guard is already in place.
- **Fix:** no code change needed — the guards (θ-blind wiring, σ_cosmo floor, masked-MSE double-where,
  static fields) are all present and correct. Existing `tests/test_emulator_{model,data}.py` (52) pass.
- **Lesson:** when a "blind" sub-model's identifiability is STRUCTURAL (it must not see θ), pin it with a
  `jacfwd==0 EXACTLY` test on the real param axis AND assert the full response equals the intended path
  (`σ_cosmo·∂r̂/∂θ`) — a forward-shape test certifies neither. A conditional-variance whitening (`σ_cosmo`)
  needs a single-sim-per-cell test (within-var 0 → div-by-0) with `simplefilter("error")` to catch a leaked
  RuntimeWarning, not just "looks finite on the production cache."

---

## N. Fourier-feature trunk over-fits the grid → interp/smoothness blow-up — **HIT** (prototype)
- **Where:** `scripts/proto_continuous_kdecoder.py` (continuous-in-k DeepONet decoder feasibility).
- **Symptom:** a `φ(k)=MLP(γ(logk))` trunk with γ = bandlimited Fourier features of log-k reconstructs the
  TRAIN k-bins beautifully (Test A: n_freq=64 → 0.27% fracRMS, ≈ SVD-12 ceiling 0.18%) but, fit to only the
  KEPT bins, HALLUCINATES between them: Test-B held-out p95 explodes to 30–95% and max to hundreds of percent,
  and the dense reconstruction oscillates (Test-C TV(dense)/TV(grid) ≈ 8–30, not ≈1). More Fourier frequencies
  IMPROVE on-grid recon but WORSEN interpolation — the classic aliasing trade-off.
- **Cause:** a free MLP on Fourier features has NO smoothness prior controlling its value at unobserved k.
  Per-row least-squares coeffs fit to the kept bins excite the basis's between-bin excursions. Nothing in the
  data loss penalises wiggle off the training grid. (Distinct from a numerical NaN trap — it is a
  modelling/identifiability trap that a forward-shape or on-grid-recon test would NOT catch.)
- **Fix:** add an explicit **curvature smoothness prior** to the fit loss — penalise the mean-square second
  derivative `∂²φ_i/∂(logk)²` of every basis function on a dense log-k grid (`jnp.diff(Pc, n=2, axis=1)/dlk²`),
  weight λ≈3e-3; plus a per-row coeff ridge ≈1e-3 to damp the few rows whose tail blows up. This is the learned
  analogue of the cubic spline's second-derivative-minimising property. With it, Test-B p95 drops to ~1.5–2.3%
  (≈ spline) and TV ratio → ~1. ALSO bandlimit the Fourier bank to the GLOBAL log-k Nyquist `f_max=(K/2)/L`
  (≈16.7 cycles/unit-logk for the 172-bin LF grid) — NOT the high-k local Nyquist (~85), which the sparse
  low-k sampling cannot constrain.
- **Residual caveat (HONEST):** even regularised, the held-out MAX stays large (~40–80%) but ONLY at the 1–2
  lowest-k bins, where the LINEAR-k grid gives a factor-~2 jump in log-k (the single widest gap) — the cubic
  spline's max is also elevated there (~13%). This is a test artefact of holding out the sparsest log-k corner,
  not a generic interpolation failure; production never holds out the lowest-k bin.
- **Lesson:** a continuous/implicit decoder (DeepONet trunk, SIREN, neural field) needs a SMOOTHNESS prior
  (curvature penalty) AND a sampling-honest bandlimit, or it over-fits the training grid and is useless OFF it.
  Test interpolation to HELD-OUT coordinates and dense-grid total-variation, never just on-grid reconstruction.

---

## N+1. Cubic-spline k-mapping = a HOST-precomputed constant matrix W, not an in-graph solve — **GUARDED**
- **Where:** `scripts/feasibility_diff_spline.py` (differentiable spline k-mapping feasibility for HMC).
- **Symptom (avoided):** the obvious implementation evaluates a cubic spline inside the forward pass — a
  tridiagonal `linalg.solve` for the node second-derivatives + a `searchsorted` bracket lookup + per-interval
  Hermite blend, all on TRACED values. Under `jit`/`grad`/`vmap` that drags a (re)solve and data-dependent
  gather into every leapfrog step, and `searchsorted` on traced data is a control-flow trap.
- **Cause/insight:** a cubic spline interpolant is **LINEAR in the node values** — the `A M = B y` solve and the
  Hermite blend are both linear maps of `y`, so the entire `sim_k -> data_k` evaluation collapses to a single
  dense matrix `W (n_data, n_sim)` that depends ONLY on the two (fixed) k-grids. Assemble `W` ONCE on the host
  (numpy `linalg.solve` + `searchsorted`), freeze it as a `jnp` constant, and the forward pass is just
  `logP_data = W @ logP_sim`. Verified: `jax.jacobian(W@y wrt y)` equals `W` to 0.0 and is point-independent
  (genuinely linear); `grad` through `exp(W@(coeffs@basis))*C_res` matches finite-diff to <1e-6 and the
  closed-form `2 W^T(W y)` to 1e-14; jit/vmap/vmap(grad) clean. No `∂P/∂k` term is ever needed — the resolution
  window `C_res=1+2 f_res R² k²` is pointwise at the fixed data k.
- **Equivalent cleaner form:** pre-apply `W` to the SVD basis (`basis_data = basis @ W.T`, built once per
  (fidelity, data-grid)); then `logP_data = coeffs @ basis_data`. Bit-identical to `W@(coeffs@basis)` (max diff
  7e-15) and cheaper (one matmul over n_basis≈12 cols instead of K=172/525). The basis vectors are NOT
  individually smoother than a logP row (higher SVD modes oscillate) — irrelevant, since `W` is the same operator.
- **Accuracy/measurement trap (HONEST):** do NOT report "full-grid spline onto data grid vs scipy" as an
  interpolation ERROR — it is `W==scipy` (Test 1) re-stated, trivially ~0 (spline-reproduces-spline; the data
  grid is COARSER than the dense sim grid within range). The honest within-range numbers are the INDEPENDENT
  round-trip (sim->data->sim vs known cache values: med 0.05–0.35%, p95 0.5–1.5%) and a conservative
  half-density DECIM bound (med ~0.4%, p95 ~1.3–2%). Also note the cache's per-row k-grids are EXACT scalar
  multiples of a single uniform-linear template (per-row std of the ratio ~5e-16 — a z-rescale), so the decoder
  emits on ONE canonical sim grid; `W` is built against that.
- **BC + range:** use **not-a-knot** (scipy default; no spurious edge curvature for a smooth power law;
  max|W|≈1.05, Lebesgue row-sum≈1.6 — no edge amplification, no NaN/Inf). CUT the likelihood to the validated
  sim range (LF k≤0.069, HR k≤~0.20 s/km); KODIAQ high-k beyond the LF Nyquist 0.069 -> HR fidelity or cut.
- **Lesson:** for ANY fixed-grid interpolation in an HMC forward model, check if the interpolant is linear in the
  values (cubic/linear splines are) → precompute it as a constant matrix on the host. Never run the solve/gather
  in-graph, and never benchmark a self-reproducing interp against itself.

---

## 20. θ-blind baseline head under-resolves a smooth 2D map with 1 hidden layer — WATCH
- **Where:** `scripts/feasibility_subpercent.py` (Exp 2), `hcd_analysis/emulator/model.py::BaselineHead`.
- **Symptom:** the deployed baseline mis-fit (term b) sat at ~0.84·σ_cosmo. Two causes, BOTH needed: (i) the
  `inv_nc` loss weighting in `joint_loss` shrinks the masked-MSE denominator's effective weight on the baseline
  term to ~6e-3 (the design's known #13); (ii) even with the loss fixed to a uniform `Σ(weight·mask)`
  normalization and trained hard, the production `BaselineHead` (a SINGLE hidden layer: `Linear(2,w)->gelu->
  Linear(w,4·n_basis)`) plateaus at ~0.10·σ_cosmo and will not go lower regardless of width or epochs.
- **Cause/insight:** the baseline target is the (z,τ₀)→(4,K) cell-mean — a smooth 2-D→high-dim regression on
  306 points whose SVD rank is ~8 (recon <0.05% in P at rank-8). The REPRESENTATION is trivially low-rank, so the
  residual ~0.10·σ_cosmo is an OPTIMIZATION/approximation floor of the shallow `(z,τ₀)`-MLP, NOT a rank/loss
  floor. Going to 3 hidden layers (width 256) collapses (b) to 0.03–0.05·σ_cosmo — below the finite-sim floor
  (~0.11·σ_cosmo). Nothing JAX-numerical here; it is a capacity/identifiability observation a one-layer head
  hides.
- **Fix:** (a) normalize the baseline loss by `Σ(weight·mask)` not the `inv_nc`-shrunk sum; (b) give the baseline
  head ≥3 hidden layers (or it caps term (b) at ~0.10·σ_cosmo). Verified on fold-0: clean (b) 0.84 -> 0.028,
  DLA 0.84 -> 0.047 σ_cosmo with d3/w256.
- **Lesson:** when a "baseline/structured-mean" head is the deployed reference (its error is amplified by the
  small σ_cosmo), separately verify its REPRESENTATION floor (SVD of the target) AND its head's ACHIEVABLE fit
  (capacity sweep) — a per-element loss that looks converged (4e-8 here) can still leave the head 20× above its
  representation floor because a tiny loss in σ_marg units is large in σ_cosmo units.

## 21. residual-head (a) is finite-sim generalization-limited, not capacity-limited — WATCH
- **Where:** `scripts/feasibility_subpercent.py` (Exp 3): encoder + `HeadB` residual path, perfect (empirical)
  baseline, trained hard on fold-0.
- **Symptom:** residual-head fit (a)=RMS(r̂−t_resid)/σ_cosmo reaches TRAIN 0.07–0.10 but VAL (held-out sims)
  0.16 (clean/LLS), 0.21 (subDLA), 0.57 (DLA). Sweeping rank (8/12/24), weight decay (1e-3..1e-2) and epochs
  barely moves the val number (clean stays ~0.155–0.16).
- **Cause/insight:** the residual head maps 9-D cosmology θ → within-cell signal, learned from only ~52 train
  cosmologies; the val (a) is the LOSO held-out-sim generalization gap = the irreducible finite-60-sim floor, NOT
  an under-regularization or capacity artifact (it is robust to wd/rank). This is the binding constraint on
  sub-percent absolute accuracy (deployed fracP ≈ √((a)²+(b)²)·σ_cosmo[logP], σ_cosmo[logP]≈0.077–0.099/class).
- **Lesson:** for an emulator trained on few simulations, separate the TRAIN fit floor (representation/optimization)
  from the VAL floor (finite-sample generalization) by reporting BOTH on a LOSO held-out-sim split. Do not chase
  the train floor with capacity — the deployed error is the val floor, set by the number of sims.

---

## 22. Fisher-bias δθ/σ explodes when J is rank-deficient (6-sim HR LOSO) — **HIT** (diagnostic)
- **Where:** `scripts/diag_tilt_bias_lf_hr.py` `fisher_bias()`: J = `jax.jacfwd(r̂)(θ_unit)·σ_cosmo`
  (∂logP̂/∂θ over the 9 unit-cube params), bias δθ = (JᵀC⁻¹J)⁻¹JᵀC⁻¹δ reported in σ_i = √diag(JᵀC⁻¹J)⁻¹.
- **Symptom:** HR (6 sims, 6-fold LOSO → **1 val cosmology**, residual head trained on **5**) gives an apparent
  ns bias of **+6.9σ** and a Fisher posterior σ_ns = **4.8 in unit-cube terms** (the whole prior box is width 1!),
  Fisher cond ≈ 4e7, and absolute `dtheta_unit[ns]` = **+33** (33× the prior width — physically impossible). LF
  (60 sims) is clean and physical: σ_ns=0.21, bias +0.057σ, dtheta_unit +0.012.
- **Cause:** a residual head trained on ~5 cosmologies learns an almost flat θ→signal map, so the autodiff
  Jacobian J is tiny and near rank-deficient over 9 params. F=JᵀC⁻¹J is then near-singular; both σ_i=√diag(F⁻¹)
  AND the pseudo-inverse F⁻¹JᵀC⁻¹δ blow up. The ratio δθ/σ is two large ill-conditioned numbers divided — it is
  NOT a meaningful "6.9σ shift", and the ridge (1e-12·tr) only stops a hard NaN, it does not restore rank.
- **Fix:** sanity-gate the Fisher on the ABSOLUTE bias vs the prior width (`dtheta_unit` ≫ 1 ⇒ J degenerate ⇒
  the σ-units are meaningless), and report the Fisher condition number. Trust per-σ bias only when σ_i ≪ prior
  width AND cond is moderate (LF: σ≈0.2–5, cond 1e6 borderline but dtheta_unit≪1 ⇒ OK). For a genuine HR
  systematic test you need MORE HR cosmologies (or a learned multi-fidelity prior on J), not this 6-sim suite.
- **Confirmation (train-vs-val tilt, k≥1e-3 clean):** LF TRAIN slope +0.00%/dex (RMS 0.02%), VAL +0.14%/dex
  (RMS 0.38%) on 8 held-out sims — fits AND generalizes, recipe removed the LF tilt. HR TRAIN +0.12%/dex
  (RMS **0.14%** — the HR REPRESENTATION is clean across the KODIAQ band, no intrinsic high-k systematic) but
  VAL **−4.51%/dex (RMS 4.21%)** on the **1** held-out HR cosmology. The HR "tilt" is therefore ENTIRELY the
  5→1-sim LOSO generalization gap, NOT a high-k representation defect. So the HR/MF high-k channel has no
  built-in systematic; only the per-σ bias is unquotable from this 6-sim suite (degenerate J).
- **Lesson:** when you autodiff an emulator for a Jacobian-based bias/Fisher, the result is only physical if the
  emulator's PARAMETER RESPONSE is itself well-resolved. With few training sims the response (hence J) is
  rank-deficient and any F⁻¹ quantity (σ, bias-in-σ, parameter covariance) is an artifact. Always cross-check
  the absolute (unit-cube) shift against the prior width and the Fisher condition number before quoting σ-units;
  and separate the TRAIN tilt (representation/high-k cleanliness) from the VAL tilt (finite-sim generalization).

## weighted-mask loss norm + frozen-baseline two-phase train — GUARDED
- **Where:** `hcd_analysis/emulator/model.py::masked_mse`, `hcd_analysis/emulator/train.py::train_fold`
  (productionize the validated sub-percent recipe).
- **Symptom (loss):** the θ-blind BaselineHead would not train in the joint loop — term (b) (baseline misfit)
  stuck at ~0.84·σ_cosmo. The baseline-term gradient norm was ~1.5e-7 on a real LF batch.
- **Cause (loss):** `masked_mse` weighted the squared diff by `inv_nc` (≈1/n_c, ~6e-3 for a populous class)
  but divided by the *unweighted* masked count `Σ(mask)`. That is NOT a weighted mean — it globally SHRINKS the
  term's gradient by the weight magnitude. (`weight·Σ(diff²)/Σ(mask)` ≠ a mean; the absolute scale rides on the
  weight.)
- **Fix (loss):** make it a TRUE weighted mean — `Σ(weight·diff²)/Σ(weight·mask)`. Preserves inv_nc's RELATIVE
  intent (each element's say = its weight ÷ total weight) but restores the absolute gradient scale. Baseline-head
  grad norm jumped ~7e4× (1.5e-7 → 1.1e-2); term (b) → 0.031·σ_cosmo after the deep head trains. NaN-safety
  unchanged (target sanitised + masked diff zeroed before the weight multiply; `max(Σ,1)` denom guard; zero-weight
  → zero contribution to BOTH numerator and `Σ(weight·mask)` denom, so padded/empty rows still contribute exactly 0).
- **Symptom (train):** even after pre-fitting the deep θ-blind baseline to its 0.03·σ_cosmo floor, term (b)
  RE-INFLATED to ~0.21 (regression test fixture: fit_ratio 0.24 → 0.49) once the JOINT loop ran.
- **Cause (train):** the joint optimizer keeps stepping the baseline leaves; the inv_nc-weighted `p_base` term at
  the shared joint LR DRIFTS the baseline off its pre-fit minimum (minibatch noise + competition with the much
  larger residual/CDDF terms). The validated feasibility recipe never jointly fine-tuned the baseline — it trained
  baseline and residual as SEPARATE fixed fits.
- **Fix (train):** two-phase, NOT the buggy 3-stage `staged` schedule — pre-fit the baseline alone to its floor
  (`_prefit_baseline`, baseline-only AdamW on the (z,τ₀)-cell table via `eqx.partition`), then FREEZE it during the
  joint loop (`freeze_baseline=True` default whenever a pre-fit ran; `eqx.partition` + `train_step_partitioned`
  so the joint optimizer never sees its leaves). The θ-blind structured mean stays pinned; the joint loop trains
  only encoder/Head A/Head B (the θ-dependent residual). Term (b) held at 0.031·σ_cosmo on production LF fold-0.
- **Lesson:** (1) any masked/weighted reduction used as a LOSS must normalize by the SAME weight it applies in the
  numerator (`Σ(w·mask)`), else per-sample weights silently rescale the gradient and a term can't train. (2) When a
  head is pre-trained to a floor that a different objective doesn't reward, FREEZE it (eqx.partition) during the
  shared loop, or it drifts back — pre-fit alone is not enough.

---

## 23. residual-head A_p Fisher-bias: overfit + seed-UNSTABLE unless the low/high-k EDGES are weighted — HIT
- **Where:** `hcd_analysis/emulator/{model.py,data.py,train.py}` (joint_loss `k_weight`, `edge_emphasis_k_weight`,
  train_fold `term_w`/`k_weight`/`early_stop_metric`), diagnosed by `scripts/diag_ap_fisher_bias.py`.
- **Symptom:** the deployed LF fold-0 emulator's A_p (primordial amplitude) Fisher-bias was **+0.26σ** (over the
  0.2σ gate) while n_s was fine (+0.009σ). The pred/true fractional residual was flat mid-band but biased at the
  band EDGES — coherent ⟨P̂/P−1⟩ ≈ −1.6% at low-k (k<0.005) and +1.5% at high-k. Because the Lyα amplitude Δ²_*
  pivots at k_*≈0.009 s/km, a coherent low-k residual maps onto A_p.
- **Two distinct causes:** (a) the residual head OVERFITS past its val minimum (~epoch 16–37 on this 60-sim fold)
  — the joint early-stop on the TOTAL val loss let it run to epoch 140 (the walkthrough), inflating the low-k
  coherent residual. (b) The joint loss trained `p_resid` with UNIFORM `term_w` (diluted 1:5 vs f_nhi/dndx/p_base/
  delta) and NO per-k weight, so the optimizer ignored the sparse, noisy band edges where A_p lives.
- **Key finding (the trap):** early-stopping on the val RESIDUAL alone FIXES the overfit but is **seed-UNSTABLE for
  A_p** — at uniform `term_w` the A_p bias swung 0.03σ (seed0) → 0.97σ (seed1) → 1.18σ (seed2). Different inits
  land the residual head in different low-k coherent-residual states; with no edge pressure the optimizer never
  removes the coherent low-k component, and A_p (a low-k-leveraged amplitude) reads it off. The bias is NOT a
  capacity floor — it is an unconstrained low-k coherent degree of freedom.
- **Fix (robust):** up-weight the cosmology term (`term_w["p_resid"]=8`) + a U-shaped per-k EDGE-emphasis weight on
  the p_resid term (`edge_emphasis_k_weight(edge_gain=3, lowk_extra=2)`, mean-1 normalized so the term scale is
  unchanged — only the relative low/high-k attention shifts) + mild weight-decay (3e-4) + early-stop on the
  (k-weighted) residual. A_p bias → **+0.030σ (seed0), +0.025σ (seed1), −0.146σ (seed2)** — ALL under the gate;
  n_s stayed <0.2σ; deployed median |P̂/P−1| dropped to 0.84% clean (sub-1% clean/LLS/subDLA); honest (a) 0.249→
  0.204·σ_cosmo. The frozen θ-blind baseline (b)=0.031·σ_cosmo was untouched.
- **Test trap (minor):** a unit test that up-weights a k-bin to prove the k_weight bites MUST pick a FINITE
  (below-Nyquist, unmasked) bin — masked_mse zeros NaN/above-Nyquist bins before the weight multiply, so weighting
  a masked bin is a numerical no-op (the loss is bit-identical and the `!=` assertion fails). Place the probe error
  on a finite bin.
- **Lesson:** when an emulator error biases a parameter that leverages a specific k-band (A_p↔low-k), an unweighted
  MSE leaves that band an unconstrained coherent d.o.f. — the bias is then seed-dependent and an early-stop only
  caps the magnitude, not the direction. Put PRINCIPLED per-k pressure (edge/inverse-CV) on the band the parameter
  reads, decompose the residual there into COHERENT (fixable) vs CV-SCATTER (budget in C_emu), and gate on the
  ABSOLUTE unit-cube shift + Fisher σ vs prior width (trap #22), not the σ-ratio alone, for the degenerate dirs.

## 24. low-k coherent residual is a TRAIN→VAL GENERALIZATION GAP, not a trainable offset — **HIT (diagnosis)** + **GUARDED (de-bias term helps)**
- **Where:** residual-head refine (`scripts/push_residual_refine.py`), pushing trap #23 further
  (the PI rejected the coherent low-k tilt the edge-emphasis only HALVED).
- **Symptom:** the coherent residual (mean over cosmologies per (z,τ₀,k)) at clean/subDLA k<0.005
  stays ~1.0–1.2% (vs the 0.73% CV floor) regardless of how hard you push the per-sim MSE / edge
  weight. An explicit COHERENT de-bias loss term (penalize `Σ_cell ⟨r̂−t_resid⟩_θ²`, the per-cell
  cosmology-mean of the residual fit error — the thing MSE leaves free because MSE is dominated by
  the per-sim CV scatter) at low/moderate weight does NOTHING to the VAL coherent.
- **Cause (the decomposition that settled it):** measure the per-cell coherent on TRAIN vs VAL
  separately. Baseline: clean k<0.005 coherent = **0.45% TRAIN vs 1.22% VAL**; subDLA 0.23% vs 1.05%.
  The model FITS the train cosmologies' low-k coherent structure fine — the val coherent is a pure
  **generalization gap**. At k<0.005 there are only ~10 modes with ~1% per-mode CV, so 52 train sims
  carry almost no information about a held-out cosmology's low-k coherent amplitude. The de-bias term
  at w=8 drove TRAIN even lower (0.31%) while VAL was unchanged/slightly worse (1.40%) — confirming
  it's not a trainable offset. The per-fold spread is itself the signature: fold-0 val coherent 0.95%,
  fold-1 0.31% (some held-out sims sit closer to the train manifold).
- **Fix (what genuinely helps):** a STRONG coherent de-bias term DOES pull the val coherent toward the
  CV floor — but ONLY with a **FLAT (uniform-k)** weight at high strength (`w_coh≈80`), acting as a
  REGULARIZER that stops the residual head chasing per-train-sim low-k fluctuations that don't
  generalize. An EDGE-weighted coherent term (reusing trap #23's U-shape) over-corrects low-k and
  TRADES it for a mid-band regression (clean mid 0.70→1.24% per-cell at w=40); INVERSE-CV weighting
  makes low-k WORSE (it down-weights the high-CV low-k). More residual rank (n_basis 24→32→48) does
  NOT shrink the gap (it is not a representation limit) and at 48 over-fits (worse + A_p −0.35σ).
- **8-fold validation (the honest net):** flat-w80 vs the production baseline (73009c5, w_coh=0):
  **gate failures (|A_p| or |n_s| > 0.2σ) 1/8 vs 3/8**; n_s RMS 0.133 vs 0.182σ; A_p RMS 0.150 vs
  0.147σ; clean low-k per-cell coherent ≤ CV floor on **6/8 vs 4/8** folds (mean 0.71 vs 0.76%);
  deployed clean median 0.55 vs 0.64%. A NET improvement — BUT no recipe is uniformly safe: a ~0.3σ
  PER-FOLD A_p/n_s swing persists in BOTH (baseline fails folds 1/4/5; flat-w80 fails fold-3). The de-
  bias REDISTRIBUTES which fold fails rather than removing the scatter, because that scatter is the
  finite-sim (60-cosmology LOSO) sampling of the low-k Jacobian, not a fittable bias.
- **Scoring trap (important):** `diag_ap_fisher_bias.coherent_vs_cv` pools ALL z=3 rows and means over
  sims — this MIXES the τ₀(alpha) cells and UNDER-states the true per-(z,τ₀) coherent (the PI's spec
  `Σ_cell⟨·⟩_θ²`). Score the coherent PER-CELL (`percell_coherent`): the per-cell numbers run higher
  (baseline clean mid 0.43% pooled vs 0.70% per-cell) and the band trades are only visible there.
- **Lesson:** before throwing capacity/loss-terms at a coherent emulator bias, split it TRAIN-vs-VAL.
  If TRAIN≪VAL it is a finite-sim generalization gap, NOT a fittable systematic — no training pressure
  removes it; a strong FLAT de-bias regularizer nudges it toward the CV floor, and the residue is
  irreducible on that sim set (budget it in C_emu). Targeted (edge/inverse-CV) weights TRADE bands;
  uniform spreads the pressure. And always score the coherent on the SAME grouping the metric specifies
  (per-cell), not a pooled proxy.

---

## 25. coherent de-bias `num_segments` static count must be a PYTHON INT in the batch, not a jnp leaf — **HIT**
- **Where:** finalizing the LF backbone — productionizing the FLAT coherent de-bias term
  (`model.coherent_debias_term`, wired through `joint_loss(w_coh)` / `train_fold`).
  `make_batch` emits the per-row `cell` id AND `n_cells` (= n_z·n_alpha, the
  `jax.ops.segment_sum(..., num_segments=n_cells)` count).
- **Symptom:** `int(batch["n_cells"])` raised `ConcretizationTypeError: Abstract tracer
  value encountered where concrete value is expected` the moment the de-bias loss ran
  inside the jit'd `train_step_partitioned`. The traceback fingered a `dynamic_nodonate`
  batch leaf — `n_cells` had been turned into a traced `int64[]` array by `_to_jnp_batch`.
- **Cause:** `num_segments` sets the segment-sum OUTPUT SHAPE, so it must be a STATIC
  python int at trace time. `_to_jnp_batch` blindly `jnp.asarray`'d every batch value,
  so `n_cells` became a traced array leaf of the jit'd step — and `int(tracer)` is illegal
  inside jit (the value isn't known until runtime).
- **Fix:** keep `n_cells` a PLAIN PYTHON INT in the batch dict (special-case it in BOTH
  `_to_jnp_batch` and `_pad_batch` — `int(v)`, never `jnp.asarray`). `eqx.filter_jit`
  partitions on `eqx.is_array`; a python int is not an array, so it lands in the STATIC
  (hashable) side and `coherent_debias_term`'s `int(batch["n_cells"])` is a no-op concrete
  read. (A numpy/jnp 0-d scalar also fails `eqx.filter`'s array test? — no: jnp 0-d IS an
  array; numpy 0-d IS an array too. Only a bare python int is reliably static.)
- **Lesson:** any quantity that sets an array SHAPE inside jit (segment counts, reshape
  dims, slice lengths) must reach the traced fn as a STATIC python scalar, not a batch
  array leaf. When a batch dict flows through a generic `{k: jnp.asarray(v)}` converter,
  special-case the shape-setting scalars to stay python ints (or pass them as separate
  static args), or jit will trace them and `int(...)`/`.reshape(...)` will raise.

## 26. `eqx.field(static=True)` does NOT freeze a sub-Module — use `stop_gradient` on its params — **HIT**
- **Where:** the multi-fidelity layer (`hcd_analysis/emulator/multifidelity.py`,
  `MultiFidelity`/`_freeze`/`lf_predict_logP`). The FROZEN LF backbone is held inside the
  trainable MF module; HMC needs `∂logP/∂θ` to flow through the LF forward (wrt the INPUT
  theta) but must NEVER perturb the LF WEIGHTS.
- **Symptom:** declaring `lf_model: Emulator = eqx.field(static=True)` did NOT freeze it.
  `eqx.partition(mf, eqx.is_array)` still recursed into the "static" sub-Module and exposed
  its weights as DYNAMIC leaves, and `eqx.filter_grad(loss)(mf)` returned NON-ZERO grads on
  `grad.lf_model.*` (0.7-scale), i.e. the LF weights were being differentiated. (The isolated
  `freeze(model)` worked, but through the static field the captured-vs-differentiated copies
  desynced so the in-forward `stop_gradient` did not bind to the leaves grad saw.)
- **Cause:** `static=True` only makes a field part of the treedef when its value is genuinely
  static (hashable / non-pytree). A field holding an `eqx.Module` is itself a PYTREE, so
  equinox recurses into it and its arrays remain dynamic leaves — `static` does not "freeze"
  a sub-network. And mixing a static-captured copy with the differentiated dynamic leaves
  means an in-forward `stop_gradient` on the static copy doesn't pin the leaves grad tracks.
- **Fix:** make `lf_model` a NORMAL dynamic field (single unambiguous leaf set) and freeze it
  in the forward with `_freeze`: `p, s = eqx.partition(model, eqx.is_array); p =
  tree_map(jax.lax.stop_gradient, p); model = eqx.combine(p, s)`. `stop_gradient` is a no-op
  on the value and zeroes the backward pass to those leaves, so the LF weights get EXACTLY
  zero grad while `∂logP/∂θ` (input) still flows. The optimizer (`train_delta_head`) filters
  to the DeltaHead alone, so the LF leaves never reach it either way. (Test:
  `test_mf_lf_backbone_is_frozen` asserts `grad.lf_model` leaves are all 0 while the head's
  are finite & nonzero.)
- **Lesson:** to FREEZE a sub-Module inside a trainable Equinox module, do NOT rely on
  `eqx.field(static=...)` — `stop_gradient` its parameters in the forward (or partition it out
  and `eqx.combine` a stop_gradient'd copy). Reserve `static=` for genuinely static scalars /
  shapes / non-array host objects.

## 27. `jax.jit(module.method)` hashes the module as a static arg → `unhashable type: 'list'` — **GUARDED**
- **Where:** the MF forward (`MultiFidelity.P_mf`/`logP_mf`, `multifidelity.py`). Trying to
  jit the forward for the HMC sampler.
- **Symptom:** `jax.jit(mf.P_mf)(x, tau0)` raised `TypeError: unhashable type: 'list'`. The
  forward, `vmap`, and grad were all fine eagerly; only this jit form broke.
- **Cause:** jitting a BOUND METHOD captures the module (`self`) as a traced/static positional
  arg; JAX hashes static args, and the `DeltaHead.layers` field is a python `list` (an Equinox
  `eqx.Module` is a pytree, not hashable). So the hash of the "static" module fails.
- **Fix:** use the Equinox jit idiom — `eqx.filter_jit(fn)(mf, x, tau0)` (partitions the module
  into array/static and only hashes the static part), or close the module over a lambda:
  `jax.jit(lambda x, t: mf.P_mf(x, t))` (module is a captured constant, not an arg). `vmap` is
  unaffected: `jax.vmap(lambda x, t: mf.P_mf(x, t))`. Documented in `P_mf`'s docstring + locked
  by `test_mf_jit_patterns`.
- **Lesson:** never `jax.jit` a bound method of an Equinox/pytree module — jit the module
  explicitly with `eqx.filter_jit`, or jit a plain function that closes over (or takes as a
  filtered arg) the module. Bound-method jit silently turns the whole module into a static arg.

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
