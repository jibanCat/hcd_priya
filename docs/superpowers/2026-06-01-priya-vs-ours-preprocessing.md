# PRIYA vs. ours — emulator preprocessing audit (2026-06-01)

Read-only audit of data preprocessing consistency between PRIYA's GP Lyα emulator
(`/home/mfho/lya_emulator_full`, Bird+2023) and our Phase-2b Equinox emulator
(`/home/mfho/hcd_priya/hcd_analysis/emulator`). Gating question: does our emulator
preprocess inputs/outputs in a way that lets it train correctly?

**TL;DR — the one finding that matters:** Our pipeline has **NO input-parameter
normalization**. PRIYA maps every emulator input to the unit cube `[0,1]` before
the GP; we feed the 9 cosmological/IGM parameters to the encoder **raw**, with
scales spanning ~9 orders of magnitude (`Ap ~ 1.2e-9` vs `ns ~ 0.9` vs
`omegamh2 ~ 0.14`). This is a real, must-fix training bug. Everything else
(P1D transform, k-grid units, τ₀ semantics, output standardization) is either an
intentional, defensible divergence or an outright agreement.

---

## PART 1 — How PRIYA preprocesses (with citations)

### (1) Parameter normalization — UNIT CUBE

PRIYA maps **all** emulator inputs onto the unit cube `[0,1]` before the GP, using
per-parameter `param_limits`, so "all the variations are similar in magnitude":

- `lya_emulator_full/lyaemu/gpemulator.py:89` (single-fidelity build) and
  `:155` (predict): `params_cube = map_to_unit_cube_list(self.params, self.param_limits)`.
- Multi-fidelity path identical: `gpemulator.py:194`, `:264`.
- The mapping is the affine rescale to `[0,1]` per dimension:
  `latin_hypercube.py:141-160` `map_to_unit_cube` →
  `(param_vec - lo) / (hi - lo)`, with `map_to_unit_cube_list` at `:162`.
- `gpemulator.py:91-93` even asserts the design *fills* the cube
  (`max(cube[:,i]) > 0.8`, `min < 0.2`) per parameter.

The 9 sim parameters and their physical limits (`coarse_grid.py:78-124`):

| idx | name        | limits            |
|-----|-------------|-------------------|
| 0   | ns          | 0.8 – 0.995       |
| 1   | Ap          | **1.2e-9 – 2.6e-9** |
| 2   | herei       | 3.5 – 4.1         |
| 3   | heref       | 2.6 – 3.2         |
| 4   | alphaq      | 1.3 – 2.5         |
| 5   | hub         | 0.65 – 0.75       |
| 6   | omegamh2    | 0.14 – 0.146      |
| 7   | hireionz    | 6.5 – 8.0         |
| 8   | bhfeedback  | 0.03 – 0.07       |

After unit-cube mapping every one of these is `[0,1]` — `Ap`'s 1e-9 magnitude is
gone. `Ap` itself is the amplitude at the Lyα pivot `k_p = π/4 /Mpc`
(`coarse_grid.py:101-102`; our `_AP_PIVOT_RATIO = 5π` transform mirrors this,
`build_emulator_cache_tau0.py:220-246`).

### (2) P1D / flux-power transform — LINEAR RATIO TO MEDIAN (not log)

PRIYA does **not** emulate `log P1D`. It emulates the **linear fractional residual
to the median training spectrum**, per redshift bin:

- `gpemulator.py:96-100`: pick the median-power spectrum `medind`, set
  `self.scalefactors = flux_vectors[medind,:]`, then
  `normspectra = flux_vectors / self.scalefactors - 1.` — a per-k linear ratio
  centered on 0, so the zero-mean GP prior is ~true. (Multi-fidelity identical:
  `:196-199`.)
- Inverse at predict: `mean = (flux_predict + 1) * self.scalefactors`
  (`gpemulator.py:157`, `:267`).
- Per-k normalization = the per-k median-spectrum vector (one scalefactor per k bin).
- k-grid: **separate GP per redshift** (`MultiBinGP`, `gpemulator.py:18-36`), each
  bin has its own `nk` k-values. k is in **km/s (angular)** units —
  `flux_power.py:121-129` `get_kf_kms` divides comoving k by `velfac`
  (= `H(z)/(1+z)` in cm/s, `flux_power.py:23`), i.e. the standard `dv = H(z)/(1+z) dx`.

### (3) Mean-flux / τ₀ handling — DIMENSIONLESS FACTOR, prepended, unit-cube mapped

PRIYA treats mean flux as an extra **"dense" emulator input prepended as column 0**:

- The mean-flux parameter is a **dimensionless multiplicative factor `t0`** on the
  observed mean optical depth: `mean_flux.py:151-157` `get_t0` returns
  `t0 * obs_mean_tau(z)`, with `obs_mean_tau = 2.3e-3 (1+z)^3.65`
  (`mean_flux.py:24-32`). It is **not** an optical depth in physical units; it is a
  scaling factor centered on 1.
- Its sampling range is the slope→factor band (`MeanFluxFactor.__init__`,
  `mean_flux.py:121-143`), roughly `[0.75, 1.25] × slope-factors`, i.e. an O(1)
  dimensionless range.
- It is prepended as the FIRST ("slow"/dense) emulator column:
  `coarse_grid.py:409-419` `get_param_limits(include_dense=True)`:
  `plimits = np.vstack([dlim, self.param_limits])`. So `t0` is param 0.
- At predict time the τ₀ dimension is scanned per redshift by multiplying that
  first column: `gpemulator.py:48` `zparams[0][0] *= tau0_factors[i]`.
- Crucially, this `t0` column **also goes through `map_to_unit_cube`** (it is part
  of the vstacked `param_limits` fed to the GP). So PRIYA's mean-flux input is a
  unit-cube `[0,1]` value, exactly like the cosmology params.

### (4) Output / GP normalization

PRIYA is a **GPy** Gaussian process (`gpemulator.py:14`, `GPy.models.GPRegression`
at `:106`; multi-fidelity `GPy.core.GP` at `:221`). There is no sklearn
`normalize=True`; output "normalization" is exactly the linear median-ratio of (2)
(`normspectra = flux/scalefactors - 1`), applied **per redshift bin, per k**. Kernel
is `Linear(ARD) + RBF(ARD)` (`:104-105`) — a different length scale per input
dimension, which is precisely *why* the unit-cube input mapping matters (ARD length
scales are meaningless if one input is 1e-9 and another is 1).

---

## PART 2 — Our setting + verdict (A–D)

### A. Do we normalize the 9 encoder params, or pass them RAW?  →  **GAP / DIVERGENCE (must fix)**

**RAW. Input normalization is MISSING.**

Trace of `params` into the encoder:
- `data.py:58` `load_cache` reads `params` straight from the cache (float64), shape
  `(R, 9)`, in physical units (built by `_read_priya_params`,
  `build_emulator_cache_tau0.py:229-258`; PARAM_ORDER = ns, Ap, herei, heref,
  alphaq, hub, omegamh2, hireionz, bhfeedback — `build_emulator_cache.py:141`).
- `load_cache` applies channel transforms and **output** normalization helpers, but
  **never** transforms `params`. There is no unit-cube map, no `param_limits`, no
  mean/std on the inputs anywhere in `data.py`.
- The encoder consumes `batch["x"]` directly: `model.py:75-77`
  `Emulator.__call__(x, tau0)` → `self.enc(x)`; `Encoder.__call__` is a plain MLP
  (`model.py:19-22`), no internal scaling.
- **There is currently no training driver and no code that builds `batch["x"]` from
  `params` at all.** The only construction of `x` in the repo is the unit test, which
  passes synthetic `jnp.zeros((2,10))` (`tests/test_emulator_model.py:196`). So the
  intended `params → x` wiring does not exist yet, and as specified it would be raw:
  spec §3 line 120 says `params(9) ⊕ z → encoder MLP [10 → …]` with **no**
  normalization step (`in_dim=10` = 9 params + z; τ₀ is the separate Head-B input,
  `model.py:51-58`).
- `fit_norm` / `apply_norm` (`data.py:102-110`) exist and are tested
  (`test_emulator_data.py:57-62`) but are **output-channel** standardizers; nothing
  calls them on inputs.

Why this breaks training: the encoder's first `Linear` would see column scales
`Ap ~ 1e-9` next to `ns ~ 0.9`, `hub ~ 0.7`, `omegamh2 ~ 0.14`, `hireionz ~ 7`,
`herei ~ 4`. With Glorot-initialized weights the `Ap` and `bhfeedback` (~0.05)
directions contribute essentially nothing to the pre-activation, while `hireionz`
(~7) dominates; gradients w.r.t. the tiny-scale inputs are ~9 orders of magnitude
suppressed. The network cannot learn the `Ap`/`bhfeedback` dependence. This is a
direct divergence from PRIYA's unit-cube design (which exists precisely so the
ARD GP can fit a sane per-dimension length scale).

**Verdict: GAP. Input parameter normalization is MISSING and must be added before
training.** The redshift `z` co-input (encoder col 10) needs scaling too (z ~ 2–4.6).

### B. Does our P1D transform (log) match PRIYA's? k-grid / units?  →  **DIFFER (transform), AGREE (units)**

- **Transform DIFFERS, by design.** We emulate **log-space** filtered class P1D
  (spec §3 line 141; `safe_log` in `data.py:99`, Head B outputs in log-space,
  `model.py:42`, exponentiated for the structural sum `structural_tier_p`,
  `model.py:80-86`). PRIYA emulates the **linear** ratio-to-median
  `flux/median - 1`. This is an intentional architecture choice (NN vs GP; log
  keeps the positive P1D positive and is the natural space for the structural
  `Σ_c w_c P_c` identity). Not a bug — but note we do **not** subtract a
  median/fiducial spectrum the way PRIYA does; our zero-mean prior comes from the
  per-channel **train-split mean/std** standardization (§D) instead. Functionally
  analogous (both center the regression target), just in different spaces.
- **k-grid units AGREE.** Our cache `kfkms` is in **s/km angular** units
  (`priya_p1d.py:167` "native k-grid (s/km, angular)"), the same convention as
  PRIYA's `get_kf_kms` (`flux_power.py:121-129`). Confirmed the convention matches.
- One structural difference: we keep dense per-row native-FFT k-bins
  (n_k=172, NaN above each row's Nyquist, masked — spec §4) and a single encoder
  over `(params, z)`, vs PRIYA's separate GP per redshift on the rebinned BOSS
  k-grid. Different binning strategy; both are angular km/s. No unit mismatch.

### C. Is our τ₀ input handled consistently with PRIYA's mean flux?  →  **DIFFER (semantics + space), benign in scale**

- **Semantics differ.** PRIYA's mean-flux input is the **dimensionless factor `t0`**
  (multiplier on `obs_mean_tau(z)`, centered ~1). Ours is
  `τ₀ = −ln(target_F)` (`data.py:81`), and `target_F = exp(−α·obs_mean_tau(z))`
  (`priya_p1d.py:65`), so `τ₀ = α · obs_mean_tau(z)` — a **dimensional effective
  optical depth** that is z-dependent, not the dimensionless factor. The underlying
  α (alpha_slope) is the analogue of PRIYA's `t0` factor; we feed the *resulting τ₀*
  instead of the factor.
- **Routing differs (intentional).** PRIYA prepends the mean-flux factor as GP input
  column 0 (one GP per z). We route τ₀ **only into Head B**
  (`model.py:57`, `concat(latent, τ₀)`), keeping Head A τ₀-invariant by construction
  (spec §3). Different architecture, deliberate.
- **Scale is benign.** `obs_mean_tau(z) = 2.3e-3 (1+z)^3.65` gives ≈0.13 at z=2.2 and
  ≈0.86 at z=4.6; with α≈0.75–1.25 our τ₀ spans roughly **0.1–1.1** — an O(1)
  quantity, so Head B's `concat(latent, τ₀)` is well-scaled even without
  normalization. (Contrast with the raw 9 cosmo params in §A, which are *not*
  well-scaled.) A mild per-feature standardization of τ₀ would still help the
  arcsinh/log targets line up but is **not** a blocker.
- **Consistency check present:** `mean_F_clean ≈ exp(−τ₀)` is enforced as a loss term
  (`model.py:119`), which is the right τ₀↔mean-flux tie. AGREE on intent.

### D. Output standardization — do we; does PRIYA; mismatch?  →  **DIFFER, but OK**

- **We standardize outputs** by **train-split** mean/std per channel in transformed
  space (`fit_norm`/`apply_norm`, `data.py:102-113`; spec §3 line 142-143, stored in
  checkpoint). Train-only fit avoids val leakage — good practice.
- **PRIYA "standardizes" outputs** only via the linear median-ratio
  (`flux/median - 1`, `gpemulator.py:96-100`), per z-bin, per k. Different mechanism
  (median-ratio vs mean/std z-score), different space (linear vs log), but the same
  *purpose*: center the regression target near zero so the prior/initialization is
  reasonable.
- **No mismatch that matters**, with one caveat to verify when the training driver is
  written: ensure `fit_norm` is computed on the **same train split** used for the
  encoder (and per the LOSO/τ₀-edge folds, `data.py:118-138`) and that
  `nanmean/nanstd` (already used, `data.py:104-105`) correctly ignore the
  Nyquist-NaN bins so the stats aren't poisoned. As written this looks correct.

---

## PART 3 — Prioritized preprocessing fixes BEFORE training

1. **[BLOCKER] Add input-parameter normalization to the loader / training driver.**
   Map the 9 params to a comparable scale before the encoder. Two equivalent
   options:
   - **Unit cube** (matches PRIYA exactly): apply `map_to_unit_cube` with PRIYA's
     `param_limits` (the table in Part 1; mirror `coarse_grid.py:93-124`). Most
     faithful, and makes our inputs identical in spirit to PRIYA's.
   - **Train-split z-score**: reuse the existing `fit_norm`/`apply_norm` on `params`
     (fit on train rows only). Equally valid for an MLP; reuses code we already have
     and test.
   Either way, the `Ap ~ 1e-9` column must not reach the encoder raw. **Also scale
   the z co-input** (col 10) onto a comparable range.
   Until this is wired, the `params → batch["x"]` path does not exist — build it
   *with* normalization from the start.

2. **[NICE-TO-HAVE] Standardize the τ₀ Head-B input.** τ₀∈~[0.1,1.1] is benign, but a
   train-split z-score (or shift to a mean-flux *factor* like PRIYA's `t0` to match
   semantics) would make the Head-B input distribution cleaner. Not a blocker.

3. **[VERIFY when writing the trainer] Output-norm stats must be fit on the exact
   train split** (LOSO / τ₀-edge fold), using the existing NaN-safe
   `nanmean/nanstd`, and stored in the checkpoint alongside the input-norm stats so
   inference inverts both. Confirm NaN-bins are excluded (they are, in `fit_norm`).

4. **[DOC, no code change] P1D transform space intentionally differs** (we: log +
   train-split z-score; PRIYA: linear ratio-to-median). Keep, but record that we do
   *not* subtract a fiducial/median spectrum — our centering is the channel z-score.
   k-grid units confirmed matching (s/km angular). No action.
