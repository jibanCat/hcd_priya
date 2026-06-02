# Phase-2b emulator — training walkthrough, normalization diagnosis & fix

**Date:** 2026-06-02. A stage-by-stage walkthrough of the emulator training pipeline (data prep →
normalization → NN init → loss → monitoring), the **diagnosis of why the first run under-resolves
cosmology**, how the field normalizes P1D, and the **recommended fix**. In the spirit of the mfbox
matter-power notebooks. Figures are in `figures/analysis/04_emulator/`.

> **TL;DR.** The pipeline is mechanically correct and trains, but the **target normalization is
> computed globally**, so the per-k std `σ_k` is set by the **τ₀+z spread (~99.5% of the variance)**,
> not the **cosmology signal (~0.4%, physically ~7–10%)**. After `(logP−μ_k)/σ_k`, the cosmology
> variation — the thing inference needs — is a ~6% sliver of a unit-variance target, so the MSE is
> dominated by the easy τ₀/z modes and cosmology is under-resolved (held-out corr 0.80, not ~1; val
> RMS ~10% ≈ the entire cosmology signal). **Fix: make the normalization (z,τ₀)-conditional** so the
> cosmology variation is the unit-scale target the loss optimizes.

---

## 1. The data (the merged v3.3 cache)
`observables_tau0_lf.h5`: **21440 rows = 60 sims × (snaps) × 20 τ₀-slopes**, n_k=172. Per row: 9 cosmo/IGM
`params`, `z_grid`, `target_F`/`τ₀`, per-class filtered P1D `P_filt[4,172]`, HCD deltas `Δ_c[3,172]`;
per snap-block (τ₀-invariant): CDDF `f_nhi[30]`, `dN/dX[3]`. The structural identity
`P_tier_p = Σ_c w_c·P_filt` holds to 1e-15.

## 2. Data preparation (`data.py::load_cache`, `make_splits`)
- **15→4 class collapse** (clean/LLS/subDLA/DLA), count-weighted; empty class → 0 (not NaN).
- **τ₀ = −ln(target_F)** per row.
- **Masks:** per-row Nyquist mask `isfinite(P_tier_p)`; f_nhi/dN/dX zero-bin validity mask.
- **Splits (`make_splits`):** τ₀-edge holdout excluded from train+val; then k-fold LOSO by sim (every
  sim held out once; no α-sibling leakage). Verified disjoint on the real cache.

![input params](../../figures/analysis/04_emulator/data_params_hist.png)
*The 9 params span ~9 orders of magnitude (Ap~1e-9 vs n_s~0.8) → input normalization is mandatory.*

![tau0 coverage](../../figures/analysis/04_emulator/data_tau0_coverage.png)
*A fixed α maps to different τ₀ across z → the τ₀-edge holdout (not α-edge) is the right extrapolation probe.*

## 3. Input normalization (unit cube; CORRECT)
The 9 params + z are mapped to **[0,1]** using PRIYA's `emulator_params.json` design box (all 60 sims
in-domain). This matches PRIYA/mfbox (`map_to_unit_cube`, "so all variations are similar in magnitude").

## 4. Output transforms + normalization (the CURRENT recipe — this is where the issue is)
`make_batch` builds targets in transformed + standardized space:
- `f_nhi, dN/dX, P_filt` → `safe_log` (=log); `Δ_c` → `arcsinh` (sign-safe, Δ flips sign at low k).
- then `apply_norm`: `(transformed − μ)/σ` where `(μ,σ)` come from `fit_norm(..., axis=0)` over **all
  train rows** → shape **(class,k)**, i.e. **per-(class,k) standardization computed GLOBALLY.**

![output transforms](../../figures/analysis/04_emulator/data_output_transforms.png)
*Log compresses the 3-decade P1D; arcsinh handles the low-k Δ_c sign flip. (Transform choice is fine.)*

This is exactly **CosmoPower's** recipe (`log` → per-feature `(μ,σ)` standardize; arXiv:2106.03846) —
*the right recipe for a NN with MSE loss*. **The problem is the scope of `(μ,σ)`, see §7.**

## 5. Architecture & init output
Encoder `[10→256→128→64]` → Head A (τ₀-invariant: f_nhi, dN/dX) + Head B (`concat(latent,τ₀)`:
P_filt via a learned low-rank basis, Δ_c dense). At init the heads output ~0 in standardized space →
`untransform` → `exp(μ_k)` = **the per-k mean spectrum** (right k-shape, identical for all params).
Training must then add the param/τ₀/z dependence on top.

![architecture](../../figures/analysis/04_emulator/emulator_architecture.png)

## 6. Loss & monitoring (`model.py::joint_loss`, `train.py`)
- **Single joint scalar:** per-element (mean) standardized-MSE over `f_nhi + dN/dX + P_filt + Δ_c`
  (uniform `term_w`), **NaN-safe masked**, weighted by `1/n_c` (per class) and `1/n_α` (Head A).
  Predicts CDDF + dN/dX + P1D + Δ **together**. τ₀ enters Head B. (All correct.)
- **Optimizer:** AdamW + cosine LR; minibatches padded to a fixed shape (single jit trace); early-stop
  on val (patience 10); SVD warm-start of the P_filt basis.

![loss curves](../../figures/analysis/04_emulator/train_val_loss_fold0.png)
![grad/lr](../../figures/analysis/04_emulator/grad_norm_lr_fold0.png)
*Fold-0 profile: train+val descend together (no overfit), still descending at early-stop — under-trained,
but the deeper issue is §7.*

## 7. DIAGNOSIS — why cosmology is under-resolved
Decompose the per-(class,k) log-P1D variance into **between-(z,τ₀)-cell** (the τ₀/z modes the NN gets as
inputs) vs **within-cell** (the cosmology variation across the 60 sims at fixed z,τ₀):

![variance decomposition](../../figures/analysis/04_emulator/norm_variance_decomp.png)
*Cosmology is **0.4–0.6%** of the per-k log-variance at every k; **~99.5% is the τ₀/z spread.***

![spectrum spread](../../figures/analysis/04_emulator/norm_spectrum_spread.png)
*The cosmology signal (dark band, ~7–10% physically) is **buried inside** the τ₀/z spread (light band).
Global `σ_k` is set by the light band.*

Consequence: the standardized target `(logP−μ_k)/σ_k` has its cosmology part at scale
`√0.004 ≈ 0.06` of unit variance. An MSE loss therefore optimizes the τ₀/z modes (trivially predictable
from the inputs) and barely weights the cosmology residual → the emulator under-resolves cosmology:

![fold-0 cosmology tracking](../../figures/analysis/04_emulator/fold0_cosmology_tracking.png)
*Fold-0 held-out: pred-vs-true deviation from the (z,τ₀)-cell mean — **corr 0.80, spread-ratio 0.80**.
It partially resolves cosmology (not collapsed-to-mean), but the residual ≈ the whole cosmology signal
(~10% val RMS) → not inference-grade.*

## 8. How the field normalizes P1D (lit review)
| Emulator | regressor | transform | normalization scope | architecture |
|---|---|---|---|---|
| Rogers&Bird 2019 (1812.04654) | GP | linear `P/median − 1` | per-k **median** reference | **one GP per z**, τ₀ input |
| Fernandez/Ho/Bird 2022 (2207.06445) | GP (multi-fid) | linear `P/median_LF − 1` | per-k median (LF) | per-z |
| **PRIYA Bird+2023 (2306.05471)** | GP | linear `P/median − 1` | per-k median, **per z**, τ₀ a **dense input** (10/z) | **one GP per z** |
| mfbox (jibanCat) | GP/NN | `log10 P` | per-k **mean-subtract** (LF) | multi-fidelity AR1/NARGP |
| **CosmoPower (2106.03846)** | **NN** | `log P` | **per-feature `(μ,σ)` standardize** | single NN (+PCA variant) |
| LaCE (2305.19064) | NN (MDN) | `log10 P/median` → **5–7 poly coeffs** | upstream median | predicts coeffs |

**The key lesson:** for a **NN**, `log` + per-k `(μ,σ)` standardize (our recipe, = CosmoPower) is right.
The Bird/PRIYA lineage divides by a per-k **median reference** with **no σ** (a GP's kernel amplitude
absorbs the scale). But crucially **PRIYA computes its reference per-z and feeds τ₀ as a dense input** —
so its normalization reference removes the z (and, via the input, τ₀) dynamic range. **Our global μ/σ
does not** — that's the gap.

## 9. RECOMMENDED FIX — make the normalization (z,τ₀)-conditional
Keep the NN + `log`/`arcsinh` + per-k `σ` (correct for MSE), but **compute the reference within (z,τ₀)
cells** so the cosmology variation becomes the unit-scale target. Three implementable options:

**(A) (z,τ₀)-conditional reference normalization (minimal, recommended first).** Replace the global
`μ_k` with `μ_k(z,τ₀)` and the global `σ_k` with the **within-cell (cosmology) σ_k**:
target `= (logP − μ_k(z,τ₀)) / σ_cosmo_k`. Build `μ_k(z,τ₀)` from the cache as the per-(z,α-cell) log-mean,
and **interpolate it smoothly in (z,τ₀)** for inference (z on a 12-grid, τ₀ smooth). The model is
unchanged; only the target scaling + the inverse change. Expected: the cosmology signal goes from ~6% to
~unit scale → the loss optimizes it → val RMS should drop toward the few-% / sub-% the field reaches.

**(B) Per-z models (most PRIYA-faithful).** Train one emulator per z (12), τ₀ a dense input, per-z
median/σ reference. Cleanest statistically, but multiplies the model count and breaks the single-model
convenience; the z-trend is no longer shared.

**(C) In-network mean/std head (your preferred — most elegant).** Add a small head that predicts the
smooth `(z,τ₀)`-conditional mean (and log-σ) spectrum from `(z,τ₀)` only; the main head predicts the
**cosmology residual**; combine `P = decode(residual)·σ̂(z,τ₀) + μ̂(z,τ₀)` (in log space). Trained
end-to-end, fully differentiable for HMC, **no external reference to store/interpolate** — the network
*is* the reference. This is the in-network embodiment of (A) and is the strongest long-term choice.

**Validation of the fix:** re-run fold-0 and check the §7 diagnostic — **corr → ~0.95+**, spread-ratio
→ ~1, and the per-class val RMS → few-% (ideally ~1–2%). If (A) already gets there, (C) is a polish; if
not, go to (C)/per-z. Also re-examine the fold-3 val-loss outlier (Head-A extrapolation) under the new
norm.

## 10. Status of the rest (unchanged by this fix)
The HCD likelihood (single-α_c + M₀-inverse), the structural identity, the covariance, the splits, the
NaN-safe loss, τ₀-as-input, and input unit-cube norm are all correct and stay. This fix is localized to
the **output target normalization** (`fit_target_norm`/`make_batch`/`untransform_prediction`, and a
reference store or a mean-head in `model.py`).

## Sources
Rogers&Bird 2019 (1812.04654 eq.2.10); Fernandez/Ho/Bird 2022 (2207.06445 eq.6); PRIYA Bird+2023
(2306.05471 §2.7/eq.2.14-2.16); CosmoPower (2106.03846 §2); LaCE (2305.19064 eq.5); mfbox
(github.com/jibanCat/matter_emu_mfbox `data_loader.py` PowerSpecs/PowerSpecsMedianNorm). Diagnostic:
`scripts/diag_p1d_normalization.py`.
