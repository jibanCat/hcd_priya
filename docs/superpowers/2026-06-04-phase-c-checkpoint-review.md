# Phase-C checkpoint review — 4-agent consolidated report

**Branch:** `phase2c-likelihood` · **Date:** 2026-06-04 · **Scope:** committed likelihood/inference/predict/closure code + HCD redesign + T4a foundation.

## Verdict: 4 / 4 GO

| Reviewer | Lens | Verdict | One-line |
|---|---|---|---|
| **CS** (correctness/regressions) | software | **GO** | 176 tests pass; library ship-ready; one broken diagnostic script (fixed). |
| **Lyα** (HCD physics) | domain | **GO** | Corrected forward model + observed-centered priors faithful; every headline number reproduced independently. |
| **Emulator** (C_emu / inference) | error budget | **GO** | `Σ coef²σ²P²` propagation verified vs 2M-sample MC to 0.1%; SBC/whitening/coverage tooling correct + tested. |
| **Stat/PPL** (SBC machinery) | statistics | **GO** | Two-leg identity 0.0e+00; ECDF FWER 0.043; whitening exact; only doc-level minors. |

**No CRITICAL findings from any reviewer.** All blocking items are pre-closure must-do's, not defects in committed code.

---

## Already fixed this checkpoint (the figure/script regressions all reviewers flagged)

- ✅ `scripts/diag_grad_fidelity.py` ported to the new `predict_P_obs(…, pf, dla_core)` signature → `grad_fidelity*.png` regenerated. Gate **PASSES** on the corrected forward model (worst-median rel-err 1.3e-7, worst-max 2.1e-5, 0 non-finite).
- ✅ `B7_delta_c_templates.png` regenerated from `predict_excess` (was plotting the deprecated `delta` head with Δ_LLS≡0). Now shows the corrected excess; LLS non-zero, DLA filtered-vs-unfiltered overlay.
- ✅ New `checkpoint_forward.png` (C_emu sub-dominance): median **C_emu/C_data = 0.028** vs a DESI-like 5% data error — partially answers the emulator reviewer's top must-do.

---

## Consolidated action items (de-duplicated, prioritized)

### A. Must-do before the SBC closure run
1. **`C_emu ≪ C_data` against the REAL cosmic_cov** (Emulator). The 0.028 figure uses an illustrative 5% diagonal; redo with the production multi-z covariance + an explicit `diag(C_emu)/diag(C_data)` (z,k) heatmap.
2. **Explicit LF-Nyquist `k_max` cut in `DATA_RANGE`** (Emulator). Currently only `k_min` is set; upper-k masking relies on native Nyquist NaNs. Add the physical LF-Nyquist cut (~0.069 angular s/km) so emulator residuals near Nyquist can't leak into C_emu.
3. **Production data-binding layer** (Emulator, Stat/PPL). No loader assembles `valid_k / P_data / cosmic_cov / sigma_zb` from real DESI vectors, and no NUTS/numpyro driver exists yet (only the kernel). Build it; feed the whitening test on real held-out sims (per-sim covariances, not a shared C).
4. **Replace `cemu_inflate = 1.0` with the bisection-calibrated value** (Emulator, Stat/PPL). Use `calibrate_cemu_inflate`; compute the ECDF band γ **once** at high `n_sim` (≥4000) and reuse across params/quantities; keep ≳1000–2000 retained draws/mock before reading coverage.
5. **Bound the subDLA prior to non-negative α** (Lyα). The broad subDLA Gaussian (2.5σ from 0) permits negative incidence; switch to half-normal/softplus in the production sampler (DLA already ~10σ from 0, harmless; comment-only today).

### B. Should-do (faithfulness / docs)
6. **Cosmology-bias figure caveat** (Lyα, Emulator). `hcd_prior_cosmology.png` is single-z, diagonal, σ=5% illustrative. At σ=2% (DR2-like) worst mis-centering bias rises to **0.192σ** — at the gate; multi-z stacking could push past it. Add the caveat to the caption + ideally a σ=2% (or production-cov) panel. Re-frame "<0.1σ" as "<0.2σ at the illustrative fiducial."
7. **Doc-number corrections** (Lyα): walkthrough §6.5 "0.30·**1.43**·w_DLA" → **1.34** (fitted pivot); subDLA σ/μ "0.25" → **0.40**; reword "α=w reproduces P_tier_p **exactly**" → "up to the DLA-core add-back (~0.34% of P)".

### C. Minor (cleanup, non-blocking)
8. Drop the stale `assemble_covariance` reference from the `inference.py` module docstring (CS, Emulator) — the live path inlines per-class C_emu; avoids two-code-path confusion.
9. Dead `delta` HeadB in `model.py`/`train.py` no longer consumed by the forward model (CS) — harmless, retire later.
10. `closure_diagnostics._ecdf_band_gamma` calibrates γ on `n_grid` points but applies on `n_grid+1` (CS) — conservative, 1-point mismatch; note only.
11. `rank_histogram_bins` degenerates to B=1 when L+1 is prime (Stat/PPL) — design uses L=99 (L+1=100) so fine; add a one-line guard so nobody picks L=100.
12. `sigma_scale_fn` docstring should state the σ-multiplier is **not** √inflate (Stat/PPL) — behavior is correct (caller-delegated), doc-only.

---

## Per-reviewer correctness confirmations (evidence)

- **Forward model:** `P_obs = P_clean + Σ α_c(P_c−P_clean) = Σ coef_c·P_c`, algebra exact to 4e-16; `∂P_obs/∂α_c = (P_c−P_clean)` matches `predict_excess` to rtol 1e-9; LLS excess now non-zero (old Δ_LLS≡0 fixed). Autodiff vs central-FD median rel-err ~7e-8.
- **Per-class C_emu:** `emu_var = Σ coef²σ²P²` matches a 2M-sample MC to <0.3%; single-channel (old 2-channel `assemble_covariance` is dead in the driver path). Known anti-conservatism (diagonal-in-k, class-independent) is the documented `cemu_inflate` caveat.
- **vmap-over-z:** `log_lik_multiz == Σ single-z` to rtol 1e-10; likelihood-only (no prior double-count).
- **NaN-safety:** all-NaN σ rows, NaN `P_data`, `valid_k` masking → finite logL + finite grads (trap-#29 fix correctly placed: sanitize interp ydata *before* `jnp.interp`).
- **SBC machinery:** `potential == log_lik + Σ log_prior` to 0.0e+00; ECDF simultaneous bands FWER 0.043 (nominal 0.05), power 1.0 at the 0.2σ design bias, N≈621; whitening `r_white=L⁻¹r` exact (var 1.000, χ²/dof 1.000; flags a 4× under-estimated C); Wilson coverage CI correct; eigenvector truth-Fisher-fixed (de-circularized).
- **HCD priors:** A6 PRIYA/obs = 0.98/1.31/0.70 reproduced; prior centers μ=(0.196,0.045,0.0124) = (1.06,0.76,0.40)·w_c; observed-centered z-slope correct; DLA weakened slope + widened σ above z=3.5 defensible.
